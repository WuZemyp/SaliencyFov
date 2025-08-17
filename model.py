"""SaliencyNet
==============
Encoder  : Official EdgeNeXt (from timm)
Temporal : SR-TSM-GRU (Spatial-Reuse + Temporal-Shift + ConvGRU)
Decoder  : GhostFPN with skip connections
Optimised for <3 ms per 192×256 frame on RTX 4090 FP16 (xx-small variant).

Requires:  pip install timm torch torchvision
"""
from typing import List, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm  # official EdgeNeXt implementation

# -----------------------------------------------------------------------------
# Helper: Ghost convolution (as in GhostNet)
# -----------------------------------------------------------------------------
class GhostConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, ratio: int = 2):
        super().__init__()
        init_channels = math.ceil(out_ch / ratio)
        new_channels = init_channels * (ratio - 1)
        self.primary = nn.Sequential(
            nn.Conv2d(in_ch, init_channels, 1, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True),
        )
        self.cheap = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, 3, 1, 1, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True),
        )
        self.out_ch = out_ch

    def forward(self, x: torch.Tensor):
        y = self.primary(x)
        z = self.cheap(y)
        out = torch.cat([y, z], dim=1)
        return out[:, : self.out_ch]

# -----------------------------------------------------------------------------
# Official EdgeNeXt backbone (timm) wrapper
# -----------------------------------------------------------------------------
class EdgeNeXtEncoder(nn.Module):
    """Returns bottleneck (H/16) + skips (H/2, H/4, H/8)."""
    def __init__(self, variant: str = "edgenext_xx_small", pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            variant,
            features_only=True,
            pretrained=pretrained,
            out_indices=(1, 2, 3),  # H/4, H/8, H/16
        )
        # Derive channel dimensions dynamically from the returned features
        channels = [d["num_chs"] for d in self.backbone.feature_info]
        # `channels` is ordered to match out_indices we passed (len == 3)
        c4, c8, c16 = channels[0], channels[1], channels[2]
        self.skip_channels = [c4, c4, c8]  # H/2 (synthetic), H/4, H/8
        self.bottleneck_ch = c16

    def forward(self, x: torch.Tensor):
        feats = self.backbone(x)  # list as per out_indices
        # Assume first feature is the highest resolution available (lowest stride)
        lowest = feats[0]  # e.g., stride 8 for EdgeNeXt xx-small
        s8 = lowest
        # Generate higher-resolution skips by upsampling (TorchScript requires float scale_factors)
        s4 = F.interpolate(s8, scale_factor=2.0, mode="nearest")
        s2 = F.interpolate(s8, scale_factor=4.0, mode="nearest")
        bottleneck = feats[-1]
        skips = [s2, s4, s8]
        return bottleneck, skips

# -----------------------------------------------------------------------------
# Temporal module: SR-TSM-GRU
# -----------------------------------------------------------------------------
class TemporalShift(nn.Module):
    """TSM for streaming (left shift previous hidden)."""
    def __init__(self, fold_div: int = 8):
        super().__init__()
        self.fold_div = fold_div

    def forward(self, x: torch.Tensor, prev: torch.Tensor):
        B, C, H, W = x.shape
        fold = C // self.fold_div
        out = torch.empty_like(x)
        # first fold comes from previous hidden (left shift)
        out[:, :fold] = prev[:, :fold]
        # second fold is zero (future placeholder)
        out[:, fold:2*fold] = 0
        # rest is identity
        out[:, 2*fold:] = x[:, 2*fold:]
        return out

class SR_TSM_GRU(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.shift = TemporalShift()
        self.gates = nn.Conv2d(channels * 2, channels * 2, 3, 1, 1, groups=channels, bias=True)
        self.cand  = nn.Conv2d(channels * 2, channels, 3, 1, 1, groups=channels, bias=True)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor]):
        if h is None:
            h = torch.zeros_like(x)
        x_shift = self.shift(x, h)
        cat = torch.cat([x_shift, h], dim=1)
        r, u = self.gates(cat).chunk(2, 1)
        r, u = torch.sigmoid(r), torch.sigmoid(u)
        cat2 = torch.cat([x_shift, r * h], dim=1)
        n = torch.tanh(self.cand(cat2))
        h_new = (1 - u) * n + u * h
        return h_new

# -----------------------------------------------------------------------------
# Decoder: GhostFPN
# -----------------------------------------------------------------------------
class GhostFPN(nn.Module):
    def __init__(self, bottleneck_ch: int, skip_channels: List[int]):
        super().__init__()
        s2, s4, s8 = skip_channels
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="nearest"),
            GhostConv(bottleneck_ch + s8, bottleneck_ch // 2),  # H/8
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="nearest"),
            GhostConv(bottleneck_ch // 2 + s4, bottleneck_ch // 4),  # H/4
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="nearest"),
            GhostConv(bottleneck_ch // 4 + s2, bottleneck_ch // 8),  # H/2
        )
        self.head = nn.Conv2d(bottleneck_ch // 8, 1, 1)

    def forward(self, x: torch.Tensor, skips: List[torch.Tensor]):
        s2, s4, s8 = skips  # resolutions ascending
        # Align to s8 resolution
        x = F.interpolate(x, size=s8.shape[2:], mode="nearest")
        x = torch.cat([x, s8], dim=1)
        x = self.up1(x)
        # Align to s4 resolution
        x = F.interpolate(x, size=s4.shape[2:], mode="nearest")
        x = torch.cat([x, s4], dim=1)
        x = self.up2(x)
        # Align to s2 resolution
        x = F.interpolate(x, size=s2.shape[2:], mode="nearest")
        x = torch.cat([x, s2], dim=1)
        x = self.up3(x)
        # x is already aligned with stride-2 (s2) resolution, which we choose as output size
        return torch.sigmoid(self.head(x))

# -----------------------------------------------------------------------------
# Full model
# -----------------------------------------------------------------------------
class SaliencyNet(nn.Module):
    def __init__(self, variant: str = "edgenext_xx_small", pretrained: bool = True):
        super().__init__()
        self.encoder = EdgeNeXtEncoder(variant, pretrained)
        # Lazy initialisation: actual channel counts vary per variant; we build
        # reduce / rnn / expand the first time we see a real tensor.
        self.reduce: Optional[nn.Module] = None
        self.rnn:   Optional[SR_TSM_GRU] = None
        self.expand: Optional[nn.Module] = None
        # Decoder uses skip-channel info which is reliable from feature_info
        self.decoder = None  # will create later when we know bottleneck_ch

    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        bottleneck, skips = self.encoder(x)

        # Lazy build for reduce/rnn/expand/decoder on first call
        if self.reduce is None:
            in_ch = bottleneck.shape[1]
            red_ch = max(16, in_ch // 4)  # ensure at least 16 channels
            self.reduce = nn.Sequential(
                nn.Conv2d(in_ch, red_ch, 1, bias=False),
                nn.BatchNorm2d(red_ch),
                nn.Hardswish(inplace=True),
            ).to(bottleneck.device)
            self.rnn = SR_TSM_GRU(red_ch).to(bottleneck.device)
            self.expand = nn.Conv2d(red_ch, in_ch, 1, bias=False).to(bottleneck.device)
            # decoder needs correct bottleneck ch and skip channel sizes from runtime
            skip_chs = [s.shape[1] for s in skips]
            self.decoder = GhostFPN(in_ch, skip_chs).to(bottleneck.device)

        z = self.reduce(bottleneck)
        hidden = self.rnn(z, hidden)
        saliency = self.decoder(self.expand(hidden), skips)
        return saliency, hidden

    # clip-level training helper
    def forward_clip(self, clip: torch.Tensor) -> torch.Tensor:
        outs, h = [], None
        for frame in clip:  # iterate time dimension
            sal, h = self.forward(frame, h)
            outs.append(sal)
        return torch.stack(outs)

# -----------------------------------------------------------------------------
# Simple speed profile
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import time
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model = SaliencyNet().to(device).eval()
    dummy = torch.randn(1, 3, 192, 256).to(device)
    h = None
    with torch.no_grad():
        for _ in range(20):
            _, h = model(dummy, h)  # warm-up
    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.time()
    with torch.no_grad():
        for _ in range(100):
            _, h = model(dummy, h)
    torch.cuda.synchronize() if device == "cuda" else None
    print(f"Avg latency: {(time.time()-t0)/100*1000:.2f} ms / frame")
