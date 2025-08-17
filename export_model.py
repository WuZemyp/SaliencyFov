import torch
from model import SaliencyNet

# Match the variant used in training
VARIANT = "edgenext_xx_small"

# Load checkpoint
ckpt = torch.load("best_model.pt", map_location="cpu")

# Build model and materialize lazy layers
model = SaliencyNet(variant=VARIANT, pretrained=False).eval()
with torch.no_grad():
    model(torch.randn(1, 3, 192, 256), None)

# Load weights (allow missing/unexpected due to lazy init differences)
missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
print("missing:", missing)
print("unexpected:", unexpected)
model.eval()

# Script (NOT trace) to keep Optional[Tensor] in the signature
scripted = torch.jit.script(model)
scripted.save("best_model_ts.pt")
print("Saved best_model_ts.pt")

# Quick sanity run (hidden=None is accepted)
with torch.no_grad():
    sal, h = scripted(torch.randn(1, 3, 192, 256), None)
    print("saliency shape:", tuple(sal.shape), "hidden shape:", tuple(h.shape))