#pragma once
#include <memory>
#include <d3d11.h>
#include <wrl.h>
#include <d3dcompiler.h>

#include "shared/d3drender.h"

#ifdef ALVR_SALIENCY
#include <torch/script.h>
#endif

using Microsoft::WRL::ComPtr;

class SaliencyPredictor {
public:
	SaliencyPredictor(std::shared_ptr<CD3DRender> d3dRender);
	~SaliencyPredictor();

	bool Initialize();
	void Process(ID3D11Texture2D* srcTexture);

private:
	std::shared_ptr<CD3DRender> m_d3d;

	// Readback resources sized like the source texture
	ComPtr<ID3D11Texture2D> m_readbackTex;
	UINT m_srcWidth = 0;
	UINT m_srcHeight = 0;

	// CPU staging buffer for downscaled RGB (float, 3x192x256)
	static constexpr int kOutW = 256;
	static constexpr int kOutH = 192;
	std::vector<float> m_downscaled;

#ifdef ALVR_SALIENCY
	torch::jit::script::Module m_module;
	bool m_modelLoaded = false;
	// Optional: cache last saliency for debugging/consumers
	torch::Tensor m_lastSaliency;
#endif

	bool m_initialized = false;
	uint64_t m_frameCounter = 0;

	bool EnsureReadbackBuffer(ID3D11Texture2D* srcTexture);
	void DownscaleBilinear(const uint8_t* src, size_t srcPitch);
}; 