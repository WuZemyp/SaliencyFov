#include "SaliencyPredictor.h"
#include "alvr_server/Utils.h"
#include "alvr_server/Logger.h"
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <sstream>
#include <cerrno>
#include <cstring>
#ifdef ALVR_SALIENCY
#include <torch/torch.h>
#include <c10/core/InferenceMode.h>
#endif
#include <Windows.h>
#ifdef ALVR_SALIENCY
#include <cuda_runtime_api.h>
#include <cuda_d3d11_interop.h>
#endif

SaliencyPredictor::SaliencyPredictor(std::shared_ptr<CD3DRender> d3dRender)
	: m_d3d(std::move(d3dRender)) {}

SaliencyPredictor::~SaliencyPredictor() {}

static bool CompileShader(const char* source, const char* entry, const char* profile, ID3DBlob** blob) {
	ComPtr<ID3DBlob> code;
	ComPtr<ID3DBlob> err;
	auto hr = D3DCompile(source, strlen(source), nullptr, nullptr, nullptr, entry, profile, 0, 0, &code, &err);
	if (FAILED(hr)) {
		if (err) {
			Error("SaliencyPredictor: shader compile error: %s\n", (const char*)err->GetBufferPointer());
		}
		return false;
	}
	*blob = code.Detach();
	return true;
}

bool SaliencyPredictor::Initialize() {
	m_downscaled.resize(3 * kOutH * kOutW);
	// Basic env diagnostics
	try {
		auto cwd = std::filesystem::current_path().string();
		Info("SaliencyPredictor: CWD=%s\n", cwd.c_str());
	} catch (...) {}
	{
		char* pathEnv = getenv("PATH");
		if (pathEnv) {
			Info("SaliencyPredictor: PATH begins: %.200s ...\n", pathEnv);
		}
	}
	// DLL probing (keep loaded)
	{
		static HMODULE hTorchCuda = LoadLibraryA("torch_cuda.dll");
		if (hTorchCuda) {
			Info("SaliencyPredictor: LoadLibrary(torch_cuda.dll) OK\n");
		} else {
			Error("SaliencyPredictor: LoadLibrary(torch_cuda.dll) FAILED (err=%lu)\n", GetLastError());
		}
		static HMODULE hNvCuda = LoadLibraryA("nvcuda.dll");
		if (hNvCuda) {
			Info("SaliencyPredictor: LoadLibrary(nvcuda.dll) OK\n");
		} else {
			Error("SaliencyPredictor: LoadLibrary(nvcuda.dll) FAILED (err=%lu)\n", GetLastError());
		}
		static HMODULE hC10Cuda = LoadLibraryA("c10_cuda.dll");
		if (hC10Cuda) {
			Info("SaliencyPredictor: LoadLibrary(c10_cuda.dll) OK\n");
		} else {
			Error("SaliencyPredictor: LoadLibrary(c10_cuda.dll) FAILED (err=%lu)\n", GetLastError());
		}
		// Some PyTorch builds split CUDA kernels
		static HMODULE hTorchCudaCu = LoadLibraryA("torch_cuda_cu.dll");
		if (hTorchCudaCu) {
			Info("SaliencyPredictor: LoadLibrary(torch_cuda_cu.dll) OK\n");
		}
	}
#ifdef ALVR_SALIENCY
	try {
		// Model path from env var or default next to binaries
		char* pathEnv = nullptr;
		size_t len = 0;
		std::string modelPath;
		if (_dupenv_s(&pathEnv, &len, "SALIENCY_MODEL_PATH") == 0 && pathEnv && len > 0) {
			modelPath.assign(pathEnv, len);
			free(pathEnv);
		} else {
			modelPath = "C:/Users/Ze/Desktop/Ieevr/code/EyeNexus/best_model_ts.pt"; // working dir
		}
		m_module = torch::jit::load(modelPath);
		m_module.eval();
		m_modelLoaded = true;
		Info("SaliencyPredictor: loaded model: %s\n", modelPath.c_str());
		// Report resolved module locations
		char modPath[MAX_PATH] = {0};
		HMODULE hTorch = GetModuleHandleA("torch.dll");
		if (hTorch && GetModuleFileNameA(hTorch, modPath, MAX_PATH)) {
			Info("SaliencyPredictor: torch.dll => %s\n", modPath);
		}
		HMODULE hTorchCpu = GetModuleHandleA("torch_cpu.dll");
		if (hTorchCpu && GetModuleFileNameA(hTorchCpu, modPath, MAX_PATH)) {
			Info("SaliencyPredictor: torch_cpu.dll => %s\n", modPath);
		}
		HMODULE hC10 = GetModuleHandleA("c10.dll");
		if (hC10 && GetModuleFileNameA(hC10, modPath, MAX_PATH)) {
			Info("SaliencyPredictor: c10.dll => %s\n", modPath);
		}
		HMODULE hTorchCudaNow = GetModuleHandleA("torch_cuda.dll");
		if (hTorchCudaNow && GetModuleFileNameA(hTorchCudaNow, modPath, MAX_PATH)) {
			Info("SaliencyPredictor: torch_cuda.dll => %s\n", modPath);
		} else {
			Info("SaliencyPredictor: torch_cuda.dll not loaded yet\n");
		}
	} catch (const c10::Error& e) {
		Error("SaliencyPredictor: failed to load model: %s\n", e.what());
		m_modelLoaded = false;
	}
#endif
	m_initialized = true;
	return true;
}

bool SaliencyPredictor::EnsureReadbackBuffer(ID3D11Texture2D* srcTexture) {
	if (!srcTexture) return false;
	D3D11_TEXTURE2D_DESC desc{};
	srcTexture->GetDesc(&desc);
	if (m_readbackTex && desc.Width == m_srcWidth && desc.Height == m_srcHeight) {
		return true;
	}
	m_srcWidth = desc.Width;
	m_srcHeight = desc.Height;
	desc.Usage = D3D11_USAGE_STAGING;
	desc.BindFlags = 0;
	desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
	if (FAILED(m_d3d->GetDevice()->CreateTexture2D(&desc, nullptr, &m_readbackTex))) {
		Error("SaliencyPredictor: CreateTexture2D(readback) failed\n");
		return false;
	}
	return true;
}

void SaliencyPredictor::DownscaleBilinear(const uint8_t* src, size_t srcPitch) {
	// src is RGBA8 at m_srcWidth x m_srcHeight
	const int dstW = kOutW;
	const int dstH = kOutH;
	for (int y = 0; y < dstH; ++y) {
		float fy = (y + 0.5f) * (float)m_srcHeight / dstH - 0.5f;
		int y0 = (int)floorf(fy); int y1 = y0 + 1;
		float wy1 = fy - y0; float wy0 = 1.0f - wy1;
		y0 = y0 < 0 ? 0 : (y0 >= (int)m_srcHeight ? (int)m_srcHeight - 1 : y0);
		y1 = y1 < 0 ? 0 : (y1 >= (int)m_srcHeight ? (int)m_srcHeight - 1 : y1);
		const uint8_t* row0 = src + y0 * srcPitch;
		const uint8_t* row1 = src + y1 * srcPitch;
		for (int x = 0; x < dstW; ++x) {
			float fx = (x + 0.5f) * (float)m_srcWidth / dstW - 0.5f;
			int x0 = (int)floorf(fx); int x1 = x0 + 1;
			float wx1 = fx - x0; float wx0 = 1.0f - wx1;
			x0 = x0 < 0 ? 0 : (x0 >= (int)m_srcWidth ? (int)m_srcWidth - 1 : x0);
			x1 = x1 < 0 ? 0 : (x1 >= (int)m_srcWidth ? (int)m_srcWidth - 1 : x1);
			const uint8_t* p00 = row0 + x0 * 4;
			const uint8_t* p10 = row0 + x1 * 4;
			const uint8_t* p01 = row1 + x0 * 4;
			const uint8_t* p11 = row1 + x1 * 4;
			for (int c = 0; c < 3; ++c) {
				float v00 = p00[c];
				float v10 = p10[c];
				float v01 = p01[c];
				float v11 = p11[c];
				float v0 = v00 * wx0 + v10 * wx1;
				float v1 = v01 * wx0 + v11 * wx1;
				float v = v0 * wy0 + v1 * wy1;
				m_downscaled[c * dstH * dstW + y * dstW + x] = v / 255.0f;
			}
		}
	}
}

bool SaliencyPredictor::EnsureGpuDownscalePipeline(ID3D11Texture2D* srcTexture) {
	if (!srcTexture) return false;
	D3D11_TEXTURE2D_DESC sdesc{};
	srcTexture->GetDesc(&sdesc);
	if (!m_srcSRV) {
		if (FAILED(m_d3d->GetDevice()->CreateShaderResourceView(srcTexture, nullptr, &m_srcSRV))) {
			Error("SaliencyPredictor: CreateShaderResourceView(src) failed\n");
			return false;
		}
	}
	if (!m_dsTex) {
		D3D11_TEXTURE2D_DESC t{};
		t.Width = kOutW; t.Height = kOutH; t.MipLevels = 1; t.ArraySize = 1;
		t.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		t.SampleDesc.Count = 1; t.SampleDesc.Quality = 0;
		t.Usage = D3D11_USAGE_DEFAULT; t.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
		t.CPUAccessFlags = 0; t.MiscFlags = 0;
		if (FAILED(m_d3d->GetDevice()->CreateTexture2D(&t, nullptr, &m_dsTex))) {
			Error("SaliencyPredictor: CreateTexture2D(ds) failed\n");
			return false;
		}
		if (FAILED(m_d3d->GetDevice()->CreateRenderTargetView(m_dsTex.Get(), nullptr, &m_dsRTV))) {
			Error("SaliencyPredictor: CreateRenderTargetView(ds) failed\n");
			return false;
		}
		m_dsViewport.TopLeftX = 0; m_dsViewport.TopLeftY = 0;
		m_dsViewport.Width = (FLOAT)kOutW; m_dsViewport.Height = (FLOAT)kOutH;
		m_dsViewport.MinDepth = 0.0f; m_dsViewport.MaxDepth = 1.0f;
		// Sampler
		D3D11_SAMPLER_DESC samp{}; samp.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
		samp.AddressU = samp.AddressV = samp.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
		if (FAILED(m_d3d->GetDevice()->CreateSamplerState(&samp, &m_linearSampler))) {
			Error("SaliencyPredictor: CreateSamplerState failed\n");
			return false;
		}
		// Shaders
		static const char* vsSrc = R"(
		struct VSOut { float4 pos:SV_Position; float2 uv:TEXCOORD; };
		VSOut main(uint vid:SV_VertexID){ VSOut o; float2 p = float2((vid<<1)&2, vid&2); o.pos=float4(p*float2(2,-2)+float2(-1,1),0,1); o.uv=p; return o; }
		)";
		static const char* psSrc = R"(
		cbuffer CB : register(b0) { float4 uvRect; } // (u0,v0,u1,v1)
		Texture2D srcTex:register(t0); SamplerState samp:register(s0);
		float4 main(float2 uv:TEXCOORD):SV_Target{
			float2 tuv = float2(lerp(uvRect.x, uvRect.z, uv.x), lerp(uvRect.y, uvRect.w, uv.y));
			return srcTex.Sample(samp, tuv);
		}
		)";
		ComPtr<ID3DBlob> vsb, psb;
		if (!CompileShader(vsSrc, "main", "vs_5_0", &vsb)) return false;
		if (!CompileShader(psSrc, "main", "ps_5_0", &psb)) return false;
		if (FAILED(m_d3d->GetDevice()->CreateVertexShader(vsb->GetBufferPointer(), vsb->GetBufferSize(), nullptr, &m_vs))) return false;
		if (FAILED(m_d3d->GetDevice()->CreatePixelShader(psb->GetBufferPointer(), psb->GetBufferSize(), nullptr, &m_ps))) return false;
		// Create constant buffer
		D3D11_BUFFER_DESC cbd{}; cbd.ByteWidth = 16; cbd.BindFlags = D3D11_BIND_CONSTANT_BUFFER; cbd.Usage = D3D11_USAGE_DYNAMIC; cbd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
		if (FAILED(m_d3d->GetDevice()->CreateBuffer(&cbd, nullptr, &m_uvCB))) return false;
	}
	return true;
}

#ifdef ALVR_SALIENCY
bool SaliencyPredictor::EnsureCudaInterop() {
	if (m_cudaRes) return true;
	if (!m_dsTex) return false;
	cudaError_t err = cudaGraphicsD3D11RegisterResource(&m_cudaRes, m_dsTex.Get(), cudaGraphicsRegisterFlagsNone);
	if (err != cudaSuccess) {
		Error("SaliencyPredictor: cudaGraphicsD3D11RegisterResource failed: %d\n", (int)err);
		m_cudaRes = nullptr;
		return false;
	}
	return true;
}

bool SaliencyPredictor::EnsureCudaBuf(size_t bytes) {
	if (m_cudaBufU8 && m_cudaBufU8Size >= bytes) return true;
	if (m_cudaBufU8) { cudaFree(m_cudaBufU8); m_cudaBufU8 = nullptr; m_cudaBufU8Size = 0; }
	cudaError_t err = cudaMalloc(&m_cudaBufU8, bytes);
	if (err != cudaSuccess) {
		Error("SaliencyPredictor: cudaMalloc failed: %d\n", (int)err);
		return false;
	}
	m_cudaBufU8Size = bytes;
	return true;
}
#endif

#ifdef ALVR_SALIENCY
static torch::Tensor make_gaussian_kernel_2d(int k, float sigma, torch::Device device, c10::ScalarType dtype) {
	// Build separable 1D gaussian and form 2D via outer product
	auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);
	auto ax = torch::arange(k, opts) - (k - 1) * 0.5f;
	auto g1 = torch::exp(-(ax * ax) / (2.0f * sigma * sigma));
	auto g2d = (g1.unsqueeze(1) * g1.unsqueeze(0));
	g2d = g2d / g2d.sum();
	g2d = g2d.to(dtype);
	return g2d.unsqueeze(0).unsqueeze(0); // [1,1,k,k]
}
#endif

void SaliencyPredictor::Process(ID3D11Texture2D* srcTexture) {
	if (!m_initialized || !srcTexture) return;

	auto t_begin = std::chrono::high_resolution_clock::now();

	bool gpu_path_ok = EnsureGpuDownscalePipeline(srcTexture)
#ifdef ALVR_SALIENCY
		&& EnsureCudaInterop()
#endif
		;
#ifdef ALVR_SALIENCY
	torch::Tensor x_from_cuda;
#endif
#ifdef ALVR_SALIENCY
	if (gpu_path_ok) {
		// Update uv rect to left half [0,0]-[0.5,1]
		D3D11_MAPPED_SUBRESOURCE m{};
		if (SUCCEEDED(m_d3d->GetContext()->Map(m_uvCB.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &m))) {
			float* p = (float*)m.pData; p[0]=0.0f; p[1]=0.0f; p[2]=0.5f; p[3]=1.0f;
			m_d3d->GetContext()->Unmap(m_uvCB.Get(), 0);
		}
		// Draw fullscreen triangle downscale
		auto ctx = m_d3d->GetContext();
		ID3D11RenderTargetView* rt = m_dsRTV.Get();
		ctx->OMSetRenderTargets(1, &rt, nullptr);
		ctx->RSSetViewports(1, &m_dsViewport);
		ctx->VSSetShader(m_vs.Get(), nullptr, 0);
		ctx->PSSetShader(m_ps.Get(), nullptr, 0);
		ctx->PSSetConstantBuffers(0, 1, m_uvCB.GetAddressOf());
		ID3D11ShaderResourceView* srv = m_srcSRV.Get();
		ctx->PSSetShaderResources(0, 1, &srv);
		ID3D11SamplerState* samp = m_linearSampler.Get();
		ctx->PSSetSamplers(0, 1, &samp);
		ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		ctx->Draw(3, 0);
		// Unbind SRV to avoid D3D warnings if src is also bound elsewhere
		ID3D11ShaderResourceView* nulls[1] = { nullptr };
		ctx->PSSetShaderResources(0, 1, nulls);

		// CUDA map + copy to tensor
		cudaError_t err = cudaGraphicsMapResources(1, &m_cudaRes, 0);
		if (err == cudaSuccess) {
			cudaArray_t arr{};
			err = cudaGraphicsSubResourceGetMappedArray(&arr, m_cudaRes, 0, 0);
			if (err == cudaSuccess) {
				size_t pitch = kOutW * 4;
				if (EnsureCudaBuf(kOutH * pitch)) {
					cudaMemcpy2DFromArray(m_cudaBufU8, pitch, arr, 0, 0, pitch, kOutH, cudaMemcpyDeviceToDevice);
					// Wrap into a CUDA byte tensor and convert to model input
					auto options_u8 = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kUInt8);
					auto raw = torch::from_blob(m_cudaBufU8, {kOutH, kOutW, 4}, options_u8);
					x_from_cuda = raw.permute({2,0,1}).narrow(0,0,3).to(torch::kHalf).div_(255.0).unsqueeze(0).contiguous();
				}
			}
			cudaGraphicsUnmapResources(1, &m_cudaRes, 0);
		}
	}
#endif

	auto t_after_resize = std::chrono::high_resolution_clock::now();

#ifdef ALVR_SALIENCY
	auto t_before_infer = t_after_resize;
	if (m_modelLoaded) {
		// Try to move module to CUDA once; avoid calling torch::cuda::is_available
		static bool module_on_cuda = false;
		static bool tried_cuda = false;
		if (!tried_cuda) {
			tried_cuda = true;
			bool cuda_dlls_loaded = GetModuleHandleA("torch_cuda.dll") != nullptr && GetModuleHandleA("c10_cuda.dll") != nullptr;
			if (!cuda_dlls_loaded) {
				Error("SaliencyPredictor: CUDA DLLs not loaded, skip moving module to CUDA\n");
			} else {
			try {
				m_module.to(torch::kCUDA);
				module_on_cuda = true;
				Info("SaliencyPredictor: moved module to CUDA\n");
				// Convert weights to FP16 for speed on RTX
				try { m_module.to(torch::kHalf); Info("SaliencyPredictor: converted module to FP16\n"); } catch (const c10::Error&) {}
				// Warm-up a few iters to initialize cuDNN kernels
				try {
					c10::InferenceMode no_grad(true);
					std::vector<torch::jit::IValue> winputs;
					winputs.emplace_back(torch::zeros({1,3,kOutH,kOutW}, torch::dtype(torch::kHalf).device(torch::kCUDA)));
					winputs.emplace_back(torch::jit::IValue());
					for (int i = 0; i < 3; ++i) {
						auto wout = m_module.forward(winputs).toTuple();
					}
					torch::cuda::synchronize();
					Info("SaliencyPredictor: CUDA warm-up done\n");
				} catch (const c10::Error&) {}
			} catch (const c10::Error& e) {
				module_on_cuda = false;
				Error("SaliencyPredictor: failed to move module to CUDA: %s\n", e.what());
			}
			}
		}

		// Build input tensor (1,3,192,256)
		torch::Tensor x;
		if (module_on_cuda && x_from_cuda.defined()) {//
			x = x_from_cuda;
		} else {
			// Fallback to CPU path
			if (!EnsureReadbackBuffer(srcTexture)) return;
			m_d3d->GetContext()->CopyResource(m_readbackTex.Get(), srcTexture);
			D3D11_MAPPED_SUBRESOURCE mapped{};
			if (FAILED(m_d3d->GetContext()->Map(m_readbackTex.Get(), 0, D3D11_MAP_READ, 0, &mapped))) {
				return;
			}
			const uint8_t* src = reinterpret_cast<const uint8_t*>(mapped.pData);
			DownscaleBilinear(src, mapped.RowPitch);
			m_d3d->GetContext()->Unmap(m_readbackTex.Get(), 0);
			auto options = torch::TensorOptions().dtype(torch::kFloat32);
			x = torch::from_blob(m_downscaled.data(), {1, 3, kOutH, kOutW}, options).clone();
			if (module_on_cuda) {
				x = x.to(torch::kCUDA, torch::kHalf);
			}
		}

		static torch::Tensor hidden; // persist across frames
		std::vector<torch::jit::IValue> inputs;
		inputs.emplace_back(x);
		if (hidden.defined()) {
			if (module_on_cuda && !hidden.is_cuda()) {
				hidden = hidden.to(torch::kCUDA, /*dtype*/torch::kHalf);
			}
			inputs.emplace_back(hidden);
		} else {
			inputs.emplace_back(torch::jit::IValue()); // None for Optional[Tensor]
		}
		try {
			c10::InferenceMode no_grad(true);
			Info("SaliencyPredictor: input device: %s, hidden defined=%d on=%s\n",
				x.is_cuda() ? "cuda" : "cpu",
				(int)hidden.defined(),
				hidden.defined() ? (hidden.is_cuda() ? "cuda" : "cpu") : "n/a");
			auto out = m_module.forward(inputs).toTuple();
			if (module_on_cuda) { torch::cuda::synchronize(); }
			torch::Tensor sal = out->elements()[0].toTensor();
			hidden = out->elements()[1].toTensor();
			// Post-process: normalize to [0,1] per-frame and apply small Gaussian blur
			{
				auto s = sal.detach(); // [1,1,H,W]
				bool on_cuda = s.is_cuda();
				c10::ScalarType work_dtype = on_cuda ? s.scalar_type() : torch::kFloat32; // CPU half conv may be unsupported
				// Normalize
				auto s_min = s.amin(std::vector<int64_t>{2,3}, /*keepdim=*/true);
				auto s_max = s.amax(std::vector<int64_t>{2,3}, /*keepdim=*/true);
				auto denom = (s_max - s_min) + 1e-8f;
				auto s_norm = (s - s_min) / denom;
				// Gaussian blur (k=7, sigma=1.5)
				int ksz = 7; float sigma = 1.5f;
				auto kernel = make_gaussian_kernel_2d(ksz, sigma, s.device(), work_dtype);
				// If CPU and original dtype is half, upcast for conv2d
				auto s_conv = s_norm;
				if (!on_cuda && s_conv.scalar_type() != work_dtype) s_conv = s_conv.to(work_dtype);
				int pad = ksz / 2;
				auto s_blur = torch::conv2d(s_conv, kernel, {}, {1,1}, {pad, pad});
				// Cast back to original dtype if needed
				if (s_blur.scalar_type() != s.scalar_type()) s_blur = s_blur.to(s.scalar_type());
				m_lastSaliency = s_blur;
			}
			Info("SaliencyPredictor: output device: %s\n", m_lastSaliency.is_cuda() ? "cuda" : "cpu");
			Info("SaliencyPredictor: inference ok. saliency sizes=%lld dims, last=(%lld,%lld)\n", (long long)m_lastSaliency.dim(), (long long)m_lastSaliency.size(-2), (long long)m_lastSaliency.size(-1));
			// quick stats
			auto cpu = m_lastSaliency.flatten().to(torch::kFloat32).cpu();
			float minv = cpu.min().item<float>();
			float maxv = cpu.max().item<float>();
			Info("SaliencyPredictor: saliency range [%.6f, %.6f]\n", minv, maxv);
		} catch (const c10::Error& e) {
			Error("SaliencyPredictor: inference failed: %s\n", e.what());
		}
	} else {
		Error("SaliencyPredictor: model not loaded\n");
	}
#else
	// No-op if saliency disabled
#endif
	auto t_after_infer = std::chrono::high_resolution_clock::now();
	double ms_resize = std::chrono::duration<double, std::milli>(t_after_resize - t_begin).count();
	double ms_infer = std::chrono::duration<double, std::milli>(t_after_infer - t_after_resize).count();
	double ms_total = std::chrono::duration<double, std::milli>(t_after_infer - t_begin).count();
	Info("Saliency latency: resize+copy=%.3fms infer=%.3fms total=%.3fms\n", ms_resize, ms_infer, ms_total);

	m_frameCounter++;
#ifdef ALVR_SALIENCY
	// CSV dump removed
#endif
} 