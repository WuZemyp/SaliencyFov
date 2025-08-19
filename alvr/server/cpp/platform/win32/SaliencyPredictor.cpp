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
#include "../../analyze_use/helper_f.h"
#include <wrl/client.h>
#include <d3dcompiler.h>
#pragma comment(lib, "d3dcompiler.lib")
#include <Windows.h>
#include <wincodec.h>
#pragma comment(lib, "windowscodecs.lib")
#ifdef ALVR_SALIENCY
#include <torch/torch.h>
#include <c10/core/InferenceMode.h>
#endif
#ifdef ALVR_SALIENCY
#include <cuda_runtime_api.h>
#include <cuda_d3d11_interop.h>
#endif

// Toggle GPU downscale (false = force CPU path)
static bool g_use_gpu_downscale = false;
// Toggle debug image outputs (PNGs)
static bool g_enable_debug_images = false;

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

void SaliencyPredictor::DownscaleBilinear(const uint8_t* src, size_t srcPitch, int roiX, int roiY, int roiW, int roiH) {
	// src is RGBA8 at m_srcWidth x m_srcHeight
	const int dstW = kOutW;
	const int dstH = kOutH;
	for (int y = 0; y < dstH; ++y) {
		float fy = (y + 0.5f) * (float)roiH / dstH - 0.5f;
		int y0 = (int)floorf(fy); int y1 = y0 + 1;
		float wy1 = fy - y0; float wy0 = 1.0f - wy1;
		y0 = y0 < 0 ? 0 : (y0 >= roiH ? roiH - 1 : y0);
		y1 = y1 < 0 ? 0 : (y1 >= roiH ? roiH - 1 : y1);
		const uint8_t* row0 = src + (roiY + y0) * srcPitch;
		const uint8_t* row1 = src + (roiY + y1) * srcPitch;
		for (int x = 0; x < dstW; ++x) {
			float fx = (x + 0.5f) * (float)roiW / dstW - 0.5f;
			int x0 = (int)floorf(fx); int x1 = x0 + 1;
			float wx1 = fx - x0; float wx0 = 1.0f - wx1;
			x0 = x0 < 0 ? 0 : (x0 >= roiW ? roiW - 1 : x0);
			x1 = x1 < 0 ? 0 : (x1 >= roiW ? roiW - 1 : x1);
			const uint8_t* p00 = row0 + (roiX + x0) * 4;
			const uint8_t* p10 = row0 + (roiX + x1) * 4;
			const uint8_t* p01 = row1 + (roiX + x0) * 4;
			const uint8_t* p11 = row1 + (roiX + x1) * 4;
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
	}
	// Sampler
	if (!m_linearSampler) {
		D3D11_SAMPLER_DESC samp{}; samp.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
		samp.AddressU = samp.AddressV = samp.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
		if (FAILED(m_d3d->GetDevice()->CreateSamplerState(&samp, &m_linearSampler))) {
			Error("SaliencyPredictor: CreateSamplerState failed\n");
			return false;
		}
	}
	// Force recompilation during debugging
	m_vs.Reset();
	m_ps.Reset();
	if (!m_vs || !m_ps) {
		// Shaders
		static const char* vsSrc = R"(
		struct VSOut { float4 pos:SV_Position; float2 uv:TEXCOORD; };
		VSOut main(uint vid:SV_VertexID){ VSOut o; float2 p = float2((vid<<1)&2, vid&2); o.pos=float4(p*float2(2,-2)+float2(-1,1),0,1); o.uv=p*0.5; return o; }
		)";
		static const char* psSrc = R"(
		cbuffer Params:register(b0){ float4 uvRect; }
		Texture2D texSrc:register(t0);
		SamplerState samp:register(s0);
		float4 main(float2 uv:TEXCOORD):SV_Target{
			uint ws, hs; texSrc.GetDimensions(ws, hs);
			float2 tuv = lerp(uvRect.xy, uvRect.zw, saturate(uv));
			float3 cS = texSrc.Sample(samp, tuv).rgb;
			uint2 xy = uint2(saturate(tuv) * float2(ws, hs));
			float3 cL = texSrc.Load(int3(xy, 0)).rgb;
			return float4(max(cS, cL), 1.0);
		}
		)";
		ComPtr<ID3DBlob> vsb, psb;
		if (!CompileShader(vsSrc, "main", "vs_5_0", &vsb)) return false;
		if (!CompileShader(psSrc, "main", "ps_5_0", &psb)) return false;
		if (FAILED(m_d3d->GetDevice()->CreateVertexShader(vsb->GetBufferPointer(), vsb->GetBufferSize(), nullptr, &m_vs))) return false;
		if (FAILED(m_d3d->GetDevice()->CreatePixelShader(psb->GetBufferPointer(), psb->GetBufferSize(), nullptr, &m_ps))) return false;
	}
	if (!m_uvCB) {
		D3D11_BUFFER_DESC cbd{}; cbd.ByteWidth = 16; cbd.BindFlags = D3D11_BIND_CONSTANT_BUFFER; cbd.Usage = D3D11_USAGE_DYNAMIC; cbd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
		if (FAILED(m_d3d->GetDevice()->CreateBuffer(&cbd, nullptr, &m_uvCB))) return false;
	}
	if (!m_rsNoCull) {
		D3D11_RASTERIZER_DESC rsd{}; rsd.FillMode = D3D11_FILL_SOLID; rsd.CullMode = D3D11_CULL_NONE; rsd.DepthClipEnable = TRUE;
		if (FAILED(m_d3d->GetDevice()->CreateRasterizerState(&rsd, &m_rsNoCull))) return false;
	}
	if (!m_bsOpaque) {
		D3D11_BLEND_DESC bd{}; bd.RenderTarget[0].BlendEnable = FALSE; bd.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
		if (FAILED(m_d3d->GetDevice()->CreateBlendState(&bd, &m_bsOpaque))) return false;
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

static bool SaveTextureToPng(ID3D11Device* dev, ID3D11DeviceContext* ctx, ID3D11Texture2D* tex, const std::wstring& path)
{
	if (!dev || !ctx || !tex) return false;
	D3D11_TEXTURE2D_DESC desc{};
	tex->GetDesc(&desc);
	D3D11_TEXTURE2D_DESC sdesc = desc;
	sdesc.Usage = D3D11_USAGE_STAGING;
	sdesc.BindFlags = 0;
	sdesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
	Microsoft::WRL::ComPtr<ID3D11Texture2D> staging;
	if (FAILED(dev->CreateTexture2D(&sdesc, nullptr, &staging))) return false;
	ctx->CopyResource(staging.Get(), tex);
	D3D11_MAPPED_SUBRESOURCE mapped{};
	if (FAILED(ctx->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped))) return false;
	bool needUninit = false;
	HRESULT hrCI = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
	if (SUCCEEDED(hrCI)) needUninit = true;
	Microsoft::WRL::ComPtr<IWICImagingFactory> factory;
	HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory));
	if (FAILED(hr)) { if (needUninit) CoUninitialize(); ctx->Unmap(staging.Get(), 0); return false; }
	Microsoft::WRL::ComPtr<IWICStream> stream;
	hr = factory->CreateStream(&stream);
	if (FAILED(hr)) { if (needUninit) CoUninitialize(); ctx->Unmap(staging.Get(), 0); return false; }
	hr = stream->InitializeFromFilename(path.c_str(), GENERIC_WRITE);
	if (FAILED(hr)) { if (needUninit) CoUninitialize(); ctx->Unmap(staging.Get(), 0); return false; }
	Microsoft::WRL::ComPtr<IWICBitmapEncoder> encoder;
	hr = factory->CreateEncoder(GUID_ContainerFormatPng, nullptr, &encoder);
	if (FAILED(hr)) { if (needUninit) CoUninitialize(); ctx->Unmap(staging.Get(), 0); return false; }
	hr = encoder->Initialize(stream.Get(), WICBitmapEncoderNoCache);
	if (FAILED(hr)) { if (needUninit) CoUninitialize(); ctx->Unmap(staging.Get(), 0); return false; }
	Microsoft::WRL::ComPtr<IWICBitmapFrameEncode> frame;
	Microsoft::WRL::ComPtr<IPropertyBag2> props;
	hr = encoder->CreateNewFrame(&frame, &props);
	if (SUCCEEDED(hr)) hr = frame->Initialize(props.Get());
	if (SUCCEEDED(hr)) hr = frame->SetSize(desc.Width, desc.Height);
	GUID pix = GUID_WICPixelFormat32bppRGBA;
	if (SUCCEEDED(hr)) hr = frame->SetPixelFormat(&pix);
	if (SUCCEEDED(hr)) {
		UINT stride = mapped.RowPitch;
		UINT bufferSize = stride * desc.Height;
		hr = frame->WritePixels(desc.Height, stride, bufferSize, static_cast<BYTE*>(mapped.pData));
	}
	if (SUCCEEDED(hr)) hr = frame->Commit();
	if (SUCCEEDED(hr)) hr = encoder->Commit();
	ctx->Unmap(staging.Get(), 0);
	if (needUninit) CoUninitialize();
	return SUCCEEDED(hr);
}

// Save an RGBA8 buffer directly to PNG via WIC
static bool SaveRgba8ToPng(int width, int height, const uint8_t* rgba, const std::wstring& path)
{
	if (!rgba || width <= 0 || height <= 0) return false;
	bool needUninit = false;
	HRESULT hrCI = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
	if (SUCCEEDED(hrCI)) needUninit = true;
	Microsoft::WRL::ComPtr<IWICImagingFactory> factory;
	HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory));
	if (FAILED(hr)) { if (needUninit) CoUninitialize(); return false; }
	Microsoft::WRL::ComPtr<IWICStream> stream;
	hr = factory->CreateStream(&stream);
	if (FAILED(hr)) { if (needUninit) CoUninitialize(); return false; }
	hr = stream->InitializeFromFilename(path.c_str(), GENERIC_WRITE);
	if (FAILED(hr)) { if (needUninit) CoUninitialize(); return false; }
	Microsoft::WRL::ComPtr<IWICBitmapEncoder> encoder;
	hr = factory->CreateEncoder(GUID_ContainerFormatPng, nullptr, &encoder);
	if (FAILED(hr)) { if (needUninit) CoUninitialize(); return false; }
	hr = encoder->Initialize(stream.Get(), WICBitmapEncoderNoCache);
	if (FAILED(hr)) { if (needUninit) CoUninitialize(); return false; }
	Microsoft::WRL::ComPtr<IWICBitmapFrameEncode> frame;
	Microsoft::WRL::ComPtr<IPropertyBag2> props;
	hr = encoder->CreateNewFrame(&frame, &props);
	if (SUCCEEDED(hr)) hr = frame->Initialize(props.Get());
	if (SUCCEEDED(hr)) hr = frame->SetSize(width, height);
	GUID pix = GUID_WICPixelFormat32bppRGBA;
	if (SUCCEEDED(hr)) hr = frame->SetPixelFormat(&pix);
	if (SUCCEEDED(hr)) {
		UINT stride = (UINT)(width * 4);
		UINT bufferSize = stride * (UINT)height;
		hr = frame->WritePixels((UINT)height, stride, bufferSize, const_cast<BYTE*>(rgba));
	}
	if (SUCCEEDED(hr)) hr = frame->Commit();
	if (SUCCEEDED(hr)) hr = encoder->Commit();
	if (needUninit) CoUninitialize();
	return SUCCEEDED(hr);
}

// Convert float RGB [0..1] to RGBA8 and save
static bool SaveFloatRgbToPng(int width, int height, const float* rgb, const std::wstring& path)
{
	if (!rgb || width <= 0 || height <= 0) return false;
	std::vector<uint8_t> rgba((size_t)width * (size_t)height * 4);
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			size_t idx = (size_t)y * (size_t)width + (size_t)x;
			float r = rgb[idx + 0 * (size_t)width * (size_t)height];
			float g = rgb[idx + 1 * (size_t)width * (size_t)height];
			float b = rgb[idx + 2 * (size_t)width * (size_t)height];
			r = r < 0.0f ? 0.0f : (r > 1.0f ? 1.0f : r);
			g = g < 0.0f ? 0.0f : (g > 1.0f ? 1.0f : g);
			b = b < 0.0f ? 0.0f : (b > 1.0f ? 1.0f : b);
			uint8_t R = (uint8_t)(r * 255.0f + 0.5f);
			uint8_t G = (uint8_t)(g * 255.0f + 0.5f);
			uint8_t B = (uint8_t)(b * 255.0f + 0.5f);
			size_t o = idx * 4;
			rgba[o + 0] = R;
			rgba[o + 1] = G;
			rgba[o + 2] = B;
			rgba[o + 3] = 255;
		}
	}
	return SaveRgba8ToPng(width, height, rgba.data(), path);
}

void SaliencyPredictor::Process(ID3D11Texture2D* srcTexture) {
	if (!m_initialized || !srcTexture) return;

	auto t_begin = std::chrono::high_resolution_clock::now();

	// Debug dump 1: raw input source texture
	if (g_enable_debug_images) {
		std::string sp = get_path_head() + std::string("debug_1_input_src_") + std::to_string((long long)(m_frameCounter + 1)) + ".png";
		std::wstring wp(sp.begin(), sp.end());
		SaveTextureToPng(m_d3d->GetDevice(), m_d3d->GetContext(), srcTexture, wp);
	}

	bool gpu_path_ok = EnsureGpuDownscalePipeline(srcTexture)
#ifdef ALVR_SALIENCY
		&& EnsureCudaInterop()
#endif
		;
#ifdef ALVR_SALIENCY
	torch::Tensor x_from_cuda;
#endif
#ifdef ALVR_SALIENCY
	if (g_use_gpu_downscale && gpu_path_ok) {
		// Update uv rect to left half [0,0]-[0.5,1]
		D3D11_MAPPED_SUBRESOURCE m{};
		if (SUCCEEDED(m_d3d->GetContext()->Map(m_uvCB.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &m))) {
			float* p = (float*)m.pData; p[0]=0.0f; p[1]=0.0f; p[2]=0.5f; p[3]=1.0f;
			m_d3d->GetContext()->Unmap(m_uvCB.Get(), 0);
		}
		// Draw fullscreen triangle downscale
		auto ctx = m_d3d->GetContext();
		// Build a per-frame SRV from a single-sample texture (resolve or copy from source)
		ComPtr<ID3D11Texture2D> srcCopyTex;
		ComPtr<ID3D11ShaderResourceView> srcCopySRV;
		{
			D3D11_TEXTURE2D_DESC sdescSrc{};
			srcTexture->GetDesc(&sdescSrc);
			Info("SaliencyPredictor: src desc w=%u h=%u fmt=%d samples=%u bind=%u\n", sdescSrc.Width, sdescSrc.Height, (int)sdescSrc.Format, sdescSrc.SampleDesc.Count, sdescSrc.BindFlags);
			D3D11_TEXTURE2D_DESC tdesc{};
			tdesc.Width = sdescSrc.Width;
			tdesc.Height = sdescSrc.Height;
			tdesc.MipLevels = 1;
			tdesc.ArraySize = 1;
			DXGI_FORMAT srvFormat = sdescSrc.Format;
			DXGI_FORMAT copyFormat = sdescSrc.Format;
			switch (sdescSrc.Format) {
			case DXGI_FORMAT_R8G8B8A8_UNORM:
			case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
			case DXGI_FORMAT_R8G8B8A8_TYPELESS:
				copyFormat = DXGI_FORMAT_R8G8B8A8_TYPELESS;
				srvFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
				break;
			case DXGI_FORMAT_B8G8R8A8_UNORM:
			case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
			case DXGI_FORMAT_B8G8R8A8_TYPELESS:
				copyFormat = DXGI_FORMAT_B8G8R8A8_TYPELESS;
				srvFormat = DXGI_FORMAT_B8G8R8A8_UNORM;
				break;
			default:
				copyFormat = sdescSrc.Format;
				srvFormat = sdescSrc.Format;
				break;
			}
			tdesc.Format = copyFormat;
			tdesc.SampleDesc.Count = 1;
			tdesc.SampleDesc.Quality = 0;
			tdesc.Usage = D3D11_USAGE_DEFAULT;
			tdesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
			tdesc.CPUAccessFlags = 0;
			tdesc.MiscFlags = 0;
			if (FAILED(m_d3d->GetDevice()->CreateTexture2D(&tdesc, nullptr, &srcCopyTex))) {
				Error("SaliencyPredictor: CreateTexture2D(srcCopyTex) failed\n");
				return;
			}
			if (sdescSrc.SampleDesc.Count > 1) {
				ctx->ResolveSubresource(srcCopyTex.Get(), 0, srcTexture, 0, sdescSrc.Format);
			} else {
				ctx->CopyResource(srcCopyTex.Get(), srcTexture);
			}
			// Create SRV for the copy with exact source format
			{
				D3D11_SHADER_RESOURCE_VIEW_DESC srvd{};
				srvd.Format = srvFormat;
				srvd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
				srvd.Texture2D.MostDetailedMip = 0;
				srvd.Texture2D.MipLevels = 1;
				if (FAILED(m_d3d->GetDevice()->CreateShaderResourceView(srcCopyTex.Get(), &srvd, &srcCopySRV))) {
					Error("SaliencyPredictor: CreateShaderResourceView(srcCopyTex) failed\n");
					return;
				}
			}
			// Debug: dump the resolved/copy source as PNG
			if (g_enable_debug_images) {
				std::string sp = get_path_head() + std::string("debug_1b_src_resolved_") + std::to_string((long long)(m_frameCounter + 1)) + ".png";
				std::wstring wp(sp.begin(), sp.end());
				SaveTextureToPng(m_d3d->GetDevice(), m_d3d->GetContext(), srcCopyTex.Get(), wp);
			}
		}
		// Create a tiny 2x2 test texture to validate sampling path
		ComPtr<ID3D11Texture2D> testTex;
		ComPtr<ID3D11ShaderResourceView> testSRV;
		{
			UINT w = 2, h = 2;
			D3D11_TEXTURE2D_DESC td{};
			td.Width = w; td.Height = h; td.MipLevels = 1; td.ArraySize = 1;
			td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
			td.SampleDesc.Count = 1; td.SampleDesc.Quality = 0;
			td.Usage = D3D11_USAGE_DEFAULT; td.BindFlags = D3D11_BIND_SHADER_RESOURCE; td.CPUAccessFlags = 0; td.MiscFlags = 0;
			const UINT pitch = w * 4;
			UINT8 pixels[16] = {
				255, 0, 0, 255,   0, 255, 0, 255,
				0, 0, 255, 255,   255, 255, 0, 255
			};
			D3D11_SUBRESOURCE_DATA init{}; init.pSysMem = pixels; init.SysMemPitch = pitch;
			if (SUCCEEDED(m_d3d->GetDevice()->CreateTexture2D(&td, &init, &testTex))) {
				D3D11_SHADER_RESOURCE_VIEW_DESC sd{}; sd.Format = td.Format; sd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D; sd.Texture2D.MostDetailedMip = 0; sd.Texture2D.MipLevels = 1;
				m_d3d->GetDevice()->CreateShaderResourceView(testTex.Get(), &sd, &testSRV);
			}
		}
		// CPU-read src via staging and create an upload texture SRV at t2
		ComPtr<ID3D11Texture2D> uploadTex;
		ComPtr<ID3D11ShaderResourceView> uploadSRV;
		{
			D3D11_TEXTURE2D_DESC sdesc{}; srcTexture->GetDesc(&sdesc);
			D3D11_TEXTURE2D_DESC stg = sdesc; stg.Usage = D3D11_USAGE_STAGING; stg.BindFlags = 0; stg.CPUAccessFlags = D3D11_CPU_ACCESS_READ; stg.SampleDesc.Count = 1; stg.SampleDesc.Quality = 0;
			ComPtr<ID3D11Texture2D> staging;
			if (SUCCEEDED(m_d3d->GetDevice()->CreateTexture2D(&stg, nullptr, &staging))) {
				ctx->CopyResource(staging.Get(), srcTexture);
				D3D11_MAPPED_SUBRESOURCE ms{};
				if (SUCCEEDED(ctx->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &ms))) {
					D3D11_TEXTURE2D_DESC ud{}; ud.Width = sdesc.Width; ud.Height = sdesc.Height; ud.MipLevels = 1; ud.ArraySize = 1; ud.Format = DXGI_FORMAT_R8G8B8A8_UNORM; ud.SampleDesc.Count = 1; ud.Usage = D3D11_USAGE_DEFAULT; ud.BindFlags = D3D11_BIND_SHADER_RESOURCE;
					D3D11_SUBRESOURCE_DATA sd{}; sd.pSysMem = ms.pData; sd.SysMemPitch = ms.RowPitch;
					// CPU inspect a central pixel from the mapped source
					{
						UINT cx = sdesc.Width / 4; UINT cy = sdesc.Height / 2;
						const uint8_t* row = reinterpret_cast<const uint8_t*>(ms.pData) + cy * ms.RowPitch;
						const uint8_t* px = row + cx * 4;
						Info("SaliencyPredictor: CPU center px (x=%u,y=%u) = RGBA(%u,%u,%u,%u)\n", cx, cy, (unsigned)px[0], (unsigned)px[1], (unsigned)px[2], (unsigned)px[3]);
					}
					if (SUCCEEDED(m_d3d->GetDevice()->CreateTexture2D(&ud, &sd, &uploadTex))) {
						D3D11_SHADER_RESOURCE_VIEW_DESC srd{}; srd.Format = ud.Format; srd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D; srd.Texture2D.MostDetailedMip = 0; srd.Texture2D.MipLevels = 1;
						m_d3d->GetDevice()->CreateShaderResourceView(uploadTex.Get(), &srd, &uploadSRV);
						// Debug: dump the upload texture as PNG
						if (g_enable_debug_images) {
							std::string sp = get_path_head() + std::string("debug_1c_upload_") + std::to_string((long long)(m_frameCounter + 1)) + ".png";
							std::wstring wp(sp.begin(), sp.end());
							SaveTextureToPng(m_d3d->GetDevice(), m_d3d->GetContext(), uploadTex.Get(), wp);
						}
					}
					ctx->Unmap(staging.Get(), 0);
				}
			}
		}
		ID3D11RenderTargetView* rt = m_dsRTV.Get();
		ctx->OMSetRenderTargets(1, &rt, nullptr);
		ctx->RSSetViewports(1, &m_dsViewport);
		if (m_rsNoCull) ctx->RSSetState(m_rsNoCull.Get());
		// Clear RT to a visible color for debugging
		const float clearColor[4] = {0,1,0,1};
		ctx->ClearRenderTargetView(m_dsRTV.Get(), clearColor);
		// Debug dump: after clear, before draw
		if (g_enable_debug_images) {
			std::string sp = get_path_head() + std::string("debug_2a_before_draw_") + std::to_string((long long)(m_frameCounter + 1)) + ".png";
			std::wstring wp(sp.begin(), sp.end());
			SaveTextureToPng(m_d3d->GetDevice(), m_d3d->GetContext(), m_dsTex.Get(), wp);
		}
		ctx->VSSetShader(m_vs.Get(), nullptr, 0);
		ctx->PSSetShader(m_ps.Get(), nullptr, 0);
		// Bind UV rect constant buffer to PS
		ID3D11Buffer* pscbs[1] = { m_uvCB.Get() };
		ctx->PSSetConstantBuffers(0, 1, pscbs);
		// Ensure null input layout for vertex-id based VS
		ctx->IASetInputLayout(nullptr);
		// Clear SRV slot 0 before binding new SRV
		ID3D11ShaderResourceView* nullSRV0[1] = { nullptr };
		ctx->PSSetShaderResources(0, 1, nullSRV0);
		// Bind src copy SRV to t0 only
		ID3D11ShaderResourceView* tex0 = (uploadSRV.Get() != nullptr) ? uploadSRV.Get() : srcCopySRV.Get();
		Info("SaliencyPredictor: binding t0=%s (upload=%p, srcCopy=%p)\n", (uploadSRV.Get()?"upload":"srcCopy"), uploadSRV.Get(), srcCopySRV.Get());
		ctx->PSSetShaderResources(0, 1, &tex0);
		ID3D11SamplerState* samp = m_linearSampler.Get();
		ctx->PSSetSamplers(0, 1, &samp);
		ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		// Bind opaque blend state
		FLOAT blendFactor[4] = {0,0,0,0}; UINT sampleMask = 0xFFFFFFFF;
		if (m_bsOpaque) ctx->OMSetBlendState(m_bsOpaque.Get(), blendFactor, sampleMask);
		ctx->Draw(3, 0);
		// Debug dump: after draw
		if (g_enable_debug_images) {
			std::string sp = get_path_head() + std::string("debug_2b_after_draw_") + std::to_string((long long)(m_frameCounter + 1)) + ".png";
			std::wstring wp(sp.begin(), sp.end());
			SaveTextureToPng(m_d3d->GetDevice(), m_d3d->GetContext(), m_dsTex.Get(), wp);
		}
		// Unbind SRV to avoid D3D warnings if src is also bound elsewhere
		ID3D11ShaderResourceView* nulls[1] = { nullptr };
		ctx->PSSetShaderResources(0, 1, nulls);

		// Debug dump 2: left-half downscaled (GPU path) render target
		if (g_enable_debug_images) {
			std::string sp = get_path_head() + std::string("debug_2_left_gpu_") + std::to_string((long long)(m_frameCounter + 1)) + ".png";
			std::wstring wp(sp.begin(), sp.end());
			SaveTextureToPng(m_d3d->GetDevice(), m_d3d->GetContext(), m_dsTex.Get(), wp);
		}

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
			// ROI: left half of the source
			int roiX = 0;
			int roiY = 0;
			int roiW = (int)m_srcWidth / 2;
			int roiH = (int)m_srcHeight;
			DownscaleBilinear(src, mapped.RowPitch, roiX, roiY, roiW, roiH);
			m_d3d->GetContext()->Unmap(m_readbackTex.Get(), 0);
			// Debug dump 3: after CPU resize (from m_downscaled)
			if (g_enable_debug_images) {
				std::string sp = get_path_head() + std::string("debug_3_after_cpu_resize_") + std::to_string((long long)(m_frameCounter + 1)) + ".png";
				std::wstring wp(sp.begin(), sp.end());
				SaveFloatRgbToPng(kOutW, kOutH, m_downscaled.data(), wp);
			}
			auto options = torch::TensorOptions().dtype(torch::kFloat32);
			x = torch::from_blob(m_downscaled.data(), {1, 3, kOutH, kOutW}, options).clone();
			if (module_on_cuda) {
				x = x.to(torch::kCUDA, torch::kHalf);
			}
		}

		// Apply ImageNet normalization: (x - mean) / std, with broadcasting
		{
			auto norm_opts = x.options();
			auto mean = torch::tensor({0.485f, 0.456f, 0.406f}, norm_opts).view({1, 3, 1, 1});
			auto std = torch::tensor({0.229f, 0.224f, 0.225f}, norm_opts).view({1, 3, 1, 1});
			x = x.sub(mean).div(std);
		}

		// Debug dump 4: final model input tensor (saved as PNG from CPU float)
		if (g_enable_debug_images) {
			auto to_save = x;
			if (to_save.is_cuda()) to_save = to_save.to(torch::kCPU);
			if (to_save.scalar_type() != torch::kFloat32) to_save = to_save.to(torch::kFloat32);
			to_save = to_save.contiguous();
			auto chw = to_save.squeeze(); // [3,H,W]
			if (chw.dim() == 3 && chw.size(0) == 3) {
				int H = (int)chw.size(1);
				int W = (int)chw.size(2);
				std::vector<float> cpu((size_t)3 * (size_t)H * (size_t)W);
				std::memcpy(cpu.data(), chw.data_ptr<float>(), cpu.size() * sizeof(float));
				std::string sp = get_path_head() + std::string("debug_4_final_input_") + std::to_string((long long)(m_frameCounter + 1)) + ".png";
				std::wstring wp(sp.begin(), sp.end());
				SaveFloatRgbToPng(W, H, cpu.data(), wp);
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
				int ksz = 10; float sigma = 3.0f;
				auto kernel = make_gaussian_kernel_2d(ksz, sigma, s.device(), work_dtype);
				// If CPU and original dtype is half, upcast for conv2d
				auto s_conv = s_norm;
				if (!on_cuda && s_conv.scalar_type() != work_dtype) s_conv = s_conv.to(work_dtype);
				int pad = ksz / 2;
				auto s_blur = torch::conv2d(s_conv, kernel, {}, {1,1}, {pad, pad});
				// Cast back to original dtype if needed
				if (s_blur.scalar_type() != s.scalar_type()) s_blur = s_blur.to(s.scalar_type());
				m_lastSaliency = s_blur;
				// Dump every 100 frames right after post-process
				bool shouldDump = ((m_frameCounter + 1) % 100) == 0;
				if (shouldDump) {
					auto cpu2d = m_lastSaliency.squeeze().to(torch::kFloat32).to(torch::kCPU).contiguous();
					int h = (int)cpu2d.size(0);
					int w = (int)cpu2d.size(1);
					std::ofstream ofs(get_path_head()+"saliency_"+std::to_string((long long)(m_frameCounter+1))+".csv");
					const float* ptr = cpu2d.data_ptr<float>();
					for (int y = 0; y < h; ++y) {
						for (int x = 0; x < w; ++x) {
							ofs << ptr[y * w + x];
							if (x + 1 < w) ofs << ",";
						}
						ofs << "\n";
					}
				}
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

#ifdef ALVR_SALIENCY
bool SaliencyPredictor::GetLastSaliencyCpu(std::vector<float>& out, int& width, int& height) {
	if (!m_lastSaliency.defined()) return false;
	auto s = m_lastSaliency;
	// Expect [1,1,H,W]
	if (s.dim() == 4 && s.size(0) == 1 && s.size(1) == 1) {
		width = (int)s.size(3);
		height = (int)s.size(2);
	} else if (s.dim() == 2) {
		height = (int)s.size(0);
		width = (int)s.size(1);
	} else {
		return false;
	}
	auto cpu = s.squeeze().to(torch::kFloat32).to(torch::kCPU).contiguous();
	out.resize((size_t)width * (size_t)height);
	std::memcpy(out.data(), cpu.data_ptr<float>(), out.size() * sizeof(float));
	return true;
}
#else
bool SaliencyPredictor::GetLastSaliencyCpu(std::vector<float>&, int&, int&) { return false; }
#endif 