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

SaliencyPredictor::SaliencyPredictor(std::shared_ptr<CD3DRender> d3dRender)
	: m_d3d(std::move(d3dRender)) {}

SaliencyPredictor::~SaliencyPredictor() {}

bool SaliencyPredictor::Initialize() {
	m_downscaled.resize(3 * kOutH * kOutW);
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

void SaliencyPredictor::Process(ID3D11Texture2D* srcTexture) {
	if (!m_initialized || !srcTexture) return;
	if (!EnsureReadbackBuffer(srcTexture)) return;

	auto t_begin = std::chrono::high_resolution_clock::now();
	// Copy GPU -> CPU staging
	m_d3d->GetContext()->CopyResource(m_readbackTex.Get(), srcTexture);
	D3D11_MAPPED_SUBRESOURCE mapped{};
	if (FAILED(m_d3d->GetContext()->Map(m_readbackTex.Get(), 0, D3D11_MAP_READ, 0, &mapped))) {
		return;
	}
	const uint8_t* src = reinterpret_cast<const uint8_t*>(mapped.pData);
	DownscaleBilinear(src, mapped.RowPitch);
	m_d3d->GetContext()->Unmap(m_readbackTex.Get(), 0);
	auto t_after_resize = std::chrono::high_resolution_clock::now();

#ifdef ALVR_SALIENCY
	auto t_before_infer = t_after_resize;
	if (m_modelLoaded) {
		// Build input tensor (1,3,192,256) float
		auto options = torch::TensorOptions().dtype(torch::kFloat32);
		torch::Tensor x = torch::from_blob(m_downscaled.data(), {1, 3, kOutH, kOutW}, options).clone();
		static torch::Tensor hidden; // persist across frames; undefined at start
		std::vector<torch::jit::IValue> inputs;
		inputs.emplace_back(x);
		if (hidden.defined()) {
			inputs.emplace_back(hidden);
		} else {
			inputs.emplace_back(torch::jit::IValue()); // None for Optional[Tensor]
		}
		try {
			auto out = m_module.forward(inputs).toTuple();
			torch::Tensor sal = out->elements()[0].toTensor();
			hidden = out->elements()[1].toTensor();
			m_lastSaliency = sal.detach();
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
	auto t_after_infer = std::chrono::high_resolution_clock::now();
	double ms_resize = std::chrono::duration<double, std::milli>(t_after_resize - t_begin).count();
	double ms_infer = std::chrono::duration<double, std::milli>(t_after_infer - t_after_resize).count();
	double ms_total = std::chrono::duration<double, std::milli>(t_after_infer - t_begin).count();
	Info("Saliency latency: resize+copy=%.3fms infer=%.3fms total=%.3fms\n", ms_resize, ms_infer, ms_total);

	// Dump saliency to CSV on every frame
	m_frameCounter++;
	if (m_lastSaliency.defined()) {
		try {
			auto sal = m_lastSaliency.squeeze(); // (1,1,H,W) -> (H,W) or (1,H,W) -> (H,W)
			sal = sal.contiguous();
			const int H = (int)sal.size(-2);
			const int W = (int)sal.size(-1);
			std::ostringstream fname;
			fname << "saliency_" << std::setw(6) << std::setfill('0') << m_frameCounter << ".csv";
			auto out_path = (std::filesystem::current_path() / fname.str()).string();
			std::ofstream ofs(out_path, std::ios::out | std::ios::trunc);
			if (!ofs.is_open()) {
				Error("SaliencyPredictor: failed to open CSV for write: %s (errno=%d %s)\n", out_path.c_str(), errno, std::strerror(errno));
			} else {
				ofs.setf(std::ios::fixed); ofs<<std::setprecision(6);
				auto cpu = sal.cpu();
				if (cpu.dtype() == torch::kFloat32) {
					float* data = cpu.data_ptr<float>();
					for (int y = 0; y < H; ++y) {
						for (int x = 0; x < W; ++x) {
							ofs << data[y * W + x];
							if (x + 1 < W) ofs << ",";
						}
						ofs << "\n";
					}
					Info("SaliencyPredictor: wrote CSV %s (%dx%d)\n", out_path.c_str(), W, H);
				} else {
					Error("SaliencyPredictor: unexpected tensor dtype, csv not written\n");
				}
				ofs.close();
			}
		} catch (const std::exception& ex) {
			Error("SaliencyPredictor: failed to write saliency csv: %s\n", ex.what());
		} catch (...) {
			Error("SaliencyPredictor: failed to write saliency csv (unknown)\n");
		}
	} else {
		Error("SaliencyPredictor: saliency undefined, skipping CSV\n");
	}
#endif
} 