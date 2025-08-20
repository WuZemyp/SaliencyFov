#include "CEncoder.h"
#include "alvr_server/Settings.h"
#include "VideoEncoderNVENC.h"
#include "../../analyze_use/config.h"

		CEncoder::CEncoder()
			: m_bExiting(false)
			, m_targetTimestampNs(0)
			, m_gaze_location_leftx(1072)
			, m_gaze_location_lefty(1168)
			, m_gaze_location_rightx(3216)
			, m_gaze_location_righty(1168)
			, fr_freq(5)
		{
			m_encodeFinished.Set();
			last_centerShiftX = 0.4;
			last_centerShiftY = 0.1;
			times = 0;
		}

		
		CEncoder::~CEncoder()
		{
			if (m_videoEncoder)
			{
				m_videoEncoder->Shutdown();
				m_videoEncoder.reset();
			}
		}

		void CEncoder::Initialize(std::shared_ptr<CD3DRender> d3dRender) {
			m_FrameRender = std::make_shared<FrameRender>(d3dRender);
			m_FrameRender->Startup();
			uint32_t encoderWidth, encoderHeight;
			m_FrameRender->GetEncodingResolution(&encoderWidth, &encoderHeight);
			m_saliency = std::make_unique<SaliencyPredictor>(d3dRender);
			m_saliency->Initialize();

			Exception vceException;
			Exception nvencException;
#ifdef ALVR_GPL
			Exception swException;
			if (Settings::Instance().m_force_sw_encoding) {
				try {
					Debug("Try to use VideoEncoderSW.\n");
					m_videoEncoder = std::make_shared<VideoEncoderSW>(d3dRender, encoderWidth, encoderHeight);
					m_videoEncoder->Initialize();
					return;
				}
				catch (Exception e) {
					swException = e;
				}
			}
#endif
			
			try {
				Debug("Try to use VideoEncoderAMF.\n");
				m_videoEncoder = std::make_shared<VideoEncoderAMF>(d3dRender, encoderWidth, encoderHeight);
				m_videoEncoder->Initialize();
				return;
			}
			catch (Exception e) {
				vceException = e;
			}
			try {
				Debug("Try to use VideoEncoderNVENC.\n");
				m_videoEncoder = std::make_shared<VideoEncoderNVENC>(d3dRender, encoderWidth, encoderHeight);
				m_videoEncoder->Initialize();
				return;
			}
			catch (Exception e) {
				nvencException = e;
			}
#ifdef ALVR_GPL
			try {
				Debug("Try to use VideoEncoderSW.\n");
				m_videoEncoder = std::make_shared<VideoEncoderSW>(d3dRender, encoderWidth, encoderHeight);
				m_videoEncoder->Initialize();
				return;
			}
			catch (Exception e) {
				swException = e;
			}
			throw MakeException("All VideoEncoder are not available. VCE: %s, NVENC: %s, SW: %s", vceException.what(), nvencException.what(), swException.what());
#else
			throw MakeException("All VideoEncoder are not available. VCE: %s, NVENC: %s", vceException.what(), nvencException.what());
#endif
		}

		bool CEncoder::CopyToStaging(ID3D11Texture2D *pTexture[][2], vr::VRTextureBounds_t bounds[][2], int layerCount, bool recentering
			, uint64_t presentationTime, uint64_t targetTimestampNs, const std::string& message, const std::string& debugText)
		{
			m_presentationTime = presentationTime;
			m_targetTimestampNs = targetTimestampNs;
			m_FrameRender->RenderFrame(pTexture, bounds, layerCount, recentering, message, debugText);
			// Run saliency inference with configurable frequency; reuse last result on skipped frames
			ReportComposed(m_targetTimestampNs, 0);
			if (m_saliency) {
				m_inferFrameCounter++;
				int every = get_infer_every_n_frames();
				if ((m_inferFrameCounter % (uint64_t)every) == 1) { // infer on the first of each N
					m_saliency->Process(m_FrameRender->GetTexture(false, m_targetTimestampNs).Get());
				}
				ReportInferenced(m_targetTimestampNs, 0);
				// Forward latest saliency map (left eye) to encoder for QP map generation
				std::vector<float> saliency;
				int salW = 0, salH = 0;
				if (m_saliency->GetLastSaliencyCpu(saliency, salW, salH)) {
					if (m_videoEncoder) {
						auto *nvenc = dynamic_cast<VideoEncoderNVENC*>(m_videoEncoder.get());
						if (nvenc) {
							nvenc->SetSaliencyMap(saliency, salW, salH, m_targetTimestampNs);
						}
					}
				}
			}else{
				ReportInferenced(m_targetTimestampNs, 0);
			}
			return true;
		}

		void CEncoder::Run()
		{
			Debug("CEncoder: Start thread. Id=%d\n", GetCurrentThreadId());
			SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_MOST_URGENT);

			while (!m_bExiting)
			{
				m_newFrameReady.Wait();
				if (m_bExiting)
					break;

				if (m_FrameRender->GetTexture(false,m_targetTimestampNs))
				{
					m_videoEncoder->Transmit(m_FrameRender->GetTexture(true,m_targetTimestampNs).Get(), m_presentationTime, m_targetTimestampNs, m_scheduler.CheckIDRInsertion(),m_gaze_location_leftx,m_gaze_location_lefty,m_gaze_location_rightx,m_gaze_location_righty,last_centerShiftX,last_centerShiftY);
				}

				m_encodeFinished.Set();
			}
		}

		void CEncoder::Stop()
		{
			m_bExiting = true;
			m_newFrameReady.Set();
			Join();
			m_FrameRender.reset();
		}

		void CEncoder::NewFrameReady()
		{
			m_encodeFinished.Reset();
			m_newFrameReady.Set();
		}

		void CEncoder::WaitForEncode()
		{
			m_encodeFinished.Wait();
		}

		void CEncoder::OnStreamStart() {
			m_scheduler.OnStreamStart();
		}

		void CEncoder::OnPacketLoss() {
			m_scheduler.OnPacketLoss();
		}

		void CEncoder::InsertIDR() {
			m_scheduler.InsertIDR();
		}

		void CEncoder::CaptureFrame() {
		}
