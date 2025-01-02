<p align="center"> <img width="500" src="resources/alvr_combined_logo_hq.png"/> </p>

# Frame Data Collection
We provide the game frame data collection for original game frame, foveated rendered frame, encoded game frame. In case you need it, please check the following proceudres, otherwise, it should be closed.

The configuration files path: alvr\server\cpp\analyze_use

config.cpp -> save_rframe_lock controls the saving of original game frame and foveated rendered frame

config.cpp -> save_eframe_lock controls the saving of encoded game frame

config.cpp -> filename_s controls the saving folder path

helper_f.cpp -> save_frame_feq controls the frame saving frequency(e.g.,  save_frame_feq = 500 -> Every 500 frames, save once. Notably, the saving process cost computational resource, a high saving frequency may be harmful for the main system.)

# System Pipeline Latency Collection
We provide the system pipeline latency collection, for each game frame, the system records the latency of each pipeline component. 

alvr\server\src\EyeNexus_Config.rs -> STATISTICS_FILE_PATH controls the saving file path.

For more details, please refer to the function report_statistics_MTP() in alvr\server\src\statistics.rs | Please ignoring the statistics file generarted by the function write_latency_to_csv() in alvr\server\src\statistics.rs->report_statistics(), this may contains some records with error in latency.

# Eye Gaze Process
Server receives the tracking packets and extract eye gaze raw data.

For more details, please refer to the tracking_receive_thread and compute_eye_gaze_location() in alvr\server\src\connection.rs.

compute_eye_gaze_location() returns the eye gaze projection location for both left eye frame and right eye frame -> (leftx, lefty, rightx, righty)

The results for every received eye gaze data will be saved in a csv file. (please update the file path in alvr\server\src\EyeNexus_Config.rs -> EYEGAZEPROCESSING_FILE_PATH)

# Dynamic Foveated Rendering 
We integrate the dynamic foveated rendering with the original ALVR server rendering code. When the eye gaze projection location changes, we re-initialize the foveated rendering shader with newest eye gaze point and do foveated rendering.

Check alvr\server\cpp\platform\win32\CEncoder.cpp -> CopyToStaging() for more details.

# Dynamic Foveated Video Encoding
alvr\server\cpp\platform\win32\VideoEncoderNVENC.cpp -> GenQPDeltaMap()

We integrate the FVE with the original ALVR frame encoding code. However, as the foveation center of foveated rendering changes during streaming, we need to update the params required during the gaze point projection (More details in paper). The high level procedure can be described as : 1. Collect foveation controller C from network monitoring component. 2. Collect eye gaze location in original frame. 3. Updating the projection required params. (alvr\server\cpp\platform\win32\NvEncoder.cpp -> Update_decompress_params()) 4. Project the eye gaze location from original frame to foveated rendered frame. (alvr\server\cpp\platform\win32\NvEncoder.cpp -> decompress_x(), decompress_y()) 5. Calculate the QO for each marco block in QP Map (alvr\server\cpp\platform\win32\NvEncoder.cpp -> EyeNexus_CalculateQPOffsetValue_leftEye(), EyeNexus_CalculateQPOffsetValue_rightEye())6. Do encoding








# ALVR - Air Light VR

[![badge-discord][]][link-discord] [![badge-matrix][]][link-matrix] [![badge-opencollective][]][link-opencollective]

Stream VR games from your PC to your headset via Wi-Fi.  
ALVR uses technologies like [Asynchronous Timewarp](https://developer.oculus.com/documentation/native/android/mobile-timewarp-overview) and [Fixed Foveated Rendering](https://developer.oculus.com/documentation/native/android/mobile-ffr) for a smoother experience.  
Most of the games that run on SteamVR or Oculus Software (using Revive) should work with ALVR.  
This is a fork of [ALVR](https://github.com/polygraphene/ALVR).

|      VR Headset       |                                Support                                 |
| :-------------------: | :--------------------------------------------------------------------: |
|    Quest 1/2/3/Pro    |                           :heavy_check_mark:                           |
|     Pico 4/Neo 3      |                           :heavy_check_mark:                           |
| Vive Focus 3/XR Elite |                           :heavy_check_mark:                           |
|        YVR 1/2        |                           :heavy_check_mark:                           |
|        Lynx R1        |                           :heavy_check_mark:                           |
|   Smartphone/Monado   |                              :warning: *                               |
|   Google Cardboard    | :warning: * ([PhoneVR](https://github.com/PhoneVR-Developers/PhoneVR)) |
|        GearVR         |                         :construction: (maybe)                         |
|       Oculus Go       |                                 :x: **                                 |

\* : Only works on some smartphones, not enough testing.  
\** : Oculus Go support was dropped, the minimum supported OS is Android 8. Download the last compatible version [here](https://github.com/alvr-org/ALVR/releases/tag/v18.2.3).

|        PC OS        |       Support       |
| :-----------------: | :-----------------: |
|   Windows 8/10/11   | :heavy_check_mark:  |
|    Windows 7/XP     |         :x:         |
|     Ubuntu/Arch     |    :warning: ***    |
| Other linux distros | :grey_question: *** |
|        macOS        |         :x:         |

\*** : Linux support is still in beta. To be able to make audio work or run ALVR at all you may need advanced knowledge of your distro for debugging or building from source.

## Requirements

-   A supported standalone VR headset (see compatibility table above)

-   SteamVR

-   High-end gaming PC
    -   See OS compatibility table above.
    -   NVIDIA GPU that supports NVENC (1000 GTX Series or higher) (or with an AMD GPU that supports AMF VCE) with the latest driver.
    -   Laptops with an onboard (Intel HD, AMD iGPU) and an additional dedicated GPU (NVidia GTX/RTX, AMD HD/R5/R7): you should assign the dedicated GPU or "high performance graphics adapter" to the applications ALVR, SteamVR for best performance and compatibility. (NVidia: Nvidia control panel->3d settings->application settings; AMD: similiar way)

-   802.11ac 5Ghz wireless or ethernet wired connection  
    -   It is recommended to use 802.11ac 5Ghz for the headset and ethernet for PC  
    -   You need to connect both the PC and the headset to same router (or use a routed connection as described [here](https://github.com/alvr-org/ALVR/wiki/ALVR-v14-and-Above))

## Install

Follow the installation guide [here](https://github.com/alvr-org/ALVR/wiki/Installation-guide).

## Troubleshooting

-   Please check the [Troubleshooting](https://github.com/alvr-org/ALVR/wiki/Troubleshooting) page. The original repository [wiki](https://github.com/polygraphene/ALVR/wiki/Troubleshooting) can also help.  
-   Configuration recommendations and information may be found [here](https://github.com/alvr-org/ALVR/wiki/PC)

## Uninstall

Open `ALVR Dashboard.exe`, go to `Installation` tab then press `Remove firewall rules`. Close ALVR window and delete the ALVR folder.

## Build from source

You can follow the guide [here](https://github.com/alvr-org/ALVR/wiki/Building-From-Source).

## License

ALVR is licensed under the [MIT License](LICENSE).

## Privacy policy

ALVR apps do not directly collect any kind of data.

## Donate

If you want to support this project you can make a donation to our [Open Source Collective account](https://opencollective.com/alvr).

You can also donate to the original author of ALVR using Paypal (polygraphene@gmail.com) or with bitcoin (1FCbmFVSjsmpnAj6oLx2EhnzQzzhyxTLEv).

[badge-discord]: https://img.shields.io/discord/720612397580025886?style=for-the-badge&logo=discord&color=5865F2 "Join us on Discord"
[link-discord]: https://discord.gg/ALVR
[badge-matrix]: https://img.shields.io/static/v1?label=chat&message=%23alvr&style=for-the-badge&logo=matrix&color=blueviolet "Join us on Matrix"
[link-matrix]: https://matrix.to/#/#alvr:ckie.dev?via=ckie.dev
[badge-opencollective]: https://img.shields.io/opencollective/all/alvr?style=for-the-badge&logo=opencollective&color=79a3e6 "Donate"
[link-opencollective]: https://opencollective.com/alvr
