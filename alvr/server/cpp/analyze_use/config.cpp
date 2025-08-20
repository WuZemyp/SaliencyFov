#include "config.h"

bool save_rframe_lock = false;// for saving original game frame and foveated rendered frame
bool save_eframe_lock = false;// for saving encoded game frame

bool get_eframe_lock() {
    return save_eframe_lock;
}

bool get_rframe_lock() {
    return save_rframe_lock;
}

std::string filename_s = "C:\\Program Files (x86)\\Steam\\steamapps\\common\\SteamVR\\records\\";

std::string get_path_head(){
    return filename_s;
}

// Run saliency inference every N frames (>=1). Can be changed at runtime.
int g_infer_every_n_frames = 2;

// Enable CSV dumps by default? Keep off for performance.
bool g_enable_qpmap_csv_dumps = false;

// QP const and QO_Max in NvEncoder

// CPU resize vs GPU resize config in SaliencyPredictor

//check modelPath = in SaliencyPredictor