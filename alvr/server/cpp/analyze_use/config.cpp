#include "config.h"

bool save_rframe_lock = true;
bool save_eframe_lock = false;

bool get_eframe_lock() {
    return save_eframe_lock;
}

bool get_rframe_lock() {
    return save_rframe_lock;
}

std::string filename_s = "C:\\Users\\13513\\ALVR_Private\\NSDI\\FovOptix_dynamicFoveation\\build\\alvr_streamer_windows\\";

std::string get_path_head(){
    return filename_s;
}