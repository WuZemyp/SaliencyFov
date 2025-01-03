#include "config.h"

bool save_rframe_lock = false;// for saving original game frame and foveated rendered frame
bool save_eframe_lock = false;// for saving encoded game frame

bool get_eframe_lock() {
    return save_eframe_lock;
}

bool get_rframe_lock() {
    return save_rframe_lock;
}

std::string filename_s = "C:\\Users\\Ze\\Desktop\\mobisys\\frame_data\\";

std::string get_path_head(){
    return filename_s;
}