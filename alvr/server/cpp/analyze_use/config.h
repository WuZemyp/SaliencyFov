#pragma once
#include <string>

extern bool save_rframe_lock;
extern bool save_eframe_lock;
extern std::string filename_s;

extern std::string get_path_head();
extern bool get_eframe_lock();

extern bool get_rframe_lock();

// Inference frequency: run saliency every N frames (>=1). Default 1
extern int g_infer_every_n_frames;
inline int get_infer_every_n_frames() { return g_infer_every_n_frames < 1 ? 1 : g_infer_every_n_frames; }

// Enable/disable CSV dumps (saliency, QP map)
extern bool g_enable_qpmap_csv_dumps;
inline bool get_enable_csv_dumps() { return g_enable_qpmap_csv_dumps; }