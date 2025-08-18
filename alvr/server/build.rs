use std::{env, path::PathBuf};

fn get_ffmpeg_path() -> PathBuf {
    let ffmpeg_path = alvr_filesystem::deps_dir()
        .join(if cfg!(target_os = "linux") {
            "linux"
        } else {
            "windows"
        })
        .join("ffmpeg");

    if cfg!(target_os = "linux") {
        ffmpeg_path.join("alvr_build")
    } else {
        ffmpeg_path
    }
}

fn main() {
    let platform_name = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cpp_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("cpp");

    // Detect features passed to the crate (build script cannot use cfg(feature))
    let saliency_enabled = env::var("CARGO_FEATURE_SALIENCY").is_ok();

    let platform_subpath = match platform_name.as_str() {
        "windows" => "cpp/platform/win32",
        "linux" => "cpp/platform/linux",
        "macos" => "cpp/platform/macos",
        _ => panic!(),
    };

    let common_iter = walkdir::WalkDir::new("cpp")
        .into_iter()
        .filter_entry(|entry| {
            entry.file_name() != "tools"
                && entry.file_name() != "platform"
                && (platform_name != "macos" || entry.file_name() != "amf")
        });

    let platform_iter = walkdir::WalkDir::new(platform_subpath).into_iter();

    let cpp_paths = common_iter
        .chain(platform_iter)
        .filter_map(|maybe_entry| maybe_entry.ok())
        .map(|entry| entry.into_path())
        .collect::<Vec<_>>();

    let source_files_paths = cpp_paths.iter().filter(|path| {
        path.extension()
            .filter(|ext| {
                let ext_str = ext.to_string_lossy();
                ext_str == "c" || ext_str == "cpp"
            })
            .is_some()
    });

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .files(source_files_paths)
        .flag_if_supported("-isystemcpp/openvr/headers") // silences many warnings from openvr headers
        .flag_if_supported("-std=c++17")
        .include("cpp/openvr/headers")
        .include("cpp");

    if platform_name == "windows" {
        build
            .debug(false) // This is because we cannot link to msvcrtd (see below)
            .flag("/std:c++17")
            .flag("/permissive-")
            .define("NOMINMAX", None)
            .define("_WINSOCKAPI_", None)
            .define("_MBCS", None)
            .define("_MT", None);
    } else if platform_name == "macos" {
        build.define("__APPLE__", None);
    }

    // #[cfg(debug_assertions)]
    // build.define("ALVR_DEBUG_LOG", None);

    let use_ffmpeg = cfg!(feature = "gpl") || cfg!(target_os = "linux");

    if use_ffmpeg {
        let ffmpeg_path = get_ffmpeg_path();

        assert!(ffmpeg_path.join("include").exists());
        build.include(ffmpeg_path.join("include"));
    }

    #[cfg(feature = "gpl")]
    build.define("ALVR_GPL", None);

    // Optional: LibTorch include (for future CPU/GPU inference in C++)
    if saliency_enabled {
        build.define("ALVR_SALIENCY", None);
        if let Ok(libtorch) = env::var("LIBTORCH") {
            let include = PathBuf::from(&libtorch).join("include");
            let include_torch = PathBuf::from(&libtorch).join("include/torch/csrc/api/include");
            build.include(include);
            build.include(include_torch);
        }
        // Optional: CUDA headers if available
        if let Ok(cuda_path) = env::var("CUDA_PATH") {
            let cuda_include = PathBuf::from(&cuda_path).join("include");
            if cuda_include.exists() {
                build.include(cuda_include);
            }
        }
    }

    build.compile("bindings");

    if use_ffmpeg {
        let ffmpeg_path = get_ffmpeg_path();
        let ffmpeg_lib_path = ffmpeg_path.join("lib");

        assert!(ffmpeg_lib_path.exists());

        println!(
            "cargo:rustc-link-search=native={}",
            ffmpeg_lib_path.to_string_lossy()
        );

        #[cfg(target_os = "linux")]
        {
            let ffmpeg_pkg_path = ffmpeg_lib_path.join("pkgconfig");
            assert!(ffmpeg_pkg_path.exists());

            let ffmpeg_pkg_path = ffmpeg_pkg_path.to_string_lossy().to_string();
            env::set_var(
                "PKG_CONFIG_PATH",
                env::var("PKG_CONFIG_PATH").map_or(ffmpeg_pkg_path.clone(), |old| {
                    format!("{ffmpeg_pkg_path}:{old}")
                }),
            );

            let pkg = pkg_config::Config::new().statik(true).to_owned();

            for lib in ["libavutil", "libavfilter", "libavcodec"] {
                pkg.probe(lib).unwrap();
            }
        }
        #[cfg(windows)]
        for lib in ["avutil", "avfilter", "avcodec", "swscale"] {
            println!("cargo:rustc-link-lib={lib}");
        }
    }

    // Link LibTorch libraries if enabled
    if saliency_enabled {
        if let Ok(libtorch) = env::var("LIBTORCH") {
            let libdir = PathBuf::from(&libtorch).join("lib");
            println!("cargo:rustc-link-search=native={}", libdir.to_string_lossy());
            for lib in [
                "c10",
                "torch_cpu",
                "torch_cuda",
                "torch",
            ] {
                println!("cargo:rustc-link-lib={}", lib);
            }
        }
        // CUDA runtime (optional)
        if let Ok(cuda_path) = env::var("CUDA_PATH") {
            let cudalib = PathBuf::from(&cuda_path).join("lib/x64");
            if cudalib.exists() {
                println!("cargo:rustc-link-search=native={}", cudalib.to_string_lossy());
                println!("cargo:rustc-link-lib=cudart");
            }
        }
        // Shader compiler on Windows
        if cfg!(target_os = "windows") {
            println!("cargo:rustc-link-lib=d3dcompiler");
        }
    }

    bindgen::builder()
        .clang_arg("-xc++")
        .header("cpp/alvr_server/bindings.h")
        .derive_default(true)
        .generate()
        .unwrap()
        .write_to_file(out_dir.join("bindings.rs"))
        .unwrap();

    if platform_name != "macos" {
        println!(
            "cargo:rustc-link-search=native={}",
            cpp_dir.join("openvr/lib").to_string_lossy()
        );
        println!("cargo:rustc-link-lib=openvr_api");
    }

    #[cfg(target_os = "linux")]
    {
        pkg_config::Config::new().probe("vulkan").unwrap();
        pkg_config::Config::new().probe("x264").unwrap();

        // fail build if there are undefined symbols in final library
        println!("cargo:rustc-cdylib-link-arg=-Wl,--no-undefined");
    }

    for path in cpp_paths {
        println!("cargo:rerun-if-changed={}", path.to_string_lossy());
    }
}
