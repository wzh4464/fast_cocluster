use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();

    // Auto-detect OpenBLAS location based on platform
    let openblas_paths = find_openblas_paths(&target_os);

    if openblas_paths.is_empty() {
        eprintln!("\n{}", "=".repeat(70));
        eprintln!("WARNING: OpenBLAS not found in common locations");
        eprintln!("{}", "=".repeat(70));
        eprintln!("\nPlease install OpenBLAS:");
        eprintln!("  macOS:        brew install openblas");
        eprintln!("  Ubuntu/Debian: sudo apt-get install libopenblas-dev");
        eprintln!("  CentOS/RHEL:   sudo yum install openblas-devel");
        eprintln!("  Conda:         conda install -c conda-forge openblas");
        eprintln!("\nOr set OPENBLAS_LIB environment variable to your OpenBLAS lib path.");
        eprintln!("{}\n", "=".repeat(70));

        // Try to link anyway - might work if in system default paths
        println!("cargo:rustc-link-lib=openblas");
        return;
    }

    // Use the first found path
    let openblas_lib_dir = &openblas_paths[0];
    println!("cargo:info=Found OpenBLAS at: {}", openblas_lib_dir.display());

    // Add library search path
    println!("cargo:rustc-link-search=native={}", openblas_lib_dir.display());
    println!("cargo:rustc-link-lib=openblas");

    // Add rpath for runtime linking (not needed for system paths)
    if !is_system_path(&openblas_lib_dir) {
        add_rpath(&target_os, &openblas_lib_dir);
    }
}

fn find_openblas_paths(target_os: &str) -> Vec<PathBuf> {
    let mut paths = Vec::new();

    // 1. Check OPENBLAS_LIB environment variable (highest priority)
    if let Ok(custom_path) = env::var("OPENBLAS_LIB") {
        let path = PathBuf::from(custom_path);
        if path.exists() {
            paths.push(path);
            return paths; // Use custom path exclusively
        }
    }

    // 2. Check Conda environment
    if let Ok(conda_prefix) = env::var("CONDA_PREFIX") {
        let conda_lib = PathBuf::from(conda_prefix).join("lib");
        if conda_lib.join("libopenblas.so").exists()
            || conda_lib.join("libopenblas.dylib").exists()
            || conda_lib.join("libopenblas.a").exists()
        {
            paths.push(conda_lib);
        }
    }

    // 3. Platform-specific detection
    match target_os {
        "macos" => {
            // Homebrew (Intel and Apple Silicon)
            for prefix in &[
                "/opt/homebrew/opt/openblas/lib",    // Apple Silicon
                "/usr/local/opt/openblas/lib",       // Intel
            ] {
                let path = PathBuf::from(prefix);
                if path.exists() && has_openblas_lib(&path) {
                    paths.push(path);
                }
            }

            // Try detecting via brew command
            if let Ok(output) = Command::new("brew")
                .args(&["--prefix", "openblas"])
                .output()
            {
                if output.status.success() {
                    let brew_prefix = String::from_utf8_lossy(&output.stdout);
                    let brew_lib = PathBuf::from(brew_prefix.trim()).join("lib");
                    if brew_lib.exists() && has_openblas_lib(&brew_lib) {
                        paths.push(brew_lib);
                    }
                }
            }

            // System paths
            paths.push(PathBuf::from("/usr/lib"));
            paths.push(PathBuf::from("/usr/local/lib"));
        }

        "linux" => {
            // Common Linux system paths
            for prefix in &[
                "/usr/lib/x86_64-linux-gnu",  // Ubuntu/Debian
                "/usr/lib64",                 // CentOS/RHEL/Fedora
                "/usr/lib",                   // Generic
                "/usr/local/lib",
            ] {
                let path = PathBuf::from(prefix);
                if path.exists() && has_openblas_lib(&path) {
                    paths.push(path);
                }
            }

            // Check pkg-config
            if let Ok(output) = Command::new("pkg-config")
                .args(&["--libs-only-L", "openblas"])
                .output()
            {
                if output.status.success() {
                    let lib_flags = String::from_utf8_lossy(&output.stdout);
                    for flag in lib_flags.split_whitespace() {
                        if flag.starts_with("-L") {
                            let path = PathBuf::from(&flag[2..]);
                            if path.exists() {
                                paths.push(path);
                            }
                        }
                    }
                }
            }
        }

        "windows" => {
            // Windows paths (if using MSYS2 or vcpkg)
            if let Ok(msys2) = env::var("MSYS2_ROOT") {
                paths.push(PathBuf::from(msys2).join("usr/lib"));
            }
        }

        _ => {}
    }

    // Deduplicate and filter existing paths
    paths.sort();
    paths.dedup();
    paths.retain(|p| p.exists() && has_openblas_lib(p));

    paths
}

fn has_openblas_lib(dir: &PathBuf) -> bool {
    let lib_files = ["libopenblas.so", "libopenblas.dylib", "libopenblas.a", "openblas.lib"];

    for lib_file in &lib_files {
        if dir.join(lib_file).exists() {
            return true;
        }
    }

    // Check for versioned libraries (libopenblas.so.0, etc.)
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                if name.starts_with("libopenblas.so")
                    || name.starts_with("libopenblas.dylib")
                {
                    return true;
                }
            }
        }
    }

    false
}

fn is_system_path(path: &PathBuf) -> bool {
    let system_paths = [
        "/usr/lib",
        "/usr/lib64",
        "/usr/local/lib",
        "/lib",
        "/lib64",
    ];

    let path_str = path.to_string_lossy();
    system_paths.iter().any(|sys_path| path_str.starts_with(sys_path))
}

fn add_rpath(target_os: &str, lib_dir: &PathBuf) {
    match target_os {
        "macos" => {
            // macOS uses -rpath directly
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
        }
        "linux" => {
            // Linux uses -Wl,-rpath,
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
        }
        _ => {}
    }
}
