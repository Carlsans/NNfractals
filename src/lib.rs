pub mod config;
pub mod dd;
pub mod formula;
pub mod genome;
pub mod known_formulas;
pub mod fractal;
pub mod recursion_model;
pub mod colormap;
pub mod fitness;
pub mod io;
pub mod display;
pub mod optimizer;
pub mod aesthetic;
pub mod novelty;
pub mod nav_predict;
pub mod vae_score;
pub mod saliency;
pub mod video_export;
pub mod formula_usage;
#[cfg(feature = "wgpu-backend")]
pub mod explore;
#[cfg(feature = "wgpu-backend")]
pub mod vae_explore;
#[cfg(feature = "wgpu-backend")]
pub mod video_zoom_explore;
#[cfg(any(feature = "viewer", feature = "browser", feature = "launcher", feature = "queue"))]
pub mod gui_font;
#[cfg(feature = "wgpu-backend")]
pub mod render_gpu;

/// Derives the project root from the running binary's own location —
/// `target/release/<bin>` (or `target/debug/<bin>`) sits exactly 2
/// directories under the root, so this is `current_exe()` minus 2
/// `.parent()` calls. Load-bearing for any GUI binary (`viewer`, `queue`)
/// that reads project-relative resources (`config.toml`, `video_queue/`):
/// unlike a CLI tool, which is invoked from a terminal where the user has
/// already `cd`'d to the project root by convention, a GUI app launched
/// via a desktop file / file-manager double-click / `xdg-open` can have
/// almost ANY working directory — confirmed a real bug, not theoretical
/// (Carl, 2026-08-11): opening a `.nn` file through the file manager and
/// using the video-export feature failed with "No such file or
/// directory" because `Config::load(Path::new("config.toml"))` and
/// `video_export::queue_dir()` were both bare CWD-relative paths. One
/// shared implementation here rather than each binary growing its own
/// copy — this project has already been bitten once by exactly that
/// (three duplicated `locate_bin` functions, see
/// `[[viewer-angle-coloring-and-binary-resolution]]`).
pub fn project_root() -> std::path::PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|exe| {
            exe.parent() // .../target/release (or debug)
                .and_then(std::path::Path::parent) // .../target
                .and_then(std::path::Path::parent) // project root
                .map(std::path::Path::to_path_buf)
        })
        .unwrap_or_else(|| std::path::PathBuf::from("."))
}

/// Resolve the python interpreter for sidecar scripts (aesthetic_scorer.py,
/// scripts/dedup.py, scripts/train_pref.py): prefer the project-local
/// virtualenv created by scripts/install-deps.sh (`<root>/.venv/bin/python3`),
/// falling back to whichever of `python3`/`python` is found on PATH.
pub fn python_bin(root: &std::path::Path) -> std::path::PathBuf {
    let venv = root.join(".venv/bin/python3");
    if venv.exists() {
        return venv;
    }
    for cmd in ["python3", "python"] {
        let works = std::process::Command::new(cmd)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if works {
            return std::path::PathBuf::from(cmd);
        }
    }
    std::path::PathBuf::from("python3")
}

#[cfg(test)]
mod python_bin_tests {
    use super::python_bin;

    #[test]
    fn prefers_venv_when_present() {
        let dir = std::env::temp_dir().join(format!("nnfractals_test_venv_{}", std::process::id()));
        let bin = dir.join(".venv/bin");
        std::fs::create_dir_all(&bin).unwrap();
        std::fs::write(bin.join("python3"), b"").unwrap();

        assert_eq!(python_bin(&dir), bin.join("python3"));

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn falls_back_to_path_without_venv() {
        let dir = std::env::temp_dir().join(format!("nnfractals_test_novenv_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let resolved = python_bin(&dir);
        assert!(resolved == std::path::PathBuf::from("python3")
            || resolved == std::path::PathBuf::from("python"));

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
