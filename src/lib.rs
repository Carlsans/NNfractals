pub mod config;
pub mod dd;
pub mod formula;
pub mod genome;
pub mod fractal;
pub mod recursion_model;
pub mod colormap;
pub mod fitness;
pub mod io;
pub mod display;
pub mod optimizer;
pub mod aesthetic;
pub mod formula_usage;
#[cfg(any(feature = "viewer", feature = "browser", feature = "launcher"))]
pub mod gui_font;
#[cfg(feature = "wgpu-backend")]
pub mod render_gpu;

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
