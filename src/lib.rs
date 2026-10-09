pub mod config;
pub mod dd;
pub mod formula;
pub mod time_program;
pub mod genome;
pub mod known_formulas;
pub mod fractal;
pub mod recursion_model;
pub mod colormap;
pub mod fitness;
pub mod fitness_profile;
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
pub mod quaternion;
pub mod quat_fractal;
pub mod quat_motion;
pub mod quat_voxel;
pub mod quat_gravity;
pub mod quat_raymarch;
pub mod quat_dag;
pub mod quat_dag_fitness;
pub mod quat_genome_ops;
pub mod quat_organization;
pub mod quat_boundedness;
pub mod quat_predator;
pub mod quat_pref;
pub mod quat_taste;
pub mod quat_live_view;
pub mod quat_map_elites;
pub mod anim_timeline;
pub mod anim_persist;
pub mod anim_profile;
pub mod orient;
pub mod anim_eval;
pub mod debug_overlay;
#[cfg(feature = "wgpu-backend")]
pub mod explore;
#[cfg(feature = "wgpu-backend")]
pub mod vae_explore;
#[cfg(feature = "wgpu-backend")]
pub mod video_zoom_explore;
#[cfg(feature = "wgpu-backend")]
pub mod time_explore;
#[cfg(feature = "wgpu-backend")]
pub mod time_ga;
#[cfg(feature = "wgpu-backend")]
pub mod auto_reel;
#[cfg(any(feature = "viewer", feature = "browser", feature = "launcher", feature = "queue"))]
pub mod gui_font;
#[cfg(feature = "wgpu-backend")]
pub mod render_gpu;
#[cfg(feature = "wgpu-backend")]
pub mod render_gpu_raymarch;
#[cfg(feature = "wgpu-backend")]
pub mod render_gpu_raymarch_dag;
#[cfg(feature = "wgpu-backend")]
pub mod render_gpu_raymarch_dag_codegen;
#[cfg(feature = "wgpu-backend")]
pub mod queue_runner;

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
/// Locate a sibling project binary: `target/release/<name>` first — even when
/// the caller is itself a debug build — then next to this executable, then
/// `target/debug/<name>`, then `~/.local/bin`, then the bare name on PATH.
///
/// Release-first is load-bearing and was a real bug: a desktop entry hardcoding
/// `target/debug/nnfractals-launcher` cascaded into debug siblings
/// indefinitely, because a check-my-own-directory-first order never got as far
/// as preferring release.
///
/// `viewer.rs`, `browser.rs` and `launcher.rs` each carry their own copy of
/// this, independently written and independently bitten by that same bug. This
/// is the one to use from here on; migrating those three is a separate,
/// mechanical change.
pub fn locate_bin(name: &str) -> std::path::PathBuf {
    use std::path::{Path, PathBuf};
    let dir = std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(Path::to_path_buf))
        .unwrap_or_else(|| PathBuf::from("."));
    if let Some(target) = dir.parent() {
        let c = target.join("release").join(name);
        if c.exists() {
            return c;
        }
    }
    let c = dir.join(name);
    if c.exists() {
        return c;
    }
    if let Some(target) = dir.parent() {
        let c = target.join("debug").join(name);
        if c.exists() {
            return c;
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        let c = PathBuf::from(home).join(".local/bin").join(name);
        if c.exists() {
            return c;
        }
    }
    PathBuf::from(name)
}

/// Whether `path` is (or is inside) a quaternion-genome folder —
/// `fractals_dag_quat/` and any `--out-dir` variant of it (a
/// `quat-dag-evolve` run naturally lands in something like
/// `fractals_dag_quat_night_subtree/` or
/// `fractals_dag_quat_metric_boxdim/`, never pinned to the exact literal
/// name). PREFIX match, not exact equality, on purpose: the original
/// exact-match version (once duplicated between `browser.rs` and never
/// added to `launcher.rs` at all) silently routed every genome from a
/// differently-named quat out-dir to the 2D viewer instead of the
/// quaternion one — found by Carl opening one and getting a 2D escape-
/// time render instead of a ray-march. Canonical version — `browser.rs`
/// and `launcher.rs` both call this now instead of carrying their own
/// copy, the exact drift that caused the bug in the first place.
pub fn is_quat_genome_path(path: &std::path::Path) -> bool {
    path.components().any(|c| {
        c.as_os_str()
            .to_str()
            .is_some_and(|s| s.starts_with("fractals_dag_quat") || s == "train_corpus_quat")
    })
}

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
mod is_quat_genome_path_tests {
    use super::is_quat_genome_path;
    use std::path::Path;

    #[test]
    fn recognizes_the_literal_folder() {
        assert!(is_quat_genome_path(Path::new("fractals_dag_quat/abc123.nn")));
        assert!(is_quat_genome_path(Path::new("train_corpus_quat/abc123.nn")));
    }

    #[test]
    fn recognizes_out_dir_variants() {
        // The actual bug Carl hit: quat-dag-evolve --out-dir isn't pinned to
        // the literal "fractals_dag_quat" name, so every one of these is a
        // real folder an overnight run/metric sanity-check produced.
        assert!(is_quat_genome_path(Path::new("fractals_dag_quat_night_subtree/x.nn")));
        assert!(is_quat_genome_path(Path::new("fractals_dag_quat_compare_legacy/x.nn")));
        assert!(is_quat_genome_path(Path::new("fractals_dag_quat_metric_boxdim/x.nn")));
        assert!(is_quat_genome_path(Path::new("/home/carl/rust_projects/NNfractals/fractals_dag_quat_metric_boxdim_autocorr_predprey/x.nn")));
    }

    #[test]
    fn does_not_match_the_2d_folder() {
        assert!(!is_quat_genome_path(Path::new("fractals_dag/abc123.nn")));
        assert!(!is_quat_genome_path(Path::new("fractals_1/abc123.nn")));
    }
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
