// Re-runnable: score every saved .nn whose CLIP+LAION are both 0 — genomes saved
// while the aesthetic sidecar was still loading (or a run where it failed to load),
// which fell back to the beauty gate and stored 0/0. Feeds each genome's already-saved
// sibling .png to the aesthetic scorer and re-saves with real clip/laion (and beauty
// = laion/10, matching the optimizer's save-time convention).
//
//   cargo run --release --bin backfill_scores -- [dir]
//   (dir defaults to config.toml's save_dir)

use nnfractals::aesthetic::AestheticScorer;
use nnfractals::colormap::apply_colormap;
use nnfractals::config::Config;
use nnfractals::fractal::{best_entropy_view, render_cpu};
use nnfractals::io::{load_genome, save_genome, save_png};
use nnfractals::render_gpu;
use std::path::{Path, PathBuf};

fn main() {
    render_gpu::init_gpu();
    let cfg = Config::load(Path::new("config.toml")).expect("load config.toml");
    let dir: PathBuf = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| cfg.output.save_dir.clone());

    let mut scorer = match AestheticScorer::new() {
        Some(s) => s,
        None => {
            eprintln!("aesthetic_scorer.py or python unavailable — cannot score. Aborting.");
            std::process::exit(1);
        }
    };
    eprintln!("Scoring 0-value fractals in {} …", dir.display());
    eprintln!("(CLIP+LAION models load on first score — this can take 30–60s.)");

    let w = cfg.rendering.default_width;
    let h = cfg.rendering.default_height;
    let tmp = PathBuf::from(format!("/tmp/nnfractals_backfill_{}.png", std::process::id()));

    let mut scanned = 0usize;
    let mut targeted = 0usize;
    let mut scored = 0usize;
    let mut failed = 0usize;

    let entries: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()))
        .flatten()
        .collect();

    for e in entries {
        let p = e.path();
        if p.extension().and_then(|x| x.to_str()) != Some("nn") {
            continue;
        }
        scanned += 1;
        let Ok(mut g) = load_genome(&p) else { continue };
        // Only touch never-scored genomes (both exactly 0.0).
        if g.clip_score != 0.0 || g.laion_score != 0.0 {
            continue;
        }
        targeted += 1;

        // Prefer the already-saved sibling PNG (faithful to the original scored image);
        // fall back to re-rendering the best-entropy view like the optimizer did at save.
        let png = p.with_extension("png");
        let score_path = if png.exists() {
            png.clone()
        } else {
            let rg = best_entropy_view(&g, &cfg);
            let et = render_cpu(&rg, &cfg, w, h);
            let rgb = apply_colormap(&et, cfg.rendering.max_iter, &cfg.rendering.colormap);
            if save_png(&rgb, w, h, &tmp).is_err() {
                failed += 1;
                continue;
            }
            tmp.clone()
        };

        match scorer.score_blocking(score_path) {
            Some(s) => {
                g.clip_score = s.clip;
                g.laion_score = s.laion;
                // Match the optimizer's save-time convention: stored beauty = LAION/10.
                g.beauty = s.laion / 10.0;
                if save_genome(&g, &p).is_ok() {
                    scored += 1;
                    if scored % 25 == 0 {
                        eprintln!("  {scored} scored…");
                    }
                } else {
                    failed += 1;
                }
            }
            None => {
                eprintln!("  no score for {} (scorer not ready / timeout)", p.display());
                failed += 1;
            }
        }
    }

    let _ = std::fs::remove_file(&tmp);
    eprintln!(
        "Done: {scored} scored, {failed} failed, {targeted} needed scoring of {scanned} .nn in {}",
        dir.display()
    );
}
