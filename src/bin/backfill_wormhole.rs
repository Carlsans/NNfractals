// One-off/re-runnable: compute wormhole_score/dx/dy/zoom for every saved
// .nn (DAG or legacy — wormhole_search routes both through
// video_export::render_escape_times, unlike fractal_recursion_score which
// is silently 0.0 for every DAG genome via formula_weights()). Discovery/
// navigation metadata only — see fractal::wormhole_search's doc comment.
// Always recomputed (not gated on "already has a value"), same rationale
// as known_formula in backfill_known_formula.rs: search parameters will
// likely be tuned after seeing real results.
//
//   cargo run --release --features wgpu-backend --bin backfill_wormhole -- [dir] [limit]
//   (dir defaults to config.toml's save_dir; limit caps how many .nn files
//   are processed, for a quick sample run before committing to the whole
//   archive — omit for no cap)

use nnfractals::config::Config;
use nnfractals::fractal::wormhole_search;
use nnfractals::io::{load_genome, save_genome};
use nnfractals::video_export::View;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn main() {
    let cfg = Config::load(Path::new("config.toml")).expect("load config.toml");
    let mut args = std::env::args().skip(1);
    let dir: PathBuf = args.next().map(PathBuf::from).unwrap_or_else(|| cfg.output.save_dir.clone());
    let limit: Option<usize> = args.next().and_then(|s| s.parse().ok());

    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()))
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("nn"))
        .collect();
    paths.sort();
    if let Some(n) = limit { paths.truncate(n); }

    let n = paths.len();
    let mut done = 0usize;
    let mut matched = 0usize;
    let t0 = Instant::now();

    for p in &paths {
        let Ok(mut g) = load_genome(p) else { continue };
        let ref_view = View::new_square(g.view_cx as f64, g.view_cy as f64, (g.view_zoom as f64).max(0.3));

        match wormhole_search(&g, &cfg, &ref_view) {
            Some(m) => {
                g.wormhole_score = m.score;
                g.wormhole_dx    = m.dx as f32;
                g.wormhole_dy    = m.dy as f32;
                g.wormhole_zoom  = m.zoom as f32;
                matched += 1;
            }
            None => {
                g.wormhole_score = 0.0;
                g.wormhole_dx    = 0.0;
                g.wormhole_dy    = 0.0;
                g.wormhole_zoom  = 0.0;
            }
        }
        if save_genome(&g, p).is_ok() { done += 1; }
        if done > 0 && done % 25 == 0 {
            let elapsed = t0.elapsed().as_secs_f32();
            let rate = done as f32 / elapsed;
            let eta = (n - done) as f32 / rate.max(0.01);
            eprintln!("  {done}/{n} scored ({matched} matched, {:.1}s/genome, ETA {:.0}s)…",
                elapsed / done as f32, eta);
        }
    }
    eprintln!("Backfilled wormhole score for {done}/{n} genomes ({matched} with a confident match) in {} — {:.1}s total.",
        dir.display(), t0.elapsed().as_secs_f32());
}
