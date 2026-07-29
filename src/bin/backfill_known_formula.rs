// One-off/re-runnable: compute known_formula_match/known_formula_score for
// every saved DAG .nn. Discovery/curiosity metadata only — see
// fractal::known_formula_match's doc comment. Always recomputed (not gated
// on "already has a value") because the reference library/threshold will
// likely be tuned after initial deployment, same rationale as
// fractal_recursion in backfill_selfrep.rs.
//
//   cargo run --release --bin backfill_known_formula -- [dir]
//   (dir defaults to config.toml's save_dir)
//
// Pure CPU — no GPU init needed (known_formula_match is built entirely on
// dag_escape_pixel, never touches render_gpu), so this runs identically with
// or without a GPU adapter present.

use nnfractals::config::Config;
use nnfractals::fractal::known_formula_match;
use nnfractals::io::{load_genome, save_genome};
use std::path::{Path, PathBuf};

fn main() {
    let cfg = Config::load(Path::new("config.toml")).expect("load config.toml");
    let dir: PathBuf = std::env::args().nth(1).map(PathBuf::from)
        .unwrap_or_else(|| cfg.output.save_dir.clone());

    let mut n = 0usize;
    let mut skipped_legacy = 0usize;
    let mut done = 0usize;
    for entry in std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()))
        .flatten()
    {
        let p = entry.path();
        if p.extension().and_then(|x| x.to_str()) != Some("nn") { continue; }
        n += 1;
        let Ok(mut g) = load_genome(&p) else { continue };
        if !g.uses_program() { skipped_legacy += 1; continue; } // DAG-only feature

        match known_formula_match(&g) {
            Some((name, score)) => { g.known_formula_match = name.to_string(); g.known_formula_score = score; }
            None => { g.known_formula_match.clear(); g.known_formula_score = 0.0; }
        }
        if save_genome(&g, &p).is_ok() { done += 1; }
        if done > 0 && done % 500 == 0 { eprintln!("  {done} scored…"); }
    }
    eprintln!("Backfilled known-formula match for {done}/{n} genomes ({skipped_legacy} legacy skipped) in {}.", dir.display());
}
