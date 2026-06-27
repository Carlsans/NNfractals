// One-off: compute self_replication AND fractal_recursion for every saved .nn
// that lacks either metric, then re-save.
use nnfractals::config::Config;
use nnfractals::io::{load_genome, save_genome};
use nnfractals::fractal::{self_replication_score, fractal_recursion_score};
use nnfractals::render_gpu;
use std::path::Path;

fn main() {
    render_gpu::init_gpu();
    let cfg = Config::load(Path::new("config.toml")).unwrap();
    let mut n = 0; let mut done = 0;
    for entry in std::fs::read_dir("fractals").unwrap().flatten() {
        let p = entry.path();
        if p.extension().and_then(|x| x.to_str()) != Some("nn") { continue; }
        n += 1;
        let Ok(mut g) = load_genome(&p) else { continue };
        if g.self_replication == 0.0 {
            g.self_replication = self_replication_score(&g, &cfg);
        }
        // Always recompute fractal_recursion: its definition evolves (boundary
        // descent, intricacy gate), so stored values must track the current metric.
        g.fractal_recursion = fractal_recursion_score(&g, &cfg);
        if save_genome(&g, &p).is_ok() { done += 1; }
        if done > 0 && done % 25 == 0 { eprintln!("  {done} scored…"); }
    }
    eprintln!("Backfilled metrics for {done}/{n} genomes.");
}
