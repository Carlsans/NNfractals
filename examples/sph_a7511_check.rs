use nnfractals::io;
use nnfractals::quat_dag::QuatDagFormula;
use nnfractals::quat_organization::sphericity_full_diagnostic;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    for path in &args[1..] {
        let genome = io::load_genome(std::path::Path::new(path)).unwrap();
        let f = QuatDagFormula {
            prog: &genome.program,
            warp: &genome.warp,
            julia: genome.julia_mode,
            jc: (genome.julia_cre, genome.julia_cim),
            phoenix: (genome.phoenix_re, genome.phoenix_im),
        };
        let bailout_sq = (genome.bailout_radius as f64) * (genome.bailout_radius as f64);
        let (hits, n_dir, mean_r, roundness, total) = sphericity_full_diagnostic(&f, 1.6, 60, bailout_sq);
        println!(
            "{:60} hit_rate={:.3} mean_hit_r={:.4} fill_ratio={:.4} roundness={:.3} => sphericity={:.4}",
            path.rsplit('/').next().unwrap(),
            hits as f64 / n_dir as f64,
            mean_r,
            (mean_r / (1.6 * 0.3)).min(1.0),
            roundness,
            total,
        );
    }
}
