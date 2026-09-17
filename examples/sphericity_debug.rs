// One-off calibration tool: dumps sphericity's raw (hit_count, mean_radius)
// for a handful of real archived genomes, to check whether domain_radius=1.6
// is a sane reference point for the "sits around the domain" fill_ratio
// factor, or whether real genomes' surfaces sit much closer in.
use nnfractals::io;
use nnfractals::quat_dag::QuatDagFormula;
use nnfractals::quat_organization::sphericity_radial_profile;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = args.get(1).map(String::as_str).unwrap_or("fractals_dag_quat");
    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(15);

    let mut files: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("nn"))
        .collect();
    files.sort();
    files.truncate(n);

    for path in &files {
        let genome = match io::load_genome(path) {
            Ok(g) if !g.program.is_empty() => g,
            _ => continue,
        };
        let f = QuatDagFormula {
            prog: &genome.program,
            warp: &genome.warp,
            julia: genome.julia_mode,
            jc: (genome.julia_cre, genome.julia_cim),
            phoenix: (genome.phoenix_re, genome.phoenix_im),
        };
        let bailout_sq = (genome.bailout_radius as f64) * (genome.bailout_radius as f64);
        let (hits, n_dir, mean_r) = sphericity_radial_profile(&f, 1.6, 60, bailout_sq);
        println!(
            "{:40} bailout_r={:.2} hit_rate={:.2} mean_hit_r={:.3} fill@1.6={:.3}",
            path.file_name().unwrap().to_string_lossy(),
            genome.bailout_radius,
            hits as f64 / n_dir as f64,
            mean_r,
            mean_r / 1.6,
        );
    }
}
