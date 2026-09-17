//! Ad-hoc diagnostic: how much does escape time / DE vary when the SAME
//! (r, rho) point is sampled at many different directions n-hat on the
//! sphere? A fractal whose iteration only ever scales the vector part by
//! a scalar (never mixes components via a genuinely 3D operation) stays
//! exactly constant across all n-hat -- perfectly spherically symmetric,
//! hence "boring" no matter how the camera orbits (this is the same
//! degeneracy already found for the hand-built Mandelbrot/Tricorn/etc.
//! under TimeAxis::R earlier this project). Run with:
//!   cargo run --release --example symmetry_check -- path/to/genome.nn
//! or with no args to check the hand-built Bulb formula for comparison.

use nnfractals::quat_dag::{quat_dag_escape_de, QuatDagFormula};
use nnfractals::quat_fractal::{quat_escape_de, QuatFormula};
use nnfractals::quaternion::Quat;

fn directions() -> Vec<(f64, f64, f64)> {
    // A handful of well-spread unit vectors (not exhaustive -- just
    // enough to catch gross degeneracy cheaply).
    let raw: [(f64, f64, f64); 8] = [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (1.0, -1.0, 0.3),
        (-0.4, 0.8, -0.9),
        (0.2, -0.6, 1.0),
        (-1.0, -1.0, 0.2),
    ];
    raw.iter()
        .map(|&(x, y, z)| {
            let n = (x * x + y * y + z * z).sqrt();
            (x / n, y / n, z / n)
        })
        .collect()
}

fn report(label: &str, r: f64, rho: f64, sample: impl Fn(Quat) -> (f32, f64)) {
    let mut ets = Vec::new();
    let mut des = Vec::new();
    for (a, b, c) in directions() {
        let q = Quat::new(r, rho * a, rho * b, rho * c);
        let (et, de) = sample(q);
        ets.push(et as f64);
        des.push(de);
    }
    let et_mean = ets.iter().sum::<f64>() / ets.len() as f64;
    let et_spread = ets.iter().map(|v| (v - et_mean).abs()).fold(0.0, f64::max);
    let de_mean = des.iter().sum::<f64>() / des.len() as f64;
    let de_spread = if de_mean.abs() > 1e-12 { des.iter().map(|v| (v - de_mean).abs()).fold(0.0, f64::max) / de_mean.abs() } else { 0.0 };
    println!(
        "  {label} r={r:.2} rho={rho:.2}: escape_time spread(max|dev|)={et_spread:.4} (mean={et_mean:.2})  DE relative_spread={de_spread:.4}"
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let points = [(0.2, 0.9), (0.5, 0.5), (-0.3, 1.1), (0.0, 1.3)];

    if let Some(path) = args.get(1) {
        let genome = nnfractals::io::load_genome(std::path::Path::new(path)).expect("load genome");
        if genome.program.is_empty() {
            panic!("genome has no DAG program");
        }
        let formula = QuatDagFormula {
            prog: &genome.program,
            warp: &genome.warp,
            julia: genome.julia_mode,
            jc: (genome.julia_cre, genome.julia_cim),
            phoenix: (genome.phoenix_re, genome.phoenix_im),
        };
        let bailout_sq = (genome.bailout_radius as f64) * (genome.bailout_radius as f64);
        println!("genome {path}:");
        for (r, rho) in points {
            report("genome", r, rho, |q| quat_dag_escape_de(&formula, q, 60, bailout_sq));
        }
    } else {
        println!("bulb (hand-built, known genuinely-3D formula):");
        for (r, rho) in points {
            report("bulb", r, rho, |q| quat_escape_de(QuatFormula::Bulb, q, 60, 16.0));
        }
        println!();
        println!("classic mandelbrot (hand-built, known SO(3)-symmetric under R):");
        for (r, rho) in points {
            report("mandelbrot", r, rho, |q| quat_escape_de(QuatFormula::Mandelbrot, q, 60, 16.0));
        }
    }
}
