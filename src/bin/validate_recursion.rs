// Sanity-check the fractal_recursion metric on known formulas.
//
// `c` is basis 7 (not implicit), so z²+c = term(basis 0, 1+0i) + term(basis 7, 1+0i).
// Mandelbrot is rich in baby-Mandelbrots (embedded whole-set copies) → should score
// high on fractal_recursion; a plain linear/degenerate map → near zero.
use nnfractals::config::Config;
use nnfractals::genome::{Genome, FormulaTerm};
use nnfractals::fractal::{self_replication_score, fractal_recursion_score, field_intricacy, render_bounds};
use nnfractals::render_gpu;
use std::path::Path;

fn make(terms: &[(u8, f32, f32)], cx: f32, cy: f32, zoom: f32) -> Genome {
    let mut g = Genome { terms: Vec::new(), ..Genome::default() };
    g.terms = terms.iter().map(|&(basis, re, im)| FormulaTerm { basis, re, im }).collect();
    g.view_cx = cx; g.view_cy = cy; g.view_zoom = zoom;
    g
}

fn main() {
    render_gpu::init_gpu();
    let cfg = Config::load(Path::new("config.toml")).unwrap();

    let cases: &[(&str, Genome)] = &[
        // Classic Mandelbrot z²+c — full view. Baby-Mandelbrots live on the boundary.
        ("mandelbrot z²+c (full)", make(&[(0, 1.0, 0.0), (7, 1.0, 0.0)], 0.0, 0.0, 1.0)),
        // Mandelbrot zoomed toward the seahorse valley / period-2 bulb neck, where a
        // prominent baby-Mandelbrot sits near (-1.75, 0).
        ("mandelbrot z²+c (zoom -1.75)", make(&[(0, 1.0, 0.0), (7, 1.0, 0.0)], -1.4, 0.0, 3.0)),
        // Cubic Mandelbrot z³+c — also self-referential.
        ("cubic z³+c", make(&[(1, 1.0, 0.0), (7, 1.0, 0.0)], 0.0, 0.0, 1.0)),
        // Degenerate non-fractal: z·c + c (linear in z) — no recursion.
        ("linear z·c+c", make(&[(10, 1.0, 0.0), (7, 1.0, 0.0)], 0.0, 0.0, 1.0)),
        // Identity-ish escape: z + c — trivial, escapes everywhere.
        ("trivial z+c", make(&[(4, 1.0, 0.0), (7, 1.0, 0.0)], 0.0, 0.0, 1.0)),
    ];

    println!("{:<32} {:>11} {:>11} {:>11}", "formula", "recursion", "intricacy", "self_repl");
    println!("{}", "-".repeat(68));
    for (name, g) in cases {
        let rec = fractal_recursion_score(g, &cfg);
        let sr  = self_replication_score(g, &cfg);
        // Intricacy of the base view (the gate input), for calibration.
        let (x0, x1, y0, y1) = g.view_bounds();
        let base = render_bounds(&g.formula_weights(), &cfg, 128, 128,
                                 cfg.rendering.max_iter, x0, x1, y0, y1);
        let intric = field_intricacy(&base, 128, 128);
        println!("{:<32} {:>11.3} {:>11.3} {:>11.3}", name, rec, intric, sr);
    }
}
