// Test the fractal_recursion metric on a family of KNOWN self-replicating
// fractals (which contain miniature copies of the whole set) versus non-recursive
// controls. Basis indices (see formula.rs): 0=z² 1=z³ 2=z⁴ 3=z⁵ 4=z 7=c 10=z·c
// 46=burning-ship fold 48=conj(z)² (tricorn). `c` is basis 7 (not implicit), so
// e.g. z²+c = term(0,1+0i) + term(7,1+0i).
use nnfractals::config::Config;
use nnfractals::genome::{Genome, FormulaTerm};
use nnfractals::fractal::{self_replication_score, fractal_recursion_score, field_intricacy, render_bounds};
use nnfractals::render_gpu;
use std::path::Path;
use std::time::Instant;

fn make(terms: &[(u8, f32, f32)], cx: f32, cy: f32, zoom: f32) -> Genome {
    let mut g = Genome { terms: Vec::new(), ..Genome::default() };
    g.terms = terms.iter().map(|&(basis, re, im)| FormulaTerm { basis, re, im }).collect();
    g.view_cx = cx; g.view_cy = cy; g.view_zoom = zoom;
    g
}

fn main() {
    render_gpu::init_gpu();
    let cfg = Config::load(Path::new("config.toml")).unwrap();

    // (label, genome, is_recursive_expected)
    let cases: &[(&str, Genome, bool)] = &[
        // ── Known self-replicating: multibrots z^d + c (mini-copies on the antennae)
        ("z²+c  mandelbrot (full)",   make(&[(0,1.,0.),(7,1.,0.)], 0., 0., 1.), true),
        ("z²+c  zoom@-1.75 (baby)",   make(&[(0,1.,0.),(7,1.,0.)], -1.4, 0., 3.), true),
        ("z³+c  cubic multibrot",     make(&[(1,1.,0.),(7,1.,0.)], 0., 0., 1.), true),
        ("z⁴+c  quartic multibrot",   make(&[(2,1.,0.),(7,1.,0.)], 0., 0., 1.), true),
        ("z⁵+c  quintic multibrot",   make(&[(3,1.,0.),(7,1.,0.)], 0., 0., 1.), true),
        // ── Tricorn / Mandelbar  z̄² + c  (contains mini-mandelbrots)
        ("z̄²+c  tricorn (full)",      make(&[(48,1.,0.),(7,1.,0.)], 0., 0., 1.), true),
        ("z̄²+c  tricorn zoom",        make(&[(48,1.,0.),(7,1.,0.)], -1.2, 0., 2.5), true),
        // ── Phoenix-ish rational with c  (1/(z²+c)) — has its own recursive structure
        ("1/(z²+c)  rational",        make(&[(54,1.,0.),(7,1.,0.)], 0., 0., 1.), true),
        // ── Non-recursive controls (should stay low / be gated to 0)
        ("z+c  trivial ramp",         make(&[(4,1.,0.),(7,1.,0.)], 0., 0., 1.), false),
        ("z·c+c  linear in z",        make(&[(10,1.,0.),(7,1.,0.)], 0., 0., 1.), false),
        ("z²  no c (plain escape)",   make(&[(0,1.,0.)], 0., 0., 1.), false),
        ("c²+z  degenerate",          make(&[(8,1.,0.),(4,1.,0.)], 0., 0., 1.), false),
    ];

    println!("{:<28} {:>10} {:>10} {:>10}   {}", "fractal", "recursion", "intricacy", "self_rep", "class");
    println!("{}", "-".repeat(78));
    let t0 = Instant::now();
    let mut min_recur = f32::MAX;
    let mut max_flat  = 0.0f32;
    for (name, g, expect) in cases {
        let rec = fractal_recursion_score(g, &cfg);
        let sr  = self_replication_score(g, &cfg);
        let (x0, x1, y0, y1) = g.view_bounds();
        let base = render_bounds(&g.formula_weights(), &cfg, 128, 128,
                                 cfg.rendering.max_iter, x0, x1, y0, y1);
        let intric = field_intricacy(&base, 128, 128);
        if *expect { min_recur = min_recur.min(rec); } else { max_flat = max_flat.max(rec); }
        let tag = if !*expect { "flat" }
                  else if rec >= 0.30 { "recur (strong)" }
                  else { "recur (faint)" };
        println!("{:<28} {:>10.3} {:>10.3} {:>10.3}   {}", name, rec, intric, sr, tag);
    }
    println!("{}", "-".repeat(78));
    // The metric is view-dependent: it scores how visibly a copy of the whole
    // appears in THIS view. Canonical fractals at full view have near-sub-pixel
    // copies (faint but nonzero); zoomed boundary views (what the GA produces)
    // score strongly. The success criterion is clean SEPARATION: every recursive
    // case must outscore every non-recursive control.
    println!("min(recursive) = {:.3}   max(control) = {:.3}   separation margin = {:+.3}  [{}]",
             min_recur, max_flat, min_recur - max_flat,
             if min_recur > max_flat { "CLEAN" } else { "OVERLAP" });
    println!("({:.1}s total)", t0.elapsed().as_secs_f32());
}
