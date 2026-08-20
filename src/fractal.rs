use rayon::prelude::*;
use crate::config::Config;
use crate::genome::Genome;
use crate::formula::apply_formula;
#[cfg(feature = "wgpu-backend")]
use crate::render_gpu;

/// CPU fitness: (raw_png_entropy, multiscale_structured_entropy, angle_structure, behavioral_descriptor).
///
/// raw_png_entropy  — used for save-gate thresholding (min/max_entropy_prefilter).
/// multiscale_entropy — used for GA selection: geometric mean of fine (64px) and
///   coarse (16px average-pool) PNG entropy. Penalises granular noise because noise
///   averages to near-uniform at coarse scale; structured fractals stay complex at
///   all scales. Replace the old single-scale fitness for selection only.
/// angle_structure — structural richness of the bailout exit-angle field (DAG
///   genomes only); 0.0 unless angle_structure_weight > 0 (see
///   fitness::angle_structure_score and dag_render_with_angle).
pub fn evaluate_fitness_full(genome: &Genome, config: &Config) -> (f32, f32, f32, Vec<f32>) {
    let ew = config.optimization.eval_width;
    let eh = config.optimization.eval_height;
    let emi = config.optimization.eval_max_iter;
    let escape_times = render_cpu_iter(genome, config, ew, eh, emi);
    let raw_png = crate::fitness::png_compression_entropy(
        &escape_times, ew, eh, emi, &config.rendering.colormap,
    );
    let structured = crate::fitness::multiscale_entropy(
        &escape_times, ew, eh, emi, &config.rendering.colormap,
    );
    let angle_score = if config.optimization.angle_structure_weight != 0.0 && genome.uses_program() {
        let (_, angles) = dag_render_with_angle(genome, config, ew, eh, emi);
        crate::fitness::angle_structure_score(&angles, ew as usize)
    } else {
        0.0
    };
    let descriptor = crate::fitness::behavior_descriptor(&escape_times, emi);
    (raw_png, structured, angle_score, descriptor)
}

/// CPU rendering — returns smooth escape times [H*W].
pub fn render_cpu(genome: &Genome, config: &Config, width: u32, height: u32) -> Vec<f32> {
    render_cpu_iter(genome, config, width, height, config.rendering.max_iter)
}

/// Try a 3×3 grid of small view offsets around the genome's current view and
/// return a clone with the view params that maximise multiscale_entropy.
/// Used in try_save() so CLIP sees the best-composed version of each candidate.
/// The genome stored in the archive is NOT modified — only the render view changes.
pub fn best_entropy_view(genome: &Genome, config: &Config) -> Genome {
    let ew  = config.optimization.eval_width;
    let eh  = config.optimization.eval_height;
    let emi = config.optimization.eval_max_iter;
    // Search radius: ±15% of the current half-width (2/zoom), so the shift
    // is proportional to zoom level and never wanders far from the evolved view.
    let pan = 0.30 / genome.view_zoom.max(0.1);
    let offsets: [(f32, f32); 9] = [
        (-pan, -pan), (0.0, -pan), (pan, -pan),
        (-pan,  0.0), (0.0,  0.0), (pan,  0.0),
        (-pan,  pan), (0.0,  pan), (pan,  pan),
    ];
    let mut best_score = -1.0f32;
    let mut best = genome.clone();
    for (dx, dy) in offsets {
        let mut candidate = genome.clone();
        candidate.view_cx += dx;
        candidate.view_cy += dy;
        let et = render_cpu_iter(&candidate, config, ew, eh, emi);
        let score = crate::fitness::multiscale_entropy(&et, ew, eh, emi, &config.rendering.colormap);
        if score > best_score {
            best_score = score;
            best = candidate;
        }
    }
    best
}

/// Escape-time (and exit angle) for one pixel under the DAG iteration with
/// Phase-3/4 dynamics: optional coordinate warp, Julia vs Mandelbrot
/// initialization, and phoenix memory z_{n+1} = f(z,c) + p·z_{n-1}. Mirrors
/// the WGSL main loop.
///
/// Returns `(escape_time, exit_angle)` — `exit_angle` is `arg(z)` at the
/// moment of bailout (`atan2(zy, zx)`), computed in the same loop and at the
/// same return points as `escape_time` so the two values can never
/// numerically diverge. `0.0` for the non-finite-escape and max-iter
/// (interior) cases, which have no meaningful exit angle. Most callers only
/// want `.0` (escape time); see `dag_render_with_angle` for a caller that
/// keeps both.
#[allow(clippy::too_many_arguments)]
pub fn dag_escape_pixel(
    prog: &[crate::formula::OpNode], warp: &[crate::formula::OpNode],
    julia: bool, jc: (f32, f32), phoenix: (f32, f32), bailout_sq: f32,
    px: f32, py: f32, max_iter: u32,
) -> (f32, f32) {
    use crate::formula::eval_program;
    // Coordinate warp bends the pixel-derived input plane.
    let (mut ix, mut iy) = (px, py);
    if !warp.is_empty() {
        let (wx, wy) = eval_program(warp, px, py, px, py);
        ix = wx; iy = wy;
    }
    // Julia: pixel → z₀, c = constant. Mandelbrot: z₀ = 0, c = pixel.
    let (mut zx, mut zy, cx, cy) = if julia { (ix, iy, jc.0, jc.1) } else { (0.0, 0.0, ix, iy) };
    let (mut pzx, mut pzy) = (0.0f32, 0.0f32);
    for it in 0..max_iter {
        let (fx, fy) = eval_program(prog, zx, zy, cx, cy);
        // + phoenix·z_prev  (complex multiply)
        let nx = fx + phoenix.0 * pzx - phoenix.1 * pzy;
        let ny = fy + phoenix.0 * pzy + phoenix.1 * pzx;
        pzx = zx; pzy = zy;
        zx = nx; zy = ny;
        let ms = zx * zx + zy * zy;
        if ms > bailout_sq {
            let et = ((it as f32 + 1.0) - (ms.log2() * 0.5).log2()).max(0.0);
            return (et, zy.atan2(zx));
        }
        if !zx.is_finite() || !zy.is_finite() { return (it as f32, 0.0); }
    }
    (max_iter as f32, 0.0)
}

/// f64 (deep-zoom) version of `dag_escape_pixel` — same dynamics, double
/// precision, for the viewer's deep-zoom CPU path. Genome dynamics (f32) are
/// widened to f64. Mirrors `dag_escape_pixel` exactly.
#[allow(clippy::too_many_arguments)]
/// Returns (escape time, bailout exit angle `zy.atan2(zx)`) — the same pair
/// `dag_escape_pixel` (f32) returns, for the same reason: the viewer's "∠"
/// angle-coloring toggle, extended to this precision tier so it doesn't
/// silently go inert as soon as a render needs f64 (see
/// `video_export::render_cpu`'s `want_angle` gate and
/// `render_f64_with_angle`). Never DD — that tier stays unsupported by
/// design (scope-limited, see the same doc comment).
pub fn dag_escape_pixel_f64(
    prog: &[crate::formula::OpNode], warp: &[crate::formula::OpNode],
    julia: bool, jc: (f64, f64), phoenix: (f64, f64), bailout_sq: f64,
    px: f64, py: f64, max_iter: u32,
) -> (f32, f32) {
    use crate::formula::f64_impl::eval_program;
    let (mut ix, mut iy) = (px, py);
    if !warp.is_empty() {
        let (wx, wy) = eval_program(warp, px, py, px, py);
        ix = wx; iy = wy;
    }
    let (mut zx, mut zy, cx, cy) = if julia { (ix, iy, jc.0, jc.1) } else { (0.0, 0.0, ix, iy) };
    let (mut pzx, mut pzy) = (0.0f64, 0.0f64);
    for it in 0..max_iter {
        let (fx, fy) = eval_program(prog, zx, zy, cx, cy);
        let nx = fx + phoenix.0 * pzx - phoenix.1 * pzy;
        let ny = fy + phoenix.0 * pzy + phoenix.1 * pzx;
        pzx = zx; pzy = zy;
        zx = nx; zy = ny;
        let ms = zx * zx + zy * zy;
        if ms > bailout_sq {
            let et = ((it as f64 + 1.0) - (ms.log2() * 0.5).log2()).max(0.0) as f32;
            return (et, zy.atan2(zx) as f32);
        }
        if !zx.is_finite() || !zy.is_finite() { return (it as f32, 0.0); }
    }
    (max_iter as f32, 0.0)
}

/// Returns `(escape_time, zx, zy)` — the raw complex z components at the
/// moment of bailout, not just `arg(z)` like `dag_escape_pixel`. Added for
/// exploring a complex-valued autoencoder (Carl's request, 2026-08-07): the
/// real/imaginary parts and magnitude of the bailout value are a genuinely
/// different signal from escape time (which only encodes iteration COUNT,
/// discarding exactly where in the complex plane the orbit left the
/// bailout disk). `(0.0, 0.0)` for zx/zy on the non-finite-escape and
/// max-iter (interior) cases — same "no meaningful bailout value" edge
/// case `dag_escape_pixel`'s angle already documents. Deliberately a
/// SEPARATE function rather than changing `dag_escape_pixel`'s return
/// type — that one has established callers (viewer angle-coloring, GPU
/// angle rendering) this shouldn't risk disturbing for an exploratory
/// feature. Same loop body as `dag_escape_pixel`, just a different return.
#[allow(clippy::too_many_arguments)]
pub fn dag_escape_pixel_z(
    prog: &[crate::formula::OpNode], warp: &[crate::formula::OpNode],
    julia: bool, jc: (f32, f32), phoenix: (f32, f32), bailout_sq: f32,
    px: f32, py: f32, max_iter: u32,
) -> (f32, f32, f32) {
    use crate::formula::eval_program;
    let (mut ix, mut iy) = (px, py);
    if !warp.is_empty() {
        let (wx, wy) = eval_program(warp, px, py, px, py);
        ix = wx; iy = wy;
    }
    let (mut zx, mut zy, cx, cy) = if julia { (ix, iy, jc.0, jc.1) } else { (0.0, 0.0, ix, iy) };
    let (mut pzx, mut pzy) = (0.0f32, 0.0f32);
    for it in 0..max_iter {
        let (fx, fy) = eval_program(prog, zx, zy, cx, cy);
        let nx = fx + phoenix.0 * pzx - phoenix.1 * pzy;
        let ny = fy + phoenix.0 * pzy + phoenix.1 * pzx;
        pzx = zx; pzy = zy;
        zx = nx; zy = ny;
        let ms = zx * zx + zy * zy;
        if ms > bailout_sq {
            let et = ((it as f32 + 1.0) - (ms.log2() * 0.5).log2()).max(0.0);
            return (et, zx, zy);
        }
        if !zx.is_finite() || !zy.is_finite() { return (it as f32, 0.0, 0.0); }
    }
    (max_iter as f32, 0.0, 0.0)
}

/// f64 sibling of `dag_escape_pixel_z` — same rationale/edge cases as
/// `dag_escape_pixel_f64` relative to `dag_escape_pixel`.
#[allow(clippy::too_many_arguments)]
pub fn dag_escape_pixel_z_f64(
    prog: &[crate::formula::OpNode], warp: &[crate::formula::OpNode],
    julia: bool, jc: (f64, f64), phoenix: (f64, f64), bailout_sq: f64,
    px: f64, py: f64, max_iter: u32,
) -> (f32, f32, f32) {
    use crate::formula::f64_impl::eval_program;
    let (mut ix, mut iy) = (px, py);
    if !warp.is_empty() {
        let (wx, wy) = eval_program(warp, px, py, px, py);
        ix = wx; iy = wy;
    }
    let (mut zx, mut zy, cx, cy) = if julia { (ix, iy, jc.0, jc.1) } else { (0.0, 0.0, ix, iy) };
    let (mut pzx, mut pzy) = (0.0f64, 0.0f64);
    for it in 0..max_iter {
        let (fx, fy) = eval_program(prog, zx, zy, cx, cy);
        let nx = fx + phoenix.0 * pzx - phoenix.1 * pzy;
        let ny = fy + phoenix.0 * pzy + phoenix.1 * pzx;
        pzx = zx; pzy = zy;
        zx = nx; zy = ny;
        let ms = zx * zx + zy * zy;
        if ms > bailout_sq {
            let et = ((it as f64 + 1.0) - (ms.log2() * 0.5).log2()).max(0.0) as f32;
            return (et, zx as f32, zy as f32);
        }
        if !zx.is_finite() || !zy.is_finite() { return (it as f32, 0.0, 0.0); }
    }
    (max_iter as f32, 0.0, 0.0)
}

/// Double-double DAG escape — used for zoom > ~10¹¹ where f64 runs out.
/// Pixel coordinates arrive as `Dd`; iteration is dd-precise for polynomial
/// ops, f64 fallback for transcendentals (whose weights are only f32 anyway).
pub fn dag_escape_pixel_dd(
    prog: &[crate::formula::OpNode], warp: &[crate::formula::OpNode],
    julia: bool, jc: (f64, f64), phoenix: (f64, f64), bailout_sq: f64,
    px: crate::dd::Dd, py: crate::dd::Dd, max_iter: u32,
) -> f32 {
    use crate::dd::{Dd, eval_program_dd};
    // Warp in f64 (the warp grid doesn't need sub-pixel DD precision)
    let (ix, iy) = if !warp.is_empty() {
        let (wx, wy) = crate::formula::f64_impl::eval_program(warp, px.hi, py.hi, px.hi, py.hi);
        (Dd::from_f64(wx), Dd::from_f64(wy))
    } else {
        (px, py)
    };
    let (mut zx, mut zy, cx, cy) = if julia {
        (ix, iy, Dd::from_f64(jc.0), Dd::from_f64(jc.1))
    } else {
        (Dd::zero(), Dd::zero(), ix, iy)
    };
    let (mut pzx, mut pzy) = (Dd::zero(), Dd::zero());
    let pr = Dd::from_f64(phoenix.0);
    let pi = Dd::from_f64(phoenix.1);
    for it in 0..max_iter {
        let (fx, fy) = eval_program_dd(prog, zx, zy, cx, cy);
        let (nx, ny) = (fx + pr*pzx - pi*pzy, fy + pr*pzy + pi*pzx);
        (pzx, pzy) = (zx, zy);
        (zx, zy) = (nx, ny);
        let ms = (zx*zx + zy*zy).hi;
        if ms > bailout_sq {
            return ((it as f64 + 1.0) - (ms.log2() * 0.5).log2()).max(0.0) as f32;
        }
        if !zx.is_finite() || !zy.is_finite() { return it as f32; }
    }
    max_iter as f32
}

/// Double-double legacy-formula escape.
pub fn legacy_escape_pixel_dd(
    weights: &[(f64, f64)], bailout_sq: f64,
    px: crate::dd::Dd, py: crate::dd::Dd, max_iter: u32,
) -> f32 {
    use crate::dd::{Dd, apply_formula_dd};
    let (cx, cy) = (px, py);
    let (mut zx, mut zy) = (Dd::zero(), Dd::zero());
    for it in 0..max_iter {
        let (nx, ny) = apply_formula_dd(weights, zx, zy, cx, cy);
        (zx, zy) = (nx, ny);
        let ms = (zx*zx + zy*zy).hi;
        if ms > bailout_sq {
            return ((it as f64 + 1.0) - (ms.log2() * 0.5).log2()).max(0.0) as f32;
        }
        if !zx.is_finite() || !zy.is_finite() { return it as f32; }
    }
    max_iter as f32
}

pub fn render_cpu_iter(
    genome: &Genome, config: &Config, width: u32, height: u32, max_iter: u32,
) -> Vec<f32> {
    let bailout_sq = config.rendering.bailout * config.rendering.bailout;
    let (xmin, xmax, ymin, ymax) = genome.view_bounds();

    // Expression-DAG genomes evaluate via the register VM (GPU when available,
    // else the Rayon CPU fallback below).
    if genome.uses_program() {
        #[cfg(feature = "wgpu-backend")]
        if render_gpu::gpu_available() {
            let item = render_gpu::dag_item(genome);
            return render_gpu::render_batch_dag(
                &[item], &[(xmin, xmax, ymin, ymax)], width, height, max_iter,
            ).into_iter().next().unwrap_or_default();
        }
        let prog = &genome.program;
        let warp = &genome.warp;
        let julia = genome.julia_mode;
        let jc = (genome.julia_cre, genome.julia_cim);
        let phoenix = (genome.phoenix_re, genome.phoenix_im);
        let bsq = genome.bailout_radius * genome.bailout_radius;
        let wf = (width.saturating_sub(1)).max(1) as f32;
        let hf = (height.saturating_sub(1)).max(1) as f32;
        let n  = (width * height) as usize;
        return (0..n).into_par_iter().map(|idx| {
            let px = idx % width as usize;
            let py = idx / width as usize;
            let cx = xmin + (px as f32 / wf) * (xmax - xmin);
            let cy = ymin + (py as f32 / hf) * (ymax - ymin);
            dag_escape_pixel(prog, warp, julia, jc, phoenix, bsq, cx, cy, max_iter).0
        }).collect();
    }

    let fw = genome.formula_weights();

    // Try GPU first (batch API, but single-genome path).
    #[cfg(feature = "wgpu-backend")]
    if render_gpu::gpu_available() {
        return render_gpu::render_fractal(
            &fw, width, height, max_iter,
            xmin, xmax, ymin, ymax, bailout_sq,
        );
    }

    // CPU fallback: parallel over pixels via Rayon.
    let wf = (width.saturating_sub(1)).max(1) as f32;
    let hf = (height.saturating_sub(1)).max(1) as f32;
    let n  = (width * height) as usize;
    (0..n).into_par_iter().map(|idx| {
        let px = idx % width as usize;
        let py = idx / width as usize;
        let cx = xmin + (px as f32 / wf) * (xmax - xmin);
        let cy = ymin + (py as f32 / hf) * (ymax - ymin);
        let mut zx = 0.0f32;
        let mut zy = 0.0f32;
        for iter in 0..max_iter {
            let (nzx, nzy) = apply_formula(&fw, zx, zy, cx, cy);
            zx = nzx; zy = nzy;
            let mod_sq = zx * zx + zy * zy;
            if mod_sq > bailout_sq {
                let log2_mod = mod_sq.log2() * 0.5;
                let nu = log2_mod.log2();
                return (iter as f32 + 1.0 - nu).max(0.0);
            }
            if !zx.is_finite() || !zy.is_finite() { return iter as f32; }
        }
        max_iter as f32
    }).collect()
}

/// DAG-genome-only sibling of `render_cpu_iter` that also captures the
/// exit-angle channel (see `dag_escape_pixel`'s doc comment). GPU when
/// available (via `render_batch_dag_angle`, capture_angle=true), else the
/// Rayon CPU fallback keeping both tuple components. Caller must ensure
/// `genome.uses_program()` — legacy genomes have no angle data, out of scope
/// for this feature.
pub fn dag_render_with_angle(
    genome: &Genome, _config: &Config, width: u32, height: u32, max_iter: u32,
) -> (Vec<f32>, Vec<f32>) {
    let (xmin, xmax, ymin, ymax) = genome.view_bounds();

    #[cfg(feature = "wgpu-backend")]
    if render_gpu::gpu_available() {
        let item = render_gpu::dag_item(genome);
        let (mut ets, mut angs) = render_gpu::render_batch_dag_angle(
            &[item], &[(xmin, xmax, ymin, ymax)], width, height, max_iter, true,
        );
        return (ets.pop().unwrap_or_default(), angs.pop().unwrap_or_default());
    }

    let prog = &genome.program;
    let warp = &genome.warp;
    let julia = genome.julia_mode;
    let jc = (genome.julia_cre, genome.julia_cim);
    let phoenix = (genome.phoenix_re, genome.phoenix_im);
    let bsq = genome.bailout_radius * genome.bailout_radius;
    let wf = (width.saturating_sub(1)).max(1) as f32;
    let hf = (height.saturating_sub(1)).max(1) as f32;
    let n  = (width * height) as usize;
    (0..n).into_par_iter().map(|idx| {
        let px = idx % width as usize;
        let py = idx / width as usize;
        let cx = xmin + (px as f32 / wf) * (xmax - xmin);
        let cy = ymin + (py as f32 / hf) * (ymax - ymin);
        dag_escape_pixel(prog, warp, julia, jc, phoenix, bsq, cx, cy, max_iter)
    }).unzip()
}

/// Render a bare DAG program at explicit bounds and mode/bailout, no
/// warp/phoenix. Used only by known-formula matching (`known_formula_match`)
/// — `render_bounds` can't be reused here: it hardcodes the legacy
/// `apply_formula` path and has no notion of a DAG `program`.
#[allow(clippy::too_many_arguments)]
fn render_program_bounds(
    prog: &[crate::formula::OpNode], julia: bool, jc: (f32, f32),
    w: u32, h: u32, max_iter: u32, bailout_sq: f32,
    xmin: f32, xmax: f32, ymin: f32, ymax: f32,
) -> Vec<f32> {
    let wf = (w.saturating_sub(1)).max(1) as f32;
    let hf = (h.saturating_sub(1)).max(1) as f32;
    (0..(w * h) as usize).into_par_iter().map(|idx| {
        let px = idx % w as usize;
        let py = idx / w as usize;
        let cx = xmin + (px as f32 / wf) * (xmax - xmin);
        let cy = ymin + (py as f32 / hf) * (ymax - ymin);
        dag_escape_pixel(prog, &[], julia, jc, (0.0, 0.0), bailout_sq, cx, cy, max_iter).0
    }).collect()
}

const KF_RES: u32  = 32;    // cheap fingerprint, not a quality render
const KF_ITER: u32 = 100;
const KF_PS: usize = 16;    // pooled vector side — mirrors self_replication/recursion's ~2-3x downsample ratio
const KF_VIEW: (f32, f32, f32, f32) = (-2.0, 2.0, -2.0, 2.0); // standard Mandelbrot-plane window
/// Similarity floor for reporting a match. Calibrated against a real
/// ~165-genome archive sample (mode/bailout-matched comparison, see
/// `known_formula_match`'s doc comment): scores spread roughly uniformly
/// over [-0.03, 0.93] with median ~0.25, so 0.65 keeps only the top ~8-9%
/// as confident matches rather than forcing a pick for everything.
pub const KNOWN_FORMULA_THRESHOLD: f32 = 0.65;

/// Closest named reference formula (`known_formulas::LIBRARY`) to a DAG
/// genome's BASE `program`, by behavioral correlation — a discovery/curiosity
/// label for the human user only. **Never used in fitness, selection, or
/// seed ranking.**
///
/// Compares behaviorally (rendered field shape), not symbolically: this
/// codebase has no canonicalization anywhere (no dead-code pruning —
/// mutation can and does orphan nodes that stay in the array and get
/// evaluated forever; no constant-folding; no canonical operand order for
/// commutative ops), so a symbolic-equality check would both miss
/// cosmetically-different-but-equivalent programs and hard-reject a formula
/// that's "basically Mandelbrot but the coefficient is 0.97, not 1.0" —
/// exactly the near-miss case this feature should catch.
///
/// Ignores the genome's `warp` and phoenix memory (not expressible in any
/// reference, and not part of "which base iteration rule is this"), and
/// always uses a fixed standard [-2,2]² view rather than the genome's own
/// (evolved for aesthetic composition, often zoomed deep into an
/// unrepresentative sub-region). **Does** use the genome's own
/// `julia_mode`/`julia_cre/cim` and `bailout_radius` for BOTH the candidate
/// and every reference render — calibrated against a real archive sample:
/// forcing every genome into Mandelbrot-mode (z0=0, c=pixel) regardless of
/// its own mode made ~90% of real Julia-mode genomes (the large majority of
/// the archive) render as a degenerate flat field at the standard view,
/// since a Julia-mode formula's evolved behavior lives at a specific fixed
/// c that has no relationship to sweeping c across [-2,2]². Rendering
/// candidate and references under the SAME mode/c/bailout keeps the
/// comparison self-consistent (still apples-to-apples between candidate and
/// reference) while actually producing a structured field for the genomes
/// that exist in practice.
///
/// Returns `None` below `KNOWN_FORMULA_THRESHOLD` rather than forcing a
/// spurious top-1 pick — "no close match" is the expected, honest answer
/// for most evolved formulas.
pub fn known_formula_match(genome: &Genome) -> Option<(&'static str, f32)> {
    if !genome.uses_program() || genome.program.is_empty() { return None; }
    let bsq = genome.bailout_radius.max(2.0).powi(2);
    let julia = genome.julia_mode;
    let jc = (genome.julia_cre, genome.julia_cim);
    let (x0, x1, y0, y1) = KF_VIEW;

    let cand_field = render_program_bounds(&genome.program, julia, jc, KF_RES, KF_RES, KF_ITER, bsq, x0, x1, y0, y1);
    let cand_vec   = structure_vec(&cand_field, KF_RES as usize, KF_RES as usize, KF_PS);
    if cand_vec.iter().all(|v| *v == 0.0) { return None; } // degenerate/flat candidate field

    let mut best: Option<(&'static str, f32)> = None;
    for kf in crate::known_formulas::LIBRARY {
        let prog  = (kf.build)();
        let field = render_program_bounds(&prog, julia, jc, KF_RES, KF_RES, KF_ITER, bsq, x0, x1, y0, y1);
        let vref  = structure_vec(&field, KF_RES as usize, KF_RES as usize, KF_PS);
        if vref.iter().all(|v| *v == 0.0) { continue; }
        let c = correlation(&cand_vec, &vref);
        if best.map_or(true, |(_, bc)| c > bc) { best = Some((kf.name, c)); }
    }
    best.filter(|&(_, c)| c >= KNOWN_FORMULA_THRESHOLD)
}

/// Render explicit fractal-plane bounds (GPU when available, else CPU) → escape times.
pub fn render_bounds(
    fw: &[(f32, f32)], config: &Config, width: u32, height: u32, max_iter: u32,
    xmin: f32, xmax: f32, ymin: f32, ymax: f32,
) -> Vec<f32> {
    let bailout_sq = config.rendering.bailout * config.rendering.bailout;

    #[cfg(feature = "wgpu-backend")]
    if render_gpu::gpu_available() {
        return render_gpu::render_fractal(fw, width, height, max_iter, xmin, xmax, ymin, ymax, bailout_sq);
    }

    let wf = (width.saturating_sub(1)).max(1) as f32;
    let hf = (height.saturating_sub(1)).max(1) as f32;
    let n  = (width * height) as usize;
    (0..n).into_par_iter().map(|idx| {
        let px = idx % width as usize;
        let py = idx / width as usize;
        let cx = xmin + (px as f32 / wf) * (xmax - xmin);
        let cy = ymin + (py as f32 / hf) * (ymax - ymin);
        let mut zx = 0.0f32;
        let mut zy = 0.0f32;
        for iter in 0..max_iter {
            let (nzx, nzy) = apply_formula(fw, zx, zy, cx, cy);
            zx = nzx; zy = nzy;
            let mod_sq = zx * zx + zy * zy;
            if mod_sq > bailout_sq {
                let nu = (mod_sq.log2() * 0.5).log2();
                return (iter as f32 + 1.0 - nu).max(0.0);
            }
            if !zx.is_finite() || !zy.is_finite() { return iter as f32; }
        }
        max_iter as f32
    }).collect()
}

/// Down-pool an escape-time field to a `PS×PS` contrast-normalised (z-scored) vector.
/// Z-scoring makes it invariant to the overall escape-time offset, which rises with
/// zoom depth — what survives is the *shape* of the structure.
pub fn structure_vec(field: &[f32], w: usize, h: usize, ps: usize) -> Vec<f32> {
    let mut pooled = vec![0.0f32; ps * ps];
    for py in 0..ps {
        for px in 0..ps {
            let x0 = px * w / ps;
            let x1 = ((px + 1) * w / ps).max(x0 + 1).min(w);
            let y0 = py * h / ps;
            let y1 = ((py + 1) * h / ps).max(y0 + 1).min(h);
            let mut sum = 0.0f32;
            let mut cnt = 0u32;
            for y in y0..y1 {
                for x in x0..x1 {
                    sum += field[y * w + x];
                    cnt += 1;
                }
            }
            pooled[py * ps + px] = sum / cnt.max(1) as f32;
        }
    }
    let n    = pooled.len() as f32;
    let mean = pooled.iter().sum::<f32>() / n;
    let var  = pooled.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let std  = var.sqrt();
    if std < 1e-6 {
        return vec![0.0; ps * ps]; // degenerate / flat → zero vector → zero correlation
    }
    pooled.iter().map(|v| (v - mean) / std).collect()
}

/// Intricacy of an escape-time field in [0, 1]: the density of gradient sign-flips
/// along horizontal and vertical scanlines — i.e. how often the field reverses
/// direction (local maxima/minima encountered when sweeping a line across it).
///
/// A *fractal* field is non-monotone: it folds into iteration bands and weaving
/// filaments, so a scanline reverses direction many times. A smooth monotone map
/// (the trivial `z+c`, whose field is ≈`2/|c|` — a self-similar but featureless
/// `1/r` radial ramp) reverses at most once per line. Self-similarity matching
/// alone can't tell a fractal from such a degenerate scale-invariant ramp; this
/// absolute gate can. (Pure noise also scores high here, but noise fails the copy
/// match, so the two together still single out genuine recursion.)
pub fn field_intricacy(field: &[f32], w: usize, h: usize) -> f32 {
    if w < 3 || h < 3 { return 0.0; }
    let n = field.len() as f32;
    let mean = field.iter().sum::<f32>() / n;
    let std  = (field.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n).sqrt();
    if std < 1e-6 { return 0.0; }
    let eps = std * 0.02; // treat sub-noise wiggles as flat (no flip)

    let mut flips = 0u32;
    let mut total = 0u32;
    // Count direction reversals along a line of values via the running sign of
    // significant consecutive differences.
    let mut scan = |get: &dyn Fn(usize) -> f32, len: usize, t: &mut u32, f: &mut u32| {
        let mut last_sign = 0i8;
        for i in 1..len {
            let d = get(i) - get(i - 1);
            *t += 1;
            if d.abs() <= eps { continue; }
            let s = if d > 0.0 { 1i8 } else { -1i8 };
            if last_sign != 0 && s != last_sign { *f += 1; }
            last_sign = s;
        }
    };
    for y in 0..h {
        scan(&|x| field[y * w + x], w, &mut total, &mut flips);
    }
    for x in 0..w {
        scan(&|y| field[y * w + x], h, &mut total, &mut flips);
    }
    flips as f32 / total.max(1) as f32
}

/// Shared by both exploration candidate-scoring paths (`explore::sweep`'s
/// `field_to_candidate` and `vae_explore::score_position`) that use
/// `local_edge_density`/`local_intricacy` — one definition so the two
/// stay in sync rather than risking silent drift between two copies.
/// 4 gives 32×32 sub-tiles at `vae_explore::COARSE_SAMPLE_RES`=128 (or
/// 16×16 at `explore::SWEEP_RES`=64) — small enough to isolate a
/// localized feature without going so small the per-tile statistic gets
/// noisy.
pub const LOCAL_TILE_GRID: usize = 4;
/// Local edge_density floor for overriding a whole-patch `is_degenerate`/
/// `rgb_is_degenerate` verdict — NOT yet calibrated against a real
/// distribution the way some of this project's other thresholds are (e.g.
/// `vae_explore::DEDUP_POS_TOL`'s own history); a reasonable starting
/// point, worth revisiting once a few live runs show whether it's over-
/// or under-permissive.
pub const LOCAL_EDGE_DEGENERATE_OVERRIDE: f32 = 0.03;
/// Minimum `local_edge_contrast`/`local_intricacy_contrast` needed before
/// a candidate's SCORE (not just its degenerate-gate eligibility) gets
/// rescued by its local max — see `tile_stats`'s doc comment for why a
/// contrast gate exists at all (a bare max caused a real regression).
/// A starting value (0.05), NOT exhaustively swept — but confirmed to
/// work on two independently real cases via `explorer debug-sweep` before
/// trusting it: it correctly fires for a genuinely-hidden feature (a flat
/// disc with a thin detailed ring at its edge — Carl's real target,
/// 2026-08-11, rescued from invisible to rank #4/972 under `edge`
/// scoring) and, per a live `vae-explore` run of 1336 zones reaching
/// zoom 5.5×10¹¹× its base, does NOT reintroduce the earlier bare-max
/// regression (zoom stayed healthily distributed across the whole range,
/// not clustered near the shallow end). Worth revisiting with a proper
/// sweep if a future case falls right at this boundary.
pub const LOCAL_CONTRAST_MIN: f32 = 0.05;

/// Splits `field` into a `tiles` × `tiles` grid and scores each sub-tile
/// on `metric`, returning `(max, contrast)` where `contrast = max -
/// mean(all tiles)` — a "local maximum" AND "local peakedness" companion
/// to a whole-patch statistic like `field_intricacy`/`edge_density`,
/// which dilutes anything confined to a small part of the patch.
/// Confirmed a REAL, not theoretical, blind spot (Carl, 2026-08-11): a
/// crisp circular boundary occupying, say, 10% of a candidate crop's area
/// still only contributes ~10% of the crop's total edge-pair count, so
/// even a perfectly sharp boundary reads as "10% edges" on the
/// whole-patch average.
///
/// `contrast` exists because raw `max` ALONE isn't enough to safely act
/// on (confirmed the hard way: an earlier version of this fix blended
/// bare `max` straight into the score and caused a REAL regression — a
/// wide/shallow crop has a much higher a-priori chance of containing
/// SOME locally-okay tile purely because it covers more ground, so
/// rewarding "best tile anywhere" made the search systematically avoid
/// zooming in at all). `contrast` tells the two cases apart: a crop
/// that's uniformly medium-busy everywhere (the regression case) has
/// every tile close to the mean, so `max - mean` stays small even if
/// `max` itself is decent; a crop with one genuinely standout feature
/// against an otherwise plain/uniform background (Carl's actual case — a
/// large flat disc with a thin detailed ring at its edge) has a `max`
/// that towers over the other tiles' mean. Verified directionally on
/// synthetic data before wiring in: a disc-plus-thin-ring field measured
/// contrast=0.087 vs. a uniformly-busy field's 0.016 — comparable
/// whole-patch/max values, clearly different contrast.
///
/// Falls back to `(metric(field), 0.0)` if the field is too small to
/// tile meaningfully.
fn tile_stats(field: &[f32], w: usize, h: usize, tiles: usize, metric: impl Fn(&[f32], usize, usize) -> f32) -> (f32, f32) {
    if tiles <= 1 || w < tiles * 2 || h < tiles * 2 {
        return (metric(field, w, h), 0.0);
    }
    let (tw, th) = (w / tiles, h / tiles);
    let mut vals = Vec::with_capacity(tiles * tiles);
    for ty in 0..tiles {
        for tx in 0..tiles {
            let (x0, y0) = (tx * tw, ty * th);
            let sub: Vec<f32> = (0..th).flat_map(|dy| (0..tw).map(move |dx| (dy, dx)))
                .map(|(dy, dx)| field[(y0 + dy) * w + (x0 + dx)])
                .collect();
            vals.push(metric(&sub, tw, th));
        }
    }
    let max = vals.iter().cloned().fold(0.0f32, f32::max);
    let mean = vals.iter().sum::<f32>() / vals.len() as f32;
    (max, (max - mean).max(0.0))
}

/// Tile-max companion to `field_intricacy` — see `tile_stats`'s doc
/// comment. Used ONLY for the `is_degenerate` gate override (a max alone
/// is safe there — see `tile_stats`'s doc comment for why it's NOT safe
/// to blend into the actual score).
pub fn local_intricacy(field: &[f32], w: usize, h: usize, tiles: usize) -> f32 {
    tile_stats(field, w, h, tiles, field_intricacy).0
}

/// Tile-max companion to `edge_density` — see `tile_stats`'s doc comment.
pub fn local_edge_density(field: &[f32], w: usize, h: usize, tiles: usize) -> f32 {
    tile_stats(field, w, h, tiles, edge_density).0
}

/// `(local_max, local_contrast)` for `edge_density` — see `tile_stats`'s
/// doc comment. Unlike `local_edge_density` (max alone), `contrast` is
/// safe to gate an actual score boost on: it specifically distinguishes a
/// genuinely-hidden standout feature from a merely uniformly-busy crop.
pub fn local_edge_contrast(field: &[f32], w: usize, h: usize, tiles: usize) -> (f32, f32) {
    tile_stats(field, w, h, tiles, edge_density)
}

/// `(local_max, local_contrast)` for `field_intricacy` — see
/// `local_edge_contrast`'s doc comment.
pub fn local_intricacy_contrast(field: &[f32], w: usize, h: usize, tiles: usize) -> (f32, f32) {
    tile_stats(field, w, h, tiles, field_intricacy)
}

/// Pearson correlation of two equal-length z-scored vectors → [-1, 1].
pub fn correlation(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() { return 0.0; }
    let n = a.len() as f32;
    a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>() / n
}

/// Fraction of adjacent pixel pairs whose escape time jumps by a notable amount —
/// a scale-free measure of how much boundary/structure is present in a frame.
pub fn edge_density(field: &[f32], w: usize, h: usize) -> f32 {
    if field.len() < 4 { return 0.0; }
    let maxv = field.iter().cloned().fold(0.0_f32, f32::max).max(1.0);
    let thr  = maxv * 0.01;
    let mut edges = 0u32;
    let mut total = 0u32;
    for y in 0..h {
        for x in 0..w {
            let t = field[y * w + x];
            if x + 1 < w { if (t - field[y * w + x + 1]).abs() > thr { edges += 1; } total += 1; }
            if y + 1 < h { if (t - field[(y + 1) * w + x]).abs() > thr { edges += 1; } total += 1; }
        }
    }
    edges as f32 / total.max(1) as f32
}

/// Richest interior boundary point: the highest-gradient pixel within the central
/// 70% of the frame (so the next zoom has room around it). Returns pixel coords.
fn richest_boundary_point(field: &[f32], w: usize, h: usize) -> Option<(usize, usize)> {
    let maxv = field.iter().cloned().fold(0.0_f32, f32::max).max(1.0);
    let (mx0, mx1) = (w * 15 / 100, w * 85 / 100);
    let (my0, my1) = (h * 15 / 100, h * 85 / 100);
    let mut best = (-1.0f32, 0usize, 0usize);
    for py in my0.max(1)..my1.min(h - 1) {
        for px in mx0.max(1)..mx1.min(w - 1) {
            let t = field[py * w + px];
            if t < maxv * 0.03 || t > maxv * 0.98 { continue; }
            let gx = (field[py * w + px + 1] - field[py * w + px - 1]).abs();
            let gy = (field[(py + 1) * w + px] - field[(py - 1) * w + px]).abs();
            let g  = gx + gy;
            if g > best.0 { best = (g, px, py); }
        }
    }
    if best.0 < 0.0 { None } else { Some((best.1, best.2)) }
}

/// Zoom self-replication score in [0, 1].
///
/// Does the fractal keep reproducing rich structure as you zoom into its
/// boundary? (The defining Mandelbrot property — infinite self-similar detail.)
///
/// Method: render the base view, then repeatedly zoom in — at each level
/// re-centring on the richest boundary point and rendering deeper. We track:
///  1. **Boundary persistence** — does edge density stay high at depth, or
///     smooth away (non-fractal)?
///  2. **Cross-scale shape correlation** — does each zoom level structurally
///     resemble the previous (contrast-normalised), i.e. self-similar?
/// The score combines the two. A smooth/degenerate map collapses to ≈0; a
/// fractal that stays complex and self-similar across scales approaches 1.
pub fn self_replication_score(genome: &Genome, config: &Config) -> f32 {
    const RES: u32   = 96;
    const PS:  usize = 32;
    const LEVELS: usize = 5;     // base + 4 deeper zooms
    const ZOOM_STEP: f32 = 5.0;  // 5⁴ ≈ 625× total depth

    let fw = genome.formula_weights();
    let mi = config.rendering.max_iter;
    let (w, h) = (RES as usize, RES as usize);

    let (x0, x1, y0, y1) = genome.view_bounds();
    let mut cx   = (x0 + x1) * 0.5;
    let mut cy   = (y0 + y1) * 0.5;
    let mut half = (x1 - x0) * 0.5;

    let mut edge_dens:  Vec<f32>      = Vec::with_capacity(LEVELS);
    let mut structs:    Vec<Vec<f32>> = Vec::with_capacity(LEVELS);

    for level in 0..LEVELS {
        let field = render_bounds(&fw, config, RES, RES, mi,
                                  cx - half, cx + half, cy - half, cy + half);
        edge_dens.push(edge_density(&field, w, h));
        structs.push(structure_vec(&field, w, h, PS));

        // Re-centre on the richest boundary point for the next (deeper) zoom.
        if level + 1 < LEVELS {
            match richest_boundary_point(&field, w, h) {
                Some((px, py)) => {
                    let wf = (w - 1).max(1) as f32;
                    let hf = (h - 1).max(1) as f32;
                    cx = (cx - half) + (px as f32 / wf) * (2.0 * half);
                    cy = (cy - half) + (py as f32 / hf) * (2.0 * half);
                    half /= ZOOM_STEP;
                }
                // No boundary left to follow → structure smoothed out; stop early.
                None => break,
            }
        }
    }

    if structs.is_empty() { return 0.0; }
    let base_ed = edge_dens[0].max(1e-4);
    if base_ed < 0.01 { return 0.0; } // base has essentially no structure

    // 1. Persistence: average retained edge density at each deeper level vs base.
    //    Reaching fewer than LEVELS levels (broke early) counts the missing levels
    //    as zero — a fractal that smooths out is correctly penalised.
    let mut persist = 0.0f32;
    for i in 1..LEVELS {
        let ratio = edge_dens.get(i).copied().unwrap_or(0.0) / base_ed;
        persist += ratio.min(1.0);
    }
    persist /= (LEVELS - 1) as f32;

    // 2. Self-similarity: mean positive correlation between consecutive scales.
    let mut corr = 0.0f32;
    let mut pairs = 0u32;
    for i in 0..structs.len().saturating_sub(1) {
        corr += correlation(&structs[i], &structs[i + 1]).max(0.0);
        pairs += 1;
    }
    if pairs > 0 { corr /= pairs as f32; }

    // Persistence is the dominant signal (it directly measures "stays complex at
    // depth"); shape correlation refines it. Weighted blend, clamped to [0,1].
    (0.65 * persist + 0.35 * corr).clamp(0.0, 1.0)
}

/// The 8 dihedral (square-symmetry) orientations of a `ps×ps` field, as flat
/// vectors. A z-scored field stays z-scored under any of these (they only permute
/// the entries), so the variants can be fed straight into `correlation`. Used to
/// match miniature copies that appear rotated or mirrored relative to the whole.
pub fn dihedral_variants(v: &[f32], ps: usize) -> Vec<Vec<f32>> {
    // (row, col) -> source (row, col) for each of the 8 transforms.
    let maps: [fn(usize, usize, usize) -> (usize, usize); 8] = [
        |r, c, _| (r, c),                       // identity
        |r, c, n| (c, n - 1 - r),               // rot 90
        |r, c, n| (n - 1 - r, n - 1 - c),       // rot 180
        |r, c, n| (n - 1 - c, r),               // rot 270
        |r, c, n| (r, n - 1 - c),               // flip horizontal
        |r, c, n| (n - 1 - r, c),               // flip vertical
        |r, c, _| (c, r),                       // transpose
        |r, c, n| (n - 1 - c, n - 1 - r),       // anti-transpose
    ];
    maps.iter().map(|m| {
        let mut out = vec![0.0f32; ps * ps];
        for r in 0..ps {
            for c in 0..ps {
                let (sr, sc) = m(r, c, ps);
                out[r * ps + c] = v[sr * ps + sc];
            }
        }
        out
    }).collect()
}

/// Localised edge density within a rectangular cell of `field`.
fn cell_edge_density(field: &[f32], w: usize, x0: usize, x1: usize, y0: usize, y1: usize) -> f32 {
    let maxv = field.iter().cloned().fold(0.0_f32, f32::max).max(1.0);
    let thr  = maxv * 0.01;
    let (mut edges, mut total) = (0u32, 0u32);
    for y in y0..y1 {
        for x in x0..x1 {
            let t = field[y * w + x];
            if x + 1 < x1 { if (t - field[y * w + x + 1]).abs() > thr { edges += 1; } total += 1; }
            if y + 1 < y1 { if (t - field[(y + 1) * w + x]).abs() > thr { edges += 1; } total += 1; }
        }
    }
    edges as f32 / total.max(1) as f32
}

/// Candidate centres likely to hold an embedded copy of the whole set.
///
/// A baby-Mandelbrot is a small *interior island* (pixels that never escape)
/// wrapped in *boundary structure*. We grid the frame and rank each cell by
/// `local_edge_density · (1 + 2·island_bonus)`, where the island bonus rewards a
/// cell that holds some — but not all — interior. Returns up to `k` cell-centre
/// pixel coordinates, best first.
/// Every cell-center of a `grid`×`grid` partition of the frame, unfiltered
/// — a full dense correlation sweep instead of `recursion_candidates`' cheap
/// island/edge heuristic prefilter. Confirmed empirically the heuristic
/// itself (not just how many of its top-K candidates survive) can miss a
/// genuinely strong candidate entirely, for genomes whose structure doesn't
/// present as a contained "island" (e.g. concentric rings rather than a
/// cardioid+bulb): a real archive genome's best embedded copy scored raw
/// template correlation 0.68 under a dense sweep, vs. a maximum of 0.11 from
/// anything the heuristic ever proposed at that grid cell density, at any K.
/// Only used at wormhole descent level 0 (see `wormhole_search_many`):
/// affordable there (native reference zoom, cheapest arithmetic tier) and
/// most consequential, since every deeper level's descent starts from
/// wherever level 0 landed — a wrong level-0 neighborhood can't be recovered
/// from later.
fn dense_grid_candidates(w: usize, h: usize, grid: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(grid * grid);
    for gy in 0..grid {
        for gx in 0..grid {
            let x0 = gx * w / grid;
            let x1 = ((gx + 1) * w / grid).max(x0 + 1).min(w);
            let y0 = gy * h / grid;
            let y1 = ((gy + 1) * h / grid).max(y0 + 1).min(h);
            out.push(((x0 + x1) / 2, (y0 + y1) / 2));
        }
    }
    out
}

fn recursion_candidates(
    field: &[f32], w: usize, h: usize, max_iter: u32, grid: usize, k: usize,
) -> Vec<(usize, usize)> {
    let interior_thr = max_iter as f32 * 0.95;
    let mut scored: Vec<(f32, usize, usize)> = Vec::with_capacity(grid * grid);
    for gy in 0..grid {
        for gx in 0..grid {
            let x0 = gx * w / grid;
            let x1 = ((gx + 1) * w / grid).max(x0 + 1).min(w);
            let y0 = gy * h / grid;
            let y1 = ((gy + 1) * h / grid).max(y0 + 1).min(h);
            let (mut interior, mut cnt) = (0u32, 0u32);
            for y in y0..y1 {
                for x in x0..x1 {
                    if field[y * w + x] >= interior_thr { interior += 1; }
                    cnt += 1;
                }
            }
            let island = interior as f32 / cnt.max(1) as f32;
            // Reward a partial island (a contained body), ignore solid/empty cells.
            let island_bonus = if island > 0.01 && island < 0.70 { island } else { 0.0 };
            let ed = cell_edge_density(field, w, x0, x1, y0, y1);
            let score = ed * (1.0 + 2.0 * island_bonus);
            scored.push((score, (x0 + x1) / 2, (y0 + y1) / 2));
        }
    }
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().take(k).filter(|s| s.0 > 0.0).map(|(_, x, y)| (x, y)).collect()
}

/// Fractal-recursion score in [0, 1] — "a fractal inside a fractal".
///
/// Detects *embedded miniature copies of the whole set* (the baby-Mandelbrot
/// phenomenon), which is what makes a fractal feel infinitely self-referential.
///
/// This is deliberately distinct from [`self_replication_score`]: that one asks
/// whether boundary *detail persists* as you zoom (correlating **consecutive**
/// scales). This one asks whether a **complete small copy of the entire
/// structure** reappears somewhere inside it — by matching the global whole-set
/// template against re-rendered sub-windows at several smaller scales and
/// locations, across all 8 dihedral orientations.
///
/// Method:
///  1. Render the base view; build the global template (contrast-normalised).
///  2. Pick candidate centres that look like contained copies (interior island
///     wrapped in boundary structure), plus the richest boundary point.
///  3. For each candidate × scale, render the sub-window and correlate it with
///     the global template (best orientation). The peak correlation is the score.
///
/// A non-recursive map yields sub-windows that look nothing like the whole (≈0);
/// a Mandelbrot-like map, whose mini-copies recur at depth, approaches 1.
pub fn fractal_recursion_score(genome: &Genome, config: &Config) -> f32 {
    const BASE_RES: u32 = 128;
    const WIN_RES:  u32 = 96;
    const PS:  usize = 24;
    const GRID: usize = 6;     // 6×6 candidate cells over the base frame
    const K:    usize = 4;     // re-render the 4 most copy-like cells
    const SCALES: [f32; 3] = [6.0, 14.0, 30.0]; // sub-window = base half-width / scale

    let fw = genome.formula_weights();
    let mi = config.rendering.max_iter;
    let (bw, bh) = (BASE_RES as usize, BASE_RES as usize);

    let (x0, x1, y0, y1) = genome.view_bounds();
    let bhalf = (x1 - x0) * 0.5;

    let base = render_bounds(&fw, config, BASE_RES, BASE_RES, mi, x0, x1, y0, y1);
    let base_ed = edge_density(&base, bw, bh);
    if base_ed < 0.01 { return 0.0; } // whole set is essentially featureless

    // Intricacy gate: the whole set must be a genuine (non-monotone) fractal, not a
    // smooth scale-invariant ramp like `z+c` whose windows trivially correlate with
    // the whole. Below LO → degenerate/monotone, no credit; ramp to full credit at HI.
    const INTRIC_LO: f32 = 0.010;
    const INTRIC_HI: f32 = 0.030;
    let intric = field_intricacy(&base, bw, bh);
    let gate = ((intric - INTRIC_LO) / (INTRIC_HI - INTRIC_LO)).clamp(0.0, 1.0);
    if gate <= 0.0 { return 0.0; }

    // Match shape on the raw (contrast-normalised) template — it carries the
    // mid-frequency "looks like the whole set" signal that a pure Laplacian throws
    // away. The roughness gate above is what rejects smooth gradients.
    let global = structure_vec(&base, bw, bh, PS);
    if global.iter().all(|v| *v == 0.0) { return 0.0; } // flat → no template to match
    let global_orients = dihedral_variants(&global, PS);

    // Boundary descent: search for copies of the whole at the base view, then
    // follow the richest boundary point down a few zoom levels and keep searching.
    // The global template stays fixed (a baby-Mandelbrot deep in the boundary is a
    // copy of the TOP-level whole set); descending just reaches the scale at which
    // the copy is large enough to resolve — without it, full-view multibrots whose
    // copies are sub-pixel score ~0.
    const DESCENT_LEVELS: usize = 3;
    const DESCENT_STEP:   f32   = 8.0;

    let mut cx_l   = (x0 + x1) * 0.5;
    let mut cy_l   = (y0 + y1) * 0.5;
    let mut half_l = bhalf;
    let mut best   = 0.0f32;

    for level in 0..DESCENT_LEVELS {
        // Reuse the base render at level 0; render the zoomed view deeper.
        let field = if level == 0 {
            base.clone()
        } else {
            render_bounds(&fw, config, BASE_RES, BASE_RES, mi,
                          cx_l - half_l, cx_l + half_l, cy_l - half_l, cy_l + half_l)
        };
        best = best.max(best_copy_match(
            &fw, config, mi, &field, BASE_RES,
            cx_l, cy_l, half_l, base_ed, &global_orients,
            GRID, K, WIN_RES, PS, &SCALES,
        ));

        // Re-centre on the richest boundary point for the next, deeper level.
        if level + 1 < DESCENT_LEVELS {
            match richest_boundary_point(&field, bw, bh) {
                Some((px, py)) => {
                    let wf = (bw - 1).max(1) as f32;
                    let hf = (bh - 1).max(1) as f32;
                    cx_l   = (cx_l - half_l) + (px as f32 / wf) * (2.0 * half_l);
                    cy_l   = (cy_l - half_l) + (py as f32 / hf) * (2.0 * half_l);
                    half_l /= DESCENT_STEP;
                }
                None => break, // boundary smoothed out → nothing deeper to find
            }
        }
    }
    (best * gate).clamp(0.0, 1.0)
}

/// Best correlation of any candidate sub-window (at the given centre/scale grid)
/// against the fixed global whole-set template orientations, for one rendered
/// `field` covering `[cx±half, cy±half]`. Factored out so the recursion search can
/// run it at several boundary-descent depths against the same global template.
#[allow(clippy::too_many_arguments)]
fn best_copy_match(
    fw: &[(f32, f32)], config: &Config, mi: u32,
    field: &[f32], res: u32, cx: f32, cy: f32, half: f32,
    base_ed: f32, global_orients: &[Vec<f32>],
    grid: usize, k: usize, win_res: u32, ps: usize, scales: &[f32],
) -> f32 {
    let (w, h) = (res as usize, res as usize);
    let mut centres = recursion_candidates(field, w, h, mi, grid, k);
    if let Some(p) = richest_boundary_point(field, w, h) { centres.push(p); }
    if centres.is_empty() { return 0.0; }

    let wf = (w - 1).max(1) as f32;
    let hf = (h - 1).max(1) as f32;
    let (x0, y0) = (cx - half, cy - half);

    let mut best = 0.0f32;
    for &(px, py) in &centres {
        let wcx = x0 + (px as f32 / wf) * (2.0 * half);
        let wcy = y0 + (py as f32 / hf) * (2.0 * half);
        for &scale in scales {
            let wh  = half / scale;
            let win = render_bounds(fw, config, win_res, win_res, mi,
                                    wcx - wh, wcx + wh, wcy - wh, wcy + wh);
            // A window that has smoothed out can't host a copy of a structured whole.
            if edge_density(&win, win_res as usize, win_res as usize) < base_ed * 0.15 { continue; }
            let wv = structure_vec(&win, win_res as usize, win_res as usize, ps);
            if wv.iter().all(|v| *v == 0.0) { continue; }
            for g in global_orients {
                let c = correlation(&wv, g);
                if c > best { best = c; }
            }
        }
    }
    best
}

// ── Wormhole: locate (not just detect) an embedded self-similar copy ───────

/// A found self-similar "wormhole": a smaller embedded copy of the
/// reference view's own structure, found via the same boundary-descent
/// template matching `fractal_recursion_score` uses to detect the
/// phenomenon — but tracking the WINNING location instead of only a
/// scalar score. `dx`/`dy` are offsets from the reference view's own
/// center (in fractal-plane units) — never absolute coordinates, for the
/// same reason `find_interesting_square`'s doc comment (viewer.rs) gives:
/// materializing an absolute position at deep zoom silently collapses
/// precision. Each descent level's own local offset is safely f64
/// (bounded by that level's own frame span, which shrinks geometrically
/// each level), so plain f64 addition across levels never loses anything
/// — only the final combination with the reference view's own (possibly
/// astronomically deep) DD center needs `Dd::from_f64(dx)`, done once by
/// the caller.
#[derive(Clone, Copy, Debug)]
pub struct WormholeMatch {
    pub dx: f64,
    pub dy: f64,
    pub zoom: f64,
    pub score: f32,
}

const WORMHOLE_RES: u32             = 96;
const WORMHOLE_PS: usize            = 24;
// GRID/K confirmed empirically to matter a lot: `recursion_candidates`'
// "island" heuristic (interior wrapped in boundary) was tuned against
// Mandelbrot-style cardioid shapes, and a real archive genome with a very
// different visual structure (concentric rings, no obvious cardioid
// "island") exposed it missing a strong, real match entirely — a dense
// brute-force scan of the same frame found it immediately (raw correlation
// 0.68 vs. nothing comparable in the sparser island-filtered set). Wider
// grid + more kept candidates costs real render time but was necessary for
// genomes whose structure doesn't look like the shape the heuristic was
// calibrated on.
const WORMHOLE_GRID: usize          = 9;
const WORMHOLE_K: usize             = 14;
const WORMHOLE_SCALES: [f64; 4]     = [4.0, 8.0, 16.0, 32.0];
const WORMHOLE_DESCENT_LEVELS: usize = 4;
const WORMHOLE_DESCENT_STEP: f64     = 7.0;
const WORMHOLE_INTRIC_LO: f32 = 0.010;
const WORMHOLE_INTRIC_HI: f32 = 0.030;
// Upper ceiling — confirmed empirically against a real archive sample:
// pixel-noise genomes (no organized structure, just per-pixel escape-time
// speckle) measure intricacy ~0.6, while genuinely structured matches
// (visually confirmed) measured 0.12-0.18. Without this, noise trivially
// self-correlates after contrast normalization (a random speckle field
// resembles another crop of itself just by having the same statistics)
// and was scoring as a false "match" as confidently as real structure —
// `fractal_recursion_score`'s intricacy gate only ramps UP to full credit
// and never back down, which never mattered there (it's a coarser
// presence/absence signal) but does here, where the score is trusted
// enough to literally jump to.
const WORMHOLE_INTRIC_CEIL_LO: f32 = 0.22;
const WORMHOLE_INTRIC_CEIL_HI: f32 = 0.40;
// A candidate whose zoom is only marginally deeper than the reference's own
// isn't a genuine embedded copy — it's a barely-shifted crop that shares
// most of its pixels with the reference outright, which trivially
// correlates near-perfectly without showing any real self-similar
// structure. Confirmed empirically: at a real archive genome's best-scoring
// (dx, dy), score peaked at 0.99 at ratio 1.0x-1.5x and fell monotonically
// with depth from there — refine_match's zoom line-search, seeded from a
// deeper raw candidate, walked straight back down into that shallow
// optimum every time, because nothing distinguished "trivially overlaps
// the reference" from "genuinely resembles the reference at a smaller
// scale". `zoom > ref_view.zoom` alone (the previous floor) permits this;
// requiring a minimum multiple does not.
const WORMHOLE_MIN_DEPTH_RATIO: f64 = 3.0;

fn wormhole_render(genome: &Genome, config: &Config, view: &crate::video_export::View) -> Vec<f32> {
    use crate::video_export::{needs_f64, render_escape_times};
    let use_f64 = needs_f64(view, WORMHOLE_RES);
    render_escape_times(genome, config, view, WORMHOLE_RES, WORMHOLE_RES,
                         config.rendering.max_iter, use_f64, true)
}

/// Largest connected component of "interior" pixels (never escaped, or
/// escaped only in the last 5% of the iteration budget), as a fraction of
/// the frame — 4-connected flood fill, iterative (a 96x96 frame can have a
/// component covering most of it; recursion would risk stack depth).
///
/// Deliberately the LARGEST component, not total interior area: a real
/// mini-Mandelbrot copy has ONE dominant body (like the reference's own
/// single cardioid+bulb); a region sitting among several same-scale
/// neighboring mini-copies can have comparable TOTAL interior area spread
/// across multiple similar-sized blobs, which total-area alone can't tell
/// apart from one genuine dominant body — confirmed empirically: a real
/// "3-4 clustered blobs" false-positive had 60% MORE total interior area
/// than a clean single-cardioid true positive, total-area comparison alone
/// would have rewarded it rather than penalizing it.
fn largest_interior_component_fraction(field: &[f32], w: usize, h: usize, max_iter: f32) -> f32 {
    if field.len() != w * h || w == 0 || h == 0 { return 0.0; }
    let thr = max_iter * 0.95;
    let mut visited = vec![false; field.len()];
    let mut best = 0usize;
    let mut stack: Vec<usize> = Vec::new();
    for start in 0..field.len() {
        if visited[start] || field[start] < thr { continue; }
        visited[start] = true;
        stack.push(start);
        let mut size = 0usize;
        while let Some(idx) = stack.pop() {
            size += 1;
            let (x, y) = (idx % w, idx / w);
            let neighbors = [
                (x > 0).then(|| idx - 1),
                (x + 1 < w).then(|| idx + 1),
                (y > 0).then(|| idx - w),
                (y + 1 < h).then(|| idx + w),
            ];
            for n in neighbors.into_iter().flatten() {
                if !visited[n] && field[n] >= thr {
                    visited[n] = true;
                    stack.push(n);
                }
            }
        }
        best = best.max(size);
    }
    best as f32 / field.len() as f32
}

/// Multiplicative penalty for a candidate whose largest connected interior
/// body is much SMALLER than the reference's own — confirmed empirically
/// to matter: raw structure-vec correlation alone ranked a region with
/// several comparable-sized clustered blobs ABOVE a visually clean,
/// obviously-correct single-body mini-copy, because dense multi-blob
/// filament texture can statistically resemble the reference's own
/// filament-heavy boundary even with no single coherent body behind it. A
/// genuine scaled copy of the same set should have ONE dominant body
/// broadly comparable in proportion to the reference's own; "much
/// smaller" is the specific, meaningful tell for "this is a cluster of
/// several unrelated small bodies that happens to correlate, not an
/// actual single copy." Deliberately asymmetric — NOT penalizing a LARGER
/// dominant body than the reference's own, since that's still consistent
/// with a genuine copy centered slightly differently (a more solid crop
/// of the same body), just not the failure mode observed.
fn interior_undershoot_penalty(cand_frac: f32, ref_frac: f32) -> f32 {
    if ref_frac <= 0.0 || cand_frac >= ref_frac { return 1.0; }
    (cand_frac / ref_frac).max(0.0).powf(0.3)
}

/// Like the old `best_copy_match`, but DD-safe (works at any zoom depth,
/// DAG or legacy) and returns EVERY non-degenerate candidate evaluated at
/// this level — offset from `level_view`'s own center, plus zoom and
/// score — instead of collapsing immediately to a single winner.
/// `fractal_recursion_score` only ever needed one number; a wormhole
/// search that's meant to offer several real options needs the whole
/// pool, gathered across every level of the descent, so the caller can
/// rank and refine from all of them at the end rather than being stuck
/// with whichever one single-mindedly won at each individual level (a
/// wide correlation peak can have a slightly-better-scoring near neighbor
/// that this level's "keep only the best" would have discarded).
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn collect_copy_matches(
    genome: &Genome, config: &Config,
    field: &[f32], res: u32, level_view: &crate::video_export::View,
    base_ed: f32, base_interior: f32, global_orients: &[Vec<f32>],
    grid: usize, k: usize, win_res: u32, ps: usize, scales: &[f64], dense: bool,
) -> Vec<(f64, f64, f64, f32)> {
    use crate::dd::Dd;
    let (w, h) = (res as usize, res as usize);
    let mut centres = if dense {
        dense_grid_candidates(w, h, grid)
    } else {
        recursion_candidates(field, w, h, config.rendering.max_iter, grid, k)
    };
    if let Some(p) = richest_boundary_point(field, w, h) { centres.push(p); }
    if centres.is_empty() { return Vec::new(); }

    let wf = (w - 1).max(1) as f64;
    let hf = (h - 1).max(1) as f64;
    let half = 2.0 / level_view.zoom;

    let mut out = Vec::new();
    for &(px, py) in &centres {
        // Offset from level_view's own center — NOT an absolute position,
        // same reasoning as find_interesting_square's candidate sweep.
        let local_dx = -half + (px as f64 / wf) * (2.0 * half);
        let local_dy = -half + (py as f64 / hf) * (2.0 * half);
        for &scale in scales {
            let cand_zoom = level_view.zoom * scale;
            let cand_cx = level_view.cx_dd() + Dd::from_f64(local_dx);
            let cand_cy = level_view.cy_dd() + Dd::from_f64(local_dy);
            let cand_view = crate::video_export::View {
                cx: cand_cx.hi, cx_lo: cand_cx.lo, cy: cand_cy.hi, cy_lo: cand_cy.lo,
                zoom: cand_zoom, aspect: 1.0,
            };
            let win = wormhole_render(genome, config, &cand_view);
            // A window that has smoothed out can't host a copy of a structured whole.
            if edge_density(&win, win_res as usize, win_res as usize) < base_ed * 0.15 { continue; }
            // A window that's degenerated into pixel noise (common at extreme
            // relative depth, where the escape dynamics haven't had enough
            // iterations yet to resolve real structure — see needs_dd's
            // sibling discussion) trivially self-correlates with anything
            // after contrast normalization, the same false-positive the
            // reference-level ceiling exists to catch (confirmed empirically:
            // a genuine deep candidate scored intricacy 0.12-0.18; a noisy one
            // scored 0.63) — apply the identical ceiling here, per-candidate.
            if field_intricacy(&win, win_res as usize, win_res as usize) > WORMHOLE_INTRIC_CEIL_HI { continue; }
            let wv = structure_vec(&win, win_res as usize, win_res as usize, ps);
            if wv.iter().all(|v| *v == 0.0) { continue; }
            let mut best_c = 0.0f32;
            for g in global_orients {
                let c = correlation(&wv, g);
                if c > best_c { best_c = c; }
            }
            let cand_interior = largest_interior_component_fraction(&win, win_res as usize, win_res as usize, config.rendering.max_iter as f32);
            best_c *= interior_undershoot_penalty(cand_interior, base_interior);
            if best_c > 0.0 { out.push((local_dx, local_dy, cand_zoom, best_c)); }
        }
    }
    out
}

/// Two matches (both expressed as offsets from the same reference) count
/// as "the same spot" if they're within 30% of the shallower one's own
/// frame width of each other AND within 2x zoom of each other — used to
/// stop a single wide correlation peak from being reported as several
/// different wormholes.
fn same_wormhole_neighborhood(a: (f64, f64, f64), b: (f64, f64, f64)) -> bool {
    let ref_zoom = a.2.min(b.2);
    let tol = (2.0 / ref_zoom) * 0.3;
    (a.0 - b.0).abs() < tol && (a.1 - b.1).abs() < tol && (a.2 / b.2).max(b.2 / a.2) < 2.0
}

/// Search for smaller embedded copies of `ref_view`'s own structure inside
/// it — the "wormhole" navigation aid. Reuses the exact template-matching
/// technique `fractal_recursion_score` already validated (contrast-
/// normalised structure correlation across all 8 dihedral orientations,
/// boundary-descent search for the scale at which a copy actually
/// resolves) but, unlike that function, tracks and returns WINNING
/// locations rather than only a quality score — and renders through
/// `video_export::render_escape_times` (DD/f64/f32-tiered, DAG+legacy+
/// warp+phoenix+julia aware) instead of the legacy-only `render_bounds`,
/// so it actually works on the DAG genomes this project evolves now.
/// (Confirmed gap: `fractal_recursion_score` silently scores 0.0 for
/// every DAG genome sampled from the real archive — `formula_weights()`
/// returns an all-zero vector whenever `terms` is empty, which it always
/// is for a DAG genome; `fractal_recursion` in every saved DAG .nn checked
/// is exactly 0.0. This function does not fix that score — it's a
/// separate, new capability — but had to route around the same gap to be
/// useful on real genomes at all.)
///
/// Returns up to `max_results` matches, best score first, deduplicated so
/// one wide correlation peak can't count as several different wormholes.
/// Empty if no confident match is found, or the reference view itself is
/// too simple/flat (or too noisy — see `WORMHOLE_INTRIC_CEIL_*`) to have a
/// template worth matching.
///
/// Candidates are gathered across the WHOLE boundary descent (every level,
/// not just each level's single local winner), because a level's runner-up
/// can end up scoring higher than its winner once BOTH are actually
/// refined — committing early to only the pre-refinement winner (the
/// original single-result design) silently discarded those. The extra
/// coverage costs real time (refinement — the expensive part — now runs on
/// several raw candidates instead of one), so callers that only need the
/// single best match should still prefer `wormhole_search`.
pub fn wormhole_search_many(
    genome: &Genome, config: &Config, ref_view: &crate::video_export::View, max_results: usize,
) -> Vec<WormholeMatch> {
    use crate::dd::Dd;
    use crate::video_export::View;
    let (bw, bh) = (WORMHOLE_RES as usize, WORMHOLE_RES as usize);
    if max_results == 0 { return Vec::new(); }

    let base = wormhole_render(genome, config, ref_view);
    let base_ed = edge_density(&base, bw, bh);
    if base_ed < 0.01 { return Vec::new(); } // reference itself is essentially featureless
    let base_interior = largest_interior_component_fraction(&base, bw, bh, config.rendering.max_iter as f32);

    // Intricacy gate: reject smooth/monotone fields whose windows would
    // trivially self-correlate (same rationale as fractal_recursion_score)
    // — AND reject pixel-noise fields at the other extreme, which do the
    // same trivial self-correlation trick from the opposite direction (see
    // WORMHOLE_INTRIC_CEIL_LO/HI's doc comment).
    let intric = field_intricacy(&base, bw, bh);
    let ramp_up   = ((intric - WORMHOLE_INTRIC_LO) / (WORMHOLE_INTRIC_HI - WORMHOLE_INTRIC_LO)).clamp(0.0, 1.0);
    let ramp_down = 1.0 - ((intric - WORMHOLE_INTRIC_CEIL_LO) / (WORMHOLE_INTRIC_CEIL_HI - WORMHOLE_INTRIC_CEIL_LO)).clamp(0.0, 1.0);
    let gate = ramp_up * ramp_down;
    if gate <= 0.0 { return Vec::new(); }

    let global = structure_vec(&base, bw, bh, WORMHOLE_PS);
    if global.iter().all(|v| *v == 0.0) { return Vec::new(); } // flat → no template to match
    let global_orients = dihedral_variants(&global, WORMHOLE_PS);

    // Boundary descent: search at ref_view, then follow the richest
    // boundary point down a few zoom levels and keep searching — a copy
    // small enough to be sub-pixel at the reference scale only resolves
    // once rendered close enough to it. `total_dx`/`total_dy` accumulate as
    // plain f64 (safe: each level's own contribution shrinks geometrically
    // by WORMHOLE_DESCENT_STEP, so the running sum never needs more
    // precision than f64 already has — only the ONE combination with
    // ref_view's own DD center, done per render, needs Dd). `pool`
    // collects EVERY non-degenerate candidate seen at every level.
    let mut total_dx = 0.0f64;
    let mut total_dy = 0.0f64;
    let mut level_zoom = ref_view.zoom;
    let mut pool: Vec<(f64, f64, f64, f32)> = Vec::new();

    for level in 0..WORMHOLE_DESCENT_LEVELS {
        let level_cx = ref_view.cx_dd() + Dd::from_f64(total_dx);
        let level_cy = ref_view.cy_dd() + Dd::from_f64(total_dy);
        let level_view = View {
            cx: level_cx.hi, cx_lo: level_cx.lo, cy: level_cy.hi, cy_lo: level_cy.lo,
            zoom: level_zoom, aspect: 1.0,
        };
        let field = if level == 0 { base.clone() } else { wormhole_render(genome, config, &level_view) };

        for (ldx, ldy, lzoom, score) in collect_copy_matches(
            genome, config, &field, WORMHOLE_RES, &level_view, base_ed, base_interior, &global_orients,
            WORMHOLE_GRID, WORMHOLE_K, WORMHOLE_RES, WORMHOLE_PS, &WORMHOLE_SCALES, level == 0,
        ) {
            pool.push((total_dx + ldx, total_dy + ldy, lzoom, score));
        }

        if level + 1 < WORMHOLE_DESCENT_LEVELS {
            match richest_boundary_point(&field, bw, bh) {
                Some((px, py)) => {
                    let half = 2.0 / level_zoom;
                    let wf = (bw - 1).max(1) as f64;
                    let hf = (bh - 1).max(1) as f64;
                    total_dx += -half + (px as f64 / wf) * (2.0 * half);
                    total_dy += -half + (py as f64 / hf) * (2.0 * half);
                    level_zoom *= WORMHOLE_DESCENT_STEP;
                }
                None => break, // boundary smoothed out → nothing deeper to find
            }
        }
    }
    if pool.is_empty() { return Vec::new(); }

    // Non-max suppression on the RAW pool: keep the best-scoring
    // representative of each distinct neighborhood, generously more than
    // `max_results` (refinement can and does reorder the ranking — see
    // this function's doc comment — so cut early candidates loosely, not
    // down to the exact final count).
    pool.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
    let raw_budget = (max_results * 3).max(3);
    let mut raw_kept: Vec<(f64, f64, f64, f32)> = Vec::new();
    for &cand in &pool {
        if raw_kept.len() >= raw_budget { break; }
        if raw_kept.iter().any(|&k| same_wormhole_neighborhood((cand.0, cand.1, cand.2), (k.0, k.1, k.2))) { continue; }
        raw_kept.push(cand);
    }

    // Local refinement: the boundary-descent grid above routinely lands
    // near a real embedded copy but not precisely ON it — confirmed
    // visually against the classic Mandelbrot, where the raw descent
    // result was a genuine but weak (~0.13) partial overlap, not a clean
    // match. Coordinate-descent-style hill-climbing (shrinking position
    // grid, alternated with a zoom line-search) locks onto the actual
    // center once descent has found the right neighborhood.
    let mut refined: Vec<WormholeMatch> = raw_kept.into_iter().map(|(dx, dy, zoom, score)| {
        let (rdx, rdy, rzoom, rscore) = refine_match(
            genome, config, ref_view, &global_orients, base_ed, base_interior, dx, dy, zoom, WORMHOLE_PS, WORMHOLE_RES,
        );
        let (dx, dy, zoom, score) = if rscore > score { (rdx, rdy, rzoom, rscore) } else { (dx, dy, zoom, score) };
        WormholeMatch { dx, dy, zoom, score: score * gate }
    }).collect();
    refined.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    // Dedupe again on the REFINED positions: hill-climbing from two
    // distinct raw starts can converge to the same actual local optimum.
    let mut out: Vec<WormholeMatch> = Vec::new();
    for m in refined {
        if out.len() >= max_results { break; }
        if out.iter().any(|k| same_wormhole_neighborhood((m.dx, m.dy, m.zoom), (k.dx, k.dy, k.zoom))) { continue; }
        out.push(m);
    }
    out
}

/// Single best match — see `wormhole_search_many`'s doc comment. Prefer
/// this over the multi-result form when only one jump target is needed
/// (interactive use, or the batch scanner): it still evaluates and refines
/// a handful of raw candidates internally (for the same "the pre-
/// refinement winner isn't always the post-refinement winner" reason), so
/// it's more accurate than the original single-track version, just without
/// the cost of ranking a wide result set.
pub fn wormhole_search(
    genome: &Genome, config: &Config, ref_view: &crate::video_export::View,
) -> Option<WormholeMatch> {
    wormhole_search_many(genome, config, ref_view, 1).into_iter().next()
}

/// Coordinate-descent refinement around a candidate match: alternates a
/// shrinking position grid with a zoom line-search, re-rendering and
/// re-correlating at each step. See `wormhole_search`'s doc comment for why
/// this exists — the coarse boundary-descent grid alone finds the right
/// neighborhood but not a precise lock.
#[allow(clippy::too_many_arguments)]
fn refine_match(
    genome: &Genome, config: &Config, ref_view: &crate::video_export::View,
    global_orients: &[Vec<f32>], base_ed: f32, base_interior: f32,
    start_dx: f64, start_dy: f64, start_zoom: f64,
    ps: usize, win_res: u32,
) -> (f64, f64, f64, f32) {
    use crate::dd::Dd;

    let try_at = |dx: f64, dy: f64, zoom: f64| -> Option<f32> {
        // A wormhole is by definition a SMALLER embedded copy — the zoom
        // line-search has no other floor tying it to the reference's own
        // depth, so without this it can (confirmed: a real chain-building
        // test caught this) drift to a zoom at or below ref_view's own,
        // which isn't a valid match at all, just a same-or-wider crop. A
        // bare `> ref_view.zoom` isn't enough on its own — see
        // WORMHOLE_MIN_DEPTH_RATIO's doc comment.
        if zoom <= ref_view.zoom * WORMHOLE_MIN_DEPTH_RATIO || !zoom.is_finite() { return None; }
        let cx = ref_view.cx_dd() + Dd::from_f64(dx);
        let cy = ref_view.cy_dd() + Dd::from_f64(dy);
        let view = crate::video_export::View { cx: cx.hi, cx_lo: cx.lo, cy: cy.hi, cy_lo: cy.lo, zoom, aspect: 1.0 };
        let win = wormhole_render(genome, config, &view);
        if edge_density(&win, win_res as usize, win_res as usize) < base_ed * 0.15 { return None; }
        if field_intricacy(&win, win_res as usize, win_res as usize) > WORMHOLE_INTRIC_CEIL_HI { return None; }
        let wv = structure_vec(&win, win_res as usize, win_res as usize, ps);
        if wv.iter().all(|v| *v == 0.0) { return None; }
        let best_c = global_orients.iter().map(|g| correlation(&wv, g)).fold(None, |acc: Option<f32>, c| {
            Some(acc.map_or(c, |a| a.max(c)))
        })?;
        let cand_interior = largest_interior_component_fraction(&win, win_res as usize, win_res as usize, config.rendering.max_iter as f32);
        Some(best_c * interior_undershoot_penalty(cand_interior, base_interior))
    };

    // Coordinate-descent hill-climb from one (dx, dy, zoom) seed. Factored
    // into a closure so it can be re-run from a wide zoom anchor below —
    // its step size shrinks geometrically every round (0.6, 0.3, 0.15, ...),
    // so cumulatively it can only ever reach roughly 2.6x away from its
    // start zoom, never a jump of the size a descent level or scale anchor
    // routinely represents.
    let hill_climb = |start: (f64, f64, f64)| -> (f64, f64, f64, f32) {
        let (mut dx, mut dy, mut zoom) = start;
        let mut best_score = try_at(dx, dy, zoom).unwrap_or(0.0);

        const ROUNDS: usize = 5;
        let mut pos_radius = 2.0 / zoom; // one full window's half-width at the starting zoom
        let mut zoom_span = 0.6f64;      // ± fraction of current zoom explored this round

        for _ in 0..ROUNDS {
            let mut round_best = (dx, dy, zoom, best_score);
            for &dzm in &[1.0 - zoom_span, 1.0, 1.0 + zoom_span] {
                let cand_zoom = zoom * dzm;
                for iy in -1..=1i32 {
                    for ix in -1..=1i32 {
                        let cdx = dx + ix as f64 * pos_radius;
                        let cdy = dy + iy as f64 * pos_radius;
                        if let Some(score) = try_at(cdx, cdy, cand_zoom) {
                            if score > round_best.3 { round_best = (cdx, cdy, cand_zoom, score); }
                        }
                    }
                }
            }
            (dx, dy, zoom, best_score) = round_best;
            pos_radius *= 0.35;
            zoom_span *= 0.5;
        }
        (dx, dy, zoom, best_score)
    };

    let (mut dx, mut dy, mut zoom, mut best_score) = hill_climb((start_dx, start_dy, start_zoom));

    // A coarse candidate grid routinely lands close enough in (dx, dy) for
    // the hill-climb above to lock on, but not close enough in ZOOM — a
    // shallow, nearly-unzoomed crop at the (now correct) position can be a
    // local optimum in its own right (two crops of similar-looking texture
    // trivially resemble each other), with a worse-scoring valley between
    // it and the true, much deeper embedded copy. Confirmed on a real
    // archive genome: hill-climbing from a raw candidate correctly found
    // the right (dx, dy) but then walked DOWN to a ~1x non-match instead of
    // climbing to the true ~25x copy, because that climb needs to cross the
    // valley, which monotonic hill-climbing can't do. Re-probe the SAME
    // (now-trusted) position at a wide, discrete spread of zoom multiples —
    // far outside the hill-climb's own reach — and re-refine from whichever
    // wins; cheap (a handful of renders) unless it actually finds something
    // better, in which case a second hill-climb (the expensive part) is
    // justified.
    const ZOOM_ANCHOR_MULTS: [f64; 7] = [0.25, 1.0, 4.0, 16.0, 64.0, 256.0, 1024.0];
    let mut best_anchor: Option<(f64, f32)> = None;
    for &mult in &ZOOM_ANCHOR_MULTS {
        let cand_zoom = start_zoom * mult;
        if let Some(score) = try_at(dx, dy, cand_zoom) {
            if best_anchor.is_none_or(|(_, s)| score > s) { best_anchor = Some((cand_zoom, score)); }
        }
    }
    // Always re-climb from the best anchor, even when its FIXED-position
    // score doesn't yet beat the current champion: position precision needs
    // get tighter as zoom deepens (an absolute dx/dy error that's
    // negligible in a shallow window can be a large fraction of a much
    // deeper one), so a genuinely better deep lock can under-score at the
    // shallow climb's slightly-off position and only reveal itself once
    // hill-climbing gets to correct position AT that depth. Confirmed
    // necessary on the same real genome: gating the re-climb on "anchor
    // already beats best" left every anchor rejected (all under-scored at
    // the shallow climb's position) and the shallow false optimum was
    // never displaced.
    if let Some((anchor_zoom, _)) = best_anchor {
        let (rdx, rdy, rzoom, rscore) = hill_climb((dx, dy, anchor_zoom));
        if rscore > best_score { (dx, dy, zoom, best_score) = (rdx, rdy, rzoom, rscore); }
    }
    (dx, dy, zoom, best_score)
}

/// Batch-evaluate all genomes in ONE GPU dispatch with per-genome view bounds.
/// Returns (raw_png_entropy, multiscale_structured_entropy, angle_structure,
/// behavioral_descriptor) per genome — see evaluate_fitness_full for the
/// meaning of each. angle_structure is 0.0 for every genome unless
/// angle_structure_weight > 0 (skips computing the angle buffer entirely —
/// see render_gpu::render_batch_dag_angle's "free when disabled" design) and
/// is always 0.0 for legacy (non-DAG) genomes.
#[cfg(feature = "wgpu-backend")]
pub fn evaluate_fitness_batch(
    genomes: &[crate::genome::Genome],
    config:  &Config,
) -> Vec<(f32, f32, f32, Vec<f32>)> {
    let ew  = config.optimization.eval_width;
    let eh  = config.optimization.eval_height;
    let emi = config.optimization.eval_max_iter;
    let bsq = config.rendering.bailout * config.rendering.bailout;
    let want_angle = config.optimization.angle_structure_weight != 0.0;

    let views: Vec<(f32,f32,f32,f32)> = genomes.iter()
        .map(|g| { let (a,b,c,d) = g.view_bounds(); (a,b,c,d) })
        .collect();

    // Dispatch by formula system. A batch is uniform in practice (whole
    // population is one system); a mixed batch falls back to per-genome CPU.
    let all_dag = !genomes.is_empty() && genomes.iter().all(|g| g.uses_program());
    let any_dag = genomes.iter().any(|g| g.uses_program());
    let (escape_batch, angle_batch): (Vec<Vec<f32>>, Vec<Vec<f32>>) = if all_dag {
        let items: Vec<render_gpu::DagItem> = genomes.iter().map(render_gpu::dag_item).collect();
        if want_angle {
            render_gpu::render_batch_dag_angle(&items, &views, ew, eh, emi, true)
        } else {
            (render_gpu::render_batch_dag(&items, &views, ew, eh, emi),
             genomes.iter().map(|_| Vec::new()).collect())
        }
    } else if any_dag {
        genomes.iter().map(|g| {
            if want_angle && g.uses_program() {
                dag_render_with_angle(g, config, ew, eh, emi)
            } else {
                (render_cpu_iter(g, config, ew, eh, emi), Vec::new())
            }
        }).unzip()
    } else {
        let fw_vecs: Vec<Vec<(f32,f32)>> = genomes.iter().map(|g| g.formula_weights()).collect();
        let fw_refs: Vec<&[(f32,f32)]> = fw_vecs.iter().map(|v| v.as_slice()).collect();
        (render_gpu::render_batch(&fw_refs, &views, ew, eh, emi, bsq),
         genomes.iter().map(|_| Vec::new()).collect())
    };

    // Parallelize PNG encoding across all CPU cores while GPU is idle post-dispatch
    escape_batch.into_par_iter().zip(angle_batch.into_par_iter()).map(|(et, ang)| {
        let raw_png   = crate::fitness::png_compression_entropy(&et, ew, eh, emi, &config.rendering.colormap);
        let structured = crate::fitness::multiscale_entropy(&et, ew, eh, emi, &config.rendering.colormap);
        // Empty `ang` (feature off, or a legacy genome in a mixed batch) → 0.0,
        // matching evaluate_fitness_full's convention (angle_structure_score
        // returns 0.0 for len<4).
        let angle_score = crate::fitness::angle_structure_score(&ang, ew as usize);
        let desc = crate::fitness::behavior_descriptor(&et, emi);
        (raw_png, structured, angle_score, desc)
    }).collect()
}


#[cfg(test)]
mod local_metric_tests {
    use super::*;

    /// A field that's uniform EXCEPT for a small checkerboard "hotspot" in
    /// one corner — the exact failure mode `local_edge_density`/
    /// `local_intricacy` exist to fix (Carl, 2026-08-11): a small,
    /// genuinely interesting feature (like a crisp circular boundary)
    /// diluted into near-invisibility by a whole-patch average.
    fn field_with_small_hotspot(size: usize, hotspot: usize) -> Vec<f32> {
        let mut field = vec![0.5f32; size * size];
        for y in 0..hotspot {
            for x in 0..hotspot {
                if (x + y) % 2 == 0 {
                    field[y * size + x] = 1.0;
                }
            }
        }
        field
    }

    #[test]
    fn local_edge_density_finds_a_small_hotspot_the_whole_patch_average_dilutes_away() {
        let size = 64;
        let field = field_with_small_hotspot(size, 8); // hotspot is <2% of the field's area
        let whole = edge_density(&field, size, size);
        let local = local_edge_density(&field, size, size, 4);
        assert!(whole < 0.05, "whole-patch average should read as near-flat, got {whole}");
        assert!(local > whole * 5.0, "the tile containing the hotspot should clearly register it: local={local} whole={whole}");
    }

    #[test]
    fn local_intricacy_finds_a_small_hotspot_the_whole_patch_average_dilutes_away() {
        let size = 64;
        let field = field_with_small_hotspot(size, 8);
        let whole = field_intricacy(&field, size, size);
        let local = local_intricacy(&field, size, size, 4);
        assert!(local > whole * 3.0, "local should score well above the diluted whole-patch value: local={local} whole={whole}");
    }

    #[test]
    fn tile_max_falls_back_to_whole_patch_for_a_field_too_small_to_tile() {
        let field = vec![0.5f32; 4 * 4];
        // tiles=4 on a 4x4 field would mean 1x1 sub-tiles — too small to
        // mean anything, so this should fall back to the whole-patch call
        // rather than silently degenerate.
        assert_eq!(local_edge_density(&field, 4, 4, 4), edge_density(&field, 4, 4));
    }

    #[test]
    fn local_metrics_never_score_below_their_whole_patch_counterpart() {
        // By construction (max over sub-tiles, one of which IS the whole
        // patch's own statistics folded smaller) local should never be
        // WORSE than whole — a uniformly-busy field (no hidden hotspot)
        // should score about the same either way, not lower locally.
        let size = 64;
        let mut field = vec![0.0f32; size * size];
        for y in 0..size {
            for x in 0..size {
                if (x + y) % 2 == 0 { field[y * size + x] = 1.0; }
            }
        }
        let whole_edge = edge_density(&field, size, size);
        let local_edge = local_edge_density(&field, size, size, 4);
        assert!(local_edge >= whole_edge * 0.9, "local={local_edge} whole={whole_edge}");
    }
}

#[cfg(test)]
mod dag_f64_tests {
    use crate::formula::{op, OpNode};

    // f64 deep-zoom DAG iteration must closely track the f32 one (same logic,
    // higher precision) — compare on a grid, allowing boundary-chaos slack.
    #[test]
    fn dag_escape_f32_vs_f64_track() {
        // (z² + c) with phoenix + julia
        let prog = vec![
            OpNode { op: op::Z,   a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::C,   a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::SQR, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::ADD, a: 2, b: 1, kre: 0.0, kim: 0.0 },
        ];
        let (julia, jc, ph, bsq) = (true, (-0.4, 0.6), (0.1, -0.05), 16.0);
        let (n, mut agree) = (40usize, 0usize);
        for gy in 0..n { for gx in 0..n {
            let px = -1.8 + 3.6 * gx as f64 / n as f64;
            let py = -1.8 + 3.6 * gy as f64 / n as f64;
            let (a, _) = super::dag_escape_pixel(&prog, &[], julia, (jc.0 as f32, jc.1 as f32),
                (ph.0 as f32, ph.1 as f32), bsq as f32, px as f32, py as f32, 128);
            let (b, _) = super::dag_escape_pixel_f64(&prog, &[], julia, jc, ph, bsq, px, py, 128);
            if (a - b).abs() < 1.0 { agree += 1; }
        }}
        let frac = agree as f32 / (n * n) as f32;
        assert!(frac > 0.95, "f32/f64 DAG escape disagree on {:.1}% of pixels", (1.0 - frac) * 100.0);
    }
}

#[cfg(test)]
mod known_formula_tests {
    use super::*;
    use crate::formula::{op, OpNode};
    use crate::genome::ProgramBuilder;

    fn dag_genome(program: Vec<OpNode>) -> Genome {
        Genome { program, bailout_radius: 4.0, view_zoom: 1.0, ..Default::default() }
    }

    #[test]
    fn exact_mandelbrot_matches_mandelbrot() {
        let mut b = ProgramBuilder::new();
        let z  = b.push(op::Z, 0, 0, 0.0, 0.0).unwrap();
        let c  = b.push(op::C, 0, 0, 0.0, 0.0).unwrap();
        let z2 = b.push(op::SQR, z, 0, 0.0, 0.0).unwrap();
        b.push(op::ADD, z2, c, 0.0, 0.0).unwrap();
        let genome = dag_genome(b.into_nodes());

        let m = known_formula_match(&genome);
        assert!(m.is_some(), "exact Mandelbrot program found no match at all");
        let (name, score) = m.unwrap();
        assert_eq!(name, "Mandelbrot");
        assert!(score > 0.95, "exact self-match scored too low: {score}");
    }

    #[test]
    fn exact_tricorn_matches_tricorn_not_mandelbrot() {
        let mut b = ProgramBuilder::new();
        let z   = b.push(op::Z, 0, 0, 0.0, 0.0).unwrap();
        let c   = b.push(op::C, 0, 0, 0.0, 0.0).unwrap();
        let cj  = b.push(op::CONJ, z, 0, 0.0, 0.0).unwrap();
        let cj2 = b.push(op::SQR, cj, 0, 0.0, 0.0).unwrap();
        b.push(op::ADD, cj2, c, 0.0, 0.0).unwrap();
        let genome = dag_genome(b.into_nodes());

        let (name, _) = known_formula_match(&genome)
            .expect("exact Tricorn program found no match at all");
        assert_eq!(name, "Tricorn (Mandelbar)",
            "Tricorn matched something else — metric isn't discriminating");
    }

    #[test]
    fn near_miss_coefficient_still_matches_mandelbrot() {
        // 0.97·z² + c instead of exactly z² + c — the case symbolic equality
        // would reject outright but behavioral comparison should still catch.
        let mut b = ProgramBuilder::new();
        let z    = b.push(op::Z, 0, 0, 0.0, 0.0).unwrap();
        let c    = b.push(op::C, 0, 0, 0.0, 0.0).unwrap();
        let z2   = b.push(op::SQR, z, 0, 0.0, 0.0).unwrap();
        let k    = b.push(op::CONST, 0, 0, 0.97, 0.0).unwrap();
        let z2s  = b.push(op::MUL, k, z2, 0.0, 0.0).unwrap();
        b.push(op::ADD, z2s, c, 0.0, 0.0).unwrap();
        let genome = dag_genome(b.into_nodes());

        let (name, score) = known_formula_match(&genome)
            .expect("near-miss Mandelbrot (0.97x) found no match — behavioral robustness failed");
        assert_eq!(name, "Mandelbrot");
        assert!(score >= KNOWN_FORMULA_THRESHOLD);
    }

    #[test]
    fn degenerate_program_has_no_match() {
        // Root is a bare constant leaf — a flat, structureless field.
        let mut b = ProgramBuilder::new();
        b.push(op::CONST, 0, 0, 0.5, 0.5).unwrap();
        let genome = dag_genome(b.into_nodes());
        assert!(known_formula_match(&genome).is_none());
    }

    #[test]
    fn legacy_genome_has_no_match() {
        let genome = Genome { terms: vec![
            crate::genome::FormulaTerm { basis: 0, re: 1.0, im: 0.0 },
            crate::genome::FormulaTerm { basis: 7, re: 1.0, im: 0.0 },
        ], view_zoom: 1.0, ..Default::default() };
        assert!(!genome.uses_program());
        assert!(known_formula_match(&genome).is_none());
    }
}

#[cfg(test)]
mod wormhole_tests {
    use super::*;
    use crate::config::{Config, DedupConfig, MassExtinctionConfig, OptimizationConfig, OutputConfig, RenderingConfig};
    use crate::formula::{op, OpNode};
    use crate::genome::ProgramBuilder;
    use crate::video_export::View;

    fn dag_genome(program: Vec<OpNode>) -> Genome {
        Genome { program, bailout_radius: 4.0, view_zoom: 1.0, ..Default::default() }
    }

    fn mandelbrot_genome() -> Genome {
        let mut b = ProgramBuilder::new();
        let z  = b.push(op::Z, 0, 0, 0.0, 0.0).unwrap();
        let c  = b.push(op::C, 0, 0, 0.0, 0.0).unwrap();
        let z2 = b.push(op::SQR, z, 0, 0.0, 0.0).unwrap();
        b.push(op::ADD, z2, c, 0.0, 0.0).unwrap();
        dag_genome(b.into_nodes())
    }

    fn default_config() -> Config {
        Config {
            dedup: DedupConfig::default(), mass_extinction: MassExtinctionConfig::default(),
            rendering: RenderingConfig {
                default_width: 800, default_height: 800, max_iter: 500, bailout: 4.0,
                colormap: "turbo".into(), view_x_min: -2.0, view_x_max: 2.0, view_y_min: -2.0, view_y_max: 2.0,
            },
            optimization: OptimizationConfig {
                population_size: 40, elitism_count: 6, mutation_rate: 0.20, mutation_scale: 0.08,
                eval_width: 64, eval_height: 64, eval_max_iter: 128, restart_after_gens: 30, novelty_weight: 0.45,
                novelty_k: 5, archive_size: 150, self_replication_weight: 0.35, fractal_recursion_weight: 0.35,
                recursion_pred_weight: 0.60, formula_diversity_weight: 0.30, clip_pred_weight: 0.50,
                formula_system: "dag".to_string(), max_nodes: 14, max_depth: 5, ood_weight: 0.0, pref_weight: 0.4,
                seed_pref_weight: 3.0, musiq_weight: 0.25, pref_elite_count: 4, archive_random_ratio: 0.30,
                duplicate_penalty_weight: 0.50, archive_seeding_enabled: false, angle_structure_weight: 0.0,
                img_novelty_weight: 0.0,
            },
            output: OutputConfig {
                save_dir: "./fractals".into(), population_dir: "./populations".into(),
                min_entropy_prefilter: 0.42, max_entropy_prefilter: 0.65, min_clip_score: 0.512, min_laion_score: 5.30,
                min_beauty: 0.35, min_save_distance: 0.04, min_ensemble: 4.6, min_musiq: 30.0, min_pref: 0.45,
            },
        }
    }

    #[test]
    fn classic_mandelbrot_has_a_wormhole() {
        // The classic Mandelbrot set is famous for embedded baby-Mandelbrots
        // — if wormhole_search can't find one here, it can't find one
        // anywhere. Regression for the DAG-rendering gap this function was
        // specifically built to route around (fractal_recursion_score,
        // built on the legacy-only formula_weights()/render_bounds() path,
        // silently scores 0.0 for every DAG genome — confirmed against 40
        // random real saved genomes, all `terms: []`).
        let genome = mandelbrot_genome();
        let config = default_config();
        let ref_view = View::new_square(-0.5, 0.0, 1.0);
        let m = wormhole_search(&genome, &config, &ref_view);
        assert!(m.is_some(), "classic Mandelbrot must have a findable embedded copy");
        let m = m.unwrap();
        assert!(m.score > 0.0 && m.score <= 1.0, "score {} out of range", m.score);
        assert!(m.zoom > ref_view.zoom, "a wormhole match must be a SMALLER (deeper) copy, not wider");
        assert!(m.dx.is_finite() && m.dy.is_finite() && m.zoom.is_finite());
    }

    #[test]
    fn wormhole_match_always_respects_minimum_depth_ratio() {
        // A match whose zoom is only marginally deeper than the reference
        // is a barely-shifted crop that shares most of its pixels with the
        // reference outright — it trivially scores well without showing any
        // real embedded self-similarity. Confirmed on a real archive genome
        // this session: the search reliably converged on a ~1.04x "match"
        // scoring 0.99 at a position whose legitimately-deeper candidates,
        // once the intricacy ceiling was correctly enforced, scored far
        // lower — the high score belonged entirely to the trivial overlap,
        // not to any genuine structure. `zoom > ref_view.zoom` alone (the
        // previous floor) permits this; WORMHOLE_MIN_DEPTH_RATIO does not.
        let genome = mandelbrot_genome();
        let config = default_config();
        let ref_view = View::new_square(-0.5, 0.0, 1.0);
        let m = wormhole_search(&genome, &config, &ref_view).expect("classic Mandelbrot must have a match");
        assert!(m.zoom >= ref_view.zoom * WORMHOLE_MIN_DEPTH_RATIO,
            "match zoom {} is not at least {}x deeper than ref zoom {}", m.zoom, WORMHOLE_MIN_DEPTH_RATIO, ref_view.zoom);
    }

    #[test]
    fn pure_noise_field_does_not_score_as_a_match() {
        // A formula whose escape times are pixel-noise (no organized
        // structure) must not exploit contrast-normalization's tendency to
        // make any two same-statistics noise crops "correlate" — this is
        // the empirically-discovered false-positive the intricacy CEILING
        // (as opposed to the floor fractal_recursion_score already had)
        // exists to catch. RECIP of Z with a near-zero epsilon is exactly
        // the kind of numerically explosive, per-pixel-chaotic formula
        // real archive genomes triggering this looked like.
        let mut b = ProgramBuilder::new();
        let z = b.push(op::Z, 0, 0, 0.0, 0.0).unwrap();
        b.push(op::RECIP, z, 0, 0.0, 0.0).unwrap();
        let genome = dag_genome(b.into_nodes());
        let config = default_config();
        // A deep, small view where 1/z is maximally sensitive to per-pixel
        // position — the regime that produced real noise-like fields.
        let ref_view = View::new_square(0.0001, 0.0001, 5000.0);
        let m = wormhole_search(&genome, &config, &ref_view);
        assert!(m.is_none() || m.unwrap().score < 0.3,
            "a noise field scored too high to be a false positive: {m:?}");
    }

    #[test]
    fn flat_field_has_no_wormhole() {
        // z_next = 0 (a CONST(0,0) program): every pixel is identical —
        // no template worth matching, must return None outright rather
        // than reporting a spurious perfect self-match.
        let mut b = ProgramBuilder::new();
        b.push(op::CONST, 0, 0, 0.0, 0.0).unwrap();
        let genome = dag_genome(b.into_nodes());
        let config = default_config();
        let ref_view = View::new_square(0.0, 0.0, 1.0);
        assert!(wormhole_search(&genome, &config, &ref_view).is_none());
    }

    #[test]
    fn largest_component_beats_total_area_for_a_clustered_field() {
        // Regression for the specific false-positive this session found by
        // rendering and looking, not trusting the number: raw structure-vec
        // correlation ranked a region with several comparable-sized
        // clustered blobs ABOVE a visually clean, obviously-correct single-
        // body mini-copy — the clustered field even had MORE total interior
        // area than the clean one, so a naive total-area check would have
        // rewarded it too; only counting the LARGEST CONNECTED body
        // discriminates them. 10x10 field, max_iter=100 (>=95 counts as
        // interior): a single 3x3 solid block vs four separated 2x2 blocks
        // (more total interior pixels: 16 vs 9) — the single block must
        // still win on largest-component-fraction.
        const W: usize = 20;
        let mi = 100.0f32;
        let mut single_block = vec![0.0f32; W * W];
        for y in 5..13 { for x in 5..13 { single_block[y * W + x] = mi; } } // 8x8 = 64 px, one component

        let mut four_blocks = vec![0.0f32; W * W];
        for &(bx, by) in &[(0usize, 0usize), (0, 15), (15, 0), (15, 15)] {
            for y in by..by + 5 { for x in bx..bx + 5 { four_blocks[y * W + x] = mi; } } // 4×(5x5=25px) = 100 px total, separated
        }

        let single_frac = largest_interior_component_fraction(&single_block, W, W, mi);
        let clustered_frac = largest_interior_component_fraction(&four_blocks, W, W, mi);
        let total_area_single: f32 = single_block.iter().filter(|&&v| v >= mi * 0.95).count() as f32 / (W * W) as f32;
        let total_area_clustered: f32 = four_blocks.iter().filter(|&&v| v >= mi * 0.95).count() as f32 / (W * W) as f32;

        assert!(total_area_clustered > total_area_single,
            "test setup must give the clustered field MORE total area, to prove this isn't just measuring total area");
        assert!(single_frac > clustered_frac,
            "largest-component fraction must favor the one dominant body over several small ones: \
             single={single_frac:.4} clustered={clustered_frac:.4}");

        // And the penalty built on top of it must then rank a "clean, one
        // dominant body" candidate above a "several small bodies" one, even
        // when the clustered one's raw correlation score is higher.
        let ref_frac = single_frac; // pretend the reference looks like the clean field
        let penalized_clean     = 0.25 * interior_undershoot_penalty(single_frac, ref_frac);
        let penalized_clustered = 0.32 * interior_undershoot_penalty(clustered_frac, ref_frac);
        assert!(penalized_clean > penalized_clustered,
            "penalty didn't fix the ranking: clean={penalized_clean:.4} clustered={penalized_clustered:.4}");
    }

    #[test]
    fn interior_penalty_is_inert_when_candidate_has_more_interior_than_reference() {
        // Deliberately asymmetric — see interior_undershoot_penalty's doc
        // comment for why overshooting must not be punished.
        assert_eq!(interior_undershoot_penalty(0.5, 0.1), 1.0);
        assert_eq!(interior_undershoot_penalty(0.1, 0.1), 1.0);
    }
}
