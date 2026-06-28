use rayon::prelude::*;
use crate::config::Config;
use crate::genome::Genome;
use crate::formula::apply_formula;
#[cfg(feature = "wgpu-backend")]
use crate::render_gpu;

/// CPU fitness: (raw_png_entropy, multiscale_structured_entropy, behavioral_descriptor).
///
/// raw_png_entropy  — used for save-gate thresholding (min/max_entropy_prefilter).
/// multiscale_entropy — used for GA selection: geometric mean of fine (64px) and
///   coarse (16px average-pool) PNG entropy. Penalises granular noise because noise
///   averages to near-uniform at coarse scale; structured fractals stay complex at
///   all scales. Replace the old single-scale fitness for selection only.
pub fn evaluate_fitness_full(genome: &Genome, config: &Config) -> (f32, f32, Vec<f32>) {
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
    let descriptor = crate::fitness::behavior_descriptor(&escape_times, emi);
    (raw_png, structured, descriptor)
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

/// Escape-time for one pixel under the DAG iteration with Phase-3/4 dynamics:
/// optional coordinate warp, Julia vs Mandelbrot initialization, and phoenix
/// memory z_{n+1} = f(z,c) + p·z_{n-1}. Mirrors the WGSL main loop.
#[allow(clippy::too_many_arguments)]
pub fn dag_escape_pixel(
    prog: &[crate::formula::OpNode], warp: &[crate::formula::OpNode],
    julia: bool, jc: (f32, f32), phoenix: (f32, f32), bailout_sq: f32,
    px: f32, py: f32, max_iter: u32,
) -> f32 {
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
        if ms > bailout_sq { return ((it as f32 + 1.0) - (ms.log2() * 0.5).log2()).max(0.0); }
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
            dag_escape_pixel(prog, warp, julia, jc, phoenix, bsq, cx, cy, max_iter)
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
fn structure_vec(field: &[f32], w: usize, h: usize, ps: usize) -> Vec<f32> {
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

/// Pearson correlation of two equal-length z-scored vectors → [-1, 1].
fn correlation(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() { return 0.0; }
    let n = a.len() as f32;
    a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>() / n
}

/// Fraction of adjacent pixel pairs whose escape time jumps by a notable amount —
/// a scale-free measure of how much boundary/structure is present in a frame.
fn edge_density(field: &[f32], w: usize, h: usize) -> f32 {
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
fn dihedral_variants(v: &[f32], ps: usize) -> Vec<Vec<f32>> {
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

/// Batch-evaluate all genomes in ONE GPU dispatch with per-genome view bounds.
#[cfg(feature = "wgpu-backend")]
pub fn evaluate_fitness_batch(
    genomes: &[crate::genome::Genome],
    config:  &Config,
) -> Vec<(f32, f32, Vec<f32>)> {
    let ew  = config.optimization.eval_width;
    let eh  = config.optimization.eval_height;
    let emi = config.optimization.eval_max_iter;
    let bsq = config.rendering.bailout * config.rendering.bailout;

    let views: Vec<(f32,f32,f32,f32)> = genomes.iter()
        .map(|g| { let (a,b,c,d) = g.view_bounds(); (a,b,c,d) })
        .collect();

    // Dispatch by formula system. A batch is uniform in practice (whole
    // population is one system); a mixed batch falls back to per-genome CPU.
    let all_dag = !genomes.is_empty() && genomes.iter().all(|g| g.uses_program());
    let any_dag = genomes.iter().any(|g| g.uses_program());
    let escape_batch = if all_dag {
        let items: Vec<render_gpu::DagItem> = genomes.iter().map(render_gpu::dag_item).collect();
        render_gpu::render_batch_dag(&items, &views, ew, eh, emi)
    } else if any_dag {
        genomes.iter().map(|g| render_cpu_iter(g, config, ew, eh, emi)).collect()
    } else {
        let fw_vecs: Vec<Vec<(f32,f32)>> = genomes.iter().map(|g| g.formula_weights()).collect();
        let fw_refs: Vec<&[(f32,f32)]> = fw_vecs.iter().map(|v| v.as_slice()).collect();
        render_gpu::render_batch(&fw_refs, &views, ew, eh, emi, bsq)
    };

    // Parallelize PNG encoding across all CPU cores while GPU is idle post-dispatch
    escape_batch.into_par_iter().map(|et| {
        let raw_png   = crate::fitness::png_compression_entropy(&et, ew, eh, emi, &config.rendering.colormap);
        let structured = crate::fitness::multiscale_entropy(&et, ew, eh, emi, &config.rendering.colormap);
        let desc = crate::fitness::behavior_descriptor(&et, emi);
        (raw_png, structured, desc)
    }).collect()
}

