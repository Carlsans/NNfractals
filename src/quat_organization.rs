//! Whole-4D-object organization/complexity metrics — Carl's ask after
//! tonight's `box_dim`/`convexity` single-metric runs each collapsed
//! into a different degenerate optimum (pure noise, then blank spheres):
//! "does our metrics consider all 4 dimensions or only a 2D picture,"
//! and then "research how we could consider entropy/noise/organization
//! of a 4D object... as a whole. Interesting fractal always has some
//! form of organisation." See the published research memo ("Organized
//! Complexity") for the literature this implements: statistical
//! complexity (López-Ruiz/Mancini/Calbet, entropy × disequilibrium,
//! zero at both pure-order and pure-noise), ordinal-pattern/permutation
//! complexity (spatial-structure-aware, the direct fix for LMC's blind
//! spot to shuffled-but-same-histogram fields), a 2-point multifractal
//! spectrum width, compression-ratio complexity (Kolmogorov-complexity
//! proxy), and a finite-difference chaoticity signal.
//!
//! The deeper fix this module makes real: every metric in
//! `quat_dag_fitness.rs` starts from a RENDERED image — a ray-marched,
//! camera-projected, single-time-axis 2D array. `quat_dag_escape_de`
//! needs no camera at all: it evaluates a raw `(R,A,B,C)` point directly.
//! Sampling it across all four dimensions at once — never picking one
//! axis as "time" and fixing the other three, the render path's own
//! limitation — is both more complete and cheaper (no sphere-tracing,
//! just direct per-point evaluation, closer in cost to the classic flat
//! Mandelbrot pixel loop).
//!
//! Deliberate scope cuts, made once and documented here rather than
//! scattered: Sobol sequences are the literature's usual first choice
//! for low-discrepancy sampling, but a Halton sequence (prime bases
//! 2/3/5/7 for R/A/B/C) needs no precomputed direction-number tables and
//! is good enough at the sample counts used here (thousands, not
//! millions). The multifractal spectrum is reduced from a full q-sweep
//! to two points (D₀, D₂) at two box-count scales — the full sweep is
//! expensive in 4D (box count grows exponentially with dimension, per
//! the research memo's own citation) and a 2-point width still answers
//! the "does structure look different under different statistical
//! weightings" question, just coarsely. Compression uses a straightforward
//! (not bit-trick-optimized) per-bit Morton/Z-order interleave for
//! locality — simpler to get right than a true Hilbert curve, and
//! correctness matters more here than the last few percent of locality
//! quality. Chaoticity is a finite-difference sensitivity of the escape-
//! time FIELD itself (perturb one sample point, see how much its escape
//! time jumps) rather than harvesting `quat_dag_escape_de`'s internal
//! derivative accumulator — avoids touching that already-tested function
//! at all, and arguably measures the more directly relevant thing (how
//! sensitive is what a viewer would see, not the raw orbit dynamics).

use std::collections::HashMap;
use std::io::Write as _;

use crate::quat_dag::{quat_dag_escape_de, QuatDagFormula};
use crate::quat_fractal::TimeAxis;
use crate::quat_motion::Vec3;
use crate::quaternion::Quat;

/// One evaluated field sample: a raw `(R,A,B,C)` point (no camera, no
/// time-axis choice — every one of the 4 dimensions is just a spatial
/// coordinate here) plus its escape time and distance estimate.
struct FieldSample {
    point: Quat,
    escape_time: f32,
}

/// Halton sequence value for 1-based `index`, prime `base` — standard
/// radical-inverse (bit/digit-reversal) construction.
fn halton(mut index: usize, base: usize) -> f64 {
    let mut result = 0.0;
    let mut f = 1.0;
    while index > 0 {
        f /= base as f64;
        result += f * (index % base) as f64;
        index /= base;
    }
    result
}

/// One low-discrepancy 4D point, mapped from `[0,1)^4` into
/// `[-domain_radius, domain_radius]^4`. `index` is 0-based; internally
/// offset by 1 so the degenerate all-zeros Halton point (index 0) is
/// never sampled.
fn halton_point_4d(index: usize, domain_radius: f64) -> Quat {
    let i = index + 1;
    let map = |h: f64| (h * 2.0 - 1.0) * domain_radius;
    Quat::new(map(halton(i, 2)), map(halton(i, 3)), map(halton(i, 5)), map(halton(i, 7)))
}

fn sample_field(f: &QuatDagFormula, n: usize, domain_radius: f64, max_iter: u32, bailout_sq: f64) -> Vec<FieldSample> {
    (0..n)
        .map(|i| {
            let point = halton_point_4d(i, domain_radius);
            let (escape_time, _de) = quat_dag_escape_de(f, point, max_iter, bailout_sq);
            FieldSample { point, escape_time }
        })
        .collect()
}

fn shannon_entropy(probs: &[f32]) -> f32 {
    probs.iter().filter(|&&p| p > 0.0).map(|&p| -p * p.log2()).sum()
}

const ENTROPY_BINS: usize = 32;

/// Escape-time values bucketed into a normalized 32-bin histogram —
/// shared by `statistical_complexity` (entropy AND disequilibrium) and
/// `escape_time_entropy_norm` (entropy alone), so the two stay built from
/// literally the same binning.
fn escape_time_hist_probs(samples: &[FieldSample], max_iter: u32) -> [f32; ENTROPY_BINS] {
    let mut hist = [0u32; ENTROPY_BINS];
    for s in samples {
        let t = (s.escape_time / max_iter.max(1) as f32).clamp(0.0, 1.0);
        let b = ((t * (ENTROPY_BINS as f32 - 1.0)) as usize).min(ENTROPY_BINS - 1);
        hist[b] += 1;
    }
    let n = samples.len().max(1) as f32;
    let mut probs = [0.0f32; ENTROPY_BINS];
    for i in 0..ENTROPY_BINS {
        probs[i] = hist[i] as f32 / n;
    }
    probs
}

/// Plain (non-LMC) normalized Shannon entropy of the escape-time
/// histogram, in [0,1]. High = escape times spread broadly across the
/// range (a visually rich/detailed field); low = mostly one value (a
/// blank/flat field). A standalone positive signal — see
/// `organized_richness`'s doc comment for why this is deliberately NOT
/// multiplied by disequilibrium the way `statistical_complexity` is.
fn escape_time_entropy_norm(samples: &[FieldSample], max_iter: u32) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let probs = escape_time_hist_probs(samples, max_iter);
    shannon_entropy(&probs) / (ENTROPY_BINS as f32).log2()
}

/// LMC-style statistical complexity: `H · D` over a histogram of escape
/// times — entropy (disorder) times disequilibrium (distance from a
/// uniform histogram). Zero for a field that's all one value (D≈0, no
/// entropy) AND zero for a field whose escape times are already spread
/// uniformly across every bin (H high but D≈0 — indistinguishable from
/// noise by this measure alone, which is exactly LMC's documented blind
/// spot to spatial arrangement; `ordinal_complexity` below is the
/// spatial-structure-aware complement, not a replacement).
fn statistical_complexity(samples: &[FieldSample], max_iter: u32) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let probs = escape_time_hist_probs(samples, max_iter);
    let h_norm = shannon_entropy(&probs) / (ENTROPY_BINS as f32).log2();
    let uniform = 1.0 / ENTROPY_BINS as f32;
    let d: f32 = probs.iter().map(|&p| (p - uniform).powi(2)).sum::<f32>().sqrt();
    let d_max = ((1.0 - uniform).powi(2) + (ENTROPY_BINS as f32 - 1.0) * uniform.powi(2)).sqrt();
    let d_norm = if d_max > 1e-9 { (d / d_max).clamp(0.0, 1.0) } else { 0.0 };
    (h_norm * d_norm).clamp(0.0, 1.0)
}

/// Ordinal-pattern (permutation) complexity: at `n_clusters` Halton
/// centers, evaluate 5 points (the center plus one small step along each
/// of the 4 axes), record which of the 5! = 120 possible RANK ORDERS
/// that quintuple falls into, and apply the same entropy × disequilibrium
/// construction as `statistical_complexity` — but over PATTERNS instead
/// of raw values. This is spatially aware where the value-histogram
/// version isn't: noise visits ordinal patterns close to uniformly
/// (unpredictable relative to its own neighbors), real structure
/// concentrates onto a small subset of patterns (the local shape of a
/// gradient/edge/ridge has a characteristic signature).
fn ordinal_complexity(f: &QuatDagFormula, domain_radius: f64, max_iter: u32, bailout_sq: f64, n_clusters: usize) -> f32 {
    let step = domain_radius * 0.01;
    let mut pattern_counts: HashMap<[u8; 5], u32> = HashMap::new();
    for i in 0..n_clusters {
        let c = halton_point_4d(i, domain_radius);
        let offsets = [
            c,
            Quat::new(c.r + step, c.a, c.b, c.c),
            Quat::new(c.r, c.a + step, c.b, c.c),
            Quat::new(c.r, c.a, c.b + step, c.c),
            Quat::new(c.r, c.a, c.b, c.c + step),
        ];
        let vals: [f32; 5] = std::array::from_fn(|k| quat_dag_escape_de(f, offsets[k], max_iter, bailout_sq).0);
        let mut order: [u8; 5] = [0, 1, 2, 3, 4];
        order.sort_by(|&a, &b| vals[a as usize].partial_cmp(&vals[b as usize]).unwrap_or(std::cmp::Ordering::Equal));
        *pattern_counts.entry(order).or_insert(0) += 1;
    }
    pattern_histogram_complexity(&pattern_counts, n_clusters)
}

/// Entropy × disequilibrium over an OBSERVED pattern-count histogram
/// (sparse: only patterns that actually occurred are keys), against the
/// full 5! = 120-pattern space — unobserved patterns implicitly
/// contribute their `(0 - uniform)²` disequilibrium term without needing
/// all 120 slots allocated. Split out from `ordinal_complexity` so the
/// entropy/disequilibrium math is directly unit-testable without needing
/// a real formula to produce a degenerate pattern distribution.
fn pattern_histogram_complexity(pattern_counts: &HashMap<[u8; 5], u32>, n_clusters: usize) -> f32 {
    const N_PATTERNS: f32 = 120.0; // 5!
    let n = n_clusters as f32;
    let observed: Vec<f32> = pattern_counts.values().map(|&c| c as f32 / n).collect();
    let h_norm = (shannon_entropy(&observed) / N_PATTERNS.log2()).clamp(0.0, 1.0);
    let uniform = 1.0 / N_PATTERNS;
    let sum_sq_observed: f32 = observed.iter().map(|&p| (p - uniform).powi(2)).sum();
    let n_unobserved = (N_PATTERNS - observed.len() as f32).max(0.0);
    let d = (sum_sq_observed + n_unobserved * uniform.powi(2)).sqrt();
    let d_max = ((1.0 - uniform).powi(2) + (N_PATTERNS - 1.0) * uniform.powi(2)).sqrt();
    let d_norm = if d_max > 1e-9 { (d / d_max).clamp(0.0, 1.0) } else { 0.0 };
    (h_norm * d_norm).clamp(0.0, 1.0)
}

/// Occupied-box count and sum-of-squared-mass over a `boxes_per_axis^4`
/// grid, from the SAME sample set `statistical_complexity` used — reused
/// rather than re-sampled, since box-counting cares about where points
/// landed, not about escape time specifically.
fn box_stats(samples: &[FieldSample], domain_radius: f64, boxes_per_axis: usize) -> (f64, f64) {
    let cell = (2.0 * domain_radius) / boxes_per_axis as f64;
    let mut counts: HashMap<(i32, i32, i32, i32), u32> = HashMap::new();
    let bin = |v: f64| (((v + domain_radius) / cell) as i32).clamp(0, boxes_per_axis as i32 - 1);
    for s in samples {
        let key = (bin(s.point.r), bin(s.point.a), bin(s.point.b), bin(s.point.c));
        *counts.entry(key).or_insert(0) += 1;
    }
    let n = samples.len().max(1) as f64;
    let occupied = counts.len() as f64;
    let sum_p2: f64 = counts.values().map(|&c| (c as f64 / n).powi(2)).sum();
    (occupied, sum_p2)
}

/// 2-point multifractal spectrum width: `|D₀ − D₂|` estimated from
/// occupied-box-count scaling (D₀, the ordinary box-counting dimension)
/// and sum-of-squared-mass scaling (D₂, the correlation dimension)
/// between a coarse (4 boxes/axis) and fine (8 boxes/axis) grid. A
/// monofractal — uniformly rough OR uniformly smooth, which includes
/// both "empty" and "pure noise" — gives D₀≈D₂ (narrow/zero width).
/// Genuine multiscale organization pulls them apart. Normalized against
/// 4.0 (the maximum possible dimension of a 4D object) so callers get a
/// roughly [0,1] value like every other metric here.
fn multifractal_spectrum_width(samples: &[FieldSample], domain_radius: f64) -> f32 {
    let (n1, p2_1) = box_stats(samples, domain_radius, 4);
    let (n2, p2_2) = box_stats(samples, domain_radius, 8);
    if n1 < 1.0 || n2 < 1.0 || p2_1 <= 0.0 || p2_2 <= 0.0 {
        return 0.0;
    }
    let ln2 = 2.0_f64.ln();
    let d0 = (n2.ln() - n1.ln()) / ln2;
    let d2 = -(p2_2.ln() - p2_1.ln()) / ln2;
    ((d0 - d2).abs() / 4.0).clamp(0.0, 1.0) as f32
}

/// Interleaves the low 16 bits of 4 coordinates into a 64-bit Morton
/// (Z-order) code — a straightforward per-bit loop rather than a
/// bit-trick "magic constant" spread, deliberately: this only runs a few
/// thousand times per genome, so the simpler-to-verify version is worth
/// more than the faster one.
fn morton4(a: u32, b: u32, c: u32, d: u32) -> u64 {
    let mut m: u64 = 0;
    for bit in 0..16 {
        m |= (((a >> bit) & 1) as u64) << (4 * bit);
        m |= (((b >> bit) & 1) as u64) << (4 * bit + 1);
        m |= (((c >> bit) & 1) as u64) << (4 * bit + 2);
        m |= (((d >> bit) & 1) as u64) << (4 * bit + 3);
    }
    m
}

/// Compression-ratio complexity (Kolmogorov-complexity proxy, via
/// Normalized-Compression-Distance-style reasoning): quantize each
/// sample's escape time to a byte, order the samples along a
/// locality-preserving 4D space-filling curve (Morton code of their
/// quantized position — order matters, an arbitrary/random scan order
/// would make even genuinely structured fields compress like noise),
/// and compress with zlib. Pure noise is near-incompressible (ratio≈1);
/// an empty/constant field compresses to almost nothing (ratio≈0); a
/// genuinely structured, self-similar fractal sits in between, because
/// the compressor can exploit exactly the kind of repetition a real
/// fractal has and noise doesn't.
fn compression_complexity(samples: &[FieldSample], domain_radius: f64, max_iter: u32) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let quantize_coord = |v: f64| (((v + domain_radius) / (2.0 * domain_radius)) * 65535.0).clamp(0.0, 65535.0) as u32;
    let mut indexed: Vec<(u64, u8)> = samples
        .iter()
        .map(|s| {
            let morton = morton4(quantize_coord(s.point.r), quantize_coord(s.point.a), quantize_coord(s.point.b), quantize_coord(s.point.c));
            let byte = ((s.escape_time / max_iter.max(1) as f32).clamp(0.0, 1.0) * 255.0) as u8;
            (morton, byte)
        })
        .collect();
    indexed.sort_by_key(|&(m, _)| m);
    let bytes: Vec<u8> = indexed.into_iter().map(|(_, b)| b).collect();

    use flate2::write::ZlibEncoder;
    use flate2::Compression;
    let mut enc = ZlibEncoder::new(Vec::new(), Compression::best());
    if enc.write_all(&bytes).is_err() {
        return 0.0;
    }
    let compressed = match enc.finish() {
        Ok(c) => c,
        Err(_) => return 0.0,
    };
    (compressed.len() as f32 / bytes.len().max(1) as f32).clamp(0.0, 1.0)
}

/// Finite-difference chaoticity: for a subsample of field points, nudge
/// one coordinate by a small `eps` and measure how much the escape time
/// jumps, `|Δescape_time| / eps`. Averaged (in log space, to keep a few
/// near-boundary outliers from dominating the mean) and squashed into
/// `[0,1)` via a saturating exponential — no threshold to tune, purely
/// monotonic. This is the practical, render-free stand-in for a Lyapunov
/// exponent this module uses (see the module doc for why it reads the
/// escape-time FIELD's sensitivity rather than harvesting
/// `quat_dag_escape_de`'s internal derivative accumulator).
fn chaoticity(f: &QuatDagFormula, samples: &[FieldSample], domain_radius: f64, max_iter: u32, bailout_sq: f64) -> f32 {
    let eps = domain_radius * 1e-3;
    let mut total = 0.0f64;
    let mut count = 0usize;
    for s in samples.iter().step_by(4) {
        let perturbed = Quat::new(s.point.r + eps, s.point.a, s.point.b, s.point.c);
        let (et2, _) = quat_dag_escape_de(f, perturbed, max_iter, bailout_sq);
        let diff = (et2 - s.escape_time).abs() as f64;
        total += (diff / eps).ln_1p();
        count += 1;
    }
    if count == 0 {
        return 0.0;
    }
    let mean = total / count as f64;
    (1.0 - (-mean / 5.0).exp()).clamp(0.0, 1.0) as f32
}

/// Same finite-difference construction as `chaoticity`, generalized from
/// perturbing R alone to perturbing all 4 axes (R, A, B, C) in turn and
/// averaging — Carl, 2026-09-28, from browsing many fractals across the
/// viewer's X/Y/Z/T sliders: "most interesting fractals have hi entropy,
/// but are continuous (differentiable) along all axis. If the fractal is
/// mostly noise, it will not be continuous." `chaoticity` alone can't
/// answer that — it only ever nudges R, so a formula that's smooth along
/// R but chaotic along A/B/C would score falsely low here. Same
/// `eps`/`ln_1p`/exponential-squash construction as `chaoticity`, so the
/// two stay comparable in scale.
fn discontinuity(f: &QuatDagFormula, samples: &[FieldSample], domain_radius: f64, max_iter: u32, bailout_sq: f64) -> f32 {
    let eps = domain_radius * 1e-3;
    let mut total = 0.0f64;
    let mut count = 0usize;
    for s in samples.iter().step_by(4) {
        let p = s.point;
        let neighbors = [
            Quat::new(p.r + eps, p.a, p.b, p.c),
            Quat::new(p.r, p.a + eps, p.b, p.c),
            Quat::new(p.r, p.a, p.b + eps, p.c),
            Quat::new(p.r, p.a, p.b, p.c + eps),
        ];
        for n in neighbors {
            let (et2, _) = quat_dag_escape_de(f, n, max_iter, bailout_sq);
            let diff = (et2 - s.escape_time).abs() as f64;
            total += (diff / eps).ln_1p();
            count += 1;
        }
    }
    if count == 0 {
        return 0.0;
    }
    let mean = total / count as f64;
    (1.0 - (-mean / 5.0).exp()).clamp(0.0, 1.0) as f32
}

/// `1 - discontinuity` — 1.0 = a tiny nudge along any of the 4 axes never
/// changes the escape time much (smooth/differentiable), 0.0 = highly
/// sensitive to position (noise-like).
fn continuity(f: &QuatDagFormula, samples: &[FieldSample], domain_radius: f64, max_iter: u32, bailout_sq: f64) -> f32 {
    1.0 - discontinuity(f, samples, domain_radius, max_iter, bailout_sq)
}

/// Carl's stated rule as a plain product, not LMC's "penalize both
/// extremes" framing: an AND, not a compromise. High entropy but noisy
/// (low continuity) scores low; smooth but bland (low entropy) also
/// scores low; only genuinely "organized" complexity — both high at
/// once — scores high. This is the presumed-good training metric he
/// asked for.
fn organized_richness(entropy_norm: f32, continuity: f32) -> f32 {
    (entropy_norm * continuity).clamp(0.0, 1.0)
}

/// One direction from the golden-angle Fibonacci-sphere spiral — a
/// deterministic, no-RNG-needed way to spread `n` directions evenly over
/// the unit sphere (used here instead of Halton since this specifically
/// wants near-uniform ANGULAR coverage, not low-discrepancy coverage of
/// a volume).
fn fibonacci_sphere_direction(i: usize, n: usize) -> Vec3 {
    const GOLDEN_ANGLE: f64 = std::f64::consts::PI * (3.0 - 2.236_067_977_499_79 /* sqrt(5) */);
    let y = 1.0 - 2.0 * (i as f64 / (n.max(2) - 1) as f64);
    let radius_at_y = (1.0 - y * y).max(0.0).sqrt();
    let theta = GOLDEN_ANGLE * i as f64;
    (theta.cos() * radius_at_y, y, theta.sin() * radius_at_y)
}

/// Finds the OUTER surface of the filled (non-escaping) set at fixed
/// `C=0` in direction `dir`, by marching INWARD from `domain_radius`
/// toward the origin (mirroring how a real camera sees this genome —
/// from outside, stopping at the first solid material) — returns that
/// distance from the origin, or `None` if nothing along the ray ever
/// escapes-vs-doesn't (no material found in that direction at all).
///
/// Two earlier designs both marched OUTWARD from the origin instead, and
/// both failed for reasons kept in git history:
/// - A DE sphere-trace: the standard analytic distance estimator
///   (`0.5 * r * ln(r) / |dz|`) is only a good approximation NEAR THE
///   BOUNDARY; deep in the interior it's mathematically degenerate, and
///   the origin (C=0, z starts at 0) is EXACTLY a deep interior point for
///   almost any Mandelbrot-style formula — no fixed exclusion zone or
///   "confirm the next step" check could tell a real hit from a
///   DE-magnitude artifact at every possible real structure scale.
/// - An escape-time (`et`) coarse-scan-then-bisect OUTWARD from the
///   origin, looking for the first inside-to-outside transition: fixed
///   the DE issue, but broke on a genome Carl pointed out as an OBVIOUSLY
///   clean, fully solid sphere by eye (`examples/sph_a7511_check.rs`,
///   kept for this kind of spot-check) — its filled region isn't
///   star-shaped from the origin: sampling straight out along +R found
///   *inside* out to r~0.8, briefly *outside* around r~1.0, then
///   *inside* again at r~1.5 (a real, if unusual, interior topology for
///   a Julia-mode formula). Marching outward finds the FIRST such
///   transition, which is essentially arbitrary noise across 300
///   different directions when the shells aren't concentric — high
///   apparent variance (looks "not round") for a genome that renders as
///   a clean circle from any normal camera. A camera never sees this
///   ambiguity because it never marches past the first surface it hits.
///
/// Marching inward matches that camera's own logic exactly (and is
/// naturally immune to the origin's own degeneracy, since t=`domain_radius`
/// starts the search far outside any interior weirdness): the escape-time
/// `et` (pinned at `max_iter` = never escapes = inside) is still the
/// signal, still with no near-origin singularity, just walked from the
/// other end.
fn march_radius(f: &QuatDagFormula, dir: Vec3, domain_radius: f64, max_iter: u32, bailout_sq: f64) -> Option<f64> {
    const COARSE_STEPS: u32 = 64;
    const BISECT_ITERS: u32 = 20;
    let is_inside = |t: f64| -> bool {
        let point: Vec3 = (dir.0 * t, dir.1 * t, dir.2 * t);
        let q = TimeAxis::C.assemble(point, 0.0);
        let (et, _) = quat_dag_escape_de(f, q, max_iter, bailout_sq);
        et >= max_iter as f32 - 0.5
    };
    let mut t_outside = domain_radius;
    let mut t_inside = None;
    for i in 1..=COARSE_STEPS {
        let t = domain_radius * (1.0 - i as f64 / COARSE_STEPS as f64);
        if is_inside(t) {
            t_inside = Some(t);
            break;
        } else {
            t_outside = t;
        }
    }
    let mut lo = t_inside?;
    let mut hi = t_outside;
    for _ in 0..BISECT_ITERS {
        let mid = (lo + hi) / 2.0;
        if is_inside(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some((lo + hi) / 2.0)
}

/// Sphericity of the genome's shape at `C=0` — REVISED after Carl looked
/// at the first version's own showcase examples and rejected them: "A
/// spherical 4d fractal is a fractal with a clear defined surface that is
/// inpenetrable using raycast. It sits usually around the domain of the
/// fractal... If you cant see through it and it's a sphere, then it's
/// spherical." The original version only measured roundness among
/// whichever of the 300 origin-rays happened to hit anything, with a
/// blunt "give up and return 0 below 25% hit rate" cutoff above that —
/// so a shape riddled with holes (well over half the rays missing
/// entirely, i.e. very much see-through) could still score a perfect 1.0
/// as long as the minority of rays that DID hit landed at a consistent
/// radius. That's backwards: "you can see through it" should itself be
/// disqualifying, continuously, not just below a hard 25% floor.
///
/// Now three independent `[0,1]` factors, MULTIPLIED together (AND, not
/// blended — failing any one factor should tank the score, not just
/// dilute it):
/// - `hit_rate` — fraction of the 300 radial rays that hit *something*
///   within the domain. This is literally "inpenetrable using raycast":
///   a shape full of holes/dust/tunnels has a low hit rate, however round
///   its surviving hits are.
/// - `fill_ratio` — mean hit radius relative to `domain_radius`. "Sits
///   usually around the domain of the fractal": a boring blob typically
///   balloons out close to the render domain's own boundary rather than
///   sitting small in the middle of empty space, so this rewards hits
///   that reach most of the way to `domain_radius`, not just any hit.
/// - `roundness` — the original coefficient-of-variation measure
///   (`radii_to_sphericity`), now just one of three factors instead of
///   the whole metric — "and it's a sphere" is still part of Carl's own
///   definition, just no longer sufficient on its own.
///
/// Deliberately NOT the same thing `quat_dag_fitness.rs`'s `convexity`
/// measures (a single camera silhouette's concavity), and still meant as
/// a PENALTY — weight it negatively in a `--fitness-metric` combo, not
/// maximized on its own (rewarding "as non-spherical as possible" on its
/// own would just be `box_dim` with extra steps).
pub fn sphericity(f: &QuatDagFormula, domain_radius: f64, max_iter: u32, bailout_sq: f64) -> f32 {
    const N_DIRECTIONS: usize = 300;
    let radii: Vec<f64> = (0..N_DIRECTIONS)
        .filter_map(|i| march_radius(f, fibonacci_sphere_direction(i, N_DIRECTIONS), domain_radius, max_iter, bailout_sq))
        .collect();
    let hit_count = radii.len();
    sphericity_from_hits(hit_count, N_DIRECTIONS, &radii, domain_radius)
}

/// Diagnostic breakdown of the raw inputs `sphericity` combines —
/// `(hit_count, n_directions, mean_hit_radius)` — without the
/// hit-rate/fill-ratio/roundness product applied. Exists so the
/// individual factors can be inspected against real archived genomes
/// (calibrating what `domain_radius` fraction a "fills the domain" blob
/// actually reaches) without re-deriving the radial march by hand.
pub fn sphericity_radial_profile(f: &QuatDagFormula, domain_radius: f64, max_iter: u32, bailout_sq: f64) -> (usize, usize, f64) {
    const N_DIRECTIONS: usize = 300;
    let radii: Vec<f64> = (0..N_DIRECTIONS)
        .filter_map(|i| march_radius(f, fibonacci_sphere_direction(i, N_DIRECTIONS), domain_radius, max_iter, bailout_sq))
        .collect();
    let hit_count = radii.len();
    let mean_radius = if radii.is_empty() { 0.0 } else { radii.iter().sum::<f64>() / radii.len() as f64 };
    (hit_count, N_DIRECTIONS, mean_radius)
}

/// Same as `sphericity_radial_profile` but also returns `roundness`
/// (`radii_to_sphericity`'s own [0,1] output) — the full three-factor
/// breakdown for one formula, for diagnosing a specific genome by hand.
pub fn sphericity_full_diagnostic(f: &QuatDagFormula, domain_radius: f64, max_iter: u32, bailout_sq: f64) -> (usize, usize, f64, f32, f32) {
    const N_DIRECTIONS: usize = 300;
    let radii: Vec<f64> = (0..N_DIRECTIONS)
        .filter_map(|i| march_radius(f, fibonacci_sphere_direction(i, N_DIRECTIONS), domain_radius, max_iter, bailout_sq))
        .collect();
    let hit_count = radii.len();
    let mean_radius = if radii.is_empty() { 0.0 } else { radii.iter().sum::<f64>() / radii.len() as f64 };
    let roundness = radii_to_sphericity(&radii);
    let total = sphericity_from_hits(hit_count, N_DIRECTIONS, &radii, domain_radius);
    (hit_count, N_DIRECTIONS, mean_radius, roundness, total)
}

/// The testable core of `sphericity`: `hit_rate` ("inpenetrable using
/// raycast") &times; `fill_ratio` ("sits usually around the domain") &times;
/// `roundness` (`radii_to_sphericity`, "and it's a sphere"), each in
/// `[0,1]`. A product rather than an average on purpose — Carl's
/// definition is a conjunction ("if you cant see through it AND it's a
/// sphere"), and only a product drives the score to ~0 when any single
/// factor fails, the way `AND` should. Split out from `sphericity` so
/// this is unit-testable with synthetic hit counts/radii, independent of
/// any real formula's actual shape.
fn sphericity_from_hits(hit_count: usize, n_directions: usize, radii: &[f64], domain_radius: f64) -> f32 {
    if radii.is_empty() || n_directions == 0 || domain_radius < 1e-9 {
        return 0.0;
    }
    let hit_rate = hit_count as f64 / n_directions as f64;
    let mean_radius = radii.iter().sum::<f64>() / radii.len() as f64;
    // Plain `mean_radius / domain_radius`, no extra calibration constant
    // — `march_radius` now finds the OUTER surface by marching inward
    // from `domain_radius` (see its doc comment), so a genuinely
    // domain-filling genome's hits land close to `domain_radius` itself
    // by construction. An earlier version of this line divided by a
    // separately-calibrated `domain_radius * 0.3` to compensate for the
    // previous outward-from-origin `march_radius`, which found structure
    // much closer to the origin — that calibration doesn't apply anymore
    // and was removed along with it (re-checked against real archived
    // genomes: raw `mean_radius/domain_radius` already spans close to the
    // full `[0,1]` range for real genomes — median ~0.5 among genomes
    // with a high hit rate).
    let fill_ratio = (mean_radius / domain_radius).clamp(0.0, 1.0);
    let roundness = radii_to_sphericity(radii) as f64;
    (hit_rate * fill_ratio * roundness) as f32
}

/// The testable core of `roundness`: coefficient-of-variation of a set
/// of radii, squashed into `[0,1]` via a saturating exponential (smooth
/// at both ends, no hard clip) — `0` variation (a true sphere) maps to
/// exactly `1.0`; growing variation decays toward `0`. Split out from
/// `sphericity_from_hits` so this transform is unit-testable with
/// synthetic radii, independent of any real formula's actual shape.
fn radii_to_sphericity(radii: &[f64]) -> f32 {
    if radii.is_empty() {
        return 0.0;
    }
    let mean = radii.iter().sum::<f64>() / radii.len() as f64;
    if mean < 1e-9 {
        return 0.0;
    }
    let variance = radii.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / radii.len() as f64;
    let coefficient_of_variation = variance.sqrt() / mean;
    // `k` calibrated so a modestly lobed shape (cv around 0.15-0.3, a
    // plausible range for real evolved genomes) reads as clearly less
    // than perfectly spherical without needing extreme variation to move
    // off of 1.0.
    const K: f64 = 0.15;
    // NOT `1 - exp(-cv/K)` — that shape starts at 0 and RISES with cv,
    // exactly backwards (caught by this module's own
    // radii_to_sphericity_is_exactly_one_for_zero_variation test, which
    // failed against the first version of this line). Plain exp(-cv/K)
    // starts at 1.0 (cv=0, a true sphere) and decays toward 0 as
    // variation grows.
    (-coefficient_of_variation / K).exp().clamp(0.0, 1.0) as f32
}

/// Every whole-4D-object organization metric, computed from one shared
/// sample of the quaternion function — no rendering, no camera, no
/// TimeAxis choice; every one of R/A/B/C is sampled as an equal spatial
/// coordinate. `domain_radius` should match the genome's own scale (the
/// same 1.6 the render-based probes use is a reasonable default —
/// callers pick). `sphericity` is the one exception to "no TimeAxis
/// choice" — it's deliberately scoped to the `C=0` slice specifically
/// (see its own doc comment), not the full 4D object.
#[derive(Debug, Clone, Copy, Default)]
pub struct OrganizationMetrics {
    pub statistical_complexity: f32,
    pub ordinal_complexity: f32,
    pub multifractal_width: f32,
    pub compression_complexity: f32,
    pub chaoticity: f32,
    pub sphericity: f32,
    /// Plain Shannon entropy of the escape-time histogram, in [0,1] —
    /// see `escape_time_entropy_norm`'s doc comment.
    pub field_entropy: f32,
    /// `1 - discontinuity` (4-axis-generalized `chaoticity`), in [0,1] —
    /// see `continuity`'s doc comment.
    pub continuity: f32,
    /// `field_entropy * continuity` — Carl's "hi entropy, but continuous"
    /// rule as a single scalar. See `organized_richness`'s doc comment.
    pub organized_richness: f32,
}

const N_FIELD_SAMPLES: usize = 3000;
const N_ORDINAL_CLUSTERS: usize = 600;

pub fn compute_organization_metrics(f: &QuatDagFormula, domain_radius: f64, max_iter: u32, bailout_sq: f64) -> OrganizationMetrics {
    let samples = sample_field(f, N_FIELD_SAMPLES, domain_radius, max_iter, bailout_sq);
    let field_entropy = escape_time_entropy_norm(&samples, max_iter);
    let continuity_val = continuity(f, &samples, domain_radius, max_iter, bailout_sq);
    OrganizationMetrics {
        statistical_complexity: statistical_complexity(&samples, max_iter),
        ordinal_complexity: ordinal_complexity(f, domain_radius, max_iter, bailout_sq, N_ORDINAL_CLUSTERS),
        multifractal_width: multifractal_spectrum_width(&samples, domain_radius),
        compression_complexity: compression_complexity(&samples, domain_radius, max_iter),
        chaoticity: chaoticity(f, &samples, domain_radius, max_iter, bailout_sq),
        sphericity: sphericity(f, domain_radius, max_iter, bailout_sq),
        field_entropy,
        continuity: continuity_val,
        organized_richness: organized_richness(field_entropy, continuity_val),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formula::{op, OpNode};

    fn mandelbrot_program() -> Vec<OpNode> {
        // z' = z² + c — the exact hand-built DAG formula.rs's own
        // `mandelbrot_dag_matches_legacy` test uses (node array value is
        // always its LAST entry; a/b index strictly-earlier nodes).
        vec![
            OpNode { op: op::Z, a: 0, b: 0, kre: 0.0, kim: 0.0 },   // 0: z
            OpNode { op: op::C, a: 0, b: 0, kre: 0.0, kim: 0.0 },   // 1: c
            OpNode { op: op::SQR, a: 0, b: 0, kre: 0.0, kim: 0.0 }, // 2: z²
            OpNode { op: op::ADD, a: 2, b: 1, kre: 0.0, kim: 0.0 }, // 3: z²+c
        ]
    }

    fn identity_program() -> Vec<OpNode> {
        // z' = c — every iteration overwrites z with the constant, so the
        // orbit becomes literally constant after one step regardless of
        // z. Far less sensitive to position than z²+c's real chaotic
        // dynamics, used as the "smooth" reference case.
        vec![
            OpNode { op: op::Z, a: 0, b: 0, kre: 0.0, kim: 0.0 }, // 0: z (unused, kept so C's index matches every other program)
            OpNode { op: op::C, a: 0, b: 0, kre: 0.0, kim: 0.0 }, // 1: c
        ]
    }

    fn formula(prog: &[OpNode]) -> QuatDagFormula<'_> {
        QuatDagFormula { prog, warp: &[], julia: false, jc: (0.0, 0.0), phoenix: (0.0, 0.0) }
    }

    #[test]
    fn statistical_complexity_is_zero_for_a_constant_field() {
        let samples: Vec<FieldSample> = (0..500).map(|_| FieldSample { point: Quat::ZERO, escape_time: 30.0 }).collect();
        assert_eq!(statistical_complexity(&samples, 60), 0.0);
    }

    #[test]
    fn statistical_complexity_is_low_for_a_uniform_histogram() {
        // Every bin equally populated == high entropy, ~zero disequilibrium
        // == low complexity by construction (LMC's own documented shape).
        let samples: Vec<FieldSample> = (0..3200)
            .map(|i| FieldSample { point: Quat::ZERO, escape_time: (i % 32) as f32 })
            .collect();
        let c = statistical_complexity(&samples, 32);
        assert!(c < 0.05, "uniform histogram should score near-zero complexity, got {c}");
    }

    #[test]
    fn statistical_complexity_is_higher_for_a_clustered_bimodal_histogram() {
        let uniform: Vec<FieldSample> = (0..3200).map(|i| FieldSample { point: Quat::ZERO, escape_time: (i % 32) as f32 }).collect();
        let bimodal: Vec<FieldSample> = (0..3200)
            .map(|i| FieldSample { point: Quat::ZERO, escape_time: if i % 2 == 0 { 5.0 } else { 27.0 } })
            .collect();
        let c_uniform = statistical_complexity(&uniform, 32);
        let c_bimodal = statistical_complexity(&bimodal, 32);
        assert!(c_bimodal > c_uniform, "bimodal ({c_bimodal}) should score above uniform ({c_uniform})");
    }

    #[test]
    fn ordinal_complexity_is_zero_when_every_cluster_gives_the_same_pattern() {
        // Every cluster lands on the identical ordinal pattern (as a
        // perfectly linear/monotonic field would) -> zero pattern
        // entropy -> zero complexity regardless of disequilibrium.
        let mut counts = HashMap::new();
        counts.insert([0u8, 1, 2, 3, 4], 200);
        assert_eq!(pattern_histogram_complexity(&counts, 200), 0.0);
    }

    /// All 120 permutations of [0,1,2,3,4], via straightforward
    /// insertion-based generation — only needed by the test below, kept
    /// deliberately simple over clever.
    fn all_5_permutations() -> Vec<[u8; 5]> {
        let mut perms = vec![vec![0u8]];
        for next in 1u8..5 {
            let mut grown = Vec::new();
            for p in &perms {
                for pos in 0..=p.len() {
                    let mut np = p.clone();
                    np.insert(pos, next);
                    grown.push(np);
                }
            }
            perms = grown;
        }
        perms.into_iter().map(|v| v.try_into().unwrap()).collect()
    }

    #[test]
    fn ordinal_complexity_is_higher_when_a_few_patterns_dominate_than_when_spread_evenly() {
        let all = all_5_permutations();
        assert_eq!(all.len(), 120);

        // Spread: every one of the 120 patterns occurs equally often —
        // the ordinal-pattern equivalent of noise (no local structure
        // favors any particular neighbor arrangement).
        let mut spread = HashMap::new();
        for p in &all {
            spread.insert(*p, 2);
        }
        let n_spread = 120 * 2;

        // Favored: only 2 of the 120 patterns ever occur — the ordinal-
        // pattern equivalent of real structure (a gradient/ridge favors
        // a small, characteristic set of local rank-orderings).
        let mut favored = HashMap::new();
        favored.insert(all[0], 180);
        favored.insert(all[1], 60);
        let n_favored = 240;

        let c_spread = pattern_histogram_complexity(&spread, n_spread);
        let c_favored = pattern_histogram_complexity(&favored, n_favored);
        assert!(
            c_favored > c_spread,
            "a few dominant patterns ({c_favored}) should score above an even spread across all patterns ({c_spread})"
        );
    }

    #[test]
    fn ordinal_complexity_on_a_real_formula_is_finite_and_in_range() {
        let prog = mandelbrot_program();
        let f = formula(&prog);
        let v = ordinal_complexity(&f, 1.6, 40, 16.0, 200);
        assert!(v.is_finite() && (0.0..=1.0).contains(&v), "ordinal_complexity out of range: {v}");
    }

    #[test]
    fn compression_complexity_is_near_zero_for_a_constant_field() {
        let samples: Vec<FieldSample> = (0..2000)
            .map(|i| {
                let t = i as f64 / 2000.0;
                FieldSample { point: Quat::new(t, 0.0, 0.0, 0.0), escape_time: 30.0 }
            })
            .collect();
        let c = compression_complexity(&samples, 1.6, 60);
        assert!(c < 0.05, "constant field should compress to near-zero ratio, got {c}");
    }

    #[test]
    fn compression_complexity_is_high_for_a_pseudorandom_field() {
        // A cheap xorshift-style PRNG, not tied to any formula — pure
        // noise input, no spatial structure at all.
        let mut state: u32 = 0x1234_5678;
        let samples: Vec<FieldSample> = (0..2000)
            .map(|i| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                let t = i as f64 / 2000.0;
                FieldSample { point: Quat::new(t, 0.0, 0.0, 0.0), escape_time: (state % 60) as f32 }
            })
            .collect();
        let c = compression_complexity(&samples, 1.6, 60);
        assert!(c > 0.5, "pseudorandom field should compress poorly (high ratio), got {c}");
    }

    #[test]
    fn multifractal_width_is_zero_for_a_single_occupied_box() {
        let samples: Vec<FieldSample> = (0..500).map(|_| FieldSample { point: Quat::ZERO, escape_time: 30.0 }).collect();
        let w = multifractal_spectrum_width(&samples, 1.6);
        assert_eq!(w, 0.0);
    }

    #[test]
    fn chaoticity_is_higher_for_an_unstable_formula_than_a_smooth_one() {
        let smooth_prog = identity_program();
        let unstable_prog = mandelbrot_program();
        let smooth = formula(&smooth_prog);
        let unstable = formula(&unstable_prog);
        let smooth_samples = sample_field(&smooth, 400, 1.6, 40, 16.0);
        let unstable_samples = sample_field(&unstable, 400, 1.6, 40, 16.0);
        let c_smooth = chaoticity(&smooth, &smooth_samples, 1.6, 40, 16.0);
        let c_unstable = chaoticity(&unstable, &unstable_samples, 1.6, 40, 16.0);
        assert!(c_unstable >= c_smooth, "mandelbrot ({c_unstable}) should be at least as chaotic as a linear field ({c_smooth})");
    }

    #[test]
    fn continuity_is_higher_for_a_smooth_formula_than_an_unstable_one() {
        let smooth_prog = identity_program();
        let unstable_prog = mandelbrot_program();
        let smooth = formula(&smooth_prog);
        let unstable = formula(&unstable_prog);
        let smooth_samples = sample_field(&smooth, 400, 1.6, 40, 16.0);
        let unstable_samples = sample_field(&unstable, 400, 1.6, 40, 16.0);
        let c_smooth = continuity(&smooth, &smooth_samples, 1.6, 40, 16.0);
        let c_unstable = continuity(&unstable, &unstable_samples, 1.6, 40, 16.0);
        assert!(c_smooth >= c_unstable, "a linear field ({c_smooth}) should be at least as continuous as mandelbrot ({c_unstable})");
    }

    #[test]
    fn organized_richness_is_the_product_of_entropy_and_continuity() {
        assert_eq!(organized_richness(0.8, 0.5), 0.4);
        assert_eq!(organized_richness(1.0, 0.0), 0.0, "zero continuity (pure noise) must zero out even high entropy");
        assert_eq!(organized_richness(0.0, 1.0), 0.0, "zero entropy (blank field) must zero out even perfect continuity");
    }

    #[test]
    fn compute_organization_metrics_returns_finite_in_range_values() {
        let prog = mandelbrot_program();
        let f = formula(&prog);
        let m = compute_organization_metrics(&f, 1.6, 40, 16.0);
        for (name, v) in [
            ("statistical_complexity", m.statistical_complexity),
            ("ordinal_complexity", m.ordinal_complexity),
            ("multifractal_width", m.multifractal_width),
            ("compression_complexity", m.compression_complexity),
            ("chaoticity", m.chaoticity),
            ("sphericity", m.sphericity),
            ("field_entropy", m.field_entropy),
            ("continuity", m.continuity),
            ("organized_richness", m.organized_richness),
        ] {
            assert!(v.is_finite() && (0.0..=1.0).contains(&v), "{name} out of range: {v}");
        }
    }

    #[test]
    fn radii_to_sphericity_is_exactly_one_for_zero_variation() {
        let radii = vec![1.2; 300];
        assert_eq!(radii_to_sphericity(&radii), 1.0);
    }

    #[test]
    fn radii_to_sphericity_decreases_as_variation_increases() {
        let uniform: Vec<f64> = vec![1.0; 300];
        let mild: Vec<f64> = (0..300).map(|i| 1.0 + 0.05 * (i as f64 / 300.0 - 0.5)).collect();
        let wild: Vec<f64> = (0..300).map(|i| if i % 2 == 0 { 0.3 } else { 1.7 }).collect();
        let s_uniform = radii_to_sphericity(&uniform);
        let s_mild = radii_to_sphericity(&mild);
        let s_wild = radii_to_sphericity(&wild);
        assert!(s_uniform > s_mild, "uniform ({s_uniform}) should score more spherical than mild variation ({s_mild})");
        assert!(s_mild > s_wild, "mild variation ({s_mild}) should score more spherical than wild variation ({s_wild})");
    }

    #[test]
    fn march_radius_hits_a_real_formula_from_the_origin() {
        // Sanity check that the sphere-tracing loop actually converges
        // for a real formula (not just synthetic radii) — the Mandelbrot
        // DAG program, straight out from the origin along +R, should hit
        // SOMETHING within the domain (it's the seed formula every other
        // test in this module already trusts).
        let prog = mandelbrot_program();
        let f = formula(&prog);
        let hit = march_radius(&f, (1.0, 0.0, 0.0), 1.6, 40, 16.0);
        assert!(hit.is_some(), "expected a hit marching outward from the origin");
        let r = hit.unwrap();
        assert!(r > 0.0 && r <= 1.6, "hit radius {r} out of the expected domain range");
    }

    #[test]
    fn sphericity_on_a_real_formula_is_finite_and_in_range() {
        let prog = mandelbrot_program();
        let f = formula(&prog);
        let s = sphericity(&f, 1.6, 40, 16.0);
        assert!(s.is_finite() && (0.0..=1.0).contains(&s), "sphericity out of range: {s}");
    }

    #[test]
    fn sphericity_from_hits_is_near_one_only_when_impenetrable_and_domain_filling_and_round() {
        // Every ray hits, right at the domain boundary, perfectly
        // uniform — the actual "boring solid ball" case Carl described.
        let radii = vec![1.6; 300];
        let s = sphericity_from_hits(300, 300, &radii, 1.6);
        assert!(s > 0.9, "expected near-1.0 for a fully-hit, domain-filling, perfectly round shape, got {s}");
    }

    #[test]
    fn sphericity_from_hits_penalizes_low_hit_rate_even_with_perfectly_uniform_hits() {
        // The bug in the first version: a shape where most of the 300
        // rays miss (you CAN see through it — holes, dust, tunnels) but
        // the minority that hit are all at the same radius used to score
        // a perfect 1.0. Carl's own words: "A spherical 4d fractal is a
        // fractal with a clear defined surface that is inpenetrable
        // using raycast" — low hit rate means it is NOT inpenetrable, so
        // this must score low regardless of how uniform the few hits are.
        let full_hit_radii = vec![1.6; 300];
        let sparse_hit_radii = vec![1.6; 30]; // only 30/300 rays hit — same uniform radius
        let s_full = sphericity_from_hits(300, 300, &full_hit_radii, 1.6);
        let s_sparse = sphericity_from_hits(30, 300, &sparse_hit_radii, 1.6);
        assert!(s_full > 0.9, "expected near-1.0 for full hit rate, got {s_full}");
        assert!(s_sparse < 0.2, "a mostly see-through shape (30/300 hits) must not score high just because its few hits are uniform, got {s_sparse}");
    }

    #[test]
    fn sphericity_from_hits_penalizes_a_tiny_blob_that_doesnt_reach_the_domain_boundary() {
        // "It sits usually around the domain of the fractal" — fully hit
        // and perfectly uniform, but tiny relative to domain_radius,
        // should score lower than the same shape sized up to fill the
        // domain (a small round pebble floating in empty space isn't the
        // boring-blob case either).
        let near_boundary = vec![1.55; 300];
        let near_center = vec![0.2; 300];
        let s_boundary = sphericity_from_hits(300, 300, &near_boundary, 1.6);
        let s_center = sphericity_from_hits(300, 300, &near_center, 1.6);
        assert!(s_boundary > s_center, "a blob filling the domain ({s_boundary}) should score more spherical than a tiny uniform blob near the center ({s_center})");
    }

    #[test]
    fn sphericity_from_hits_is_zero_with_no_hits() {
        assert_eq!(sphericity_from_hits(0, 300, &[], 1.6), 0.0);
    }

    #[test]
    fn morton4_round_trip_preserves_ordering_within_each_axis() {
        // Not a full round-trip test (no de-interleave implemented, none
        // needed) — just confirms the interleave is monotonic in each
        // axis independently, i.e. increasing one coordinate with the
        // others fixed never decreases the resulting code less often
        // than it increases it (a basic locality sanity check).
        let a = morton4(10, 10, 10, 10);
        let b = morton4(11, 10, 10, 10);
        assert!(b > a);
    }
}
