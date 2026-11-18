//! Pen-plotter output for a 2D escape-time field (a slice through the 3D view):
//! contour lines → cubic Bézier curves → SVG and Marlin G-code.
//!
//! - Levels: equal-area bands of the escaped samples (each escaped sample is
//!   replaced by its rank fraction, the histogram equalisation the slice view
//!   uses), plus the set's outline (escaped vs never escaped).
//! - Crowding: contour lines bunch up where the field is steep (near the set) and
//!   would merge into ink. Level `i` is drawn in a grid cell only if, at its own
//!   hierarchy step (every 2^trailing_zeros(i)-th level), the gap to the next
//!   line of that step is at least `min_gap` mm. Steep places keep every 2nd,
//!   4th, 8th … line, like the index contours of a topographic map.
//! - Curves: Schneider's least-squares cubic fit ("An algorithm for automatically
//!   fitting digitized curves", Graphics Gems 1990) within `tol` mm.
//! - Order: greedy nearest-neighbour over path ends (open paths may be reversed,
//!   closed loops start at their vertex nearest the pen) to cut pen-up travel.

use rayon::prelude::*;
use std::collections::HashMap;
use std::fmt::Write as _;

pub type P2 = (f64, f64);
/// Start, control 1, control 2, end.
pub type Cubic = [P2; 4];

fn sub(a: P2, b: P2) -> P2 { (a.0 - b.0, a.1 - b.1) }
fn add(a: P2, b: P2) -> P2 { (a.0 + b.0, a.1 + b.1) }
fn mul(a: P2, s: f64) -> P2 { (a.0 * s, a.1 * s) }
fn dot(a: P2, b: P2) -> f64 { a.0 * b.0 + a.1 * b.1 }
fn dist(a: P2, b: P2) -> f64 { dot(sub(a, b), sub(a, b)).sqrt() }
fn unit(a: P2) -> Option<P2> {
    let l = dot(a, a).sqrt();
    (l > 1e-12).then(|| mul(a, 1.0 / l))
}

/// A polyline; when `closed` its last point equals its first.
#[derive(Clone, Debug, PartialEq)]
pub struct Polyline {
    pub pts: Vec<P2>,
    pub closed: bool,
}

impl Polyline {
    pub fn length(&self) -> f64 { self.pts.windows(2).map(|w| dist(w[0], w[1])).sum() }
}

/// A path of joined cubics (each starts where the previous ends).
#[derive(Clone, Debug, PartialEq)]
pub struct CurvePath {
    pub curves: Vec<Cubic>,
    pub closed: bool,
}

/// Contour lines of the `w`×`h` field `f` (row-major, row 0 = top) at `level`,
/// in grid coordinates (sample (x, y) sits at (x, y)). A cell's segments are
/// kept only where `keep(x, y)` (x, y = the cell's top-left sample); dropped
/// cells break a line into pieces. Saddle cells are split by the cell's mean.
pub fn contour(f: &[f32], w: usize, h: usize, level: f32, keep: &(dyn Fn(usize, usize) -> bool + Sync)) -> Vec<Polyline> {
    contour_levels(f, w, h, &[level], &|_, x, y| keep(x, y)).pop().unwrap_or_default()
}

/// Contour lines of `f` at every value of `levels` (ascending) in one pass over
/// the cells: a cell only works on the levels its corners straddle, so many
/// levels cost little more than one. `keep(i, x, y)` drops level `i`'s segment
/// in cell (x, y). Cells with a non-finite corner are skipped. One list of
/// polylines per level, in grid coordinates.
pub fn contour_levels(f: &[f32], w: usize, h: usize, levels: &[f32], keep: &(dyn Fn(usize, usize, usize) -> bool + Sync)) -> Vec<Vec<Polyline>> {
    let nl = levels.len();
    let mut out: Vec<Vec<Polyline>> = vec![Vec::new(); nl];
    if nl == 0 || w < 2 || h < 2 {
        return out;
    }
    let nlu = nl as u64;
    // Edge ids: 2·sample (horizontal edge to the right) and 2·sample+1 (vertical
    // edge down); key = id·levels + level, so every level has its own nodes.
    let hid = |x: usize, y: usize| 2 * (y * w + x) as u64;
    let vid = |x: usize, y: usize| 2 * (y * w + x) as u64 + 1;
    let rows: Vec<usize> = (0..h - 1).collect();
    let parts: Vec<(HashMap<u64, P2>, Vec<(u64, u64)>)> = rows
        .par_chunks(32)
        .map(|chunk| {
            let mut pts: HashMap<u64, P2> = HashMap::new();
            let mut segs: Vec<(u64, u64)> = Vec::new();
            for &y in chunk {
                for x in 0..w - 1 {
                    let c = [f[y * w + x], f[y * w + x + 1], f[(y + 1) * w + x + 1], f[(y + 1) * w + x]];
                    if !c.iter().all(|v| v.is_finite()) {
                        continue;
                    }
                    let lo = c[0].min(c[1]).min(c[2]).min(c[3]);
                    let hi = c[0].max(c[1]).max(c[2]).max(c[3]);
                    if lo >= hi {
                        continue;
                    }
                    // Levels L with lo < L <= hi are the ones some corner is below.
                    let i0 = levels.partition_point(|&l| l <= lo);
                    let i1 = levels.partition_point(|&l| l <= hi);
                    for i in i0..i1 {
                        if !keep(i, x, y) {
                            continue;
                        }
                        let level = levels[i];
                        let ins = c.map(|q| q >= level);
                        let corner = [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)];
                        // Cyclic order top, right, bottom, left; each edge runs from the same
                        // corner in both cells that share it, so the crossing point agrees.
                        let edges = [(hid(x, y), 0, 1), (vid(x + 1, y), 1, 2), (hid(x, y + 1), 3, 2), (vid(x, y), 0, 3)];
                        let mut cross = [u64::MAX; 4];
                        for (k, &(id, a, b)) in edges.iter().enumerate() {
                            if ins[a] != ins[b] {
                                let t = ((level - c[a]) / (c[b] - c[a])).clamp(0.0, 1.0) as f64;
                                let (pa, pb) = (corner[a], corner[b]);
                                let key = id * nlu + i as u64;
                                pts.entry(key).or_insert((pa.0 as f64 + t * (pb.0 as f64 - pa.0 as f64), pa.1 as f64 + t * (pb.1 as f64 - pa.1 as f64)));
                                cross[k] = key;
                            }
                        }
                        let ks: Vec<usize> = (0..4).filter(|&k| cross[k] != u64::MAX).collect();
                        if ks.len() == 2 {
                            segs.push((cross[ks[0]], cross[ks[1]]));
                        } else if ks.len() == 4 {
                            // Corners alternate in/out. The two corners on the centre's side are
                            // joined through it; the other two are cut off.
                            let centre_in = c.iter().sum::<f32>() / 4.0 >= level;
                            if centre_in == ins[0] {
                                segs.push((cross[0], cross[1])); // cut off top-right
                                segs.push((cross[2], cross[3])); // cut off bottom-left
                            } else {
                                segs.push((cross[3], cross[0])); // cut off top-left
                                segs.push((cross[1], cross[2])); // cut off bottom-right
                            }
                        }
                    }
                }
            }
            (pts, segs)
        })
        .collect();
    let mut pts: HashMap<u64, P2> = HashMap::with_capacity(parts.iter().map(|p| p.0.len()).sum());
    let mut segs: Vec<(u64, u64)> = Vec::with_capacity(parts.iter().map(|p| p.1.len()).sum());
    for (p, sg) in parts {
        for (k, v) in p {
            pts.entry(k).or_insert(v);
        }
        segs.extend(sg);
    }
    for (key, line) in chain(&pts, &segs) {
        out[(key % nlu) as usize].push(line);
    }
    out
}

/// Joins segments that share a crossing point into polylines: open chains from
/// their free ends first, then the remaining closed loops. Each line comes with
/// the key of its first point.
fn chain(pts: &HashMap<u64, P2>, segs: &[(u64, u64)]) -> Vec<(u64, Polyline)> {
    let mut adj: HashMap<u64, [u32; 2]> = HashMap::with_capacity(pts.len());
    for (i, &(a, b)) in segs.iter().enumerate() {
        for n in [a, b] {
            let e = adj.entry(n).or_insert([u32::MAX; 2]);
            if e[0] == u32::MAX { e[0] = i as u32 } else { e[1] = i as u32 }
        }
    }
    let mut used = vec![false; segs.len()];
    let walk = |start: u64, first: usize, used: &mut Vec<bool>| -> (u64, Polyline) {
        let mut line = vec![pts[&start]];
        let (mut node, mut s) = (start, first);
        loop {
            used[s] = true;
            let (a, b) = segs[s];
            node = if a == node { b } else { a };
            line.push(pts[&node]);
            match adj[&node].iter().find(|&&j| j != u32::MAX && !used[j as usize]) {
                Some(&j) => s = j as usize,
                None => break,
            }
        }
        (start, Polyline { closed: node == start && line.len() > 3, pts: line })
    };
    let mut nodes: Vec<u64> = adj.keys().copied().collect();
    nodes.sort_unstable();
    let mut out = Vec::new();
    for n in nodes {
        let e = adj[&n];
        if e[1] == u32::MAX && !used[e[0] as usize] {
            out.push(walk(n, e[0] as usize, &mut used));
        }
    }
    for i in 0..segs.len() {
        if !used[i] {
            out.push(walk(segs[i].0, i, &mut used));
        }
    }
    out
}

/// Equal-area equalisation: each escaped sample (< `inside`) → its mid-rank
/// fraction in [0, 1) among the escaped samples; never-escaped samples → 2.
pub fn equalize(f: &[f32], inside: f32) -> Vec<f32> {
    let mut esc: Vec<f32> = f.iter().copied().filter(|&v| v < inside).collect();
    esc.sort_by(|a, b| a.total_cmp(b));
    let n = esc.len().max(1) as f64;
    f.par_iter()
        .map(|&v| {
            if !(v < inside) {
                return 2.0;
            }
            let lo = esc.partition_point(|&e| e < v);
            let hi = esc.partition_point(|&e| e <= v);
            ((lo + hi) as f64 * 0.5 / n) as f32
        })
        .collect()
}

/// What to draw and how finely.
#[derive(Clone, Copy, Debug)]
pub struct ContourStyle {
    /// Equal-area bands; levels 1..levels are drawn (a power of two keeps the
    /// thinning hierarchy regular).
    pub levels: usize,
    /// Smallest gap between neighbouring drawn lines, mm.
    pub min_gap: f64,
    /// Pieces shorter than this are dropped (dust), mm.
    pub min_len: f64,
    pub contours: bool,
    pub outline: bool,
}

/// The hierarchy step of level `i`: 2^(trailing zeros of i).
pub fn level_step(i: usize) -> usize { 1 << i.trailing_zeros().min(30) }

/// Contour polylines of an equalised field `g` (see [`equalize`]), in mm:
/// sample (x, y) maps to `origin + (x + 0.5, y + 0.5) · cell`. Returns
/// (contour lines with their level index 1..levels, outline).
pub fn plot_lines(g: &[f32], w: usize, h: usize, cell: f64, origin: P2, style: &ContourStyle) -> (Vec<(usize, Polyline)>, Vec<Polyline>) {
    let contours = if style.contours {
        contour_family(g, w, h, cell, origin, style.levels, style.min_gap, style.min_len, &|_, _, _| true)
    } else {
        Vec::new()
    };
    let to_mm = |l: Polyline| grid_to_mm(l, cell, origin);
    let outline = if style.outline { contour(g, w, h, 1.5, &|_, _| true).into_iter().map(to_mm).filter(|l| l.length() >= style.min_len).collect() } else { Vec::new() };
    (contours, outline)
}

/// A grid-coordinate polyline in mm: sample (x, y) sits at `origin + (x + 0.5, y + 0.5) · cell`.
pub fn grid_to_mm(l: Polyline, cell: f64, origin: P2) -> Polyline {
    Polyline { closed: l.closed, pts: l.pts.iter().map(|&(x, y)| (origin.0 + (x + 0.5) * cell, origin.1 + (y + 0.5) * cell)).collect() }
}

/// One family of contour lines: `g` (normalised to [0, 1)) at levels i/levels,
/// i = 1..levels, in mm. Level i is drawn in a cell only where the gap to the
/// next line of its hierarchy step (2^trailing_zeros(i) levels) is at least
/// `min_gap` mm (0 = no thinning), and only where `mask(i, x, y)` holds (the
/// family's region, or no wrap-around seam for an angle). Pieces shorter than
/// `min_len` mm are dropped.
#[allow(clippy::too_many_arguments)]
pub fn contour_family(g: &[f32], w: usize, h: usize, cell: f64, origin: P2, levels: usize, min_gap: f64, min_len: f64,
                      mask: &(dyn Fn(usize, usize, usize) -> bool + Sync)) -> Vec<(usize, Polyline)> {
    let n = levels.max(2);
    let delta = 1.0 / n as f64;
    let values: Vec<f32> = (1..n).map(|i| (i as f64 * delta) as f32).collect();
    let keep = |k: usize, x: usize, y: usize| {
        let i = k + 1;
        if !mask(i, x, y) {
            return false;
        }
        if min_gap <= 0.0 {
            return true;
        }
        let (a, b, c, d) = (g[y * w + x] as f64, g[y * w + x + 1] as f64, g[(y + 1) * w + x + 1] as f64, g[(y + 1) * w + x] as f64);
        let gx = 0.5 * ((b - a) + (c - d));
        let gy = 0.5 * ((d - a) + (c - b));
        let grad = (gx * gx + gy * gy).sqrt();
        grad <= 0.0 || level_step(i) as f64 * delta / grad * cell >= min_gap
    };
    contour_levels(g, w, h, &values, &keep)
        .into_iter()
        .enumerate()
        .flat_map(|(k, ls)| ls.into_iter().map(move |l| (k + 1, l)))
        .map(|(i, l)| (i, grid_to_mm(l, cell, origin)))
        .filter(|(_, l)| l.length() >= min_len)
        .collect()
}

/// Splits `line` where `band` (per point) changes, at the midpoint between the
/// two points; a closed loop whose first and last runs share a band is rejoined.
/// Runs shorter than `min_run` (mm) take the band of their longer neighbour
/// first, so a noisy band field does not chop a line into specks.
pub fn split_by_band(line: &Polyline, band: &dyn Fn(P2) -> usize, min_run: f64) -> Vec<(usize, Polyline)> {
    if line.pts.len() < 2 {
        return Vec::new();
    }
    let mut bands: Vec<usize> = line.pts.iter().map(|&p| band(p)).collect();
    absorb_short_runs(&line.pts, &mut bands, min_run);
    let mut out: Vec<(usize, Polyline)> = Vec::new();
    let mut cur = vec![line.pts[0]];
    let mut b = bands[0];
    for i in 1..line.pts.len() {
        if bands[i] != b {
            let mid = mul(add(line.pts[i - 1], line.pts[i]), 0.5);
            cur.push(mid);
            out.push((b, Polyline { pts: std::mem::take(&mut cur), closed: false }));
            cur.push(mid);
            b = bands[i];
        }
        cur.push(line.pts[i]);
    }
    if out.is_empty() {
        return vec![(b, line.clone())];
    }
    out.push((b, Polyline { pts: cur, closed: false }));
    if line.closed && out[0].0 == out[out.len() - 1].0 {
        let (_, first) = out.remove(0);
        let last = out.last_mut().unwrap();
        last.1.pts.extend_from_slice(&first.pts[1..]);
    }
    out
}

/// Reassigns runs of equal `bands` (per point) shorter than `min_run` along
/// `pts` to the band of their longer neighbouring run, shortest first.
fn absorb_short_runs(pts: &[P2], bands: &mut [usize], min_run: f64) {
    if min_run <= 0.0 {
        return;
    }
    loop {
        // (band, first point, last point, length): a run's length is measured
        // between the midpoints where it starts and ends.
        let mut runs: Vec<(usize, usize, usize, f64)> = Vec::new();
        for i in 0..pts.len() {
            let half = |a: usize, b: usize| 0.5 * dist(pts[a], pts[b]);
            match runs.last_mut() {
                Some(r) if r.0 == bands[i] => {
                    r.3 += half(i - 1, i) * 2.0;
                    r.2 = i;
                }
                _ => {
                    if let Some(r) = runs.last_mut() {
                        r.3 += half(i - 1, i);
                    }
                    let start = if i > 0 { half(i - 1, i) } else { 0.0 };
                    runs.push((bands[i], i, i, start));
                }
            }
        }
        if runs.len() < 2 {
            return;
        }
        let Some((k, _)) = runs.iter().enumerate().filter(|(_, r)| r.3 < min_run).min_by(|a, b| a.1 .3.total_cmp(&b.1 .3)) else { return };
        let prev = (k > 0).then(|| runs[k - 1]);
        let next = runs.get(k + 1).copied();
        let to = match (prev, next) {
            (Some(p), Some(n)) => if p.3 >= n.3 { p.0 } else { n.0 },
            (Some(p), None) => p.0,
            (None, Some(n)) => n.0,
            (None, None) => return,
        };
        for b in &mut bands[runs[k].1..=runs[k].2] {
            *b = to;
        }
    }
}

// ---------------------------------------------------------------- Bézier fit

fn bez(c: &Cubic, t: f64) -> P2 {
    let s = 1.0 - t;
    let (b0, b1, b2, b3) = (s * s * s, 3.0 * s * s * t, 3.0 * s * t * t, t * t * t);
    (b0 * c[0].0 + b1 * c[1].0 + b2 * c[2].0 + b3 * c[3].0, b0 * c[0].1 + b1 * c[1].1 + b2 * c[2].1 + b3 * c[3].1)
}

fn bez_d1(c: &Cubic, t: f64) -> P2 {
    let s = 1.0 - t;
    let d = [mul(sub(c[1], c[0]), 3.0), mul(sub(c[2], c[1]), 3.0), mul(sub(c[3], c[2]), 3.0)];
    add(add(mul(d[0], s * s), mul(d[1], 2.0 * s * t)), mul(d[2], t * t))
}

fn bez_d2(c: &Cubic, t: f64) -> P2 {
    let a = mul(add(sub(c[2], mul(c[1], 2.0)), c[0]), 6.0);
    let b = mul(add(sub(c[3], mul(c[2], 2.0)), c[1]), 6.0);
    add(mul(a, 1.0 - t), mul(b, t))
}

/// Cubic Béziers through `line` within `tol` (max distance at the sample points).
pub fn fit_polyline(line: &Polyline, tol: f64) -> CurvePath {
    let mut p: Vec<P2> = Vec::with_capacity(line.pts.len());
    for &q in &line.pts {
        if p.last().is_none_or(|&l| dist(l, q) > 1e-9) {
            p.push(q);
        }
    }
    let closed = line.closed && p.len() >= 4 && dist(p[0], p[p.len() - 1]) < 1e-9;
    let mut out = Vec::new();
    let n = p.len();
    if n < 2 {
        return CurvePath { curves: out, closed: false };
    }
    if closed {
        // A loop's chord is zero, so fit two halves with a shared tangent at the start.
        let t0 = unit(sub(p[1], p[n - 2])).unwrap_or((1.0, 0.0));
        let m = n / 2;
        let tm = mid_tangent(&p, m);
        fit_cubic(&p[..=m], t0, tm, tol, &mut out, 0);
        fit_cubic(&p[m..], mul(tm, -1.0), mul(t0, -1.0), tol, &mut out, 0);
    } else {
        let t1 = unit(sub(p[1], p[0])).unwrap_or((1.0, 0.0));
        let t2 = unit(sub(p[n - 2], p[n - 1])).unwrap_or((-1.0, 0.0));
        fit_cubic(&p, t1, t2, tol, &mut out, 0);
    }
    CurvePath { curves: out, closed }
}

/// The tangent at interior sample `k`, pointing back along the line.
fn mid_tangent(p: &[P2], k: usize) -> P2 {
    unit(sub(p[k - 1], p[k + 1])).or_else(|| unit(sub(p[k - 1], p[k]))).unwrap_or((-1.0, 0.0))
}

fn fit_cubic(d: &[P2], t1: P2, t2: P2, tol: f64, out: &mut Vec<Cubic>, depth: u32) {
    let n = d.len();
    let (first, last) = (d[0], d[n - 1]);
    if n == 2 || depth > 40 {
        let l = dist(first, last) / 3.0;
        out.push([first, add(first, mul(t1, l)), add(last, mul(t2, l)), last]);
        return;
    }
    let mut u = chord_params(d);
    let mut c = generate(d, &u, t1, t2);
    let (mut err, mut split) = max_error(d, &c, &u);
    if err <= tol * tol {
        out.push(c);
        return;
    }
    if err <= 16.0 * tol * tol {
        for _ in 0..4 {
            let Some(u2) = reparameterize(d, &u, &c) else { break };
            u = u2;
            c = generate(d, &u, t1, t2);
            (err, split) = max_error(d, &c, &u);
            if err <= tol * tol {
                out.push(c);
                return;
            }
        }
    }
    let tc = mid_tangent(d, split);
    fit_cubic(&d[..=split], t1, tc, tol, out, depth + 1);
    fit_cubic(&d[split..], mul(tc, -1.0), t2, tol, out, depth + 1);
}

fn chord_params(d: &[P2]) -> Vec<f64> {
    let mut u = Vec::with_capacity(d.len());
    u.push(0.0);
    for i in 1..d.len() {
        u.push(u[i - 1] + dist(d[i], d[i - 1]));
    }
    let total = u[d.len() - 1].max(1e-12);
    u.iter_mut().for_each(|x| *x /= total);
    u
}

/// Least-squares handle lengths along the fixed end tangents.
fn generate(d: &[P2], u: &[f64], t1: P2, t2: P2) -> Cubic {
    let (first, last) = (d[0], d[d.len() - 1]);
    let (mut c00, mut c01, mut c11, mut x0, mut x1) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (i, &t) in u.iter().enumerate() {
        let s = 1.0 - t;
        let (b0, b1, b2, b3) = (s * s * s, 3.0 * s * s * t, 3.0 * s * t * t, t * t * t);
        let (a1, a2) = (mul(t1, b1), mul(t2, b2));
        c00 += dot(a1, a1);
        c01 += dot(a1, a2);
        c11 += dot(a2, a2);
        let tmp = sub(d[i], add(mul(first, b0 + b1), mul(last, b2 + b3)));
        x0 += dot(a1, tmp);
        x1 += dot(a2, tmp);
    }
    let det = c00 * c11 - c01 * c01;
    let (mut al, mut ar) = if det.abs() > 1e-12 { ((x0 * c11 - x1 * c01) / det, (c00 * x1 - c01 * x0) / det) } else { (0.0, 0.0) };
    let seg = dist(first, last);
    let arc: f64 = d.windows(2).map(|w| dist(w[0], w[1])).sum();
    // Non-positive or overlong handles mean the fit failed; a third of the chord
    // (or of the arc for near-closed pieces) gives a sane curve that the error
    // check then splits.
    if al < 1e-6 * arc || ar < 1e-6 * arc || al > arc || ar > arc {
        let l = if seg > 1e-6 * arc { seg } else { arc } / 3.0;
        al = l;
        ar = l;
    }
    [first, add(first, mul(t1, al)), add(last, mul(t2, ar)), last]
}

/// (largest squared distance, its index in 1..n-1).
fn max_error(d: &[P2], c: &Cubic, u: &[f64]) -> (f64, usize) {
    let mut best = (0.0, d.len() / 2);
    for i in 1..d.len() - 1 {
        let e = dot(sub(bez(c, u[i]), d[i]), sub(bez(c, u[i]), d[i]));
        if e >= best.0 {
            best = (e, i);
        }
    }
    best
}

/// One Newton step per sample towards its nearest curve point; None if the
/// parameters stop increasing (the step would fold the curve).
fn reparameterize(d: &[P2], u: &[f64], c: &Cubic) -> Option<Vec<f64>> {
    let mut v: Vec<f64> = Vec::with_capacity(u.len());
    for (i, &t) in u.iter().enumerate() {
        let (q, q1, q2) = (bez(c, t), bez_d1(c, t), bez_d2(c, t));
        let diff = sub(q, d[i]);
        let den = dot(q1, q1) + dot(diff, q2);
        let nt = if den.abs() > 1e-12 { (t - dot(diff, q1) / den).clamp(0.0, 1.0) } else { t };
        if i > 0 && nt <= v[i - 1] {
            return None;
        }
        v.push(nt);
    }
    Some(v)
}

/// Straight segments (end points after `c[0]`) within `tol` of the curve.
pub fn flatten(c: &Cubic, tol: f64, out: &mut Vec<P2>) {
    fn rec(c: &Cubic, tol: f64, depth: u32, out: &mut Vec<P2>) {
        let chord = sub(c[3], c[0]);
        let len = dot(chord, chord).sqrt();
        let off = |p: P2| if len > 1e-12 { ((p.0 - c[0].0) * chord.1 - (p.1 - c[0].1) * chord.0).abs() / len } else { dist(p, c[0]) };
        // The curve lies within 3/4 of the larger control-point offset of its chord.
        if depth >= 16 || 0.75 * off(c[1]).max(off(c[2])) <= tol {
            out.push(c[3]);
            return;
        }
        let m = |a: P2, b: P2| mul(add(a, b), 0.5);
        let (p01, p12, p23) = (m(c[0], c[1]), m(c[1], c[2]), m(c[2], c[3]));
        let (p012, p123) = (m(p01, p12), m(p12, p23));
        let mid = m(p012, p123);
        rec(&[c[0], p01, p012, mid], tol, depth + 1, out);
        rec(&[mid, p123, p23, c[3]], tol, depth + 1, out);
    }
    rec(c, tol, 0, out);
}

// ------------------------------------------------------------------ ordering

/// Reorders `lines` to cut pen-up travel, starting from `start`: greedy nearest
/// path end; an open path may be drawn backwards, a closed loop is rotated to
/// start at its vertex nearest the pen. Returns (ordered lines, travel mm).
pub fn order_paths(lines: Vec<Polyline>, start: P2) -> (Vec<Polyline>, f64) {
    if lines.is_empty() {
        return (lines, 0.0);
    }
    // Spatial hash of path ends (a loop's ends are its first point).
    let (mut lo, mut hi) = ((f64::MAX, f64::MAX), (f64::MIN, f64::MIN));
    for l in &lines {
        for &p in &l.pts {
            lo = (lo.0.min(p.0), lo.1.min(p.1));
            hi = (hi.0.max(p.0), hi.1.max(p.1));
        }
    }
    let span = (hi.0 - lo.0).max(hi.1 - lo.1).max(1e-6);
    let cells = ((lines.len() as f64).sqrt().ceil() as i64).clamp(1, 512);
    let cs = span / cells as f64;
    let key = |p: P2| (((p.0 - lo.0) / cs).floor() as i64, ((p.1 - lo.1) / cs).floor() as i64);
    let mut grid: HashMap<(i64, i64), Vec<(u32, bool)>> = HashMap::new();
    for (i, l) in lines.iter().enumerate() {
        grid.entry(key(l.pts[0])).or_default().push((i as u32, false));
        if !l.closed {
            grid.entry(key(*l.pts.last().unwrap())).or_default().push((i as u32, true));
        }
    }
    let mut used = vec![false; lines.len()];
    let mut lines: Vec<Option<Polyline>> = lines.into_iter().map(Some).collect();
    let mut out = Vec::with_capacity(lines.len());
    let (mut pen, mut travel) = (start, 0.0);
    for _ in 0..lines.len() {
        // A pen outside the grid searches from the nearest grid cell: distances from
        // the pen are never shorter than from its projection, so the ring bound holds.
        let (kx, ky) = key(pen);
        let (kx, ky) = (kx.clamp(0, cells), ky.clamp(0, cells));
        let mut best: Option<(f64, u32, bool)> = None;
        for r in 0i64.. {
            // Every point in ring r+1 or beyond is at least r·cs from the pen.
            if best.is_some_and(|b| b.0 < (r as f64 - 1.0) * cs) || r > 2 * cells + 2 {
                break;
            }
            for gx in kx - r..=kx + r {
                for gy in ky - r..=ky + r {
                    if (gx - kx).abs() != r && (gy - ky).abs() != r {
                        continue;
                    }
                    let Some(bucket) = grid.get_mut(&(gx, gy)) else { continue };
                    bucket.retain(|&(i, _)| !used[i as usize]);
                    for &(i, at_end) in bucket.iter() {
                        let l = lines[i as usize].as_ref().unwrap();
                        let p = if at_end { *l.pts.last().unwrap() } else { l.pts[0] };
                        let dd = dist(pen, p);
                        if best.is_none_or(|b| dd < b.0) {
                            best = Some((dd, i, at_end));
                        }
                    }
                }
            }
        }
        let Some((_, i, at_end)) = best else { break };
        used[i as usize] = true;
        let mut l = lines[i as usize].take().unwrap();
        if at_end {
            l.pts.reverse();
        } else if l.closed {
            let k = (0..l.pts.len() - 1).min_by(|&a, &b| dist(pen, l.pts[a]).total_cmp(&dist(pen, l.pts[b]))).unwrap_or(0);
            if k > 0 {
                let mut pts: Vec<P2> = l.pts[k..l.pts.len() - 1].to_vec();
                pts.extend_from_slice(&l.pts[..=k]);
                l.pts = pts;
            }
        }
        travel += dist(pen, l.pts[0]);
        pen = *l.pts.last().unwrap();
        out.push(l);
    }
    (out, travel)
}

// -------------------------------------------------------------------- output

/// A named group of paths, drawn with one pen; `color` is the SVG stroke.
pub struct Layer {
    pub name: String,
    pub color: String,
    pub paths: Vec<CurvePath>,
}

/// The pen (0..pens) for contour level `i` of `levels`: equal runs of levels,
/// pen 0 the outermost (fastest escaping) band.
pub fn level_pen(i: usize, levels: usize, pens: usize) -> usize { (i * pens / levels.max(1)).min(pens.saturating_sub(1)) }

/// The paper, in mm (SVG coordinates: origin top-left, y down).
#[derive(Clone, Copy, Debug)]
pub struct Sheet {
    pub w: f64,
    pub h: f64,
    pub pen: f64,
}

fn f3(x: f64) -> String {
    let s = format!("{x:.3}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s == "-0" || s.is_empty() { "0".into() } else { s.into() }
}

/// An SVG in mm, one Inkscape layer per [`Layer`], paths as `M … C … [Z]`.
pub fn to_svg(sheet: &Sheet, layers: &[Layer], comment: &str) -> String {
    let mut s = String::new();
    let _ = writeln!(s, r#"<?xml version="1.0" encoding="UTF-8"?>"#);
    let _ = writeln!(s, "<!-- {} -->", comment.replace("--", "–"));
    let _ = writeln!(
        s,
        r#"<svg xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" width="{w}mm" height="{h}mm" viewBox="0 0 {w} {h}">"#,
        w = f3(sheet.w),
        h = f3(sheet.h)
    );
    let _ = writeln!(s, r#"<rect width="{}" height="{}" fill="white"/>"#, f3(sheet.w), f3(sheet.h));
    for (k, layer) in layers.iter().enumerate() {
        let _ = writeln!(
            s,
            r#"<g inkscape:groupmode="layer" id="layer{}" inkscape:label="{} {}" fill="none" stroke="{}" stroke-width="{}" stroke-linecap="round" stroke-linejoin="round">"#,
            k + 1,
            k + 1,
            layer.name,
            layer.color,
            f3(sheet.pen)
        );
        for p in &layer.paths {
            let Some(first) = p.curves.first() else { continue };
            let mut d = format!("M{} {}", f3(first[0].0), f3(first[0].1));
            for c in &p.curves {
                let _ = write!(d, "C{} {} {} {} {} {}", f3(c[1].0), f3(c[1].1), f3(c[2].0), f3(c[2].1), f3(c[3].0), f3(c[3].1));
            }
            if p.closed {
                d.push('Z');
            }
            let _ = writeln!(s, r#"<path d="{d}"/>"#);
        }
        let _ = writeln!(s, "</g>");
    }
    s.push_str("</svg>\n");
    s
}

/// Marlin G-code settings.
#[derive(Clone, Debug)]
pub struct GcodeOpts {
    /// Bed position of the paper's lower-left corner (flip_y) or upper-left.
    pub origin: P2,
    /// Bed Y grows away from the operator, SVG y grows down the page: flip so
    /// the plot reads the same as the SVG seen from the front.
    pub flip_y: bool,
    /// Drawing and travel speed, mm/min.
    pub feed: f64,
    pub travel: f64,
    pub pen_up: Vec<String>,
    pub pen_down: Vec<String>,
    /// `G5` cubic moves (Marlin BEZIER_CURVE_SUPPORT) instead of `G1` segments.
    pub bezier: bool,
    /// Max deviation of the `G1` segments from the curves, mm.
    pub flat_tol: f64,
    /// `G28 X Y` before plotting.
    pub home: bool,
    /// Pause (`pen_change`) between layers, to change pens.
    pub pause_between_layers: bool,
    /// Lines after `G21`/`G90` (e.g. a Z offset) and before the final move.
    pub header: Vec<String>,
    pub footer: Vec<String>,
    /// Travel shorter than `.0` mm lifts the pen with `.1` instead of `pen_up`.
    pub short_lift: Option<(f64, Vec<String>)>,
    /// The pen-change pause; `{k}` and `{name}` become the layer number and name.
    pub pen_change: Vec<String>,
    /// Also pause before the first layer (to put the first pen in).
    pub pause_first: bool,
    /// Pause (the `pen_change` block, name "refill …") after every this many mm drawn in a layer.
    pub refill_mm: Option<f64>,
}

impl GcodeOpts {
    /// Plain Marlin defaults: absolute mm, Z pen, `M0` pause between pens.
    pub fn marlin(pen_up: Vec<String>, pen_down: Vec<String>) -> Self {
        GcodeOpts {
            origin: (0.0, 0.0), flip_y: true, feed: 3000.0, travel: 6000.0, pen_up, pen_down, bezier: false, flat_tol: 0.02,
            home: false, pause_between_layers: false, header: Vec::new(), footer: Vec::new(), short_lift: None,
            pen_change: vec!["M0 Change pen for layer {k} {name}".into()], pause_first: false, refill_mm: None,
        }
    }
}

/// Lengths and an estimated time for a plot.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PlotStats {
    pub paths: usize,
    pub curves: usize,
    pub g_lines: usize,
    pub draw_mm: f64,
    pub travel_mm: f64,
    pub minutes: f64,
}

/// Marlin G-code for the layers, and its statistics. Pen state is always
/// explicit: every path is `G0` to its start, pen down, moves, pen up.
pub fn to_gcode(sheet: &Sheet, layers: &[Layer], o: &GcodeOpts, comment: &str) -> (String, PlotStats) {
    let sy = if o.flip_y { -1.0 } else { 1.0 };
    let tx = |p: P2| (p.0 + o.origin.0, (if o.flip_y { sheet.h - p.1 } else { p.1 }) + o.origin.1);
    let mut body = String::new();
    let mut st = PlotStats::default();
    let mut pen = o.origin;
    let mut lifts = 0usize;
    let pause = |body: &mut String, k: usize, name: &str| {
        for c in &o.pen_change {
            let _ = writeln!(body, "{}", c.replace("{k}", &k.to_string()).replace("{name}", name));
        }
    };
    let mut down = false;
    for (k, layer) in layers.iter().enumerate() {
        let _ = writeln!(body, "; ---- layer {} {} ({} paths)", k + 1, layer.name, layer.paths.len());
        if o.pause_between_layers && !layer.paths.is_empty() && (k > 0 || o.pause_first) {
            pause(&mut body, k + 1, &layer.name);
        }
        let mut since_refill = 0.0;
        let mut refills = 0;
        for p in &layer.paths {
            let Some(first) = p.curves.first() else { continue };
            let s = tx(first[0]);
            let hop = dist(pen, s);
            if down {
                let cmds = match &o.short_lift { Some((d, c)) if hop < *d => c, _ => &o.pen_up };
                for c in cmds {
                    let _ = writeln!(body, "{c}");
                }
                lifts += 1;
                down = false;
            }
            if o.refill_mm.is_some_and(|r| since_refill >= r) {
                refills += 1;
                pause(&mut body, k + 1, &format!("refill {} #{refills}", layer.name));
                since_refill = 0.0;
            }
            let drawn_before = st.draw_mm;
            st.travel_mm += hop;
            let _ = writeln!(body, "G0 X{} Y{} F{}", f3(s.0), f3(s.1), f3(o.travel));
            for c in &o.pen_down {
                let _ = writeln!(body, "{c}");
            }
            down = true;
            let mut f_set = false;
            let mut cur = s;
            for c in &p.curves {
                if o.bezier {
                    let e = tx(c[3]);
                    let _ = write!(
                        body,
                        "G5 I{} J{} P{} Q{} X{} Y{}",
                        f3(c[1].0 - c[0].0),
                        f3(sy * (c[1].1 - c[0].1)),
                        f3(c[2].0 - c[3].0),
                        f3(sy * (c[2].1 - c[3].1)),
                        f3(e.0),
                        f3(e.1)
                    );
                    let mut flat = Vec::new();
                    flatten(c, o.flat_tol, &mut flat);
                    let mut q = cur;
                    for f in flat {
                        let f = tx(f);
                        st.draw_mm += dist(q, f);
                        q = f;
                    }
                    cur = e;
                    if !f_set {
                        let _ = write!(body, " F{}", f3(o.feed));
                        f_set = true;
                    }
                    body.push('\n');
                    st.g_lines += 1;
                } else {
                    let mut flat = Vec::new();
                    flatten(c, o.flat_tol, &mut flat);
                    for f in flat {
                        let f = tx(f);
                        if dist(cur, f) < 1e-4 {
                            continue;
                        }
                        st.draw_mm += dist(cur, f);
                        cur = f;
                        let _ = write!(body, "G1 X{} Y{}", f3(f.0), f3(f.1));
                        if !f_set {
                            let _ = write!(body, " F{}", f3(o.feed));
                            f_set = true;
                        }
                        body.push('\n');
                        st.g_lines += 1;
                    }
                }
                st.curves += 1;
            }
            since_refill += st.draw_mm - drawn_before;
            st.paths += 1;
            pen = cur;
        }
        if down {
            for c in &o.pen_up {
                let _ = writeln!(body, "{c}");
            }
            lifts += 1;
            down = false;
        }
    }
    let end = o.origin;
    st.travel_mm += dist(pen, end);
    // Pen moves at ~0.3 s each way on a typical Z or servo lift.
    st.minutes = st.draw_mm / o.feed.max(1.0) + st.travel_mm / o.travel.max(1.0) + lifts as f64 * 0.6 / 60.0;
    let mut s = String::new();
    for line in comment.lines() {
        let _ = writeln!(s, "; {line}");
    }
    let _ = writeln!(s, "; paper {}x{} mm, pen {} mm, paper corner at bed X{} Y{}{}", f3(sheet.w), f3(sheet.h), f3(sheet.pen), f3(o.origin.0), f3(o.origin.1), if o.flip_y { " (front-left)" } else { "" });
    let _ = writeln!(
        s,
        "; {} paths, {} {} moves, draw {:.1} m, travel {:.1} m, about {:.0} min at F{}/F{}",
        st.paths,
        st.g_lines,
        if o.bezier { "G5" } else { "G1" },
        st.draw_mm / 1000.0,
        st.travel_mm / 1000.0,
        st.minutes,
        f3(o.feed),
        f3(o.travel)
    );
    if o.bezier {
        let _ = writeln!(s, "; G5 needs Marlin built with BEZIER_CURVE_SUPPORT; otherwise use the G1 file");
    }
    let _ = writeln!(s, "; pen down: {} | pen up: {}", o.pen_down.join(" ; "), o.pen_up.join(" ; "));
    let _ = writeln!(s, "G21 ; millimetres\nG90 ; absolute positions");
    for c in &o.header {
        let _ = writeln!(s, "{c}");
    }
    for c in &o.pen_up {
        let _ = writeln!(s, "{c}");
    }
    if o.home {
        let _ = writeln!(s, "G28 X Y");
    }
    s.push_str(&body);
    let _ = writeln!(s, "; ---- end");
    for c in &o.footer {
        let _ = writeln!(s, "{c}");
    }
    let _ = writeln!(s, "G0 X{} Y{} F{}\nM400", f3(end.0), f3(end.1), f3(o.travel));
    (s, st)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn radial(n: usize, c: f64) -> Vec<f32> {
        (0..n * n).map(|i| (((i % n) as f64 - c).hypot((i / n) as f64 - c)) as f32).collect()
    }

    #[test]
    fn a_radial_field_contours_to_one_closed_circle() {
        let (n, c, r) = (64, 31.5, 20.0);
        let f = radial(n, c);
        let lines = contour(&f, n, n, r as f32, &|_, _| true);
        assert_eq!(lines.len(), 1);
        let l = &lines[0];
        assert!(l.closed && l.pts.first() == l.pts.last());
        for &(x, y) in &l.pts {
            assert!(((x - c).hypot(y - c) - r).abs() < 0.05, "crossings interpolate onto the circle");
        }
        // Dropping cells where the loop crosses once (top half of column 40) opens
        // it into one piece; the whole column would cut it twice.
        let cut = contour(&f, n, n, r as f32, &|x, y| !(x == 40 && y < 31));
        assert_eq!(contour(&f, n, n, r as f32, &|x, _| x != 40).len(), 2);
        assert_eq!(cut.len(), 1);
        assert!(!cut[0].closed);
    }

    #[test]
    fn one_pass_over_many_levels_matches_one_level_at_a_time() {
        let n = 80;
        let f: Vec<f32> = (0..n * n).map(|i| { let (x, y) = ((i % n) as f32 * 0.13, (i / n) as f32 * 0.11); (x.sin() * y.cos() + 0.3 * (x * y).sin()) as f32 }).collect();
        let levels: Vec<f32> = (0..15).map(|k| -1.2 + 0.17 * k as f32).collect();
        let keep = |i: usize, x: usize, _y: usize| (x + i) % 7 != 0;
        let all = contour_levels(&f, n, n, &levels, &keep);
        for (i, &l) in levels.iter().enumerate() {
            let one = contour(&f, n, n, l, &|x, y| keep(i, x, y));
            assert_eq!(all[i], one, "level {i}");
        }
        assert!(all.iter().map(|v| v.len()).sum::<usize>() > 30);
    }

    #[test]
    fn saddles_never_give_a_point_three_segments() {
        // A checkerboard is all saddles; every chain must still be a simple path.
        let n = 9;
        let f: Vec<f32> = (0..n * n).map(|i| if (i % n + i / n) % 2 == 0 { 1.0 } else { 0.0 }).collect();
        let lines = contour(&f, n, n, 0.5, &|_, _| true);
        let total: usize = lines.iter().map(|l| l.pts.len() - 1).sum();
        assert_eq!(total, 2 * (n - 1) * (n - 1), "two segments per saddle cell, each used once");
    }

    #[test]
    fn the_cubic_fit_stays_within_tolerance_of_a_circle() {
        let pts: Vec<P2> = (0..=400).map(|i| { let a = i as f64 / 400.0 * std::f64::consts::TAU; (50.0 + 30.0 * a.cos(), 50.0 + 30.0 * a.sin()) }).collect();
        let mut line = Polyline { pts, closed: true };
        *line.pts.last_mut().unwrap() = line.pts[0];
        let tol = 0.05;
        let cp = fit_polyline(&line, tol);
        assert!(cp.closed && cp.curves.len() >= 4 && cp.curves.len() < 40, "{} curves", cp.curves.len());
        for c in &cp.curves {
            for k in 0..=20 {
                let p = bez(c, k as f64 / 20.0);
                assert!(((p.0 - 50.0).hypot(p.1 - 50.0) - 30.0).abs() < 3.0 * tol);
            }
        }
        for w in cp.curves.windows(2) {
            assert_eq!(w[0][3], w[1][0], "curves join");
        }
        // An open straight line is one cubic.
        let st = fit_polyline(&Polyline { pts: (0..10).map(|i| (i as f64, 2.0 * i as f64)).collect(), closed: false }, 0.01);
        assert_eq!(st.curves.len(), 1);
    }

    #[test]
    fn ordering_reverses_and_rotates_to_the_nearest_end() {
        let a = Polyline { pts: vec![(10.0, 0.0), (1.0, 0.0)], closed: false };
        let b = Polyline { pts: vec![(20.0, 0.0), (20.0, 5.0), (25.0, 5.0), (20.0, 0.0)], closed: true };
        let (o, travel) = order_paths(vec![b.clone(), a], (0.0, 0.0));
        assert_eq!(o[0].pts, vec![(1.0, 0.0), (10.0, 0.0)], "open path drawn backwards");
        assert_eq!(o[1].pts[0], (20.0, 0.0));
        assert!((travel - (1.0 + 10.0)).abs() < 1e-9);
        // A loop entered from near its second vertex starts there and still closes.
        let (o, _) = order_paths(vec![b], (20.0, 6.0));
        assert_eq!(o[0].pts.first(), Some(&(20.0, 5.0)));
        assert_eq!(o[0].pts.first(), o[0].pts.last());
        assert_eq!(o[0].pts.len(), 4);
    }

    #[test]
    fn crowded_levels_thin_hierarchically() {
        assert_eq!((1..9).map(level_step).collect::<Vec<_>>(), vec![1, 2, 1, 4, 1, 2, 1, 8]);
        // A field that is shallow on the left and steep on the right: the right
        // half keeps only the coarser levels.
        let (w, h) = (200, 20);
        let f: Vec<f32> = (0..w * h).map(|i| { let x = (i % w) as f64; (if x < 100.0 { x * 0.002 } else { 0.2 + (x - 100.0) * 0.008 }) as f32 }).collect();
        let style = ContourStyle { levels: 64, min_gap: 1.5, min_len: 0.0, contours: true, outline: false };
        let (lines, _) = plot_lines(&f, w, h, 0.2, (0.0, 0.0), &style);
        let xs = |lo: f64, hi: f64| lines.iter().filter(|(_, l)| (lo..hi).contains(&l.pts[0].0)).count();
        // left: 1/64 per 7.8 px = 1.56 mm ≥ 1.5 → every level; right: every 4th.
        assert_eq!(xs(0.0, 20.0), 12, "levels 1..12 cross x < 100 px");
        let right = xs(20.0, 40.0);
        assert!((10..=14).contains(&right), "{right} lines: 51 levels / 4");
    }

    #[test]
    fn marlin_output_matches_the_curves() {
        let sheet = Sheet { w: 100.0, h: 100.0, pen: 0.3 };
        let c: Cubic = [(10.0, 10.0), (20.0, 10.0), (30.0, 20.0), (30.0, 30.0)];
        let layers = [Layer { name: "x".into(), color: "black".into(), paths: vec![CurvePath { curves: vec![c], closed: false }] }];
        let mut o = GcodeOpts { origin: (5.0, 0.0), bezier: true, flat_tol: 0.01, ..GcodeOpts::marlin(vec!["G0 Z2".into()], vec!["G1 Z0".into()]) };
        let (g5, st) = to_gcode(&sheet, &layers, &o, "test");
        assert!(g5.contains("G0 X15 Y90 F6000\nG1 Z0\nG5 I10 J0 P0 Q10 X35 Y70 F3000\nG0 Z2\n"), "{g5}");
        o.bezier = false;
        let (g1, st1) = to_gcode(&sheet, &layers, &o, "test");
        assert!((st.draw_mm - st1.draw_mm).abs() < 1e-9 && st1.draw_mm > 28.2 && st1.draw_mm < 32.0);
        let last = g1.lines().filter(|l| l.starts_with("G1 X")).last().unwrap();
        assert_eq!(last, "G1 X35 Y70");
        let svg = to_svg(&sheet, &layers, "t");
        // Carl's machine: short hops lift 2.5 mm, long ones 7 mm, and every pen
        // (the first too) is put in at the pause; a refill pause after 10 mm drawn.
        let two = [Layer { name: "a".into(), color: "black".into(), paths: vec![
            CurvePath { curves: vec![[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (12.0, 0.0)]], closed: false },
            CurvePath { curves: vec![[(13.0, 0.0), (14.0, 0.0), (15.0, 0.0), (16.0, 0.0)]], closed: false },
            CurvePath { curves: vec![[(40.0, 0.0), (41.0, 0.0), (42.0, 0.0), (43.0, 0.0)]], closed: false },
        ] }];
        let c = GcodeOpts { flip_y: false, header: vec!["M851 Z-1.2".into()], short_lift: Some((5.0, vec!["G1 Z2.5".into()])),
            pen_change: vec!["M117 {k} {name}".into(), "M226 P121".into()], pause_first: true, pause_between_layers: true, refill_mm: Some(10.0),
            ..GcodeOpts::marlin(vec!["G1 Z7".into()], vec!["G1 Z0".into()]) };
        let (g, st) = to_gcode(&sheet, &two, &c, "t");
        let lifts: Vec<&str> = g.lines().filter(|l| l.starts_with("G1 Z2.5") || l.starts_with("G1 Z7") || l.starts_with("M117")).collect();
        // Start lift, pen in, hop of 1 mm (low lift), refill (12 mm drawn >= 10), long hop (high lift), end of layer.
        assert_eq!(lifts, vec!["G1 Z7", "M117 1 a", "G1 Z2.5", "M117 1 refill a #1", "G1 Z7", "G1 Z7"], "{g}");
        assert!(g.contains("G90 ; absolute positions\nM851 Z-1.2\n"));
        assert!((st.draw_mm - 18.0).abs() < 1e-6);
        assert!(svg.contains(r#"<path d="M10 10C20 10 30 20 30 30"/>"#));
        assert!(svg.contains(r#"stroke="black""#));
    }

    #[test]
    fn lines_split_where_the_band_changes() {
        let line = Polyline { pts: (0..=10).map(|i| (i as f64, 0.0)).collect(), closed: false };
        let parts = split_by_band(&line, &|p| if p.0 < 4.0 { 0 } else if p.0 < 7.0 { 1 } else { 0 }, 0.0);
        assert_eq!(parts.iter().map(|p| p.0).collect::<Vec<_>>(), vec![0, 1, 0]);
        assert_eq!(parts[0].1.pts.last(), Some(&(3.5, 0.0)));
        assert_eq!(parts[1].1.pts.first(), Some(&(3.5, 0.0)));
        let total: f64 = parts.iter().map(|p| p.1.length()).sum();
        assert!((total - 10.0).abs() < 1e-9, "no length lost or added");
        // A loop starting and ending in band 0 comes back as 2 runs, not 3.
        let sq = Polyline { pts: vec![(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0), (0.0, 0.0)], closed: true };
        let parts = split_by_band(&sq, &|p| (p.1 > 2.0) as usize, 0.0);
        assert_eq!(parts.len(), 2);
        assert!((parts.iter().map(|p| p.1.length()).sum::<f64>() - 16.0).abs() < 1e-9);
        assert_eq!(split_by_band(&sq, &|_| 3, 0.0), vec![(3, sq.clone())]);
        // A 1-unit speck of band 1 inside band 0 is absorbed at min_run 2; a 3-unit run is not.
        let speck = split_by_band(&line, &|p| if (5.0..6.0).contains(&p.0) { 1 } else { 0 }, 2.0);
        assert_eq!(speck.len(), 1);
        assert_eq!(speck[0].0, 0);
        let kept = split_by_band(&line, &|p| if p.0 < 4.0 { 0 } else if p.0 < 7.0 { 1 } else { 0 }, 2.0);
        assert_eq!(kept.iter().map(|p| p.0).collect::<Vec<_>>(), vec![0, 1, 0]);
    }

    #[test]
    fn pens_take_equal_runs_of_levels() {
        let pens: Vec<usize> = (1..256).map(|i| level_pen(i, 256, 6)).collect();
        assert_eq!(pens[0], 0);
        assert_eq!(*pens.last().unwrap(), 5);
        for k in 0..6 {
            let n = pens.iter().filter(|&&p| p == k).count();
            assert!((42..=43).contains(&n), "pen {k}: {n} levels");
        }
        assert!(pens.windows(2).all(|w| w[1] >= w[0]), "outer to inner");
        assert_eq!(level_pen(10, 64, 1), 0);
    }
}
