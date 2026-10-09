//! Per-axis "does this fractal actually have a set limit when observed
//! from outside with raycasting" — Carl, 2026-09-23: "I want to know what
//! fractals actually have a set limit when observed from outside with
//! raycasting. For a specific fractal, that means the 4 axis have
//! actually limits on all 4 axis. Please also note these limits once
//! extracted. Please run the metric for every quaternion on the GA."
//!
//! Nothing existing already answers this. `quat_dag_fitness.rs`'s render-
//! based metrics and `quat_organization.rs`'s whole-4D-object metrics both
//! sample WITHIN an assumed `domain_radius` (1.6, or the render's own
//! working box) — they characterize shape inside that box, never ask
//! whether the box itself is big enough. `bailout_radius` (a per-genome
//! escape-detection threshold, `quat_dag.rs`'s `|z|² > bailout²` check) is
//! not a spatial extent either — a real check against archived genomes
//! found saved values of 4.0 (default), 9.95, 9.95, 10.28: nothing
//! today records how far a genome's actual bounded/filled set reaches.
//!
//! The approach mirrors `quat_organization.rs::march_radius` exactly (same
//! coarse-scan-then-bisect shape, same `et >= max_iter - 0.5` "never
//! escaped" test, same reasoning: bisection assumes a single inside/
//! outside crossing along the line, which isn't guaranteed for chaotic
//! dynamics in general — an accepted limitation `march_radius` already
//! lives with, not a new one introduced here) — but `march_radius` marches
//! within a CALLER-SUPPLIED `domain_radius` it assumes is big enough,
//! walking a `Vec3` DIRECTION through `TimeAxis::assemble`. This module
//! instead walks each of R/A/B/C SEPARATELY (the other three pinned at
//! 0 — the natural symmetric center, and the only choice that treats all
//! four axes identically rather than privileging whichever three
//! `TimeAxis` would call "space") out to a generous, genome-independent
//! `SEARCH_MAX`, specifically to discover whether a safe domain size even
//! exists, not to characterize shape inside one already assumed correct.
//!
//! `SEARCH_MAX = 64.0`: roughly 6× the largest `bailout_radius` seen in
//! the real archive (10.28) — generous enough for the common case, and
//! cheap (128 coarse steps at this scale is a tight, fast scan). Not
//! provably sufficient for every possible DAG program on its own (no
//! finite radius could be, in general) — which is exactly why there is a
//! second, escalating phase past it (see `axis_limit`'s own doc comment):
//! a real, Carl-reported false negative (2026-09-23 — "it say no discovery
//! limit for all axis" on genomes that DO have one) turned out to be a
//! genuine, non-pathological formula (one that happens to use a bounded
//! transcendental op — `COS` — partway through its DAG) that stays bounded
//! well past `SEARCH_MAX` on one axis — confirmed by temporarily
//! instrumenting `axis_limit` with an env-var-gated trace and probing that
//! exact genome directly, not guessed at. `SEARCH_MAX` alone silently
//! misreported this (and Carl's other reported genome, same pattern) as
//! unbounded. The exact crossing found this far out isn't a single stable
//! number, though — see the note on phase 2's own resolution below.
//!
//! An axis reports `None` (no limit found) for two different underlying
//! reasons this module deliberately does not distinguish in its return
//! type (Carl asked for the limits, not a taxonomy of why one might be
//! missing): the bounded set may genuinely extend past `HARD_CAP` in that
//! direction (or truly to infinity), OR the origin-outward line in that
//! direction may never re-enter the bounded set at all inside
//! `SEARCH_MAX` (a "shell" or "hollow" formula whose bounded region
//! doesn't reach that particular axis — phase 2 does not help here; see
//! `axis_limit`'s doc comment for why). Either way, that axis does not
//! have a discoverable limit — the only distinction "fully bounded on all
//! 4 axes" needs.
//!
//! One more accepted imprecision, inherited from (not introduced by)
//! `march_radius`'s own already-documented limitation just below: when an
//! axis's escape/re-entry structure is genuinely non-monotonic across
//! phase 2's `[SEARCH_MAX, HARD_CAP]` range — real, observed on the exact
//! genome that motivated phase 2 — a coarse-then-bisect pass at ANY finite
//! resolution can land on a DIFFERENT real crossing than another resolution
//! would, rather than some uniquely "correct" outermost one. The value
//! returned is still always a genuine inside/outside boundary (never
//! fabricated), just not necessarily THE outermost one in a case like
//! that — good enough to answer "does this axis have a limit at all"
//! (Carl's actual question), not a promise of the single truest number
//! for a chaotically-structured axis.

use crate::quat_dag::{quat_dag_escape_de, QuatDagFormula};
use crate::quaternion::Quat;

/// Phase 1's search radius — see this module's own doc comment for the
/// calibration. Still the right choice to try FIRST even now that phase 2
/// exists: it is what correctly catches "shell" structures (a bounded
/// region that doesn't include the origin — see `axis_limit`'s doc
/// comment), which phase 2's log-spaced scan starts from `SEARCH_MAX`
/// rather than the origin and so cannot.
pub const SEARCH_MAX: f64 = 64.0;
/// Phase 2's absolute ceiling — past this, an axis is reported as having
/// no limit at all, not just "none found yet". 1e9 was picked with
/// generous headroom over the real case that motivated phase 2 existing
/// (a genuine crossing found well past 64, up in the hundreds-of-
/// thousands range — see the module doc), while still being small enough
/// that `f64` squaring (up to `HARD_CAP²` = 1e18) stays nowhere near
/// overflow — a real, checked constraint, not an arbitrary round number.
const HARD_CAP: f64 = 1.0e9;
const COARSE_STEPS: u32 = 128;
/// [`SEARCH_MAX`, `HARD_CAP`] spans 7 orders of magnitude — far more than
/// phase 1's own `[0, SEARCH_MAX]`, so phase 2's log-spaced scan uses more
/// steps for comparable per-decade resolution (~37/decade vs. phase 1's
/// 128 steps over less than 2 decades). Still cheap: point evaluation is
/// the cost here, not step count, and 256 extra evaluations is nothing
/// next to running this over ~2500 genomes.
const COARSE_STEPS_PHASE2: u32 = 256;
const BISECT_ITERS: u32 = 20;
/// Phase 2's bracket can be up to `HARD_CAP` wide (vs. `SEARCH_MAX` for
/// phase 1) — a few more bisection steps keeps final precision comparable
/// at that much larger scale.
const BISECT_ITERS_PHASE2: u32 = 34;

/// `axis`: 0=R, 1=A, 2=B, 3=C. `sign`: +1.0 or -1.0.
fn point_on_axis(axis: usize, sign: f64, t: f64) -> Quat {
    let v = sign * t;
    match axis {
        0 => Quat::new(v, 0.0, 0.0, 0.0),
        1 => Quat::new(0.0, v, 0.0, 0.0),
        2 => Quat::new(0.0, 0.0, v, 0.0),
        _ => Quat::new(0.0, 0.0, 0.0, v),
    }
}

/// The outermost point along ONE axis, in ONE direction, still inside the
/// bounded/filled set — `None` if no limit was found within `HARD_CAP` (see
/// the module doc for the three folded-together underlying reasons).
///
/// Two phases:
/// 1. Coarse-scan-then-bisect within `[0, SEARCH_MAX]`, copied from
///    `quat_organization::march_radius`'s own proven shape — scanning from
///    `SEARCH_MAX` INWARD so the boundary found is the OUTERMOST crossing
///    (an inner shell nested inside an unbounded outer region would
///    otherwise read as falsely bounded). Catches the common case AND
///    "shell" structures (a bounded region that does not include the
///    origin) cheaply.
/// 2. Only reached when phase 1 finds the object STILL bounded at
///    `SEARCH_MAX` itself: the SAME outside-in coarse-scan-then-bisect
///    shape as phase 1, but over `[SEARCH_MAX, HARD_CAP]` on a LOG scale
///    (that range spans several orders of magnitude, so linear steps
///    would badly under-sample it) — scanning from `HARD_CAP` inward so
///    the boundary found is still the OUTERMOST crossing, for the same
///    reason phase 1 does. An earlier version instead doubled the probe
///    outward and stopped at the FIRST escape it hit; caught by this
///    module's own regression test, which found a closer, non-outermost
///    crossing (230) instead of the real one (~820,313) — escape/
///    re-entry structure past `SEARCH_MAX` is not guaranteed monotonic
///    any more than it is within phase 1's own range.
fn axis_limit(f: &QuatDagFormula, axis: usize, sign: f64, max_iter: u32, bailout_sq: f64) -> Option<f64> {
    let is_inside = |t: f64| -> bool {
        let (et, _) = quat_dag_escape_de(f, point_on_axis(axis, sign, t), max_iter, bailout_sq);
        et >= max_iter as f32 - 0.5
    };
    if !is_inside(SEARCH_MAX) {
        let mut t_outside = SEARCH_MAX;
        let mut t_inside = None;
        for i in 1..=COARSE_STEPS {
            let t = SEARCH_MAX * (1.0 - i as f64 / COARSE_STEPS as f64);
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
        return Some((lo + hi) / 2.0);
    }

    // Phase 2: still inside at SEARCH_MAX. Still bounded at HARD_CAP itself
    // means genuinely no limit found (or beyond any practical one).
    if is_inside(HARD_CAP) {
        return None;
    }
    // [SEARCH_MAX, HARD_CAP] spans several orders of magnitude, so a
    // LINEARLY spaced coarse scan (phase 1's own approach) would badly
    // under-sample it — scan geometrically (log-spaced) instead, same
    // "outside-in, stop at the first inside point" shape as phase 1, for
    // the same reason: the OUTERMOST crossing is the one that answers
    // whether this axis has a limit at all, not just the nearest one an
    // outward walk happens to hit first (an earlier version of this phase
    // doubled outward and stopped at the FIRST escape found instead —
    // caught by this exact test, which landed on a much closer, non-
    // outermost crossing instead: this axis's escape/re-entry structure
    // is genuinely non-monotonic across several orders of magnitude, the
    // same class of limitation `march_radius` already accepts (see this
    // module's own top doc comment) rather than something solvable by a
    // single coarse-then-bisect pass at ANY finite resolution.
    let log_lo = SEARCH_MAX.log10();
    let log_hi = HARD_CAP.log10();
    let mut t_outside = HARD_CAP;
    let mut t_inside = None;
    for i in 1..=COARSE_STEPS_PHASE2 {
        let frac = i as f64 / COARSE_STEPS_PHASE2 as f64;
        let t = 10f64.powf(log_hi - frac * (log_hi - log_lo));
        if is_inside(t) {
            t_inside = Some(t);
            break;
        } else {
            t_outside = t;
        }
    }
    // Guaranteed Some: the LAST coarse sample (frac=1.0) is SEARCH_MAX
    // itself, which this branch is only reached after confirming is_inside.
    let mut lo = t_inside.expect("SEARCH_MAX itself is always the final, guaranteed-inside coarse sample");
    let mut hi = t_outside;
    for _ in 0..BISECT_ITERS_PHASE2 {
        let mid = (lo + hi) / 2.0;
        if is_inside(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some((lo + hi) / 2.0)
}

/// One axis's outcome in both directions.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct AxisLimits {
    pub pos: Option<f64>,
    pub neg: Option<f64>,
}

impl AxisLimits {
    /// A limit was found in BOTH directions — a one-sided limit (bounded
    /// going +R but not -R, say) does not count as "this axis has a
    /// limit" for Carl's "all 4 axis have actually limits" framing.
    pub fn bounded(&self) -> bool {
        self.pos.is_some() && self.neg.is_some()
    }
}

/// The full result for one genome: each of R/A/B/C's `AxisLimits`, plus
/// whether every one of the 4 is bounded in both directions.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct BoundednessReport {
    pub r: AxisLimits,
    pub a: AxisLimits,
    pub b: AxisLimits,
    pub c: AxisLimits,
}

impl BoundednessReport {
    /// Carl's own definition: "the 4 axis have actually limits on all 4
    /// axis" — every one of R/A/B/C bounded in both directions.
    pub fn fully_bounded(&self) -> bool {
        self.r.bounded() && self.a.bounded() && self.b.bounded() && self.c.bounded()
    }
}

/// Scans all 4 axes (8 direction-scans total) for one formula.
/// `bailout_sq` should be the genome's own `bailout_radius²` — matching
/// every other quat metric's convention, not a fixed constant, since a
/// genome's own evolved escape threshold is what its render actually uses.
pub fn compute_boundedness(f: &QuatDagFormula, max_iter: u32, bailout_sq: f64) -> BoundednessReport {
    let scan = |axis: usize| AxisLimits {
        pos: axis_limit(f, axis, 1.0, max_iter, bailout_sq),
        neg: axis_limit(f, axis, -1.0, max_iter, bailout_sq),
    };
    BoundednessReport { r: scan(0), a: scan(1), b: scan(2), c: scan(3) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formula::{op, OpNode};

    fn mandelbrot_program() -> Vec<OpNode> {
        // z' = z² + c — same hand-built DAG `quat_organization.rs`'s own
        // test module uses (duplicated here rather than shared: that
        // helper is private to that module's `#[cfg(test)]` block).
        vec![
            OpNode { op: op::Z, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::C, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::SQR, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::ADD, a: 2, b: 1, kre: 0.0, kim: 0.0 },
        ]
    }

    /// z' = z (ignores c entirely) — with z starting at `Quat::ZERO` in
    /// non-Julia mode, the orbit is 0 forever regardless of the point
    /// tested, for ANY point at ANY distance. A genuinely, deliberately
    /// unbounded formula: every axis in every direction must report `None`.
    fn always_bounded_program() -> Vec<OpNode> {
        vec![OpNode { op: op::Z, a: 0, b: 0, kre: 0.0, kim: 0.0 }]
    }

    fn formula(prog: &[OpNode]) -> QuatDagFormula<'_> {
        QuatDagFormula { prog, warp: &[], julia: false, jc: (0.0, 0.0), phoenix: (0.0, 0.0) }
    }

    #[test]
    fn always_bounded_formula_reports_no_limit_on_every_axis() {
        let prog = always_bounded_program();
        let f = formula(&prog);
        let report = compute_boundedness(&f, 40, 16.0);
        for axis in [report.r, report.a, report.b, report.c] {
            assert_eq!(axis.pos, None);
            assert_eq!(axis.neg, None);
            assert!(!axis.bounded());
        }
        assert!(!report.fully_bounded());
    }

    #[test]
    fn mandelbrot_r_axis_matches_the_known_asymmetric_boundary() {
        // The real-axis extent of the classic Mandelbrot set is NOT ±2 —
        // that's a common misconception (checked here directly: an
        // earlier, looser version of this test asserted symmetric ±2 and
        // failed with a measured +0.2502, which is actually EXACTLY
        // right). The true bounded range on the real axis is
        // [-2, +0.25]: c=1/4 is the textbook cusp of the main cardioid
        // (the fixed point z=1/2 has derivative exactly 1 there), and
        // c=-2 is the tip of the period-2 bulb's antenna. Both directions
        // checked against their real, asymmetric values — strong
        // confirmation the algorithm is finding the actual mathematical
        // boundary, not a coincidentally-plausible-looking number.
        let prog = mandelbrot_program();
        let f = formula(&prog);
        let pos = axis_limit(&f, 0, 1.0, 200, 16.0).expect("bounded going +R");
        let neg = axis_limit(&f, 0, -1.0, 200, 16.0).expect("bounded going -R");
        assert!((0.24..0.26).contains(&pos), "+R limit {pos} should land at the cardioid cusp, c=0.25");
        assert!((1.95..2.05).contains(&neg), "-R limit {neg} should land near c=-2");
    }

    #[test]
    fn axis_limit_is_none_when_bounded_all_the_way_past_hard_cap() {
        // Bounded at SEARCH_MAX now escalates into phase 2 (exponential
        // doubling toward HARD_CAP) rather than short-circuiting straight
        // to None — always_bounded_program is bounded EVERYWHERE (even at
        // HARD_CAP itself), so phase 2 also exhausts and this still
        // correctly ends in None, just after trying much harder first.
        let prog = always_bounded_program();
        let f = formula(&prog);
        assert_eq!(axis_limit(&f, 2, -1.0, 40, 16.0), None);
    }

    #[test]
    fn phase_two_finds_a_real_limit_far_beyond_search_max() {
        // The exact genome that motivated phase 2 existing — Carl reported
        // (2026-09-23) "go to bounds" saying "no discovery limit for all
        // axis" on genomes that, per this test, genuinely DO have one:
        // fractals_dag_quat_taste_me5/me_sph2_sol3_ord1_9ff5d021e4bab5d7.nn.
        // Its +A axis is bounded well past SEARCH_MAX=64, caused by a
        // bounded transcendental op (COS, node 17) partway through the
        // DAG — traced directly (temporary env-var-gated instrumentation,
        // since removed). NOT asserting a single exact crossing value:
        // this axis's escape/re-entry structure is genuinely non-monotonic
        // across several orders of magnitude (different sample
        // resolutions land on different real crossings — see this
        // module's own top doc comment), so only the property that
        // actually matters is checked: phase 2 must find a real limit
        // meaningfully beyond SEARCH_MAX, not incorrectly report none at
        // all the way this genome originally did. Program/warp/jc copied
        // verbatim from that real genome file, not simplified — the whole
        // point is reproducing the exact case that was silently
        // misreported before.
        let program = vec![
            OpNode { op: 0, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: 13, a: 0, b: 0, kre: -0.551412, kim: -0.051999748 },
            OpNode { op: 10, a: 1, b: 1, kre: 0.0, kim: 0.0 },
            OpNode { op: 7, a: 2, b: 1, kre: 0.0, kim: 0.0 },
            OpNode { op: 10, a: 0, b: 0, kre: -0.73896825, kim: 0.67076576 },
            OpNode { op: 20, a: 2, b: 4, kre: 0.0, kim: 0.0 },
            OpNode { op: 20, a: 2, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: 17, a: 6, b: 6, kre: 0.0, kim: 0.0 },
            OpNode { op: 17, a: 5, b: 5, kre: 0.0, kim: 0.0 },
            OpNode { op: 17, a: 8, b: 6, kre: 0.0, kim: 0.0 },
            OpNode { op: 17, a: 9, b: 7, kre: 0.0, kim: 0.0 },
            OpNode { op: 9, a: 1, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: 15, a: 1, b: 11, kre: -0.87846756, kim: -0.1812272 },
            OpNode { op: 6, a: 11, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: 4, a: 13, b: 1, kre: 0.0, kim: 0.0 },
            OpNode { op: 18, a: 14, b: 3, kre: 0.0, kim: 0.0 },
            OpNode { op: 20, a: 11, b: 13, kre: 0.0, kim: 0.0 },
            OpNode { op: 8, a: 15, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: 19, a: 10, b: 17, kre: 0.0, kim: 0.0 },
            OpNode { op: 17, a: 18, b: 4, kre: 0.0, kim: 0.0 },
            OpNode { op: 17, a: 19, b: 14, kre: 0.0, kim: 0.0 },
            OpNode { op: 17, a: 20, b: 14, kre: 0.0, kim: 0.0 },
        ];
        let warp = vec![
            OpNode { op: 0, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: 1, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: 3, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: 5, a: 1, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: 14, a: 1, b: 0, kre: 0.0, kim: 0.0 },
        ];
        let f = QuatDagFormula { prog: &program, warp: &warp, julia: true, jc: (-0.05037065, -0.6250651), phoenix: (0.0, 0.0) };
        let bailout_sq = 6.505473_f64 * 6.505473_f64;
        let limit = axis_limit(&f, 1, 1.0, 200, bailout_sq).expect("this axis IS bounded, just far past SEARCH_MAX");
        assert!(limit > SEARCH_MAX * 2.0, "phase 2 should have found something well beyond SEARCH_MAX, got {limit}");
        assert!(limit < HARD_CAP, "a real crossing was found; it must be strictly inside HARD_CAP, got {limit}");
    }

    #[test]
    fn axis_limits_bounded_requires_both_directions() {
        let one_sided = AxisLimits { pos: Some(1.5), neg: None };
        assert!(!one_sided.bounded(), "a limit in only one direction must not count as bounded");
        let both = AxisLimits { pos: Some(1.5), neg: Some(1.2) };
        assert!(both.bounded());
    }

    #[test]
    fn fully_bounded_requires_all_four_axes() {
        let mostly = BoundednessReport {
            r: AxisLimits { pos: Some(1.0), neg: Some(1.0) },
            a: AxisLimits { pos: Some(1.0), neg: Some(1.0) },
            b: AxisLimits { pos: Some(1.0), neg: Some(1.0) },
            c: AxisLimits { pos: Some(1.0), neg: None }, // one axis, one direction, missing
        };
        assert!(!mostly.fully_bounded());
    }
}
