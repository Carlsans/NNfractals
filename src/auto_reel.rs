//! Automated video generation, stage 1: turn an archived `.nn` into a finished
//! shot without a human choosing anything.
//!
//! This module holds the *choosing* — which is the part the project did not have.
//! Scoring a video was already solved and well calibrated
//! (`video_export::probe_frames_score`, `time_explore`'s five gates,
//! `vae_explore::ZoneGate`, `video_zoom_explore::dd_wall_zoom`); what was always
//! done by eye was picking the fractal, the opening frame, the destination, and
//! the animation.
//!
//! # Framing: where the fractal ends
//!
//! An establishing shot has to contain the whole fractal, and neither of the two
//! obvious sources for that is reliable. A genome's saved `view_cx/cy/zoom` is
//! the frame the GA liked while scoring it — a beauty-scored crop, often already
//! well inside the set. A fixed wide zoom frames some fractals and strands
//! others in empty space, because these are evolved formulas whose bodies are not
//! all near the origin at scale 1.
//!
//! So [`auto_frame`] measures it: render deliberately wide, find the pixels that
//! belong to the fractal rather than to the background, and frame their bounding
//! box. Everything else in the pipeline hangs off getting this right, which is
//! why it reports what it measured rather than only what it decided.

use crate::config::Config;
use crate::genome::Genome;
use crate::video_export::{needs_f64, render_escape_times, CapturedView, View};

/// Resolution of a framing scan. Small on purpose: this measures WHERE the
/// fractal is, not what it looks like, and a bounding box is not improved by
/// more pixels. It also keeps every scan on the fast f32/GPU tier.
pub const FRAME_RES: u32 = 256;

/// Square view spans tried when framing, tightest first.
///
/// `View::new_square` spans `4/zoom`, so 1.0 is the classic `-2..2` window and
/// 0.04 is a 100-unit-wide view. Ordered tightest-first because the first scan
/// that contains the whole body is the one that measures it most precisely —
/// a body occupying 4% of a very wide frame quantises its own bounding box to
/// a handful of pixels.
pub const FRAME_SCAN_ZOOMS: &[f64] = &[2.0, 1.0, 0.5, 0.25, 0.1, 0.04, 0.015];

/// A pixel counts as part of the fractal when it either never escapes, or takes
/// at least this fraction of the frame's own deepest escape time.
///
/// Relative rather than absolute because escape times are not comparable across
/// formulas — one genome's boundary may take 300 iterations where another's
/// takes 20, and an absolute floor would call the second one empty. Escape time
/// falls off roughly logarithmically with distance from the set, so this
/// threshold is not sharp.
///
/// # Calibration
///
/// Swept over the top ten `fractals_1` genomes at 0.25 / 0.10 / 0.05 and judged
/// from a contact sheet of the resulting opening frames (2026-09-10). At 0.25 the
/// threshold tracks only the bright core, so the dim outer halo falls outside the
/// measured box and four of seven shots opened mid-crop — a radial burst with its
/// rays cut at the frame edge, a four-armed fractal with its arms cut top and
/// bottom, a blob cut at both poles. 0.10 framed all four with margin and changed
/// nothing on the rest. 0.05 was marginally wider again with no visible gain, and
/// pushed an already-thin fractal further into the middle distance.
pub const BODY_ESCAPE_FRACTION: f32 = 0.10;

/// How much of the frame the fractal should occupy in the opening shot.
pub const FRAME_FILL: f64 = 0.85;

/// A body pixel this close to the border means the fractal continues past the
/// frame, so this scan cannot have measured all of it.
pub const EDGE_MARGIN_PX: u32 = 2;

/// How many body pixels must reach the border ring before the scan counts as
/// cropping the fractal.
///
/// More than one, because a lone speck at the border is noise; but measured on
/// RAW body pixels rather than on the trimmed box, because the two questions are
/// different statistics. "How big is the body" is a percentile — one stray
/// pixel must not set the frame size. "Does the fractal continue past the edge"
/// is not: for a connected body a few pixels at the border genuinely answer yes,
/// and trimming them away hides exactly the signal the test is for. Conflating
/// them was measured to matter: reusing the trimmed box for both took one real
/// genome from a 0.14x frame to a 1.76x one by silently discarding its outermost
/// petals.
pub const EDGE_TOUCH_MIN_PIXELS: usize = 3;

/// Above this body fraction at the widest scan, there is no meaningful
/// "outside" to open on — the formula fills the plane.
pub const MAX_BODY_FRACTION: f32 = 0.60;

/// Below this, the scan found no fractal at all.
pub const MIN_BODY_FRACTION: f32 = 0.001;

/// A sampled point must reach this fraction of the line's OWN best richness.
///
/// `video_zoom_explore`'s absolute floor of 0.30 is calibrated on bright,
/// high-contrast frames, and a dark low-contrast fractal compresses better at
/// every depth simply because it is dark. Measured on a real archived genome:
/// the entire straight line scored 0.10-0.32 against a maximum of 0.319, so an
/// absolute 0.30 rejected the whole shot while the frames themselves were fine.
/// Taking the stricter of the absolute floor and this relative one keeps the
/// calibrated behaviour for bright fractals and stops punishing dark ones for
/// being dark.
pub const RICHNESS_RELATIVE: f32 = 0.60;

/// Hard floor no relative rule may go below — genuinely flat is genuinely flat,
/// and 60% of nothing is still nothing.
pub const RICHNESS_ABSOLUTE_MIN: f32 = 0.12;

/// Fraction of body pixels trimmed from each side before the bounding box is
/// taken.
///
/// A raw min/max box is decided by its two most extreme pixels, so one stray
/// speck in a corner — and evolved formulas produce plenty of those — inflates
/// the whole frame and pushes the fractal into the middle distance. Trimming
/// half a percent from each edge costs nothing on a solid body and removes the
/// outlier sensitivity entirely.
pub const BBOX_TRIM: f64 = 0.005;

/// Knobs for [`auto_frame_with`]. The defaults are [`BODY_ESCAPE_FRACTION`] and
/// [`FRAME_FILL`]; they are parameters so the thresholds can be swept against
/// real archive genomes instead of argued about.
#[derive(Clone, Copy, Debug)]
pub struct FrameOpts {
    pub body_escape_fraction: f32,
    pub fill: f64,
    pub res: u32,
}

impl Default for FrameOpts {
    fn default() -> Self {
        FrameOpts {
            body_escape_fraction: BODY_ESCAPE_FRACTION,
            fill: FRAME_FILL,
            res: FRAME_RES,
        }
    }
}

/// What a framing scan measured, not just what it decided.
///
/// The measurements are reported because the framing rule is the one piece of
/// this pipeline with no prior art in the codebase to inherit calibration from:
/// when a shot opens badly, these numbers say whether the threshold, the scan
/// range, or the fill fraction was wrong.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct FrameFit {
    /// The opening shot.
    pub view: CapturedView,
    /// Which entry of [`FRAME_SCAN_ZOOMS`] produced it.
    pub scan_zoom: f64,
    /// Fraction of the scan's pixels that belong to the fractal.
    pub body_fraction: f32,
    /// The larger side of the measured bounding box, in fractal units.
    pub extent: f64,
    /// Deepest escape time seen in the scan, against the compute cap.
    pub max_escape: f32,
}

/// One framing scan's raw measurements.
#[derive(Clone, Copy, Debug)]
struct Scan {
    body_fraction: f32,
    /// `None` when no pixel qualified.
    bbox: Option<(f64, f64, f64, f64)>,
    touches_edge: bool,
    max_escape: f32,
    /// Fraction of pixels that never escaped at all.
    interior_fraction: f32,
    /// Essentially every pixel escaped at the same time — a flat frame, not a
    /// fractal. Checked separately because the relative body threshold cannot
    /// see it: if every escape time is equal then every pixel clears
    /// `BODY_ESCAPE_FRACTION` of the maximum, and a blank frame measures as a
    /// body filling the plane. Two opposite diagnoses for the same picture.
    degenerate: bool,
}

/// Which pixels belong to the fractal, and their bounding box in fractal units.
fn scan_body(et: &[f32], res: u32, view: &View, compute_iter: u32, body_frac: f32) -> Scan {
    let n = (res * res) as usize;
    let max_escape = et.iter().copied().fold(0.0f32, f32::max);
    // Interior pixels sit exactly at the cap (see `fractal::dag_escape_pixel`),
    // so they are counted directly rather than via the relative threshold —
    // a fractal that is ALL interior has no escaping pixels to take a fraction
    // of, and would otherwise measure as empty.
    let interior = compute_iter as f32 - 0.5;
    let thresh = (body_frac * max_escape).min(interior);

    let mut interior_count = 0usize;
    let (mut xs, mut ys): (Vec<u32>, Vec<u32>) = (Vec::new(), Vec::new());

    for i in 0..n.min(et.len()) {
        if et[i] >= interior {
            interior_count += 1;
        }
        if et[i] < thresh {
            continue;
        }
        xs.push((i as u32) % res);
        ys.push((i as u32) / res);
    }

    let count = xs.len();
    let margin = EDGE_MARGIN_PX.min(res / 2);
    let border = xs.iter().zip(&ys)
        .filter(|&(&px, &py)| px < margin || py < margin || px + margin >= res || py + margin >= res)
        .count();
    let touches_edge = border >= EDGE_TOUCH_MIN_PIXELS;

    let bbox = trimmed_span(&mut xs, &mut ys).map(|(px0, px1, py0, py1)| {
        let (xmin, xmax, ymin, ymax) = view.bounds();
        let to_x = |p: u32| xmin + (p as f64 / (res.max(2) - 1) as f64) * (xmax - xmin);
        let to_y = |p: u32| ymin + (p as f64 / (res.max(2) - 1) as f64) * (ymax - ymin);
        (to_x(px0), to_x(px1), to_y(py0), to_y(py1))
    });

    Scan {
        body_fraction: count as f32 / n as f32,
        bbox,
        touches_edge,
        max_escape,
        interior_fraction: interior_count as f32 / n as f32,
        degenerate: crate::fitness::is_degenerate(et),
    }
}

/// Pixel bounds of the body with [`BBOX_TRIM`] shaved off each side, per axis.
///
/// Sorts and indexes rather than tracking a running min/max, so the trim is a
/// percentile rather than a guess about which pixels are outliers.
fn trimmed_span(xs: &mut [u32], ys: &mut [u32]) -> Option<(u32, u32, u32, u32)> {
    if xs.is_empty() {
        return None;
    }
    xs.sort_unstable();
    ys.sort_unstable();
    let k = ((xs.len() as f64 * BBOX_TRIM) as usize).min(xs.len().saturating_sub(1) / 2);
    Some((xs[k], xs[xs.len() - 1 - k], ys[k], ys[ys.len() - 1 - k]))
}

/// The opening shot for `g`: the whole fractal, filling [`FRAME_FILL`] of frame.
///
/// Scans [`FRAME_SCAN_ZOOMS`] tightest-first and takes the first view that
/// contains the entire body with room to spare, then frames that body's bounding
/// box. Refuses rather than guesses when the formula fills the plane (nothing to
/// open on) or when nothing is there (nothing to film).
///
/// This answers WHERE the fractal is, not whether it is worth filming. A smooth
/// radial gradient with no structure has a perfectly well-defined body and will
/// be framed happily; rejecting it is the job of `vae_explore::ZoneGate`, which
/// the destination search already applies to every candidate. Folding a
/// structure test in here would put the same judgement in two places with two
/// calibrations.
///
/// The returned zoom is derived from the LARGER side of the bounding box against
/// the view's vertical half-extent. That is deliberately conservative for a wide
/// output: `chain_frame_views` overrides aspect with the export's own, which
/// widens the horizontal field and never crops the vertical, so a body framed
/// this way still fits at 16:9.
pub fn auto_frame(g: &Genome, config: &Config) -> Result<FrameFit, String> {
    auto_frame_with(g, config, &FrameOpts::default())
}

/// [`auto_frame`] with the thresholds spelled out.
pub fn auto_frame_with(
    g: &Genome, config: &Config, fo: &FrameOpts,
) -> Result<FrameFit, String> {
    let compute_iter = config.rendering.max_iter;
    let res = fo.res.max(16);
    let mut widest: Option<(f64, Scan)> = None;

    for &zoom in FRAME_SCAN_ZOOMS {
        let view = View::new_square(0.0, 0.0, zoom);
        let et = render_escape_times(
            g, config, &view, res, res, compute_iter, needs_f64(&view, res), false,
        );
        let scan = scan_body(&et, res, &view, compute_iter, fo.body_escape_fraction);
        widest = Some((zoom, scan));

        if scan.degenerate || scan.body_fraction < MIN_BODY_FRACTION || scan.touches_edge {
            continue;
        }
        let Some((lo_x, hi_x, lo_y, hi_y)) = scan.bbox else { continue };
        let extent = (hi_x - lo_x).max(hi_y - lo_y);
        if extent <= 0.0 || !extent.is_finite() {
            continue;
        }
        return Ok(FrameFit {
            view: CapturedView {
                cx: (lo_x + hi_x) * 0.5,
                cx_lo: 0.0,
                cy: (lo_y + hi_y) * 0.5,
                cy_lo: 0.0,
                zoom: 4.0 / (extent / fo.fill),
                aspect: 1.0,
            },
            scan_zoom: zoom,
            body_fraction: scan.body_fraction,
            extent,
            max_escape: scan.max_escape,
        });
    }

    // Every scan failed. Say which way, because the two call for opposite fixes.
    // A uniform frame and a plane-filling one look identical to the body
    // threshold — every pixel is at the maximum either way — so the interior
    // fraction is what separates them, and they call for opposite fixes.
    match widest {
        Some((zoom, s)) if s.interior_fraction > MAX_BODY_FRACTION => Err(format!(
            "the formula fills the plane ({:.0}% of a {:.3}x view never escapes) — \
             there is no outside to open on",
            s.interior_fraction * 100.0, zoom
        )),
        Some((zoom, s)) if s.degenerate => Err(format!(
            "nothing to film: the frame is uniform at the widest scan ({zoom:.3}x) — \
             every pixel escapes at the same time"
        )),
        Some((zoom, s)) if s.body_fraction > MAX_BODY_FRACTION => Err(format!(
            "the formula fills the plane ({:.0}% of a {:.3}x view is still fractal) — \
             there is no outside to open on",
            s.body_fraction * 100.0, zoom
        )),
        Some((zoom, s)) if s.body_fraction < MIN_BODY_FRACTION => Err(format!(
            "nothing to film: {:.4}% body at the widest scan ({:.3}x), deepest escape {:.1}",
            s.body_fraction * 100.0, zoom, s.max_escape
        )),
        Some((zoom, s)) => Err(format!(
            "the fractal runs past the widest scan ({:.3}x): {:.1}% body still touching the frame edge",
            zoom, s.body_fraction * 100.0
        )),
        None => Err("no scan zooms configured".into()),
    }
}

// ════════════════════════════════════════════════════════════════════════════
// The destination: a straight zoom that stays interesting the whole way
// ════════════════════════════════════════════════════════════════════════════

/// Canvas resolution for a descent step. Matches `video_zoom_explore`'s own
/// choice for the same reason: `coarse_scan` strided-samples every crop down to
/// `COARSE_SAMPLE_RES` and the smallest scan scale is 0.125, so 1024 is the
/// smallest canvas that fully feeds the sampling. Larger measured both slower
/// AND worse there.
pub const DESCENT_CANVAS_RES: u32 = 1024;

/// How far a descent ply may recentre, as a fraction of the current view's own
/// half-extent.
///
/// This is what makes a STRAIGHT zoom possible at all, and it is a geometric
/// argument rather than a tuned number. An unconstrained descent recentres
/// freely at every ply — that is exactly what makes it good at finding
/// structure, and exactly what makes the straight line to its endpoint miss:
/// the target ends up somewhere the opening frame was never pointing.
///
/// Bound the per-ply drift to `k` half-extents while each ply multiplies zoom by
/// at least `z`, and the total remaining drift from any ply is at most
/// `k·h·z/(z−1)` — a convergent geometric series. At `k = 0.5` and the `z ≥ 2`
/// the scan scales guarantee, that is `≤ h`: the final target stays inside every
/// intermediate frame, so the straight line to it passes through all of them.
///
/// Measured before this existed: on ten top-ranked archive genomes, four
/// straight lines died within a couple of samples of the opening while the
/// descent itself had reached 1e12.
pub const MAX_PLY_DRIFT: f64 = 0.5;

/// Safety factor applied to the precision wall.
///
/// `dd_wall_zoom` is where f64 quantisation makes adjacent pixels collapse onto
/// the same coordinate. Stopping exactly there means the last frames of every
/// reel are the blockiest they can possibly be, so back off one doubling — the
/// cost is one frame of zoom out of ~45.
pub const DD_WALL_SAFETY: f64 = 0.5;

/// How the automated pipeline picks where to zoom to.
#[derive(Clone, Debug)]
pub struct DestinationOpts {
    /// Width the precision wall is computed against. Must be the FINAL export
    /// width, not the preview's — deriving it from a 512px preview would set an
    /// end zoom a 1920px render cannot resolve.
    pub final_width: u32,
    pub dd_margin_ulps: f64,
    /// Maximum descent plies. Each step multiplies zoom by 2-8x, so ~25 reaches
    /// the wall from a wide start.
    pub max_steps: usize,
    pub canvas_res: u32,
    pub gate: crate::vae_explore::ZoneGate,
    pub method: crate::explore::ScoreMethod,
    /// Per-ply recentring limits to try. Every one is descended and its
    /// straight line validated; the deepest surviving shot wins.
    ///
    /// Two strategies, because neither dominates. An UNCONSTRAINED descent
    /// (infinite drift) finds the richest structure but wanders, so the straight
    /// line to it often misses. A BOUNDED one (see [`MAX_PLY_DRIFT`]) guarantees
    /// the target stays inside every intermediate frame but has to choose from a
    /// smaller set of candidates, so it sometimes settles for a weaker zone.
    /// Measured on the same archived genome: unconstrained reached 35.8
    /// doublings where bounded managed 11.3 — and on others the bounded descent
    /// was the only one that survived the line check at all.
    ///
    /// Running both roughly doubles the aim stage, which is minutes in a batch
    /// that takes hours, and the comparison is free because the doubling count
    /// is already computed.
    pub drift_limits: Vec<f64>,
    /// Stop trying further descent strategies once one produces a shot at least
    /// this deep.
    ///
    /// The aim stage is the expensive half of planning a reel — every ply
    /// renders a full canvas, and past zoom ~512 that is on the CPU/f64 path —
    /// so trying both strategies unconditionally doubles it. In practice the
    /// unconstrained descent already gives a good shot most of the time; the
    /// bounded one earns its cost only when the free line dies early.
    pub good_enough_doublings: f64,
    /// How many points along the straight line get a real render.
    pub check_frames: usize,
    /// Frame richness a sampled point must reach, on
    /// `video_zoom_explore::file_size_entropy`'s scale.
    pub min_frame_richness: f32,
    /// How many consecutive sparse samples may be bridged inside the shot
    /// before the line counts as having run out of structure.
    pub max_gap: usize,
    /// Minimum doublings of zoom for a shot to be worth rendering.
    pub min_doublings: f64,
}

impl Default for DestinationOpts {
    fn default() -> Self {
        DestinationOpts {
            final_width: 1920,
            dd_margin_ulps: crate::video_export::DD_MARGIN_ULPS_PIXELATE,
            max_steps: 25,
            canvas_res: DESCENT_CANVAS_RES,
            // Same already-tuned values `video_zoom_explore` defaults to; this
            // is the identical candidate shape from the identical `coarse_scan`.
            gate: crate::vae_explore::ZoneGate { max_intricacy: 0.30, min_edge_density: 0.05 },
            method: crate::explore::ScoreMethod::GatedEntropy,
            drift_limits: vec![f64::INFINITY, MAX_PLY_DRIFT],
            // A 2^25 ≈ 34-million-fold zoom. Deep enough that a second strategy
            // is unlikely to be worth minutes of extra rendering.
            good_enough_doublings: 25.0,
            check_frames: 24,
            min_frame_richness: 0.30,
            // One, not zero: a single probe frame dipping below the floor
            // mid-shot is within the noise of a 128px probe, and cutting a
            // whole reel short on one sample would be brittle. Two in a row is
            // structure genuinely running out.
            max_gap: 1,
            // Ten doublings is a 1000x zoom — the shortest travel that still
            // reads as a journey rather than a slow push-in. Below that the
            // render time is better spent on another fractal.
            min_doublings: 10.0,
        }
    }
}

/// Where a reel zooms to, and how confidently.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct Destination {
    pub end: CapturedView,
    /// The f64 precision wall at the final export width.
    pub dd_wall: f64,
    /// Zoom the descent reached before the straight line was validated.
    pub descent_zoom: f64,
    pub descent_steps: usize,
    /// Zoom the straight line was cut back to, if it was.
    pub trimmed_to: Option<f64>,
    /// Sampled points along the line, and how many passed.
    pub checked: usize,
    pub passed: usize,
    /// Leading samples below the richness floor — the establishing shot. This
    /// is expected, not a defect: the opening frame is deliberately a wide view
    /// of the whole fractal, which is mostly background.
    pub sparse_opening: usize,
    /// Worst richness seen along the kept span.
    pub worst_richness: f32,
    /// Which drift limit produced this shot — `INFINITY` for the
    /// unconstrained descent. Reported so the balance between the two
    /// strategies is visible over a batch rather than guessed at.
    #[serde(default = "unconstrained")]
    pub ply_drift: f64,
}

fn unconstrained() -> f64 {
    f64::INFINITY
}

impl Destination {
    /// How close the shot gets to the precision wall.
    ///
    /// The one diagnostic worth reading per reel: a destination at 1e9 against a
    /// 9e12 wall ran out of STRUCTURE, one at 9e12 ran out of PRECISION, and
    /// those call for opposite fixes.
    pub fn wall_fraction(&self) -> f64 {
        if self.dd_wall > 0.0 { self.end.zoom / self.dd_wall } else { 0.0 }
    }

    /// How many doublings of zoom were left on the table.
    ///
    /// Reported instead of the raw fraction because these ratios span ten
    /// orders of magnitude: a real result printed as "0.0% of the wall" says
    /// nothing, where "18 doublings short" is immediately legible and is also
    /// the unit the zoom is actually built from.
    pub fn doublings_short(&self) -> f64 {
        if self.dd_wall > 0.0 && self.end.zoom > 0.0 {
            (self.dd_wall / self.end.zoom).log2().max(0.0)
        } else {
            0.0
        }
    }

    /// Total doublings the shot travels.
    pub fn doublings_travelled(&self, start_zoom: f64) -> f64 {
        if start_zoom > 0.0 { (self.end.zoom / start_zoom).log2().max(0.0) } else { 0.0 }
    }
}

/// The usable span of a checked line: the first passing sample to the last one
/// reachable without crossing more than `max_gap` consecutive failures.
///
/// Leading failures are skipped rather than fatal. That is the whole correction
/// this function exists to make: the opening frame of a reel is a deliberately
/// wide establishing view of the entire fractal, so it is mostly flat
/// background and scores LOW on a richness measure calibrated for deep-zoom
/// frames. Treating that as "the line is dead" rejected four of five real
/// shots on the first run. `video_zoom_explore` already draws the same
/// distinction for its chains — "a trimmed head just means the opening was
/// sparse" — and only a trimmed TAIL means the shot ran out of structure.
pub fn usable_span(good: &[bool], max_gap: usize) -> Option<(usize, usize)> {
    let first = good.iter().position(|&g| g)?;
    let (mut last, mut gap) = (first, 0usize);
    for (i, &g) in good.iter().enumerate().skip(first + 1) {
        if g {
            last = i;
            gap = 0;
        } else {
            gap += 1;
            if gap > max_gap {
                break;
            }
        }
    }
    Some((first, last))
}

/// Descend from `start`, recentring on the best sub-square each ply, and return
/// the deepest centre reached.
///
/// This is not the shot — it is only how the destination POINT is located. The
/// shot itself is the straight line to it, which is why the path this walks is
/// thrown away.
fn descend(
    g: &Genome, config: &Config, start: &View, wall: f64, drift: f64, opts: &DestinationOpts,
) -> (View, usize) {
    let mut view = start.clone();
    let mut steps = 0usize;

    for _ in 0..opts.max_steps {
        let eff_iter = crate::video_export::effective_max_iter(&view, config.rendering.max_iter);
        let field = render_escape_times(
            g, config, &view, opts.canvas_res, opts.canvas_res, eff_iter,
            needs_f64(&view, opts.canvas_res), false,
        );
        let cands = crate::vae_explore::coarse_scan(
            &field, opts.canvas_res, opts.canvas_res, &view, config, opts.method,
        );
        let ranked = crate::explore::rank_by_zscore(&cands, 0.0);
        // The drift limit is in units of THIS view's half-extent, so it tightens
        // automatically as the descent goes deeper.
        let max_drift = drift * 2.0 / view.zoom;
        let next = ranked.iter().find(|c| {
            c.zoom <= wall
                && c.dx.hypot(c.dy) <= max_drift
                && c.metrics.edge_density >= opts.gate.min_edge_density
                && c.metrics.intricacy <= opts.gate.max_intricacy
        });
        let Some(c) = next else { break };
        view = crate::explore::apply_offset(&view, c.dx, c.dy, c.zoom);
        steps += 1;
    }
    (view, steps)
}

/// The end view for a straight zoom out of `start`.
///
/// Three stages, because a straight line is genuinely harder than the chain
/// search this reuses. The descent finds a deep point by RECENTRING at every
/// ply — that is exactly what makes it good at finding structure, and exactly
/// what makes the straight line to its endpoint suspect, since the interesting
/// region moved off-centre on the way down. So the line is then rendered and
/// checked point by point, and cut back to the deepest prefix that holds up.
///
/// Trimming is expected to be the common case, not a failure. What matters is
/// that the reel says how deep it actually got and against what wall — see
/// [`Destination::wall_fraction`].
pub fn find_destination(
    g: &Genome, config: &Config, start: &CapturedView, opts: &DestinationOpts,
) -> Result<Destination, String> {
    let wall = crate::video_zoom_explore::dd_wall_zoom(opts.final_width, opts.dd_margin_ulps)
        * DD_WALL_SAFETY;
    if !(wall > start.zoom) {
        return Err(format!(
            "the opening shot at {:.3e}x is already past the f64 wall for a {}px export ({wall:.3e})",
            start.zoom, opts.final_width
        ));
    }

    let limits = if opts.drift_limits.is_empty() {
        vec![f64::INFINITY]
    } else {
        opts.drift_limits.clone()
    };
    let mut best: Option<Destination> = None;
    let mut last_err = String::new();
    for drift in limits {
        match aim_once(g, config, start, wall, drift, opts) {
            Ok(d) => {
                let deep_enough = d.doublings_travelled(start.zoom) >= opts.good_enough_doublings;
                if best.as_ref().is_none_or(|b| d.end.zoom > b.end.zoom) {
                    best = Some(d);
                }
                if deep_enough {
                    break;
                }
            }
            Err(e) => last_err = e,
        }
    }
    best.ok_or(last_err)
}

/// One descent strategy, descended and validated.
fn aim_once(
    g: &Genome, config: &Config, start: &CapturedView, wall: f64, drift: f64,
    opts: &DestinationOpts,
) -> Result<Destination, String> {
    let (deep, descent_steps) = descend(g, config, &start.to_view(), wall, drift, opts);
    if descent_steps == 0 {
        return Err("nowhere to zoom: no candidate cleared the structure gate at the opening view".into());
    }

    // The straight line goes to the descent's POSITION at the wall's DEPTH —
    // not to the depth the descent happened to stop at, which is a property of
    // the search budget rather than of the fractal.
    let far = CapturedView {
        cx: deep.cx, cx_lo: deep.cx_lo, cy: deep.cy, cy_lo: deep.cy_lo,
        zoom: wall, aspect: start.aspect,
    };

    // Sample the whole line first, then decide. Deciding as we go is what made
    // the sparse establishing shot look like a dead line.
    let n = opts.check_frames.max(2);
    let mut richness = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 / (n - 1) as f64;
        let v = crate::video_export::lerp_view(start, &far, t);
        richness.push(crate::video_zoom_explore::file_size_entropy(
            g, config, &v, crate::video_zoom_explore::FILE_SIZE_PROBE_RES, opts.final_width,
        ));
    }
    let peak = richness.iter().copied().fold(0.0f32, f32::max);
    let floor = opts.min_frame_richness
        .min(RICHNESS_RELATIVE * peak)
        .max(RICHNESS_ABSOLUTE_MIN);
    let good: Vec<bool> = richness.iter().map(|&r| r >= floor).collect();

    let Some((first, last)) = usable_span(&good, opts.max_gap) else {
        return Err(format!(
            "nothing along the straight line to ({:.6},{:.6}) holds up — \
             the descent found structure the line does not pass through\n         {}",
            deep.cx, deep.cy, richness_profile(&richness, floor)
        ));
    };
    if last <= first {
        return Err(format!(
            "only one of {n} sampled points along the line holds up — nothing to zoom into\n         {}",
            richness_profile(&richness, floor)
        ));
    }

    let worst = richness[first..=last].iter().copied().fold(f32::MAX, f32::min);
    let end = crate::video_export::lerp_view(start, &far, last as f64 / (n - 1) as f64);
    let trimmed_to = (last + 1 < n).then_some(end.zoom);

    let doublings = (end.zoom / start.zoom).log2();
    if doublings < opts.min_doublings {
        return Err(format!(
            "the shot only travels {doublings:.1} doublings (want {:.0}) before the structure \
             runs out\n         {}",
            opts.min_doublings, richness_profile(&richness, floor)
        ));
    }

    Ok(Destination {
        end: CapturedView::from_view(&end),
        dd_wall: wall,
        descent_zoom: deep.zoom,
        descent_steps,
        trimmed_to,
        checked: n,
        passed: good.iter().filter(|&&g| g).count(),
        sparse_opening: first,
        worst_richness: if worst == f32::MAX { 0.0 } else { worst },
        ply_drift: drift,
    })
}

/// The richness of every sampled point as a sparkline, with the ones clearing
/// the floor in upper case.
///
/// A refusal that only says "nothing holds up" sends you guessing between three
/// different causes — the floor is too high, the descent aimed somewhere the
/// line misses, or the fractal genuinely has one interesting depth. The shape of
/// the profile distinguishes them at a glance, which is why it is attached to
/// the message rather than left to a rerun with tracing on.
pub fn richness_profile(richness: &[f32], floor: f32) -> String {
    // Letters rather than a `.:-=+*#%@` ramp, because the bar has to carry TWO
    // things at once: how rich each sample is, and whether it cleared the floor.
    // Case does the second for free; punctuation has no case to carry it.
    const RAMP: &[u8] = b"abcdefghi";
    let bar: String = richness.iter().map(|&r| {
        let i = ((r.clamp(0.0, 1.0) * RAMP.len() as f32) as usize).min(RAMP.len() - 1);
        let c = RAMP[i] as char;
        if r >= floor { c.to_ascii_uppercase() } else { c }
    }).collect();
    format!("richness |{bar}| a→i low→high, CAPS clear the floor {floor:.2}  max {:.3}",
            richness.iter().copied().fold(0.0f32, f32::max))
}

/// One log line per destination — the audit trail for the zoom rule.
pub fn destination_report(label: &str, start_zoom: f64, d: &Destination) -> String {
    format!(
        "[zoom] {label}  drift {}  descent {} plies to {:.2e}x  line {}/{} rich (opening {} sparse)  \
         worst {:.3}  → end {:.3e}x  travels {:.1} doublings, stops {:.1} short of the \
         {:.2e} wall{}",
        if d.ply_drift.is_finite() { format!("{:.2}", d.ply_drift) } else { "free".into() },
        d.descent_steps, d.descent_zoom, d.passed, d.checked, d.sparse_opening,
        d.worst_richness, d.end.zoom, d.doublings_travelled(start_zoom),
        d.doublings_short(), d.dd_wall,
        if d.trimmed_to.is_some() { "  (trimmed)" } else { "" }
    )
}

// ════════════════════════════════════════════════════════════════════════════
// The reel: one finished decision, written down
// ════════════════════════════════════════════════════════════════════════════

/// Where a batch of reels lives.
pub fn reels_dir() -> std::path::PathBuf {
    crate::project_root().join("reels")
}

/// What a reel is waiting for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ReelStatus {
    /// Rendered as a preview; a human has not decided yet.
    Pending,
    /// Approved and queued for a full-resolution render.
    Approved,
    Rejected,
}

/// One automatically-planned shot: the complete recipe plus every measurement
/// behind it.
///
/// Written next to its preview `.mp4` and a copy of its genome, so a reel is
/// self-contained — the review GUI reads only this, and re-rendering at full
/// resolution needs nothing that is not in here. The measurements travel with
/// it because the reason to trust or distrust a shot (how far it got against
/// the precision wall, how sparse its opening was, which depth its time formula
/// was worst at) is exactly what a human triaging it wants to see.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ReelRecord {
    pub id: String,
    /// The archived genome this came from, for de-duplicating future batches.
    pub source: String,
    pub label: String,
    /// Filename of the genome copy beside this record.
    pub nn_file: String,
    /// Filename of the preview clip beside this record.
    pub preview_file: String,
    pub start: CapturedView,
    pub end: CapturedView,
    #[serde(default)]
    pub time_prog: Vec<crate::time_program::TimeProgram>,
    pub frame: FrameFit,
    pub destination: Destination,
    /// The time formula's worst-depth score, and whether the clip loops.
    #[serde(default)]
    pub time_score: f64,
    #[serde(default)]
    pub time_loops: bool,
    #[serde(default)]
    pub time_summary: String,
    /// Fraction of the shot over which the formula can still animate, and the
    /// zoom where that stops — see `time_ga::animatable_span`. Past it the
    /// camera keeps descending and the formula necessarily holds still, which
    /// is a precision limit rather than a defect.
    #[serde(default = "one")]
    pub animated_fraction: f64,
    #[serde(default)]
    pub animatable_to: f64,
    pub preview_w: u32,
    pub preview_h: u32,
    pub preview_fps: u32,
    pub preview_frames: u32,
    /// Seconds spent in each stage, so a batch's cost is attributable.
    #[serde(default)]
    pub secs_frame: f32,
    #[serde(default)]
    pub secs_aim: f32,
    #[serde(default)]
    pub secs_evolve: f32,
    #[serde(default)]
    pub secs_render: f32,
    pub created_at: u64,
    #[serde(default = "pending")]
    pub status: ReelStatus,
}

fn pending() -> ReelStatus {
    ReelStatus::Pending
}

fn one() -> f64 {
    1.0
}

impl ReelRecord {
    /// Doublings of zoom this shot travels.
    pub fn doublings(&self) -> f64 {
        if self.start.zoom > 0.0 { (self.end.zoom / self.start.zoom).log2().max(0.0) } else { 0.0 }
    }

    pub fn total_secs(&self) -> f32 {
        self.secs_frame + self.secs_aim + self.secs_evolve + self.secs_render
    }
}

/// Read every reel record in a batch directory, newest first.
///
/// Tolerant by design: a malformed or half-written record is skipped rather
/// than losing the whole batch. A batch is written incrementally over hours,
/// and being killed partway through is an expected way for it to end.
pub fn load_batch(dir: &std::path::Path) -> Vec<ReelRecord> {
    let mut out: Vec<ReelRecord> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("json"))
        .filter_map(|p| std::fs::read_to_string(&p).ok())
        .filter_map(|t| serde_json::from_str::<ReelRecord>(&t).ok())
        .collect();
    out.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    out
}

/// Every batch directory under `reels/`, newest first.
pub fn list_batches() -> Vec<std::path::PathBuf> {
    let mut out: Vec<std::path::PathBuf> = std::fs::read_dir(reels_dir())
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    out.sort();
    out.reverse();
    out
}

/// Persist one reel record.
pub fn save_record(dir: &std::path::Path, rec: &ReelRecord) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let json = serde_json::to_string_pretty(rec)
        .map_err(|e| std::io::Error::other(e.to_string()))?;
    std::fs::write(dir.join(format!("{}.json", rec.id)), json)
}

/// Which source genomes a set of batches has already used.
///
/// A batch draws from the top of a ranked pool, so without this every run would
/// re-plan the same ten fractals.
pub fn already_reeled(batches: &[std::path::PathBuf]) -> std::collections::HashSet<String> {
    batches.iter()
        .flat_map(|d| load_batch(d))
        .map(|r| r.source)
        .collect()
}

/// One log line per framed genome — the audit trail for the framing rule.
pub fn frame_report(label: &str, fit: &FrameFit) -> String {
    format!(
        "[frame] {label}  scan {:.3}x  body {:.1}%  extent {:.4}  max_escape {:.1}  \
         → ({:.6},{:.6}) @ {:.4}x",
        fit.scan_zoom, fit.body_fraction * 100.0, fit.extent, fit.max_escape,
        fit.view.cx, fit.view.cy, fit.view.zoom
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        Config, DedupConfig, MassExtinctionConfig, OptimizationConfig, OutputConfig, RenderingConfig,
    };
    use crate::formula::{op, OpNode};

    fn test_config() -> Config {
        // Pins the render backend for this module's tests — `render_cpu` picks
        // the GPU tier only once `init_gpu` has run, and that flips mid-process
        // in a test binary. See `video_export::tests::chain_test_config`.
        #[cfg(feature = "wgpu-backend")]
        crate::render_gpu::init_gpu();
        Config {
            dedup: DedupConfig::default(),
            mass_extinction: MassExtinctionConfig::default(),
            rendering: RenderingConfig {
                default_width: 800, default_height: 800, max_iter: 300, bailout: 4.0,
                colormap: "turbo".into(),
                view_x_min: -2.0, view_x_max: 2.0, view_y_min: -2.0, view_y_max: 2.0,
            },
            optimization: OptimizationConfig::default(),
            output: OutputConfig {
                save_dir: "./fractals".into(), population_dir: "./populations".into(),
                ..Default::default()
            },
        }
    }

    fn node(op: u8, a: u8, b: u8) -> OpNode {
        OpNode { op, a, b, kre: 0.0, kim: 0.0 }
    }

    /// z² + c — the reference fractal with known geometry.
    fn mandelbrot() -> Genome {
        let mut g = Genome::default();
        g.program = vec![node(op::Z, 0, 0), node(op::C, 0, 0), node(op::SQR, 0, 0), node(op::ADD, 2, 1)];
        g.bailout_radius = 4.0;
        g
    }

    #[test]
    fn mandelbrot_is_framed_where_it_actually_lives() {
        let fit = auto_frame(&mandelbrot(), &test_config()).expect("the reference fractal must frame");
        // The set spans about x ∈ [-2, 0.5], y ∈ [-1.2, 1.2]: centred left of
        // the origin, on the real axis, roughly 2.5 units across.
        assert!((-1.1..=-0.3).contains(&fit.view.cx), "cx {} is not on the set", fit.view.cx);
        assert!(fit.view.cy.abs() < 0.25, "the set is symmetric about the real axis, got cy {}", fit.view.cy);
        assert!((1.5..=4.0).contains(&fit.extent), "extent {} does not match the known set", fit.extent);
        assert!(fit.view.zoom > 0.5 && fit.view.zoom < 3.0, "zoom {} is implausible", fit.view.zoom);
    }

    #[test]
    fn the_framed_view_itself_contains_the_whole_body() {
        // The invariant that matters more than any single number: whatever
        // auto_frame returns must, when rendered, hold the fractal clear of the
        // frame edge. Anything else is a shot that opens mid-crop.
        let (g, config) = (mandelbrot(), test_config());
        let fit = auto_frame(&g, &config).unwrap();
        let view = fit.view.to_view();
        let et = render_escape_times(
            &g, &config, &view, FRAME_RES, FRAME_RES, config.rendering.max_iter,
            needs_f64(&view, FRAME_RES), false,
        );
        let scan = scan_body(&et, FRAME_RES, &view, config.rendering.max_iter, BODY_ESCAPE_FRACTION);
        assert!(!scan.touches_edge, "the framed shot crops the fractal");
        assert!(scan.body_fraction > 0.05,
                "only {:.2}% body in the framed shot — framed too wide", scan.body_fraction * 100.0);
    }

    #[test]
    fn a_formula_that_never_escapes_is_refused_with_the_reason() {
        // z' = z (the identity): nothing ever leaves the bailout radius, so
        // every pixel is interior at every scale. There is no outside to open on.
        let mut g = mandelbrot();
        g.program = vec![node(op::Z, 0, 0), node(op::CONST, 0, 0), node(op::ADD, 0, 1)];
        let err = auto_frame(&g, &test_config()).expect_err("an all-interior formula has no shot");
        assert!(err.contains("fills the plane"), "{err}");
        assert!(err.contains("never escapes"),
                "an all-interior frame and a flat one are both uniform — the reason must \
                 say which: {err}");
    }

    #[test]
    fn a_uniform_frame_is_refused_with_the_reason() {
        // z' = z + 10⁶ escapes on the first iteration at the same time for
        // every pixel: a flat frame. The relative body threshold cannot tell
        // this from a fractal filling the plane — every pixel is at the maximum,
        // so every pixel is "body" — which is why degeneracy is its own check.
        let mut g = mandelbrot();
        g.program = vec![
            node(op::Z, 0, 0),
            OpNode { op: op::CONST, a: 0, b: 0, kre: 1.0e6, kim: 0.0 },
            node(op::ADD, 0, 1),
        ];
        let err = auto_frame(&g, &test_config()).expect_err("a flat frame has no shot");
        assert!(err.contains("nothing to film"), "{err}");
        assert!(err.contains("uniform"), "the reason must distinguish it from a full plane: {err}");
    }

    #[test]
    fn framing_finds_a_body_that_is_nowhere_near_the_origin() {
        // The whole reason the saved view is not trusted: an evolved formula's
        // body need not sit at the origin. Iterating z² + (c-3) is the standard
        // map in w = c-3, so the set is {c : c-3 ∈ M} = M shifted RIGHT by 3 —
        // centred near +2.25, well outside the classic -2..2 window.
        let mut g = mandelbrot();
        g.program = vec![
            node(op::Z, 0, 0),
            node(op::C, 0, 0),
            OpNode { op: op::CONST, a: 0, b: 0, kre: -3.0, kim: 0.0 },
            node(op::ADD, 1, 2),   // c - 3
            node(op::SQR, 0, 0),   // z²
            node(op::ADD, 4, 3),   // z² + (c-3)
        ];
        let fit = auto_frame(&g, &test_config()).expect("an off-centre body must still frame");
        assert!((1.7..=2.9).contains(&fit.view.cx),
                "the shifted set should be found near +2.25, got cx {}", fit.view.cx);
        assert!(fit.view.cy.abs() < 0.25, "still symmetric about the real axis, got {}", fit.view.cy);
        assert!(fit.scan_zoom < 1.0, "it must have widened to find it, stayed at {}", fit.scan_zoom);
    }

    #[test]
    fn both_descent_strategies_are_tried_by_default() {
        // Neither dominates — measured on one archived genome, unconstrained
        // reached 35.8 doublings where bounded managed 11.3, and on others only
        // the bounded descent survived the line check. Dropping either would
        // silently lose shots.
        let d = DestinationOpts::default();
        assert_eq!(d.drift_limits.len(), 2, "{:?}", d.drift_limits);
        assert!(d.drift_limits.iter().any(|v| v.is_infinite()), "the free descent must be tried");
        assert!(d.drift_limits.iter().any(|v| v.is_finite()), "the bounded descent must be tried");
    }

    #[test]
    fn a_good_enough_shot_stops_the_search_early() {
        let d = DestinationOpts::default();
        assert!(d.good_enough_doublings > 0.0 && d.good_enough_doublings.is_finite(),
                "an infinite threshold would always run every strategy");
        // The threshold must be reachable: the deepest shots measured on real
        // archive genomes travel 35-42 doublings.
        assert!(d.good_enough_doublings < 35.0, "got {}", d.good_enough_doublings);
    }

    #[test]
    fn the_drift_bound_keeps_the_target_inside_every_intermediate_frame() {
        // The geometric argument MAX_PLY_DRIFT rests on, checked numerically
        // rather than trusted: with per-ply drift capped at k half-extents and
        // zoom multiplying by at least z each ply, the total remaining drift
        // from any ply must stay within that ply's own half-extent — otherwise
        // the straight line leaves the frame partway down.
        let (k, z) = (MAX_PLY_DRIFT, 2.0f64);
        for start_ply in 0..25 {
            // Worst case: every remaining ply drifts the full allowance, all in
            // the same direction.
            let mut drift = 0.0f64;
            let mut half = 1.0f64;
            for _ in start_ply..40 {
                drift += k * half;
                half /= z;
            }
            assert!(drift <= 1.0 + 1e-9,
                    "from ply {start_ply}, worst-case drift {drift} exceeds the frame half-extent");
        }
    }

    #[test]
    fn the_scan_range_reaches_fractals_far_larger_than_the_classic_window() {
        // Two of ten top-ranked archive genomes ran past the widest scan.
        let widest = FRAME_SCAN_ZOOMS.last().copied().expect("at least one scan zoom");
        assert!(4.0 / widest >= 200.0, "widest scan spans only {} units", 4.0 / widest);
        assert!(FRAME_SCAN_ZOOMS.windows(2).all(|w| w[0] > w[1]),
                "scans must be ordered tightest-first");
    }

    #[test]
    fn a_sparse_opening_does_not_kill_the_line() {
        // The bug this function exists to fix, measured on real archive
        // genomes: four of five shots were rejected because their FIRST sampled
        // point — the deliberately wide establishing view — scored below a
        // richness floor calibrated for deep-zoom frames.
        let good = [false, false, false, true, true, true, true, true];
        assert_eq!(usable_span(&good, 1), Some((3, 7)));
    }

    #[test]
    fn a_dead_tail_cuts_the_shot_short() {
        let good = [true, true, true, false, false, false];
        assert_eq!(usable_span(&good, 1), Some((0, 2)), "the shot must stop where structure does");
    }

    #[test]
    fn one_sparse_frame_mid_shot_is_bridged_but_two_are_not() {
        // A single dip is within a 128px probe's noise; two in a row is the
        // structure genuinely running out.
        assert_eq!(usable_span(&[true, false, true, true], 1), Some((0, 3)));
        assert_eq!(usable_span(&[true, false, false, true, true], 1), Some((0, 0)));
        assert_eq!(usable_span(&[true, false, false, true, true], 2), Some((0, 4)));
    }

    #[test]
    fn a_line_with_nothing_rich_has_no_span() {
        assert_eq!(usable_span(&[false, false, false], 1), None);
        assert_eq!(usable_span(&[], 1), None);
    }

    #[test]
    fn the_richness_floor_adapts_to_a_dark_fractal_but_not_to_a_flat_one() {
        // The rule, stated three ways. `find_destination` computes it inline;
        // this pins the arithmetic so a change to either constant is visible.
        let floor = |peak: f32| {
            DestinationOpts::default().min_frame_richness
                .min(RICHNESS_RELATIVE * peak)
                .max(RICHNESS_ABSOLUTE_MIN)
        };
        // Bright fractal: the calibrated absolute floor is unchanged.
        assert!((floor(0.80) - 0.30).abs() < 1e-6, "got {}", floor(0.80));
        // Dark fractal (the measured case: peak 0.319, whole line rejected).
        let dark = floor(0.319);
        assert!(dark < 0.30 && dark > RICHNESS_ABSOLUTE_MIN,
                "a dark fractal must get a lower bar, got {dark}");
        assert!(0.319 > dark, "its own best frame must clear its own floor");
        // Genuinely flat: 60% of nothing is still nothing.
        assert!((floor(0.10) - RICHNESS_ABSOLUTE_MIN).abs() < 1e-6, "got {}", floor(0.10));
        assert!(0.10 < floor(0.10), "flat frames must still be refused");
    }

    #[test]
    fn the_richness_profile_marks_which_samples_cleared_the_floor() {
        let line = richness_profile(&[0.05, 0.4, 0.9, 0.2], 0.30);
        let bar = line.split('|').nth(1).expect("a bar between pipes");
        assert_eq!(bar.len(), 4);
        let cleared: Vec<bool> = bar.chars().map(|c| c.is_uppercase()).collect();
        assert_eq!(cleared, vec![false, true, true, false], "{line}");
        assert!(line.contains("max 0.900"), "{line}");
    }

    #[test]
    fn doublings_are_reported_instead_of_an_unreadable_fraction() {
        let d = Destination {
            end: CapturedView { cx: 0.0, cx_lo: 0.0, cy: 0.0, cy_lo: 0.0, zoom: 2.84e7, aspect: 1.0 },
            dd_wall: 4.69e12, descent_zoom: 3.8e12, descent_steps: 20,
            trimmed_to: Some(2.84e7), checked: 24, passed: 15, sparse_opening: 3,
            worst_richness: 0.346, ply_drift: f64::INFINITY,
        };
        // The real numbers from the first run, where the fraction printed 0.0%.
        assert!((d.doublings_short() - 17.3).abs() < 0.2, "got {}", d.doublings_short());
        assert!((d.doublings_travelled(0.2167) - 27.0).abs() < 0.2,
                "got {}", d.doublings_travelled(0.2167));
        let line = destination_report("x", 0.2167, &d);
        assert!(line.contains("doublings"), "{line}");
    }

    #[test]
    fn a_report_line_names_every_measurement_behind_the_decision() {
        let fit = auto_frame(&mandelbrot(), &test_config()).unwrap();
        let line = frame_report("abc123", &fit);
        for want in ["abc123", "scan", "body", "extent", "max_escape"] {
            assert!(line.contains(want), "report is missing `{want}`: {line}");
        }
    }
}
