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
pub const FRAME_SCAN_ZOOMS: &[f64] = &[2.0, 1.0, 0.5, 0.25, 0.1, 0.04];

/// A pixel counts as part of the fractal when it either never escapes, or takes
/// at least this fraction of the frame's own deepest escape time.
///
/// Relative rather than absolute because escape times are not comparable across
/// formulas — one genome's boundary may take 300 iterations where another's
/// takes 20, and an absolute floor would call the second one empty. Escape time
/// falls off roughly logarithmically with distance from the set, so this
/// threshold is not sharp; it is calibrated by looking at real output, not
/// derived.
pub const BODY_ESCAPE_FRACTION: f32 = 0.25;

/// How much of the frame the fractal should occupy in the opening shot.
pub const FRAME_FILL: f64 = 0.85;

/// A body pixel this close to the border means the fractal continues past the
/// frame, so this scan cannot have measured all of it.
pub const EDGE_MARGIN_PX: u32 = 2;

/// Above this body fraction at the widest scan, there is no meaningful
/// "outside" to open on — the formula fills the plane.
pub const MAX_BODY_FRACTION: f32 = 0.60;

/// Below this, the scan found no fractal at all.
pub const MIN_BODY_FRACTION: f32 = 0.001;

/// What a framing scan measured, not just what it decided.
///
/// The measurements are reported because the framing rule is the one piece of
/// this pipeline with no prior art in the codebase to inherit calibration from:
/// when a shot opens badly, these numbers say whether the threshold, the scan
/// range, or the fill fraction was wrong.
#[derive(Clone, Copy, Debug)]
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
fn scan_body(et: &[f32], res: u32, view: &View, compute_iter: u32) -> Scan {
    let n = (res * res) as usize;
    let max_escape = et.iter().copied().fold(0.0f32, f32::max);
    // Interior pixels sit exactly at the cap (see `fractal::dag_escape_pixel`),
    // so they are counted directly rather than via the relative threshold —
    // a fractal that is ALL interior has no escaping pixels to take a fraction
    // of, and would otherwise measure as empty.
    let interior = compute_iter as f32 - 0.5;
    let thresh = (BODY_ESCAPE_FRACTION * max_escape).min(interior);

    let (xmin, xmax, ymin, ymax) = view.bounds();
    let (mut lo_x, mut hi_x, mut lo_y, mut hi_y) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
    let mut count = 0usize;
    let mut interior_count = 0usize;
    let mut touches_edge = false;
    let margin = EDGE_MARGIN_PX.min(res / 2);

    for i in 0..n.min(et.len()) {
        if et[i] >= interior {
            interior_count += 1;
        }
        if et[i] < thresh {
            continue;
        }
        count += 1;
        let px = (i as u32) % res;
        let py = (i as u32) / res;
        if px < margin || py < margin || px + margin >= res || py + margin >= res {
            touches_edge = true;
        }
        let fx = xmin + (px as f64 / (res.max(2) - 1) as f64) * (xmax - xmin);
        let fy = ymin + (py as f64 / (res.max(2) - 1) as f64) * (ymax - ymin);
        lo_x = lo_x.min(fx);
        hi_x = hi_x.max(fx);
        lo_y = lo_y.min(fy);
        hi_y = hi_y.max(fy);
    }

    Scan {
        body_fraction: count as f32 / n as f32,
        bbox: (count > 0).then_some((lo_x, hi_x, lo_y, hi_y)),
        touches_edge,
        max_escape,
        interior_fraction: interior_count as f32 / n as f32,
        degenerate: crate::fitness::is_degenerate(et),
    }
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
    let compute_iter = config.rendering.max_iter;
    let mut widest: Option<(f64, Scan)> = None;

    for &zoom in FRAME_SCAN_ZOOMS {
        let view = View::new_square(0.0, 0.0, zoom);
        let et = render_escape_times(
            g, config, &view, FRAME_RES, FRAME_RES, compute_iter,
            needs_f64(&view, FRAME_RES), false,
        );
        let scan = scan_body(&et, FRAME_RES, &view, compute_iter);
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
                zoom: 4.0 / (extent / FRAME_FILL),
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
        let scan = scan_body(&et, FRAME_RES, &view, config.rendering.max_iter);
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
    fn a_report_line_names_every_measurement_behind_the_decision() {
        let fit = auto_frame(&mandelbrot(), &test_config()).unwrap();
        let line = frame_report("abc123", &fit);
        for want in ["abc123", "scan", "body", "extent", "max_escape"] {
            assert!(line.contains(want), "report is missing `{want}`: {line}");
        }
    }
}
