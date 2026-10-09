//! The single per-frame render-parameter construction function shared by
//! the live preview (`anim_viewer.rs`) and the batch renderer
//! (`explorer.rs`'s `quat-render-timeline` CLI command) —
//! animation-viewer plan, Phase 7.
//!
//! Section 6 of the plan: "no other call site anywhere constructs
//! `RaymarchDagParams`/`RaymarchCamera` from timeline state by hand." This
//! also fixes a real, pre-existing latent bug-risk the plan's own research
//! flagged: before this module existed, `quat_viewer.rs` and the CLI each
//! hardcoded their own copies of framing constants (`domain_radius`,
//! `fov_deg`, march steps, epsilons) that "happened to agree" only by
//! hand-kept-in-sync literals. Everything the animation viewer renders —
//! live or batched — now goes through exactly one function.

use crate::anim_timeline::{accumulated_angle, evaluate_axis_track, evaluate_difference, evaluate_slide_iteration, AnimationTimeline, AxisBound, BoundingBox};
use crate::orient::OrientQuat;
use crate::quat_dag::{QuatDagFormula, RaymarchDagParams};
use crate::quat_fractal::TimeAxis;
use crate::quat_raymarch::RaymarchCamera;

/// `max_iter`/`aa` — separated from the rest of `RaymarchDagParams` since
/// both the live preview (which drops these during scrub/play/orbit-drag,
/// per the confirmed plan answer on preview quality) and the batch
/// renderer (always full quality) need to plug in a quality tier without
/// touching anything else `build_frame_params` computes.
#[derive(Copy, Clone, Debug)]
pub struct RenderQuality {
    pub max_iter: u32,
    pub aa: u32,
}

/// Fixed vertical FOV every animation-viewer render uses — not user
/// adjustable in v1 (matches `quat_viewer.rs`'s own fixed 45°).
pub const FOV_DEG: f64 = 45.0;

/// How far back the camera needs to sit so a box of the given
/// characteristic scale occupies a sensible, well-margined fraction of
/// frame — the aspect-aware framing calibration `explorer.rs`'s own
/// `recommended_orbit_radius` and `quat_viewer.rs`'s copy of it already
/// use (duplicated here as the lib-side canonical copy now that a batch
/// CLI command and the live viewer both need it; see that function's own
/// doc comment in `explorer.rs` for the FILL_FRAC calibration derivation).
pub fn recommended_orbit_radius(characteristic_scale: f64, fov_deg: f64, width: u32, height: u32) -> f64 {
    const FILL_FRAC: f64 = 0.82;
    let aspect = width as f64 / height.max(1) as f64;
    let half_fov_y = (fov_deg / 2.0).to_radians();
    let half_fov_x = (half_fov_y.tan() * aspect).atan();
    let tight_half_fov = half_fov_y.min(half_fov_x);
    let target_angular_half_size = FILL_FRAC * tight_half_fov;
    characteristic_scale / target_angular_half_size.sin()
}

/// `pivot` is the box center after this frame's X/Y/Z Translation-track
/// contributions — the camera orbits AROUND the pivot, not always the
/// origin, so a Translation clip on X moving the box also carries the
/// camera's look-at point with it. `orientation` is this frame's X->Y->Z
/// Rotation-track composition (`anim_timeline::camera_orientation`)
/// applied to the manual orbit's own Cartesian offset — with no active
/// Translation/Rotation clips, pivot is (0,0,0) and orientation is
/// `OrientQuat::IDENTITY` (a proven no-op, see `orient.rs`'s own tests),
/// so this reduces exactly to a plain manual orbit.
///
/// `up_hint` is rotated by `orientation` too, not left fixed at world
/// `(0,1,0)` — Carl reported (2026-09-22) a render where "the 3D position
/// is consistent for a while, but there appear to be a jump on the
/// rotation axis, the point of view changes very suddenly." Root cause: a
/// FIXED world up_hint against an EYE that orbits under an animating
/// Rotation track sweeps `forward = target - eye` through a wide range of
/// directions relative to that fixed hint, including past-vertical — and
/// `look_at_basis` (`quat_motion.rs`) discretely SWITCHES its reference
/// axis to `(0,0,1)` once `forward` gets within ~18° of `up_hint`
/// (`dot(...).abs() >= 0.95`), with no blending across that threshold. As
/// `forward` crosses it, `right`/`up` snap to a different basis in one
/// frame — exactly a sudden, position-unaffected viewpoint jump. Rotating
/// `up_hint` by the same `orientation` keeps it co-rotating with the eye
/// offset, so the pole is only reachable via the manual yaw/pitch orbit
/// itself (the same pre-existing, narrower edge case a plain orbit with no
/// Rotation clips already had), not newly introduced by an animating
/// Rotation track.
pub fn camera_from_orbit(yaw: f64, pitch: f64, distance: f64, fov_y: f64, pivot: (f64, f64, f64), orientation: &OrientQuat) -> RaymarchCamera {
    let cp = pitch.cos();
    let base_offset = (distance * cp * yaw.sin(), distance * pitch.sin(), -distance * cp * yaw.cos());
    let offset = orientation.rotate(base_offset);
    let eye = (pivot.0 + offset.0, pivot.1 + offset.1, pivot.2 + offset.2);
    let up_hint = orientation.rotate((0.0, 1.0, 0.0));
    RaymarchCamera { eye, target: pivot, up_hint, fov_y }
}

/// The manual orbit's own (forward, right, up) unit basis at the given
/// yaw/pitch, deliberately IGNORING any Rotation-track `orientation` — used
/// by the live preview's middle-drag PAN control (`anim_viewer.rs`,
/// MeshLab-style: middle-drag moves `camera_pivot_offset`) to convert a 2D
/// screen drag into a 3D world-space offset. Panning is a manual,
/// authoring-time framing adjustment, not an animated timeline behavior —
/// same reasoning `camera_yaw`/`pitch`/`distance` themselves already have
/// for staying orientation-free. Shares `look_at_basis` with the actual
/// renderer (`quat_raymarch.rs`) so the pan directions always match what's
/// on screen, not a separately hand-derived (and possibly mismatched) basis.
pub fn manual_orbit_basis(yaw: f64, pitch: f64) -> ((f64, f64, f64), (f64, f64, f64), (f64, f64, f64)) {
    let cp = pitch.cos();
    let base_offset = (cp * yaw.sin(), pitch.sin(), -cp * yaw.cos());
    let forward = crate::quat_motion::normalize((-base_offset.0, -base_offset.1, -base_offset.2));
    let (right, up) = crate::quat_motion::look_at_basis(forward, (0.0, 1.0, 0.0));
    (forward, right, up)
}

/// Projects a world-space point through `cam`'s actual perspective
/// projection into normalized (x right, y UP) screen coordinates in
/// roughly `[-1,1]` — for a UI overlay to convert into pixel coordinates
/// over the rendered image, never the GPU raymarch path itself. `None`
/// when the point is behind the camera (nothing sensible to draw).
///
/// Exists so `anim_viewer.rs` can draw a marker at the orbit pivot WHILE
/// the user drags (left-button orbit or middle-button pan) — Carl,
/// 2026-09-23: "when changing the origin, it often become very hard to
/// monitor rotation, the center of rotation become unknown or infered at
/// best... let the user see where is the origin and where he is situated
/// next to it (also like meshlab browsing)."
pub fn project_to_screen_ndc(cam: &RaymarchCamera, point: (f64, f64, f64), aspect: f64) -> Option<(f64, f64)> {
    let forward = crate::quat_motion::normalize((
        cam.target.0 - cam.eye.0, cam.target.1 - cam.eye.1, cam.target.2 - cam.eye.2,
    ));
    let (right, up) = crate::quat_motion::look_at_basis(forward, cam.up_hint);
    let v = (point.0 - cam.eye.0, point.1 - cam.eye.1, point.2 - cam.eye.2);
    let depth = crate::quat_motion::dot(v, forward);
    if depth <= 1e-6 {
        return None;
    }
    let tan_y = (cam.fov_y * 0.5).tan().max(1e-6);
    let tan_x = tan_y * aspect;
    let ndc_x = crate::quat_motion::dot(v, right) / depth / tan_x;
    let ndc_y = crate::quat_motion::dot(v, up) / depth / tan_y;
    Some((ndc_x, ndc_y))
}

/// Largest half-extent across X/Y/Z — the "characteristic scale" that
/// stands in for a single `domain_radius` wherever epsilon derivations or
/// the initial camera distance need one scalar (`RaymarchDagParams::box_bounds`'s
/// doc comment explains why `domain_radius` still carries this role even
/// when box-bounded).
pub fn characteristic_scale(bounds: &BoundingBox) -> f64 {
    (bounds.x.max - bounds.x.min).max(bounds.y.max - bounds.y.min).max(bounds.z.max - bounds.z.min).max(1e-6) * 0.5
}

/// This frame's camera orientation from the X/Y/Z tracks' Rotation clips —
/// section 2 of the plan's documented, FIXED composition order (rotation
/// is non-commutative, so this order is a stated convention, not
/// arbitrary): apply X, then Y, then Z.
pub fn camera_orientation(timeline: &AnimationTimeline, t: f64) -> OrientQuat {
    let qx = OrientQuat::from_axis_angle((1.0, 0.0, 0.0), accumulated_angle(&timeline.x_track, t).to_radians());
    let qy = OrientQuat::from_axis_angle((0.0, 1.0, 0.0), accumulated_angle(&timeline.y_track, t).to_radians());
    let qz = OrientQuat::from_axis_angle((0.0, 0.0, 1.0), accumulated_angle(&timeline.z_track, t).to_radians());
    qz.mul(&qy).mul(&qx)
}

/// The box (min, max), camera pivot, and "characteristic scale" (for
/// epsilon derivations) after applying this frame's X/Y/Z track
/// contributions at authored time `t` — Translation clips sum into that
/// axis's pivot offset, Scale clips multiply that axis's half-extent
/// around its own (possibly translated) center; see
/// `anim_timeline::evaluate_axis_track`'s doc comment for the exact
/// composition rules. With every track empty, this reduces exactly to the
/// static `timeline.bounds`.
pub fn effective_box_and_pivot(timeline: &AnimationTimeline, t: f64) -> ((f64, f64, f64), (f64, f64, f64), (f64, f64, f64), f64) {
    let b = &timeline.bounds;
    let axis = |bound: &AxisBound, track: &crate::anim_timeline::AxisTimeline| -> (f64, f64, f64) {
        let contrib = evaluate_axis_track(track, t);
        let center = (bound.min + bound.max) * 0.5 + contrib.translation;
        let half = (bound.max - bound.min) * 0.5 * contrib.scale;
        (center - half, center + half, center)
    };
    let (xmin, xmax, xc) = axis(&b.x, &timeline.x_track);
    let (ymin, ymax, yc) = axis(&b.y, &timeline.y_track);
    let (zmin, zmax, zc) = axis(&b.z, &timeline.z_track);
    let box_min = (xmin, ymin, zmin);
    let box_max = (xmax, ymax, zmax);
    let pivot = (xc, yc, zc);
    let scale = (xmax - xmin).max(ymax - ymin).max(zmax - zmin).max(1e-6) * 0.5;
    (box_min, box_max, pivot, scale)
}

/// The one function both the live preview and the batch renderer call to
/// go from "timeline state at authored time `t`" to "what to actually
/// render this frame". `bailout_radius` comes from the loaded genome (a
/// per-genome constant, not something the timeline itself animates).
///
/// This function only ever reaches `AnimParam` values indirectly, through
/// `evaluate_axis_track`/`evaluate_difference` — it never calls
/// `eval_anim_param` itself. That's why `AnimParam`'s "Ramp" mode (a
/// clip-relative from→to transition, added alongside the pre-existing
/// constant/wave-drive modes) needed zero changes here: any new evaluation
/// mode added inside `anim_timeline.rs` flows through to both the live
/// preview and the batch CLI renderer automatically, by construction, not
/// by remembering to update this function too. Keep it that way — never
/// add a second, parallel place that reads `AnimParam`/`EffectClip` state.
pub fn build_frame_params<'a>(
    timeline: &AnimationTimeline,
    formula: QuatDagFormula<'a>,
    t: f64,
    bailout_radius: f64,
    quality: RenderQuality,
) -> (RaymarchDagParams<'a>, RaymarchCamera) {
    let (box_min, box_max, pivot, domain_radius) = effective_box_and_pivot(timeline, t);
    let t_track_offset = evaluate_axis_track(&timeline.t_track, t).translation;
    let time_val = timeline.t_value + t_track_offset;
    let orientation = camera_orientation(timeline, t);
    // The manual pan offset shifts WHERE the orbit is centered, added in
    // world space before any Rotation-track orientation — see
    // `AnimationTimeline::camera_pivot_offset`'s doc comment.
    let o = timeline.camera_pivot_offset;
    let panned_pivot = (pivot.0 + o.0, pivot.1 + o.1, pivot.2 + o.2);
    let cam = camera_from_orbit(timeline.camera_yaw, timeline.camera_pitch, timeline.camera_distance, FOV_DEG.to_radians(), panned_pivot, &orientation);
    // Computed AFTER `cam`: a `CameraFixed` plane needs this frame's actual
    // eye/target/up_hint to resolve into world space — see
    // `crate::anim_timeline::DifferencePlane`'s doc comment.
    let clip_plane = evaluate_difference(&timeline.effects_lanes, t).map(|d| match d {
        crate::anim_timeline::DifferencePlane::WorldFixed { pos, normal } =>
            ((pos[0], pos[1], pos[2]), (normal[0], normal[1], normal[2])),
        crate::anim_timeline::DifferencePlane::CameraFixed { camera_offset, normal } => {
            let forward = crate::quat_motion::normalize((
                cam.target.0 - cam.eye.0, cam.target.1 - cam.eye.1, cam.target.2 - cam.eye.2,
            ));
            let (right, up) = crate::quat_motion::look_at_basis(forward, cam.up_hint);
            let wp = (
                cam.eye.0 + right.0 * camera_offset[0] + up.0 * camera_offset[1] + forward.0 * camera_offset[2],
                cam.eye.1 + right.1 * camera_offset[0] + up.1 * camera_offset[1] + forward.1 * camera_offset[2],
                cam.eye.2 + right.2 * camera_offset[0] + up.2 * camera_offset[1] + forward.2 * camera_offset[2],
            );
            let wn = (
                right.0 * normal[0] + up.0 * normal[1] + forward.0 * normal[2],
                right.1 * normal[0] + up.1 * normal[1] + forward.1 * normal[2],
                right.2 * normal[0] + up.2 * normal[1] + forward.2 * normal[2],
            );
            (wp, wn)
        }
    });
    let slide_iter = evaluate_slide_iteration(&timeline.effects_lanes, t);
    let params = RaymarchDagParams {
        formula,
        time_axis: TimeAxis::C,
        time_val,
        domain_radius,
        box_bounds: Some((box_min, box_max)),
        axis_assignment: Some(timeline.axis_assignment),
        clip_plane,
        slide_iter,
        // Resolved by the caller (`anim_viewer.rs`'s render loop, and
        // `explorer.rs`'s batch renderer) from an actual probe render's
        // own escape-time distribution — see
        // `percentile_escape_time_cutoff` below. This function only ever
        // builds params from timeline state, never renders, so it has no
        // distribution to resolve a cutoff from yet.
        hide_above: None,
        max_iter: quality.max_iter,
        bailout: bailout_radius,
        max_march_steps: 200,
        hit_epsilon: domain_radius * 1e-4,
        step_safety: 0.8,
        light_dir: (0.5, 0.8, 0.3),
        normal_eps: domain_radius * 1e-3,
        color_probe_offset: domain_radius * 1e-2,
        aa: quality.aa,
    };
    (params, cam)
}

/// The raw escape-time value at the boundary of the top `fraction` of
/// visible hit pixels, by rank. Mirrors `colormap::empirical_cdf`'s own
/// population exactly (hit pixels — `shading > 0.0` — whose `color_t <
/// max_iter`, i.e. actually escaped) so the resulting cutoff precisely
/// matches "whatever the equalized colormap currently puts at the very
/// top of the palette" for THIS frame, regardless of what raw iteration
/// count that happens to be — see `RaymarchDagParams::hide_above`'s doc
/// comment for why a fixed absolute number can't do this reliably.
///
/// Intended use (the "hide top N%" control): render once with
/// `hide_above: None` to get `shading`/`color_t`, call this on that
/// result, then render again with `hide_above` set to the returned
/// cutoff. `fraction` is in [0,1] (e.g. 0.15 = hide the top 15%).
/// Returns `None` when `fraction <= 0` or there's nothing to compute a
/// percentile from (an empty/background-only frame) — callers should
/// treat `None` the same as leaving `hide_above` at `None`.
pub fn percentile_escape_time_cutoff(shading: &[f32], color_t: &[f32], max_iter: u32, fraction: f64) -> Option<f64> {
    if fraction <= 0.0 || shading.len() != color_t.len() {
        return None;
    }
    let mut escaped: Vec<f32> = shading.iter().zip(color_t.iter())
        .filter(|&(&s, &t)| s > 0.0 && (t as u32) < max_iter)
        .map(|(_, &t)| t)
        .collect();
    if escaped.is_empty() {
        return None;
    }
    escaped.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = escaped.len();
    let idx = (((1.0 - fraction.clamp(0.0, 1.0)) * n as f64).floor() as usize).min(n - 1);
    Some(escaped[idx] as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formula::OpNode;

    const MANDEL_PROG: [OpNode; 4] = [
        OpNode { op: crate::formula::op::Z, a: 0, b: 0, kre: 0.0, kim: 0.0 },
        OpNode { op: crate::formula::op::C, a: 0, b: 0, kre: 0.0, kim: 0.0 },
        OpNode { op: crate::formula::op::SQR, a: 0, b: 0, kre: 0.0, kim: 0.0 },
        OpNode { op: crate::formula::op::ADD, a: 2, b: 1, kre: 0.0, kim: 0.0 },
    ];

    fn formula() -> QuatDagFormula<'static> {
        QuatDagFormula { prog: &MANDEL_PROG, warp: &[], julia: false, jc: (0.0, 0.0), phoenix: (0.0, 0.0) }
    }

    #[test]
    fn defaults_produce_a_sensible_box_bounded_static_frame() {
        let tl = AnimationTimeline::new(0);
        let (params, cam) = build_frame_params(&tl, formula(), 0.0, 4.0, RenderQuality { max_iter: 40, aa: 1 });
        assert_eq!(params.box_bounds, Some(((-1.6, -1.6, -1.6), (1.6, 1.6, 1.6))));
        assert!(params.clip_plane.is_none());
        assert_eq!(params.time_val, 0.0);
        // Default camera looks at the origin from the default distance.
        assert_eq!(cam.target, (0.0, 0.0, 0.0));
        let dist = (cam.eye.0 * cam.eye.0 + cam.eye.1 * cam.eye.1 + cam.eye.2 * cam.eye.2).sqrt();
        assert!((dist - tl.camera_distance).abs() < 1e-9);
    }

    #[test]
    fn project_to_screen_ndc_puts_the_target_at_the_center() {
        let cam = RaymarchCamera { eye: (0.0, 0.0, -5.0), target: (0.0, 0.0, 0.0), up_hint: (0.0, 1.0, 0.0), fov_y: FOV_DEG.to_radians() };
        let (x, y) = project_to_screen_ndc(&cam, cam.target, 1.0).expect("target is in front of the eye");
        assert!(x.abs() < 1e-9 && y.abs() < 1e-9, "the look-at target must project to dead center: ({x},{y})");
    }

    #[test]
    fn project_to_screen_ndc_is_none_behind_the_camera() {
        let cam = RaymarchCamera { eye: (0.0, 0.0, -5.0), target: (0.0, 0.0, 0.0), up_hint: (0.0, 1.0, 0.0), fov_y: FOV_DEG.to_radians() };
        assert!(project_to_screen_ndc(&cam, (0.0, 0.0, -10.0), 1.0).is_none(), "a point behind the eye has nothing sensible to project to");
    }

    #[test]
    fn project_to_screen_ndc_puts_a_right_offset_point_on_the_positive_x_side() {
        let cam = RaymarchCamera { eye: (0.0, 0.0, -5.0), target: (0.0, 0.0, 0.0), up_hint: (0.0, 1.0, 0.0), fov_y: FOV_DEG.to_radians() };
        // Slightly to the world +X side of the target, still well within FOV.
        let (x, y) = project_to_screen_ndc(&cam, (0.3, 0.0, 0.0), 1.0).expect("in front of the eye");
        assert!(x > 0.0, "a point to the +right of the view direction must land on the +x side: x={x}");
        assert!(y.abs() < 1e-9, "no vertical offset was introduced: y={y}");
    }

    #[test]
    fn fixed_difference_plane_tracks_the_camera_as_it_orbits() {
        use crate::anim_timeline::{AxisTimeline, Effect, EffectClip};
        let mut tl = AnimationTimeline::new(0);
        tl.camera_pitch = 0.0; // isolate yaw for a clean basis, per manual_orbit_basis's own test.
        tl.effects_lanes.push(AxisTimeline { clips: vec![EffectClip {
            effect: Effect::FixedDifference {
                camera_offset: [AnimParam::constant(0.0), AnimParam::constant(0.0), AnimParam::constant(2.0)],
                normal: [AnimParam::constant(0.0), AnimParam::constant(0.0), AnimParam::constant(1.0)],
            },
            start_s: 0.0, end_s: 100.0, label: String::new(),
        }] });

        // At yaw=0: eye=(0,0,-distance), forward=(0,0,1) — the plane sits
        // 2 units in front of the eye along that forward direction.
        let (params0, cam0) = build_frame_params(&tl, formula(), 0.0, 4.0, RenderQuality { max_iter: 40, aa: 1 });
        let (pos0, normal0) = params0.clip_plane.expect("a FixedDifference clip is active");
        let expected_pos0 = (cam0.eye.0, cam0.eye.1, cam0.eye.2 + 2.0);
        assert!((pos0.0 - expected_pos0.0).abs() < 1e-9 && (pos0.1 - expected_pos0.1).abs() < 1e-9 && (pos0.2 - expected_pos0.2).abs() < 1e-9,
            "pos0={pos0:?} expected={expected_pos0:?}");
        assert!((normal0.0 - 0.0).abs() < 1e-9 && (normal0.1 - 0.0).abs() < 1e-9 && (normal0.2 - 1.0).abs() < 1e-9, "normal0={normal0:?}");

        // Orbit 90 degrees — a WorldFixed plane would stay put; this one
        // must rotate WITH the camera, staying "2 units in front" of the
        // NEW eye position along the NEW forward direction.
        tl.camera_yaw = std::f64::consts::FRAC_PI_2;
        let (params1, cam1) = build_frame_params(&tl, formula(), 0.0, 4.0, RenderQuality { max_iter: 40, aa: 1 });
        let (pos1, normal1) = params1.clip_plane.expect("still active after orbiting");
        assert!((normal1.0 - normal0.0).abs() > 0.5 || (normal1.2 - normal0.2).abs() > 0.5,
            "the normal must have rotated with the camera, not stayed fixed: normal0={normal0:?} normal1={normal1:?}");
        let expected_pos1 = (
            cam1.eye.0 + (cam1.target.0 - cam1.eye.0) / tl.camera_distance * 2.0,
            cam1.eye.1 + (cam1.target.1 - cam1.eye.1) / tl.camera_distance * 2.0,
            cam1.eye.2 + (cam1.target.2 - cam1.eye.2) / tl.camera_distance * 2.0,
        );
        assert!((pos1.0 - expected_pos1.0).abs() < 1e-6 && (pos1.1 - expected_pos1.1).abs() < 1e-6 && (pos1.2 - expected_pos1.2).abs() < 1e-6,
            "pos1={pos1:?} expected={expected_pos1:?}");
    }

    #[test]
    fn camera_pivot_offset_shifts_both_eye_and_target_together() {
        let mut tl = AnimationTimeline::new(0);
        tl.camera_pivot_offset = (1.0, 2.0, 3.0);
        let (_, cam) = build_frame_params(&tl, formula(), 0.0, 4.0, RenderQuality { max_iter: 40, aa: 1 });
        assert_eq!(cam.target, (1.0, 2.0, 3.0), "panning moves the look-at point");
        // A pure pan is a translation of the whole rig — eye moves by the
        // exact same offset as target, so eye-minus-target (and therefore
        // distance/direction) is unchanged from the no-offset case.
        let (_, cam0) = build_frame_params(&AnimationTimeline::new(0), formula(), 0.0, 4.0, RenderQuality { max_iter: 40, aa: 1 });
        let rel = (cam.eye.0 - cam.target.0, cam.eye.1 - cam.target.1, cam.eye.2 - cam.target.2);
        let rel0 = (cam0.eye.0 - cam0.target.0, cam0.eye.1 - cam0.target.1, cam0.eye.2 - cam0.target.2);
        assert!((rel.0 - rel0.0).abs() < 1e-9 && (rel.1 - rel0.1).abs() < 1e-9 && (rel.2 - rel0.2).abs() < 1e-9);
    }

    #[test]
    fn manual_orbit_basis_is_the_standard_frame_at_zero_yaw_pitch() {
        let (forward, right, up) = manual_orbit_basis(0.0, 0.0);
        let close = |a: (f64, f64, f64), b: (f64, f64, f64)| {
            (a.0 - b.0).abs() < 1e-9 && (a.1 - b.1).abs() < 1e-9 && (a.2 - b.2).abs() < 1e-9
        };
        assert!(close(forward, (0.0, 0.0, 1.0)), "forward: {forward:?}");
        assert!(close(right, (1.0, 0.0, 0.0)), "right: {right:?}");
        assert!(close(up, (0.0, 1.0, 0.0)), "up: {up:?}");
    }

    #[test]
    fn t_value_and_track_offset_both_reach_time_val() {
        let mut tl = AnimationTimeline::new(0);
        tl.t_value = 0.7;
        let (params, _) = build_frame_params(&tl, formula(), 0.0, 4.0, RenderQuality { max_iter: 40, aa: 1 });
        assert_eq!(params.time_val, 0.7, "t_value alone (no track clips) must reach time_val unchanged");
    }

    #[test]
    fn persisted_camera_orbit_is_reproduced_exactly() {
        let mut tl = AnimationTimeline::new(0);
        tl.camera_yaw = 1.2;
        tl.camera_pitch = -0.3;
        tl.camera_distance = 9.0;
        let (_, cam) = build_frame_params(&tl, formula(), 0.0, 4.0, RenderQuality { max_iter: 40, aa: 1 });
        let expected = camera_from_orbit(1.2, -0.3, 9.0, FOV_DEG.to_radians(), (0.0, 0.0, 0.0), &OrientQuat::IDENTITY);
        assert!((cam.eye.0 - expected.eye.0).abs() < 1e-9 && (cam.eye.1 - expected.eye.1).abs() < 1e-9 && (cam.eye.2 - expected.eye.2).abs() < 1e-9);
    }

    // ── moved from anim_viewer.rs's camera_orientation_tests when
    // camera_orientation/camera_from_orbit moved into this module ──────

    use crate::anim_timeline::{AnimParam, AxisTimeline, Drive, Effect, EffectClip, WaveShape};

    fn rotation_clip(deg_per_s: f64, end_s: f64) -> EffectClip {
        EffectClip { effect: Effect::Rotation { degrees_per_second: AnimParam::constant(deg_per_s) }, start_s: 0.0, end_s, label: String::new() }
    }

    #[test]
    fn no_rotation_clips_is_the_identity_orientation() {
        let tl = AnimationTimeline::new(0);
        let o = camera_orientation(&tl, 3.7);
        assert_eq!(o, OrientQuat::IDENTITY);
    }

    #[test]
    fn a_z_track_rotation_clip_spins_the_orbit_offset_around_z() {
        let mut tl = AnimationTimeline::new(0);
        tl.z_track = AxisTimeline { clips: vec![rotation_clip(90.0, 100.0) /* 90 deg/s */] };
        // At t=1s, accumulated_angle = 90 degrees around Z.
        let o = camera_orientation(&tl, 1.0);
        let r = o.rotate((1.0, 0.0, 0.0));
        assert!((r.0).abs() < 1e-6 && (r.1 - 1.0).abs() < 1e-6, "expected (0,1,0)-ish, got {r:?}");
    }

    /// Regression for Carl's 2026-09-22 report: a smoothly-moving eye
    /// (position "consistent for a while") whose viewpoint nonetheless
    /// "jumps very suddenly." Root cause was `up_hint` staying fixed at
    /// world (0,1,0) while `eye` orbited under an animating orientation —
    /// see `camera_from_orbit`'s doc comment for the full mechanism
    /// (`look_at_basis`'s discrete reference-axis switch near the pole).
    /// `up_hint` must co-rotate with the eye offset by the exact same
    /// quaternion, for any non-identity orientation.
    #[test]
    fn up_hint_rotates_with_the_orbit_offset_instead_of_staying_fixed_at_world_up() {
        let mut tl = AnimationTimeline::new(0);
        tl.z_track = AxisTimeline { clips: vec![rotation_clip(90.0, 100.0)] };
        let orientation = camera_orientation(&tl, 1.0); // 90 degrees around Z, as above.
        let cam = camera_from_orbit(0.0, 0.0, 5.0, FOV_DEG.to_radians(), (0.0, 0.0, 0.0), &orientation);
        let expected = orientation.rotate((0.0, 1.0, 0.0));
        assert!(
            (cam.up_hint.0 - expected.0).abs() < 1e-9
                && (cam.up_hint.1 - expected.1).abs() < 1e-9
                && (cam.up_hint.2 - expected.2).abs() < 1e-9,
            "up_hint must be rotated by the same orientation as the eye offset, got {:?} expected {expected:?}",
            cam.up_hint
        );
        // And it must actually have MOVED off world (0,1,0) — otherwise this
        // test would pass vacuously even with the old, fixed-up_hint code.
        assert!((cam.up_hint.1 - 1.0).abs() > 1e-3, "up_hint must no longer be fixed at world (0,1,0): {:?}", cam.up_hint);
    }

    /// With no Rotation clips (`OrientQuat::IDENTITY`), up_hint must stay
    /// exactly world (0,1,0) — the fix must not change behavior for the
    /// overwhelmingly common case of a plain manual orbit.
    #[test]
    fn identity_orientation_keeps_up_hint_at_world_up() {
        let cam = camera_from_orbit(1.2, -0.3, 9.0, FOV_DEG.to_radians(), (0.0, 0.0, 0.0), &OrientQuat::IDENTITY);
        assert_eq!(cam.up_hint, (0.0, 1.0, 0.0));
    }

    /// The plan's own elliptic-orbit proof of concept (section 2): a
    /// rotation transform (Z track) combined with a translation function
    /// of sin(t) (X/Y tracks, unequal amplitudes) — proves the pivot the
    /// orbit traces is a genuine ellipse, not a circle.
    #[test]
    fn elliptic_orbit_poc_pivot_traces_a_genuine_ellipse_not_a_circle() {
        let x_amp = 0.5;
        let y_amp = 0.25;
        let freq = 0.1; // period = 10s
        let x_track = AxisTimeline { clips: vec![EffectClip {
            effect: Effect::Translation { offset: AnimParam { constant: 0.0, drive: Some(Drive { shape: WaveShape::Sine, amp: x_amp, freq, phase: 0.0 }), ramp: None } },
            start_s: 0.0, end_s: 10.0, label: String::new(),
        }] };
        let y_track = AxisTimeline { clips: vec![EffectClip {
            effect: Effect::Translation { offset: AnimParam { constant: 0.0, drive: Some(Drive { shape: WaveShape::Sine, amp: y_amp, freq, phase: 0.25 }), ramp: None } },
            start_s: 0.0, end_s: 10.0, label: String::new(),
        }] };

        let mut max_x = 0.0f64;
        let mut max_y = 0.0f64;
        let mut samples = Vec::new();
        for i in 0..100 {
            let t = i as f64 * 0.1;
            let x = evaluate_axis_track(&x_track, t).translation;
            let y = evaluate_axis_track(&y_track, t).translation;
            max_x = max_x.max(x.abs());
            max_y = max_y.max(y.abs());
            samples.push((x, y));
        }
        assert!((max_x - x_amp).abs() < 1e-6, "x should reach its full amplitude {x_amp} somewhere in one period, got max {max_x}");
        assert!((max_y - y_amp).abs() < 1e-6, "y should reach its full amplitude {y_amp} somewhere in one period, got max {max_y}");
        assert!((max_x / max_y - 2.0).abs() < 1e-6, "the 2:1 amplitude ratio is what makes this an ELLIPSE, not a circle");

        let radii: Vec<f64> = samples.iter().map(|&(x, y)| (x * x + y * y).sqrt()).collect();
        let min_r = radii.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_r = radii.iter().cloned().fold(0.0, f64::max);
        assert!(max_r - min_r > 0.1, "an ellipse's distance from center must vary noticeably; min={min_r} max={max_r} looks circular");

        assert!(samples.iter().any(|&(x, y)| x.abs() > 0.05 && y.abs() > 0.05), "expected genuine 2D motion, not motion confined to one axis at a time");
    }

    #[test]
    fn percentile_escape_time_cutoff_hides_roughly_the_requested_top_fraction() {
        // 100 escaped hit pixels evenly spread 0..99 (background pixels
        // interleaved, which must be ignored — matches `shading > 0.0`
        // filtering `color_t`'s own definition).
        let mut shading = Vec::new();
        let mut color_t = Vec::new();
        for i in 0..100 {
            shading.push(1.0);
            color_t.push(i as f32);
            shading.push(0.0); // background — must not influence the ranking
            color_t.push(999.0);
        }
        let cutoff = percentile_escape_time_cutoff(&shading, &color_t, 200, 0.10).expect("a real distribution to rank");
        // Top 10% of 0..99 is roughly [90, 99] — the cutoff should land in that band.
        assert!((85.0..100.0).contains(&cutoff), "expected the cutoff near the top decile, got {cutoff}");
    }

    #[test]
    fn percentile_escape_time_cutoff_excludes_never_escaped_pixels_from_the_ranking() {
        // Every hit pixel never escapes (color_t == max_iter) — there's
        // nothing ESCAPED to rank against, so this must return None
        // rather than a bogus cutoff.
        let shading = vec![1.0; 50];
        let color_t = vec![60.0; 50];
        assert_eq!(percentile_escape_time_cutoff(&shading, &color_t, 60, 0.10), None);
    }

    #[test]
    fn percentile_escape_time_cutoff_none_for_zero_or_negative_fraction() {
        let shading = vec![1.0; 10];
        let color_t: Vec<f32> = (0..10).map(|i| i as f32).collect();
        assert_eq!(percentile_escape_time_cutoff(&shading, &color_t, 60, 0.0), None);
        assert_eq!(percentile_escape_time_cutoff(&shading, &color_t, 60, -0.5), None);
    }
}
