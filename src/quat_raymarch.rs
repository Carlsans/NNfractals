//! True volumetric ray-marching, for the genuinely-3D quaternion formulas
//! (`Bulb`, `BurningShip*`, `Perpendicular*` — see `quat_fractal`'s module
//! docs on the "solid of revolution" / spherical-symmetry degeneracy for
//! why `Mandelbrot`/`Tricorn`/`Cubic`/`Quartic`/`Celtic` are the WRONG
//! formulas to point this at under `TimeAxis::R`: their escape-time field
//! is a pure function of one radial parameter there, so a full volumetric
//! render of them would show nothing a flat 2D plot of that one parameter
//! doesn't already show).
//!
//! Unlike `quat_fractal::Slice::pixel_to_spatial`, which evaluates the
//! field at exactly ONE point per pixel (a flat cross-section, even with
//! `PerspectiveCamera`'s foreshortening), this casts a genuine ray per
//! pixel through the 3D spatial subspace and finds where it crosses the
//! fractal's own boundary — giving a real shaded 3D surface.
//!
//! Uses true adaptive sphere tracing: each step advances by the local
//! `quat_escape_de` distance estimate rather than a fixed increment, so
//! rays through empty space cover it in a few large jumps while rays near
//! fine detail automatically take small, safe steps. The v1 of this module
//! used fixed-step marching + bisection instead (simpler, no per-formula
//! derivative math needed) — kept working and correctly shaped, but
//! shading was visibly speckled from a noisy normal source; switching the
//! normal to `quat_escape_de`'s distance estimate fixed that, and this is
//! the natural following step: use the SAME distance estimate to drive the
//! march itself, not just the normal. `quat_escape_de`'s doc comment has
//! the caveat this still isn't a certified bound (these quaternion
//! power-maps aren't holomorphic) — a `step_safety` factor below 1.0
//! trades some speed for overshoot robustness against that.

use rayon::prelude::*;

use crate::quat_fractal::{quat_escape_de_params, QuatFormula, TimeAxis};
use crate::quat_motion::{add, cross, dot, look_at_basis, normalize, scale, sub, Vec3};

#[derive(Clone, Copy, Debug)]
pub struct RaymarchCamera {
    pub eye: Vec3,
    pub target: Vec3,
    pub up_hint: Vec3,
    /// Vertical field of view, in radians.
    pub fov_y: f64,
}

/// Turntable orbit: `eye` traces a circle of `radius` around `target`, in
/// the plane perpendicular to `axis`, over `turns` revolutions across a
/// clip. Unlike `quat_motion::OrbitParams` (which orbits a flat `Slice`
/// through the fractal's OWN field, where a basis-aligned `axis` froze a
/// raw coordinate across the clip — see the eleventh follow-up in project
/// memory), the fractal here is a fixed 3D object in ordinary space and the
/// camera just looks at it from different angles, so there's no analogous
/// degeneracy: `axis = (0,1,0)` is the natural, unremarkable default,
/// exactly like any ordinary turntable render.
pub struct RaymarchOrbitParams {
    pub target: Vec3,
    pub axis: Vec3,
    pub radius: f64,
    pub turns: f64,
    pub phase0: f64,
    pub fov_y: f64,
}

impl RaymarchOrbitParams {
    /// `t ∈ [0,1)` — same convention as `video_export::time_frames` and
    /// every other motion sampler in this project.
    pub fn sample(&self, t: f64) -> RaymarchCamera {
        let axis = normalize(self.axis);
        let reference = if dot(axis, (1.0, 0.0, 0.0)).abs() < 0.9 {
            (1.0, 0.0, 0.0)
        } else {
            (0.0, 1.0, 0.0)
        };
        let e1 = normalize(add(reference, scale(axis, -dot(reference, axis))));
        let e2 = normalize(cross(axis, e1));
        let theta = self.phase0 + self.turns * std::f64::consts::TAU * t;
        let offset = add(scale(e1, self.radius * theta.cos()), scale(e2, self.radius * theta.sin()));
        RaymarchCamera {
            eye: add(self.target, offset),
            target: self.target,
            up_hint: axis,
            fov_y: self.fov_y,
        }
    }
}

#[derive(Copy, Clone)]
pub struct RaymarchParams {
    pub formula: QuatFormula,
    pub time_axis: TimeAxis,
    pub time_val: f64,
    /// Radius of the bounding sphere (centered on the spatial subspace's
    /// own origin) a ray is marched across — rays that miss this sphere
    /// entirely are background, no formula evaluation needed.
    pub domain_radius: f64,
    pub max_iter: u32,
    pub bailout: f64,
    /// Hard cap on sphere-tracing steps per ray, so a ray that's grazing
    /// along a near-tangent surface (many tiny safe steps in a row) can't
    /// loop forever — hitting this cap without converging counts as a
    /// miss, same as exiting the bounding sphere.
    pub max_march_steps: u32,
    /// Stop and report a hit once the distance estimate drops below this —
    /// small relative to `domain_radius`, since it's the residual gap left
    /// unresolved at the reported surface point.
    pub hit_epsilon: f64,
    /// Multiplies each step's distance-estimate-sized advance (`(0,1]`).
    /// `1.0` trusts the estimate fully (fastest, most overshoot risk on a
    /// formula whose DE isn't a tight/certified bound); `<1.0` (e.g. `0.8`)
    /// takes smaller, safer steps — the standard mitigation for exactly
    /// the "this DE is only an approximation" caveat `quat_escape_de`
    /// documents.
    pub step_safety: f64,
    /// Directional light, in the same spatial coordinates as everything
    /// else here — need not be normalized.
    pub light_dir: Vec3,
    /// Supersampling grid factor: `aa x aa` jittered sub-rays per pixel,
    /// averaged. `Bulb`'s fine filamentary surface detail is thinner than
    /// one pixel's angular footprint at typical framings, so a single ray
    /// per pixel aliases into a sparkly/noisy-looking speckle pattern —
    /// confirmed empirically (this is real geometric detail being
    /// under-sampled, not a normal-estimation bug: the pattern persisted
    /// even after switching to the smoother distance-estimate normal).
    /// `1` = no supersampling (cheapest, speckled); `2`-`3` cleans it up
    /// substantially at 4x-9x the cost.
    pub aa: u32,
    /// Finite-difference offset for `estimate_normal`. This is against the
    /// (already much smoother) distance-estimate potential, not raw escape
    /// time — see `estimate_normal`'s doc comment for why that switch
    /// itself was the fix for the ORIGINAL speckled shading, before
    /// sphere-tracing existed here at all.
    pub normal_eps: f64,
    /// How far outward from the surface (along the normal) to sample the
    /// color-source escape time — see `march_ray`'s doc comment. Too large
    /// and every probe lands in near-immediate-escape territory (low
    /// iteration count everywhere, a flat dark color); too small and it's
    /// numerically indistinguishable from the surface itself (pinned near
    /// `max_iter` everywhere, also flat). Wants to be small relative to
    /// `domain_radius` but not as tight as `normal_eps`.
    pub color_probe_offset: f64,
    /// Override for `Bulb`'s power (module default `BULB_POWER=8.0` in
    /// `quat_fractal.rs`) — meaningless for every other formula. Added so
    /// imported Mandelbulber parameter files (which vary this per-file)
    /// can actually render at their own power instead of always
    /// collapsing onto one fixed shape; see `quat_fractal::
    /// quat_escape_de_params`.
    pub bulb_power: f64,
    /// Override for `Mandelbox`'s scale (module default
    /// `MANDELBOX_SCALE=-1.5`) — meaningless for every other formula.
    /// Same motivation as `bulb_power`.
    pub mandelbox_scale: f64,
}

/// Ray/bounding-sphere intersection, sphere centered at the spatial
/// subspace's origin `(0,0,0)` with radius `radius`. Returns `(t_near,
/// t_far)` (both `>= 0`; `t_near = 0` if `eye` is already inside the
/// sphere) when the ray hits it at all.
pub(crate) fn ray_sphere(eye: Vec3, dir: Vec3, radius: f64) -> Option<(f64, f64)> {
    // |eye + t*dir|^2 = radius^2 => a t^2 + b t + c = 0, a = |dir|^2 (=1 for
    // a normalized dir, kept general here for robustness).
    let a = dot(dir, dir);
    let b = 2.0 * dot(eye, dir);
    let c = dot(eye, eye) - radius * radius;
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 || a < 1e-300 {
        return None;
    }
    let sqrt_disc = disc.sqrt();
    let t0 = (-b - sqrt_disc) / (2.0 * a);
    let t1 = (-b + sqrt_disc) / (2.0 * a);
    if t1 < 0.0 {
        return None; // sphere is entirely behind the eye
    }
    Some((t0.max(0.0), t1))
}

/// Ray/axis-aligned-box intersection (the standard "slab method"): `bmin`
/// and `bmax` are the box's per-axis low/high corners. Returns `(t_near,
/// t_far)` (both `>= 0`; `t_near = 0` if `eye` starts inside the box) when
/// the ray crosses it at all. Added for the animation-viewer plan's
/// bounding-box axis (replacing `ray_sphere`'s single radius with
/// independent per-axis extents) — additive, next to `ray_sphere`, which
/// stays exactly as-is for every existing caller (the hand-built
/// `QuatFormula` stack via `RaymarchParams`, and any `RaymarchDagParams`
/// caller that leaves `box_bounds` as `None`).
pub(crate) fn ray_box(eye: Vec3, dir: Vec3, bmin: Vec3, bmax: Vec3) -> Option<(f64, f64)> {
    let mut t0 = 0.0f64;
    let mut t1 = f64::INFINITY;
    let e = [eye.0, eye.1, eye.2];
    let d = [dir.0, dir.1, dir.2];
    let lo = [bmin.0, bmin.1, bmin.2];
    let hi = [bmax.0, bmax.1, bmax.2];
    for axis in 0..3 {
        if d[axis].abs() < 1e-300 {
            if e[axis] < lo[axis] || e[axis] > hi[axis] {
                return None; // parallel to this axis' slab and outside it
            }
        } else {
            let inv = 1.0 / d[axis];
            let mut ta = (lo[axis] - e[axis]) * inv;
            let mut tb = (hi[axis] - e[axis]) * inv;
            if ta > tb {
                std::mem::swap(&mut ta, &mut tb);
            }
            t0 = t0.max(ta);
            t1 = t1.min(tb);
            if t0 > t1 {
                return None;
            }
        }
    }
    Some((t0.max(0.0), t1))
}

/// Tetrahedral 4-tap gradient estimate of the distance-estimate potential
/// (`quat_escape_de`) at `p` — NOT the raw escape time. First attempt used
/// raw escape time here and the shading came out visibly speckled/noisy no
/// matter how large `normal_eps` was pushed (confirmed empirically, not
/// just suspected): near a chaotic fractal boundary, iteration count is
/// extremely sensitive to position — that IS the boundary's defining
/// property — so its finite-difference gradient stays noisy at every
/// scale. The distance estimate is designed to behave like an actual
/// (approximate) signed distance near the boundary, which is Lipschitz-ish
/// and gives a clean gradient — see `quat_escape_de`'s doc comment.
///
/// Uses 4 potential samples at regular-tetrahedron offsets (the standard
/// SDF-normal trick — each offset weighted by itself and summed) instead
/// of the more obvious 6-tap axis-aligned central difference (2 samples ×
/// 3 axes): the four tetrahedral offsets sum to zero and their outer
/// products sum to a multiple of the identity, so the weighted sum
/// recovers the same gradient DIRECTION as central differences (up to a
/// positive scale `normalize` removes) for 4 evaluations instead of 6 —
/// this is called at every ray hit, so it's worth the 33% cut. Kept
/// exactly the same outward-facing sign convention (verified via the
/// existing `normal_points_outward_away_from_the_domain_center` test, not
/// just assumed).
fn estimate_normal(p: &RaymarchParams, point: Vec3) -> Vec3 {
    let eps = p.normal_eps.max(1e-6);
    let bailout_sq = p.bailout * p.bailout;
    let potential = |q: Vec3| -> f64 {
        quat_escape_de_params(p.formula, p.time_axis.assemble(q, p.time_val), p.max_iter, bailout_sq, p.bulb_power, p.mandelbox_scale).1
    };
    const K0: Vec3 = (1.0, -1.0, -1.0);
    const K1: Vec3 = (-1.0, -1.0, 1.0);
    const K2: Vec3 = (-1.0, 1.0, -1.0);
    const K3: Vec3 = (1.0, 1.0, 1.0);
    let g = add(
        add(scale(K0, potential(add(point, scale(K0, eps)))), scale(K1, potential(add(point, scale(K1, eps))))),
        add(scale(K2, potential(add(point, scale(K2, eps)))), scale(K3, potential(add(point, scale(K3, eps))))),
    );
    // Same outward-facing convention as the old central-difference version
    // (the distance estimate is lowest AT the boundary and increases
    // outward, so its gradient already points outward — no sign flip).
    normalize(g)
}

/// Adaptive sphere tracing: `Some((surface_point, outward_normal,
/// color_escape_time))` on a hit. Each step advances `t` by (a
/// safety-scaled) `quat_escape_de` at the current point rather than a fixed
/// increment — safe because the distance estimate lower-bounds
/// (approximately, for these non-holomorphic formulas — see
/// `step_safety`) how far the nearest surface point can be, so stepping by
/// less than that can't skip past it.
///
/// `color_escape_time` is the plain `quat_escape` value at a point nudged
/// slightly OUTWARD from the surface along the normal — not AT the
/// surface, where escape time is uselessly pinned near `max_iter` for
/// every hit by construction. A short distance out, it varies richly with
/// how deep into a filament/crevice vs. how exposed a given patch of
/// surface is, giving a real per-point color source tied to the fractal's
/// own structure — the same value `explorer.rs` feeds into
/// `colormap::apply_colormap` for actual color, keeping this consistent
/// with every other render in the project.
fn march_ray(p: &RaymarchParams, eye: Vec3, dir: Vec3) -> Option<(Vec3, Vec3, f32)> {
    let (t0, t1) = ray_sphere(eye, dir, p.domain_radius)?;
    let bailout_sq = p.bailout * p.bailout;
    let hit_eps = p.hit_epsilon.max(1e-9);
    // A floor on the per-step advance: without one, a spot where DE reads
    // as (near-)zero without actually being a hit yet — plausible given
    // `quat_escape_de` is only an approximation for these formulas — could
    // stall the march in place for its entire step budget instead of
    // either resolving a hit or moving on.
    let min_step = (p.domain_radius * 1e-6).max(1e-9);
    let mut t = t0;
    for _ in 0..p.max_march_steps.max(1) {
        if t > t1 {
            return None;
        }
        let point = add(eye, scale(dir, t));
        let (_, de) = quat_escape_de_params(p.formula, p.time_axis.assemble(point, p.time_val), p.max_iter, bailout_sq, p.bulb_power, p.mandelbox_scale);
        if de < hit_eps {
            let normal = estimate_normal(p, point);
            let probe = add(point, scale(normal, p.color_probe_offset));
            let (color_et, _) = quat_escape_de_params(p.formula, p.time_axis.assemble(probe, p.time_val), p.max_iter, bailout_sq, p.bulb_power, p.mandelbox_scale);
            return Some((point, normal, color_et));
        }
        t += (de * p.step_safety).max(min_step);
    }
    None
}

/// Rayon-parallel over pixels. Returns two `width*height` buffers: shading
/// values in `[0,1]` (exactly `0.0` for a ray that never hit anything) and
/// a color-source escape time (`0.0` for background, meaningless there —
/// callers should gate on shading, not this, to tell background from a
/// genuinely dark hit). See `explorer.rs`'s CLI for how these combine into
/// an actual RGB image via `colormap::apply_colormap`.
pub fn render_raymarch_frame(p: &RaymarchParams, cam: &RaymarchCamera, width: u32, height: u32) -> (Vec<f32>, Vec<f32>) {
    let forward = normalize(sub(cam.target, cam.eye));
    let (right, up) = look_at_basis(forward, cam.up_hint);
    let half_h = (cam.fov_y * 0.5).tan();
    let aspect = width as f64 / (height.max(1)) as f64;
    let half_w = half_h * aspect;
    let light = normalize(p.light_dir);
    let wf = width.max(1) as f64;
    let hf = height.max(1) as f64;
    let aa = p.aa.max(1);
    (0..(width as u64 * height as u64))
        .into_par_iter()
        .map(|idx| {
            let px = (idx % width as u64) as u32;
            let py = (idx / width as u64) as u32;
            let mut sum_shade = 0.0f32;
            let mut sum_color = 0.0f32;
            for sy in 0..aa {
                for sx in 0..aa {
                    let jx = (sx as f64 + 0.5) / aa as f64;
                    let jy = (sy as f64 + 0.5) / aa as f64;
                    let u = ((px as f64 + jx) / wf * 2.0 - 1.0) * half_w;
                    let v = (1.0 - (py as f64 + jy) / hf * 2.0) * half_h;
                    let dir = normalize(add(add(forward, scale(right, u)), scale(up, v)));
                    match march_ray(p, cam.eye, dir) {
                        Some((_, normal, color_et)) => {
                            let ndotl = dot(normal, light).max(0.0);
                            sum_shade += (0.15 + 0.85 * ndotl) as f32;
                            sum_color += color_et;
                        }
                        None => {}
                    };
                }
            }
            let n = (aa * aa) as f32;
            (sum_shade / n, sum_color / n)
        })
        .unzip()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_params(formula: QuatFormula) -> RaymarchParams {
        RaymarchParams {
            formula,
            time_axis: TimeAxis::C,
            time_val: 0.0,
            domain_radius: 1.6,
            max_iter: 40,
            bailout: 4.0,
            max_march_steps: 200,
            hit_epsilon: 1.6 * 1e-4,
            step_safety: 0.8,
            light_dir: (0.5, 0.8, 0.3),
            normal_eps: 1.6 * 1e-3,
            color_probe_offset: 1.6 * 1e-2,
            aa: 1,
            bulb_power: 8.0,
            mandelbox_scale: -1.5,
        }
    }

    #[test]
    fn orbit_returns_to_start_after_one_full_turn() {
        let params = RaymarchOrbitParams { target: (0.1, -0.2, 0.05), axis: (0.0, 1.0, 0.0), radius: 4.0, turns: 1.0, phase0: 0.0, fov_y: 0.8 };
        let cam0 = params.sample(0.0);
        let cam1 = params.sample(1.0);
        let d = |a: Vec3, b: Vec3| ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2) + (a.2 - b.2).powi(2)).sqrt();
        assert!(d(cam0.eye, cam1.eye) < 1e-9, "expected the eye to return to its start after one full turn, cam0={:?} cam1={:?}", cam0.eye, cam1.eye);
    }

    #[test]
    fn orbit_eye_stays_at_a_constant_radius_from_the_target() {
        let params = RaymarchOrbitParams { target: (0.2, 0.1, -0.1), axis: (0.3, 1.0, 0.2), radius: 3.5, turns: 2.0, phase0: 0.4, fov_y: 0.8 };
        for i in 0..20 {
            let t = i as f64 / 20.0;
            let cam = params.sample(t);
            let d = sub(cam.eye, cam.target);
            let r = (d.0 * d.0 + d.1 * d.1 + d.2 * d.2).sqrt();
            assert!((r - 3.5).abs() < 1e-9, "t={t}: radius={r}, expected 3.5");
        }
    }

    #[test]
    fn orbit_camera_always_looks_at_the_target() {
        let params = RaymarchOrbitParams { target: (1.0, 2.0, 3.0), axis: (0.0, 1.0, 0.0), radius: 2.0, turns: 1.5, phase0: 0.0, fov_y: 0.8 };
        for i in 0..10 {
            let cam = params.sample(i as f64 / 10.0);
            assert_eq!(cam.target, (1.0, 2.0, 3.0));
        }
    }

    #[test]
    fn ray_sphere_hits_a_centered_sphere_head_on() {
        let hit = ray_sphere((0.0, 0.0, -5.0), (0.0, 0.0, 1.0), 1.6);
        let (t0, t1) = hit.expect("ray through the center must hit");
        assert!((t0 - 3.4).abs() < 1e-9, "t0={t0}");
        assert!((t1 - 6.6).abs() < 1e-9, "t1={t1}");
    }

    #[test]
    fn ray_sphere_misses_when_aimed_well_clear() {
        let hit = ray_sphere((0.0, 10.0, -5.0), (0.0, 0.0, 1.0), 1.6);
        assert!(hit.is_none());
    }

    #[test]
    fn ray_sphere_returns_none_for_a_sphere_entirely_behind_the_eye() {
        let hit = ray_sphere((0.0, 0.0, -5.0), (0.0, 0.0, -1.0), 1.6);
        assert!(hit.is_none());
    }

    #[test]
    fn ray_box_hits_a_centered_cube_head_on() {
        let hit = ray_box((0.0, 0.0, -5.0), (0.0, 0.0, 1.0), (-1.6, -1.6, -1.6), (1.6, 1.6, 1.6));
        let (t0, t1) = hit.expect("ray through the center must hit");
        assert!((t0 - 3.4).abs() < 1e-9, "t0={t0}");
        assert!((t1 - 6.6).abs() < 1e-9, "t1={t1}");
    }

    #[test]
    fn ray_box_misses_when_aimed_well_clear() {
        let hit = ray_box((0.0, 10.0, -5.0), (0.0, 0.0, 1.0), (-1.6, -1.6, -1.6), (1.6, 1.6, 1.6));
        assert!(hit.is_none());
    }

    #[test]
    fn ray_box_returns_none_for_a_box_entirely_behind_the_eye() {
        let hit = ray_box((0.0, 0.0, -5.0), (0.0, 0.0, -1.0), (-1.6, -1.6, -1.6), (1.6, 1.6, 1.6));
        assert!(hit.is_none());
    }

    #[test]
    fn ray_box_reaches_further_into_the_corners_than_the_equivalent_sphere() {
        // The whole point of switching to a box: a ray aimed at a diagonal
        // corner should reach materially farther than a same-half-extent
        // sphere would let it, since the sphere caps every direction at
        // exactly `radius` while the box's corners extend to
        // radius*sqrt(3).
        let dir = normalize((1.0, 1.0, 1.0));
        let eye = (-5.0, -5.0, -5.0);
        let (_, t1_box) = ray_box(eye, dir, (-1.6, -1.6, -1.6), (1.6, 1.6, 1.6)).unwrap();
        let (_, t1_sphere) = ray_sphere(eye, dir, 1.6).unwrap();
        assert!(t1_box > t1_sphere, "box t1={t1_box} should exceed sphere t1={t1_sphere}");
    }

    #[test]
    fn ray_box_respects_independent_per_axis_extents() {
        // A box squashed flat on X (min=max=0) must be missed by a ray
        // that would have hit a symmetric box/sphere at that same origin —
        // this is the actual behavior the animation viewer's per-axis
        // bounding-box editor depends on (shrinking one axis crops the
        // render on that axis specifically, not uniformly).
        let hit = ray_box((0.0, 0.0, -5.0), (0.0, 0.0, 1.0), (0.0, -1.6, -1.6), (0.0, 1.6, 1.6));
        // Straight down the Z axis at x=0 should still clip it (edge case,
        // x stays exactly 0 the whole ray), but a ray offset in X must miss.
        assert!(hit.is_some(), "a ray exactly on the flattened plane still crosses it");
        let offset_hit = ray_box((0.5, 0.0, -5.0), (0.0, 0.0, 1.0), (0.0, -1.6, -1.6), (0.0, 1.6, 1.6));
        assert!(offset_hit.is_none(), "a ray off the flattened X=0 plane must miss a box with zero X extent");
    }

    #[test]
    fn a_ray_aimed_at_the_origin_hits_bulb_from_outside_the_domain() {
        // Q=(0,0,0,0) is exactly Quat::ZERO, provably interior forever for
        // every formula (see quat_fractal's interior_point_never_escapes
        // test) — so a ray aimed straight at the spatial origin, from
        // outside the bounding sphere, must find a crossing somewhere
        // along the way in for a formula whose set has any real extent.
        let params = base_params(QuatFormula::Bulb);
        let eye = (0.0, 0.0, -5.0);
        let dir = normalize((0.0, 0.0, 1.0));
        let hit = march_ray(&params, eye, dir);
        assert!(hit.is_some(), "expected the ray toward the origin to hit the bulb");
        let (surface, normal, color_et) = hit.unwrap();
        assert!(surface.0.is_finite() && surface.1.is_finite() && surface.2.is_finite());
        let n = (normal.0 * normal.0 + normal.1 * normal.1 + normal.2 * normal.2).sqrt();
        assert!((n - 1.0).abs() < 1e-6, "normal should be unit length, got {n}");
        assert!(color_et.is_finite() && color_et >= 0.0, "color_et={color_et}");
    }

    #[test]
    fn a_ray_that_misses_the_domain_entirely_produces_no_hit() {
        let params = base_params(QuatFormula::Bulb);
        let hit = march_ray(&params, (0.0, 20.0, -5.0), normalize((0.0, 0.0, 1.0)));
        assert!(hit.is_none());
    }

    #[test]
    fn render_frame_background_pixels_are_exactly_zero_and_hits_are_in_bounds() {
        let params = base_params(QuatFormula::Bulb);
        let cam = RaymarchCamera {
            eye: (0.0, 0.0, -4.0),
            target: (0.0, 0.0, 0.0),
            up_hint: (0.0, 1.0, 0.0),
            fov_y: 50.0_f64.to_radians(),
        };
        let (w, h) = (48u32, 48u32);
        let (shading, color_t) = render_raymarch_frame(&params, &cam, w, h);
        assert_eq!(shading.len(), (w * h) as usize);
        assert_eq!(color_t.len(), (w * h) as usize);
        let mut any_hit = false;
        let mut any_background = false;
        for (&v, &c) in shading.iter().zip(color_t.iter()) {
            assert!(v.is_finite() && (0.0..=1.0).contains(&v), "shading value out of range: {v}");
            assert!(c.is_finite() && c >= 0.0, "color_t out of range: {c}");
            if v == 0.0 {
                any_background = true;
            } else {
                any_hit = true;
            }
        }
        assert!(any_hit, "expected at least some pixels to hit the bulb, looking straight at it");
        assert!(any_background, "expected at least some background pixels at the frame edges");
    }

    #[test]
    fn normal_points_outward_away_from_the_domain_center() {
        // A crude but effective sanity check: at a hit point reasonably
        // close to the domain's own origin, the estimated outward normal
        // should have a non-negative dot product with the point's own
        // direction from the origin more often than not — a normal that's
        // silently backwards (pointing INTO the set) would fail this
        // outright, since it'd systematically point the opposite way.
        let params = base_params(QuatFormula::Bulb);
        let eye = (0.0, 0.0, -5.0);
        let mut outward_votes = 0;
        let mut total = 0;
        for i in -3..=3 {
            for j in -3..=3 {
                let dir = normalize((i as f64 * 0.08, j as f64 * 0.08, 1.0));
                if let Some((surface, normal, _)) = march_ray(&params, eye, dir) {
                    let radial = normalize(surface);
                    if dot(radial, normal) > 0.0 {
                        outward_votes += 1;
                    }
                    total += 1;
                }
            }
        }
        assert!(total > 5, "expected several rays to hit for this sanity check, got {total}");
        assert!(outward_votes * 2 >= total, "expected most normals to point outward, got {outward_votes}/{total}");
    }
}
