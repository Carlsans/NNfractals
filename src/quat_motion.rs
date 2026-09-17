//! Animates a `Slice` (and the time-driven C axis) over a clip's normalized
//! time `t ∈ [0,1)` — the same convention `video_export::time_frames` uses
//! (i/n, never reaching 1.0 for a real render). Two motions for the "first
//! try" prototype: `Orbit` (the plane circles a pivot while C ramps) and
//! `PanZoom` (the plane translates+zooms while C ramps). A third motion is
//! a new enum variant + one new `match` arm in `SliceMotion::sample` —
//! existing variants untouched.

use crate::quat_fractal::Slice;

pub(crate) type Vec3 = (f64, f64, f64);

pub(crate) fn add(u: Vec3, v: Vec3) -> Vec3 {
    (u.0 + v.0, u.1 + v.1, u.2 + v.2)
}
pub(crate) fn sub(u: Vec3, v: Vec3) -> Vec3 {
    (u.0 - v.0, u.1 - v.1, u.2 - v.2)
}
pub(crate) fn scale(u: Vec3, s: f64) -> Vec3 {
    (u.0 * s, u.1 * s, u.2 * s)
}
pub(crate) fn dot(u: Vec3, v: Vec3) -> f64 {
    u.0 * v.0 + u.1 * v.1 + u.2 * v.2
}
pub(crate) fn cross(u: Vec3, v: Vec3) -> Vec3 {
    (
        u.1 * v.2 - u.2 * v.1,
        u.2 * v.0 - u.0 * v.2,
        u.0 * v.1 - u.1 * v.0,
    )
}
pub(crate) fn normalize(u: Vec3) -> Vec3 {
    let n = dot(u, u).sqrt();
    if n < 1e-12 {
        (0.0, 0.0, 0.0)
    } else {
        scale(u, 1.0 / n)
    }
}

/// Given a unit `forward` direction, builds an orthonormal (basis_u=right,
/// basis_v=up) pair spanning the plane perpendicular to it — a standard
/// look-at construction, using `up_hint` to fix the roll (falls back to a
/// different reference axis when `forward` is nearly parallel to it, same
/// degenerate-case handling `OrbitParams` uses for its own reference axis).
pub(crate) fn look_at_basis(forward: Vec3, up_hint: Vec3) -> (Vec3, Vec3) {
    let f = normalize(forward);
    let hint = if dot(f, normalize(up_hint)).abs() < 0.95 {
        up_hint
    } else {
        (0.0, 0.0, 1.0)
    };
    let right = normalize(cross(hint, f));
    let up = cross(f, right);
    (right, up)
}

/// The plane's origin travels a circle of `radius` around `pivot`, over
/// `turns` full revolutions across the clip, always facing the direction of
/// travel — screen +x = the orbit's tangent (direction of motion), screen
/// +y = `axis` (constant for the whole clip, normalized internally). The
/// circular path lies entirely in the plane perpendicular to `axis`, so
/// `axis` is automatically perpendicular to the tangent at every instant.
/// While C ramps linearly c0 → c1.
///
/// Because this is an orthographic (non-perspective) slice, `pivot` is never
/// exactly ON the rendered plane once `radius > 0` — the plane's closest
/// point to `pivot` is the screen center, at perpendicular distance exactly
/// `radius`. Keep `radius` small relative to the visible half-extent
/// (2.0/zoom) to keep the pivot's neighborhood near screen center.
#[derive(Copy, Clone, Debug)]
pub struct OrbitParams {
    pub pivot: Vec3,
    pub axis: Vec3,
    pub radius: f64,
    pub turns: f64,
    pub phase0: f64,
    pub zoom: f64,
    pub c0: f64,
    pub c1: f64,
}

impl OrbitParams {
    fn sample(&self, t: f64) -> (Slice, f64) {
        let axis = normalize(self.axis);
        let reference = if dot(axis, (1.0, 0.0, 0.0)).abs() < 0.9 {
            (1.0, 0.0, 0.0)
        } else {
            (0.0, 1.0, 0.0)
        };
        let e1 = normalize(add(reference, scale(axis, -dot(reference, axis))));
        let e2 = normalize(cross(axis, e1));
        let radius = self.radius.max(1e-9);
        let theta = self.phase0 + self.turns * std::f64::consts::TAU * t;
        let origin = add(
            self.pivot,
            add(scale(e1, radius * theta.cos()), scale(e2, radius * theta.sin())),
        );
        let tangent = normalize(add(scale(e1, -theta.sin()), scale(e2, theta.cos())));
        let slice = Slice {
            origin,
            basis_u: tangent,
            basis_v: axis,
            zoom: self.zoom,
            camera: None,
        };
        (slice, self.c0 + (self.c1 - self.c0) * t)
    }
}

/// The plane's origin translates linearly along `direction` by a total of
/// `distance` (absolute R,A,B units) while zoom interpolates zoom0→zoom1
/// GEOMETRICALLY (exp(lerp(ln z0, ln z1, t))) — the same convention
/// `video_export::lerp_view` uses for zoom — orientation (basis_u/basis_v)
/// held fixed. C ramps linearly c0 → c1.
#[derive(Copy, Clone, Debug)]
pub struct PanZoomParams {
    pub origin0: Vec3,
    pub direction: Vec3,
    pub distance: f64,
    pub basis_u: Vec3,
    pub basis_v: Vec3,
    pub zoom0: f64,
    pub zoom1: f64,
    pub c0: f64,
    pub c1: f64,
}

impl PanZoomParams {
    fn sample(&self, t: f64) -> (Slice, f64) {
        let dir = normalize(self.direction);
        let origin = add(self.origin0, scale(dir, self.distance * t));
        let (lz0, lz1) = (self.zoom0.max(1e-300).ln(), self.zoom1.max(1e-300).ln());
        let zoom = (lz0 + (lz1 - lz0) * t).exp();
        let slice = Slice {
            origin,
            basis_u: normalize(self.basis_u),
            basis_v: normalize(self.basis_v),
            zoom,
            camera: None,
        };
        (slice, self.c0 + (self.c1 - self.c0) * t)
    }
}

/// A closed-form motion is a `Vec3`-full of scalars and can be `Copy`; a
/// motion that comes from simulating something (see `quat_gravity`) is a
/// precomputed per-frame sequence instead, since the physics has to be
/// integrated forward step by step rather than evaluated at an arbitrary
/// `t` — hence `SliceMotion` itself is `Clone` only, not `Copy`.
#[derive(Clone, Debug)]
pub enum SliceMotion {
    Orbit(OrbitParams),
    PanZoom(PanZoomParams),
    /// A precomputed (Slice, c) per frame, e.g. from `quat_gravity::simulate_projectile`.
    /// Built with exactly as many entries as the render has frames, so
    /// `t = i/n` maps back to index `i` exactly (mirrors `time_frames`'s
    /// own `t=i/n` convention — see `sample` below).
    Trajectory(Vec<(Slice, f64)>),
}

impl SliceMotion {
    /// `t ∈ [0,1)` — same convention as `video_export::time_frames`.
    pub fn sample(&self, t: f64) -> (Slice, f64) {
        match self {
            SliceMotion::Orbit(p) => p.sample(t),
            SliceMotion::PanZoom(p) => p.sample(t),
            SliceMotion::Trajectory(frames) => {
                let n = frames.len().max(1);
                let idx = ((t * n as f64).round() as usize).min(n - 1);
                frames[idx]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(u: Vec3, v: Vec3, eps: f64) -> bool {
        (u.0 - v.0).abs() < eps && (u.1 - v.1).abs() < eps && (u.2 - v.2).abs() < eps
    }

    #[test]
    fn orbit_returns_to_start_after_one_full_turn() {
        let p = OrbitParams {
            pivot: (0.0, 0.0, 0.0),
            axis: (0.0, 1.0, 0.0),
            radius: 0.7,
            turns: 1.0,
            phase0: 0.3,
            zoom: 1.0,
            c0: 0.0,
            c1: 0.0,
        };
        let (s0, _) = p.sample(0.0);
        let (s1, _) = p.sample(1.0);
        assert!(approx(s0.origin, s1.origin, 1e-9), "{:?} vs {:?}", s0.origin, s1.origin);
        assert!(approx(s0.basis_u, s1.basis_u, 1e-9));
        assert!(approx(s0.basis_v, s1.basis_v, 1e-9));
    }

    #[test]
    fn orbit_basis_vectors_stay_orthonormal() {
        let p = OrbitParams {
            pivot: (1.0, -2.0, 0.5),
            axis: (0.0, 1.0, 0.0),
            radius: 1.3,
            turns: 2.0,
            phase0: 0.0,
            zoom: 4.0,
            c0: -1.0,
            c1: 1.0,
        };
        for i in 0..10 {
            let t = i as f64 / 10.0;
            let (slice, _) = p.sample(t);
            let nu = dot(slice.basis_u, slice.basis_u).sqrt();
            let nv = dot(slice.basis_v, slice.basis_v).sqrt();
            assert!((nu - 1.0).abs() < 1e-9, "t={t} |basis_u|={nu}");
            assert!((nv - 1.0).abs() < 1e-9, "t={t} |basis_v|={nv}");
            assert!(dot(slice.basis_u, slice.basis_v).abs() < 1e-9, "t={t} basis_u·basis_v not ~0");
        }
    }

    #[test]
    fn orbit_c_ramps_linearly() {
        let p = OrbitParams {
            pivot: (0.0, 0.0, 0.0),
            axis: (0.0, 1.0, 0.0),
            radius: 1.0,
            turns: 1.0,
            phase0: 0.0,
            zoom: 1.0,
            c0: -2.0,
            c1: 6.0,
        };
        let (_, c) = p.sample(0.25);
        assert!((c - 0.0).abs() < 1e-9, "c={c}");
    }

    #[test]
    fn panzoom_zoom_is_geometric_not_linear() {
        let p = PanZoomParams {
            origin0: (0.0, 0.0, 0.0),
            direction: (1.0, 0.0, 0.0),
            distance: 0.0,
            basis_u: (1.0, 0.0, 0.0),
            basis_v: (0.0, 1.0, 0.0),
            zoom0: 1.0,
            zoom1: 100.0,
            c0: 0.0,
            c1: 0.0,
        };
        let (slice, _) = p.sample(0.5);
        assert!((slice.zoom - 10.0).abs() < 1e-6, "zoom={}", slice.zoom);
    }

    #[test]
    fn panzoom_origin_translates_and_c_ramps_linearly() {
        let p = PanZoomParams {
            origin0: (0.0, 0.0, 0.0),
            direction: (1.0, 0.0, 0.0),
            distance: 2.0,
            basis_u: (1.0, 0.0, 0.0),
            basis_v: (0.0, 1.0, 0.0),
            zoom0: 1.0,
            zoom1: 1.0,
            c0: 0.0,
            c1: 10.0,
        };
        let (slice, c) = p.sample(0.25);
        assert!(approx(slice.origin, (0.5, 0.0, 0.0), 1e-9), "{:?}", slice.origin);
        assert!((c - 2.5).abs() < 1e-9, "c={c}");
    }
}
