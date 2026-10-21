//! Ways of collapsing the 4D quaternion set to the 3D picture the raymarcher
//! draws. A render carries its own [`Collapse`] (`RaymarchDagParams::collapse`);
//! `None` falls back to a process-wide default (set by experiments / `NN_COLLAPSE`).
//!
//! The raymarcher's `time_val` is interpreted by the method: a value of the
//! held-fixed C axis for the slice family, a 4D rotation angle (radians) for
//! [`Method::S3`]. `map_point` is the CPU twin of the shader's `de_one`.

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU32, Ordering};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Method {
    /// Flat hyperplane at C = t (the original behaviour).
    #[default]
    Slice = 0,
    /// Hyperplane tilted by 4D rotations in the (z,t) and (x,t) planes.
    Tilt = 1,
    /// Union of three slices at t-h, t, t+h (min distance).
    Slab = 2,
    /// Surface inflated by delta: the level set DE = delta.
    Offset = 3,
    /// Points on the 3-sphere via inverse stereographic projection; t is a
    /// 4D rotation angle.
    S3 = 4,
    /// Orbit confined to R,A,B: the C component of z is zeroed every step.
    Proj3 = 5,
    /// The flat slice at distance `rho` from the origin, turned by the time
    /// angle in one 4D plane (the slice sweeps through the whole space and the
    /// loop closes after a full turn).
    Spin = 6,
    /// Like `Spin`, but the turn is isoclinic (two planes at once), so no
    /// direction stays fixed.
    IsoSpin = 7,
    /// The 3-sphere with an isoclinic time rotation: every point of the sphere
    /// moves at the same speed, there is no fixed axis (Hopf-like motion).
    IsoS3 = 8,
}

pub const ALL: [Method; 9] = [
    Method::Slice, Method::Tilt, Method::Slab, Method::Offset, Method::S3, Method::Proj3,
    Method::Spin, Method::IsoSpin, Method::IsoS3,
];

/// The projections the GUIs offer for now: the legacy slice and S3. The other
/// methods stay available from the command line and in saved timelines.
pub const OFFERED: [Method; 2] = [Method::Slice, Method::S3];

impl Method {
    /// Time is a rotation angle (one loop = 2π), not a C value.
    pub fn time_is_angle(self) -> bool {
        matches!(self, Method::S3 | Method::Spin | Method::IsoSpin | Method::IsoS3)
    }
    /// Samples the 3-sphere (view space is stereographic), so the planes view
    /// draws the sphere on each tile.
    pub fn is_sphere(self) -> bool { matches!(self, Method::S3 | Method::IsoS3) }
    /// The per-axis "equal scale" stretch only makes sense when view axes stay axes.
    pub fn allows_axis_remap(self) -> bool {
        matches!(self, Method::Slice | Method::Slab | Method::Offset | Method::Proj3)
    }
    pub fn describe(self) -> &'static str {
        match self {
            Method::Slice => "Flat hyperplane at the time value (original)",
            Method::Tilt => "Hyperplane tilted by 4D rotations",
            Method::Slab => "Union of three nearby slices (less flashing)",
            Method::Offset => "Inflated surface (level set of the distance field)",
            Method::S3 => "The 3-sphere, stereographically projected; time is a 4D rotation angle",
            Method::Proj3 => "Orbit confined to three components",
            Method::Spin => "A flat slice turned through 4D by the time angle (one plane)",
            Method::IsoSpin => "A flat slice turned through 4D by an isoclinic rotation (two planes at once)",
            Method::IsoS3 => "The 3-sphere with an isoclinic (Hopf-like) time rotation: no fixed axis",
        }
    }
}

impl Method {
    pub fn name(self) -> &'static str {
        match self { Method::Slice => "slice", Method::Tilt => "tilt", Method::Slab => "slab", Method::Offset => "offset", Method::S3 => "s3", Method::Proj3 => "proj3", Method::Spin => "spin", Method::IsoSpin => "isospin", Method::IsoS3 => "isos3" }
    }
    pub fn parse(s: &str) -> Option<Method> {
        ALL.iter().copied().find(|m| m.name() == s)
    }
}

/// A method plus its parameters. Parameters not used by `method` are ignored.
#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Collapse {
    pub method: Method,
    /// S3: radius of the 3-sphere the set is sampled on.
    pub rho: f64,
    /// S3: which plane the time angle rotates: 0 = (x,w), 1 = (y,w), 2 = (z,w).
    /// (Planes among x,y,z only spin the picture, so they are not offered.)
    /// Default 2: with the usual axes that is the B-C plane, two imaginary
    /// directions, which the formulas treat almost symmetrically, so the sweep
    /// is smooth. Plane 0 mixes in the real axis and roughly doubles the
    /// frame-to-frame change and the flash index (CPU study, 2026-10-15).
    pub plane: u8,
    /// S3: static 4D rotations in the (x,w), (y,w) and (z,w) planes that move
    /// the projection point off the w axis (radians). Applied in the order
    /// zw, yw, xw, before the time rotation.
    pub pole: [f64; 3],
    /// Tilt: rotation angles in the (z,t) and (x,t) planes (radians).
    pub tilt: (f64, f64),
    /// Slab: half-thickness along t.
    pub slab_half: f64,
    /// Offset: inflation of the surface (distance units).
    pub offset: f64,
}

impl Default for Collapse {
    fn default() -> Self {
        Collapse { method: Method::Slice, rho: 1.0, plane: 2, pole: [0.0; 3], tilt: (0.6, 0.45), slab_half: 1.6 * 0.06, offset: 1.6 * 0.02 }
    }
}

static CURRENT: AtomicU32 = AtomicU32::new(0);
/// Process-wide parameter overrides (radius, plane, ...) for the default collapse.
static GLOBAL_PARAMS: std::sync::Mutex<Option<Collapse>> = std::sync::Mutex::new(None);

pub fn set(m: Method) { CURRENT.store(m as u32, Ordering::Relaxed); }
pub fn current() -> Method { ALL[(CURRENT.load(Ordering::Relaxed) as usize).min(ALL.len() - 1)] }
/// Sets the process-wide default method together with its parameters.
pub fn set_global(c: Collapse) {
    set(c.method);
    *GLOBAL_PARAMS.lock().unwrap() = Some(c);
}

/// Reads `NN_COLLAPSE` once at startup if set.
pub fn init_from_env() {
    if let Ok(s) = std::env::var("NN_COLLAPSE") {
        match Method::parse(&s) { Some(m) => set(m), None => eprintln!("[collapse] unknown NN_COLLAPSE={s:?}") }
    }
}

impl Collapse {
    /// The process-wide default method with default parameters.
    pub fn global() -> Collapse {
        let base = GLOBAL_PARAMS.lock().unwrap().unwrap_or_default();
        Collapse { method: current(), ..base }
    }
    pub fn of(method: Method) -> Collapse {
        let rho = match method { Method::Spin | Method::IsoSpin => 0.5, _ => 1.0 };
        Collapse { method, rho, ..Collapse::default() }
    }
    /// What a render with `explicit` actually uses.
    pub fn effective(explicit: Option<Collapse>) -> Collapse { explicit.unwrap_or_else(Collapse::global) }

    /// Mode word + 5 parameters for the shader (`col_mode`, `col_p0..p4`).
    pub fn uniform_words(&self) -> (u32, [f64; 5]) {
        let p = match self.method {
            Method::Slice | Method::Proj3 => [0.0; 5],
            Method::Tilt => [self.tilt.0, self.tilt.1, 0.0, 0.0, 0.0],
            Method::Slab => [self.slab_half, 0.0, 0.0, 0.0, 0.0],
            Method::Offset => [self.offset, 0.0, 0.0, 0.0, 0.0],
            Method::S3 | Method::IsoS3 => [self.rho, self.plane.min(2) as f64, self.pole[0], self.pole[1], self.pole[2]],
            Method::Spin | Method::IsoSpin => [self.rho, self.plane.min(2) as f64, 0.0, 0.0, 0.0],
        };
        (self.method as u32, p)
    }

    /// The 3-sphere point for view coordinates `v` at time `t`, plus the
    /// conformal factor (view distance = 4D distance / factor). CPU twin of
    /// the shader's point mapping; only Tilt and S3 move the point.
    /// Inverse of the S3 mapping for a point `p4` on the sphere: the view
    /// coordinates (stereographic) that show it at time `t`, or `None` at the
    /// projection pole. Used by the Planes window to mark what is visible.
    pub fn s3_view_coords(&self, p4: [f64; 4], t: f64) -> Option<(f64, f64, f64)> {
        let rot = |a: f64, b: f64, ang: f64| { let (s, c) = ang.sin_cos(); (a * c - b * s, a * s + b * c) };
        let [mut x, mut y, mut z, mut w] = p4;
        if self.method == Method::IsoS3 {
            let q = iso_rotate(p4, self.plane, -t);
            x = q[0]; y = q[1]; z = q[2]; w = q[3];
        } else {
            match self.plane.min(2) {
                0 => { let r = rot(x, w, -t); x = r.0; w = r.1; }
                1 => { let r = rot(y, w, -t); y = r.0; w = r.1; }
                _ => { let r = rot(z, w, -t); z = r.0; w = r.1; }
            }
        }
        let r = rot(x, w, -self.pole[0]); x = r.0; w = r.1;
        let r = rot(y, w, -self.pole[1]); y = r.0; w = r.1;
        let r = rot(z, w, -self.pole[2]); z = r.0; w = r.1;
        let d = self.rho - w;
        if d.abs() < 1e-9 { return None; }
        Some((x / d, y / d, z / d))
    }

    pub fn map_point(&self, v: (f64, f64, f64), t: f64) -> ([f64; 4], f64) {
        let rot = |a: f64, b: f64, ang: f64| { let (s, c) = ang.sin_cos(); (a * c - b * s, a * s + b * c) };
        match self.method {
            Method::Tilt => {
                let (mut x, y, mut z, mut w) = (v.0, v.1, v.2, t);
                let r = rot(z, w, self.tilt.0); z = r.0; w = r.1;
                let r = rot(x, w, self.tilt.1); x = r.0; w = r.1;
                ([x, y, z, w], 1.0)
            }
            Method::S3 => {
                let rho = self.rho;
                let s2 = v.0 * v.0 + v.1 * v.1 + v.2 * v.2;
                let k = 1.0 / (1.0 + s2);
                let (mut x, mut y, mut z) = (2.0 * rho * k * v.0, 2.0 * rho * k * v.1, 2.0 * rho * k * v.2);
                let mut w = rho * (s2 - 1.0) * k;
                let r = rot(z, w, self.pole[2]); z = r.0; w = r.1;
                let r = rot(y, w, self.pole[1]); y = r.0; w = r.1;
                let r = rot(x, w, self.pole[0]); x = r.0; w = r.1;
                match self.plane.min(2) {
                    0 => { let r = rot(x, w, t); x = r.0; w = r.1; }
                    1 => { let r = rot(y, w, t); y = r.0; w = r.1; }
                    _ => { let r = rot(z, w, t); z = r.0; w = r.1; }
                }
                ([x, y, z, w], 2.0 * rho * k)
            }
            Method::IsoS3 => {
                let rho = self.rho;
                let s2 = v.0 * v.0 + v.1 * v.1 + v.2 * v.2;
                let k = 1.0 / (1.0 + s2);
                let (mut x, mut y, mut z) = (2.0 * rho * k * v.0, 2.0 * rho * k * v.1, 2.0 * rho * k * v.2);
                let mut w = rho * (s2 - 1.0) * k;
                let r = rot(z, w, self.pole[2]); z = r.0; w = r.1;
                let r = rot(y, w, self.pole[1]); y = r.0; w = r.1;
                let r = rot(x, w, self.pole[0]); x = r.0; w = r.1;
                (iso_rotate([x, y, z, w], self.plane, t), 2.0 * rho * k)
            }
            Method::Spin => {
                let (mut x, y, mut z, mut w) = (v.0, v.1, v.2, self.rho);
                let mut yy = y;
                match self.plane.min(2) {
                    0 => { let r = rot(x, w, t); x = r.0; w = r.1; }
                    1 => { let r = rot(yy, w, t); yy = r.0; w = r.1; }
                    _ => { let r = rot(z, w, t); z = r.0; w = r.1; }
                }
                ([x, yy, z, w], 1.0)
            }
            Method::IsoSpin => (iso_rotate([v.0, v.1, v.2, self.rho], self.plane, t), 1.0),
            _ => ([v.0, v.1, v.2, t], 1.0),
        }
    }
}

/// Range limits for the per-genome range factor (log scale around 1).
pub const RANGE_MIN: f64 = 0.15;
pub const RANGE_MAX: f64 = 8.0;
/// Default view domain radius (fractal-space units) at range 1.
pub const BASE_DOMAIN: f64 = 1.6;

/// The projection and view domain a genome is rendered with: `own` (or the
/// process default) with the genome's range factor applied. Under a sphere
/// method the factor scales the sphere radius and the view box stays put;
/// under the others it scales the view box. `range` 0 = unset = 1.
pub fn view_setup(own: Option<Collapse>, range: f32) -> (Option<Collapse>, f64) {
    let s = if range > 0.0 { (range as f64).clamp(RANGE_MIN, RANGE_MAX) } else { 1.0 };
    let c = Collapse::effective(own);
    if c.method.is_sphere() {
        (Some(Collapse { rho: c.rho * s, ..c }), BASE_DOMAIN)
    } else {
        (own, BASE_DOMAIN * s)
    }
}

/// Isoclinic rotation of a 4D point by `t`: the (a,w) plane and the plane of
/// the two remaining axes turn together. `plane` 0 pairs (x,w)+(y,z), 1 pairs
/// (y,w)+(x,z), 2 pairs (z,w)+(x,y). WGSL twin: `iso_rot` in the shader.
pub fn iso_rotate(p: [f64; 4], plane: u8, t: f64) -> [f64; 4] {
    let (s, c) = t.sin_cos();
    let [x, y, z, w] = p;
    match plane.min(2) {
        0 => [x * c - w * s, y * c - z * s, y * s + z * c, x * s + w * c],
        1 => [x * c - z * s, y * c - w * s, x * s + z * c, y * s + w * c],
        _ => [x * c - y * s, x * s + y * c, z * c - w * s, z * s + w * c],
    }
}

/// File-name suffix of the taste model / rating corpus for the process-wide
/// method: S3 renders are judged by their own model and ratings (`_s3`).
pub fn taste_suffix() -> &'static str {
    if current().time_is_angle() { "_s3" } else { "" }
}

/// [`taste_suffix`] for a genome that may carry its own projection.
pub fn taste_suffix_for(own: Option<Collapse>) -> &'static str {
    match own {
        Some(c) => if c.method.time_is_angle() { "_s3" } else { "" },
        None => taste_suffix(),
    }
}

/// The shader `time_val` for sweep phase `phase` in [0,1) (one full loop).
pub fn time_for_phase(bailout: f64, phase: f64) -> f64 {
    time_for_phase_m(current(), bailout, phase)
}

/// [`time_for_phase`] for an explicit method (genomes carry their own).
/// Non-angle methods (slice family) sweep T LINEARLY over [-h, +h), so phases
/// 0, ¼, ½, ¾ give -h, -h/2, 0, +h/2. The old h·sin(2π·phase) gave 0, +h, 0, -h: and
/// since DAG constants only use the (r, a) parts, most genomes (13 of 21 moving ones
/// measured 2026-10-18) look the same at -T and +T (mirrored), so only 2 of those 4
/// samples were distinct shapes. Linear keeps 3 distinct for those and full coverage
/// for the genomes that are not T-symmetric.
pub fn time_for_phase_m(method: Method, bailout: f64, phase: f64) -> f64 {
    if method.time_is_angle() { std::f64::consts::TAU * phase } else { crate::quat_live_view::view_c_half(bailout) * (2.0 * phase.rem_euclid(1.0) - 1.0) }
}

#[cfg(test)]
mod projection_tests {
    use super::*;

    fn norm(p: [f64; 4]) -> f64 { p.iter().map(|v| v * v).sum::<f64>().sqrt() }

    #[test]
    fn iso_rotate_keeps_length_and_is_periodic() {
        for plane in 0..3u8 {
            let p = [0.3, -0.7, 0.5, 0.2];
            let q = iso_rotate(p, plane, 1.234);
            assert!((norm(p) - norm(q)).abs() < 1e-12);
            let full = iso_rotate(p, plane, std::f64::consts::TAU);
            for k in 0..4 { assert!((p[k] - full[k]).abs() < 1e-12); }
        }
    }

    #[test]
    fn spin_at_zero_is_the_slice_at_rho_and_loops() {
        let c = Collapse { method: Method::Spin, rho: 0.4, plane: 2, ..Collapse::default() };
        let (p0, lam) = c.map_point((0.1, -0.2, 0.3), 0.0);
        assert_eq!(p0, [0.1, -0.2, 0.3, 0.4]);
        assert_eq!(lam, 1.0);
        let (p1, _) = c.map_point((0.1, -0.2, 0.3), std::f64::consts::TAU);
        for k in 0..4 { assert!((p0[k] - p1[k]).abs() < 1e-9); }
    }

    #[test]
    fn iso_s3_stays_on_the_sphere_and_inverts() {
        let c = Collapse { method: Method::IsoS3, rho: 1.3, plane: 1, pole: [0.2, -0.1, 0.3], ..Collapse::default() };
        for t in [0.0, 0.7, 2.9] {
            let (p4, _) = c.map_point((0.4, -0.8, 0.25), t);
            assert!((norm(p4) - 1.3).abs() < 1e-9);
            let back = c.s3_view_coords(p4, t).expect("not at the pole");
            assert!((back.0 - 0.4).abs() < 1e-8 && (back.1 + 0.8).abs() < 1e-8 && (back.2 - 0.25).abs() < 1e-8);
        }
    }

    #[test]
    fn view_setup_scales_the_right_thing() {
        let (c, d) = view_setup(Some(Collapse::of(Method::S3)), 2.0);
        assert_eq!(c.unwrap().rho, 2.0);
        assert_eq!(d, BASE_DOMAIN);
        let (c, d) = view_setup(Some(Collapse::of(Method::Slice)), 2.0);
        assert_eq!(c.unwrap().method, Method::Slice);
        assert_eq!(d, BASE_DOMAIN * 2.0);
        // unset = 1, and the factor is clamped
        assert_eq!(view_setup(Some(Collapse::of(Method::Slice)), 0.0).1, BASE_DOMAIN);
        assert_eq!(view_setup(Some(Collapse::of(Method::Slice)), 100.0).1, BASE_DOMAIN * RANGE_MAX);
    }

    #[test]
    fn genome_range_and_collapse_roundtrip_and_default() {
        let mut g = crate::genome::Genome::default();
        let json = serde_json::to_string(&g).unwrap();
        assert!(!json.contains("view_range") && !json.contains("\"collapse\""), "unset fields are not written");
        let back: crate::genome::Genome = serde_json::from_str(&json).unwrap();
        assert_eq!(back.view_range, 0.0);
        assert!(back.collapse.is_none());
        g.view_range = 2.5;
        g.collapse = Some(Collapse::of(Method::S3));
        let back: crate::genome::Genome = serde_json::from_str(&serde_json::to_string(&g).unwrap()).unwrap();
        assert_eq!(back.view_range, 2.5);
        assert_eq!(back.collapse.unwrap().method, Method::S3);
    }

    #[test]
    fn names_roundtrip_and_time_kind() {
        for m in ALL { assert_eq!(Method::parse(m.name()), Some(m)); }
        assert!(Method::Spin.time_is_angle() && Method::IsoS3.time_is_angle() && !Method::Slab.time_is_angle());
    }
}
