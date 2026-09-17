//! Quaternion escape-time renderer: q₀ = 0, q ← step(q) + Q, where Q is a 4D
//! point assembled from a 2D `Slice`'s 3 SPATIAL scalars plus a
//! caller-supplied time value, via a `TimeAxis` saying which of R/A/B/C
//! plays the time role for a given render (see `quat_motion` for how the
//! time value itself is driven, and `TimeAxis` below for the assembly).
//! Mandelbrot convention, not quaternion-Julia: the sampled point is the
//! additive constant, not a fixed offset applied to a varying starting
//! point.
//!
//! `step` is one of several `QuatFormula` variants — quaternion
//! generalizations of the known 2D formula catalog in `known_formulas.rs`
//! (Tricorn, Burning Ship family, Celtic, cubic/quartic power maps). Every
//! generalization is chosen so that restricted to the R,A-plane (B=C=0,
//! which that plane is closed under for every formula here) it reduces
//! EXACTLY to its named 2D counterpart — see the parity tests below.

use rayon::prelude::*;

use crate::quaternion::Quat;

/// A 2D affine projection/slice plane embedded in a 3D subspace of the
/// quaternion — WHICH 3D subspace depends on `TimeAxis` (see below);
/// `Slice` itself just carries 3 generic spatial scalars, oblivious to what
/// they'll be assembled into. `origin` is the point at screen center;
/// `basis_u`/`basis_v` are unit vectors giving screen +x/+y directions.
/// `zoom` follows `video_export::View`'s convention exactly: vertical
/// half-extent = 2.0/zoom, horizontal half-extent = that times the OUTPUT
/// canvas's aspect ratio (width/height) — aspect is never a field on
/// `Slice` itself.
#[derive(Copy, Clone, Debug)]
pub struct Slice {
    pub origin: (f64, f64, f64),
    pub basis_u: (f64, f64, f64),
    pub basis_v: (f64, f64, f64),
    pub zoom: f64,
    /// `None` (the default everywhere except an opted-in gravity camera) =
    /// orthographic, byte-identical to this struct's original behavior. See
    /// `PerspectiveCamera` for what `Some` changes and why a naive "pull the
    /// eye straight back" is NOT enough to turn a centered circle into an
    /// ellipse.
    pub camera: Option<PerspectiveCamera>,
}

/// An eye positioned off the `Slice`'s own plane, still aimed at `origin`,
/// used to build a genuine perspective (pinhole) pixel→spatial-point mapping
/// instead of the plain orthographic affine one.
///
/// **Why `tilt_u`/`tilt_v` matter and pure `distance` doesn't fix roundness
/// on its own**: pulling the eye straight back along the plane's own normal
/// (`tilt_u = tilt_v = 0`) keeps the camera's viewing axis EXACTLY aligned
/// with that normal — a rotationally symmetric setup. A pattern centered on
/// `origin` (e.g. the spherically-symmetric formulas' concentric circles
/// under `TimeAxis::R` — see the module docs) stays exactly circular under
/// any such straight-back pullback, no matter the distance, because nothing
/// breaks the rotational symmetry about the viewing axis. What actually
/// produces foreshortening (a circle appearing elliptical) is an angle
/// between the viewing axis and the plane's own normal — like viewing a
/// round table from the side rather than from directly overhead, even while
/// still looking straight at its center. `tilt_u`/`tilt_v` displace the eye
/// SIDEWAYS (in the plane's own `basis_u`/`basis_v` units) while the camera
/// keeps aiming at `origin` — that's what introduces the tilt.
#[derive(Copy, Clone, Debug)]
pub struct PerspectiveCamera {
    pub tilt_u: f64,
    pub tilt_v: f64,
    /// Pullback distance behind the plane along its own normal. Must be > 0.
    pub distance: f64,
}

impl Slice {
    /// The default/"classic" slice: screen x = local-0, screen y = local-1,
    /// local-2 held at 0. Under the default `TimeAxis::C` this is screen
    /// x=R, y=A, B=0 — with C also held at 0 this must reduce EXACTLY to
    /// the ordinary 2D formula of the same name, see the parity tests below.
    pub fn classic() -> Self {
        Slice {
            origin: (0.0, 0.0, 0.0),
            basis_u: (1.0, 0.0, 0.0),
            basis_v: (0.0, 1.0, 0.0),
            zoom: 1.0,
            camera: None,
        }
    }

    /// Pixel (px,py) of a width×height canvas → 3 spatial scalars, in
    /// Slice-local order (NOT necessarily R,A,B — see `TimeAxis`). Mirrors
    /// `View::pixel_to_fractal`'s pixel mapping exactly
    /// (`cx = xmin + (px/wf)*(xmax-xmin)`), generalized from axis-aligned
    /// (xmin/xmax on R, ymin/ymax on A) to an arbitrary origin + basis pair.
    ///
    /// With `camera: None` this is the plain orthographic formula, unchanged
    /// from before `PerspectiveCamera` existed. With `camera: Some(cam)`,
    /// `(u,v)` instead locate a point on a virtual near-image-plane one unit
    /// in front of a pinhole eye (positioned via `cam`, aimed at `origin`),
    /// and the returned point is where the ray from the eye through that
    /// point hits THIS plane — see `PerspectiveCamera`'s doc comment.
    pub fn pixel_to_spatial(&self, px: u32, py: u32, width: u32, height: u32) -> (f64, f64, f64) {
        let half_y = 2.0 / self.zoom;
        let aspect = if height > 0 {
            width as f64 / height as f64
        } else {
            1.0
        };
        let half_x = half_y * aspect;
        let wf = width.saturating_sub(1).max(1) as f64;
        let hf = height.saturating_sub(1).max(1) as f64;
        let u = ((px as f64 / wf) * 2.0 - 1.0) * half_x;
        let v = ((py as f64 / hf) * 2.0 - 1.0) * half_y;
        let ortho = (
            self.origin.0 + u * self.basis_u.0 + v * self.basis_v.0,
            self.origin.1 + u * self.basis_u.1 + v * self.basis_v.1,
            self.origin.2 + u * self.basis_u.2 + v * self.basis_v.2,
        );
        let Some(cam) = self.camera else {
            return ortho;
        };
        use crate::quat_motion::{add, cross, dot, look_at_basis, normalize, scale, sub};
        let normal = normalize(cross(self.basis_u, self.basis_v));
        let eye = sub(
            add(add(self.origin, scale(self.basis_u, cam.tilt_u)), scale(self.basis_v, cam.tilt_v)),
            scale(normal, cam.distance),
        );
        let to_origin = sub(self.origin, eye);
        let dist_to_origin = dot(to_origin, to_origin).sqrt();
        if dist_to_origin < 1e-9 {
            return ortho; // eye coincides with origin — degenerate, fall back
        }
        let forward = scale(to_origin, 1.0 / dist_to_origin);
        let (right, up) = look_at_basis(forward, self.basis_v);
        let ray_dir = add(add(forward, scale(right, u)), scale(up, v));
        let denom = dot(ray_dir, normal);
        if denom.abs() < 1e-9 {
            return ortho; // ray parallel to the target plane — degenerate, fall back
        }
        let t = dot(sub(self.origin, eye), normal) / denom;
        add(eye, scale(ray_dir, t))
    }
}

/// Which quaternion component is driven by TIME for a given render; the
/// other three are the SPATIAL subspace `Slice` lives in (in Slice-local
/// order). Swapping this is a genuine change of subject, not just a
/// relabeling: for every formula whose `step` only ever touches the vector
/// part by SCALING it (never by transforming its individual components) —
/// Mandelbrot/Tricorn/Cubic/Quartic, and also `Celtic` (it only takes abs
/// of the resulting REAL component after squaring, leaving the vector part
/// untouched) — the orbit's vector part is provably confined to a fixed
/// line through the origin (see `quat_fractal`'s module docs on the "solid
/// of revolution" finding). Under the default `C`, that line sits in the
/// (A,B) plane the SPATIAL slice only partly explores, so a slice through
/// (R,A,B) still shows a real cardioid-and-bulb (or, for Celtic, leaf)
/// profile. Under `R`, though, R becomes the FIXED axis and the spatial
/// subspace is the ENTIRE vector part (A,B,C) — so escape time for those
/// same formulas collapses to a function of `ρ=|(A,B,C)|` alone, i.e. a
/// perfect sphere: any slice through it is just concentric circles, no
/// cardioid visible at all. Only the formulas that abs the vector
/// COMPONENTS individually before squaring — `BurningShip*` and
/// `Perpendicular*` — plus `Bulb`, actually break the "vector part stays on
/// one line" invariant and stay genuinely structured under any `TimeAxis`
/// — see the tests below for both halves of this claim, checked directly
/// rather than just asserted (this file's own first attempt at this list
/// wrongly included Celtic in the "breaks it" group — caught by actually
/// testing it, not just re-deriving the abs-on-vector-components argument
/// by eye).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TimeAxis {
    R,
    A,
    B,
    C,
}

impl TimeAxis {
    pub fn name(self) -> &'static str {
        match self {
            TimeAxis::R => "r",
            TimeAxis::A => "a",
            TimeAxis::B => "b",
            TimeAxis::C => "c",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "r" => TimeAxis::R,
            "a" => TimeAxis::A,
            "b" => TimeAxis::B,
            "c" => TimeAxis::C,
            _ => return None,
        })
    }

    /// Assembles a full `Quat` from 3 spatial scalars (in `Slice`-local
    /// order) plus the time-driven scalar, placing each in the component
    /// this axis says it belongs in.
    #[inline]
    pub fn assemble(self, spatial: (f64, f64, f64), time_val: f64) -> Quat {
        let (x, y, z) = spatial;
        match self {
            TimeAxis::R => Quat::new(time_val, x, y, z),
            TimeAxis::A => Quat::new(x, time_val, y, z),
            TimeAxis::B => Quat::new(x, y, time_val, z),
            TimeAxis::C => Quat::new(x, y, z, time_val),
        }
    }
}

/// Quaternion generalizations of `known_formulas.rs`'s classic power-map /
/// non-holomorphic-fold catalog. Each variant computes `step(q)`; the caller
/// adds `q_const` afterward (see `quat_escape`), mirroring the 2D
/// convention of every formula in that catalog (`f(z) + c`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum QuatFormula {
    /// q² — the archetypal escape-time fractal, generalized to 4D.
    Mandelbrot,
    /// conj(q)² — Milnor's antiholomorphic analogue, generalized.
    Tricorn,
    /// abs every component, then square — Burning Ship, generalized.
    BurningShip,
    /// abs every component, then cube.
    BurningShipCubic,
    /// abs and negate the vector part only, then square — "imaginary axis
    /// only" fold, generalized from (Re z − i|Im z|)².
    PerpendicularBurningShip,
    /// square, then abs only the resulting r component — Celtic Mandelbrot,
    /// generalized from |Re(z²)| + i·Im(z²).
    Celtic,
    /// abs the vector part only (no negation), then square — generalized
    /// from (Re z + i|Im z|)².
    PerpendicularMandelbrot,
    /// q³ — degree-3 power map.
    Cubic,
    /// q⁴ — degree-4 power map.
    Quartic,
    /// The quaternion generalization of the Mandelbulb's spherical-power
    /// trick: multiply the ANGLES by n, not just the magnitude. Unlike
    /// every formula above, this one breaks the "orbit stays on a fixed
    /// line through the origin" degeneracy (see module docs) — it produces
    /// genuinely 3D structure, not a solid of revolution of the 2D set.
    Bulb,
    /// Tom Lowe's ("Tglad") Mandelbox (2010): `v = scale·ballFold(boxFold(v))`,
    /// generalized to all 4 components instead of just 3 — a fundamentally
    /// different iteration mechanic from every formula above (fold + radial
    /// clamp + scale, not a power map), added on Carl's ask to bring in
    /// fractal types known-good from the Mandelbulb3D/Mandelbulber
    /// community rather than only quaternion-generalized 2D escape-time
    /// maps. Needs its own distance-estimate derivative recurrence — see
    /// `quat_escape_de`'s dispatch on `self` — the generic power-law one
    /// every other formula shares does not apply (this isn't a power map).
    Mandelbox,
}

/// Power used by `Bulb` — 8 is the Mandelbulb's own "most popular choice"
/// (see the distance-estimated-fractals write-up); not yet exposed as a CLI
/// flag, but the whole point of factoring `step` this way is that it could
/// be.
const BULB_POWER: f64 = 8.0;

/// `Mandelbox` parameters — Tom Lowe's original ("Amazing Box") defaults,
/// the single most commonly cited starting point across every Mandelbulb3D/
/// Mandelbulber preset library and write-up: fold limit ±1 on each
/// component, ball-fold minimum/fixed radius 0.5/1.0, scale -1.5 (negative
/// scale is what gives the classic result its non-trivial rotation/
/// reflection character; the plain positive-scale version looks far more
/// repetitive).
const MANDELBOX_FOLD_LIMIT: f64 = 1.0;
const MANDELBOX_MIN_RADIUS_SQ: f64 = 0.25; // 0.5²
const MANDELBOX_FIXED_RADIUS_SQ: f64 = 1.0; // 1.0²
const MANDELBOX_SCALE: f64 = -1.5;

/// Reflects each component exceeding `MANDELBOX_FOLD_LIMIT` back across the
/// nearest face of the hypercube `[-limit, limit]^4` — the "box" in
/// box-fold. Local derivative magnitude is exactly 1 everywhere (a
/// reflection, not a scaling), which is why it doesn't appear in
/// `mandelbox_de_factor` below.
#[inline]
fn box_fold(q: Quat) -> Quat {
    let f = |x: f64| {
        if x > MANDELBOX_FOLD_LIMIT {
            2.0 * MANDELBOX_FOLD_LIMIT - x
        } else if x < -MANDELBOX_FOLD_LIMIT {
            -2.0 * MANDELBOX_FOLD_LIMIT - x
        } else {
            x
        }
    };
    Quat::new(f(q.r), f(q.a), f(q.b), f(q.c))
}

/// Radial clamp: points inside `MANDELBOX_MIN_RADIUS_SQ` get pushed outward
/// by a fixed factor, points in the shell between min and fixed radius get
/// inverted through the sphere (`fixedR²/r²`), points outside fixed radius
/// pass through unchanged. Returns the scaled quaternion AND the factor it
/// scaled by (1.0 in the unchanged case) so the DE derivative tracker can
/// apply the identical multiplier to `dr`.
#[inline]
fn ball_fold(q: Quat) -> (Quat, f64) {
    let r2 = q.norm_sq();
    let factor = if r2 < MANDELBOX_MIN_RADIUS_SQ {
        MANDELBOX_FIXED_RADIUS_SQ / MANDELBOX_MIN_RADIUS_SQ
    } else if r2 < MANDELBOX_FIXED_RADIUS_SQ {
        MANDELBOX_FIXED_RADIUS_SQ / r2
    } else {
        1.0
    };
    (Quat::new(q.r * factor, q.a * factor, q.b * factor, q.c * factor), factor)
}

impl QuatFormula {
    pub fn name(self) -> &'static str {
        match self {
            QuatFormula::Mandelbrot => "mandelbrot",
            QuatFormula::Tricorn => "tricorn",
            QuatFormula::BurningShip => "burning-ship",
            QuatFormula::BurningShipCubic => "burning-ship-cubic",
            QuatFormula::PerpendicularBurningShip => "perpendicular-burning-ship",
            QuatFormula::Celtic => "celtic",
            QuatFormula::PerpendicularMandelbrot => "perpendicular-mandelbrot",
            QuatFormula::Cubic => "cubic",
            QuatFormula::Quartic => "quartic",
            QuatFormula::Bulb => "bulb",
            QuatFormula::Mandelbox => "mandelbox",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "mandelbrot" => QuatFormula::Mandelbrot,
            "tricorn" => QuatFormula::Tricorn,
            "burning-ship" => QuatFormula::BurningShip,
            "burning-ship-cubic" => QuatFormula::BurningShipCubic,
            "perpendicular-burning-ship" => QuatFormula::PerpendicularBurningShip,
            "celtic" => QuatFormula::Celtic,
            "perpendicular-mandelbrot" => QuatFormula::PerpendicularMandelbrot,
            "cubic" => QuatFormula::Cubic,
            "quartic" => QuatFormula::Quartic,
            "bulb" => QuatFormula::Bulb,
            "mandelbox" => QuatFormula::Mandelbox,
            _ => return None,
        })
    }

    /// This formula's effective magnitude-scaling power — used ONLY by
    /// `quat_escape_de`'s running-derivative recurrence
    /// (`dr' = n·ρ^(n-1)·dr + 1`), never by `step`/`quat_escape`
    /// themselves. `abs()` operations (`BurningShip*`, `Perpendicular*`,
    /// `Celtic`) don't change a value's MAGNITUDE, so each shares its
    /// non-abs sibling's power — the same approximation real Mandelbulb-
    /// family renderers use for their own abs-based variants (Burning
    /// Ship 3D etc.), since none of these maps are truly holomorphic
    /// enough for an exact analytic derivative anyway.
    fn power(self) -> f64 {
        match self {
            QuatFormula::Mandelbrot
            | QuatFormula::Tricorn
            | QuatFormula::BurningShip
            | QuatFormula::PerpendicularBurningShip
            | QuatFormula::Celtic
            | QuatFormula::PerpendicularMandelbrot => 2.0,
            QuatFormula::Cubic | QuatFormula::BurningShipCubic => 3.0,
            QuatFormula::Quartic => 4.0,
            QuatFormula::Bulb => BULB_POWER,
            // Unused: quat_escape_de special-cases Mandelbox entirely
            // (it isn't a power map), never reads this. Any finite value
            // is fine here purely so the match stays exhaustive.
            QuatFormula::Mandelbox => 1.0,
        }
    }

    pub const ALL: [QuatFormula; 11] = [
        QuatFormula::Mandelbrot,
        QuatFormula::Tricorn,
        QuatFormula::BurningShip,
        QuatFormula::BurningShipCubic,
        QuatFormula::PerpendicularBurningShip,
        QuatFormula::Celtic,
        QuatFormula::PerpendicularMandelbrot,
        QuatFormula::Cubic,
        QuatFormula::Quartic,
        QuatFormula::Bulb,
        QuatFormula::Mandelbox,
    ];

    /// One iteration step, BEFORE adding `q_const` — `quat_escape` does
    /// `q = formula.step(q) + q_const`. Uses the module defaults
    /// (`BULB_POWER`/`MANDELBOX_SCALE`) — see `step_with_params` for the
    /// override-capable version every real render call site now uses.
    #[inline]
    fn step(self, q: Quat) -> Quat {
        self.step_with_params(q, BULB_POWER, MANDELBOX_SCALE)
    }

    /// Same as `step`, but with `Bulb`'s power / `Mandelbox`'s scale
    /// overridable per-call instead of hardcoded — added so imported
    /// Mandelbulber parameter files (which vary these per-file) can
    /// actually render at their own values. Meaningless (ignored) for
    /// every formula other than the one it names.
    #[inline]
    fn step_with_params(self, q: Quat, bulb_power: f64, mandelbox_scale: f64) -> Quat {
        match self {
            QuatFormula::Mandelbrot => q.mul(q),
            QuatFormula::Tricorn => {
                let c = q.conj();
                c.mul(c)
            }
            QuatFormula::BurningShip => {
                let a = q.abs_components();
                a.mul(a)
            }
            QuatFormula::BurningShipCubic => {
                let a = q.abs_components();
                a.mul(a).mul(a)
            }
            QuatFormula::PerpendicularBurningShip => {
                let p = Quat::new(q.r, -q.a.abs(), -q.b.abs(), -q.c.abs());
                p.mul(p)
            }
            QuatFormula::Celtic => {
                let sq = q.mul(q);
                Quat::new(sq.r.abs(), sq.a, sq.b, sq.c)
            }
            QuatFormula::PerpendicularMandelbrot => {
                let p = Quat::new(q.r, q.a.abs(), q.b.abs(), q.c.abs());
                p.mul(p)
            }
            QuatFormula::Cubic => q.mul(q).mul(q),
            QuatFormula::Quartic => {
                let sq = q.mul(q);
                sq.mul(sq)
            }
            QuatFormula::Bulb => {
                // Polar decomposition: rho = |q|, theta1 = angle from the
                // real axis (like the Mandelbulb's theta from the z-axis),
                // theta2/phi = the vector part's OWN direction within its
                // 3-space (like the Mandelbulb's theta/phi for x,y,z).
                // Scaling rho by n while multiplying ALL THREE angles by n
                // is what makes the output direction genuinely different
                // from the input direction — unlike plain quaternion
                // multiplication, which can only ever scale the existing
                // vector direction, never rotate it (see module docs).
                // atan2 is well-defined at (0,0) (returns 0.0), so no
                // special-casing is needed for on-axis or zero inputs.
                let n = bulb_power;
                let v_mag = (q.a * q.a + q.b * q.b + q.c * q.c).sqrt();
                let rho = q.norm_sq().sqrt();
                let theta1 = v_mag.atan2(q.r);
                let theta2 = (q.a * q.a + q.b * q.b).sqrt().atan2(q.c);
                let phi = q.b.atan2(q.a);
                let rho_n = rho.powf(n);
                let (t1, t2, ph) = (theta1 * n, theta2 * n, phi * n);
                let new_r = rho_n * t1.cos();
                let new_vmag = rho_n * t1.sin();
                Quat::new(
                    new_r,
                    new_vmag * t2.sin() * ph.cos(),
                    new_vmag * t2.sin() * ph.sin(),
                    new_vmag * t2.cos(),
                )
            }
            QuatFormula::Mandelbox => {
                let (folded, _factor) = ball_fold(box_fold(q));
                Quat::new(
                    folded.r * mandelbox_scale,
                    folded.a * mandelbox_scale,
                    folded.b * mandelbox_scale,
                    folded.c * mandelbox_scale,
                )
            }
        }
    }
}

/// Quaternion escape time under `formula`: q₀ = 0, q ← formula.step(q) +
/// q_const (Mandelbrot convention). Mirrors `dag_escape_pixel_f64`'s loop
/// shape and smooth-coloring formula exactly (`fractal.rs`) so quaternion
/// renders read consistently with the rest of the app.
pub fn quat_escape(formula: QuatFormula, q_const: Quat, max_iter: u32, bailout_sq: f64) -> f32 {
    let mut q = Quat::ZERO;
    for it in 0..max_iter {
        q = formula.step(q).add(q_const);
        let ms = q.norm_sq();
        if ms > bailout_sq {
            return ((it as f64 + 1.0) - (ms.log2() * 0.5).log2()).max(0.0) as f32;
        }
        if !q.is_finite() {
            return it as f32;
        }
    }
    max_iter as f32
}

/// `base.powf(exp)` for the common case in this file: `exp` is always one
/// of `power() - 1.0` (1, 2, 3, or 7 for the current formula set — every
/// `power()` is a small positive integer) — exponentiation by squaring
/// does the same computation as `f64::powf` for an integer exponent in a
/// handful of multiplications instead of a full transcendental (log/exp)
/// call, which matters here since this runs on every single iteration of
/// every march step of every ray. Falls back to `powf` for a non-integer
/// or out-of-range exponent so this stays correct even if a future
/// formula's `power()` ever isn't a small integer.
#[inline]
fn pow_fast(base: f64, exp: f64) -> f64 {
    if exp >= 0.0 && exp <= 32.0 && exp.fract() == 0.0 {
        let mut result = 1.0;
        let mut b = base;
        let mut e = exp as u32;
        while e > 0 {
            if e & 1 == 1 {
                result *= b;
            }
            b *= b;
            e >>= 1;
        }
        result
    } else {
        base.powf(exp)
    }
}

/// Like `quat_escape`, but also returns a distance ESTIMATE, not just the
/// escape time — the standard running-derivative technique real Mandelbulb/
/// Burning-Ship-3D renderers use for ray-marching normals and adaptive step
/// sizes, since raw escape time is famously a terrible source for either:
/// near a chaotic fractal boundary, iteration count is extremely sensitive
/// to position (that IS the boundary's defining property), so any
/// finite-difference gradient of it stays noisy at every scale — confirmed
/// empirically in `quat_raymarch` (see its module docs) before reaching
/// for this. A distance estimate instead tracks how fast the ORBIT's
/// magnitude is diverging (`dr' = n·ρ^(n-1)·dr + 1`, `n` = `power()`) and
/// converts that into `DE = 0.5·ln(ρ)·ρ/dr` at bailout — a much
/// better-behaved (Lipschitz-ish) quantity near the boundary.
///
/// Deep-interior points (loop reaches `max_iter` without escaping) get a
/// DE computed from the final ρ/dr and clamped to `0.0` if negative
/// (`ln(ρ)<0` for ρ<1, common well inside the set) — not a meaningful
/// distance there, but callers here only ever need DE close to a boundary
/// crossing, so this is left unrefined rather than special-cased.
pub fn quat_escape_de(formula: QuatFormula, q_const: Quat, max_iter: u32, bailout_sq: f64) -> (f32, f64) {
    quat_escape_de_params(formula, q_const, max_iter, bailout_sq, BULB_POWER, MANDELBOX_SCALE)
}

/// Same as `quat_escape_de`, but with `Bulb`'s power / `Mandelbox`'s scale
/// overridable per-call — see `RaymarchParams::bulb_power`/
/// `mandelbox_scale`'s doc comments for why (imported Mandelbulber
/// parameter files). `n` (used for every power-map formula's DE
/// recurrence) is recomputed from `bulb_power` rather than
/// `formula.power()` when `formula` is `Bulb`, so a non-default power
/// actually affects the distance estimate too, not just `step`.
pub fn quat_escape_de_params(
    formula: QuatFormula,
    q_const: Quat,
    max_iter: u32,
    bailout_sq: f64,
    bulb_power: f64,
    mandelbox_scale: f64,
) -> (f32, f64) {
    let n = if formula == QuatFormula::Bulb { bulb_power } else { formula.power() };
    let mut q = Quat::ZERO;
    let mut dr = 1.0_f64;
    for it in 0..max_iter {
        // Mandelbox isn't a power map (`step` folds + radially clamps +
        // scales, never raises `q` to a power), so the generic
        // `n·ρ^(n-1)` recurrence below doesn't apply — box-fold's local
        // derivative magnitude is 1 everywhere (a reflection), so only
        // ball-fold's radial factor and the final `scale` multiply the
        // running derivative. Recomputing the fold here (rather than
        // having `step` return its factor) duplicates a cheap vector op,
        // not the expensive part of a march step.
        if formula == QuatFormula::Mandelbox {
            let (_, ball_factor) = ball_fold(box_fold(q));
            dr = mandelbox_scale.abs() * ball_factor * dr + 1.0;
        } else {
            let rho = q.norm_sq().sqrt();
            dr = n * pow_fast(rho, n - 1.0) * dr + 1.0;
        }
        q = formula.step_with_params(q, bulb_power, mandelbox_scale).add(q_const);
        let ms = q.norm_sq();
        if ms > bailout_sq {
            let et = ((it as f64 + 1.0) - (ms.log2() * 0.5).log2()).max(0.0) as f32;
            let r = ms.sqrt();
            let de = (0.5 * r.ln() * r / dr.max(1e-300)).max(0.0);
            return (et, de);
        }
        if !q.is_finite() {
            return (it as f32, 0.0);
        }
    }
    let r = q.norm_sq().sqrt().max(1e-300);
    let de = (0.5 * r.ln() * r / dr.max(1e-300)).max(0.0);
    (max_iter as f32, de)
}

/// One pixel: `Slice` → 3 spatial scalars, paired with the caller-supplied
/// `time_val` (driven by the animation's time axis, never by the pixel) →
/// one quaternion escape time under `formula`, assembled per `time_axis`.
#[allow(clippy::too_many_arguments)]
pub fn quat_mandelbrot_pixel(
    formula: QuatFormula,
    time_axis: TimeAxis,
    slice: &Slice,
    time_val: f64,
    px: u32,
    py: u32,
    width: u32,
    height: u32,
    max_iter: u32,
    bailout_sq: f64,
) -> f32 {
    let spatial = slice.pixel_to_spatial(px, py, width, height);
    quat_escape(formula, time_axis.assemble(spatial, time_val), max_iter, bailout_sq)
}

/// One full frame's escape-time buffer [H*W], Rayon-parallel over pixels.
/// Feeds straight into `colormap::apply_colormap`.
#[allow(clippy::too_many_arguments)]
pub fn render_quat_frame(
    formula: QuatFormula,
    time_axis: TimeAxis,
    slice: &Slice,
    time_val: f64,
    width: u32,
    height: u32,
    max_iter: u32,
    bailout_sq: f64,
) -> Vec<f32> {
    let n = (width as usize) * (height as usize);
    (0..n)
        .into_par_iter()
        .map(|idx| {
            let px = (idx as u32) % width;
            let py = (idx as u32) / width;
            quat_mandelbrot_pixel(formula, time_axis, slice, time_val, px, py, width, height, max_iter, bailout_sq)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Independent, standalone re-implementation of a classic 2D
    /// escape-time formula — deliberately NOT calling into `fractal.rs` or
    /// `known_formulas.rs`, so this is a genuine cross-check rather than the
    /// same code compared to itself. `step2d` takes (zx,zy) and returns the
    /// next (zx,zy) BEFORE adding (cx,cy), mirroring `QuatFormula::step`.
    fn classic_2d_escape(
        step2d: impl Fn(f64, f64) -> (f64, f64),
        cx: f64,
        cy: f64,
        max_iter: u32,
        bailout_sq: f64,
    ) -> f32 {
        let (mut zx, mut zy) = (0.0f64, 0.0f64);
        for it in 0..max_iter {
            let (sx, sy) = step2d(zx, zy);
            zx = sx + cx;
            zy = sy + cy;
            let ms = zx * zx + zy * zy;
            if ms > bailout_sq {
                return ((it as f64 + 1.0) - (ms.log2() * 0.5).log2()).max(0.0) as f32;
            }
        }
        max_iter as f32
    }

    /// Runs `formula` restricted to the classic R,A slice (B=C=0) against an
    /// independent 2D re-implementation of `step2d`, over a small grid, and
    /// asserts they match pixel-for-pixel. This is the load-bearing
    /// correctness check for every formula: it works because every `step`
    /// above keeps b=c=0 invariant when it started at 0 (proven per-formula
    /// by inspection: each is built from `mul`, `conj`, and per-component
    /// `abs`, none of which can introduce a nonzero b/c from an all-real
    /// R,A input), so the quaternion iteration degenerates to exactly the
    /// 2D formula at every step.
    fn assert_formula_matches_2d(formula: QuatFormula, step2d: impl Fn(f64, f64) -> (f64, f64) + Copy) {
        let slice = Slice::classic();
        let (max_iter, bailout_sq, w, h) = (96u32, 16.0, 32u32, 32u32);
        for py in 0..h {
            for px in 0..w {
                let (r, a, b) = slice.pixel_to_spatial(px, py, w, h);
                assert_eq!(b, 0.0);
                let quat_et = quat_mandelbrot_pixel(formula, TimeAxis::C, &slice, 0.0, px, py, w, h, max_iter, bailout_sq);
                let classic_et = classic_2d_escape(step2d, r, a, max_iter, bailout_sq);
                assert!(
                    (quat_et - classic_et).abs() < 1e-4,
                    "{}: px={px} py={py} quat={quat_et} classic={classic_et}",
                    formula.name()
                );
            }
        }
    }

    #[test]
    fn mandelbrot_matches_2d_mandelbrot_exactly() {
        // z^2
        assert_formula_matches_2d(QuatFormula::Mandelbrot, |x, y| (x * x - y * y, 2.0 * x * y));
    }

    #[test]
    fn tricorn_matches_2d_tricorn_exactly() {
        // conj(z)^2 = (x - iy)^2 = (x^2 - y^2, -2xy)
        assert_formula_matches_2d(QuatFormula::Tricorn, |x, y| (x * x - y * y, -2.0 * x * y));
    }

    #[test]
    fn burning_ship_matches_2d_burning_ship_exactly() {
        // (|x| + i|y|)^2
        assert_formula_matches_2d(QuatFormula::BurningShip, |x, y| {
            let (ax, ay) = (x.abs(), y.abs());
            (ax * ax - ay * ay, 2.0 * ax * ay)
        });
    }

    #[test]
    fn perpendicular_burning_ship_matches_2d_exactly() {
        // (x - i|y|)^2
        assert_formula_matches_2d(QuatFormula::PerpendicularBurningShip, |x, y| {
            let ay = y.abs();
            (x * x - ay * ay, -2.0 * x * ay)
        });
    }

    #[test]
    fn celtic_matches_2d_celtic_exactly() {
        // |Re(z^2)| + i*Im(z^2)
        assert_formula_matches_2d(QuatFormula::Celtic, |x, y| ((x * x - y * y).abs(), 2.0 * x * y));
    }

    #[test]
    fn perpendicular_mandelbrot_matches_2d_exactly() {
        // (x + i|y|)^2
        assert_formula_matches_2d(QuatFormula::PerpendicularMandelbrot, |x, y| {
            let ay = y.abs();
            (x * x - ay * ay, 2.0 * x * ay)
        });
    }

    #[test]
    fn cubic_matches_2d_cubic_exactly() {
        // z^3 via complex multiplication
        assert_formula_matches_2d(QuatFormula::Cubic, |x, y| {
            let (x2, y2) = (x * x - y * y, 2.0 * x * y);
            (x2 * x - y2 * y, x2 * y + y2 * x)
        });
    }

    #[test]
    fn quartic_matches_2d_quartic_exactly() {
        // z^4 = (z^2)^2
        assert_formula_matches_2d(QuatFormula::Quartic, |x, y| {
            let (x2, y2) = (x * x - y * y, 2.0 * x * y);
            (x2 * x2 - y2 * y2, 2.0 * x2 * y2)
        });
    }

    #[test]
    fn burning_ship_cubic_matches_2d_exactly() {
        // (|x| + i|y|)^3
        assert_formula_matches_2d(QuatFormula::BurningShipCubic, |x, y| {
            let (ax, ay) = (x.abs(), y.abs());
            let (x2, y2) = (ax * ax - ay * ay, 2.0 * ax * ay);
            (x2 * ax - y2 * ay, x2 * ay + y2 * ax)
        });
    }

    #[test]
    fn formula_name_and_parse_round_trip() {
        for f in QuatFormula::ALL {
            assert_eq!(QuatFormula::parse(f.name()), Some(f));
        }
    }

    #[test]
    fn parse_rejects_unknown_name() {
        assert_eq!(QuatFormula::parse("not-a-formula"), None);
    }

    #[test]
    fn classic_slice_pixel_mapping_matches_view_bounds_shape() {
        // Center pixel of an odd-ish canvas should land near the origin;
        // corners should land near +/- half_y (zoom=1 => half_y=2.0).
        let slice = Slice::classic();
        let (w, h) = (65u32, 65u32);
        let (r0, a0, _b0) = slice.pixel_to_spatial(0, 0, w, h);
        assert!((r0 - (-2.0)).abs() < 1e-9, "r0={r0}");
        assert!((a0 - (-2.0)).abs() < 1e-9, "a0={a0}");
        let (r1, a1, _b1) = slice.pixel_to_spatial(w - 1, h - 1, w, h);
        assert!((r1 - 2.0).abs() < 1e-9, "r1={r1}");
        assert!((a1 - 2.0).abs() < 1e-9, "a1={a1}");
    }

    #[test]
    fn perspective_zero_tilt_stays_rotationally_symmetric_about_origin() {
        // Root-cause check for "why does mandelbrot only draw circles":
        // pulling the eye straight back (tilt=0) must NOT break rotational
        // symmetry about `origin` — four pixels equidistant from screen
        // center along +u/-u/+v/-v must map to spatial points equidistant
        // from `origin`, same as the orthographic case (just rescaled).
        let slice = Slice {
            camera: Some(PerspectiveCamera { tilt_u: 0.0, tilt_v: 0.0, distance: 2.0 }),
            ..Slice::classic()
        };
        let (w, h) = (65u32, 65u32);
        let (cx, cy, r) = (32, 32, 10);
        let pts = [
            slice.pixel_to_spatial(cx + r, cy, w, h),
            slice.pixel_to_spatial(cx - r, cy, w, h),
            slice.pixel_to_spatial(cx, cy + r, w, h),
            slice.pixel_to_spatial(cx, cy - r, w, h),
        ];
        let dist_from_origin = |p: (f64, f64, f64)| {
            let d = (p.0 - slice.origin.0, p.1 - slice.origin.1, p.2 - slice.origin.2);
            (d.0 * d.0 + d.1 * d.1 + d.2 * d.2).sqrt()
        };
        let radii: Vec<f64> = pts.iter().map(|&p| dist_from_origin(p)).collect();
        for r in &radii[1..] {
            assert!((r - radii[0]).abs() < 1e-6, "expected equal radii (still a circle), got {radii:?}");
        }
    }

    #[test]
    fn perspective_nonzero_tilt_breaks_rotational_symmetry_into_an_ellipse_shape() {
        // The actual fix: a lateral tilt (eye offset within the plane's own
        // u/v, not just pulled back along the normal) tilts the viewing
        // axis away from the plane's normal, so the same four points no
        // longer land at equal distances from `origin` — the circle
        // foreshortens into an ellipse.
        let slice = Slice {
            camera: Some(PerspectiveCamera { tilt_u: 1.5, tilt_v: 0.0, distance: 2.0 }),
            ..Slice::classic()
        };
        let (w, h) = (65u32, 65u32);
        let (cx, cy, r) = (32, 32, 10);
        let pts = [
            slice.pixel_to_spatial(cx + r, cy, w, h),
            slice.pixel_to_spatial(cx - r, cy, w, h),
            slice.pixel_to_spatial(cx, cy + r, w, h),
            slice.pixel_to_spatial(cx, cy - r, w, h),
        ];
        let dist_from_origin = |p: (f64, f64, f64)| {
            let d = (p.0 - slice.origin.0, p.1 - slice.origin.1, p.2 - slice.origin.2);
            (d.0 * d.0 + d.1 * d.1 + d.2 * d.2).sqrt()
        };
        let radii: Vec<f64> = pts.iter().map(|&p| dist_from_origin(p)).collect();
        let (max, min) = (radii.iter().cloned().fold(f64::MIN, f64::max), radii.iter().cloned().fold(f64::MAX, f64::min));
        assert!(max - min > 1e-3, "expected tilt to break circular symmetry, radii={radii:?}");
    }

    #[test]
    fn perspective_degenerate_eye_falls_back_to_orthographic_without_panicking() {
        // distance=0, tilt=0 => eye coincides with origin — must not panic
        // or produce NaN, just fall back to the plain orthographic point.
        let slice = Slice {
            camera: Some(PerspectiveCamera { tilt_u: 0.0, tilt_v: 0.0, distance: 0.0 }),
            ..Slice::classic()
        };
        let ortho = Slice::classic();
        let (w, h) = (33u32, 33u32);
        for (px, py) in [(0, 0), (16, 16), (32, 32)] {
            let p = slice.pixel_to_spatial(px, py, w, h);
            let o = ortho.pixel_to_spatial(px, py, w, h);
            assert!(p.0.is_finite() && p.1.is_finite() && p.2.is_finite(), "p={p:?}");
            assert!((p.0 - o.0).abs() < 1e-9 && (p.1 - o.1).abs() < 1e-9 && (p.2 - o.2).abs() < 1e-9, "p={p:?} o={o:?}");
        }
    }

    #[test]
    fn interior_point_never_escapes_within_max_iter() {
        // q_const = 0 stays at q=0 forever, for every formula (step(0)=0
        // for all of them: mul/conj/abs of zero is zero).
        for f in QuatFormula::ALL {
            let et = quat_escape(f, Quat::ZERO, 64, 16.0);
            assert_eq!(et, 64.0, "formula {} did not stay interior", f.name());
        }
    }

    #[test]
    fn pow_fast_matches_powf_for_every_exponent_this_file_actually_uses() {
        // exp = power()-1.0 for each formula: 1,2,3,7. Also check exp=0
        // (identity) and a couple of bases spanning the ranges rho takes
        // in practice (near-zero, ~1, and a few units past bailout).
        for exp in [0.0, 1.0, 2.0, 3.0, 7.0] {
            for base in [0.0, 0.3, 1.0, 1.7, 4.0] {
                let fast = pow_fast(base, exp);
                let slow = base.powf(exp);
                assert!((fast - slow).abs() < 1e-9 * slow.abs().max(1.0), "base={base} exp={exp}: fast={fast} slow={slow}");
            }
        }
    }

    #[test]
    fn pow_fast_falls_back_to_powf_for_a_non_integer_exponent() {
        let base = 2.0;
        let exp = 2.5;
        assert!((pow_fast(base, exp) - base.powf(exp)).abs() < 1e-9);
    }

    #[test]
    fn quat_escape_de_reports_the_same_escape_time_as_quat_escape() {
        // Adding DE tracking as a side computation must not change the
        // escape-time result at all — same loop, same bailout condition.
        for f in QuatFormula::ALL {
            for q in [
                Quat::new(0.3, 0.2, -0.1, 0.05),
                Quat::new(-1.2, 0.6, 0.4, -0.3),
                Quat::new(0.0, 0.0, 0.0, 0.0),
                Quat::new(2.0, 2.0, 2.0, 2.0),
            ] {
                let et_plain = quat_escape(f, q, 48, 16.0);
                let (et_de, de) = quat_escape_de(f, q, 48, 16.0);
                assert_eq!(et_plain, et_de, "formula {} q={:?}: escape time diverged when DE tracking was added", f.name(), q);
                assert!(de.is_finite() && de >= 0.0, "formula {} q={:?}: de={de} should be finite and non-negative", f.name(), q);
            }
        }
    }

    #[test]
    fn quat_escape_de_grows_with_distance_for_a_point_far_outside_the_set() {
        // A point escaping almost immediately, far from the set, should
        // get a distance estimate roughly on the same order as its own
        // distance from the origin — a coarse but useful sanity check that
        // the DE formula isn't off by some wild factor.
        for f in [QuatFormula::Bulb, QuatFormula::BurningShip, QuatFormula::PerpendicularMandelbrot] {
            let (et, de) = quat_escape_de(f, Quat::new(10.0, 0.0, 0.0, 0.0), 48, 16.0);
            assert!(et < 3.0, "formula {}: expected near-immediate escape far outside, et={et}", f.name());
            assert!(de > 0.5 && de < 50.0, "formula {}: de={de} outside a sane range for a point at distance 10", f.name());
        }
    }

    #[test]
    fn bulb_matches_real_power_on_the_real_axis() {
        // On a pure real input (no vector part), Bulb should behave like
        // ordinary real exponentiation: r^n for BULB_POWER=8 (even), even
        // when r is negative.
        for r in [2.0_f64, -2.0, 0.5, -0.5] {
            let out = QuatFormula::Bulb.step(Quat::new(r, 0.0, 0.0, 0.0));
            let want = r.powf(BULB_POWER);
            assert!((out.r - want).abs() < 1e-6, "r={r} out.r={} want={want}", out.r);
            assert!(out.a.abs() < 1e-9 && out.b.abs() < 1e-9 && out.c.abs() < 1e-9, "{:?}", out);
        }
    }

    #[test]
    fn bulb_magnitude_scales_as_rho_to_the_n() {
        for q in [
            Quat::new(0.3, 0.2, -0.1, 0.4),
            Quat::new(-0.5, 0.6, 0.3, -0.2),
            Quat::new(1.2, 0.0, 0.0, 0.9),
        ] {
            let rho = q.norm_sq().sqrt();
            let out = QuatFormula::Bulb.step(q);
            let out_mag = out.norm_sq().sqrt();
            let want = rho.powf(BULB_POWER);
            assert!((out_mag - want).abs() < 1e-6 * want.max(1.0), "rho={rho} out_mag={out_mag} want={want}");
        }
    }

    /// After 2 iterations from q0=0 with the given q_const, is the vector
    /// part still exactly parallel to where it started (cross product ~0)?
    fn stays_on_fixed_line(formula: QuatFormula, q_const: Quat) -> bool {
        let q1 = formula.step(Quat::ZERO).add(q_const);
        let v1 = (q1.a, q1.b, q1.c);
        let q2 = formula.step(q1).add(q_const);
        let v2 = (q2.a, q2.b, q2.c);
        let cross = (
            v1.1 * v2.2 - v1.2 * v2.1,
            v1.2 * v2.0 - v1.0 * v2.2,
            v1.0 * v2.1 - v1.1 * v2.0,
        );
        let cross_mag = (cross.0 * cross.0 + cross.1 * cross.1 + cross.2 * cross.2).sqrt();
        cross_mag < 1e-9
    }

    #[test]
    fn degeneracy_classification_is_exactly_the_scale_only_formulas() {
        // The claim behind TimeAxis::R's doc comment, checked directly
        // rather than re-derived by eye (an earlier version of this file's
        // docs got Celtic's classification wrong by reasoning about "abs
        // means it breaks the invariant" without checking WHICH part of
        // the quaternion the abs touches — Celtic only abs's the resulting
        // REAL component, leaving the vector part exactly as plain
        // squaring left it, so it stays on the fixed line same as
        // Mandelbrot/Tricorn/Cubic/Quartic).
        let q_const = Quat::new(0.4, 0.3, -0.2, 0.5);
        let degenerate = [
            QuatFormula::Mandelbrot,
            QuatFormula::Tricorn,
            QuatFormula::Cubic,
            QuatFormula::Quartic,
            QuatFormula::Celtic,
            // Mandelbox at THIS q_const only: every component of q_const
            // (and of the one iterate computed before it) has magnitude
            // <1.0, so box_fold never triggers a per-component reflection
            // and reduces to identity — leaving ball_fold's uniform radial
            // scalar and the final `scale` multiply, both direction-
            // preserving. Unlike its neighbors in this list, this isn't a
            // structural guarantee: it holds only while every iterate stays
            // inside the ±MANDELBOX_FOLD_LIMIT box. A real render's "c"
            // coordinates span well past that (domain_radius=1.6, camera
            // frustum extends further), where box_fold reflects individual
            // components differently and the degeneracy breaks — confirmed
            // by actually rendering it under TimeAxis::R (real structure,
            // not a sphere), not just by this narrow small-magnitude probe.
            QuatFormula::Mandelbox,
        ];
        let structured = [
            QuatFormula::Bulb,
            QuatFormula::BurningShip,
            QuatFormula::BurningShipCubic,
            QuatFormula::PerpendicularBurningShip,
            QuatFormula::PerpendicularMandelbrot,
        ];
        for f in degenerate {
            assert!(stays_on_fixed_line(f, q_const), "{} should stay on a fixed line (spherical under TimeAxis::R) but didn't", f.name());
        }
        for f in structured {
            assert!(!stays_on_fixed_line(f, q_const), "{} should break the fixed-line degeneracy but didn't", f.name());
        }
        // Every formula accounted for — this must partition QuatFormula::ALL exactly.
        assert_eq!(degenerate.len() + structured.len(), QuatFormula::ALL.len());
    }

    #[test]
    fn time_axis_name_and_parse_round_trip() {
        for axis in [TimeAxis::R, TimeAxis::A, TimeAxis::B, TimeAxis::C] {
            assert_eq!(TimeAxis::parse(axis.name()), Some(axis));
        }
    }

    #[test]
    fn assemble_places_time_in_the_right_slot() {
        let spatial = (1.0, 2.0, 3.0);
        assert_eq!(TimeAxis::R.assemble(spatial, 9.0), Quat::new(9.0, 1.0, 2.0, 3.0));
        assert_eq!(TimeAxis::A.assemble(spatial, 9.0), Quat::new(1.0, 9.0, 2.0, 3.0));
        assert_eq!(TimeAxis::B.assemble(spatial, 9.0), Quat::new(1.0, 2.0, 9.0, 3.0));
        assert_eq!(TimeAxis::C.assemble(spatial, 9.0), Quat::new(1.0, 2.0, 3.0, 9.0));
    }

    #[test]
    fn mandelbrot_under_time_axis_r_is_spherically_symmetric_in_the_spatial_part() {
        // With R fixed (time), Mandelbrot's orbit is provably confined to
        // the line through (A,B,C) (the module-doc "solid of revolution"
        // argument, now applied with the SPATIAL subspace being the full
        // vector part) — so escape time can only depend on rho=|(A,B,C)|,
        // not on direction. Any two spatial points at the same rho must
        // give identical escape times, regardless of R.
        let r_time = 0.3;
        let rho = 0.6;
        let points = [
            (rho, 0.0, 0.0),
            (0.0, rho, 0.0),
            (0.0, 0.0, rho),
            (rho / 3f64.sqrt(), rho / 3f64.sqrt(), rho / 3f64.sqrt()),
            (-rho, 0.0, 0.0),
        ];
        let (max_iter, bailout_sq) = (64u32, 16.0);
        let first = quat_escape(QuatFormula::Mandelbrot, TimeAxis::R.assemble(points[0], r_time), max_iter, bailout_sq);
        for &p in &points[1..] {
            let et = quat_escape(QuatFormula::Mandelbrot, TimeAxis::R.assemble(p, r_time), max_iter, bailout_sq);
            assert!((et - first).abs() < 1e-4, "point {p:?}: et={et} vs first={first} (should match, same rho)");
        }
    }

    #[test]
    fn bulb_under_time_axis_r_is_not_spherically_symmetric() {
        // Bulb breaks the "vector part stays on one line" invariant (see
        // the earlier degeneracy tests), so this must NOT hold for it —
        // different directions at the same rho should give genuinely
        // different escape times. This is the concrete claim behind
        // "swapping time axes still shows real structure for bulb but
        // turns the pure-power formulas into plain spheres."
        let r_time = 0.3;
        let rho = 1.0; // near the boundary, where direction genuinely matters (checked empirically)
        let a = quat_escape(QuatFormula::Bulb, TimeAxis::R.assemble((rho, 0.0, 0.0), r_time), 64, 16.0);
        let b = quat_escape(QuatFormula::Bulb, TimeAxis::R.assemble((0.0, rho, 0.0), r_time), 64, 16.0);
        let c = quat_escape(QuatFormula::Bulb, TimeAxis::R.assemble((0.0, 0.0, rho), r_time), 64, 16.0);
        assert!(
            (a - b).abs() > 1e-4 || (b - c).abs() > 1e-4 || (a - c).abs() > 1e-4,
            "expected direction-dependent escape times for bulb, got a={a} b={b} c={c}"
        );
    }
}

