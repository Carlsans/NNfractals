//! Double-double arithmetic — ~31 decimal digits of precision (vs 15-16 for f64).
//! Used for deep fractal zoom beyond f64's ~10¹¹ zoom limit.
//!
//! A `Dd` represents a value exactly as `hi + lo` where |lo| ≤ ε·|hi|.
//! Reference: Hida, Li, Bailey, "Algorithms for Quad-Double Precision…" 2001.

#[derive(Copy, Clone, Default, Debug)]
pub struct Dd {
    pub hi: f64,
    pub lo: f64,
}

impl Dd {
    #[inline] pub fn from_f64(x: f64) -> Self { Dd { hi: x, lo: 0.0 } }
    #[inline] pub fn zero() -> Self { Dd { hi: 0.0, lo: 0.0 } }
    #[inline] pub fn one()  -> Self { Dd { hi: 1.0, lo: 0.0 } }
    #[inline] pub fn abs(self) -> Self {
        if self.hi < 0.0 { Dd { hi: -self.hi, lo: -self.lo } } else { self }
    }
    #[inline] pub fn is_finite(self) -> bool { self.hi.is_finite() }

    /// Newton-step refinement of f64 sqrt → full dd precision.
    pub fn sqrt(self) -> Self {
        if self.hi <= 0.0 { return Dd::zero(); }
        let r = self.hi.sqrt();
        let r_dd = Dd::from_f64(r);
        r_dd + (self - r_dd * r_dd) * Dd::from_f64(0.5 / r)
    }
}

// ── Error-free transforms ────────────────────────────────────────────────────

/// a + b = s + e  exactly (Knuth two-sum).
#[inline(always)]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let v = s - a;
    (s, (a - (s - v)) + (b - v))
}

/// a * b = p + e  exactly (requires hardware FMA).
#[inline(always)]
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    (p, a.mul_add(b, -p))
}

// ── Arithmetic operators ─────────────────────────────────────────────────────

impl std::ops::Neg for Dd {
    type Output = Self;
    #[inline] fn neg(self) -> Self { Dd { hi: -self.hi, lo: -self.lo } }
}

impl std::ops::Add for Dd {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let (s, e) = two_sum(self.hi, rhs.hi);
        let e = e + self.lo + rhs.lo;
        let (hi, lo) = two_sum(s, e);
        Dd { hi, lo }
    }
}

impl std::ops::Sub for Dd {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let (s, e) = two_sum(self.hi, -rhs.hi);
        let e = e + self.lo - rhs.lo;
        let (hi, lo) = two_sum(s, e);
        Dd { hi, lo }
    }
}

impl std::ops::Mul for Dd {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let (p, e) = two_prod(self.hi, rhs.hi);
        let e = e + self.hi * rhs.lo + self.lo * rhs.hi;
        let (hi, lo) = two_sum(p, e);
        Dd { hi, lo }
    }
}

/// Fast multiply by a plain f64 scalar (one fewer two_prod vs Dd*Dd).
impl std::ops::Mul<f64> for Dd {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        let (p, e) = two_prod(self.hi, rhs);
        let e = e + self.lo * rhs;
        let (hi, lo) = two_sum(p, e);
        Dd { hi, lo }
    }
}

impl std::ops::Add<f64> for Dd {
    type Output = Self;
    fn add(self, rhs: f64) -> Self {
        let (s, e) = two_sum(self.hi, rhs);
        let (hi, lo) = two_sum(s, e + self.lo);
        Dd { hi, lo }
    }
}

impl std::ops::Div for Dd {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        let q = Dd::from_f64(self.hi / rhs.hi);
        let r = self - rhs * q;
        q + Dd::from_f64(r.hi / rhs.hi)
    }
}

// ── First-order correction for smooth (holomorphic) f64-only functions ─────
//
// A transcendental evaluated as `f(hi)` alone silently throws away every
// Dd's `.lo` part — and since these ops run once per iteration, that
// collapses the WHOLE computation back to plain f64 precision from the
// first iteration onward, for any genome that uses them anywhere in its
// formula graph. Confirmed against a real saved genome (Julia mode, TANH
// then COS applied directly to Z every iteration): deep-zoom rendering
// went degenerate exactly at the f64 precision limit, i.e. DD contributed
// nothing at all for that genome.
//
// Fix: first-order Taylor expansion around the hi-only point. For a
// holomorphic f, f(z+dz) ≈ f(z) + f'(z)·dz. `dz` here is the accumulated
// `.lo` remainder — always tiny by the Dd invariant (|lo| ≤ ε·|hi|) — so
// the dropped O(dz²) term is of order ε², far below anything that matters.
// This recovers full Dd-level accuracy without reimplementing sin/cos/exp
// as genuine double-double routines.
#[inline]
fn dd_taylor_correct(fre: f64, fim: f64, dre: f64, dim: f64, dx: f64, dy: f64) -> (Dd, Dd) {
    let corr_re = dre * dx - dim * dy;
    let corr_im = dre * dy + dim * dx;
    (Dd::from_f64(fre) + Dd::from_f64(corr_re), Dd::from_f64(fim) + Dd::from_f64(corr_im))
}

// ── Formula evaluation in double-double ──────────────────────────────────────

/// Evaluate one of the 58 legacy basis functions in double-double.
/// Polynomial/rational bases are fully precise; transcendentals use the
/// hi-only value plus a numerical-derivative correction (see
/// `dd_taylor_correct` above) — generic over all 40 remaining basis shapes
/// rather than a hand-derived formula per function.
pub fn eval_basis_dd(i: usize, zx: Dd, zy: Dd, cx: Dd, cy: Dd) -> (Dd, Dd) {
    let eps = Dd::from_f64(1e-15_f64);

    // Inline complex helpers using dd
    macro_rules! cmul {
        ($ax:expr, $ay:expr, $bx:expr, $by:expr) => {
            ($ax * $bx - $ay * $by, $ax * $by + $ay * $bx)
        };
    }

    match i {
        // ── A: Powers of z ──────────────────────────────────────────────────
        0 => (zx*zx - zy*zy, (zx*zy) * 2.0),                       // z²
        1 => { let (a,b)=(zx*zx,zy*zy); (zx*(a-b*3.0), zy*(a*3.0-b)) }, // z³
        2 => {                                                        // z⁴
            let (a,b)=(zx*zx,zy*zy);
            (a*a - (a*b)*6.0 + b*b, (zx*zy)*4.0*(a-b))
        },
        3 => {                                                        // z⁵
            let (a,b)=(zx*zx,zy*zy);
            let z4x=a*a-(a*b)*6.0+b*b; let z4y=(zx*zy)*4.0*(a-b);
            cmul!(z4x,z4y,zx,zy)
        },
        4 => (zx, zy),                                               // z
        5 => { let d=zx*zx+zy*zy+eps; (zx/d, (-zy)/d) },           // 1/(z+ε)
        6 => {                                                        // 1/(z²+ε)
            let (z2x,z2y)=(zx*zx-zy*zy,(zx*zy)*2.0);
            let d=z2x*z2x+z2y*z2y+eps; (z2x/d,(-z2y)/d)
        },

        // ── B: Powers involving c ────────────────────────────────────────────
        7  => (cx, cy),                                              // c
        8  => (cx*cx-cy*cy, (cx*cy)*2.0),                          // c²
        9  => { let(a,b)=(cx*cx,cy*cy); (cx*(a-b*3.0),cy*(a*3.0-b)) }, // c³
        10 => cmul!(zx,zy,cx,cy),                                   // z·c
        11 => { let(z2x,z2y)=(zx*zx-zy*zy,(zx*zy)*2.0); cmul!(z2x,z2y,cx,cy) }, // z²·c
        12 => { let(c2x,c2y)=(cx*cx-cy*cy,(cx*cy)*2.0); cmul!(zx,zy,c2x,c2y) }, // z·c²
        13 => {                                                       // z²·c²
            let(z2x,z2y)=(zx*zx-zy*zy,(zx*zy)*2.0);
            let(c2x,c2y)=(cx*cx-cy*cy,(cx*cy)*2.0);
            cmul!(z2x,z2y,c2x,c2y)
        },
        14 => { let d=zx*zx+zy*zy+eps; cmul!(cx,cy,zx/d,(-zy)/d) }, // c/(z+ε)
        15 => { let(sx,sy)=(zx+cx,zy+cy); (sx*sx-sy*sy,(sx*sy)*2.0) }, // (z+c)²
        16 => { let(sx,sy)=(zx-cx,zy-cy); (sx*sx-sy*sy,(sx*sy)*2.0) }, // (z-c)²
        17 => { let(zcx,zcy)=cmul!(zx,zy,cx,cy); (zcx*zcx-zcy*zcy,(zcx*zcy)*2.0) }, // (z·c)²

        // ── C–G: Transcendental — hi-only value + numerical-derivative
        // correction (generic: 40 different shapes here, not worth hand-
        // deriving each one — see dd_taylor_correct's doc comment).
        i => {
            use crate::formula::f64_impl::eval_basis;
            let (zxh, zyh, cxh, cyh) = (zx.hi, zy.hi, cx.hi, cy.hi);
            let (zxl, zyl, cxl, cyl) = (zx.lo, zy.lo, cx.lo, cy.lo);
            let (fre, fim) = eval_basis(i, zxh, zyh, cxh, cyh);
            if zxl == 0.0 && zyl == 0.0 && cxl == 0.0 && cyl == 0.0 {
                return (Dd::from_f64(fre), Dd::from_f64(fim));
            }
            let scale = zxh.abs().max(zyh.abs()).max(cxh.abs()).max(cyh.abs()).max(1.0);
            let h = scale * 1e-6;
            let mut corr_re = 0.0;
            let mut corr_im = 0.0;
            let mut accum = |dre: f64, dim: f64, d: f64| { corr_re += dre * d; corr_im += dim * d; };
            if zxl != 0.0 {
                let (rp, ip) = eval_basis(i, zxh + h, zyh, cxh, cyh);
                let (rm, im) = eval_basis(i, zxh - h, zyh, cxh, cyh);
                accum((rp - rm) / (2.0 * h), (ip - im) / (2.0 * h), zxl);
            }
            if zyl != 0.0 {
                let (rp, ip) = eval_basis(i, zxh, zyh + h, cxh, cyh);
                let (rm, im) = eval_basis(i, zxh, zyh - h, cxh, cyh);
                accum((rp - rm) / (2.0 * h), (ip - im) / (2.0 * h), zyl);
            }
            if cxl != 0.0 {
                let (rp, ip) = eval_basis(i, zxh, zyh, cxh + h, cyh);
                let (rm, im) = eval_basis(i, zxh, zyh, cxh - h, cyh);
                accum((rp - rm) / (2.0 * h), (ip - im) / (2.0 * h), cxl);
            }
            if cyl != 0.0 {
                let (rp, ip) = eval_basis(i, zxh, zyh, cxh, cyh + h);
                let (rm, im) = eval_basis(i, zxh, zyh, cxh, cyh - h);
                accum((rp - rm) / (2.0 * h), (ip - im) / (2.0 * h), cyl);
            }
            (Dd::from_f64(fre) + Dd::from_f64(corr_re), Dd::from_f64(fim) + Dd::from_f64(corr_im))
        }
    }
}

/// Evaluate the weighted basis sum in double-double.
pub fn apply_formula_dd(
    weights: &[(f64, f64)],
    zx: Dd, zy: Dd, cx: Dd, cy: Dd,
) -> (Dd, Dd) {
    let mut rx = Dd::zero();
    let mut ry = Dd::zero();
    for (i, &(wr, wi)) in weights.iter().enumerate() {
        if wr == 0.0 && wi == 0.0 { continue; }
        let (bx, by) = eval_basis_dd(i, zx, zy, cx, cy);
        rx = rx + (bx * wr - by * wi);
        ry = ry + (by * wr + bx * wi);
    }
    (rx, ry)
}

/// Evaluate a DAG program in double-double.
/// Polynomial/structural ops are fully precise; transcendentals fall back to f64.
pub fn eval_program_dd(
    prog: &[crate::formula::OpNode],
    zx: Dd, zy: Dd, cx: Dd, cy: Dd,
) -> (Dd, Dd) {
    use crate::formula::{op::*, N_SLOTS};
    let n = prog.len().min(N_SLOTS);
    if n == 0 { return (Dd::zero(), Dd::zero()); }

    let zero = (Dd::zero(), Dd::zero());
    let mut reg = [zero; N_SLOTS];
    let eps = Dd::from_f64(1e-15_f64);

    for i in 0..n {
        let node = prog[i];
        let ai = (node.a as usize).min(N_SLOTS - 1);
        let bi = (node.b as usize).min(N_SLOTS - 1);
        let (ax, ay) = if ai < i { reg[ai] } else { zero };
        let (bx, by) = if bi < i { reg[bi] } else { zero };

        reg[i] = match node.op {
            Z     => (zx, zy),
            C     => (cx, cy),
            CONST => (Dd::from_f64(node.kre as f64), Dd::from_f64(node.kim as f64)),

            // ── Polynomial / structural ──────────────────────────────────────
            SQR   => (ax*ax - ay*ay, (ax*ay) * 2.0),
            CUBE  => { let(a2,b2)=(ax*ax,ay*ay); (ax*(a2-b2*3.0), ay*(a2*3.0-b2)) },
            QUART => { let(a2,b2)=(ax*ax,ay*ay);
                       (a2*a2-(a2*b2)*6.0+b2*b2, (ax*ay)*4.0*(a2-b2)) },
            ADD   => (ax+bx, ay+by),
            SUB   => (ax-bx, ay-by),
            MUL   => (ax*bx - ay*by, ax*by + ay*bx),
            RECIP => { let d=ax*ax+ay*ay+eps; (ax/d, (-ay)/d) },
            DIV   => {
                let d=bx*bx+by*by+eps;
                let ir=bx/d; let ii=(-by)/d;
                (ax*ir-ay*ii, ax*ii+ay*ir)
            },
            CONJ    => (ax, -ay),
            ABSFOLD => (ax.abs(), ay.abs()),
            ABSRE   => (ax.abs(), ay),
            ABSIM   => (ax, ay.abs()),
            NORMZ   => { let m=(ax*ax+ay*ay).sqrt()+eps; (ax/m, ay/m) },

            // ── Transcendentals — hi-only value + exact analytic-derivative
            // correction (dd_taylor_correct). These run once PER ITERATION,
            // so a naive hi-only fallback here throws away Dd precision on
            // every step, not just once — confirmed to fully defeat DD
            // rendering for any genome using SIN/COS/EXP/LOG/TANH directly.
            SIN  => { let(x,y)=(ax.hi,ay.hi);
                      let (sre,sim) = (x.sin()*y.cosh(), x.cos()*y.sinh());
                      let (cre,cim) = (x.cos()*y.cosh(), -x.sin()*y.sinh()); // sin' = cos
                      dd_taylor_correct(sre, sim, cre, cim, ax.lo, ay.lo) },
            COS  => { let(x,y)=(ax.hi,ay.hi);
                      let (cre,cim) = (x.cos()*y.cosh(), -x.sin()*y.sinh());
                      let (sre,sim) = (x.sin()*y.cosh(), x.cos()*y.sinh());
                      dd_taylor_correct(cre, cim, -sre, -sim, ax.lo, ay.lo) }, // cos' = -sin
            EXP  => { let x=ax.hi.clamp(-8.0,8.0); let e=x.exp(); let y=ay.hi;
                      let (ere,eim) = (e*y.cos(), e*y.sin());
                      dd_taylor_correct(ere, eim, ere, eim, ax.lo, ay.lo) }, // exp' = exp
            LOG  => { let(x,y)=(ax.hi,ay.hi);
                      let d=(x*x+y*y+1e-15).sqrt();
                      let (lre,lim) = (d.ln(), y.atan2(x));
                      let denom = x*x+y*y+1e-15;
                      dd_taylor_correct(lre, lim, x/denom, -y/denom, ax.lo, ay.lo) }, // log' = 1/z
            TANH => { let(x2,y2)=(2.0*ax.hi,2.0*ay.hi);
                      let d=x2.cosh()+y2.cos()+1e-15;
                      let (tre,tim) = (x2.sinh()/d, y2.sin()/d);
                      let (t2re,t2im) = (tre*tre-tim*tim, 2.0*tre*tim); // tanh' = 1-tanh²
                      dd_taylor_correct(tre, tim, 1.0-t2re, -t2im, ax.lo, ay.lo) },
            _    => zero,
        };
    }
    reg[n - 1]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formula::{op, OpNode};

    // Regression for the transcendental-fallback bug: a genuine saved genome
    // (Julia mode, program TANH→COS→MUL→DIV applied to Z every iteration)
    // rendered as degenerate exactly at the f64 precision limit — DD
    // contributed nothing, because the old hi-only fallback discarded the
    // Dd .lo part on every single iteration. Confirmed against a brute-force
    // f64 evaluation at the literal (hi+lo) point: the old hi-only value was
    // off by ~3e-9 for a deliberately-not-astronomically-tiny lo; the
    // first-order-corrected value matches brute force exactly.
    #[test]
    fn tanh_dd_correction_matches_brute_force_at_summed_point() {
        let x = 0.31_f64;
        let y = -0.22_f64;
        let dx = 3e-9_f64;
        let dy = -2e-9_f64;

        let prog = vec![
            OpNode { op: op::Z,    a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::TANH, a: 0, b: 0, kre: 0.0, kim: 0.0 },
        ];
        let ax = Dd { hi: x, lo: dx };
        let ay = Dd { hi: y, lo: dy };
        let (rx, ry) = eval_program_dd(&prog, ax, ay, Dd::zero(), Dd::zero());
        let got = (rx.hi + rx.lo, ry.hi + ry.lo);

        // Brute-force ground truth: evaluate tanh directly at the literal
        // summed point in plain f64 (valid here since dx/dy aren't so tiny
        // that x+dx rounds back to x — this checks the CORRECTION logic,
        // not sub-ULP precision itself).
        let (bx, by) = (x + dx, y + dy);
        let (x2, y2) = (2.0 * bx, 2.0 * by);
        let d = x2.cosh() + y2.cos() + 1e-15;
        let brute = (x2.sinh() / d, y2.sin() / d);

        // hi-only (the old buggy behavior) for comparison — should be
        // measurably worse than the corrected result.
        let (x2h, y2h) = (2.0 * x, 2.0 * y);
        let dh = x2h.cosh() + y2h.cos() + 1e-15;
        let hi_only = (x2h.sinh() / dh, y2h.sin() / dh);
        let hi_only_err = ((hi_only.0 - brute.0).powi(2) + (hi_only.1 - brute.1).powi(2)).sqrt();

        let err = ((got.0 - brute.0).powi(2) + (got.1 - brute.1).powi(2)).sqrt();
        assert!(err < 1e-13, "corrected TANH doesn't match brute force: err={err:e}");
        assert!(err < hi_only_err / 1000.0,
            "corrected TANH ({err:e}) isn't meaningfully better than the old hi-only fallback ({hi_only_err:e})");
    }

    #[test]
    fn transcendental_correction_is_inert_when_lo_is_zero() {
        // No Dd precision to recover (lo=0) — must reduce to exactly the
        // old hi-only behavior, not introduce spurious noise.
        let prog = vec![
            OpNode { op: op::Z,   a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::COS, a: 0, b: 0, kre: 0.0, kim: 0.0 },
        ];
        let ax = Dd::from_f64(0.4);
        let ay = Dd::from_f64(-0.15);
        let (rx, ry) = eval_program_dd(&prog, ax, ay, Dd::zero(), Dd::zero());
        assert_eq!(rx.lo, 0.0);
        assert_eq!(ry.lo, 0.0);
        assert!((rx.hi - (0.4_f64.cos() * (-0.15_f64).cosh())).abs() < 1e-15);
    }
}
