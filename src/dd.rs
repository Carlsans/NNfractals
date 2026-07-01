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

// ── Formula evaluation in double-double ──────────────────────────────────────

/// Evaluate one of the 58 legacy basis functions in double-double.
/// Polynomial/rational bases are fully precise; transcendentals fall back to
/// f64 (their weights are only f32-precise anyway).
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

        // ── C–G: Transcendental — fall back to f64 (weights are f32 anyway) ─
        i => {
            use crate::formula::f64_impl::eval_basis;
            let (rx, ry) = eval_basis(i, zx.hi, zy.hi, cx.hi, cy.hi);
            (Dd::from_f64(rx), Dd::from_f64(ry))
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

            // ── Transcendentals — fall back to f64 hi parts ─────────────────
            SIN  => { let(x,y)=(ax.hi,ay.hi);
                      (Dd::from_f64(x.sin()*y.cosh()), Dd::from_f64(x.cos()*y.sinh())) },
            COS  => { let(x,y)=(ax.hi,ay.hi);
                      (Dd::from_f64(x.cos()*y.cosh()), Dd::from_f64(-x.sin()*y.sinh())) },
            EXP  => { let x=ax.hi.clamp(-8.0,8.0); let e=x.exp(); let y=ay.hi;
                      (Dd::from_f64(e*y.cos()), Dd::from_f64(e*y.sin())) },
            LOG  => { let(x,y)=(ax.hi,ay.hi);
                      let d=(x*x+y*y+1e-15).sqrt();
                      (Dd::from_f64(d.ln()), Dd::from_f64(y.atan2(x))) },
            TANH => { let(x2,y2)=(2.0*ax.hi,2.0*ay.hi);
                      let d=x2.cosh()+y2.cos()+1e-15;
                      (Dd::from_f64(x2.sinh()/d), Dd::from_f64(y2.sin()/d)) },
            _    => zero,
        };
    }
    reg[n - 1]
}
