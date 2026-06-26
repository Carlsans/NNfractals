pub const N_BASIS: usize = 58;

// The formula evaluator is generated for both f32 (fast, GPU-matched, used by the
// GA and normal rendering) and f64 (deep-zoom precision, used by the viewer once
// the per-pixel coordinate step underflows f32). A single macro keeps the two
// precisions byte-for-byte identical in logic.
macro_rules! define_precision {
    ($modname:ident, $ty:ident) => {
        pub mod $modname {
            use std::$ty::consts::PI;

            const EPS: $ty = 1e-6;

            // ── Scalar helpers ──────────────────────────────────────────────
            #[inline] fn cmul(ar: $ty, ai: $ty, br: $ty, bi: $ty) -> ($ty, $ty) {
                (ar * br - ai * bi, ar * bi + ai * br)
            }
            #[inline] fn csin(x: $ty, y: $ty) -> ($ty, $ty) {
                (x.sin() * y.cosh(), x.cos() * y.sinh())
            }
            #[inline] fn ccos(x: $ty, y: $ty) -> ($ty, $ty) {
                (x.cos() * y.cosh(), -x.sin() * y.sinh())
            }
            #[inline] fn cexp(x: $ty, y: $ty) -> ($ty, $ty) {
                let e = x.clamp(-8.0, 8.0).exp();
                (e * y.cos(), e * y.sin())
            }
            #[inline] fn clog(x: $ty, y: $ty) -> ($ty, $ty) {
                ((x * x + y * y + EPS).sqrt().ln(), y.atan2(x))
            }

            /// Evaluate basis function i at (z, c) → complex result.
            pub fn eval_basis(i: usize, zx: $ty, zy: $ty, cx: $ty, cy: $ty) -> ($ty, $ty) {
                match i {
                    // ── A: Powers of z ──────────────────────────────────────
                    0  => (zx*zx - zy*zy, 2.0*zx*zy),                       // z²
                    1  => {                                                    // z³
                        let (a, b) = (zx*zx, zy*zy);
                        (zx*(a - 3.0*b), zy*(3.0*a - b))
                    },
                    2  => {                                                    // z⁴
                        let (a, b) = (zx*zx, zy*zy);
                        (a*a - 6.0*a*b + b*b, 4.0*zx*zy*(a - b))
                    },
                    3  => {                                                    // z⁵ = z⁴·z
                        let (a, b) = (zx*zx, zy*zy);
                        let (z4x, z4y) = (a*a - 6.0*a*b + b*b, 4.0*zx*zy*(a - b));
                        cmul(z4x, z4y, zx, zy)
                    },
                    4  => (zx, zy),                                           // z  (identity)
                    5  => {                                                    // 1/(z+ε)
                        let d = zx*zx + zy*zy + EPS;
                        (zx/d, -zy/d)
                    },
                    6  => {                                                    // 1/(z²+ε)
                        let (z2x, z2y) = (zx*zx - zy*zy, 2.0*zx*zy);
                        let d = z2x*z2x + z2y*z2y + EPS;
                        (z2x/d, -z2y/d)
                    },

                    // ── B: Powers involving c ───────────────────────────────
                    7  => (cx, cy),                                           // c
                    8  => (cx*cx - cy*cy, 2.0*cx*cy),                        // c²
                    9  => {                                                    // c³
                        let (a, b) = (cx*cx, cy*cy);
                        (cx*(a - 3.0*b), cy*(3.0*a - b))
                    },
                    10 => cmul(zx, zy, cx, cy),                              // z·c
                    11 => {                                                    // z²·c
                        let (z2x, z2y) = (zx*zx - zy*zy, 2.0*zx*zy);
                        cmul(z2x, z2y, cx, cy)
                    },
                    12 => {                                                    // z·c²
                        let (c2x, c2y) = (cx*cx - cy*cy, 2.0*cx*cy);
                        cmul(zx, zy, c2x, c2y)
                    },
                    13 => {                                                    // z²·c²
                        let (z2x, z2y) = (zx*zx - zy*zy, 2.0*zx*zy);
                        let (c2x, c2y) = (cx*cx - cy*cy, 2.0*cx*cy);
                        cmul(z2x, z2y, c2x, c2y)
                    },
                    14 => {                                                    // c/(z+ε)
                        let d = zx*zx + zy*zy + EPS;
                        cmul(cx, cy, zx/d, -zy/d)
                    },
                    15 => {                                                    // (z+c)²
                        let (sx, sy) = (zx+cx, zy+cy);
                        (sx*sx - sy*sy, 2.0*sx*sy)
                    },
                    16 => {                                                    // (z−c)²
                        let (sx, sy) = (zx-cx, zy-cy);
                        (sx*sx - sy*sy, 2.0*sx*sy)
                    },
                    17 => {                                                    // (z·c)²
                        let (zcx, zcy) = cmul(zx, zy, cx, cy);
                        (zcx*zcx - zcy*zcy, 2.0*zcx*zcy)
                    },

                    // ── C: Trigonometric (complex entire functions) ─────────
                    18 => csin(zx, zy),                                       // sin(z)
                    19 => ccos(zx, zy),                                       // cos(z)
                    20 => csin(zx * PI, zy * PI),                             // sin(πz)
                    21 => ccos(zx * PI, zy * PI),                             // cos(πz)
                    22 => { let (z2x,z2y)=(zx*zx-zy*zy,2.0*zx*zy); csin(z2x,z2y) }, // sin(z²)
                    23 => { let (z2x,z2y)=(zx*zx-zy*zy,2.0*zx*zy); ccos(z2x,z2y) }, // cos(z²)
                    24 => csin(zx+cx, zy+cy),                                 // sin(z+c)
                    25 => ccos(zx+cx, zy+cy),                                 // cos(z+c)
                    26 => { let (zcx,zcy)=cmul(zx,zy,cx,cy); csin(zcx,zcy) },// sin(z·c)
                    27 => { let (zcx,zcy)=cmul(zx,zy,cx,cy); ccos(zcx,zcy) },// cos(z·c)
                    28 => { let (sx,sy)=csin(zx,zy); cmul(zx,zy,sx,sy) },    // z·sin(z)
                    29 => { let (cx2,cy2)=ccos(zx,zy); cmul(zx,zy,cx2,cy2) },// z·cos(z)
                    30 => {                                                    // tan(z)
                        let (sx,sy) = csin(zx,zy);
                        let (cxv,cyv) = ccos(zx,zy);
                        let d = cxv*cxv + cyv*cyv + EPS;
                        ((sx*cxv + sy*cyv)/d, (sy*cxv - sx*cyv)/d)
                    },
                    31 => {                                                    // sinh(z)
                        let (ex_r,ex_i) = cexp(zx,zy);
                        let (enx_r,enx_i) = cexp(-zx,-zy);
                        ((ex_r-enx_r)*0.5, (ex_i-enx_i)*0.5)
                    },
                    32 => {                                                    // cosh(z)
                        let (ex_r,ex_i) = cexp(zx,zy);
                        let (enx_r,enx_i) = cexp(-zx,-zy);
                        ((ex_r+enx_r)*0.5, (ex_i+enx_i)*0.5)
                    },
                    33 => {                                                    // tanh(z)
                        let (x2,y2) = (2.0*zx, 2.0*zy);
                        let d = x2.cosh() + y2.cos() + EPS;
                        (x2.sinh()/d, y2.sin()/d)
                    },

                    // ── D: Exponential and logarithmic ──────────────────────
                    34 => cexp(zx, zy),                                       // exp(z)
                    35 => cexp(-zx, -zy),                                     // exp(−z)
                    36 => { let (zcx,zcy)=cmul(zx.clamp(-4.0,4.0),zy,cx,cy); cexp(zcx,zcy) }, // exp(z·c)
                    37 => { let (ex_r,ex_i)=cexp(zx.clamp(-4.0,4.0),zy); cmul(zx,zy,ex_r,ex_i) }, // z·exp(z)
                    38 => { let (ex_r,ex_i)=cexp(zx.clamp(-4.0,4.0),zy); cmul(ex_r,ex_i,cx,cy) }, // exp(z)·c
                    39 => clog(zx+1.0, zy),                                   // log(z+1)
                    40 => { let (z2x,z2y)=(zx*zx-zy*zy+1.0,2.0*zx*zy); clog(z2x,z2y) }, // log(z²+1)
                    41 => { let (lx,ly)=clog(zx+1.0,zy); cmul(zx,zy,lx,ly) },// z·log(z+1)
                    42 => {                                                    // sin(1/z)
                        let d = zx*zx + zy*zy + EPS;
                        csin((zx/d).clamp(-10.0,10.0), (-zy/d).clamp(-10.0,10.0))
                    },
                    43 => {                                                    // exp(1/z)
                        let d = zx*zx + zy*zy + EPS;
                        cexp((zx/d).clamp(-4.0,4.0), (-zy/d).clamp(-8.0,8.0))
                    },

                    // ── E: Non-holomorphic / Burning Ship family ───────────
                    44 => (zx.abs(), zy),                                     // |Re(z)| + i·Im(z)
                    45 => (zx, zy.abs()),                                     // Re(z) + i·|Im(z)|
                    46 => (zx.abs(), zy.abs()),                               // Burning Ship fold
                    47 => (zx, -zy),                                          // conj(z)
                    48 => (zx*zx - zy*zy, -2.0*zx*zy),                       // conj(z)² (Tricorn)
                    49 => { let m = (zx*zx+zy*zy).sqrt(); (zx*m, zy*m) },    // z·|z|
                    50 => { let m = (zx*zx+zy*zy).sqrt()+EPS; (zx/m, zy/m) },// z/|z| (normalized)

                    // ── F: Rational functions (meromorphic) ─────────────────
                    51 => {                                                    // z/(z²+1)
                        let (z2x,z2y) = (zx*zx-zy*zy+1.0, 2.0*zx*zy);
                        let d = z2x*z2x + z2y*z2y + EPS;
                        ((zx*z2x + zy*z2y)/d, (zy*z2x - zx*z2y)/d)
                    },
                    52 => {                                                    // (z²−1)/(z²+1)
                        let (z2x,z2y) = (zx*zx-zy*zy, 2.0*zx*zy);
                        let (nr,ni) = (z2x-1.0, z2y);
                        let (dr,di) = (z2x+1.0, z2y);
                        let d = dr*dr + di*di + EPS;
                        ((nr*dr + ni*di)/d, (ni*dr - nr*di)/d)
                    },
                    53 => {                                                    // z²/(z−1+ε)
                        let (z2x,z2y) = (zx*zx-zy*zy, 2.0*zx*zy);
                        let (dr,di) = (zx-1.0, zy);
                        let d = dr*dr + di*di + EPS;
                        ((z2x*dr + z2y*di)/d, (z2y*dr - z2x*di)/d)
                    },
                    54 => {                                                    // 1/(z²+c)
                        let (z2x,z2y) = (zx*zx-zy*zy, 2.0*zx*zy);
                        let (dr,di) = (z2x+cx, z2y+cy);
                        let d = dr*dr + di*di + EPS;
                        (dr/d, -di/d)
                    },
                    55 => {                                                    // z²·c/(z+c+ε)
                        let (z2x,z2y) = (zx*zx-zy*zy, 2.0*zx*zy);
                        let (dr,di) = (zx+cx, zy+cy);
                        let d = dr*dr + di*di + EPS;
                        let (irx,iiy) = (dr/d, -di/d);
                        let (z2cx,z2cy) = cmul(z2x,z2y,cx,cy);
                        cmul(z2cx, z2cy, irx, iiy)
                    },

                    // ── G: Constants ───────────────────────────────────────
                    56 => (1.0, 0.0),                                         // 1
                    57 => (0.0, 1.0),                                         // i

                    _  => (0.0, 0.0),
                }
            }

            /// Evaluate weighted sum: z_new = Σᵢ (w_re[i]+i·w_im[i]) * φᵢ(z, c).
            pub fn apply_formula(weights: &[($ty, $ty)], zx: $ty, zy: $ty, cx: $ty, cy: $ty) -> ($ty, $ty) {
                let mut rx = 0.0 as $ty;
                let mut ry = 0.0 as $ty;
                for (i, &(wr, wi)) in weights.iter().enumerate() {
                    if wr == 0.0 && wi == 0.0 { continue; }
                    let (bx, by) = eval_basis(i, zx, zy, cx, cy);
                    rx += wr * bx - wi * by;
                    ry += wr * by + wi * bx;
                }
                (rx, ry)
            }
        }
    };
}

define_precision!(f32_impl, f32);
define_precision!(f64_impl, f64);

// Backward-compatible re-exports: the rest of the codebase (GA, GPU CPU fallback,
// normal rendering) uses the f32 versions exactly as before.
pub use f32_impl::{apply_formula, eval_basis};

/// Human-readable label for a basis function.
pub fn basis_name(i: usize) -> &'static str {
    match i {
        0 =>"z²",     1 =>"z³",       2 =>"z⁴",       3 =>"z⁵",
        4 =>"z",      5 =>"1/z",      6 =>"1/z²",
        7 =>"c",      8 =>"c²",       9 =>"c³",
        10=>"zc",     11=>"z²c",      12=>"zc²",       13=>"z²c²",
        14=>"c/z",    15=>"(z+c)²",   16=>"(z−c)²",    17=>"(zc)²",
        18=>"sin",    19=>"cos",      20=>"sin(π)",     21=>"cos(π)",
        22=>"sin(z²)",23=>"cos(z²)",  24=>"sin(z+c)",   25=>"cos(z+c)",
        26=>"sin(zc)",27=>"cos(zc)",  28=>"z·sin",      29=>"z·cos",
        30=>"tan",    31=>"sinh",     32=>"cosh",        33=>"tanh",
        34=>"exp",    35=>"exp(−z)",  36=>"exp(zc)",    37=>"z·exp",
        38=>"exp·c",  39=>"log(z+1)", 40=>"log(z²+1)", 41=>"z·log",
        42=>"sin(1/z)",43=>"exp(1/z)",
        44=>"|Re|+Im", 45=>"Re+|Im|", 46=>"|BS|",      47=>"conj",
        48=>"conj²",  49=>"z|z|",    50=>"z/|z|",
        51=>"z/(z²+1)",52=>"(z²−1)/(z²+1)",53=>"z²/(z−1)",
        54=>"1/(z²+c)",55=>"z²c/(z+c)",
        56=>"1",      57=>"i",
        _  =>"?",
    }
}
