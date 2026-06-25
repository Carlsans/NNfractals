use burn::tensor::{Tensor, backend::Backend};

pub const N_BASIS: usize = 58;

const EPS: f32 = 1e-6;

// ── Scalar helpers ────────────────────────────────────────────────────────────

#[inline] fn cmul(ar: f32, ai: f32, br: f32, bi: f32) -> (f32, f32) {
    (ar * br - ai * bi, ar * bi + ai * br)
}
#[inline] fn csin(x: f32, y: f32) -> (f32, f32) {
    (x.sin() * y.cosh(), x.cos() * y.sinh())
}
#[inline] fn ccos(x: f32, y: f32) -> (f32, f32) {
    (x.cos() * y.cosh(), -x.sin() * y.sinh())
}
#[inline] fn cexp(x: f32, y: f32) -> (f32, f32) {
    let e = x.clamp(-8.0, 8.0).exp();
    (e * y.cos(), e * y.sin())
}
#[inline] fn clog(x: f32, y: f32) -> (f32, f32) {
    ((x * x + y * y + EPS).sqrt().ln(), y.atan2(x))
}

// ── Scalar eval ───────────────────────────────────────────────────────────────

/// Evaluate basis function i at (z, c) → complex result.
/// All 58 functions are defined here.
pub fn eval_basis(i: usize, zx: f32, zy: f32, cx: f32, cy: f32) -> (f32, f32) {
    match i {
        // ── A: Powers of z ────────────────────────────────────────────────
        0  => (zx*zx - zy*zy, 2.0*zx*zy),                               // z²
        1  => {                                                            // z³
            let (a, b) = (zx*zx, zy*zy);
            (zx*(a - 3.0*b), zy*(3.0*a - b))
        },
        2  => {                                                            // z⁴
            let (a, b) = (zx*zx, zy*zy);
            (a*a - 6.0*a*b + b*b, 4.0*zx*zy*(a - b))
        },
        3  => {                                                            // z⁵ = z⁴·z
            let (a, b) = (zx*zx, zy*zy);
            let (z4x, z4y) = (a*a - 6.0*a*b + b*b, 4.0*zx*zy*(a - b));
            cmul(z4x, z4y, zx, zy)
        },
        4  => (zx, zy),                                                   // z  (identity)
        5  => {                                                            // 1/(z+ε)
            let d = zx*zx + zy*zy + EPS;
            (zx/d, -zy/d)
        },
        6  => {                                                            // 1/(z²+ε)
            let (z2x, z2y) = (zx*zx - zy*zy, 2.0*zx*zy);
            let d = z2x*z2x + z2y*z2y + EPS;
            (z2x/d, -z2y/d)
        },

        // ── B: Powers involving c ─────────────────────────────────────────
        7  => (cx, cy),                                                   // c
        8  => (cx*cx - cy*cy, 2.0*cx*cy),                                // c²
        9  => {                                                            // c³
            let (a, b) = (cx*cx, cy*cy);
            (cx*(a - 3.0*b), cy*(3.0*a - b))
        },
        10 => cmul(zx, zy, cx, cy),                                      // z·c
        11 => {                                                            // z²·c
            let (z2x, z2y) = (zx*zx - zy*zy, 2.0*zx*zy);
            cmul(z2x, z2y, cx, cy)
        },
        12 => {                                                            // z·c²
            let (c2x, c2y) = (cx*cx - cy*cy, 2.0*cx*cy);
            cmul(zx, zy, c2x, c2y)
        },
        13 => {                                                            // z²·c²
            let (z2x, z2y) = (zx*zx - zy*zy, 2.0*zx*zy);
            let (c2x, c2y) = (cx*cx - cy*cy, 2.0*cx*cy);
            cmul(z2x, z2y, c2x, c2y)
        },
        14 => {                                                            // c/(z+ε)
            let d = zx*zx + zy*zy + EPS;
            cmul(cx, cy, zx/d, -zy/d)
        },
        15 => {                                                            // (z+c)²
            let (sx, sy) = (zx+cx, zy+cy);
            (sx*sx - sy*sy, 2.0*sx*sy)
        },
        16 => {                                                            // (z−c)²
            let (sx, sy) = (zx-cx, zy-cy);
            (sx*sx - sy*sy, 2.0*sx*sy)
        },
        17 => {                                                            // (z·c)²
            let (zcx, zcy) = cmul(zx, zy, cx, cy);
            (zcx*zcx - zcy*zcy, 2.0*zcx*zcy)
        },

        // ── C: Trigonometric (complex entire functions) ───────────────────
        18 => csin(zx, zy),                                               // sin(z)
        19 => ccos(zx, zy),                                               // cos(z)
        20 => csin(zx * std::f32::consts::PI, zy * std::f32::consts::PI), // sin(πz)
        21 => ccos(zx * std::f32::consts::PI, zy * std::f32::consts::PI), // cos(πz)
        22 => { let (z2x,z2y)=(zx*zx-zy*zy,2.0*zx*zy); csin(z2x,z2y) }, // sin(z²)
        23 => { let (z2x,z2y)=(zx*zx-zy*zy,2.0*zx*zy); ccos(z2x,z2y) }, // cos(z²)
        24 => csin(zx+cx, zy+cy),                                         // sin(z+c)
        25 => ccos(zx+cx, zy+cy),                                         // cos(z+c)
        26 => { let (zcx,zcy)=cmul(zx,zy,cx,cy); csin(zcx,zcy) },        // sin(z·c)
        27 => { let (zcx,zcy)=cmul(zx,zy,cx,cy); ccos(zcx,zcy) },        // cos(z·c)
        28 => { let (sx,sy)=csin(zx,zy); cmul(zx,zy,sx,sy) },            // z·sin(z)
        29 => { let (cx2,cy2)=ccos(zx,zy); cmul(zx,zy,cx2,cy2) },        // z·cos(z)
        30 => {                                                            // tan(z)
            let (sx,sy) = csin(zx,zy);
            let (cxv,cyv) = ccos(zx,zy);
            let d = cxv*cxv + cyv*cyv + EPS;
            ((sx*cxv + sy*cyv)/d, (sy*cxv - sx*cyv)/d)
        },
        31 => {                                                            // sinh(z)
            let (ex_r,ex_i) = cexp(zx,zy);
            let (enx_r,enx_i) = cexp(-zx,-zy);
            ((ex_r-enx_r)*0.5, (ex_i-enx_i)*0.5)
        },
        32 => {                                                            // cosh(z)
            let (ex_r,ex_i) = cexp(zx,zy);
            let (enx_r,enx_i) = cexp(-zx,-zy);
            ((ex_r+enx_r)*0.5, (ex_i+enx_i)*0.5)
        },
        33 => {                                                            // tanh(z)
            let (x2,y2) = (2.0*zx, 2.0*zy);
            let d = x2.cosh() + y2.cos() + EPS;
            (x2.sinh()/d, y2.sin()/d)
        },

        // ── D: Exponential and logarithmic ────────────────────────────────
        34 => cexp(zx, zy),                                               // exp(z)
        35 => cexp(-zx, -zy),                                             // exp(−z)
        36 => { let (zcx,zcy)=cmul(zx.clamp(-4.0,4.0),zy,cx,cy); cexp(zcx,zcy) }, // exp(z·c)
        37 => { let (ex_r,ex_i)=cexp(zx.clamp(-4.0,4.0),zy); cmul(zx,zy,ex_r,ex_i) }, // z·exp(z)
        38 => { let (ex_r,ex_i)=cexp(zx.clamp(-4.0,4.0),zy); cmul(ex_r,ex_i,cx,cy) }, // exp(z)·c
        39 => clog(zx+1.0, zy),                                           // log(z+1)
        40 => { let (z2x,z2y)=(zx*zx-zy*zy+1.0,2.0*zx*zy); clog(z2x,z2y) }, // log(z²+1)
        41 => { let (lx,ly)=clog(zx+1.0,zy); cmul(zx,zy,lx,ly) },        // z·log(z+1)
        42 => {                                                            // sin(1/z)
            let d = zx*zx + zy*zy + EPS;
            csin((zx/d).clamp(-10.0,10.0), (-zy/d).clamp(-10.0,10.0))
        },
        43 => {                                                            // exp(1/z)
            let d = zx*zx + zy*zy + EPS;
            cexp((zx/d).clamp(-4.0,4.0), (-zy/d).clamp(-8.0,8.0))
        },

        // ── E: Non-holomorphic / Burning Ship family ─────────────────────
        44 => (zx.abs(), zy),                                             // |Re(z)| + i·Im(z)
        45 => (zx, zy.abs()),                                             // Re(z) + i·|Im(z)|
        46 => (zx.abs(), zy.abs()),                                       // Burning Ship fold
        47 => (zx, -zy),                                                  // conj(z)
        48 => (zx*zx - zy*zy, -2.0*zx*zy),                              // conj(z)² (Tricorn)
        49 => { let m = (zx*zx+zy*zy).sqrt(); (zx*m, zy*m) },           // z·|z|
        50 => { let m = (zx*zx+zy*zy).sqrt()+EPS; (zx/m, zy/m) },      // z/|z| (normalized)

        // ── F: Rational functions (meromorphic) ───────────────────────────
        51 => {                                                            // z/(z²+1)
            let (z2x,z2y) = (zx*zx-zy*zy+1.0, 2.0*zx*zy);
            let d = z2x*z2x + z2y*z2y + EPS;
            ((zx*z2x + zy*z2y)/d, (zy*z2x - zx*z2y)/d)
        },
        52 => {                                                            // (z²−1)/(z²+1)
            let (z2x,z2y) = (zx*zx-zy*zy, 2.0*zx*zy);
            let (nr,ni) = (z2x-1.0, z2y);
            let (dr,di) = (z2x+1.0, z2y);
            let d = dr*dr + di*di + EPS;
            ((nr*dr + ni*di)/d, (ni*dr - nr*di)/d)
        },
        53 => {                                                            // z²/(z−1+ε)
            let (z2x,z2y) = (zx*zx-zy*zy, 2.0*zx*zy);
            let (dr,di) = (zx-1.0, zy);
            let d = dr*dr + di*di + EPS;
            ((z2x*dr + z2y*di)/d, (z2y*dr - z2x*di)/d)
        },
        54 => {                                                            // 1/(z²+c)
            let (z2x,z2y) = (zx*zx-zy*zy, 2.0*zx*zy);
            let (dr,di) = (z2x+cx, z2y+cy);
            let d = dr*dr + di*di + EPS;
            (dr/d, -di/d)
        },
        55 => {                                                            // z²·c/(z+c+ε)
            let (z2x,z2y) = (zx*zx-zy*zy, 2.0*zx*zy);
            let (dr,di) = (zx+cx, zy+cy);
            let d = dr*dr + di*di + EPS;
            let (irx,iiy) = (dr/d, -di/d);
            let (z2cx,z2cy) = cmul(z2x,z2y,cx,cy);
            cmul(z2cx, z2cy, irx, iiy)
        },

        // ── G: Constants ─────────────────────────────────────────────────
        56 => (1.0, 0.0),                                                 // 1
        57 => (0.0, 1.0),                                                 // i

        _  => (0.0, 0.0),
    }
}

/// Evaluate weighted sum: z_new = Σᵢ (w_re[i]+i·w_im[i]) * φᵢ(z, c).
/// `weights` is a slice of N_BASIS complex pairs.
pub fn apply_formula(weights: &[(f32, f32)], zx: f32, zy: f32, cx: f32, cy: f32) -> (f32, f32) {
    let mut rx = 0.0f32;
    let mut ry = 0.0f32;
    for (i, &(wr, wi)) in weights.iter().enumerate() {
        let (bx, by) = eval_basis(i, zx, zy, cx, cy);
        rx += wr * bx - wi * by;
        ry += wr * by + wi * bx;
    }
    (rx, ry)
}

// ── Tensor helpers ────────────────────────────────────────────────────────────

fn t_cmul<B: Backend>(
    ar: &Tensor<B, 2>, ai: &Tensor<B, 2>,
    br: &Tensor<B, 2>, bi: &Tensor<B, 2>,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    (ar.clone() * br.clone() - ai.clone() * bi.clone(),
     ar.clone() * bi.clone() + ai.clone() * br.clone())
}

fn t_csin<B: Backend>(x: &Tensor<B, 2>, y: &Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
    (x.clone().sin() * y.clone().cosh(), x.clone().cos() * y.clone().sinh())
}

fn t_ccos<B: Backend>(x: &Tensor<B, 2>, y: &Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
    (x.clone().cos() * y.clone().cosh(), x.clone().sin().neg() * y.clone().sinh())
}

fn t_cexp<B: Backend>(x: &Tensor<B, 2>, y: &Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let e = x.clone().clamp(-8.0, 8.0).exp();
    (e.clone() * y.clone().cos(), e * y.clone().sin())
}

fn t_clog<B: Backend>(x: &Tensor<B, 2>, y: &Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let r = (x.clone() * x.clone() + y.clone() * y.clone() + EPS).sqrt().log();
    let th = y.clone().atan2(x.clone());
    (r, th)
}

/// Evaluate basis function i using tensors (for autodiff backprop).
/// z_re/z_im/c_re/c_im: [N, 1] each. Returns (phi_re, phi_im) [N, 1].
pub fn eval_basis_tensor<B: Backend>(
    i: usize,
    z_re: &Tensor<B, 2>,
    z_im: &Tensor<B, 2>,
    c_re: &Tensor<B, 2>,
    c_im: &Tensor<B, 2>,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let zr = z_re;
    let zi = z_im;
    let cr = c_re;
    let ci = c_im;

    match i {
        // A: Powers of z
        0  => (zr.clone()*zr.clone() - zi.clone()*zi.clone(), zr.clone()*zi.clone()*2.0), // z²
        1  => {                                                                              // z³
            let a = zr.clone()*zr.clone();
            let b = zi.clone()*zi.clone();
            (zr.clone()*(a.clone() - b.clone()*3.0), zi.clone()*(a*3.0 - b))
        },
        2  => {                                                                              // z⁴
            let a = zr.clone()*zr.clone();
            let b = zi.clone()*zi.clone();
            (a.clone()*a.clone() - a.clone()*b.clone()*6.0 + b.clone()*b.clone(),
             zr.clone()*zi.clone()*(a - b)*4.0)
        },
        3  => {                                                                              // z⁵
            let a = zr.clone()*zr.clone();
            let b = zi.clone()*zi.clone();
            let (z4r,z4i) = (a.clone()*a.clone()-a.clone()*b.clone()*6.0+b.clone()*b.clone(),
                             zr.clone()*zi.clone()*(a-b)*4.0);
            t_cmul(&z4r, &z4i, zr, zi)
        },
        4  => (zr.clone(), zi.clone()),                                                     // z
        5  => {                                                                              // 1/(z+ε)
            let d = zr.clone()*zr.clone() + zi.clone()*zi.clone() + EPS;
            (zr.clone()/d.clone(), zi.clone().neg()/d)
        },
        6  => {                                                                              // 1/(z²+ε)
            let (z2r,z2i) = (zr.clone()*zr.clone()-zi.clone()*zi.clone(), zr.clone()*zi.clone()*2.0);
            let d = z2r.clone()*z2r.clone() + z2i.clone()*z2i.clone() + EPS;
            (z2r/d.clone(), z2i.neg()/d)
        },

        // B: Powers involving c
        7  => (cr.clone(), ci.clone()),                                                     // c
        8  => (cr.clone()*cr.clone()-ci.clone()*ci.clone(), cr.clone()*ci.clone()*2.0),    // c²
        9  => {                                                                              // c³
            let a = cr.clone()*cr.clone();
            let b = ci.clone()*ci.clone();
            (cr.clone()*(a.clone()-b.clone()*3.0), ci.clone()*(a*3.0-b))
        },
        10 => t_cmul(zr, zi, cr, ci),                                                       // z·c
        11 => {                                                                              // z²·c
            let (z2r,z2i) = (zr.clone()*zr.clone()-zi.clone()*zi.clone(), zr.clone()*zi.clone()*2.0);
            t_cmul(&z2r, &z2i, cr, ci)
        },
        12 => {                                                                              // z·c²
            let (c2r,c2i) = (cr.clone()*cr.clone()-ci.clone()*ci.clone(), cr.clone()*ci.clone()*2.0);
            t_cmul(zr, zi, &c2r, &c2i)
        },
        13 => {                                                                              // z²·c²
            let (z2r,z2i) = (zr.clone()*zr.clone()-zi.clone()*zi.clone(), zr.clone()*zi.clone()*2.0);
            let (c2r,c2i) = (cr.clone()*cr.clone()-ci.clone()*ci.clone(), cr.clone()*ci.clone()*2.0);
            t_cmul(&z2r, &z2i, &c2r, &c2i)
        },
        14 => {                                                                              // c/(z+ε)
            let d = zr.clone()*zr.clone()+zi.clone()*zi.clone()+EPS;
            let (ir,ii) = (zr.clone()/d.clone(), zi.clone().neg()/d);
            t_cmul(cr, ci, &ir, &ii)
        },
        15 => {                                                                              // (z+c)²
            let (sr,si) = (zr.clone()+cr.clone(), zi.clone()+ci.clone());
            (sr.clone()*sr.clone()-si.clone()*si.clone(), sr*si*2.0)
        },
        16 => {                                                                              // (z−c)²
            let (sr,si) = (zr.clone()-cr.clone(), zi.clone()-ci.clone());
            (sr.clone()*sr.clone()-si.clone()*si.clone(), sr*si*2.0)
        },
        17 => {                                                                              // (z·c)²
            let (zcr,zci) = t_cmul(zr, zi, cr, ci);
            (zcr.clone()*zcr.clone()-zci.clone()*zci.clone(), zcr*zci*2.0)
        },

        // C: Trigonometric
        18 => t_csin(zr, zi),                                                               // sin(z)
        19 => t_ccos(zr, zi),                                                               // cos(z)
        20 => t_csin(&(zr.clone()*std::f32::consts::PI), &(zi.clone()*std::f32::consts::PI)), // sin(πz)
        21 => t_ccos(&(zr.clone()*std::f32::consts::PI), &(zi.clone()*std::f32::consts::PI)), // cos(πz)
        22 => { let (z2r,z2i)=(zr.clone()*zr.clone()-zi.clone()*zi.clone(),zr.clone()*zi.clone()*2.0); t_csin(&z2r,&z2i) }, // sin(z²)
        23 => { let (z2r,z2i)=(zr.clone()*zr.clone()-zi.clone()*zi.clone(),zr.clone()*zi.clone()*2.0); t_ccos(&z2r,&z2i) }, // cos(z²)
        24 => t_csin(&(zr.clone()+cr.clone()), &(zi.clone()+ci.clone())),                  // sin(z+c)
        25 => t_ccos(&(zr.clone()+cr.clone()), &(zi.clone()+ci.clone())),                  // cos(z+c)
        26 => { let (zcr,zci)=t_cmul(zr,zi,cr,ci); t_csin(&zcr,&zci) },                   // sin(z·c)
        27 => { let (zcr,zci)=t_cmul(zr,zi,cr,ci); t_ccos(&zcr,&zci) },                   // cos(z·c)
        28 => { let (sr,si)=t_csin(zr,zi); t_cmul(zr,zi,&sr,&si) },                       // z·sin(z)
        29 => { let (cr2,ci2)=t_ccos(zr,zi); t_cmul(zr,zi,&cr2,&ci2) },                   // z·cos(z)
        30 => {                                                                              // tan(z)
            let (sr,si) = t_csin(zr, zi);
            let (cvr,cvi) = t_ccos(zr, zi);
            let d = cvr.clone()*cvr.clone() + cvi.clone()*cvi.clone() + EPS;
            ((sr.clone()*cvr.clone()+si.clone()*cvi.clone())/d.clone(),
             (si*cvr-sr*cvi)/d)
        },
        31 => {                                                                              // sinh(z)
            let (er,ei) = t_cexp(zr, zi);
            let (enr,eni) = t_cexp(&zr.clone().neg(), &zi.clone().neg());
            ((er-enr)*0.5, (ei-eni)*0.5)
        },
        32 => {                                                                              // cosh(z)
            let (er,ei) = t_cexp(zr, zi);
            let (enr,eni) = t_cexp(&zr.clone().neg(), &zi.clone().neg());
            ((er+enr)*0.5, (ei+eni)*0.5)
        },
        33 => {                                                                              // tanh(z)
            let x2 = zr.clone()*2.0;
            let y2 = zi.clone()*2.0;
            let d = x2.clone().cosh() + y2.clone().cos() + EPS;
            (x2.sinh()/d.clone(), y2.sin()/d)
        },

        // D: Exponential / logarithmic
        34 => t_cexp(zr, zi),                                                               // exp(z)
        35 => t_cexp(&zr.clone().neg(), &zi.clone().neg()),                                 // exp(−z)
        36 => { let (zcr,zci)=t_cmul(&zr.clone().clamp(-4.0,4.0),zi,cr,ci); t_cexp(&zcr,&zci) }, // exp(z·c)
        37 => { let (er,ei)=t_cexp(&zr.clone().clamp(-4.0,4.0),zi); t_cmul(zr,zi,&er,&ei) }, // z·exp(z)
        38 => { let (er,ei)=t_cexp(&zr.clone().clamp(-4.0,4.0),zi); t_cmul(&er,&ei,cr,ci) }, // exp(z)·c
        39 => t_clog(&(zr.clone()+1.0), zi),                                                // log(z+1)
        40 => { let (z2r,z2i)=(zr.clone()*zr.clone()-zi.clone()*zi.clone()+1.0,zr.clone()*zi.clone()*2.0); t_clog(&z2r,&z2i) }, // log(z²+1)
        41 => { let (lr,li)=t_clog(&(zr.clone()+1.0),zi); t_cmul(zr,zi,&lr,&li) },        // z·log(z+1)
        42 => {                                                                              // sin(1/z)
            let d = zr.clone()*zr.clone()+zi.clone()*zi.clone()+EPS;
            let (ir,ii) = ((zr.clone()/d.clone()).clamp(-10.0,10.0), (zi.clone().neg()/d).clamp(-10.0,10.0));
            t_csin(&ir, &ii)
        },
        43 => {                                                                              // exp(1/z)
            let d = zr.clone()*zr.clone()+zi.clone()*zi.clone()+EPS;
            let (ir,ii) = ((zr.clone()/d.clone()).clamp(-4.0,4.0), (zi.clone().neg()/d).clamp(-8.0,8.0));
            t_cexp(&ir, &ii)
        },

        // E: Non-holomorphic
        44 => (zr.clone().abs(), zi.clone()),                                               // |Re(z)| + i·Im(z)
        45 => (zr.clone(), zi.clone().abs()),                                               // Re(z) + i·|Im(z)|
        46 => (zr.clone().abs(), zi.clone().abs()),                                         // Burning Ship
        47 => (zr.clone(), zi.clone().neg()),                                               // conj(z)
        48 => (zr.clone()*zr.clone()-zi.clone()*zi.clone(), zr.clone()*zi.clone()*(-2.0)), // conj(z)² Tricorn
        49 => { let m=(zr.clone()*zr.clone()+zi.clone()*zi.clone()).sqrt(); (zr.clone()*m.clone(),zi.clone()*m) }, // z·|z|
        50 => { let m=(zr.clone()*zr.clone()+zi.clone()*zi.clone()).sqrt()+EPS; (zr.clone()/m.clone(),zi.clone()/m) }, // z/|z|

        // F: Rational
        51 => {                                                                              // z/(z²+1)
            let (z2r,z2i)=(zr.clone()*zr.clone()-zi.clone()*zi.clone()+1.0, zr.clone()*zi.clone()*2.0);
            let d = z2r.clone()*z2r.clone()+z2i.clone()*z2i.clone()+EPS;
            ((zr.clone()*z2r.clone()+zi.clone()*z2i.clone())/d.clone(),
             (zi.clone()*z2r-zr.clone()*z2i)/d)
        },
        52 => {                                                                              // (z²−1)/(z²+1)
            let (z2r,z2i)=(zr.clone()*zr.clone()-zi.clone()*zi.clone(), zr.clone()*zi.clone()*2.0);
            let (nr,ni)=(z2r.clone()-1.0, z2i.clone());
            let (dr,di)=(z2r+1.0, z2i);
            let d = dr.clone()*dr.clone()+di.clone()*di.clone()+EPS;
            ((nr.clone()*dr.clone()+ni.clone()*di.clone())/d.clone(),
             (ni*dr-nr*di)/d)
        },
        53 => {                                                                              // z²/(z−1+ε)
            let (z2r,z2i)=(zr.clone()*zr.clone()-zi.clone()*zi.clone(), zr.clone()*zi.clone()*2.0);
            let (dr,di)=(zr.clone()-1.0, zi.clone());
            let d = dr.clone()*dr.clone()+di.clone()*di.clone()+EPS;
            ((z2r.clone()*dr.clone()+z2i.clone()*di.clone())/d.clone(),
             (z2i*dr-z2r*di)/d)
        },
        54 => {                                                                              // 1/(z²+c)
            let (z2r,z2i)=(zr.clone()*zr.clone()-zi.clone()*zi.clone(), zr.clone()*zi.clone()*2.0);
            let (dr,di)=(z2r+cr.clone(), z2i+ci.clone());
            let d = dr.clone()*dr.clone()+di.clone()*di.clone()+EPS;
            (dr/d.clone(), di.neg()/d)
        },
        55 => {                                                                              // z²·c/(z+c+ε)
            let (z2r,z2i)=(zr.clone()*zr.clone()-zi.clone()*zi.clone(), zr.clone()*zi.clone()*2.0);
            let (dr,di)=(zr.clone()+cr.clone(), zi.clone()+ci.clone());
            let d = dr.clone()*dr.clone()+di.clone()*di.clone()+EPS;
            let (ir,ii)=(dr/d.clone(), di.neg()/d);
            let (z2cr,z2ci)=t_cmul(&z2r,&z2i,cr,ci);
            t_cmul(&z2cr,&z2ci,&ir,&ii)
        },

        // G: Constants
        56 => {                                                                             // 1 (broadcast to shape of z_re)
            let shape = zr.dims();
            (Tensor::ones(shape, &zr.device()), Tensor::zeros(shape, &zr.device()))
        },
        57 => {                                                                             // i
            let shape = zr.dims();
            (Tensor::zeros(shape, &zr.device()), Tensor::ones(shape, &zr.device()))
        },

        _  => {
            let shape = zr.dims();
            (Tensor::zeros(shape, &zr.device()), Tensor::zeros(shape, &zr.device()))
        },
    }
}

/// Tensor weighted sum: z_new = Σᵢ (fw_re[i]+i·fw_im[i]) * φᵢ(z, c).
/// fw_re/fw_im: [1, N_BASIS] (broadcast over N pixels).
/// z/c: [N, 2].  Returns [N, 2].
pub fn apply_formula_tensor<B: Backend>(
    fw_re: &Tensor<B, 2>,
    fw_im: &Tensor<B, 2>,
    z:     &Tensor<B, 2>,
    c:     &Tensor<B, 2>,
) -> Tensor<B, 2> {
    let z_re = z.clone().narrow(1, 0, 1);  // [N, 1]
    let z_im = z.clone().narrow(1, 1, 1);  // [N, 1]
    let c_re = c.clone().narrow(1, 0, 1);  // [N, 1]
    let c_im = c.clone().narrow(1, 1, 1);  // [N, 1]
    let device = z.device();
    let n = z.dims()[0];
    let mut sum_re: Tensor<B, 2> = Tensor::zeros([n, 1], &device);
    let mut sum_im: Tensor<B, 2> = Tensor::zeros([n, 1], &device);

    for i in 0..N_BASIS {
        let wr = fw_re.clone().narrow(1, i, 1);  // [1, 1] → broadcasts to [N, 1]
        let wi = fw_im.clone().narrow(1, i, 1);  // [1, 1]
        let (phi_re, phi_im) = eval_basis_tensor(i, &z_re, &z_im, &c_re, &c_im);
        // (wr + i·wi)(phi_re + i·phi_im) = (wr·phi_re − wi·phi_im) + i·(wr·phi_im + wi·phi_re)
        sum_re = sum_re + wr.clone() * phi_re.clone() - wi.clone() * phi_im.clone();
        sum_im = sum_im + wr * phi_im + wi * phi_re;
    }

    Tensor::cat(vec![sum_re, sum_im], 1)  // [N, 2]
}

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
