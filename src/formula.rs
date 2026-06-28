pub const N_BASIS: usize = 58;

// ════════════════════════════════════════════════════════════════════════════
// Expression-DAG formula system (Phase 1).
//
// A genome's formula can be represented as a small directed acyclic graph of
// primitive operations instead of the flat 58-basis weighted sum. Each node
// reads at most two earlier nodes (strict topological order: a,b < own index),
// so evaluation is a single forward pass over a register file. The root (last
// node) is z_{n+1}. This subsumes every legacy basis (e.g. z²c = Mul(Sqr Z, C))
// while unlocking composition/products/division the flat sum cannot express.
//
// INVARIANT: the opcode semantics here, the WGSL register-VM in fractal.wgsl,
// and (via the macro) both f32/f64 paths must stay byte-for-byte identical.
// ════════════════════════════════════════════════════════════════════════════

/// Register-file size of the program VM (shared with the WGSL VM). This is the
/// hard ceiling on node count; *evolved* programs are capped lower via
/// `config.max_nodes`. Sized to also fit legacy→DAG conversion of typical
/// multi-term genomes (an 8-term genome with complex bases may still overflow
/// and stay on the legacy path).
pub const N_SLOTS: usize = 24;

/// Opcodes. Values are part of the on-disk genome + GPU upload format — append
/// only, never renumber.
pub mod op {
    pub const Z:       u8 = 0;  // leaf: current iterate z
    pub const C:       u8 = 1;  // leaf: parameter c
    pub const CONST:   u8 = 2;  // leaf: complex constant (kre, kim)
    pub const SQR:     u8 = 3;  // a²
    pub const CUBE:    u8 = 4;  // a³
    pub const QUART:   u8 = 5;  // a⁴
    pub const RECIP:   u8 = 6;  // 1/a
    pub const SIN:     u8 = 7;
    pub const COS:     u8 = 8;
    pub const EXP:     u8 = 9;
    pub const LOG:     u8 = 10; // log(a) (guarded)
    pub const TANH:    u8 = 11;
    pub const CONJ:    u8 = 12; // (re, -im)
    pub const ABSFOLD: u8 = 13; // (|re|, |im|)  — burning-ship fold
    pub const ABSRE:   u8 = 14; // (|re|, im)
    pub const ABSIM:   u8 = 15; // (re, |im|)
    pub const NORMZ:   u8 = 16; // a/|a|
    pub const ADD:     u8 = 17; // a + b
    pub const SUB:     u8 = 18; // a − b
    pub const MUL:     u8 = 19; // a · b
    pub const DIV:     u8 = 20; // a / b
    pub const N_OPS:   usize = 21;

    /// Number of node inputs each op reads (0 leaf, 1 unary, 2 binary). Used by
    /// random program generation and mutation to wire valid DAGs.
    pub fn arity(o: u8) -> u8 {
        match o {
            Z | C | CONST => 0,
            ADD | SUB | MUL | DIV => 2,
            _ => 1,
        }
    }

    pub fn name(o: u8) -> &'static str {
        match o {
            Z=>"z", C=>"c", CONST=>"k", SQR=>"sqr", CUBE=>"cube", QUART=>"quart",
            RECIP=>"recip", SIN=>"sin", COS=>"cos", EXP=>"exp", LOG=>"log",
            TANH=>"tanh", CONJ=>"conj", ABSFOLD=>"absfold", ABSRE=>"absre",
            ABSIM=>"absim", NORMZ=>"normz", ADD=>"add", SUB=>"sub", MUL=>"mul",
            DIV=>"div", _=>"?",
        }
    }
}

/// One node of an expression-DAG program. `a`/`b` index strictly-earlier nodes
/// (ignored for ops whose arity is below 2). `kre`/`kim` used only by CONST.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct OpNode {
    pub op: u8,
    pub a:  u8,
    pub b:  u8,
    #[serde(default)] pub kre: f32,
    #[serde(default)] pub kim: f32,
}

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

            /// Evaluate an expression-DAG program → z_new. Single forward pass
            /// over a register file; node i reads earlier registers a,b. Returns
            /// the root (last node). MUST match the WGSL register-VM exactly.
            pub fn eval_program(prog: &[super::OpNode], zx: $ty, zy: $ty, cx: $ty, cy: $ty) -> ($ty, $ty) {
                use super::op::*;
                let n = prog.len().min(super::N_SLOTS);
                if n == 0 { return (0.0 as $ty, 0.0 as $ty); }
                let mut reg = [(0.0 as $ty, 0.0 as $ty); super::N_SLOTS];
                for i in 0..n {
                    let node = prog[i];
                    let ai = (node.a as usize).min(super::N_SLOTS - 1);
                    let bi = (node.b as usize).min(super::N_SLOTS - 1);
                    // Topological safety: inputs must precede i; out-of-order → 0.
                    let (ax, ay) = if ai < i { reg[ai] } else { (0.0 as $ty, 0.0 as $ty) };
                    let (bx, by) = if bi < i { reg[bi] } else { (0.0 as $ty, 0.0 as $ty) };
                    reg[i] = match node.op {
                        Z     => (zx, zy),
                        C     => (cx, cy),
                        CONST => (node.kre as $ty, node.kim as $ty),
                        SQR   => (ax*ax - ay*ay, 2.0*ax*ay),
                        CUBE  => { let (a2,b2)=(ax*ax, ay*ay); (ax*(a2 - 3.0*b2), ay*(3.0*a2 - b2)) },
                        QUART => { let (a2,b2)=(ax*ax, ay*ay); (a2*a2 - 6.0*a2*b2 + b2*b2, 4.0*ax*ay*(a2 - b2)) },
                        RECIP => { let d = ax*ax + ay*ay + EPS; (ax/d, -ay/d) },
                        SIN   => csin(ax, ay),
                        COS   => ccos(ax, ay),
                        EXP   => cexp(ax, ay),
                        LOG   => clog(ax, ay),
                        TANH  => { let (x2,y2)=(2.0*ax, 2.0*ay); let d = x2.cosh() + y2.cos() + EPS; (x2.sinh()/d, y2.sin()/d) },
                        CONJ  => (ax, -ay),
                        ABSFOLD => (ax.abs(), ay.abs()),
                        ABSRE => (ax.abs(), ay),
                        ABSIM => (ax, ay.abs()),
                        NORMZ => { let m = (ax*ax + ay*ay).sqrt() + EPS; (ax/m, ay/m) },
                        ADD   => (ax + bx, ay + by),
                        SUB   => (ax - bx, ay - by),
                        MUL   => cmul(ax, ay, bx, by),
                        DIV   => { let d = bx*bx + by*by + EPS; cmul(ax, ay, bx/d, -by/d) },
                        _     => (0.0 as $ty, 0.0 as $ty),
                    };
                }
                reg[n - 1]
            }
        }
    };
}

define_precision!(f32_impl, f32);
define_precision!(f64_impl, f64);

// Backward-compatible re-exports: the rest of the codebase (GA, GPU CPU fallback,
// normal rendering) uses the f32 versions exactly as before.
pub use f32_impl::{apply_formula, eval_basis, eval_program};

/// Human-readable label for a basis function.
#[cfg(test)]
mod dag_tests {
    use super::*;

    // Hand-built Mandelbrot DAG: Add(Sqr(Z), C) must equal legacy z²+c.
    #[test]
    fn mandelbrot_dag_matches_legacy() {
        let prog = vec![
            OpNode { op: op::Z,   a: 0, b: 0, kre: 0.0, kim: 0.0 }, // 0: z
            OpNode { op: op::C,   a: 0, b: 0, kre: 0.0, kim: 0.0 }, // 1: c
            OpNode { op: op::SQR, a: 0, b: 0, kre: 0.0, kim: 0.0 }, // 2: z²
            OpNode { op: op::ADD, a: 2, b: 1, kre: 0.0, kim: 0.0 }, // 3: z²+c
        ];
        for &(zx, zy, cx, cy) in &[(0.3f32,0.4,-0.5,0.6),(1.2,-0.7,0.1,0.2),(-0.9,0.3,0.4,-0.8)] {
            let (px, py) = eval_program(&prog, zx, zy, cx, cy);
            let legacy = (zx*zx - zy*zy + cx, 2.0*zx*zy + cy);
            assert!((px - legacy.0).abs() < 1e-6 && (py - legacy.1).abs() < 1e-6,
                "mismatch at ({zx},{zy},{cx},{cy}): dag={px:?},{py:?} legacy={legacy:?}");
        }
    }

    // f32 and f64 eval_program must agree (3-way parity, CPU legs).
    #[test]
    fn f32_f64_program_parity() {
        let prog = vec![
            OpNode { op: op::Z,    a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::C,    a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::SQR,  a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::SIN,  a: 2, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::ADD,  a: 3, b: 1, kre: 0.0, kim: 0.0 },
        ];
        let (a, b)   = f32_impl::eval_program(&prog, 0.4, 0.3, -0.2, 0.5);
        let (a2, b2) = f64_impl::eval_program(&prog, 0.4, 0.3, -0.2, 0.5);
        assert!((a as f64 - a2).abs() < 1e-4 && (b as f64 - b2).abs() < 1e-4);
    }
}

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
