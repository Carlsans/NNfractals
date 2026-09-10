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
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct OpNode {
    pub op: u8,
    pub a:  u8,
    pub b:  u8,
    #[serde(default)] pub kre: f32,
    #[serde(default)] pub kim: f32,
}

/// Which nodes of `prog` actually contribute to its result.
///
/// A program's value is its LAST node, and evolved DAGs routinely accumulate
/// unreachable subtrees — introns, in GP terms. On a real archived genome
/// measured while building the time sweep, 5 of 9 nodes were dead. Anything
/// that reasons about "changing this node" has to know that, or it will spend
/// its budget perturbing values nothing reads and then report the result as a
/// mysterious no-op.
///
/// Returns an empty vec for an empty program.
pub fn reachable_from_root(prog: &[OpNode]) -> Vec<bool> {
    let mut live = vec![false; prog.len()];
    if prog.is_empty() {
        return live;
    }
    let root = prog.len() - 1;
    live[root] = true;
    // Operands always index strictly-earlier nodes, so one backward sweep
    // suffices — no work list needed.
    for i in (0..prog.len()).rev() {
        if !live[i] {
            continue;
        }
        let arity = op::arity(prog[i].op);
        if arity >= 1 {
            if let Some(slot) = live.get_mut(prog[i].a as usize) {
                *slot = true;
            }
        }
        if arity >= 2 {
            if let Some(slot) = live.get_mut(prog[i].b as usize) {
                *slot = true;
            }
        }
    }
    live
}

// ── Time modulation ─────────────────────────────────────────────────────────
//
// A fractal is normally a function of two variables (the pixel coordinate). A
// `TimeMod` adds a third: it names one scalar inside the genome and varies it
// over t ∈ [0,1), so a clip becomes a continuous walk through a one-parameter
// family of fractals rather than a camera move over a fixed one.
//
// Every target below is a value the renderer already re-reads per render — the
// DAG program's own `kre`/`kim`, the Julia constant, phoenix, bailout — and all
// of them are re-uploaded to the GPU on every dispatch (`render_gpu::dag_item`
// packs them fresh each time). So modulation is applied by building a modified
// `Genome` per frame (`Genome::at_time`) and rendering it normally: no change to
// the CPU kernel, the f64/DD kernels, or `fractal.wgsl`.

/// Which scalar inside a genome a `TimeMod` drives.
///
/// Variants are serialized by NAME into the `.nn` file, so they may be added
/// but must not be renamed — same append-only rule as the opcodes above.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ModTarget {
    /// The Julia constant `c`. Two-channel. Present on ~98% of the archive, and
    /// the most reliably beautiful axis: the set morphs continuously while the
    /// camera stays put. Inert unless `julia_mode` is set.
    JuliaC,
    /// The phoenix memory coefficient `p` in `z' = f(z,c) + p·z_prev`.
    /// Two-channel. Changes the order of the recurrence itself.
    Phoenix,
    /// The escape radius. Single-channel, and the weakest of the set: it does
    /// not change the set's topology, only where the smooth escape-time bands
    /// fall, so it reads as a pulsing rather than a morph.
    Bailout,
    /// The `kre`/`kim` of an existing CONST node in the main program.
    /// Two-channel. The most direct "scale one term of the formula", but only
    /// ~25% of archived genomes have a CONST node at all — see `ProgScale`.
    ProgConst { node: u8 },
    /// The `kre`/`kim` of an existing CONST node in the coordinate-warp program.
    /// Two-channel. Bends the input plane over time.
    WarpConst { node: u8 },
    /// For the three-quarters of genomes with no CONST node: splice
    /// `MUL(CONST(k), program[node])` in at render time and drive `k`, so any
    /// subtree can be scaled even when the evolved program never allocated a
    /// constant. Costs 2 of the 24 register slots and is skipped when the
    /// program has no room.
    ProgScale { node: u8 },
}

impl ModTarget {
    /// Whether this target drives a complex pair rather than a lone scalar.
    /// `Bailout` is the only single-channel target.
    pub fn is_two_channel(self) -> bool {
        !matches!(self, ModTarget::Bailout)
    }

    /// Short label for logs, manifests and the viewer's list.
    pub fn label(self) -> String {
        match self {
            ModTarget::JuliaC => "julia c".to_string(),
            ModTarget::Phoenix => "phoenix p".to_string(),
            ModTarget::Bailout => "bailout".to_string(),
            ModTarget::ProgConst { node } => format!("prog const #{node}"),
            ModTarget::WarpConst { node } => format!("warp const #{node}"),
            ModTarget::ProgScale { node } => format!("prog scale #{node}"),
        }
    }
}

/// The shape of the modulation over one clip.
///
/// `Sine`, `Cosine` and `Triangle` are continuous AND periodic, so at an integer
/// `freq` the clip loops seamlessly — the last frame flows back into the first.
/// `Sawtooth` is periodic but NOT continuous: it snaps back once per cycle, so
/// expect a visible cut there. `Pulse` and `Ramp` are one-shot and do not loop.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ModShape {
    Sine,
    Cosine,
    Triangle,
    Sawtooth,
    /// A single Gaussian bump centred at `phase`; `freq` narrows it.
    Pulse,
    /// A straight line from 0, reaching `amp` at t = 1/`freq`.
    Ramp,
    /// Two-channel only: walks a circle of radius `amp` in the complex plane.
    /// The natural shape for `JuliaC` and `Phoenix` — a circular orbit of the
    /// Julia constant is the classic animated-Julia move. On a single-channel
    /// target it degenerates to its real part, i.e. a cosine.
    Orbit,
}

impl ModShape {
    pub fn label(self) -> &'static str {
        match self {
            ModShape::Sine => "sine",
            ModShape::Cosine => "cosine",
            ModShape::Triangle => "triangle",
            ModShape::Sawtooth => "sawtooth",
            ModShape::Pulse => "pulse",
            ModShape::Ramp => "ramp",
            ModShape::Orbit => "orbit",
        }
    }

    pub fn parse(s: &str) -> Option<ModShape> {
        Some(match s.trim().to_ascii_lowercase().as_str() {
            "sine" | "sin" => ModShape::Sine,
            "cosine" | "cos" => ModShape::Cosine,
            "triangle" | "tri" => ModShape::Triangle,
            "sawtooth" | "saw" => ModShape::Sawtooth,
            "pulse" => ModShape::Pulse,
            "ramp" => ModShape::Ramp,
            "orbit" => ModShape::Orbit,
            _ => return None,
        })
    }

    /// Whether the clip returns to its starting value, so an integer number of
    /// cycles plays as a seamless loop.
    pub fn loops(self) -> bool {
        matches!(self, ModShape::Sine | ModShape::Cosine | ModShape::Triangle | ModShape::Orbit)
    }

    pub const ALL: &'static [ModShape] = &[
        ModShape::Sine, ModShape::Cosine, ModShape::Triangle,
        ModShape::Sawtooth, ModShape::Pulse, ModShape::Ramp, ModShape::Orbit,
    ];
}

/// One time-modulation channel: what to drive, how, and by how much.
///
/// The value is an OFFSET added to the genome's own stored value, never a
/// replacement — so `amp = 0` is always exactly the original fractal, and a
/// modulation can be dialled continuously from "off" to "wild" without the
/// genome's identity jumping.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TimeMod {
    pub target: ModTarget,
    pub shape: ModShape,
    /// Peak offset applied to the target. Interpreted in the target's own
    /// units (the complex plane for julia/phoenix/consts, radius for bailout).
    #[serde(default)]
    pub amp: f32,
    /// Cycles over the full clip. 1.0 = exactly one loop.
    #[serde(default = "default_mod_freq")]
    pub freq: f32,
    /// Where in the cycle the clip starts, in turns (0..1).
    #[serde(default)]
    pub phase: f32,
}

fn default_mod_freq() -> f32 { 1.0 }

impl TimeMod {
    pub fn new(target: ModTarget, shape: ModShape, amp: f32) -> Self {
        TimeMod { target, shape, amp, freq: 1.0, phase: 0.0 }
    }
}

/// The scalar offset for `m` at time `t ∈ [0,1)`.
///
/// This is the real channel; for two-channel targets see [`mod_offset`].
pub fn mod_value(m: &TimeMod, t: f32) -> f32 {
    mod_offset(m, t).0
}

/// The complex offset `(re, im)` for `m` at time `t ∈ [0,1)`.
///
/// Every shape except `Orbit` drives the real channel only, leaving the
/// imaginary part of the target untouched — so switching a modulation from
/// `Sine` to `Orbit` is the difference between sliding along a line and walking
/// a circle, which is exactly the distinction worth having.
pub fn mod_offset(m: &TimeMod, t: f32) -> (f32, f32) {
    use std::f32::consts::TAU;
    let theta = TAU * (m.freq * t + m.phase);
    match m.shape {
        ModShape::Sine => (m.amp * theta.sin(), 0.0),
        ModShape::Cosine => (m.amp * theta.cos(), 0.0),
        ModShape::Triangle => {
            // 4|x - round(x)| - 1 over one period: a continuous -1..1 ramp pair.
            let x = m.freq * t + m.phase;
            let tri = 4.0 * (x - (x + 0.5).floor()).abs() - 1.0;
            (m.amp * tri, 0.0)
        }
        ModShape::Sawtooth => {
            let x = m.freq * t + m.phase;
            (m.amp * (2.0 * (x - x.floor()) - 1.0), 0.0)
        }
        ModShape::Pulse => {
            // Gaussian bump centred at `phase`; `freq` narrows it. The factor 6
            // makes freq = 1 span roughly a third of the clip.
            let d = (t - m.phase) * m.freq.max(1e-3) * 6.0;
            (m.amp * (-d * d).exp(), 0.0)
        }
        ModShape::Ramp => (m.amp * (m.freq * t + m.phase), 0.0),
        ModShape::Orbit => (m.amp * theta.cos(), m.amp * theta.sin()),
    }
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

#[cfg(test)]
mod reachability_tests {
    use super::*;

    fn n(o: u8, a: u8, b: u8) -> OpNode { OpNode { op: o, a, b, kre: 0.0, kim: 0.0 } }

    #[test]
    fn an_empty_program_has_no_live_nodes() {
        assert!(reachable_from_root(&[]).is_empty());
    }

    #[test]
    fn a_fully_used_program_is_all_live() {
        // z² + c: every node feeds the root.
        let prog = [n(op::Z, 0, 0), n(op::C, 0, 0), n(op::SQR, 0, 0), n(op::ADD, 2, 1)];
        assert_eq!(reachable_from_root(&prog), vec![true; 4]);
    }

    #[test]
    fn introns_are_reported_dead() {
        // Shape taken from a real archived genome (0b3199d357fc16e0), where
        // five of nine nodes never reach the root.
        let prog = [
            n(op::Z, 0, 0),        // 0
            n(op::NORMZ, 0, 0),    // 1
            n(op::ABSRE, 1, 0),    // 2
            n(op::SQR, 1, 0),      // 3  dead
            n(op::CUBE, 0, 0),     // 4  dead
            n(op::SIN, 1, 0),      // 5  dead
            n(op::CONST, 0, 0),    // 6  dead
            n(op::DIV, 4, 1),      // 7  dead
            n(op::ABSIM, 2, 0),    // 8  root
        ];
        assert_eq!(
            reachable_from_root(&prog),
            vec![true, true, true, false, false, false, false, false, true]
        );
    }

    #[test]
    fn the_root_is_always_live_even_as_a_lone_leaf() {
        assert_eq!(reachable_from_root(&[n(op::Z, 0, 0)]), vec![true]);
    }

    #[test]
    fn a_leafs_stale_operand_bytes_do_not_revive_nodes() {
        // Leaves ignore a/b, so garbage there must not mark anything live.
        let prog = [n(op::Z, 0, 0), n(op::CONST, 0, 0), n(op::C, 7, 7)];
        assert_eq!(reachable_from_root(&prog), vec![false, false, true]);
    }
}

#[cfg(test)]
mod time_mod_tests {
    use super::*;

    fn m(shape: ModShape, amp: f32, freq: f32) -> TimeMod {
        TimeMod { target: ModTarget::JuliaC, shape, amp, freq, phase: 0.0 }
    }

    #[test]
    fn looping_shapes_return_to_their_start_at_integer_freq() {
        // This is what makes a clip loop seamlessly: the frame after the last
        // one would be the first one again.
        for shape in [ModShape::Sine, ModShape::Cosine, ModShape::Triangle, ModShape::Orbit] {
            for freq in [1.0f32, 2.0, 3.0] {
                let tm = m(shape, 0.4, freq);
                let (a_re, a_im) = mod_offset(&tm, 0.0);
                let (b_re, b_im) = mod_offset(&tm, 1.0);
                assert!((a_re - b_re).abs() < 1e-5 && (a_im - b_im).abs() < 1e-5,
                    "{shape:?} at freq {freq} does not loop: {a_re},{a_im} vs {b_re},{b_im}");
                assert!(shape.loops(), "{shape:?} should report loops() = true");
            }
        }
    }

    #[test]
    fn sawtooth_is_periodic_but_reports_that_it_does_not_loop() {
        // It returns to the same value, but by snapping rather than flowing —
        // so `loops()` must say false or the UI would promise a seamless clip
        // it cannot deliver.
        let tm = m(ModShape::Sawtooth, 1.0, 1.0);
        assert!(!ModShape::Sawtooth.loops());
        // Just before the wrap it is near +amp; just after, near -amp.
        assert!(mod_value(&tm, 0.99) > 0.9, "got {}", mod_value(&tm, 0.99));
        assert!(mod_value(&tm, 0.01) < -0.9, "got {}", mod_value(&tm, 0.01));
    }

    #[test]
    fn continuous_shapes_have_no_jumps_across_the_clip() {
        for shape in [ModShape::Sine, ModShape::Cosine, ModShape::Triangle, ModShape::Orbit] {
            let tm = m(shape, 1.0, 2.0);
            let steps = 400;
            let mut prev = mod_offset(&tm, 0.0);
            for i in 1..=steps {
                let t = i as f32 / steps as f32;
                let cur = mod_offset(&tm, t);
                let jump = ((cur.0 - prev.0).powi(2) + (cur.1 - prev.1).powi(2)).sqrt();
                assert!(jump < 0.1, "{shape:?} jumps {jump} at t={t}");
                prev = cur;
            }
        }
    }

    #[test]
    fn zero_amplitude_is_always_the_identity() {
        // The invariant the whole design rests on: dialling a modulation to 0
        // must give back exactly the original fractal, for every shape.
        for shape in ModShape::ALL {
            let tm = m(*shape, 0.0, 1.7);
            for i in 0..20 {
                let t = i as f32 / 20.0;
                assert_eq!(mod_offset(&tm, t), (0.0, 0.0), "{shape:?} at t={t}");
            }
        }
    }

    #[test]
    fn orbit_walks_a_circle_of_radius_amp() {
        let tm = m(ModShape::Orbit, 0.3, 1.0);
        for i in 0..32 {
            let t = i as f32 / 32.0;
            let (re, im) = mod_offset(&tm, t);
            let r = (re * re + im * im).sqrt();
            assert!((r - 0.3).abs() < 1e-5, "radius {r} at t={t}");
        }
    }

    #[test]
    fn only_orbit_drives_the_imaginary_channel() {
        for shape in ModShape::ALL {
            let tm = m(*shape, 0.5, 1.0);
            let (_, im) = mod_offset(&tm, 0.37);
            if *shape == ModShape::Orbit {
                assert!(im.abs() > 1e-6, "orbit must move in im");
            } else {
                assert_eq!(im, 0.0, "{shape:?} must leave im alone");
            }
        }
    }

    #[test]
    fn pulse_peaks_at_its_phase_and_decays_away() {
        let tm = TimeMod { target: ModTarget::Bailout, shape: ModShape::Pulse,
                           amp: 1.0, freq: 1.0, phase: 0.5 };
        let peak = mod_value(&tm, 0.5);
        assert!((peak - 1.0).abs() < 1e-5, "peak {peak}");
        assert!(mod_value(&tm, 0.0) < 0.02, "should be ~0 far from the peak");
        assert!(mod_value(&tm, 1.0) < 0.02);
    }

    #[test]
    fn shape_names_round_trip() {
        for shape in ModShape::ALL {
            assert_eq!(ModShape::parse(shape.label()), Some(*shape));
        }
        assert_eq!(ModShape::parse("SIN"), Some(ModShape::Sine));
        assert_eq!(ModShape::parse("nonsense"), None);
    }

    #[test]
    fn bailout_is_the_only_single_channel_target() {
        assert!(!ModTarget::Bailout.is_two_channel());
        for t in [ModTarget::JuliaC, ModTarget::Phoenix,
                  ModTarget::ProgConst { node: 0 }, ModTarget::WarpConst { node: 0 },
                  ModTarget::ProgScale { node: 0 }] {
            assert!(t.is_two_channel(), "{t:?}");
        }
    }

    #[test]
    fn time_mod_survives_a_json_round_trip() {
        let tm = TimeMod { target: ModTarget::ProgScale { node: 7 }, shape: ModShape::Orbit,
                           amp: 0.25, freq: 2.0, phase: 0.125 };
        let json = serde_json::to_string(&tm).unwrap();
        assert_eq!(serde_json::from_str::<TimeMod>(&json).unwrap(), tm);
    }
}
