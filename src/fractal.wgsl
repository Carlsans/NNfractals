// WGSL compute shader: fractal escape-time renderer
// Each thread computes one pixel. apply_formula loops over all 58 basis functions
// with weights supplied via a storage buffer.
// No control-flow divergence: all threads in a warp take the same switch branch per iteration.

struct Params {
    width:        u32,
    height:       u32,
    max_iter:     u32,
    genome_count: u32,
    bailout_sq:   f32,
    use_dag:      u32,   // 0 = legacy 58-basis sum, 1 = expression-DAG VM
    _pad2:        f32,
    _pad3:        f32,
}

// Batched layout (Z dispatch dimension = genome index):
//   all_fw      : genome_count × 116 f32s  (58 complex weights each)
//   view_bounds : genome_count × 4 f32s    (xmin, xmax, ymin, ymax per genome)
//   output      : genome_count × width×height f32s (escape times)
@group(0) @binding(0) var<uniform>             params      : Params;
@group(0) @binding(1) var<storage, read>       all_fw      : array<f32>;
@group(0) @binding(2) var<storage, read>       view_bounds : array<f32>;
@group(0) @binding(3) var<storage, read_write> output      : array<f32>;

const EPS : f32 = 1e-6;
const PI  : f32 = 3.14159265358979;

// ── Complex helpers ────────────────────────────────────────────────────────────
fn cmul(a: vec2f, b: vec2f) -> vec2f {
    return vec2f(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}
fn cinv(v: vec2f) -> vec2f {
    let d = v.x*v.x + v.y*v.y + EPS;
    return vec2f(v.x/d, -v.y/d);
}
fn csin(v: vec2f) -> vec2f {
    return vec2f(sin(v.x)*cosh(v.y), cos(v.x)*sinh(v.y));
}
fn ccos(v: vec2f) -> vec2f {
    return vec2f(cos(v.x)*cosh(v.y), -sin(v.x)*sinh(v.y));
}
fn cexp(v: vec2f) -> vec2f {
    let e = exp(clamp(v.x, -8.0, 8.0));
    return vec2f(e*cos(v.y), e*sin(v.y));
}
fn clog(v: vec2f) -> vec2f {
    return vec2f(log(sqrt(v.x*v.x + v.y*v.y + EPS)), atan2(v.y, v.x));
}

// ── 58 basis functions (no divergence in caller loop) ─────────────────────────
fn eval_basis(i: u32, z: vec2f, c: vec2f) -> vec2f {
    let zx = z.x; let zy = z.y;
    let cx = c.x; let cy = c.y;
    switch i {
        // A: powers of z
        case  0u { return vec2f(zx*zx - zy*zy, 2.0*zx*zy); }
        case  1u { let a=zx*zx; let b=zy*zy; return vec2f(zx*(a-3.0*b), zy*(3.0*a-b)); }
        case  2u { let a=zx*zx; let b=zy*zy; return vec2f(a*a-6.0*a*b+b*b, 4.0*zx*zy*(a-b)); }
        case  3u { let a=zx*zx; let b=zy*zy; let z4=vec2f(a*a-6.0*a*b+b*b,4.0*zx*zy*(a-b)); return cmul(z4,z); }
        case  4u { return z; }
        case  5u { return cinv(z); }
        case  6u { let z2=vec2f(zx*zx-zy*zy,2.0*zx*zy); return cinv(z2); }
        // B: powers involving c
        case  7u { return c; }
        case  8u { return vec2f(cx*cx-cy*cy, 2.0*cx*cy); }
        case  9u { let a=cx*cx; let b=cy*cy; return vec2f(cx*(a-3.0*b), cy*(3.0*a-b)); }
        case 10u { return cmul(z,c); }
        case 11u { return cmul(vec2f(zx*zx-zy*zy,2.0*zx*zy), c); }
        case 12u { return cmul(z, vec2f(cx*cx-cy*cy,2.0*cx*cy)); }
        case 13u { return cmul(vec2f(zx*zx-zy*zy,2.0*zx*zy), vec2f(cx*cx-cy*cy,2.0*cx*cy)); }
        case 14u { return cmul(c, cinv(z)); }
        case 15u { let s=z+c; return vec2f(s.x*s.x-s.y*s.y, 2.0*s.x*s.y); }
        case 16u { let s=z-c; return vec2f(s.x*s.x-s.y*s.y, 2.0*s.x*s.y); }
        case 17u { let zc=cmul(z,c); return vec2f(zc.x*zc.x-zc.y*zc.y, 2.0*zc.x*zc.y); }
        // C: trig
        case 18u { return csin(z); }
        case 19u { return ccos(z); }
        case 20u { return csin(z*PI); }
        case 21u { return ccos(z*PI); }
        case 22u { let z2=vec2f(zx*zx-zy*zy,2.0*zx*zy); return csin(z2); }
        case 23u { let z2=vec2f(zx*zx-zy*zy,2.0*zx*zy); return ccos(z2); }
        case 24u { return csin(z+c); }
        case 25u { return ccos(z+c); }
        case 26u { return csin(cmul(z,c)); }
        case 27u { return ccos(cmul(z,c)); }
        case 28u { return cmul(z, csin(z)); }
        case 29u { return cmul(z, ccos(z)); }
        case 30u { // tan
            let s=csin(z); let cv=ccos(z);
            let d=cv.x*cv.x+cv.y*cv.y+EPS;
            return vec2f((s.x*cv.x+s.y*cv.y)/d, (s.y*cv.x-s.x*cv.y)/d);
        }
        case 31u { let e=cexp(z); let en=cexp(-z); return (e-en)*0.5; }
        case 32u { let e=cexp(z); let en=cexp(-z); return (e+en)*0.5; }
        case 33u { // tanh
            let x2=2.0*zx; let y2=2.0*zy;
            let d=cosh(x2)+cos(y2)+EPS;
            return vec2f(sinh(x2)/d, sin(y2)/d);
        }
        // D: exponential/log
        case 34u { return cexp(z); }
        case 35u { return cexp(-z); }
        case 36u { return cexp(cmul(vec2f(clamp(zx,-4.0,4.0),zy), c)); }
        case 37u { return cmul(z, cexp(vec2f(clamp(zx,-4.0,4.0),zy))); }
        case 38u { return cmul(cexp(vec2f(clamp(zx,-4.0,4.0),zy)), c); }
        case 39u { return clog(vec2f(zx+1.0, zy)); }
        case 40u { return clog(vec2f(zx*zx-zy*zy+1.0, 2.0*zx*zy)); }
        case 41u { return cmul(z, clog(vec2f(zx+1.0,zy))); }
        case 42u { let inv=cinv(z); return csin(clamp(inv,-vec2f(10.0),vec2f(10.0))); }
        case 43u { let inv=cinv(z); return cexp(vec2f(clamp(inv.x,-4.0,4.0), clamp(inv.y,-8.0,8.0))); }
        // E: non-holomorphic
        case 44u { return vec2f(abs(zx), zy); }
        case 45u { return vec2f(zx, abs(zy)); }
        case 46u { return abs(z); }
        case 47u { return vec2f(zx, -zy); }
        case 48u { return vec2f(zx*zx-zy*zy, -2.0*zx*zy); }
        case 49u { let m=length(z); return z*m; }
        case 50u { let m=length(z)+EPS; return z/m; }
        // F: rational
        case 51u { return cmul(z, cinv(vec2f(zx*zx-zy*zy+1.0, 2.0*zx*zy))); }
        case 52u {
            let z2=vec2f(zx*zx-zy*zy, 2.0*zx*zy);
            return cmul(z2-vec2f(1.0,0.0), cinv(z2+vec2f(1.0,0.0)));
        }
        case 53u {
            let z2=vec2f(zx*zx-zy*zy, 2.0*zx*zy);
            return cmul(z2, cinv(vec2f(zx-1.0, zy)));
        }
        case 54u {
            let z2=vec2f(zx*zx-zy*zy, 2.0*zx*zy);
            return cinv(z2+c);
        }
        case 55u {
            let z2=vec2f(zx*zx-zy*zy, 2.0*zx*zy);
            return cmul(cmul(z2,c), cinv(vec2f(zx+cx, zy+cy)));
        }
        // G: constants
        case 56u { return vec2f(1.0, 0.0); }
        case 57u { return vec2f(0.0, 1.0); }
        default  { return vec2f(0.0, 0.0); }
    }
}

fn apply_formula(z: vec2f, c: vec2f, fw_offset: u32) -> vec2f {
    var rx = 0.0f;
    var ry = 0.0f;
    for (var i = 0u; i < 58u; i++) {
        let base = fw_offset + i * 2u;
        let wr   = all_fw[base];
        let wi   = all_fw[base + 1u];
        let b    = eval_basis(i, z, c);
        rx += wr * b.x - wi * b.y;
        ry += wr * b.y + wi * b.x;
    }
    return vec2f(rx, ry);
}

// ── Expression-DAG register VM ─────────────────────────────────────────────────
// Mirrors eval_program() in formula.rs byte-for-byte. Program stored as up to
// N_SLOTS nodes of 5 f32s [op, a, b, kre, kim] at all_fw[prog_base..]. op==255
// terminates. Returns the last evaluated register (the root).
const N_SLOTS: u32 = 24u;
fn eval_program(z: vec2f, c: vec2f, prog_base: u32) -> vec2f {
    var reg: array<vec2f, 24>;
    var last: u32 = 0u;
    for (var i = 0u; i < N_SLOTS; i++) {
        let nb  = prog_base + i * 5u;
        let o   = u32(all_fw[nb]);
        if (o == 255u) { break; }
        let ai  = u32(all_fw[nb + 1u]);
        let bi  = u32(all_fw[nb + 2u]);
        let kre = all_fw[nb + 3u];
        let kim = all_fw[nb + 4u];
        var av = vec2f(0.0, 0.0);
        var bv = vec2f(0.0, 0.0);
        if (ai < i) { av = reg[ai]; }
        if (bi < i) { bv = reg[bi]; }
        var r = vec2f(0.0, 0.0);
        switch o {
            case 0u  { r = z; }                                                   // Z
            case 1u  { r = c; }                                                   // C
            case 2u  { r = vec2f(kre, kim); }                                     // CONST
            case 3u  { r = vec2f(av.x*av.x - av.y*av.y, 2.0*av.x*av.y); }         // SQR
            case 4u  { let a2=av.x*av.x; let b2=av.y*av.y; r = vec2f(av.x*(a2-3.0*b2), av.y*(3.0*a2-b2)); } // CUBE
            case 5u  { let a2=av.x*av.x; let b2=av.y*av.y; r = vec2f(a2*a2-6.0*a2*b2+b2*b2, 4.0*av.x*av.y*(a2-b2)); } // QUART
            case 6u  { r = cinv(av); }                                            // RECIP
            case 7u  { r = csin(av); }                                            // SIN
            case 8u  { r = ccos(av); }                                            // COS
            case 9u  { r = cexp(av); }                                            // EXP
            case 10u { r = clog(av); }                                            // LOG
            case 11u { let x2=2.0*av.x; let y2=2.0*av.y; let d=cosh(x2)+cos(y2)+EPS; r=vec2f(sinh(x2)/d, sin(y2)/d); } // TANH
            case 12u { r = vec2f(av.x, -av.y); }                                  // CONJ
            case 13u { r = vec2f(abs(av.x), abs(av.y)); }                         // ABSFOLD
            case 14u { r = vec2f(abs(av.x), av.y); }                              // ABSRE
            case 15u { r = vec2f(av.x, abs(av.y)); }                              // ABSIM
            case 16u { let m=sqrt(av.x*av.x+av.y*av.y)+EPS; r=vec2f(av.x/m, av.y/m); } // NORMZ
            case 17u { r = av + bv; }                                             // ADD
            case 18u { r = av - bv; }                                             // SUB
            case 19u { r = cmul(av, bv); }                                        // MUL
            case 20u { r = cmul(av, cinv(bv)); }                                  // DIV
            default  { r = vec2f(0.0, 0.0); }
        }
        reg[i] = r;
        last = i;
    }
    return reg[last];
}

// Dispatch: (ceil(w*h/256), 1, genome_count) — Z = genome index.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let pixel_idx  = gid.x;
    let genome_idx = gid.z;
    let pixels     = params.width * params.height;

    if pixel_idx >= pixels || genome_idx >= params.genome_count { return; }

    // Per-genome view bounds
    let vb     = genome_idx * 4u;
    let xmin   = view_bounds[vb];
    let xmax   = view_bounds[vb + 1u];
    let ymin   = view_bounds[vb + 2u];
    let ymax   = view_bounds[vb + 3u];

    let px  = pixel_idx % params.width;
    let py  = pixel_idx / params.width;
    let wf  = f32(max(params.width,  2u) - 1u);
    let hf  = f32(max(params.height, 2u) - 1u);
    let cx  = xmin + (f32(px) / wf) * (xmax - xmin);
    let cy  = ymin + (f32(py) / hf) * (ymax - ymin);
    let c   = vec2f(cx, cy);

    let out_offset = genome_idx * pixels;
    // Per-genome stride differs by mode: 116 legacy weights vs 120 program floats.
    let prog_base  = genome_idx * 120u;
    let fw_offset  = genome_idx * 116u;
    let dag        = params.use_dag != 0u;

    var z     = vec2f(0.0, 0.0);
    var etime = f32(params.max_iter);

    for (var iter = 0u; iter < params.max_iter; iter++) {
        if (dag) { z = eval_program(z, c, prog_base); }
        else     { z = apply_formula(z, c, fw_offset); }
        let ms = dot(z, z);
        if ms > params.bailout_sq {
            let nu = log2(log2(sqrt(ms) + 1e-10));
            etime = max(0.0, f32(iter) + 1.0 - nu);
            break;
        }
        if z.x != z.x || z.y != z.y || abs(z.x) > 1e15 || abs(z.y) > 1e15 {
            etime = f32(iter);
            break;
        }
    }
    output[out_offset + pixel_idx] = etime;
}
