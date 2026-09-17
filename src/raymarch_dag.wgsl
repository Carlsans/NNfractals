// WGSL compute shader: quaternion ray-marching for an ARBITRARY GA-evolved
// genome's expression-DAG (not a fixed formula switch — see raymarch.wgsl
// for that). Mirrors src/quat_dag.rs's eval_program_quat/
// eval_program_quat_deriv/quat_dag_escape_de EXACTLY — this file has no
// independent design, it's a straight port, validated against the CPU
// path pixel-for-pixel (see render_gpu_raymarch_dag.rs's tests).

struct Params {
    width:              u32,
    height:             u32,
    max_iter:           u32,
    prog_len:           u32,
    warp_len:           u32,
    time_axis:          u32,
    max_march_steps:    u32,
    julia:              u32,
    domain_radius:      f32,
    time_val:           f32,
    bailout_sq:         f32,
    hit_epsilon:        f32,
    step_safety:        f32,
    normal_eps:         f32,
    color_probe_offset: f32,
    half_w:             f32,
    half_h:             f32,
    jc_re:              f32,
    jc_im:               f32,
    phoenix_re:         f32,
    phoenix_im:         f32,
    eye_x: f32, eye_y: f32, eye_z: f32,
    fwd_x: f32, fwd_y: f32, fwd_z: f32,
    right_x: f32, right_y: f32, right_z: f32,
    up_x: f32, up_y: f32, up_z: f32,
    light_x: f32, light_y: f32, light_z: f32,
    aa: u32,
    _pad0: u32,
    _pad1: u32,
}

struct DagNode { op: u32, a: u32, b: u32, kre: f32, kim: f32 }

@group(0) @binding(0) var<uniform>             params         : Params;
@group(0) @binding(1) var<storage, read>       prog           : array<DagNode>;
@group(0) @binding(2) var<storage, read>       warp_prog      : array<DagNode>;
@group(0) @binding(3) var<storage, read_write> output_shading : array<f32>;
@group(0) @binding(4) var<storage, read_write> output_color   : array<f32>;

const N_SLOTS: u32 = 24u;
const EPS: f32 = 1e-9;

// ── Quaternion helpers — mirror quat_dag.rs's q*/formula.rs's c* exactly ──
struct Quat { r: f32, a: f32, b: f32, c: f32 }

fn qmul(p: Quat, q: Quat) -> Quat {
    return Quat(
        p.r*q.r - p.a*q.a - p.b*q.b - p.c*q.c,
        p.r*q.a + p.a*q.r + p.b*q.c - p.c*q.b,
        p.r*q.b - p.a*q.c + p.b*q.r + p.c*q.a,
        p.r*q.c + p.a*q.b - p.b*q.a + p.c*q.r,
    );
}
fn qadd(p: Quat, q: Quat) -> Quat { return Quat(p.r+q.r, p.a+q.a, p.b+q.b, p.c+q.c); }
fn qsub(p: Quat, q: Quat) -> Quat { return Quat(p.r-q.r, p.a-q.a, p.b-q.b, p.c-q.c); }
fn qnorm_sq(q: Quat) -> f32 { return q.r*q.r + q.a*q.a + q.b*q.b + q.c*q.c; }
fn qnorm(q: Quat) -> f32 { return sqrt(qnorm_sq(q)); }
fn qconj(q: Quat) -> Quat { return Quat(q.r, -q.a, -q.b, -q.c); }
fn qabs_components(q: Quat) -> Quat { return Quat(abs(q.r), abs(q.a), abs(q.b), abs(q.c)); }
fn vec_norm(q: Quat) -> f32 { return sqrt(q.a*q.a + q.b*q.b + q.c*q.c); }
fn qfinite(q: Quat) -> bool {
    return !(any(vec4<bool>(
        !(q.r == q.r) || abs(q.r) > 3.4e38,
        !(q.a == q.a) || abs(q.a) > 3.4e38,
        !(q.b == q.b) || abs(q.b) > 3.4e38,
        !(q.c == q.c) || abs(q.c) > 3.4e38,
    )));
}

fn qexp(q: Quat) -> Quat {
    let rho = vec_norm(q);
    let r = clamp(q.r, -8.0, 8.0);
    let e = exp(r);
    if (rho < EPS) { return Quat(e, 0.0, 0.0, 0.0); }
    let s = e * sin(rho) / rho;
    return Quat(e*cos(rho), q.a*s, q.b*s, q.c*s);
}
fn qlog(q: Quat) -> Quat {
    let rho = vec_norm(q);
    let norm = sqrt(q.r*q.r + rho*rho) + EPS;
    let theta = atan2(rho, q.r);
    if (rho < EPS) { return Quat(log(norm), 0.0, 0.0, 0.0); }
    let s = theta / rho;
    return Quat(log(norm), q.a*s, q.b*s, q.c*s);
}
fn qsin(q: Quat) -> Quat {
    let rho = vec_norm(q);
    let re = sin(q.r) * cosh(rho);
    if (rho < EPS) { return Quat(re, 0.0, 0.0, 0.0); }
    let s = cos(q.r) * sinh(rho) / rho;
    return Quat(re, q.a*s, q.b*s, q.c*s);
}
fn qcos(q: Quat) -> Quat {
    let rho = vec_norm(q);
    let re = cos(q.r) * cosh(rho);
    if (rho < EPS) { return Quat(re, 0.0, 0.0, 0.0); }
    let s = -sin(q.r) * sinh(rho) / rho;
    return Quat(re, q.a*s, q.b*s, q.c*s);
}
fn qtanh(q: Quat) -> Quat {
    let rho = vec_norm(q);
    let x2 = 2.0*q.r;
    let y2 = 2.0*rho;
    let d = cosh(x2) + cos(y2) + EPS;
    let re = sinh(x2)/d;
    if (rho < EPS) { return Quat(re, 0.0, 0.0, 0.0); }
    let s = (sin(y2)/d) / rho;
    return Quat(re, q.a*s, q.b*s, q.c*s);
}
fn qrecip(q: Quat) -> Quat {
    let d = qnorm_sq(q) + EPS;
    let cj = qconj(q);
    return Quat(cj.r/d, cj.a/d, cj.b/d, cj.c/d);
}
fn qnormz(q: Quat) -> Quat {
    let m = qnorm(q) + EPS;
    return Quat(q.r/m, q.a/m, q.b/m, q.c/m);
}

// ── DAG evaluators — mirror quat_dag.rs's eval_program_quat /
// eval_program_quat_deriv register-VMs exactly, opcode for opcode. ──

fn eval_warp(point: Quat) -> Quat {
    let n = min(params.warp_len, N_SLOTS);
    if (n == 0u) { return point; }
    var reg: array<Quat, 24>;
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        let node = warp_prog[i];
        let ai = min(node.a, N_SLOTS - 1u);
        let bi = min(node.b, N_SLOTS - 1u);
        var a = Quat(0.0, 0.0, 0.0, 0.0);
        var b = Quat(0.0, 0.0, 0.0, 0.0);
        if (ai < i) { a = reg[ai]; }
        if (bi < i) { b = reg[bi]; }
        var val: Quat;
        switch node.op {
            case 0u: { val = point; }       // Z (warp reads point for both z and c)
            case 1u: { val = point; }       // C
            case 2u: { val = Quat(node.kre, node.kim, 0.0, 0.0); }
            case 3u: { val = qmul(a, a); }
            case 4u: { val = qmul(qmul(a, a), a); }
            case 5u: { let sq = qmul(a, a); val = qmul(sq, sq); }
            case 6u: { val = qrecip(a); }
            case 7u: { val = qsin(a); }
            case 8u: { val = qcos(a); }
            case 9u: { val = qexp(a); }
            case 10u: { val = qlog(a); }
            case 11u: { val = qtanh(a); }
            case 12u: { val = qconj(a); }
            case 13u: { val = qabs_components(a); }
            case 14u: { val = Quat(abs(a.r), a.a, a.b, a.c); }
            case 15u: { val = Quat(a.r, abs(a.a), abs(a.b), abs(a.c)); }
            case 16u: { val = qnormz(a); }
            case 17u: { val = qadd(a, b); }
            case 18u: { val = qsub(a, b); }
            case 19u: { val = qmul(a, b); }
            case 20u: { val = qmul(a, qrecip(b)); }
            default: { val = Quat(0.0, 0.0, 0.0, 0.0); }
        }
        reg[i] = val;
    }
    return reg[n - 1u];
}

struct ValDeriv { val: Quat, dval: f32 }

fn eval_main_deriv(z: Quat, c: Quat, dz: f32, dc: f32) -> ValDeriv {
    let n = min(params.prog_len, N_SLOTS);
    if (n == 0u) { return ValDeriv(Quat(0.0,0.0,0.0,0.0), 0.0); }
    var reg: array<Quat, 24>;
    var dreg: array<f32, 24>;
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        let node = prog[i];
        let ai = min(node.a, N_SLOTS - 1u);
        let bi = min(node.b, N_SLOTS - 1u);
        var a = Quat(0.0, 0.0, 0.0, 0.0);
        var da = 0.0;
        var b = Quat(0.0, 0.0, 0.0, 0.0);
        var db = 0.0;
        if (ai < i) { a = reg[ai]; da = dreg[ai]; }
        if (bi < i) { b = reg[bi]; db = dreg[bi]; }
        let na = qnorm(a);
        let nb = qnorm(b);
        var val: Quat;
        var dval: f32;
        switch node.op {
            case 0u: { val = z; dval = dz; }
            case 1u: { val = c; dval = dc; }
            case 2u: { val = Quat(node.kre, node.kim, 0.0, 0.0); dval = 0.0; }
            case 3u: { val = qmul(a, a); dval = 2.0 * na * da; }
            case 4u: { val = qmul(qmul(a, a), a); dval = 3.0 * na * na * da; }
            case 5u: { let sq = qmul(a, a); val = qmul(sq, sq); dval = 4.0 * na*na*na * da; }
            case 6u: { val = qrecip(a); dval = da / (na*na + EPS); }
            case 7u: { val = qsin(a); dval = qnorm(qcos(a)) * da; }
            case 8u: { val = qcos(a); dval = qnorm(qsin(a)) * da; }
            case 9u: { let e = qexp(a); val = e; dval = qnorm(e) * da; }
            case 10u: { val = qlog(a); dval = da / (na + EPS); }
            case 11u: { let t = qtanh(a); let tn = qnorm(t); val = t; dval = abs(1.0 - tn*tn) * da; }
            case 12u: { val = qconj(a); dval = da; }
            case 13u: { val = qabs_components(a); dval = da; }
            case 14u: { val = Quat(abs(a.r), a.a, a.b, a.c); dval = da; }
            case 15u: { val = Quat(a.r, abs(a.a), abs(a.b), abs(a.c)); dval = da; }
            case 16u: { val = qnormz(a); dval = da / (na + EPS); }
            case 17u: { val = qadd(a, b); dval = da + db; }
            case 18u: { val = qsub(a, b); dval = da + db; }
            case 19u: { val = qmul(a, b); dval = na*db + nb*da; }
            case 20u: {
                let rb = qrecip(b);
                let d_rb = db / (nb*nb + EPS);
                val = qmul(a, rb);
                dval = da / (nb + EPS) + na * d_rb;
            }
            default: { val = Quat(0.0, 0.0, 0.0, 0.0); dval = 0.0; }
        }
        reg[i] = val;
        dreg[i] = dval;
    }
    return ValDeriv(reg[n - 1u], dreg[n - 1u]);
}

// Mirrors TimeAxis::assemble exactly (0=R 1=A 2=B 3=C).
fn assemble(time_axis: u32, x: f32, y: f32, z: f32, time_val: f32) -> Quat {
    switch time_axis {
        case 0u: { return Quat(time_val, x, y, z); }
        case 1u: { return Quat(x, time_val, y, z); }
        case 2u: { return Quat(x, y, time_val, z); }
        default: { return Quat(x, y, z, time_val); }
    }
}

// Mirrors quat_dag_escape_de exactly.
fn quat_dag_escape_de(point: Quat) -> vec2f {
    var warped = point;
    if (params.warp_len > 0u) { warped = eval_warp(point); }

    var z: Quat;
    var c: Quat;
    var dz: f32;
    var dc: f32;
    if (params.julia != 0u) {
        z = warped;
        c = Quat(params.jc_re, params.jc_im, 0.0, 0.0);
        dz = 1.0; dc = 0.0;
    } else {
        z = Quat(0.0, 0.0, 0.0, 0.0);
        c = warped;
        dz = 0.0; dc = 1.0;
    }
    let phoenix_q = Quat(params.phoenix_re, params.phoenix_im, 0.0, 0.0);
    let phoenix_mag = qnorm(phoenix_q);
    var pz = Quat(0.0, 0.0, 0.0, 0.0);
    var dpz: f32 = 0.0;

    for (var it: u32 = 0u; it < params.max_iter; it = it + 1u) {
        let fd = eval_main_deriv(z, c, dz, dc);
        let dnext = fd.dval + phoenix_mag * dpz;
        let next = qadd(fd.val, qmul(phoenix_q, pz));
        dpz = dz;
        pz = z;
        dz = dnext;
        z = next;

        let ms = qnorm_sq(z);
        if (ms > params.bailout_sq) {
            let et = max(f32(it) + 1.0 - log2(log2(ms) * 0.5), 0.0);
            let r = sqrt(ms);
            let de = max(0.5 * log(r) * r / max(dz, 1e-30), 0.0);
            return vec2f(et, de);
        }
        if (!qfinite(z)) {
            return vec2f(f32(it), 0.0);
        }
    }
    let r = max(qnorm(z), 1e-30);
    let de = max(0.5 * log(r) * r / max(dz, 1e-30), 0.0);
    return vec2f(f32(params.max_iter), de);
}

fn de_at(p: vec3f) -> f32 {
    let q = assemble(params.time_axis, p.x, p.y, p.z, params.time_val);
    return quat_dag_escape_de(q).y;
}

// Tetrahedral 4-tap normal estimate — same technique as raymarch.wgsl.
fn estimate_normal(p: vec3f) -> vec3f {
    let eps = max(params.normal_eps, 1e-6);
    let k0 = vec3f(1.0, -1.0, -1.0);
    let k1 = vec3f(-1.0, -1.0, 1.0);
    let k2 = vec3f(-1.0, 1.0, -1.0);
    let k3 = vec3f(1.0, 1.0, 1.0);
    let g = k0 * de_at(p + k0*eps) + k1 * de_at(p + k1*eps)
          + k2 * de_at(p + k2*eps) + k3 * de_at(p + k3*eps);
    return normalize(g);
}

fn march_sample(u: f32, v: f32) -> vec2f {
    let forward = vec3f(params.fwd_x, params.fwd_y, params.fwd_z);
    let right   = vec3f(params.right_x, params.right_y, params.right_z);
    let up      = vec3f(params.up_x, params.up_y, params.up_z);
    let eye     = vec3f(params.eye_x, params.eye_y, params.eye_z);
    let dir = normalize(forward + right*u + up*v);

    let a = dot(dir, dir);
    let b = 2.0 * dot(eye, dir);
    let cc = dot(eye, eye) - params.domain_radius * params.domain_radius;
    let disc = b*b - 4.0*a*cc;
    if (disc < 0.0) { return vec2f(0.0, 0.0); }
    let sq = sqrt(disc);
    var t0 = (-b - sq) / (2.0*a);
    let t1 = (-b + sq) / (2.0*a);
    if (t1 < 0.0) { return vec2f(0.0, 0.0); }
    t0 = max(t0, 0.0);

    let hit_eps = max(params.hit_epsilon, 1e-9);
    let min_step = max(params.domain_radius * 1e-6, 1e-9);
    var t = t0;
    var hit = false;
    var hit_point = vec3f(0.0, 0.0, 0.0);
    for (var i: u32 = 0u; i < params.max_march_steps; i = i + 1u) {
        if (t > t1) { break; }
        let p = eye + dir * t;
        let de = de_at(p);
        if (de < hit_eps) {
            hit = true;
            hit_point = p;
            break;
        }
        t = t + max(de * params.step_safety, min_step);
    }
    if (!hit) { return vec2f(0.0, 0.0); }

    let normal = estimate_normal(hit_point);
    let light = normalize(vec3f(params.light_x, params.light_y, params.light_z));
    let ndotl = max(dot(normal, light), 0.0);
    let shading = 0.15 + 0.85 * ndotl;

    let probe = hit_point + normal * params.color_probe_offset;
    let probe_q = assemble(params.time_axis, probe.x, probe.y, probe.z, params.time_val);
    let color_et = quat_dag_escape_de(probe_q).x;

    return vec2f(shading, color_et);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }
    let idx = gid.y * params.width + gid.x;

    let wf = f32(params.width);
    let hf = f32(params.height);
    let aa = max(params.aa, 1u);
    let aa_f = f32(aa);

    var sum = vec2f(0.0, 0.0);
    for (var sy: u32 = 0u; sy < aa; sy = sy + 1u) {
        for (var sx: u32 = 0u; sx < aa; sx = sx + 1u) {
            let jx = (f32(sx) + 0.5) / aa_f;
            let jy = (f32(sy) + 0.5) / aa_f;
            let u = ((f32(gid.x) + jx) / wf * 2.0 - 1.0) * params.half_w;
            let v = (1.0 - (f32(gid.y) + jy) / hf * 2.0) * params.half_h;
            sum = sum + march_sample(u, v);
        }
    }
    let result = sum / (aa_f * aa_f);
    output_shading[idx] = result.x;
    output_color[idx] = result.y;
}
