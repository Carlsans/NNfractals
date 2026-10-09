// WGSL template for a PER-GENOME SPECIALIZED ray-march shader — the
// "compiled" twin of raymarch_dag.wgsl's general DAG interpreter.
//
// raymarch_dag.wgsl uploads a genome's program as DATA (a storage buffer
// of DagNodes) and interprets it with a runtime loop + opcode switch —
// correct, but ~9x slower than the hand-built QuatFormula GPU path
// (raymarch.wgsl) because of the dynamic branching and array indexing on
// every node of every march step. The hand-built path is fast precisely
// because each formula is fixed, unrolled Rust match arms — this template
// gets the same property for an ARBITRARY genome by generating the
// program's straight-line WGSL AT RENDER-SETUP TIME (once per genome, not
// once per frame) instead of interpreting it. Each marker comment inside
// eval_warp/eval_main_deriv below is replaced by
// render_gpu_raymarch_dag_codegen.rs's generate_shader_source with
// unrolled "let nI = ...;" chains — one per DAG node, in program order,
// each referencing only strictly-earlier locals (mirrors
// eval_program_quat/eval_program_quat_deriv's register-VM semantics
// exactly, just resolved at codegen time instead of at runtime).
//
// Everything else here — Quat helpers, assemble, quat_dag_escape_de,
// estimate_normal, march_sample, main — is copied verbatim from
// raymarch_dag.wgsl (already verified pixel-for-pixel against the CPU
// path); this file has no independent design beyond the two placeholders.

struct Params {
    width:              u32,
    height:             u32,
    max_iter:           u32,
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
    jc_im:              f32,
    phoenix_re:         f32,
    phoenix_im:         f32,
    eye_x: f32, eye_y: f32, eye_z: f32,
    fwd_x: f32, fwd_y: f32, fwd_z: f32,
    right_x: f32, right_y: f32, right_z: f32,
    up_x: f32, up_y: f32, up_z: f32,
    light_x: f32, light_y: f32, light_z: f32,
    aa: u32,
    // ── Animation-viewer plan, Phase 2: additive fields (were _pad0/_pad1
    // padding). `use_box`/`use_axis_perm` are 0 for every caller before
    // this plan (quat_viewer.rs, the CLI) — march_sample/assemble fall
    // back to the original domain_radius-sphere/time_axis behavior
    // exactly, byte-for-byte, when both are 0. ──────────────────────────
    use_box: u32,
    use_axis_perm: u32,
    box_min_x: f32, box_min_y: f32, box_min_z: f32,
    box_max_x: f32, box_max_y: f32, box_max_z: f32,
    axis_part_x: u32, axis_part_y: u32, axis_part_z: u32, axis_part_t: u32,
    // ── Animation-viewer plan, Phase 6: orientable difference/cutaway
    // clip plane. `clip_enabled` is 0 for every caller before this plan.
    clip_enabled: u32,
    clip_pos_x: f32, clip_pos_y: f32, clip_pos_z: f32,
    clip_normal_x: f32, clip_normal_y: f32, clip_normal_z: f32,
    // ── Slide Iteration effect: a hidden escape-iteration band — a point
    // whose escape time falls in [slide_iter_min, slide_iter_max) is never
    // accepted as a raymarch hit. `slide_iter_enabled` is 0 for every
    // caller before this effect.
    slide_iter_enabled: u32,
    slide_iter_min: f32, slide_iter_max: f32,
    // "hide top N%" resolved cutoff — a point whose escape time is >=
    // hide_above is never accepted as a raymarch hit. Resolved fresh per
    // render from an actual rendered frame's escape-time distribution
    // (see anim_eval::percentile_escape_time_cutoff); a fixed absolute
    // iteration count can't reliably target "whatever currently renders
    // white," since coloring is histogram-equalized against each frame's
    // own distribution. hide_above_enabled is 0 for every caller before
    // this control existed, and for every render's own probe pass.
    hide_above_enabled: u32,
    hide_above: f32,
}

@group(0) @binding(0) var<uniform>             params         : Params;
@group(0) @binding(1) var<storage, read_write> output_shading : array<f32>;
@group(0) @binding(2) var<storage, read_write> output_color   : array<f32>;

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

// ── Specialized (codegen'd) DAG evaluators — one straight-line WGSL
// statement per program node, program-order, no runtime loop or switch. ──

fn eval_warp(point: Quat) -> Quat {
// GENERATED-CODE-MARKER-WARP
}

struct ValDeriv { val: Quat, dval: f32 }

fn eval_main_deriv(z: Quat, c: Quat, dz: f32, dc: f32) -> ValDeriv {
// GENERATED-CODE-MARKER-MAIN-DERIV
}

// When params.use_axis_perm != 0, mirrors AxisAssignment::assemble exactly
// (a full permutation of R/A/B/C=0/1/2/3 across the 4 viewer axes). Else
// mirrors TimeAxis::assemble exactly (0=R 1=A 2=B 3=C) — every caller
// before the animation-viewer plan (quat_viewer.rs, the CLI) leaves
// use_axis_perm at 0.
fn assemble(x: f32, y: f32, z: f32, time_val: f32) -> Quat {
    if (params.use_axis_perm != 0u) {
        var vals = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
        let axis_vals = array<f32, 4>(x, y, z, time_val);
        vals[params.axis_part_x] = axis_vals[0];
        vals[params.axis_part_y] = axis_vals[1];
        vals[params.axis_part_z] = axis_vals[2];
        vals[params.axis_part_t] = axis_vals[3];
        return Quat(vals[0], vals[1], vals[2], vals[3]);
    }
    switch params.time_axis {
        case 0u: { return Quat(time_val, x, y, z); }
        case 1u: { return Quat(x, time_val, y, z); }
        case 2u: { return Quat(x, y, time_val, z); }
        default: { return Quat(x, y, z, time_val); }
    }
}

// Mirrors quat_dag_escape_de exactly, except `warped` is unconditional —
// the generated eval_warp already returns `point` unchanged when this
// genome's warp program is empty, so no runtime branch is needed here.
fn quat_dag_escape_de(point: Quat) -> vec2f {
    let warped = eval_warp(point);

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

// Returns (escape_time, distance_estimate) — the escape time is needed at
// the Slide Iteration gate in march_sample; estimate_normal below only
// ever uses the distance-estimate half.
fn de_at(p: vec3f) -> vec2f {
    let q = assemble(p.x, p.y, p.z, params.time_val);
    return quat_dag_escape_de(q);
}

// Tetrahedral 4-tap normal estimate — same technique as raymarch.wgsl.
fn estimate_normal(p: vec3f) -> vec3f {
    let eps = max(params.normal_eps, 1e-6);
    let k0 = vec3f(1.0, -1.0, -1.0);
    let k1 = vec3f(-1.0, -1.0, 1.0);
    let k2 = vec3f(-1.0, 1.0, -1.0);
    let k3 = vec3f(1.0, 1.0, 1.0);
    let g = k0 * de_at(p + k0*eps).y + k1 * de_at(p + k1*eps).y
          + k2 * de_at(p + k2*eps).y + k3 * de_at(p + k3*eps).y;
    return normalize(g);
}

fn march_sample(u: f32, v: f32) -> vec2f {
    let forward = vec3f(params.fwd_x, params.fwd_y, params.fwd_z);
    let right   = vec3f(params.right_x, params.right_y, params.right_z);
    let up      = vec3f(params.up_x, params.up_y, params.up_z);
    let eye     = vec3f(params.eye_x, params.eye_y, params.eye_z);
    let dir = normalize(forward + right*u + up*v);

    var t0: f32;
    var t1: f32;
    if (params.use_box != 0u) {
        // Slab method — mirrors quat_raymarch::ray_box exactly.
        let dir_a = array<f32, 3>(dir.x, dir.y, dir.z);
        let eye_a = array<f32, 3>(eye.x, eye.y, eye.z);
        let lo_a = array<f32, 3>(params.box_min_x, params.box_min_y, params.box_min_z);
        let hi_a = array<f32, 3>(params.box_max_x, params.box_max_y, params.box_max_z);
        var tn: f32 = 0.0;
        var tf: f32 = 3.4e38;
        var missed = false;
        for (var axis: u32 = 0u; axis < 3u; axis = axis + 1u) {
            let d = dir_a[axis];
            let e = eye_a[axis];
            let lo = lo_a[axis];
            let hi = hi_a[axis];
            if (abs(d) < 1e-30) {
                if (e < lo || e > hi) { missed = true; }
            } else {
                let inv = 1.0 / d;
                var ta = (lo - e) * inv;
                var tb = (hi - e) * inv;
                if (ta > tb) { let tmp = ta; ta = tb; tb = tmp; }
                tn = max(tn, ta);
                tf = min(tf, tb);
                if (tn > tf) { missed = true; }
            }
        }
        if (missed) { return vec2f(0.0, 0.0); }
        t0 = max(tn, 0.0);
        t1 = tf;
    } else {
        let a = dot(dir, dir);
        let b = 2.0 * dot(eye, dir);
        let cc = dot(eye, eye) - params.domain_radius * params.domain_radius;
        let disc = b*b - 4.0*a*cc;
        if (disc < 0.0) { return vec2f(0.0, 0.0); }
        let sq = sqrt(disc);
        let tt0 = (-b - sq) / (2.0*a);
        let tt1 = (-b + sq) / (2.0*a);
        if (tt1 < 0.0) { return vec2f(0.0, 0.0); }
        t0 = max(tt0, 0.0);
        t1 = tt1;
    }

    // Orientable difference/cutaway clip plane (Phase 6) — narrows [t0,t1]
    // once per ray, mirrors quat_dag.rs's apply_clip_plane exactly. Ray
    // interval narrowing, not a distance-field modification: the escape-
    // time distance estimate has no inside/outside sign convention, so the
    // usual signed-SDF CSG trick doesn't apply here.
    if (params.clip_enabled != 0u) {
        let clip_pos = vec3f(params.clip_pos_x, params.clip_pos_y, params.clip_pos_z);
        let clip_n = vec3f(params.clip_normal_x, params.clip_normal_y, params.clip_normal_z);
        let denom = dot(dir, clip_n);
        if (denom > 1e-6) {
            let t_plane = dot(clip_pos - eye, clip_n) / denom;
            t0 = max(t0, t_plane);
        } else if (denom < -1e-6) {
            let t_plane = dot(clip_pos - eye, clip_n) / denom;
            t1 = min(t1, t_plane);
        } else if (dot(eye - clip_pos, clip_n) < 0.0) {
            return vec2f(0.0, 0.0);
        }
        if (t0 > t1) { return vec2f(0.0, 0.0); }
    }

    let hit_eps = max(params.hit_epsilon, 1e-9);
    let min_step = max(params.domain_radius * 1e-6, 1e-9);
    var t = t0;
    var hit = false;
    var hit_point = vec3f(0.0, 0.0, 0.0);
    for (var i: u32 = 0u; i < params.max_march_steps; i = i + 1u) {
        if (t > t1) { break; }
        let p = eye + dir * t;
        let r = de_at(p);
        let et = r.x;
        let de = r.y;
        // Slide Iteration / "hide top N%": a point in the hidden band, or
        // at/past the resolved hide_above cutoff, is never accepted as a
        // hit, even when its DE is small enough — the real DE still
        // drives the step below, so marching through stays adaptively
        // efficient.
        let in_band = params.slide_iter_enabled != 0u && et >= params.slide_iter_min && et < params.slide_iter_max;
        let above_cutoff = params.hide_above_enabled != 0u && et >= params.hide_above;
        if (de < hit_eps && !in_band && !above_cutoff) {
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
    let probe_q = assemble(probe.x, probe.y, probe.z, params.time_val);
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
