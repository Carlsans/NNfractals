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
    // Per-axis stretch between marched (view) space and fractal space
    // (RaymarchDagParams::axis_remap): fractal = off + sc * view.
    // remap_enabled is 0 for every caller that doesn't set it.
    remap_enabled: u32,
    remap_sc_x: f32, remap_sc_y: f32, remap_sc_z: f32,
    remap_off_x: f32, remap_off_y: f32, remap_off_z: f32,
    // 4D->3D collapse method (quat_collapse::Method) and its parameters.
    col_mode: u32,
    col_p0: f32, col_p1: f32, col_p2: f32, col_p3: f32, col_p4: f32,
    // Zone filter (RaymarchDagParams::zone, Slide/hide already taken out):
    // solid where the escape time is in one of zone_n ranges
    // [zone_lo_i, zone_hi_i). zone_enabled is 0 for every caller without one.
    zone_enabled: u32,
    zone_n: u32,
    zone_max_step: f32,
    zone_min_step: f32,
    zone_de_trust: f32,
    zone_lo0: f32, zone_lo1: f32, zone_lo2: f32, zone_lo3: f32,
    zone_lo4: f32, zone_lo5: f32, zone_lo6: f32, zone_lo7: f32,
    zone_hi0: f32, zone_hi1: f32, zone_hi2: f32, zone_hi3: f32,
    zone_hi4: f32, zone_hi5: f32, zone_hi6: f32, zone_hi7: f32,
    zone_speckle_len: f32,
    pixel_lod: f32, solid_interior: u32,
}

@group(0) @binding(0) var<uniform>             params         : Params;
@group(0) @binding(1) var<storage, read_write> output_shading : array<f32>;
@group(0) @binding(2) var<storage, read_write> output_color   : array<f32>;
// DAG constants, uploaded per render instead of baked into the generated
// code, so editing a constant (the anim viewer's live formula editor) needs
// no shader rebuild. Entry i: .xy = program node i's (kre, kim), .zw = warp
// node i's. 24 = formula::N_SLOTS.
@group(0) @binding(3) var<uniform>             consts         : array<vec4<f32>, 24>;

const EPS: f32 = 1e-9;
const ZONE_BISECT: u32 = 10u; // quat_dag::ZONE_BISECT
const ZONE_SPECKLE_SAMPLES: u32 = 6u; // quat_dag::ZONE_SPECKLE_*
const ZONE_SPECKLE_DEPTH: u32 = 5u;
const ZONE_SPECKLE_NEED: u32 = 5u;

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

// Generalised Mandelbulb power (op BULB) — mirrors quat_dag::qbulb exactly.
fn safe_atan2(y: f32, x: f32) -> f32 {
    if (abs(x) + abs(y) < 1e-30) { return 0.0; }
    return atan2(y, x);
}
fn qbulb(q: Quat, p: Quat) -> Quat {
    let nr = clamp(p.r, 1.5, 10.0);
    let na = clamp(p.a, 1.0, 10.0);
    let v = sqrt(q.a*q.a + q.b*q.b + q.c*q.c);
    let rho = sqrt(q.r*q.r + v*v);
    let t1 = na * safe_atan2(v, q.r) + p.b;
    let t2 = na * safe_atan2(sqrt(q.a*q.a + q.b*q.b), q.c);
    let ph = na * safe_atan2(q.b, q.a) + p.c;
    let rn = pow(max(rho, 1e-30), nr);
    let s1 = rn * sin(t1);
    return Quat(rn * cos(t1), s1 * sin(t2) * cos(ph), s1 * sin(t2) * sin(ph), s1 * cos(t2));
}
fn qbulb_deriv(q: Quat, p: Quat, da: f32, db: f32) -> f32 {
    let nr = clamp(p.r, 1.5, 10.0);
    let na = clamp(p.a, 1.0, 10.0);
    let rho = max(qnorm(q), 1e-30);
    let rn = pow(rho, nr);
    return max(nr, na) * pow(rho, nr - 1.0) * da + rn * (abs(log(rho)) + 1.0) * db;
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
fn assemble(vx: f32, vy: f32, vz: f32, time_val: f32) -> Quat {
    var x = vx; var y = vy; var z = vz;
    if (params.remap_enabled != 0u) {
        x = params.remap_off_x + params.remap_sc_x * vx;
        y = params.remap_off_y + params.remap_sc_y * vy;
        z = params.remap_off_z + params.remap_sc_z * vz;
    }
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
        if (params.col_mode == 5u) { z.c = 0.0; }

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
fn rot2(a: f32, b: f32, ang: f32) -> vec2f {
    let cs = cos(ang); let sn = sin(ang);
    return vec2f(a*cs - b*sn, a*sn + b*cs);
}

// Isoclinic rotation of a 4D point by t (CPU twin: quat_collapse::iso_rotate).
fn iso_rot(p: vec4f, plane: f32, t: f32) -> vec4f {
    let s = sin(t); let c = cos(t);
    if (plane < 0.5) {
        return vec4f(p.x*c - p.w*s, p.y*c - p.z*s, p.y*s + p.z*c, p.x*s + p.w*c);
    } else if (plane < 1.5) {
        return vec4f(p.x*c - p.z*s, p.y*c - p.w*s, p.x*s + p.z*c, p.y*s + p.w*c);
    }
    return vec4f(p.x*c - p.y*s, p.x*s + p.y*c, p.z*c - p.w*s, p.z*s + p.w*c);
}

// One evaluation of the 4D set at view point p with time value t, through
// the active collapse method's point mapping.
fn de_one(p: vec3f, t: f32) -> vec2f {
    var x = p.x; var y = p.y; var z = p.z; var w = t;
    var lam: f32 = 1.0;
    if (params.col_mode == 1u) {
        let r1 = rot2(z, w, params.col_p0); z = r1.x; w = r1.y;
        let r2 = rot2(x, w, params.col_p1); x = r2.x; w = r2.y;
    } else if (params.col_mode == 6u) {
        // Spin: the flat slice at distance col_p0, turned by t in the plane chosen by col_p1.
        w = params.col_p0;
        if (params.col_p1 < 0.5) {
            let r = rot2(x, w, t); x = r.x; w = r.y;
        } else if (params.col_p1 < 1.5) {
            let r = rot2(y, w, t); y = r.x; w = r.y;
        } else {
            let r = rot2(z, w, t); z = r.x; w = r.y;
        }
    } else if (params.col_mode == 7u) {
        // IsoSpin: the same slice, turned isoclinically.
        let q = iso_rot(vec4f(x, y, z, params.col_p0), params.col_p1, t);
        x = q.x; y = q.y; z = q.z; w = q.w;
    } else if (params.col_mode == 8u) {
        // IsoS3: the 3-sphere (as S3) with the isoclinic time rotation.
        let rho = params.col_p0;
        let s2 = x*x + y*y + z*z;
        let k = 1.0 / (1.0 + s2);
        var X = 2.0*rho*k*x; var Y = 2.0*rho*k*y; var Z = 2.0*rho*k*z;
        var W = rho*(s2 - 1.0)*k;
        let r0 = rot2(Z, W, params.col_p4); Z = r0.x; W = r0.y;
        let r1 = rot2(Y, W, params.col_p3); Y = r1.x; W = r1.y;
        let r2 = rot2(X, W, params.col_p2); X = r2.x; W = r2.y;
        let q = iso_rot(vec4f(X, Y, Z, W), params.col_p1, t);
        x = q.x; y = q.y; z = q.z; w = q.w;
        lam = 2.0*rho*k;
    } else if (params.col_mode == 4u) {
        // 3-sphere: inverse stereographic projection, then the 4D pole tilts
        // (col_p2 in xw, col_p3 in yw, col_p4 in zw) and the time rotation in the plane
        // chosen by col_p1 (0 = xw, 1 = yw, 2 = zw).
        let rho = params.col_p0;
        let s2 = x*x + y*y + z*z;
        let k = 1.0 / (1.0 + s2);
        var X = 2.0*rho*k*x; var Y = 2.0*rho*k*y; var Z = 2.0*rho*k*z;
        var W = rho*(s2 - 1.0)*k;
        let r0 = rot2(Z, W, params.col_p4); Z = r0.x; W = r0.y;
        let r1 = rot2(Y, W, params.col_p3); Y = r1.x; W = r1.y;
        let r2 = rot2(X, W, params.col_p2); X = r2.x; W = r2.y;
        if (params.col_p1 < 0.5) {
            let r = rot2(X, W, t); X = r.x; W = r.y;
        } else if (params.col_p1 < 1.5) {
            let r = rot2(Y, W, t); Y = r.x; W = r.y;
        } else {
            let r = rot2(Z, W, t); Z = r.x; W = r.y;
        }
        x = X; y = Y; z = Z; w = W;
        lam = 2.0*rho*k;
    }
    let q = assemble(x, y, z, w);
    let r = quat_dag_escape_de(q);
    return vec2f(r.x, r.y / lam);
}

// Returns (escape_time, distance_estimate) — the escape time is needed at
// the Slide Iteration gate in march_sample; estimate_normal below only
// ever uses the distance-estimate half.
fn de_at(p: vec3f) -> vec2f {
    var r: vec2f;
    if (params.col_mode == 2u) {
        let h = params.col_p0;
        r = de_one(p, params.time_val - h);
        let b = de_one(p, params.time_val);
        if (b.y < r.y) { r = b; }
        let c = de_one(p, params.time_val + h);
        if (c.y < r.y) { r = c; }
    } else {
        r = de_one(p, params.time_val);
    }
    if (params.col_mode == 3u) { r.y = r.y - params.col_p0; }
    if (params.remap_enabled != 0u) {
        // A view-space step of d moves at most d * max(scale) in fractal space.
        let smax = max(max(params.remap_sc_x, params.remap_sc_y), max(params.remap_sc_z, 1e-12));
        return vec2f(r.x, r.y / smax);
    }
    return r;
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

// ── Zone filter — mirrors quat_dag.rs's ZoneParams / march_zone ──────────
fn zone_lo(i: u32) -> f32 {
    switch i {
        case 0u: { return params.zone_lo0; }
        case 1u: { return params.zone_lo1; }
        case 2u: { return params.zone_lo2; }
        case 3u: { return params.zone_lo3; }
        case 4u: { return params.zone_lo4; }
        case 5u: { return params.zone_lo5; }
        case 6u: { return params.zone_lo6; }
        default: { return params.zone_lo7; }
    }
}
fn zone_hi(i: u32) -> f32 {
    switch i {
        case 0u: { return params.zone_hi0; }
        case 1u: { return params.zone_hi1; }
        case 2u: { return params.zone_hi2; }
        case 3u: { return params.zone_hi3; }
        case 4u: { return params.zone_hi4; }
        case 5u: { return params.zone_hi5; }
        case 6u: { return params.zone_hi6; }
        default: { return params.zone_hi7; }
    }
}
fn zone_contains(et: f32) -> bool {
    for (var i: u32 = 0u; i < params.zone_n; i = i + 1u) {
        if (et >= zone_lo(i) && et < zone_hi(i)) { return true; }
    }
    return false;
}
// ZoneParams::step_bound: de * (1 - 2^-gap), gap = escape-time distance to
// the nearest range end; max_step where there is no usable estimate.
fn zone_step(et: f32, de: f32) -> f32 {
    if (et >= f32(params.max_iter) || !(de > 0.0) || de > 3.0e38 || params.zone_de_trust <= 0.0) {
        return params.zone_max_step;
    }
    var gap: f32 = 3.4e38;
    for (var i: u32 = 0u; i < params.zone_n; i = i + 1u) {
        let lo = zone_lo(i);
        let hi = zone_hi(i);
        if (lo < 1.0e38) { gap = min(gap, abs(et - lo)); }
        if (hi < 1.0e38) { gap = min(gap, abs(et - hi)); }
    }
    if (gap >= 3.0e38) { return params.zone_max_step; }
    return min(params.zone_de_trust * de * (1.0 - exp2(-gap)), params.zone_max_step);
}
// Escape-time gradient turned to face the camera (zone_normal on the CPU).
fn zone_normal(p: vec3f, dir: vec3f) -> vec3f {
    let eps = max(params.normal_eps, 1e-6);
    let k0 = vec3f(1.0, -1.0, -1.0);
    let k1 = vec3f(-1.0, -1.0, 1.0);
    let k2 = vec3f(-1.0, 1.0, -1.0);
    let k3 = vec3f(1.0, 1.0, 1.0);
    let cap = f32(params.max_iter);
    let g = k0 * min(de_at(p + k0*eps).x, cap) + k1 * min(de_at(p + k1*eps).x, cap)
          + k2 * min(de_at(p + k2*eps).x, cap) + k3 * min(de_at(p + k3*eps).x, cap);
    var n: vec3f;
    if (length(g) > 1e-9) {
        n = normalize(g);
    } else {
        n = estimate_normal(p);
        if (!(dot(n, n) > 0.5)) { n = -dir; }
    }
    if (dot(n, dir) > 0.0) { n = -n; }
    return n;
}
fn zone_shade(n: vec3f, et: f32) -> vec2f {
    let light = normalize(vec3f(params.light_x, params.light_y, params.light_z));
    return vec2f(0.15 + 0.85 * max(dot(n, light), 0.0), et);
}
// speck_test on the CPU: a hit counts only when the zone carries on along its
// own surface. ZONE_SPECKLE_SAMPLES probes sit speckle_len/2 from the hit in the
// plane perpendicular to the surface normal n and look across the surface
// (ZONE_SPECKLE_DEPTH points over +-speckle_len/2 along n); a probe meets the
// zone when a point is in it or the escape time runs monotonically across a
// whole range (a shell thinner than the point spacing). ZONE_SPECKLE_NEED must.
fn zone_probe_crosses(e: array<f32, 5>) -> bool {
    var up = true;
    var down = true;
    for (var j: u32 = 1u; j < ZONE_SPECKLE_DEPTH; j = j + 1u) {
        if (e[j] < e[j - 1u]) { up = false; }
        if (e[j] > e[j - 1u]) { down = false; }
    }
    if (!(up || down)) { return false; }
    let lo = min(e[0], e[ZONE_SPECKLE_DEPTH - 1u]);
    let hi = max(e[0], e[ZONE_SPECKLE_DEPTH - 1u]);
    for (var i: u32 = 0u; i < params.zone_n; i = i + 1u) {
        if (lo < zone_lo(i) && hi >= zone_hi(i)) { return true; }
    }
    return false;
}
fn zone_solid_behind(hit: vec3f, n_in: vec3f) -> bool {
    if (params.zone_speckle_len <= 0.0) { return true; }
    var n = vec3f(0.0, 0.0, 1.0);
    if (dot(n_in, n_in) > 1e-12) { n = normalize(n_in); }
    var helper = vec3f(0.0, 1.0, 0.0);
    if (abs(n.x) < 0.9) { helper = vec3f(1.0, 0.0, 0.0); }
    let u = normalize(cross(n, helper));
    let v = cross(n, u);
    let r = 0.5 * params.zone_speckle_len;
    var met: u32 = 0u;
    for (var k: u32 = 0u; k < ZONE_SPECKLE_SAMPLES; k = k + 1u) {
        if (met >= ZONE_SPECKLE_NEED || met + (ZONE_SPECKLE_SAMPLES - k) < ZONE_SPECKLE_NEED) { break; }
        let a = 6.283185307 * f32(k) / f32(ZONE_SPECKLE_SAMPLES);
        let base = hit + u * (r * cos(a)) + v * (r * sin(a));
        var e: array<f32, 5>;
        var inside = false;
        for (var j: u32 = 0u; j < ZONE_SPECKLE_DEPTH; j = j + 1u) {
            let s = r * (2.0 * f32(j) / f32(ZONE_SPECKLE_DEPTH - 1u) - 1.0);
            e[j] = de_at(base + n * s).x;
            if (zone_contains(e[j])) { inside = true; break; }
        }
        if (inside || zone_probe_crosses(e)) { met = met + 1u; }
    }
    return met >= ZONE_SPECKLE_NEED;
}

fn march_zone(eye: vec3f, dir: vec3f, t0: f32, t1: f32, entry_n: vec3f, has_entry: bool) -> vec2f {
    if (params.zone_n == 0u) { return vec2f(0.0, 0.0); }
    let min_step = max(max(params.zone_min_step, params.hit_epsilon), 1e-9);
    var t = t0;
    var prev = t0;
    for (var i: u32 = 0u; i < params.max_march_steps; i = i + 1u) {
        let p = eye + dir * t;
        let r = de_at(p);
        if (zone_contains(r.x)) {
            var th = t;
            if (i != 0u) {
                var a = prev;
                var b = t;
                for (var k: u32 = 0u; k < ZONE_BISECT; k = k + 1u) {
                    let m = 0.5 * (a + b);
                    let em = de_at(eye + dir * m).x;
                    if (zone_contains(em)) { b = m; } else { a = m; }
                }
                th = b;
            }
            // The hit's normal first: the speck test probes along its surface.
            let hp = eye + dir * th;
            var n = -dir;
            var et_hit = r.x;
            if (i == 0u) {
                if (has_entry) { n = entry_n; if (dot(n, dir) > 0.0) { n = -n; } }
            } else {
                n = zone_normal(hp, dir);
                et_hit = de_at(hp).x;
            }
            if (zone_solid_behind(hp, n)) {
                return zone_shade(n, et_hit);
            }
            // A speck: march on from halfway through the tested stretch.
            prev = th + 0.5 * params.zone_speckle_len;
            t = prev;
            if (t >= t1) { break; }
            continue;
        }
        if (t >= t1) { break; }
        prev = t;
        t = min(t + max(zone_step(r.x, r.y) * params.step_safety, min_step), t1);
    }
    return vec2f(0.0, 0.0);
}

fn march_sample(u: f32, v: f32) -> vec2f {
    let forward = vec3f(params.fwd_x, params.fwd_y, params.fwd_z);
    let right   = vec3f(params.right_x, params.right_y, params.right_z);
    let up      = vec3f(params.up_x, params.up_y, params.up_z);
    let eye     = vec3f(params.eye_x, params.eye_y, params.eye_z);
    let dir = normalize(forward + right*u + up*v);

    var t0: f32;
    var t1: f32;
    // Outward normal of the surface the ray enters through (box face, sphere
    // or clip plane) — the lighting of a zone's flat cut there.
    var entry_n = vec3f(0.0, 0.0, 0.0);
    var has_entry = false;
    if (params.use_box != 0u) {
        // Slab method — mirrors quat_raymarch::ray_box exactly.
        let dir_a = array<f32, 3>(dir.x, dir.y, dir.z);
        let eye_a = array<f32, 3>(eye.x, eye.y, eye.z);
        let lo_a = array<f32, 3>(params.box_min_x, params.box_min_y, params.box_min_z);
        let hi_a = array<f32, 3>(params.box_max_x, params.box_max_y, params.box_max_z);
        var tn: f32 = 0.0;
        var tf: f32 = 3.4e38;
        var missed = false;
        var entry_axis: u32 = 3u; // 3 = the eye is inside the box
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
                if (ta > tn) { tn = ta; entry_axis = axis; }
                tf = min(tf, tb);
                if (tn > tf) { missed = true; }
            }
        }
        if (missed) { return vec2f(0.0, 0.0); }
        t0 = max(tn, 0.0);
        t1 = tf;
        if (entry_axis < 3u && tn > 0.0) {
            var n = array<f32, 3>(0.0, 0.0, 0.0);
            n[entry_axis] = select(1.0, -1.0, dir_a[entry_axis] > 0.0);
            entry_n = vec3f(n[0], n[1], n[2]);
            has_entry = true;
        }
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
        if (tt0 > 0.0) { entry_n = normalize(eye + dir * tt0); has_entry = true; }
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
            if (t_plane > t0) { t0 = t_plane; entry_n = normalize(clip_n); has_entry = true; }
        } else if (denom < -1e-6) {
            let t_plane = dot(clip_pos - eye, clip_n) / denom;
            t1 = min(t1, t_plane);
        } else if (dot(eye - clip_pos, clip_n) < 0.0) {
            return vec2f(0.0, 0.0);
        }
        if (t0 > t1) { return vec2f(0.0, 0.0); }
    }

    if (params.zone_enabled != 0u) {
        return march_zone(eye, dir, t0, t1, entry_n, has_entry);
    }

    let hit_eps = max(params.hit_epsilon, 1e-9);
    let min_step = max(params.domain_radius * 1e-6, 1e-9);
    // RaymarchDagParams::pixel_lod: one pixel's width per unit of distance.
    let pixel = 2.0 * params.half_h / f32(params.height);
    var t = t0;
    var prev = t0;
    var hit = false;
    var interior_hit = false;
    var started_inside = false;
    var hit_point = vec3f(0.0, 0.0, 0.0);
    let cap = f32(params.max_iter);
    // quat_dag::SOLID_STEP_DIVISIONS: with solid_interior no step exceeds the
    // region's diagonal / 128 (an overstated distance estimate jumped the set).
    var step_cap: f32 = 3.4e38;
    if (params.solid_interior != 0u) {
        var extent = 2.0 * params.domain_radius;
        if (params.use_box != 0u) {
            extent = length(vec3f(params.box_max_x - params.box_min_x, params.box_max_y - params.box_min_y, params.box_max_z - params.box_min_z));
        }
        step_cap = extent / 128.0;
    }
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
        // RaymarchDagParams::solid_interior: a ray stepping into the set stops,
        // the surface bisected back to the last point outside.
        if (params.solid_interior != 0u && et >= cap && !in_band && !above_cutoff) {
            var a = prev;
            var b = t;
            if (i != 0u) {
                for (var k: u32 = 0u; k < ZONE_BISECT; k = k + 1u) {
                    let m = 0.5 * (a + b);
                    if (de_at(eye + dir * m).x >= cap) { b = m; } else { a = m; }
                }
            }
            hit = true;
            interior_hit = true;
            started_inside = i == 0u;
            hit_point = eye + dir * b;
            break;
        }
        if (de < max(hit_eps, params.pixel_lod * pixel * t) && !in_band && !above_cutoff) {
            hit = true;
            hit_point = p;
            break;
        }
        prev = t;
        t = t + min(max(de * params.step_safety, min_step), step_cap);
    }
    if (!hit) { return vec2f(0.0, 0.0); }

    var normal = estimate_normal(hit_point);
    if (interior_hit) {
        if (started_inside) {
            // Started inside the set: the flat cut of the entry face (box, sphere, clip plane).
            normal = -dir;
            if (has_entry) { normal = entry_n; if (dot(normal, dir) > 0.0) { normal = -normal; } }
        } else {
            normal = zone_normal(hit_point, dir);
        }
    }
    let light = normalize(vec3f(params.light_x, params.light_y, params.light_z));
    let ndotl = max(dot(normal, light), 0.0);
    let shading = 0.15 + 0.85 * ndotl;

    let probe = hit_point + normal * params.color_probe_offset;
    let color_et = de_at(probe).x;

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
