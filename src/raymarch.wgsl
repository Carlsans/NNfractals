// WGSL compute shader: quaternion ray-marching (adaptive sphere tracing).
// Each thread computes one pixel. Mirrors src/quat_raymarch.rs + the
// relevant parts of src/quat_fractal.rs EXACTLY — this file has no
// independent design, it's a straight port, validated against the CPU
// path pixel-for-pixel (see render_gpu_raymarch.rs's tests) rather than
// trusted on its own.

struct Params {
    width:              u32,
    height:             u32,
    max_iter:           u32,
    formula:            u32,
    time_axis:          u32,
    max_march_steps:    u32,
    domain_radius:      f32,
    time_val:           f32,
    bailout_sq:         f32,
    hit_epsilon:        f32,
    step_safety:        f32,
    normal_eps:         f32,
    color_probe_offset: f32,
    bulb_power:         f32,
    mandelbox_scale:    f32,
    half_w:             f32,
    half_h:             f32,
    eye_x: f32, eye_y: f32, eye_z: f32,
    fwd_x: f32, fwd_y: f32, fwd_z: f32,
    right_x: f32, right_y: f32, right_z: f32,
    up_x: f32, up_y: f32, up_z: f32,
    light_x: f32, light_y: f32, light_z: f32,
    aa: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform>             params         : Params;
@group(0) @binding(1) var<storage, read_write> output_shading : array<f32>;
@group(0) @binding(2) var<storage, read_write> output_color   : array<f32>;

// ── Quaternion helpers ──────────────────────────────────────────────────────
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
fn qnorm_sq(q: Quat) -> f32 { return q.r*q.r + q.a*q.a + q.b*q.b + q.c*q.c; }
fn qconj(q: Quat) -> Quat { return Quat(q.r, -q.a, -q.b, -q.c); }
fn qabs_components(q: Quat) -> Quat { return Quat(abs(q.r), abs(q.a), abs(q.b), abs(q.c)); }
fn qfinite(q: Quat) -> bool {
    return !(any(vec4<bool>(
        !(q.r == q.r) || abs(q.r) > 3.4e38,
        !(q.a == q.a) || abs(q.a) > 3.4e38,
        !(q.b == q.b) || abs(q.b) > 3.4e38,
        !(q.c == q.c) || abs(q.c) > 3.4e38,
    )));
}

// Mirrors quat_fractal.rs's MANDELBOX_* consts exactly. Scale is a runtime
// uniform now (`params.bulb_power`/`params.mandelbox_scale`, see Params
// struct above) so imported per-file values actually take effect; fold
// limit and ball-fold radii stay fixed constants (not yet exposed the same
// way — every real .fract example seen so far leaves them at these
// defaults anyway).
const MANDELBOX_FOLD_LIMIT: f32 = 1.0;
const MANDELBOX_MIN_RADIUS_SQ: f32 = 0.25;
const MANDELBOX_FIXED_RADIUS_SQ: f32 = 1.0;

fn box_fold_component(x: f32) -> f32 {
    if (x > MANDELBOX_FOLD_LIMIT) { return 2.0 * MANDELBOX_FOLD_LIMIT - x; }
    if (x < -MANDELBOX_FOLD_LIMIT) { return -2.0 * MANDELBOX_FOLD_LIMIT - x; }
    return x;
}
fn box_fold(q: Quat) -> Quat {
    return Quat(box_fold_component(q.r), box_fold_component(q.a), box_fold_component(q.b), box_fold_component(q.c));
}
// Returns the folded quaternion; ball_factor_out receives the scale factor
// applied (mirrors quat_fractal.rs's ball_fold returning (Quat, f64) — WGSL
// has no tuple return, so the factor comes back via a pointer instead).
fn ball_fold(q: Quat, ball_factor_out: ptr<function, f32>) -> Quat {
    let r2 = qnorm_sq(q);
    var factor: f32 = 1.0;
    if (r2 < MANDELBOX_MIN_RADIUS_SQ) {
        factor = MANDELBOX_FIXED_RADIUS_SQ / MANDELBOX_MIN_RADIUS_SQ;
    } else if (r2 < MANDELBOX_FIXED_RADIUS_SQ) {
        factor = MANDELBOX_FIXED_RADIUS_SQ / r2;
    }
    *ball_factor_out = factor;
    return Quat(q.r * factor, q.a * factor, q.b * factor, q.c * factor);
}

// Mirrors QuatFormula::step exactly — formula IDs match QuatFormula::ALL's
// order in quat_fractal.rs: 0=Mandelbrot 1=Tricorn 2=BurningShip
// 3=BurningShipCubic 4=PerpendicularBurningShip 5=Celtic
// 6=PerpendicularMandelbrot 7=Cubic 8=Quartic 9=Bulb 10=Mandelbox.
fn step_formula(formula: u32, q: Quat) -> Quat {
    switch formula {
        case 0u: { return qmul(q, q); }
        case 1u: { let c = qconj(q); return qmul(c, c); }
        case 2u: { let a = qabs_components(q); return qmul(a, a); }
        case 3u: { let a = qabs_components(q); return qmul(qmul(a, a), a); }
        case 4u: { let p = Quat(q.r, -abs(q.a), -abs(q.b), -abs(q.c)); return qmul(p, p); }
        case 5u: { let sq = qmul(q, q); return Quat(abs(sq.r), sq.a, sq.b, sq.c); }
        case 6u: { let p = Quat(q.r, abs(q.a), abs(q.b), abs(q.c)); return qmul(p, p); }
        case 7u: { return qmul(qmul(q, q), q); }
        case 8u: { let sq = qmul(q, q); return qmul(sq, sq); }
        case 9u: {
            let n = params.bulb_power;
            let v_mag = sqrt(q.a*q.a + q.b*q.b + q.c*q.c);
            let rho = sqrt(qnorm_sq(q));
            let theta1 = atan2(v_mag, q.r);
            let theta2 = atan2(sqrt(q.a*q.a + q.b*q.b), q.c);
            let phi = atan2(q.b, q.a);
            let rho_n = pow(rho, n);
            let t1 = theta1 * n; let t2 = theta2 * n; let ph = phi * n;
            let new_r = rho_n * cos(t1);
            let new_vmag = rho_n * sin(t1);
            return Quat(new_r, new_vmag*sin(t2)*cos(ph), new_vmag*sin(t2)*sin(ph), new_vmag*cos(t2));
        }
        default: { // 10 = Mandelbox
            var ball_factor: f32 = 1.0;
            let folded = ball_fold(box_fold(q), &ball_factor);
            return Quat(folded.r * params.mandelbox_scale, folded.a * params.mandelbox_scale, folded.b * params.mandelbox_scale, folded.c * params.mandelbox_scale);
        }
    }
}

// Mirrors QuatFormula::power() exactly. Meaningless for Mandelbox (10) —
// quat_escape_de below never calls this for that formula, same as the CPU
// side special-cases it entirely instead of using the power-law recurrence.
fn formula_power(formula: u32) -> f32 {
    switch formula {
        case 7u, 3u: { return 3.0; }
        case 8u: { return 4.0; }
        case 9u: { return params.bulb_power; }
        default: { return 2.0; } // 0,1,2,4,5,6
    }
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

// Mirrors quat_escape_de exactly: returns (escape_time, distance_estimate).
fn quat_escape_de(formula: u32, q_const: Quat, max_iter: u32, bailout_sq: f32) -> vec2f {
    let n = formula_power(formula);
    var q = Quat(0.0, 0.0, 0.0, 0.0);
    var dr: f32 = 1.0;
    for (var it: u32 = 0u; it < max_iter; it = it + 1u) {
        if (formula == 10u) { // Mandelbox: not a power map, see step_formula's default case
            var ball_factor: f32 = 1.0;
            _ = ball_fold(box_fold(q), &ball_factor);
            dr = abs(params.mandelbox_scale) * ball_factor * dr + 1.0;
        } else {
            let rho = sqrt(qnorm_sq(q));
            dr = n * pow(rho, n - 1.0) * dr + 1.0;
        }
        q = qadd(step_formula(formula, q), q_const);
        let ms = qnorm_sq(q);
        if (ms > bailout_sq) {
            let et = max(f32(it) + 1.0 - log2(log2(ms) * 0.5), 0.0);
            let r = sqrt(ms);
            let de = max(0.5 * log(r) * r / max(dr, 1e-30), 0.0);
            return vec2f(et, de);
        }
        if (!qfinite(q)) {
            return vec2f(f32(it), 0.0);
        }
    }
    let r = max(sqrt(qnorm_sq(q)), 1e-30);
    let de = max(0.5 * log(r) * r / max(dr, 1e-30), 0.0);
    return vec2f(f32(max_iter), de);
}

fn de_at(p: vec3f) -> f32 {
    let q = assemble(params.time_axis, p.x, p.y, p.z, params.time_val);
    return quat_escape_de(params.formula, q, params.max_iter, params.bailout_sq).y;
}

// Mirrors estimate_normal exactly: tetrahedral 4-tap gradient.
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

// One sub-ray through normalized screen offset (u,v). Mirrors march_ray +
// the shading computation in render_raymarch_frame exactly. Returns
// (shading, color_et), both 0 on a miss.
fn march_sample(u: f32, v: f32) -> vec2f {
    let forward = vec3f(params.fwd_x, params.fwd_y, params.fwd_z);
    let right   = vec3f(params.right_x, params.right_y, params.right_z);
    let up      = vec3f(params.up_x, params.up_y, params.up_z);
    let eye     = vec3f(params.eye_x, params.eye_y, params.eye_z);
    let dir = normalize(forward + right*u + up*v);

    // ray/bounding-sphere intersection (mirrors ray_sphere exactly)
    let a = dot(dir, dir);
    let b = 2.0 * dot(eye, dir);
    let cc = dot(eye, eye) - params.domain_radius * params.domain_radius;
    let disc = b*b - 4.0*a*cc;
    if (disc < 0.0) {
        return vec2f(0.0, 0.0);
    }
    let sq = sqrt(disc);
    var t0 = (-b - sq) / (2.0*a);
    let t1 = (-b + sq) / (2.0*a);
    if (t1 < 0.0) {
        return vec2f(0.0, 0.0);
    }
    t0 = max(t0, 0.0);

    let hit_eps = max(params.hit_epsilon, 1e-9);
    let min_step = max(params.domain_radius * 1e-6, 1e-9);
    var t = t0;
    var hit = false;
    var hit_point = vec3f(0.0, 0.0, 0.0);
    for (var i: u32 = 0u; i < params.max_march_steps; i = i + 1u) {
        if (t > t1) {
            break;
        }
        let p = eye + dir * t;
        let de = de_at(p);
        if (de < hit_eps) {
            hit = true;
            hit_point = p;
            break;
        }
        t = t + max(de * params.step_safety, min_step);
    }

    if (!hit) {
        return vec2f(0.0, 0.0);
    }

    let normal = estimate_normal(hit_point);
    let light = normalize(vec3f(params.light_x, params.light_y, params.light_z));
    let ndotl = max(dot(normal, light), 0.0);
    let shading = 0.15 + 0.85 * ndotl;

    let probe = hit_point + normal * params.color_probe_offset;
    let probe_q = assemble(params.time_axis, probe.x, probe.y, probe.z, params.time_val);
    let color_et = quat_escape_de(params.formula, probe_q, params.max_iter, params.bailout_sq).x;

    return vec2f(shading, color_et);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
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
