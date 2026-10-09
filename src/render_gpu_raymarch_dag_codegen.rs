//! Per-genome CODE-GENERATED GPU ray-marching — a faster alternative to
//! `render_gpu_raymarch_dag.rs`'s general DAG interpreter.
//!
//! That module uploads a genome's program as DATA (a storage buffer of
//! `DagNode`s) and a fixed shader interprets it with a runtime loop +
//! opcode `switch` on every DAG node, of every march step, of every pixel.
//! Benchmarked ~9x slower than the hand-built `QuatFormula` GPU path
//! (`render_gpu_raymarch.rs`) at identical settings — and the hand-built
//! path is fast for an unsurprising reason: each of its 10 formulas is
//! fixed, unrolled Rust match arms, not a runtime-interpreted loop.
//!
//! This module gets an arbitrary genome the same property by generating
//! the program's straight-line WGSL — one `let` per DAG node, in program
//! order, referencing only already-computed locals — ONCE when a genome
//! is loaded (compiling a dedicated `wgpu::ComputePipeline` for it), not
//! once per frame. A 1800-frame video amortizes that one-time shader
//! compile to nothing. See `raymarch_dag_codegen_template.wgsl` for the
//! shared (formula-independent) surrounding machinery, and
//! `generate_shader_source` below for the per-genome specialization.
//!
//! Correctness rests entirely on the codegen mirroring
//! `quat_dag.rs`'s `eval_program_quat`/`eval_program_quat_deriv` and
//! `render_gpu_raymarch_dag.rs`'s interpreter (`raymarch_dag.wgsl`)
//! opcode-for-opcode — verified by this module's own pixel-for-pixel
//! parity tests against the CPU path, the same discipline every prior
//! quaternion generalization in this project has used.

#![cfg(feature = "wgpu-backend")]

use std::sync::{Mutex, OnceLock};

use crate::formula::{op, OpNode, N_SLOTS};
use crate::quat_dag::RaymarchDagParams;
use crate::quat_motion::{look_at_basis, normalize, sub};
use crate::quat_fractal::TimeAxis;
use crate::quat_raymarch::RaymarchCamera;

const TEMPLATE: &str = include_str!("raymarch_dag_codegen_template.wgsl");
/// 59 plain scalar fields (see the template's `Params` struct): the
/// original 37 (2 of which — `use_box`/`use_axis_perm` — were unused
/// `_pad0`/`_pad1` padding before the animation-viewer plan's Phase 2)
/// plus 10 from Phase 2 (box min/max, axis permutation) plus 7 from
/// Phase 6 (clip plane: enabled flag + pos + normal) plus 3 from the
/// Slide Iteration effect (enabled flag + min + max) plus 2 for the
/// "hide top N%" resolved cutoff (enabled flag + value).
const PARAMS_BYTES: u64 = 59 * 4;
const WG: u32 = 8;

fn wgsl_f32(v: f32) -> String {
    if v.is_finite() {
        format!("{v:.9}")
    } else {
        "0.0".to_string()
    }
}

fn operand(prefix: &str, idx: u8, i: usize, zero: &str) -> String {
    let idx = (idx as usize).min(N_SLOTS - 1);
    if idx < i {
        format!("{prefix}{idx}")
    } else {
        zero.to_string()
    }
}

/// Straight-line WGSL for `eval_warp`'s body — mirrors `eval_program_quat`
/// called with `z=c=point` (the warp program's self-referential reading
/// convention, see `quat_dag.rs`'s `eval_program_quat_deriv`'s doc
/// comment) opcode-for-opcode.
fn codegen_warp(warp: &[OpNode]) -> String {
    let n = warp.len().min(N_SLOTS);
    if n == 0 {
        return "    return point;\n".to_string();
    }
    let mut out = String::new();
    for (i, node) in warp.iter().take(n).enumerate() {
        let a = operand("w", node.a, i, "Quat(0.0, 0.0, 0.0, 0.0)");
        let b = operand("w", node.b, i, "Quat(0.0, 0.0, 0.0, 0.0)");
        let expr = match node.op {
            op::Z | op::C => "point".to_string(),
            op::CONST => format!("Quat({}, {}, 0.0, 0.0)", wgsl_f32(node.kre), wgsl_f32(node.kim)),
            op::SQR => format!("qmul({a}, {a})"),
            op::CUBE => format!("qmul(qmul({a}, {a}), {a})"),
            op::QUART => format!("qmul(qmul({a}, {a}), qmul({a}, {a}))"),
            op::RECIP => format!("qrecip({a})"),
            op::SIN => format!("qsin({a})"),
            op::COS => format!("qcos({a})"),
            op::EXP => format!("qexp({a})"),
            op::LOG => format!("qlog({a})"),
            op::TANH => format!("qtanh({a})"),
            op::CONJ => format!("qconj({a})"),
            op::ABSFOLD => format!("qabs_components({a})"),
            op::ABSRE => format!("Quat(abs({a}.r), {a}.a, {a}.b, {a}.c)"),
            op::ABSIM => format!("Quat({a}.r, abs({a}.a), abs({a}.b), abs({a}.c))"),
            op::NORMZ => format!("qnormz({a})"),
            op::ADD => format!("qadd({a}, {b})"),
            op::SUB => format!("qsub({a}, {b})"),
            op::MUL => format!("qmul({a}, {b})"),
            op::DIV => format!("qmul({a}, qrecip({b}))"),
            _ => "Quat(0.0, 0.0, 0.0, 0.0)".to_string(),
        };
        out.push_str(&format!("    let w{i}: Quat = {expr};\n"));
    }
    out.push_str(&format!("    return w{};\n", n - 1));
    out
}

/// Straight-line WGSL for `eval_main_deriv`'s body — mirrors
/// `eval_program_quat_deriv`/`raymarch_dag.wgsl`'s interpreted switch
/// opcode-for-opcode, including the derivative (chain-rule) side.
/// `EXP`/`TANH` reuse the node's own just-computed value (`n{i}`) instead
/// of recomputing `qexp`/`qtanh` inside the derivative expression — a
/// free simplification since the interpreter's `let e = qexp(a); val = e;
/// dval = qnorm(e) * da;` pattern already names that reuse explicitly.
fn codegen_main_deriv(prog: &[OpNode]) -> String {
    let n = prog.len().min(N_SLOTS);
    if n == 0 {
        return "    return ValDeriv(Quat(0.0, 0.0, 0.0, 0.0), 0.0);\n".to_string();
    }
    let mut out = String::new();
    for (i, node) in prog.iter().take(n).enumerate() {
        let a = operand("n", node.a, i, "Quat(0.0, 0.0, 0.0, 0.0)");
        let da = operand("d", node.a, i, "0.0");
        let b = operand("n", node.b, i, "Quat(0.0, 0.0, 0.0, 0.0)");
        let db = operand("d", node.b, i, "0.0");
        let (val, dval) = match node.op {
            op::Z => ("z".to_string(), "dz".to_string()),
            op::C => ("c".to_string(), "dc".to_string()),
            op::CONST => (format!("Quat({}, {}, 0.0, 0.0)", wgsl_f32(node.kre), wgsl_f32(node.kim)), "0.0".to_string()),
            op::SQR => (format!("qmul({a}, {a})"), format!("2.0 * qnorm({a}) * {da}")),
            op::CUBE => (format!("qmul(qmul({a}, {a}), {a})"), format!("3.0 * qnorm({a}) * qnorm({a}) * {da}")),
            op::QUART => (
                format!("qmul(qmul({a}, {a}), qmul({a}, {a}))"),
                format!("4.0 * qnorm({a}) * qnorm({a}) * qnorm({a}) * {da}"),
            ),
            op::RECIP => (format!("qrecip({a})"), format!("{da} / (qnorm({a})*qnorm({a}) + EPS)")),
            op::SIN => (format!("qsin({a})"), format!("qnorm(qcos({a})) * {da}")),
            op::COS => (format!("qcos({a})"), format!("qnorm(qsin({a})) * {da}")),
            op::EXP => (format!("qexp({a})"), format!("qnorm(n{i}) * {da}")),
            op::LOG => (format!("qlog({a})"), format!("{da} / (qnorm({a}) + EPS)")),
            op::TANH => (format!("qtanh({a})"), format!("abs(1.0 - qnorm(n{i})*qnorm(n{i})) * {da}")),
            op::CONJ => (format!("qconj({a})"), da.clone()),
            op::ABSFOLD => (format!("qabs_components({a})"), da.clone()),
            op::ABSRE => (format!("Quat(abs({a}.r), {a}.a, {a}.b, {a}.c)"), da.clone()),
            op::ABSIM => (format!("Quat({a}.r, abs({a}.a), abs({a}.b), abs({a}.c))"), da.clone()),
            op::NORMZ => (format!("qnormz({a})"), format!("{da} / (qnorm({a}) + EPS)")),
            op::ADD => (format!("qadd({a}, {b})"), format!("{da} + {db}")),
            op::SUB => (format!("qsub({a}, {b})"), format!("{da} + {db}")),
            op::MUL => (format!("qmul({a}, {b})"), format!("qnorm({a})*{db} + qnorm({b})*{da}")),
            op::DIV => (
                format!("qmul({a}, qrecip({b}))"),
                format!("{da} / (qnorm({b}) + EPS) + qnorm({a}) * ({db} / (qnorm({b})*qnorm({b}) + EPS))"),
            ),
            _ => ("Quat(0.0, 0.0, 0.0, 0.0)".to_string(), "0.0".to_string()),
        };
        out.push_str(&format!("    let n{i}: Quat = {val};\n"));
        out.push_str(&format!("    let d{i}: f32 = {dval};\n"));
    }
    out.push_str(&format!("    return ValDeriv(n{}, d{});\n", n - 1, n - 1));
    out
}

pub fn generate_shader_source(prog: &[OpNode], warp: &[OpNode]) -> String {
    TEMPLATE
        .replace("// GENERATED-CODE-MARKER-WARP", &codegen_warp(warp))
        .replace("// GENERATED-CODE-MARKER-MAIN-DERIV", &codegen_main_deriv(prog))
}

fn time_axis_id(t: TimeAxis) -> u32 {
    match t {
        TimeAxis::R => 0,
        TimeAxis::A => 1,
        TimeAxis::B => 2,
        TimeAxis::C => 3,
    }
}

struct ParamsWriter<'a> {
    buf: &'a mut [u8],
    pos: usize,
}
impl<'a> ParamsWriter<'a> {
    fn u32(&mut self, v: u32) {
        self.buf[self.pos..self.pos + 4].copy_from_slice(&v.to_le_bytes());
        self.pos += 4;
    }
    fn f32(&mut self, v: f64) {
        self.buf[self.pos..self.pos + 4].copy_from_slice(&(v as f32).to_le_bytes());
        self.pos += 4;
    }
}

/// The shared, expensive-to-create GPU handle (adapter/device/queue) —
/// created once for the process and reused across every compiled genome
/// pipeline, exactly like `render_gpu_raymarch.rs`/`render_gpu_raymarch_dag.rs`'s
/// own singletons.
struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
}
unsafe impl Send for Gpu {}
unsafe impl Sync for Gpu {}

static GPU: OnceLock<Option<Mutex<Gpu>>> = OnceLock::new();
static GPU_INIT_ERROR: OnceLock<Mutex<Option<String>>> = OnceLock::new();

fn gpu() -> Option<&'static Mutex<Gpu>> {
    GPU.get_or_init(|| pollster::block_on(init_gpu()).map(|(device, queue)| Mutex::new(Gpu { device, queue }))).as_ref()
}

pub fn gpu_available() -> bool {
    gpu().is_some()
}

/// The reason the most recent GPU init attempt failed — `None` if init has
/// never run yet, or it succeeded. Adapter/device request failures used to
/// be logged with `eprintln!` and then discarded, leaving every caller
/// (`CompiledDagPipeline::compile` returns a bare `Option`) with no way to
/// tell WHY it got `None` — every failure surfaced identically as "no GPU
/// adapter available (or shader failed to compile) — see stderr", which
/// was also just wrong: shader compilation errors don't produce `None`
/// here at all (wgpu reports them asynchronously via the device's error
/// scope, not a `Result` from `create_shader_module`), so that parenthetical
/// never actually described a real code path. This is the fix: the actual
/// reason, queryable, not just logged.
pub fn last_gpu_init_error() -> Option<String> {
    GPU_INIT_ERROR.get().and_then(|m| m.lock().ok().and_then(|g| g.clone()))
}

async fn init_gpu() -> Option<(wgpu::Device, wgpu::Queue)> {
    let record_error = |msg: String| {
        eprintln!("[gpu-raymarch-dag-codegen] {msg}");
        if let Ok(mut slot) = GPU_INIT_ERROR.get_or_init(|| Mutex::new(None)).lock() {
            *slot = Some(msg);
        }
    };
    let inst = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = match inst
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
    {
        Ok(a) => a,
        Err(e) => {
            record_error(format!("adapter request failed: {e} (no compatible GPU found, or all adapters are already claimed by other processes)"));
            return None;
        }
    };
    match adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            experimental_features: Default::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        })
        .await
    {
        Ok(dq) => Some(dq),
        Err(e) => {
            record_error(format!("device request failed: {e} (adapter was found, but creating a logical device from it failed)"));
            None
        }
    }
}

/// A shader/pipeline compiled ONCE for one specific genome's `(prog,
/// warp)` — reused for every frame of that genome's video. Building one
/// of these is the whole point of this module: pay the shader-compile
/// cost once, not per frame.
pub struct CompiledDagPipeline {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    params_buf: wgpu::Buffer,
    shading_buf: wgpu::Buffer,
    color_buf: wgpu::Buffer,
    shading_staging: wgpu::Buffer,
    color_staging: wgpu::Buffer,
    max_pixels: u64,
}

fn mk_buf(device: &wgpu::Device, label: &'static str, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: size.max(16), usage, mapped_at_creation: false })
}

#[allow(clippy::type_complexity)]
fn alloc_outputs(device: &wgpu::Device, max_pixels: u64) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer) {
    let out_sz = max_pixels * 4;
    (
        mk_buf(device, "rmdc_shading", out_sz, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC),
        mk_buf(device, "rmdc_color", out_sz, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC),
        mk_buf(device, "rmdc_shading_stage", out_sz, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST),
        mk_buf(device, "rmdc_color_stage", out_sz, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST),
    )
}

impl CompiledDagPipeline {
    /// Compiles a fresh shader/pipeline for this exact `(prog, warp)`
    /// pair. `None` if no GPU adapter is available.
    pub fn compile(prog: &[OpNode], warp: &[OpNode]) -> Option<Self> {
        let g = gpu()?;
        let g = g.lock().ok()?;
        let device = &g.device;

        let src = generate_shader_source(prog, warp);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raymarch_dag_codegen"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });

        let entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                entry(0, wgpu::BufferBindingType::Uniform),
                entry(1, wgpu::BufferBindingType::Storage { read_only: false }),
                entry(2, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let params_buf = mk_buf(device, "rmdc_params", PARAMS_BYTES, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        let max_pixels = 128 * 128;
        let (shading_buf, color_buf, shading_staging, color_staging) = alloc_outputs(device, max_pixels);

        Some(Self { pipeline, bgl, params_buf, shading_buf, color_buf, shading_staging, color_staging, max_pixels })
    }

    pub fn render(&mut self, params: &RaymarchDagParams, cam: &RaymarchCamera, width: u32, height: u32) -> (Vec<f32>, Vec<f32>) {
        let g = gpu().expect("compile() already proved a GPU is available");
        let g = g.lock().expect("gpu mutex poisoned");
        let device = &g.device;
        let queue = &g.queue;

        let pix = (width as u64) * (height as u64);
        if pix > self.max_pixels {
            let (shading_buf, color_buf, shading_staging, color_staging) = alloc_outputs(device, pix);
            self.shading_buf = shading_buf;
            self.color_buf = color_buf;
            self.shading_staging = shading_staging;
            self.color_staging = color_staging;
            self.max_pixels = pix;
        }

        let forward = normalize(sub(cam.target, cam.eye));
        let (right, up) = look_at_basis(forward, cam.up_hint);
        let half_h = (cam.fov_y * 0.5).tan();
        let aspect = width as f64 / (height.max(1)) as f64;
        let half_w = half_h * aspect;

        let mut pb = [0u8; PARAMS_BYTES as usize];
        {
            let mut w = ParamsWriter { buf: &mut pb, pos: 0 };
            w.u32(width);
            w.u32(height);
            w.u32(params.max_iter);
            w.u32(time_axis_id(params.time_axis));
            w.u32(params.max_march_steps);
            w.u32(params.formula.julia as u32);
            w.f32(params.domain_radius);
            w.f32(params.time_val);
            w.f32(params.bailout * params.bailout);
            w.f32(params.hit_epsilon);
            w.f32(params.step_safety);
            w.f32(params.normal_eps);
            w.f32(params.color_probe_offset);
            w.f32(half_w);
            w.f32(half_h);
            w.f32(params.formula.jc.0 as f64);
            w.f32(params.formula.jc.1 as f64);
            w.f32(params.formula.phoenix.0 as f64);
            w.f32(params.formula.phoenix.1 as f64);
            w.f32(cam.eye.0);
            w.f32(cam.eye.1);
            w.f32(cam.eye.2);
            w.f32(forward.0);
            w.f32(forward.1);
            w.f32(forward.2);
            w.f32(right.0);
            w.f32(right.1);
            w.f32(right.2);
            w.f32(up.0);
            w.f32(up.1);
            w.f32(up.2);
            w.f32(params.light_dir.0);
            w.f32(params.light_dir.1);
            w.f32(params.light_dir.2);
            w.u32(params.aa);
            // ── Animation-viewer plan, Phase 2: additive fields (were
            // 2 padding u32s). `box_bounds`/`axis_assignment` are `None`
            // for every caller before this plan, so use_box/use_axis_perm
            // write 0 and the shader falls back to the original
            // domain_radius-sphere/time_axis behavior exactly. ─────────
            w.u32(params.box_bounds.is_some() as u32);
            w.u32(params.axis_assignment.is_some() as u32);
            let (bmin, bmax) = params.box_bounds.unwrap_or(((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)));
            w.f32(bmin.0);
            w.f32(bmin.1);
            w.f32(bmin.2);
            w.f32(bmax.0);
            w.f32(bmax.1);
            w.f32(bmax.2);
            let parts = params.axis_assignment
                .map(|a| (a.x, a.y, a.z, a.t))
                .unwrap_or((crate::anim_timeline::QuatPart::R, crate::anim_timeline::QuatPart::A, crate::anim_timeline::QuatPart::B, crate::anim_timeline::QuatPart::C));
            w.u32(parts.0.index() as u32);
            w.u32(parts.1.index() as u32);
            w.u32(parts.2.index() as u32);
            w.u32(parts.3.index() as u32);
            // ── Animation-viewer plan, Phase 6: orientable difference/
            // cutaway clip plane. clip_plane=None (every caller before
            // this plan) writes clip_enabled=0, which march_sample's
            // "if (params.clip_enabled != 0u)" branch skips entirely.
            w.u32(params.clip_plane.is_some() as u32);
            let (clip_pos, clip_normal) = params.clip_plane.unwrap_or(((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)));
            w.f32(clip_pos.0);
            w.f32(clip_pos.1);
            w.f32(clip_pos.2);
            w.f32(clip_normal.0);
            w.f32(clip_normal.1);
            w.f32(clip_normal.2);
            // ── Slide Iteration effect: hidden escape-iteration band.
            // slide_iter=None (every caller before this effect) writes
            // slide_iter_enabled=0, which de_at's own gating skips entirely.
            w.u32(params.slide_iter.is_some() as u32);
            let (slide_min, slide_max) = params.slide_iter.unwrap_or((0.0, 0.0));
            w.f32(slide_min);
            w.f32(slide_max);
            // ── "hide top N%" resolved cutoff. hide_above=None (every
            // caller's own probe pass, and every caller before this
            // control existed) writes hide_above_enabled=0.
            w.u32(params.hide_above.is_some() as u32);
            w.f32(params.hide_above.unwrap_or(0.0));
        }
        queue.write_buffer(&self.params_buf, 0, &pb);

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.shading_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.color_buf.as_entire_binding() },
            ],
        });

        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut p = enc.begin_compute_pass(&Default::default());
            p.set_pipeline(&self.pipeline);
            p.set_bind_group(0, &bg, &[]);
            p.dispatch_workgroups(width.div_ceil(WG), height.div_ceil(WG), 1);
        }
        let out_sz = pix * 4;
        enc.copy_buffer_to_buffer(&self.shading_buf, 0, &self.shading_staging, 0, out_sz);
        enc.copy_buffer_to_buffer(&self.color_buf, 0, &self.color_staging, 0, out_sz);
        queue.submit(Some(enc.finish()));

        let (tx1, rx1) = std::sync::mpsc::channel();
        self.shading_staging.slice(..out_sz).map_async(wgpu::MapMode::Read, move |r| {
            tx1.send(r).ok();
        });
        let (tx2, rx2) = std::sync::mpsc::channel();
        self.color_staging.slice(..out_sz).map_async(wgpu::MapMode::Read, move |r| {
            tx2.send(r).ok();
        });
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        rx1.recv().unwrap().unwrap();
        rx2.recv().unwrap().unwrap();

        let shading: Vec<f32> = {
            let mapped = self.shading_staging.slice(..out_sz).get_mapped_range();
            let v = mapped.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
            drop(mapped);
            v
        };
        self.shading_staging.unmap();

        let color: Vec<f32> = {
            let mapped = self.color_staging.slice(..out_sz).get_mapped_range();
            let v = mapped.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
            drop(mapped);
            v
        };
        self.color_staging.unmap();

        (shading, color)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quat_dag::{render_raymarch_dag_frame, QuatDagFormula};

    fn base_params<'a>(prog: &'a [OpNode], warp: &'a [OpNode]) -> RaymarchDagParams<'a> {
        RaymarchDagParams {
            formula: QuatDagFormula { prog, warp, julia: false, jc: (0.0, 0.0), phoenix: (0.0, 0.0) },
            time_axis: TimeAxis::C,
            time_val: 0.0,
            domain_radius: 1.6,
            box_bounds: None, axis_assignment: None, clip_plane: None, slide_iter: None, hide_above: None,
            max_iter: 40,
            bailout: 4.0,
            max_march_steps: 200,
            hit_epsilon: 1.6 * 1e-4,
            step_safety: 0.8,
            light_dir: (0.5, 0.8, 0.3),
            normal_eps: 1.6 * 1e-3,
            color_probe_offset: 1.6 * 1e-2,
            aa: 1,
        }
    }

    fn base_cam() -> RaymarchCamera {
        RaymarchCamera { eye: (1.6, 1.0, -3.4), target: (0.0, 0.0, 0.0), up_hint: (0.0, 1.0, 0.0), fov_y: 45.0_f64.to_radians() }
    }

    fn assert_matches_cpu(prog: &[OpNode], warp: &[OpNode], params: RaymarchDagParams, w: u32, h: u32) {
        if !gpu_available() {
            eprintln!("skipping: no GPU adapter available in this environment");
            return;
        }
        let cam = base_cam();
        let (cpu_shading, cpu_color) = render_raymarch_dag_frame(&params, &cam, w, h);
        let mut pipeline = CompiledDagPipeline::compile(prog, warp).expect("gpu available");
        let (gpu_shading, gpu_color) = pipeline.render(&params, &cam, w, h);

        assert_eq!(cpu_shading.len(), gpu_shading.len());
        let mut mismatched_hits = 0;
        let mut max_shade_diff = 0.0f32;
        let mut large_shade_diffs = 0usize;
        let mut agreeing_shade = 0usize;
        for i in 0..cpu_shading.len() {
            let (cs, gs) = (cpu_shading[i], gpu_shading[i]);
            if (cs > 0.0) != (gs > 0.0) {
                mismatched_hits += 1;
                continue;
            }
            if cs <= 0.0 {
                continue;
            }
            agreeing_shade += 1;
            let d = (cs - gs).abs();
            max_shade_diff = max_shade_diff.max(d);
            if d > 0.15 {
                large_shade_diffs += 1;
                eprintln!("  pixel {i}: cpu_shading={cs} gpu_shading={gs} diff={d} cpu_color={} gpu_color={}", cpu_color[i], gpu_color[i]);
            }
        }
        let mismatch_frac = mismatched_hits as f64 / cpu_shading.len() as f64;
        assert!(mismatch_frac < 0.05, "{mismatched_hits}/{} pixels disagree on hit/miss ({:.1}%)", cpu_shading.len(), mismatch_frac * 100.0);
        let large_shade_frac = large_shade_diffs as f64 / agreeing_shade.max(1) as f64;
        assert!(
            large_shade_frac < 0.05,
            "{large_shade_diffs}/{agreeing_shade} agreeing-hit pixels have shading diff > 0.15 ({:.1}%); max diff {max_shade_diff}",
            large_shade_frac * 100.0
        );

        let mut agreeing = 0usize;
        let mut large_diffs = 0usize;
        for i in 0..cpu_color.len() {
            if cpu_shading[i] > 0.0 && gpu_shading[i] > 0.0 {
                agreeing += 1;
                if (cpu_color[i] - gpu_color[i]).abs() > 5.0 {
                    large_diffs += 1;
                }
            }
        }
        let large_frac = large_diffs as f64 / agreeing.max(1) as f64;
        assert!(large_frac < 0.05, "{large_diffs}/{agreeing} agreeing-hit pixels have a large color_t diff ({:.1}%)", large_frac * 100.0);
    }

    #[test]
    fn codegen_matches_cpu_for_a_representative_dag_program() {
        let prog = [
            OpNode { op: op::Z, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::SIN, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::C, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::MUL, a: 0, b: 2, kre: 0.0, kim: 0.0 },
            OpNode { op: op::SUB, a: 3, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::DIV, a: 4, b: 1, kre: 0.0, kim: 0.0 },
            OpNode { op: op::LOG, a: 5, b: 0, kre: 0.0, kim: 0.0 },
        ];
        assert_matches_cpu(&prog, &[], base_params(&prog, &[]), 64, 64);
    }

    /// Animation-viewer plan, Phase 2: exercises the NEW `use_box`/
    /// `box_min/max` shader branch (not just the `None` fallback every
    /// other test in this file uses) — GPU-vs-CPU parity, same discipline
    /// as every test in this module.
    #[test]
    fn codegen_matches_cpu_with_box_bounds() {
        let prog = [
            OpNode { op: op::Z, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::C, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::SQR, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::ADD, a: 2, b: 1, kre: 0.0, kim: 0.0 },
        ];
        let mut params = base_params(&prog, &[]);
        // A genuinely non-cubic box (asymmetric per-axis extents) so a
        // bug that only handles the symmetric-cube case wouldn't be caught.
        params.box_bounds = Some(((-1.2, -0.9, -1.6), (1.4, 1.1, 1.6)));
        assert_matches_cpu(&prog, &[], params, 64, 64);
    }

    /// Exercises the NEW `use_axis_perm`/`axis_part_*` shader branch.
    #[test]
    fn codegen_matches_cpu_with_axis_assignment() {
        use crate::anim_timeline::{AxisAssignment, QuatPart};
        let prog = [
            OpNode { op: op::Z, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::C, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::SQR, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::ADD, a: 2, b: 1, kre: 0.0, kim: 0.0 },
        ];
        let mut params = base_params(&prog, &[]);
        // A genuine (non-identity-like) permutation: time on R, spatial
        // axes reshuffled across A/B/C rather than left in order.
        params.axis_assignment = Some(AxisAssignment { x: QuatPart::C, y: QuatPart::B, z: QuatPart::A, t: QuatPart::R });
        params.time_val = 0.35;
        assert_matches_cpu(&prog, &[], params, 64, 64);
    }

    /// Both new fields at once — the actual combination the animation
    /// viewer's live render loop will use once axis assignment + the
    /// bounding-box editor are wired up.
    #[test]
    fn codegen_matches_cpu_with_box_bounds_and_axis_assignment_together() {
        use crate::anim_timeline::{AxisAssignment, QuatPart};
        let prog = [
            OpNode { op: op::Z, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::C, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::SQR, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::ADD, a: 2, b: 1, kre: 0.0, kim: 0.0 },
        ];
        let mut params = base_params(&prog, &[]);
        params.box_bounds = Some(((-1.0, -1.3, -1.6), (1.0, 1.3, 1.6)));
        params.axis_assignment = Some(AxisAssignment { x: QuatPart::A, y: QuatPart::C, z: QuatPart::R, t: QuatPart::B });
        params.time_val = -0.2;
        assert_matches_cpu(&prog, &[], params, 64, 64);
    }

    /// Animation-viewer plan, Phase 6: exercises the NEW `clip_enabled`/
    /// `clip_pos`/`clip_normal` shader branch — GPU-vs-CPU parity.
    #[test]
    fn codegen_matches_cpu_with_clip_plane() {
        let prog = [
            OpNode { op: op::Z, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::C, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::SQR, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::ADD, a: 2, b: 1, kre: 0.0, kim: 0.0 },
        ];
        let mut params = base_params(&prog, &[]);
        params.box_bounds = Some(((-1.6, -1.6, -1.6), (1.6, 1.6, 1.6)));
        // A deliberately non-axis-aligned plane through a point not at the
        // box's own origin, so a bug that only handles the "through the
        // exact center, axis-aligned" case wouldn't be caught.
        params.clip_plane = Some(((0.2, -0.1, 0.0), (0.6, 0.8, 0.0)));
        assert_matches_cpu(&prog, &[], params, 64, 64);
    }

    /// Slide Iteration effect: exercises the NEW `slide_iter_enabled`/
    /// `slide_iter_min`/`slide_iter_max` shader branch — GPU-vs-CPU parity.
    /// A mid-range band (neither empty nor covering the whole [0,max_iter)
    /// span) so both "hidden" and "still visible" points are exercised.
    #[test]
    fn codegen_matches_cpu_with_slide_iteration() {
        let prog = [
            OpNode { op: op::Z, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::C, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::SQR, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::ADD, a: 2, b: 1, kre: 0.0, kim: 0.0 },
        ];
        let mut params = base_params(&prog, &[]);
        params.box_bounds = Some(((-1.6, -1.6, -1.6), (1.6, 1.6, 1.6)));
        params.slide_iter = Some((5.0, 20.0));
        assert_matches_cpu(&prog, &[], params, 64, 64);
    }

    /// `hide_above` — exercises the NEW `hide_above_enabled`/`hide_above`
    /// shader branch — GPU-vs-CPU parity.
    #[test]
    fn codegen_matches_cpu_with_hide_above() {
        let prog = [
            OpNode { op: op::Z, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::C, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::SQR, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::ADD, a: 2, b: 1, kre: 0.0, kim: 0.0 },
        ];
        let mut params = base_params(&prog, &[]);
        params.box_bounds = Some(((-0.3, -0.3, -0.3), (0.3, 0.3, 0.3)));
        params.hide_above = Some(params.max_iter as f64);
        assert_matches_cpu(&prog, &[], params, 64, 64);
    }

    #[test]
    fn codegen_matches_cpu_with_julia_mode_and_a_warp_program() {
        let prog = [
            OpNode { op: op::Z, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::SQR, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::C, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::ADD, a: 1, b: 2, kre: 0.0, kim: 0.0 },
        ];
        let warp = [
            OpNode { op: op::Z, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::COS, a: 0, b: 0, kre: 0.0, kim: 0.0 },
        ];
        let mut params = base_params(&prog, &warp);
        params.formula.julia = true;
        params.formula.jc = (0.28, 0.53);
        assert_matches_cpu(&prog, &warp, params, 48, 48);
    }

    /// Exercises every opcode this codegen path implements a rule for,
    /// including CUBE/QUART/EXP/TANH/RECIP/CONJ/ABS*/NORMZ — the
    /// representative-program test above only touches Z/C/SIN/MUL/SUB/
    /// DIV/LOG, so this is the one that would actually catch a wrong rule
    /// for the rest.
    #[test]
    fn codegen_matches_cpu_exercising_every_remaining_opcode() {
        let prog = [
            OpNode { op: op::Z, a: 0, b: 0, kre: 0.0, kim: 0.0 },       // 0
            OpNode { op: op::CONST, a: 0, b: 0, kre: 0.4, kim: -0.2 },  // 1
            OpNode { op: op::ADD, a: 0, b: 1, kre: 0.0, kim: 0.0 },     // 2
            OpNode { op: op::CUBE, a: 2, b: 0, kre: 0.0, kim: 0.0 },    // 3
            OpNode { op: op::C, a: 0, b: 0, kre: 0.0, kim: 0.0 },       // 4
            OpNode { op: op::QUART, a: 4, b: 0, kre: 0.0, kim: 0.0 },   // 5
            OpNode { op: op::RECIP, a: 5, b: 0, kre: 0.0, kim: 0.0 },   // 6
            OpNode { op: op::EXP, a: 6, b: 0, kre: 0.0, kim: 0.0 },     // 7
            OpNode { op: op::TANH, a: 3, b: 0, kre: 0.0, kim: 0.0 },    // 8
            OpNode { op: op::CONJ, a: 7, b: 0, kre: 0.0, kim: 0.0 },    // 9
            OpNode { op: op::ABSFOLD, a: 8, b: 0, kre: 0.0, kim: 0.0 }, // 10
            OpNode { op: op::ABSRE, a: 9, b: 0, kre: 0.0, kim: 0.0 },   // 11
            OpNode { op: op::ABSIM, a: 10, b: 0, kre: 0.0, kim: 0.0 },  // 12
            OpNode { op: op::NORMZ, a: 11, b: 0, kre: 0.0, kim: 0.0 },  // 13
            OpNode { op: op::MUL, a: 12, b: 13, kre: 0.0, kim: 0.0 },   // 14
        ];
        assert_matches_cpu(&prog, &[], base_params(&prog, &[]), 48, 48);
    }

    #[test]
    fn generated_source_is_deterministic_and_contains_both_bodies() {
        let prog = [OpNode { op: op::Z, a: 0, b: 0, kre: 0.0, kim: 0.0 }];
        let src1 = generate_shader_source(&prog, &[]);
        let src2 = generate_shader_source(&prog, &[]);
        assert_eq!(src1, src2);
        assert!(src1.contains("fn eval_main_deriv"));
        assert!(src1.contains("fn eval_warp"));
        assert!(!src1.contains("GENERATED-CODE-MARKER"));
    }
}
