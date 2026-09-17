//! GPU-accelerated ray-marching for an ARBITRARY GA-evolved genome's
//! expression-DAG (`raymarch_dag.wgsl`) — the DAG-genome twin of
//! `render_gpu_raymarch.rs`. Unlike that module (a fixed 10-way `switch`
//! on `QuatFormula`), this uploads the genome's actual program as DATA
//! (a storage buffer of `DagNode`s) and the shader interprets it with a
//! general register-VM loop — the same architecture `eval_program_quat`/
//! `eval_program_quat_deriv` (`quat_dag.rs`) use on the CPU, ported
//! op-for-op, not redesigned.

#![cfg(feature = "wgpu-backend")]

use std::sync::{Mutex, OnceLock};

use crate::formula::OpNode;
use crate::quat_dag::RaymarchDagParams;
use crate::quat_fractal::TimeAxis;
use crate::quat_motion::{look_at_basis, normalize, sub};
use crate::quat_raymarch::RaymarchCamera;

const SHADER_SRC: &str = include_str!("raymarch_dag.wgsl");
/// Must match `raymarch_dag.wgsl`'s `Params` struct exactly: 39 plain
/// scalar fields (no vec3/vec4 in the struct itself, so WGSL's
/// host-shareable layout is sequential 4-byte packing — no alignment
/// padding to reason about, same approach `render_gpu_raymarch.rs`
/// already validated).
const PARAMS_BYTES: u64 = 39 * 4;
/// Bytes per `DagNode` (`op:u32, a:u32, b:u32, kre:f32, kim:f32`).
const NODE_BYTES: u64 = 20;
/// Must match `raymarch_dag.wgsl`'s `@workgroup_size(8, 8, 1)`.
const WG: u32 = 8;
/// `quat_dag::N_SLOTS` isn't public; mirrored here (also matches
/// `raymarch_dag.wgsl`'s own `N_SLOTS` constant) — both are the DAG
/// register-VM's hard node-count ceiling, unlikely to ever change without
/// a matching change to `formula.rs::N_SLOTS` itself.
const N_SLOTS: usize = 24;

struct GpuRaymarchDagRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    params_buf: wgpu::Buffer,
    prog_buf: wgpu::Buffer,
    warp_buf: wgpu::Buffer,
    shading_buf: wgpu::Buffer,
    color_buf: wgpu::Buffer,
    shading_staging: wgpu::Buffer,
    color_staging: wgpu::Buffer,
    max_pixels: u64,
}

unsafe impl Send for GpuRaymarchDagRenderer {}
unsafe impl Sync for GpuRaymarchDagRenderer {}

static GPU: OnceLock<Option<Mutex<GpuRaymarchDagRenderer>>> = OnceLock::new();

fn gpu() -> Option<&'static Mutex<GpuRaymarchDagRenderer>> {
    GPU.get_or_init(|| pollster::block_on(GpuRaymarchDagRenderer::new(128 * 128)).map(Mutex::new)).as_ref()
}

pub fn gpu_available() -> bool {
    gpu().is_some()
}

/// Renders one frame on the GPU — same inputs, same `(shading, color_t)`
/// output shape as `quat_dag::render_raymarch_dag_frame`. `None` if no GPU
/// adapter is available; callers should fall back to the CPU renderer.
pub fn render_raymarch_dag_frame_gpu(
    params: &RaymarchDagParams,
    cam: &RaymarchCamera,
    width: u32,
    height: u32,
) -> Option<(Vec<f32>, Vec<f32>)> {
    let m = gpu()?;
    let mut r = m.lock().ok()?;
    Some(r.dispatch(params, cam, width, height))
}

fn time_axis_id(t: TimeAxis) -> u32 {
    match t {
        TimeAxis::R => 0,
        TimeAxis::A => 1,
        TimeAxis::B => 2,
        TimeAxis::C => 3,
    }
}

fn encode_nodes(prog: &[OpNode]) -> Vec<u8> {
    let n = prog.len().min(N_SLOTS);
    let mut bytes = Vec::with_capacity(n.max(1) * NODE_BYTES as usize);
    for node in &prog[..n] {
        bytes.extend_from_slice(&(node.op as u32).to_le_bytes());
        bytes.extend_from_slice(&(node.a as u32).to_le_bytes());
        bytes.extend_from_slice(&(node.b as u32).to_le_bytes());
        bytes.extend_from_slice(&node.kre.to_le_bytes());
        bytes.extend_from_slice(&node.kim.to_le_bytes());
    }
    if bytes.is_empty() {
        bytes.resize(NODE_BYTES as usize, 0); // keep the buffer non-empty (min_binding_size safety)
    }
    bytes
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

impl GpuRaymarchDagRenderer {
    async fn new(max_pixels: u64) -> Option<Self> {
        let inst = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = inst
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .map_err(|e| eprintln!("[gpu-raymarch-dag] adapter: {e}"))
            .ok()?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                experimental_features: Default::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| eprintln!("[gpu-raymarch-dag] device: {e}"))
            .ok()?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raymarch_dag"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
        });

        let bgl = Self::make_bgl(&device);
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

        let (params_buf, prog_buf, warp_buf, shading_buf, color_buf, shading_staging, color_staging) = Self::alloc(&device, max_pixels);

        Some(Self {
            device,
            queue,
            pipeline,
            bgl,
            params_buf,
            prog_buf,
            warp_buf,
            shading_buf,
            color_buf,
            shading_staging,
            color_staging,
            max_pixels,
        })
    }

    fn make_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        let entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                entry(0, wgpu::BufferBindingType::Uniform),
                entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                entry(3, wgpu::BufferBindingType::Storage { read_only: false }),
                entry(4, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        })
    }

    fn mk_buf(device: &wgpu::Device, label: &'static str, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: size.max(16), usage, mapped_at_creation: false })
    }

    #[allow(clippy::type_complexity)]
    fn alloc(
        device: &wgpu::Device,
        max_pixels: u64,
    ) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer) {
        let out_sz = max_pixels * 4;
        let node_sz = (N_SLOTS as u64) * NODE_BYTES;
        (
            Self::mk_buf(device, "rmd_params", PARAMS_BYTES, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST),
            Self::mk_buf(device, "rmd_prog", node_sz, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST),
            Self::mk_buf(device, "rmd_warp", node_sz, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST),
            Self::mk_buf(device, "rmd_shading", out_sz, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC),
            Self::mk_buf(device, "rmd_color", out_sz, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC),
            Self::mk_buf(device, "rmd_shading_stage", out_sz, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST),
            Self::mk_buf(device, "rmd_color_stage", out_sz, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST),
        )
    }

    fn dispatch(&mut self, params: &RaymarchDagParams, cam: &RaymarchCamera, width: u32, height: u32) -> (Vec<f32>, Vec<f32>) {
        let pix = (width as u64) * (height as u64);
        if pix > self.max_pixels {
            let (_, _, _, shading_buf, color_buf, shading_staging, color_staging) = Self::alloc(&self.device, pix);
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
            w.u32(params.formula.prog.len().min(N_SLOTS) as u32);
            w.u32(params.formula.warp.len().min(N_SLOTS) as u32);
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
            w.u32(0);
            w.u32(0);
        }
        self.queue.write_buffer(&self.params_buf, 0, &pb);
        self.queue.write_buffer(&self.prog_buf, 0, &encode_nodes(params.formula.prog));
        self.queue.write_buffer(&self.warp_buf, 0, &encode_nodes(params.formula.warp));

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.prog_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.warp_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.shading_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.color_buf.as_entire_binding() },
            ],
        });

        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut p = enc.begin_compute_pass(&Default::default());
            p.set_pipeline(&self.pipeline);
            p.set_bind_group(0, &bg, &[]);
            p.dispatch_workgroups(width.div_ceil(WG), height.div_ceil(WG), 1);
        }
        let out_sz = pix * 4;
        enc.copy_buffer_to_buffer(&self.shading_buf, 0, &self.shading_staging, 0, out_sz);
        enc.copy_buffer_to_buffer(&self.color_buf, 0, &self.color_staging, 0, out_sz);
        self.queue.submit(Some(enc.finish()));

        let (tx1, rx1) = std::sync::mpsc::channel();
        self.shading_staging.slice(..out_sz).map_async(wgpu::MapMode::Read, move |r| {
            tx1.send(r).ok();
        });
        let (tx2, rx2) = std::sync::mpsc::channel();
        self.color_staging.slice(..out_sz).map_async(wgpu::MapMode::Read, move |r| {
            tx2.send(r).ok();
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
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
    use crate::formula::op;
    use crate::quat_dag::{render_raymarch_dag_frame, QuatDagFormula};

    fn base_params<'a>(prog: &'a [OpNode], warp: &'a [OpNode]) -> RaymarchDagParams<'a> {
        RaymarchDagParams {
            formula: QuatDagFormula { prog, warp, julia: false, jc: (0.0, 0.0), phoenix: (0.0, 0.0) },
            time_axis: TimeAxis::C,
            time_val: 0.0,
            domain_radius: 1.6,
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

    /// The load-bearing test: GPU output must match the CPU reference for
    /// a real, representative multi-op program (sin/log/div/mul — the
    /// same shape used in `quat_dag.rs`'s own CPU parity test), not just
    /// the trivial z²+c case.
    #[test]
    fn gpu_matches_cpu_for_a_representative_dag_program() {
        if !gpu_available() {
            eprintln!("skipping: no GPU adapter available in this environment");
            return;
        }
        let prog = [
            OpNode { op: op::Z, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::SIN, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::C, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::MUL, a: 0, b: 2, kre: 0.0, kim: 0.0 },
            OpNode { op: op::SUB, a: 3, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::DIV, a: 4, b: 1, kre: 0.0, kim: 0.0 },
            OpNode { op: op::LOG, a: 5, b: 0, kre: 0.0, kim: 0.0 },
        ];
        let (w, h) = (64u32, 64u32);
        let params = base_params(&prog, &[]);
        let cam = base_cam();
        let (cpu_shading, cpu_color) = render_raymarch_dag_frame(&params, &cam, w, h);
        let (gpu_shading, gpu_color) = render_raymarch_dag_frame_gpu(&params, &cam, w, h).expect("gpu available");

        assert_eq!(cpu_shading.len(), gpu_shading.len());
        let mut mismatched_hits = 0;
        let mut max_shade_diff = 0.0f32;
        for i in 0..cpu_shading.len() {
            let (cs, gs) = (cpu_shading[i], gpu_shading[i]);
            if (cs > 0.0) != (gs > 0.0) {
                mismatched_hits += 1;
                continue;
            }
            max_shade_diff = max_shade_diff.max((cs - gs).abs());
        }
        let mismatch_frac = mismatched_hits as f64 / cpu_shading.len() as f64;
        assert!(mismatch_frac < 0.05, "{mismatched_hits}/{} pixels disagree on hit/miss ({:.1}%)", cpu_shading.len(), mismatch_frac * 100.0);
        assert!(max_shade_diff < 0.15, "max shading diff {max_shade_diff} too large");

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
    fn gpu_matches_cpu_with_julia_mode_and_a_warp_program() {
        if !gpu_available() {
            eprintln!("skipping: no GPU adapter available in this environment");
            return;
        }
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
        let (w, h) = (48u32, 48u32);
        let cam = base_cam();
        let (cpu_shading, _) = render_raymarch_dag_frame(&params, &cam, w, h);
        let (gpu_shading, _) = render_raymarch_dag_frame_gpu(&params, &cam, w, h).expect("gpu available");
        let mut mismatched = 0;
        for i in 0..cpu_shading.len() {
            if (cpu_shading[i] > 0.0) != (gpu_shading[i] > 0.0) {
                mismatched += 1;
            }
        }
        let frac = mismatched as f64 / cpu_shading.len() as f64;
        assert!(frac < 0.05, "{mismatched}/{} pixels disagree on hit/miss under julia+warp ({:.1}%)", cpu_shading.len(), frac * 100.0);
    }

    #[test]
    fn gpu_background_pixels_are_exactly_zero() {
        if !gpu_available() {
            eprintln!("skipping: no GPU adapter available in this environment");
            return;
        }
        let prog = [
            OpNode { op: op::Z, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::SQR, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::C, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::ADD, a: 1, b: 2, kre: 0.0, kim: 0.0 },
        ];
        let params = base_params(&prog, &[]);
        let cam = base_cam();
        let (shading, _) = render_raymarch_dag_frame_gpu(&params, &cam, 48, 48).expect("gpu available");
        assert!(shading.iter().any(|&v| v == 0.0), "expected some background pixels");
        assert!(shading.iter().any(|&v| v > 0.0), "expected some hit pixels");
        for &v in &shading {
            assert!(v.is_finite() && (0.0..=1.0).contains(&v), "shading value out of range: {v}");
        }
    }
}
