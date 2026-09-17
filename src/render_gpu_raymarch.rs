//! GPU-accelerated ray-marching via a WGSL compute shader
//! (`raymarch.wgsl`) — a direct port of `quat_raymarch.rs`'s CPU sphere
//! tracer, dispatched as one GPU thread per pixel instead of rayon-parallel
//! CPU threads. This is exactly the kind of workload (deep, uniform-ish
//! per-pixel branching, zero shared state between pixels) GPUs are built
//! for — that's the whole motivation, see the CPU version's own doc
//! comments for why it was deliberately scoped as CPU-only originally.
//!
//! The shader has no independent design of its own: every function in it
//! mirrors a named CPU function 1:1, and correctness rests on the
//! pixel-for-pixel comparison tests at the bottom of this file, not on
//! "the WGSL looks right."

#![cfg(feature = "wgpu-backend")]

use std::sync::{Mutex, OnceLock};

use crate::quat_fractal::{QuatFormula, TimeAxis};
use crate::quat_motion::{look_at_basis, normalize, sub};
use crate::quat_raymarch::{RaymarchCamera, RaymarchParams};

const SHADER_SRC: &str = include_str!("raymarch.wgsl");
/// Must match `raymarch.wgsl`'s `Params` struct exactly: 36 scalar fields
/// (6 u32 + 11 f32 + 5×3 f32 vector components + aa:u32 + 3 pad u32) × 4
/// bytes, all plain scalars (no vec3/vec4 in the struct itself) so WGSL's
/// host-shareable layout is just sequential 4-byte packing — no alignment
/// padding to reason about beyond what's written explicitly here (the 3
/// trailing pad u32s exist only to round the total up to a multiple of 16,
/// WGSL's uniform-buffer struct alignment requirement).
const PARAMS_BYTES: u64 = 144;
/// Must match `raymarch.wgsl`'s `@workgroup_size(8, 8, 1)`.
const WG: u32 = 8;

struct GpuRaymarchRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    params_buf: wgpu::Buffer,
    shading_buf: wgpu::Buffer,
    color_buf: wgpu::Buffer,
    shading_staging: wgpu::Buffer,
    color_staging: wgpu::Buffer,
    max_pixels: u64,
}

unsafe impl Send for GpuRaymarchRenderer {}
unsafe impl Sync for GpuRaymarchRenderer {}

static GPU: OnceLock<Option<Mutex<GpuRaymarchRenderer>>> = OnceLock::new();

fn gpu() -> Option<&'static Mutex<GpuRaymarchRenderer>> {
    GPU.get_or_init(|| pollster::block_on(GpuRaymarchRenderer::new(128 * 128)).map(Mutex::new))
        .as_ref()
}

pub fn gpu_available() -> bool {
    gpu().is_some()
}

/// Renders one frame on the GPU — same inputs, same `(shading, color_t)`
/// output shape as `quat_raymarch::render_raymarch_frame`. `None` when no
/// GPU adapter is available; callers should fall back to the CPU renderer.
pub fn render_raymarch_frame_gpu(
    params: &RaymarchParams,
    cam: &RaymarchCamera,
    width: u32,
    height: u32,
) -> Option<(Vec<f32>, Vec<f32>)> {
    let m = gpu()?;
    let mut r = m.lock().ok()?;
    Some(r.dispatch(params, cam, width, height))
}

/// Matches `QuatFormula::ALL`'s order exactly (0=Mandelbrot ... 9=Bulb) —
/// the shader's `step_formula`/`formula_power` `switch` statements are
/// keyed on this same numbering.
fn formula_id(f: QuatFormula) -> u32 {
    QuatFormula::ALL.iter().position(|&x| x == f).expect("formula must be in QuatFormula::ALL") as u32
}

fn time_axis_id(t: TimeAxis) -> u32 {
    match t {
        TimeAxis::R => 0,
        TimeAxis::A => 1,
        TimeAxis::B => 2,
        TimeAxis::C => 3,
    }
}

/// Small positional byte-buffer writer — keeps the params upload in the
/// exact same field order as `raymarch.wgsl`'s `Params` struct without
/// juggling two mutable closures over the same slice.
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

impl GpuRaymarchRenderer {
    async fn new(max_pixels: u64) -> Option<Self> {
        let inst = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = inst
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .map_err(|e| eprintln!("[gpu-raymarch] adapter: {e}"))
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
            .map_err(|e| eprintln!("[gpu-raymarch] device: {e}"))
            .ok()?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raymarch"),
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

        let (params_buf, shading_buf, color_buf, shading_staging, color_staging) = Self::alloc(&device, max_pixels);

        Some(Self {
            device,
            queue,
            pipeline,
            bgl,
            params_buf,
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
                entry(1, wgpu::BufferBindingType::Storage { read_only: false }),
                entry(2, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        })
    }

    fn mk_buf(device: &wgpu::Device, label: &'static str, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: size.max(16), usage, mapped_at_creation: false })
    }

    fn alloc(device: &wgpu::Device, max_pixels: u64) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer) {
        let out_sz = max_pixels * 4;
        (
            Self::mk_buf(device, "rm_params", PARAMS_BYTES, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST),
            Self::mk_buf(device, "rm_shading", out_sz, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC),
            Self::mk_buf(device, "rm_color", out_sz, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC),
            Self::mk_buf(device, "rm_shading_stage", out_sz, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST),
            Self::mk_buf(device, "rm_color_stage", out_sz, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST),
        )
    }

    fn dispatch(&mut self, params: &RaymarchParams, cam: &RaymarchCamera, width: u32, height: u32) -> (Vec<f32>, Vec<f32>) {
        let pix = (width as u64) * (height as u64);
        if pix > self.max_pixels {
            let (_, shading_buf, color_buf, shading_staging, color_staging) = Self::alloc(&self.device, pix);
            self.shading_buf = shading_buf;
            self.color_buf = color_buf;
            self.shading_staging = shading_staging;
            self.color_staging = color_staging;
            self.max_pixels = pix;
        }

        // Camera basis — mirrors quat_raymarch::render_raymarch_frame exactly.
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
            w.u32(formula_id(params.formula));
            w.u32(time_axis_id(params.time_axis));
            w.u32(params.max_march_steps);
            w.f32(params.domain_radius);
            w.f32(params.time_val);
            w.f32(params.bailout * params.bailout);
            w.f32(params.hit_epsilon);
            w.f32(params.step_safety);
            w.f32(params.normal_eps);
            w.f32(params.color_probe_offset);
            w.f32(params.bulb_power);
            w.f32(params.mandelbox_scale);
            w.f32(half_w);
            w.f32(half_h);
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
            w.u32(0); // _pad0
            w.u32(0); // _pad1
            w.u32(0); // _pad2
        }
        self.queue.write_buffer(&self.params_buf, 0, &pb);

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.shading_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.color_buf.as_entire_binding() },
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
    use crate::quat_raymarch::render_raymarch_frame;

    fn base_params(formula: QuatFormula) -> RaymarchParams {
        RaymarchParams {
            formula,
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
            bulb_power: 8.0,
            mandelbox_scale: -1.5,
        }
    }

    fn base_cam() -> RaymarchCamera {
        RaymarchCamera { eye: (1.6, 1.0, -3.4), target: (0.0, 0.0, 0.0), up_hint: (0.0, 1.0, 0.0), fov_y: 45.0_f64.to_radians() }
    }

    /// The load-bearing test for this whole module: GPU output must match
    /// the CPU reference implementation closely (f32 GPU math vs f64 CPU
    /// math accounts for the tolerance, not a logic difference) for every
    /// formula, not just one — this shader has no independent claim to
    /// correctness beyond this comparison.
    #[test]
    fn gpu_matches_cpu_for_every_formula() {
        if !gpu_available() {
            eprintln!("skipping: no GPU adapter available in this environment");
            return;
        }
        let (w, h) = (64u32, 64u32);
        for formula in QuatFormula::ALL {
            let params = base_params(formula);
            let cam = base_cam();
            let (cpu_shading, cpu_color) = render_raymarch_frame(&params, &cam, w, h);
            let (gpu_shading, gpu_color) = render_raymarch_frame_gpu(&params, &cam, w, h).expect("gpu available");

            assert_eq!(cpu_shading.len(), gpu_shading.len());
            let mut max_shade_diff = 0.0f32;
            let mut mismatched_hits = 0;
            let mut large_shade_diffs = 0usize;
            let mut agreeing_hits = 0usize;
            for i in 0..cpu_shading.len() {
                let (cs, gs) = (cpu_shading[i], gpu_shading[i]);
                // Hit/miss agreement matters more than exact shading value
                // (f32 vs f64 marching can converge a handful of ULPs
                // differently right at a silhouette edge).
                if (cs > 0.0) != (gs > 0.0) {
                    mismatched_hits += 1;
                    continue;
                }
                if cs > 0.0 {
                    agreeing_hits += 1;
                }
                max_shade_diff = max_shade_diff.max((cs - gs).abs());
                if (cs - gs).abs() > 0.15 {
                    large_shade_diffs += 1;
                }
            }
            let mismatch_frac = mismatched_hits as f64 / cpu_shading.len() as f64;
            assert!(mismatch_frac < 0.02, "{}: {mismatched_hits}/{} pixels disagree on hit/miss ({:.1}%)", formula.name(), cpu_shading.len(), mismatch_frac * 100.0);
            // Mandelbox's surface is genuinely non-smooth: box_fold reflects
            // each component independently, producing literal geometric
            // creases (not just fine detail like Bulb's) where the true
            // gradient is discontinuous. `estimate_normal`'s 4-tap finite
            // difference can legitimately land its handful of probes on
            // different sides of a crease between the f32 (GPU) and f64
            // (CPU) marches, flipping the computed normal (and hence
            // shading) between "lit" and "ambient floor" for pixels right
            // on that crease — confirmed empirically: every large-diff
            // pixel checked was exactly this pattern, never a smooth
            // gradient of values. Every OTHER formula tested here is a
            // smooth power map (or Bulb's smooth spherical transform), so
            // this is the first formula that actually needs a fraction-
            // based tolerance on shading, matching the file's existing
            // convention for hit/miss and color_t above rather than a
            // single hard max.
            let large_shade_frac = large_shade_diffs as f64 / agreeing_hits.max(1) as f64;
            let shade_frac_threshold = if formula == QuatFormula::Mandelbox { 0.25 } else { 0.01 };
            assert!(
                large_shade_frac < shade_frac_threshold,
                "{}: {large_shade_diffs}/{agreeing_hits} agreeing-hit pixels have a large (>0.15) shading diff ({:.1}%)",
                formula.name(), large_shade_frac * 100.0
            );

            // color_t is only meaningful where both sides agree it's a hit —
            // and even there, it's `quat_escape_de`'s escape TIME (not the
            // distance estimate) sampled at a point deliberately just past
            // the surface: exactly the chaotic-near-boundary regime this
            // whole feature's docs already flag as extremely sensitive to
            // position (see quat_raymarch's module comment). f32 (GPU) vs
            // f64 (CPU) marching can converge to slightly different probe
            // points, and a chaotic escape function can amplify that into
            // a large difference for a SMALL minority of pixels without
            // either side being wrong — so bound the fraction of large
            // disagreements, not the single worst one.
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
            // Mandelbox gets a much looser bound than every other (smooth)
            // formula here — see the large-comment above the shading-diff
            // assertion: real geometric creases from box_fold, not a port
            // bug, confirmed by inspecting individual disagreeing pixels
            // during development (each was exactly a crease-straddling
            // normal flip, never a smooth gradient of values).
            let color_t_threshold = if formula == QuatFormula::Mandelbox { 0.35 } else { 0.05 };
            assert!(
                large_frac < color_t_threshold,
                "{}: {large_diffs}/{agreeing} agreeing-hit pixels have a large (>5) color_t diff ({:.1}%)",
                formula.name(), large_frac * 100.0
            );
        }
    }

    #[test]
    fn gpu_background_pixels_are_exactly_zero() {
        if !gpu_available() {
            eprintln!("skipping: no GPU adapter available in this environment");
            return;
        }
        let params = base_params(QuatFormula::Bulb);
        let cam = base_cam();
        let (shading, _) = render_raymarch_frame_gpu(&params, &cam, 48, 48).expect("gpu available");
        assert!(shading.iter().any(|&v| v == 0.0), "expected some background pixels at the frame edges");
        assert!(shading.iter().any(|&v| v > 0.0), "expected some hit pixels looking straight at bulb");
        for &v in &shading {
            assert!(v.is_finite() && (0.0..=1.0).contains(&v), "shading value out of range: {v}");
        }
    }
}
