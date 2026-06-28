// GPU fractal renderer — batched WGSL compute shader.
// All genomes render in ONE dispatch (Z dimension = genome index).
// Per-genome view bounds in buffer 2; pre-allocated reusable buffers.

#![cfg(feature = "wgpu-backend")]

use std::sync::{OnceLock, Mutex};

const SHADER_SRC: &str = include_str!("fractal.wgsl");
const WG_SIZE: u32 = 256;
const FW_F32S:   usize = 116; // 58 complex weights (legacy stride)
const PROG_F32S: usize = 248; // DAG stride: 120 main + 120 warp + 8 dynamics
const STRIDE_F32S: usize = 248; // max(FW_F32S, PROG_F32S): shared fw_buf sizing
const VIEW_F32S: usize = 4;   // xmin, xmax, ymin, ymax

pub struct GpuRenderer {
    device:      wgpu::Device,
    queue:       wgpu::Queue,
    pipeline:    wgpu::ComputePipeline,
    bgl:         wgpu::BindGroupLayout,
    params_buf:  wgpu::Buffer,
    fw_buf:      wgpu::Buffer,
    view_buf:    wgpu::Buffer,
    out_buf:     wgpu::Buffer,
    staging_buf: wgpu::Buffer,
    max_genomes: u32,
    max_pixels:  u32,
    // Output buffer capacity in f32 elements (gc × pix), tracked independently
    // from max_genomes × max_pixels to avoid blowing the 128 MB Vulkan binding limit.
    max_out:     u64,
}

unsafe impl Send for GpuRenderer {}
unsafe impl Sync for GpuRenderer {}

static GPU: OnceLock<Option<Mutex<GpuRenderer>>> = OnceLock::new();

pub fn init_gpu() {
    GPU.get_or_init(|| {
        match pollster::block_on(GpuRenderer::new(64, 128, 128)) {
            Some(mut r) => {
                // Warm up: compile the compute pipeline and prime the driver with a
                // real dispatch now, so the first interactive render isn't stalled by
                // multi-second lazy Vulkan pipeline compilation.
                let dummy_fw: Vec<(f32,f32)> = vec![(0.0,0.0); FW_F32S/2];
                let _ = r.dispatch(&[&dummy_fw], &[(-2.0,2.0,-2.0,2.0)], 64, 64, 16, 16.0);
                eprintln!("[gpu] Fractal WGSL compute shader ready (warmed up).");
                Some(Mutex::new(r))
            }
            None    => { eprintln!("[gpu] No wgpu adapter — using CPU renderer."); None }
        }
    });
}

pub fn gpu_available() -> bool {
    GPU.get().is_some_and(|o| o.is_some())
}

pub fn render_fractal(
    fw: &[(f32,f32)], w: u32, h: u32, mi: u32,
    xmin: f32, xmax: f32, ymin: f32, ymax: f32, bsq: f32,
) -> Vec<f32> {
    render_batch(&[fw], &[(xmin,xmax,ymin,ymax)], w, h, mi, bsq)
        .into_iter().next().unwrap_or_default()
}

/// One GPU dispatch for all genomes. `views[i]` = (xmin,xmax,ymin,ymax) for genome i.
pub fn render_batch(
    fw_batch: &[&[(f32,f32)]],
    views:    &[(f32,f32,f32,f32)],
    w: u32, h: u32, mi: u32, bsq: f32,
) -> Vec<Vec<f32>> {
    assert_eq!(fw_batch.len(), views.len());
    if let Some(Some(m)) = GPU.get() {
        if let Ok(mut r) = m.lock() {
            return r.dispatch(fw_batch, views, w, h, mi, bsq);
        }
    }
    fw_batch.iter().zip(views).map(|(fw, &(xmin,xmax,ymin,ymax))| {
        render_cpu_seq(fw, w, h, mi, xmin, xmax, ymin, ymax, bsq)
    }).collect()
}

pub fn render_cpu_seq(
    fw: &[(f32,f32)], w: u32, h: u32, mi: u32,
    xmin: f32, xmax: f32, ymin: f32, ymax: f32, bsq: f32,
) -> Vec<f32> {
    use crate::formula::apply_formula;
    let wf = (w.saturating_sub(1)).max(1) as f32;
    let hf = (h.saturating_sub(1)).max(1) as f32;
    (0..(w*h) as usize).map(|i| {
        let cx = xmin + (i%w as usize) as f32 / wf * (xmax-xmin);
        let cy = ymin + (i/w as usize) as f32 / hf * (ymax-ymin);
        let (mut zx, mut zy) = (0.0f32, 0.0f32);
        for it in 0..mi {
            let (nx,ny) = apply_formula(fw, zx, zy, cx, cy);
            zx = nx; zy = ny;
            let ms = zx*zx+zy*zy;
            if ms > bsq { return ((it as f32+1.0)-(ms.log2()*0.5).log2()).max(0.0); }
            if !zx.is_finite() || !zy.is_finite() { return it as f32; }
        }
        mi as f32
    }).collect()
}

// ── Expression-DAG GPU path ────────────────────────────────────────────────────

// Per-genome DAG upload layout (f32), DAG_F32S total = 248:
//   [0..120)    main program  (24 nodes × 5: op,a,b,kre,kim; op==255 terminates)
//   [120..240)  warp program  (same encoding; empty ⇒ first op == 255)
//   [240]       julia (0/1)   [241] julia_cre  [242] julia_cim
//   [243] phoenix_re  [244] phoenix_im  [245] bailout_sq  [246,247] reserved
const NODE_BLOCK: usize = 120; // N_SLOTS(24) × 5

/// A DAG genome's render inputs (borrowed from a Genome).
pub struct DagItem<'a> {
    pub prog:    &'a [crate::formula::OpNode],
    pub warp:    &'a [crate::formula::OpNode],
    pub julia:   bool,
    pub jc:      (f32, f32),
    pub phoenix: (f32, f32),
    pub bailout_sq: f32,
}

/// Build a DagItem borrowing a genome's program/warp + dynamics.
pub fn dag_item(g: &crate::genome::Genome) -> DagItem<'_> {
    DagItem {
        prog: &g.program,
        warp: &g.warp,
        julia: g.julia_mode,
        jc: (g.julia_cre, g.julia_cim),
        phoenix: (g.phoenix_re, g.phoenix_im),
        bailout_sq: g.bailout_radius * g.bailout_radius,
    }
}

/// Encode one node block (120 f32, sentinel-terminated).
fn encode_nodes(prog: &[crate::formula::OpNode]) -> [f32; NODE_BLOCK] {
    let mut out = [255.0f32; NODE_BLOCK];
    let n = prog.len().min(crate::formula::N_SLOTS);
    for (i, node) in prog.iter().take(n).enumerate() {
        let b = i * 5;
        out[b]     = node.op as f32;
        out[b + 1] = node.a  as f32;
        out[b + 2] = node.b  as f32;
        out[b + 3] = node.kre;
        out[b + 4] = node.kim;
    }
    out
}

/// Encode a full DAG item into the 248-f32 GPU layout.
fn encode_dag_item(item: &DagItem) -> Vec<f32> {
    let mut out = Vec::with_capacity(PROG_F32S);
    out.extend_from_slice(&encode_nodes(item.prog));
    out.extend_from_slice(&encode_nodes(item.warp));
    out.push(if item.julia { 1.0 } else { 0.0 });
    out.push(item.jc.0); out.push(item.jc.1);
    out.push(item.phoenix.0); out.push(item.phoenix.1);
    out.push(item.bailout_sq);
    out.push(0.0); out.push(0.0); // reserved
    debug_assert_eq!(out.len(), PROG_F32S);
    out
}

/// Batched render of expression-DAG genomes.
pub fn render_batch_dag(
    items: &[DagItem],
    views: &[(f32,f32,f32,f32)],
    w: u32, h: u32, mi: u32,
) -> Vec<Vec<f32>> {
    assert_eq!(items.len(), views.len());
    if let Some(Some(m)) = GPU.get() {
        if let Ok(mut r) = m.lock() {
            return r.dispatch_dag(items, views, w, h, mi);
        }
    }
    items.iter().zip(views).map(|(it, &(xmin,xmax,ymin,ymax))| {
        render_cpu_seq_dag(it, w, h, mi, xmin, xmax, ymin, ymax)
    }).collect()
}

pub fn render_cpu_seq_dag(
    item: &DagItem, w: u32, h: u32, mi: u32,
    xmin: f32, xmax: f32, ymin: f32, ymax: f32,
) -> Vec<f32> {
    let wf = (w.saturating_sub(1)).max(1) as f32;
    let hf = (h.saturating_sub(1)).max(1) as f32;
    (0..(w*h) as usize).map(|i| {
        let cx = xmin + (i%w as usize) as f32 / wf * (xmax-xmin);
        let cy = ymin + (i/w as usize) as f32 / hf * (ymax-ymin);
        crate::fractal::dag_escape_pixel(
            item.prog, item.warp, item.julia, item.jc, item.phoenix, item.bailout_sq,
            cx, cy, mi,
        )
    }).collect()
}

#[cfg(test)]
mod gpu_dag_tests {
    use super::*;
    use crate::formula::{op, OpNode};

    fn mandelbrot_prog() -> Vec<OpNode> {
        vec![
            OpNode { op: op::Z,   a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::C,   a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::SQR, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::ADD, a: 2, b: 1, kre: 0.0, kim: 0.0 },
        ]
    }

    // GPU DAG VM must match (a) the CPU DAG eval and (b) the legacy GPU z²+c path.
    #[test]
    fn gpu_dag_parity() {
        init_gpu();
        if !gpu_available() { eprintln!("[test] no GPU — skipping gpu_dag_parity"); return; }
        let (w, h, mi, bsq) = (64u32, 64u32, 128u32, 16.0f32);
        let view = (-2.0f32, 2.0, -2.0, 2.0);
        let prog = mandelbrot_prog();

        // Legacy z²+c weights: basis 0 (z²) = 1, basis 7 (c) = 1.
        let mut fw = vec![(0.0f32, 0.0f32); 58];
        fw[0] = (1.0, 0.0); fw[7] = (1.0, 0.0);

        // Plain Mandelbrot DAG item: no warp, no julia, no phoenix, bailout²=16.
        let item = DagItem { prog: &prog, warp: &[], julia: false, jc: (0.0,0.0), phoenix: (0.0,0.0), bailout_sq: bsq };
        let gpu_dag    = render_batch_dag(std::slice::from_ref(&item), &[view], w, h, mi).remove(0);
        let gpu_legacy = render_batch(&[&fw], &[view], w, h, mi, bsq).remove(0);
        let cpu_dag    = render_cpu_seq_dag(&item, w, h, mi, view.0, view.1, view.2, view.3);
        let cpu_legacy = render_cpu_seq(&fw, w, h, mi, view.0, view.1, view.2, view.3, bsq);

        // Within-hardware parity: the VM must reproduce the legacy basis path on
        // the SAME arithmetic (GPU↔GPU, CPU↔CPU). Cross GPU-vs-CPU is intentionally
        // not asserted — f32 rounding/FMA differences make escape time chaotic at
        // the fractal boundary (the legacy path has the identical property).
        let gpu_diff = gpu_dag.iter().zip(&gpu_legacy).map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);
        let cpu_diff = cpu_dag.iter().zip(&cpu_legacy).map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);
        eprintln!("[parity] gpu_dag↔gpu_legacy max={gpu_diff}  cpu_dag↔cpu_legacy max={cpu_diff}");
        assert!(gpu_diff < 1e-3, "GPU register-VM diverges from legacy basis path: {gpu_diff}");
        assert!(cpu_diff < 1e-3, "CPU eval_program diverges from legacy apply_formula: {cpu_diff}");
    }

    // Dynamics path (Julia + phoenix + warp): GPU and CPU must agree on the bulk
    // of pixels (boundary pixels are chaotic under f32, so compare the fraction
    // that match closely rather than the max).
    #[test]
    fn gpu_dag_dynamics_parity() {
        init_gpu();
        if !gpu_available() { eprintln!("[test] no GPU — skipping"); return; }
        let (w, h, mi) = (96u32, 96u32, 128u32);
        let view = (-1.8f32, 1.8, -1.8, 1.8);
        let prog = mandelbrot_prog();
        // warp: sin(pixel) — bends the plane
        let warp = vec![
            OpNode { op: op::Z, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::SIN, a: 0, b: 0, kre: 0.0, kim: 0.0 },
        ];
        let item = DagItem {
            prog: &prog, warp: &warp, julia: true, jc: (-0.4, 0.6),
            phoenix: (0.15, -0.05), bailout_sq: 16.0,
        };
        let gpu = render_batch_dag(std::slice::from_ref(&item), &[view], w, h, mi).remove(0);
        let cpu = render_cpu_seq_dag(&item, w, h, mi, view.0, view.1, view.2, view.3);
        let n = gpu.len();
        let close = gpu.iter().zip(&cpu).filter(|(a,b)| (**a - **b).abs() < 1.0).count();
        let frac = close as f32 / n as f32;
        // not degenerate: escape times must actually vary
        let mx = cpu.iter().cloned().fold(0.0f32, f32::max);
        let mn = cpu.iter().cloned().fold(f32::INFINITY, f32::min);
        eprintln!("[dyn-parity] {frac:.3} of pixels agree (<1.0); range [{mn},{mx}]");
        assert!(frac > 0.90, "GPU/CPU dynamics disagree on {:.1}% of pixels", (1.0-frac)*100.0);
        assert!(mx - mn > 1.0, "degenerate render (no structure)");
    }
}

// ── GpuRenderer ──────────────────────────────────────────────────────────────

impl GpuRenderer {
    async fn new(max_g: u32, max_w: u32, max_h: u32) -> Option<Self> {
        let inst = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = inst.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }).await.map_err(|e| eprintln!("[gpu] adapter: {e}")).ok()?;

        eprintln!("[gpu] Adapter: {} ({:?})", adapter.get_info().name, adapter.get_info().backend);

        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            experimental_features: Default::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }).await.map_err(|e| eprintln!("[gpu] device: {e}")).ok()?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fractal"), source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
        });

        let bgl = Self::make_bgl(&device);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[Some(&bgl)], immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None, layout: Some(&layout), module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None,
        });

        let max_pixels = max_w * max_h;
        let max_out    = (max_g as u64) * (max_pixels as u64);
        let (params_buf, fw_buf, view_buf, out_buf, staging_buf) =
            Self::alloc(&device, max_g, max_out);

        Some(Self { device, queue, pipeline, bgl,
            params_buf, fw_buf, view_buf, out_buf, staging_buf,
            max_genomes: max_g, max_pixels, max_out })
    }

    fn make_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        let entry = |binding: u32, ro: bool| wgpu::BindGroupLayoutEntry {
            binding, visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: if binding == 0 { wgpu::BufferBindingType::Uniform }
                    else { wgpu::BufferBindingType::Storage { read_only: ro } },
                has_dynamic_offset: false, min_binding_size: None,
            }, count: None,
        };
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[entry(0,true), entry(1,true), entry(2,true), entry(3,false)],
        })
    }

    fn mk_buf(device: &wgpu::Device, label: &'static str, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label), size: size.max(16), usage, mapped_at_creation: false,
        })
    }

    fn alloc(device: &wgpu::Device, max_g: u32, max_out: u64) -> (
        wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer
    ) {
        let fw_sz   = (max_g as u64) * (STRIDE_F32S as u64) * 4;
        let view_sz = (max_g as u64) * (VIEW_F32S as u64) * 4;
        let out_sz  = max_out * 4;
        (
            Self::mk_buf(device, "params", 32,      wgpu::BufferUsages::UNIFORM  | wgpu::BufferUsages::COPY_DST),
            Self::mk_buf(device, "fw",     fw_sz,   wgpu::BufferUsages::STORAGE  | wgpu::BufferUsages::COPY_DST),
            Self::mk_buf(device, "view",   view_sz, wgpu::BufferUsages::STORAGE  | wgpu::BufferUsages::COPY_DST),
            Self::mk_buf(device, "out",    out_sz,  wgpu::BufferUsages::STORAGE  | wgpu::BufferUsages::COPY_SRC),
            Self::mk_buf(device, "stage",  out_sz,  wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST),
        )
    }

    fn dispatch(
        &mut self,
        fw_batch: &[&[(f32,f32)]],
        views:    &[(f32,f32,f32,f32)],
        w: u32, h: u32, mi: u32, bsq: f32,
    ) -> Vec<Vec<f32>> {
        let gc  = fw_batch.len() as u32;
        let pix = w * h;

        // Grow input buffers (fw, view) when more genomes than ever seen.
        if gc > self.max_genomes {
            let fw_sz   = (gc as u64) * (STRIDE_F32S as u64) * 4;
            let view_sz = (gc as u64) * (VIEW_F32S as u64) * 4;
            self.fw_buf   = Self::mk_buf(&self.device, "fw",   fw_sz,   wgpu::BufferUsages::STORAGE  | wgpu::BufferUsages::COPY_DST);
            self.view_buf = Self::mk_buf(&self.device, "view", view_sz, wgpu::BufferUsages::STORAGE  | wgpu::BufferUsages::COPY_DST);
            self.max_genomes = gc;
        }
        if pix > self.max_pixels { self.max_pixels = pix; }

        // Grow output buffers only for the actual gc × pix needed this call.
        // This avoids the Vulkan 128 MB binding limit: e.g. 64 genomes × 800×800
        // = 163 MB would crash; viewer uses 1 genome × 800×800 = 2.5 MB.
        let out_needed = gc as u64 * pix as u64;
        if out_needed > self.max_out {
            let out_sz = out_needed * 4;
            self.out_buf     = Self::mk_buf(&self.device, "out",   out_sz, wgpu::BufferUsages::STORAGE  | wgpu::BufferUsages::COPY_SRC);
            self.staging_buf = Self::mk_buf(&self.device, "stage", out_sz, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
            self.max_out = out_needed;
        }

        // Params
        let mut pb = [0u8; 32];
        let p32 = |b: &mut [u8], o: usize, v: u32| b[o..o+4].copy_from_slice(&v.to_le_bytes());
        let pf  = |b: &mut [u8], o: usize, v: f32| b[o..o+4].copy_from_slice(&v.to_le_bytes());
        p32(&mut pb, 0, w); p32(&mut pb, 4, h); p32(&mut pb, 8, mi); p32(&mut pb, 12, gc);
        pf(&mut pb, 16, bsq);
        self.queue.write_buffer(&self.params_buf, 0, &pb);

        // FW weights
        let mut fw_bytes = Vec::with_capacity(fw_batch.len() * FW_F32S * 4);
        for fw in fw_batch {
            for &(re,im) in fw.iter() {
                fw_bytes.extend_from_slice(&re.to_le_bytes());
                fw_bytes.extend_from_slice(&im.to_le_bytes());
            }
        }
        self.queue.write_buffer(&self.fw_buf, 0, &fw_bytes);

        // View bounds
        let mut vb = Vec::with_capacity(views.len() * VIEW_F32S * 4);
        for &(xn,xx,yn,yx) in views {
            for v in [xn,xx,yn,yx] { vb.extend_from_slice(&v.to_le_bytes()); }
        }
        self.queue.write_buffer(&self.view_buf, 0, &vb);

        // Bind group + dispatch
        let out_sz = (gc as u64) * (pix as u64) * 4;
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.fw_buf.as_entire_binding()     },
                wgpu::BindGroupEntry { binding: 2, resource: self.view_buf.as_entire_binding()   },
                wgpu::BindGroupEntry { binding: 3, resource: self.out_buf.as_entire_binding()    },
            ],
        });

        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut p = enc.begin_compute_pass(&Default::default());
            p.set_pipeline(&self.pipeline);
            p.set_bind_group(0, &bg, &[]);
            p.dispatch_workgroups(pix.div_ceil(WG_SIZE), 1, gc);
        }
        enc.copy_buffer_to_buffer(&self.out_buf, 0, &self.staging_buf, 0, out_sz);
        self.queue.submit(Some(enc.finish()));

        // Readback
        let (tx, rx) = std::sync::mpsc::channel();
        self.staging_buf.slice(..out_sz).map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        rx.recv().unwrap().unwrap();

        let mapped = self.staging_buf.slice(..out_sz).get_mapped_range();
        let all: Vec<f32> = mapped.chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0],b[1],b[2],b[3]])).collect();
        drop(mapped);
        self.staging_buf.unmap();

        let p = pix as usize;
        (0..fw_batch.len()).map(|i| all[i*p..(i+1)*p].to_vec()).collect()
    }

    /// Expression-DAG dispatch: same pipeline as `dispatch` but uploads full DAG
    /// items (248 f32/genome: main+warp programs + dynamics) and sets use_dag=1.
    fn dispatch_dag(
        &mut self,
        items: &[DagItem],
        views: &[(f32,f32,f32,f32)],
        w: u32, h: u32, mi: u32,
    ) -> Vec<Vec<f32>> {
        let gc  = items.len() as u32;
        let pix = w * h;

        if gc > self.max_genomes {
            let fw_sz   = (gc as u64) * (STRIDE_F32S as u64) * 4;
            let view_sz = (gc as u64) * (VIEW_F32S as u64) * 4;
            self.fw_buf   = Self::mk_buf(&self.device, "fw",   fw_sz,   wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
            self.view_buf = Self::mk_buf(&self.device, "view", view_sz, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
            self.max_genomes = gc;
        }
        if pix > self.max_pixels { self.max_pixels = pix; }
        let out_needed = gc as u64 * pix as u64;
        if out_needed > self.max_out {
            let out_sz = out_needed * 4;
            self.out_buf     = Self::mk_buf(&self.device, "out",   out_sz, wgpu::BufferUsages::STORAGE  | wgpu::BufferUsages::COPY_SRC);
            self.staging_buf = Self::mk_buf(&self.device, "stage", out_sz, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
            self.max_out = out_needed;
        }

        // Params (use_dag = 1 at offset 20)
        let mut pb = [0u8; 32];
        let p32 = |b: &mut [u8], o: usize, v: u32| b[o..o+4].copy_from_slice(&v.to_le_bytes());
        let pf  = |b: &mut [u8], o: usize, v: f32| b[o..o+4].copy_from_slice(&v.to_le_bytes());
        p32(&mut pb, 0, w); p32(&mut pb, 4, h); p32(&mut pb, 8, mi); p32(&mut pb, 12, gc);
        // bailout_sq is per-genome for DAG (read from each block); params slot unused.
        pf(&mut pb, 16, 16.0); p32(&mut pb, 20, 1);
        self.queue.write_buffer(&self.params_buf, 0, &pb);

        // DAG item data: 248 f32 per genome (main program + warp program + dynamics)
        let mut prog_bytes = Vec::with_capacity(gc as usize * PROG_F32S * 4);
        for it in items {
            for v in encode_dag_item(it) { prog_bytes.extend_from_slice(&v.to_le_bytes()); }
        }
        self.queue.write_buffer(&self.fw_buf, 0, &prog_bytes);

        // View bounds
        let mut vb = Vec::with_capacity(views.len() * VIEW_F32S * 4);
        for &(xn,xx,yn,yx) in views {
            for v in [xn,xx,yn,yx] { vb.extend_from_slice(&v.to_le_bytes()); }
        }
        self.queue.write_buffer(&self.view_buf, 0, &vb);

        let out_sz = (gc as u64) * (pix as u64) * 4;
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.fw_buf.as_entire_binding()     },
                wgpu::BindGroupEntry { binding: 2, resource: self.view_buf.as_entire_binding()   },
                wgpu::BindGroupEntry { binding: 3, resource: self.out_buf.as_entire_binding()    },
            ],
        });

        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut p = enc.begin_compute_pass(&Default::default());
            p.set_pipeline(&self.pipeline);
            p.set_bind_group(0, &bg, &[]);
            p.dispatch_workgroups(pix.div_ceil(WG_SIZE), 1, gc);
        }
        enc.copy_buffer_to_buffer(&self.out_buf, 0, &self.staging_buf, 0, out_sz);
        self.queue.submit(Some(enc.finish()));

        let (tx, rx) = std::sync::mpsc::channel();
        self.staging_buf.slice(..out_sz).map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        rx.recv().unwrap().unwrap();

        let mapped = self.staging_buf.slice(..out_sz).get_mapped_range();
        let all: Vec<f32> = mapped.chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0],b[1],b[2],b[3]])).collect();
        drop(mapped);
        self.staging_buf.unmap();

        let p = pix as usize;
        (0..items.len()).map(|i| all[i*p..(i+1)*p].to_vec()).collect()
    }
}
