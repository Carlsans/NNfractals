// GPU fractal renderer — batched WGSL compute shader.
// All genomes render in ONE dispatch (Z dimension = genome index).
// Per-genome view bounds in buffer 2; pre-allocated reusable buffers.

#![cfg(feature = "wgpu-backend")]

use std::sync::{OnceLock, Mutex};

const SHADER_SRC: &str = include_str!("fractal.wgsl");
const WG_SIZE: u32 = 256;
const FW_F32S:   usize = 116; // 58 complex weights
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
        let fw_sz   = (max_g as u64) * (FW_F32S   as u64) * 4;
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
            let fw_sz   = (gc as u64) * (FW_F32S   as u64) * 4;
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
}
