/// NNFractals interactive viewer.
///
/// Controls:
///   Drag box     Zoom into selected region (square)
///   Right-click  Zoom out 2×
///   Backspace    Undo last zoom
///   R            Reset to default view
///   S            Save PNG (prompts for resolution)
///   H / ? badge  Toggle help overlay
///   Q / Esc      Quit
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::mpsc;
use std::thread;

use font8x8::UnicodeFonts;
use softbuffer::{Context, Surface};
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalPosition};
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

use nnfractals::colormap::apply_colormap;
use nnfractals::config::Config;
use nnfractals::formula::apply_formula;
use nnfractals::genome::Genome;
use nnfractals::io::{load_genome, save_png};
#[cfg(feature = "wgpu-backend")]
use nnfractals::render_gpu;

// ── View ──────────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct View {
    cx: f32,
    cy: f32,
    zoom: f32,
}

impl View {
    fn bounds(&self) -> (f32, f32, f32, f32) {
        let half = 2.0 / self.zoom;
        (self.cx - half, self.cx + half, self.cy - half, self.cy + half)
    }

    fn pixel_to_fractal(&self, px: f64, py: f64, w: u32, h: u32) -> (f32, f32) {
        let (xmin, xmax, ymin, ymax) = self.bounds();
        (
            xmin + (px as f32 / w as f32) * (xmax - xmin),
            ymin + (py as f32 / h as f32) * (ymax - ymin),
        )
    }
}

// ── Interaction mode ──────────────────────────────────────────────────────────

enum Mode {
    Normal,
    Selecting { anchor: (f64, f64) },
    ResInput  { digits: String },
}

// ── CPU renderer ──────────────────────────────────────────────────────────────

/// Returns true when the per-pixel coordinate step has shrunk below what f32 can
/// resolve at this view's location — i.e. the GPU/f32 render would show blocky
/// pixelation and we must fall back to the f64 CPU path.
fn needs_f64(view: &View, w: u32) -> bool {
    let span      = 4.0 / view.zoom;                       // xmax - xmin
    let pixel_step = span / w.max(1) as f32;
    // Absolute f32 resolution near the view centre (magnitude × machine epsilon).
    let coord_mag = view.cx.abs().max(view.cy.abs()).max(1.0);
    let f32_ulp   = coord_mag * f32::EPSILON;              // ≈ coord_mag · 1.19e-7
    // Switch a bit before artifacts become visible (≈ 64 ulps per pixel).
    pixel_step < f32_ulp * 64.0
}

/// `compute_iter` controls how deep the iteration loop goes (used for fast
/// progressive passes). Colour normalisation ALWAYS uses the full
/// `config.rendering.max_iter`, so progressive passes only add boundary detail —
/// they never shift the palette. This keeps the first rough pass colour-matched
/// to the final image (and to the saved PNG).
///
/// When `use_f64` is set, the iteration runs in double precision on the CPU —
/// this is the deep-zoom path that eliminates f32 pixelation (WGSL/GPU is f32-only).
fn render_cpu(
    genome: &Genome, config: &Config, view: &View, w: u32, h: u32,
    compute_iter: u32, use_f64: bool,
) -> Vec<u8> {
    let color_iter = config.rendering.max_iter;

    // ── Deep-zoom path: f64 CPU iteration ────────────────────────────────────
    if use_f64 {
        use rayon::prelude::*;
        let view64 = (view.cx as f64, view.cy as f64, view.zoom as f64);
        let half   = 2.0 / view64.2;
        let (xmin, xmax) = (view64.0 - half, view64.0 + half);
        let (ymin, ymax) = (view64.1 - half, view64.1 + half);
        let fw: Vec<(f64, f64)> = genome.formula_weights()
            .iter().map(|&(r, i)| (r as f64, i as f64)).collect();
        let bailout_sq = (config.rendering.bailout * config.rendering.bailout) as f64;
        let wf = (w.saturating_sub(1)).max(1) as f64;
        let hf = (h.saturating_sub(1)).max(1) as f64;
        let escape_times: Vec<f32> = (0..(w * h) as usize)
            .into_par_iter()
            .map(|idx| {
                let cx = xmin + (idx % w as usize) as f64 / wf * (xmax - xmin);
                let cy = ymin + (idx / w as usize) as f64 / hf * (ymax - ymin);
                let (mut zx, mut zy) = (0.0f64, 0.0f64);
                for iter in 0..compute_iter {
                    let (nx, ny) = nnfractals::formula::f64_impl::apply_formula(&fw, zx, zy, cx, cy);
                    zx = nx; zy = ny;
                    let ms = zx * zx + zy * zy;
                    if ms > bailout_sq {
                        return (iter as f64 + 1.0 - (ms.log2() * 0.5).log2()).max(0.0) as f32;
                    }
                    if !zx.is_finite() || !zy.is_finite() { return iter as f32; }
                }
                color_iter as f32
            })
            .collect();
        return apply_colormap(&escape_times, color_iter, &config.rendering.colormap);
    }

    let (xmin, xmax, ymin, ymax) = view.bounds();
    let fw         = genome.formula_weights();
    let bailout_sq = config.rendering.bailout * config.rendering.bailout;

    // GPU path — much faster for interactive use.
    #[cfg(feature = "wgpu-backend")]
    if render_gpu::gpu_available() {
        let escape_times = render_gpu::render_fractal(
            &fw, w, h, compute_iter, xmin, xmax, ymin, ymax, bailout_sq,
        );
        return apply_colormap(&escape_times, color_iter, &config.rendering.colormap);
    }

    // CPU fallback (f32).
    use rayon::prelude::*;
    let wf = (w.saturating_sub(1)).max(1) as f32;
    let hf = (h.saturating_sub(1)).max(1) as f32;
    let escape_times: Vec<f32> = (0..(w * h) as usize)
        .into_par_iter()
        .map(|idx| {
            let cx = xmin + (idx % w as usize) as f32 / wf * (xmax - xmin);
            let cy = ymin + (idx / w as usize) as f32 / hf * (ymax - ymin);
            let (mut zx, mut zy) = (0.0f32, 0.0f32);
            for iter in 0..compute_iter {
                let (nx, ny) = apply_formula(&fw, zx, zy, cx, cy);
                zx = nx; zy = ny;
                let ms = zx * zx + zy * zy;
                if ms > bailout_sq {
                    return (iter as f32 + 1.0 - (ms.log2() * 0.5).log2()).max(0.0);
                }
                if !zx.is_finite() || !zy.is_finite() { return iter as f32; }
            }
            // Not yet escaped at this depth → treat as "inside" using the full
            // colour range so unfinished pixels match the final pass's interior.
            color_iter as f32
        })
        .collect();
    apply_colormap(&escape_times, color_iter, &config.rendering.colormap)
}

fn rgb_to_xrgb(rgb: &[u8], out: &mut Vec<u32>) {
    out.clear();
    out.reserve(rgb.len() / 3);
    for px in rgb.chunks_exact(3) {
        out.push(((px[0] as u32) << 16) | ((px[1] as u32) << 8) | (px[2] as u32));
    }
}

fn stretch_blit(src: &[u32], sw: usize, sh: usize, dst: &mut [u32], dw: usize, dh: usize) {
    for dy in 0..dh {
        let sy = (dy * sh / dh).min(sh.saturating_sub(1));
        for dx in 0..dw {
            let sx = (dx * sw / dw).min(sw.saturating_sub(1));
            dst[dy * dw + dx] = src[sy * sw + sx];
        }
    }
}

// ── Pixel drawing helpers ─────────────────────────────────────────────────────

fn draw_char(buf: &mut [u32], stride: usize, x: usize, y: usize, ch: char, fg: u32, scale: usize) {
    let ch = if ch.is_ascii() { ch } else { '?' };
    if let Some(glyph) = font8x8::BASIC_FONTS.get(ch) {
        for (row, &bits) in glyph.iter().enumerate() {
            for col in 0..8usize {
                if bits & (1u8 << col) != 0 {
                    for sy in 0..scale {
                        for sx in 0..scale {
                            let px = x + col * scale + sx;
                            let py = y + row * scale + sy;
                            let idx = py * stride + px;
                            if idx < buf.len() { buf[idx] = fg; }
                        }
                    }
                }
            }
        }
    }
}

fn draw_text(buf: &mut [u32], stride: usize, x: usize, y: usize, text: &str, fg: u32, scale: usize) {
    let cw = 8 * scale;
    for (i, ch) in text.chars().enumerate() {
        draw_char(buf, stride, x + i * cw, y, ch, fg, scale);
    }
}

/// Multiply every pixel in the region by `factor/255` (fade toward black).
fn darken_rect(buf: &mut [u32], stride: usize, x: usize, y: usize, w: usize, h: usize, factor: u32) {
    if stride == 0 || w == 0 || h == 0 { return; }
    let max_x = (x + w).min(stride);
    let rows   = buf.len() / stride;
    let max_y  = (y + h).min(rows);
    for row in y..max_y {
        for col in x..max_x {
            let idx = row * stride + col;
            let p = buf[idx];
            let r = (((p >> 16) & 0xFF) * factor / 255) << 16;
            let g = (((p >>  8) & 0xFF) * factor / 255) <<  8;
            let b =   (p        & 0xFF) * factor / 255;
            buf[idx] = r | g | b;
        }
    }
}

fn draw_rect_outline(buf: &mut [u32], stride: usize, x: usize, y: usize, w: usize, h: usize, color: u32) {
    if w == 0 || h == 0 { return; }
    for col in 0..w {
        let t = y * stride + x + col;
        let b = (y + h - 1) * stride + x + col;
        if t < buf.len() { buf[t] = color; }
        if b < buf.len() { buf[b] = color; }
    }
    for row in 0..h {
        let l = (y + row) * stride + x;
        let r = (y + row) * stride + x + w - 1;
        if l < buf.len() { buf[l] = color; }
        if r < buf.len() { buf[r] = color; }
    }
}

// ── Render channel ────────────────────────────────────────────────────────────

struct RenderRequest {
    view:       View,
    w:          u32,
    h:          u32,
    /// true → render at 1/4 res for fast exploration feedback; auto-upgrades when idle.
    preview:    bool,
    generation: u64,
}

struct RenderResult {
    pixels:     Vec<u32>,
    w:          u32,
    h:          u32,
    /// true = 1/4-res low-quality preview
    is_preview: bool,
    /// false = more progressive passes coming for this generation
    complete:   bool,
    generation: u64,
}

// ── Constants ─────────────────────────────────────────────────────────────────

const MIN_SEL_PX: f64  = 12.0;
const MAX_UNDO:   usize = 20;
const BADGE_W:    usize = 60;
const BADGE_H:    usize = 18;

// ── Application ───────────────────────────────────────────────────────────────

struct App {
    window:  Option<Rc<Window>>,
    surface: Option<Surface<Rc<Window>, Rc<Window>>>,
    sb_ctx:  Option<Context<Rc<Window>>>,

    genome:  Genome,
    config:  Config,
    nn_path: PathBuf,

    view:         View,
    default_view: View,
    view_stack:   Vec<View>,

    mouse_pos:       PhysicalPosition<f64>,
    mode:            Mode,
    show_help:       bool,
    save_resolution: u32,

    current_pixels:   Vec<u32>,
    current_w:        u32,
    current_h:        u32,
    current_preview:  bool,
    render_complete:  bool,   // false while progressive passes are still arriving
    render_gen:       u64,
    displayed_gen:    u64,

    req_tx: mpsc::SyncSender<RenderRequest>,
    res_rx: mpsc::Receiver<RenderResult>,
    proxy:  EventLoopProxy<()>,
}

impl App {
    fn new(nn_path: PathBuf, proxy: EventLoopProxy<()>) -> anyhow::Result<Self> {
        let config = Config::load(Path::new("config.toml"))
            .unwrap_or_else(|_| default_config());
        let genome = load_genome(&nn_path)?;

        let default_view = View {
            cx:   genome.view_cx,
            cy:   genome.view_cy,
            zoom: genome.view_zoom.max(0.1),
        };

        // Capacity 2 allows a preview + a full-quality request to queue simultaneously.
        let (req_tx, req_rx) = mpsc::sync_channel::<RenderRequest>(2);
        let (res_tx, res_rx) = mpsc::sync_channel::<RenderResult>(4);

        {
            let genome     = genome.clone();
            let config     = config.clone();
            let wake_proxy = proxy.clone();
            thread::spawn(move || {
                // Progressive iteration steps for full-quality renders.
                // Each step sends an intermediate result immediately so the window
                // shows something long before max_iter is reached.
                let full_iter = config.rendering.max_iter;
                let full_steps: &[u32] = &[8, 24, 64, full_iter];

                // `pending` carries a request that interrupted progressive refinement
                // so it gets rendered next instead of being discarded.
                let mut pending = req_rx.recv().ok();
                while let Some(req) = pending.take() {
                    let mut latest = req;
                    while let Ok(newer) = req_rx.try_recv() { latest = newer; }

                    // Deep zoom past f32 precision → use the f64 CPU path. It's slower,
                    // so cap its resolution (long side ≤ 720) and let stretch_blit upscale;
                    // structure, not pixel count, is what matters at depth.
                    let use_f64 = !latest.preview && needs_f64(&latest.view, latest.w);

                    let (rw, rh) = if latest.preview {
                        ((latest.w / 4).max(1), (latest.h / 4).max(1))
                    } else if use_f64 {
                        let long = latest.w.max(latest.h).max(1);
                        if long > 720 {
                            ((latest.w * 720 / long).max(1), (latest.h * 720 / long).max(1))
                        } else {
                            (latest.w, latest.h)
                        }
                    } else {
                        (latest.w, latest.h)
                    };

                    // Previews (1/4-res) are already tiny — one pass at full depth.
                    // Full-quality renders use progressive steps to show results fast.
                    let steps: &[u32] = if latest.preview {
                        &[full_iter]
                    } else {
                        full_steps
                    };

                    for (i, &max_iter) in steps.iter().enumerate() {
                        let max_iter  = max_iter.min(full_iter);
                        let is_last   = i == steps.len() - 1;
                        let rgb       = render_cpu(&genome, &config, &latest.view, rw, rh, max_iter, use_f64);
                        let mut pixels = Vec::new();
                        rgb_to_xrgb(&rgb, &mut pixels);
                        if res_tx.send(RenderResult {
                            pixels, w: rw, h: rh,
                            is_preview: latest.preview,
                            complete:   is_last,
                            generation: latest.generation,
                        }).is_err() { return; }
                        // Wake the event loop so the result is displayed immediately,
                        // without waiting for the next user input event.
                        wake_proxy.send_event(()).ok();

                        if is_last { break; }
                        // If a newer request arrived, stop refining this (stale) view and
                        // render the newer one next — but DON'T discard it.
                        if let Ok(newer) = req_rx.try_recv() {
                            pending = Some(newer);
                            break;
                        }
                    }

                    // If progressive refinement finished cleanly, block for the next request.
                    if pending.is_none() {
                        pending = req_rx.recv().ok();
                    }
                }
            });
        }

        Ok(Self {
            window: None, surface: None, sb_ctx: None,
            genome, config, nn_path,
            view: default_view.clone(),
            default_view,
            view_stack: Vec::new(),
            mouse_pos: PhysicalPosition::new(0.0, 0.0),
            mode: Mode::Normal,
            show_help: false,
            save_resolution: 2048,
            current_pixels: Vec::new(),
            current_w: 0, current_h: 0,
            current_preview: false,
            render_complete: true,
            render_gen: 0, displayed_gen: 0,
            req_tx, res_rx, proxy,
        })
    }

    fn win_size(&self) -> (u32, u32) {
        self.window.as_ref()
            .map(|w| { let s = w.inner_size(); (s.width.max(1), s.height.max(1)) })
            .unwrap_or((800, 800))
    }

    fn request_render(&mut self, preview: bool) {
        let (w, h) = self.win_size();
        self.render_gen += 1;
        let _ = self.req_tx.try_send(RenderRequest {
            view: self.view.clone(), w, h, preview,
            generation: self.render_gen,
        });
    }

    fn poll_render(&mut self) -> bool {
        let mut got = false;
        while let Ok(res) = self.res_rx.try_recv() {
            if res.generation >= self.displayed_gen {
                self.current_pixels  = res.pixels;
                self.current_w       = res.w;
                self.current_h       = res.h;
                self.current_preview = res.is_preview;
                self.render_complete = res.complete;
                self.displayed_gen   = res.generation;
                got = true;
            }
        }
        got
    }

    fn push_view(&mut self) {
        let old = self.view.clone();
        if self.view_stack.len() >= MAX_UNDO { self.view_stack.remove(0); }
        self.view_stack.push(old);
    }

    fn commit_selection(&mut self, anchor: (f64, f64)) {
        let (ax, ay) = anchor;
        let (cx, cy) = (self.mouse_pos.x, self.mouse_pos.y);
        let dx = cx - ax;
        let dy = cy - ay;
        let size = dx.abs().min(dy.abs());
        if size < MIN_SEL_PX { return; }

        let ex = ax + size * if dx >= 0.0 { 1.0 } else { -1.0 };
        let ey = ay + size * if dy >= 0.0 { 1.0 } else { -1.0 };
        let x1 = ax.min(ex);
        let y1 = ay.min(ey);
        let x2 = ax.max(ex);
        let y2 = ay.max(ey);

        let (w, h) = self.win_size();
        let (fx1, fy1) = self.view.pixel_to_fractal(x1, y1, w, h);
        let (fx2, fy2) = self.view.pixel_to_fractal(x2, y2, w, h);
        let new_cx   = (fx1 + fx2) * 0.5;
        let new_cy   = (fy1 + fy2) * 0.5;
        let new_zoom = ((self.view.zoom as f64) * (w as f64 / size)) as f32;

        self.push_view();
        self.view = View {
            cx:   new_cx,
            cy:   new_cy,
            zoom: new_zoom.clamp(0.05, 2_000_000.0),
        };
        self.request_render(true); // fast 1/4-res preview; about_to_wait upgrades to full
    }

    fn zoom_out(&mut self) {
        self.push_view();
        self.view.zoom = (self.view.zoom * 0.5).max(0.05);
        self.request_render(true);
    }

    fn undo_zoom(&mut self) {
        if let Some(prev) = self.view_stack.pop() {
            self.view = prev;
            self.request_render(true);
            if let Some(w) = &self.window { w.request_redraw(); }
        }
    }

    fn badge_origin(&self) -> (usize, usize) {
        let (w, h) = self.win_size();
        (w as usize - BADGE_W - 4, h as usize - BADGE_H - 4)
    }

    fn is_badge(&self, px: f64, py: f64) -> bool {
        let (bx, by) = self.badge_origin();
        px >= bx as f64 && px <= (bx + BADGE_W) as f64
            && py >= by as f64 && py <= (by + BADGE_H) as f64
    }

    fn selection_screen_rect(&self) -> Option<(usize, usize, usize, usize)> {
        let anchor = match &self.mode {
            Mode::Selecting { anchor } => *anchor,
            _ => return None,
        };
        let (ax, ay) = anchor;
        let (cx, cy) = (self.mouse_pos.x, self.mouse_pos.y);
        let dx = cx - ax;
        let dy = cy - ay;
        let size = dx.abs().min(dy.abs());
        if size < MIN_SEL_PX { return None; }
        let ex = ax + size * if dx >= 0.0 { 1.0 } else { -1.0 };
        let ey = ay + size * if dy >= 0.0 { 1.0 } else { -1.0 };
        let (ww, wh) = self.win_size();
        let x1 = ax.min(ex).max(0.0) as usize;
        let y1 = ay.min(ey).max(0.0) as usize;
        let x2 = (ax.max(ex) as usize).min(ww as usize);
        let y2 = (ay.max(ey) as usize).min(wh as usize);
        Some((x1, y1, x2, y2))
    }

    fn blit(&mut self) {
        // Pre-compute self-borrowed values before the surface buffer mutable borrow.
        let sel_rect   = self.selection_screen_rect();
        let badge      = self.badge_origin();
        let show_help  = self.show_help;
        let save_res   = self.save_resolution;
        let res_digits = if let Mode::ResInput { ref digits } = self.mode {
            Some(digits.clone())
        } else { None };
        let view_title = {
            let (xmin, xmax, _, _) = self.view.bounds();
            match &self.mode {
                Mode::ResInput { digits } => format!(
                    "NNFractals — save size: {digits}  (Enter=confirm  Esc=cancel)"
                ),
                _ => {
                    let spin = if !self.render_complete || self.displayed_gen < self.render_gen {
                        "  [rendering…]"
                    } else { "" };
                    let (w, _) = self.win_size();
                    let prec = if needs_f64(&self.view, w) { "  [f64]" } else { "" };
                    format!(
                        "NNFractals  ({:.4}, {:.4})  zoom {:.1}×  range {:.5}{}{}",
                        self.view.cx, self.view.cy, self.view.zoom, xmax - xmin, prec, spin
                    )
                }
            }
        };

        let Some(surface) = self.surface.as_mut() else { return };
        let Some(window)  = self.window.as_ref()  else { return };
        let s = window.inner_size();
        let (w, h) = (s.width.max(1), s.height.max(1));
        let (nw, nh) = match (NonZeroU32::try_from(w), NonZeroU32::try_from(h)) {
            (Ok(nw), Ok(nh)) => (nw, nh),
            _ => return,
        };
        if surface.resize(nw, nh).is_err() { return; }
        let Ok(mut buf) = surface.buffer_mut() else { return };
        let stride = w as usize;
        let bh     = h as usize;

        // ── 1. Fractal pixels ─────────────────────────────────────────────────
        let (sw, sh) = (self.current_w as usize, self.current_h as usize);
        if sw == 0 || sh == 0 {
            buf.fill(0);
        } else if sw == stride && sh == bh {
            buf.copy_from_slice(&self.current_pixels);
        } else {
            stretch_blit(&self.current_pixels, sw, sh, &mut buf, stride, bh);
        }

        // ── 2. Selection rectangle ────────────────────────────────────────────
        if let Some((x1, y1, x2, y2)) = sel_rect {
            let sw2 = x2.saturating_sub(x1).max(1);
            let sh2 = y2.saturating_sub(y1).max(1);
            if y1 > 0  { darken_rect(&mut buf, stride, 0,  0,  stride,      y1,       160); }
            if y2 < bh { darken_rect(&mut buf, stride, 0,  y2, stride,      bh - y2,  160); }
            if x1 > 0  { darken_rect(&mut buf, stride, 0,  y1, x1,          sh2,      160); }
            if x2 < stride { darken_rect(&mut buf, stride, x2, y1, stride - x2, sh2,  160); }
            draw_rect_outline(&mut buf, stride, x1, y1, sw2, sh2, 0xFFFFFF);
            if sw2 > 2 && sh2 > 2 {
                draw_rect_outline(&mut buf, stride, x1+1, y1+1, sw2-2, sh2-2, 0xFFCC00);
            }
        }

        // ── 3. Help overlay ───────────────────────────────────────────────────
        if show_help {
            const LINES: &[(&str, u32)] = &[
                (" CONTROLS           ", 0xFFFF88),
                ("                    ", 0x000000),
                (" Drag box  Zoom in  ", 0xDDDDDD),
                (" R-click   Zoom out ", 0xDDDDDD),
                (" Bksp      Undo     ", 0xDDDDDD),
                (" R         Reset    ", 0xDDDDDD),
                (" S         Save PNG ", 0xDDDDDD),
                (" H or ?    Help     ", 0xDDDDDD),
                (" Q / Esc   Quit     ", 0xDDDDDD),
            ];
            let scale = 2usize;
            let cw    = 8 * scale;
            let lh    = 8 * scale + 2;
            let pad   = 8usize;
            let max_c = LINES.iter().map(|(t, _)| t.len()).max().unwrap_or(0);
            let bx    = 8usize;
            let by    = 8usize;
            let bw    = max_c * cw + pad * 2;
            let bh2   = LINES.len() * lh + pad * 2;
            darken_rect(&mut buf, stride, bx, by, bw, bh2, 30);
            draw_rect_outline(&mut buf, stride, bx, by, bw, bh2, 0x666666);
            for (i, (text, color)) in LINES.iter().enumerate() {
                draw_text(&mut buf, stride, bx + pad, by + pad + i * lh, text, *color, scale);
            }
        }

        // ── 4. Resolution input prompt ────────────────────────────────────────
        if let Some(ref digits) = res_digits {
            let prompt = format!(
                "  {}x{} (was {}) Enter=save Esc=cancel  ",
                digits, digits, save_res
            );
            let scale = 2usize;
            let cw    = 8 * scale;
            let ch    = 8 * scale;
            let pad   = 10usize;
            let bw    = prompt.len() * cw + pad * 2;
            let bh2   = ch + pad * 2;
            let bx    = stride.saturating_sub(bw) / 2;
            let by2   = bh.saturating_sub(bh2) / 2;
            darken_rect(&mut buf, stride, bx, by2, bw, bh2, 25);
            draw_rect_outline(&mut buf, stride, bx, by2, bw, bh2, 0xFFFF88);
            draw_text(&mut buf, stride, bx + pad, by2 + pad, &prompt, 0xFFFF88, scale);
        }

        // ── 5. "? Help" badge ─────────────────────────────────────────────────
        {
            let (bx, by) = badge;
            let active   = show_help;
            darken_rect(&mut buf, stride, bx, by, BADGE_W, BADGE_H,
                if active { 80 } else { 20 });
            draw_rect_outline(&mut buf, stride, bx, by, BADGE_W, BADGE_H,
                if active { 0xFFFF88 } else { 0x666666 });
            let label = if active { "X HELP" } else { "? HELP" };
            let fg    = if active { 0xFFFF88 } else { 0x999999 };
            // scale=1: 8px tall chars, 6 chars × 8px = 48px. bx+6 centers in BADGE_W=60.
            draw_text(&mut buf, stride, bx + 6, by + 5, label, fg, 1);
        }

        // ── 6. Title bar ──────────────────────────────────────────────────────
        window.set_title(&view_title);
        let _ = buf.present();
    }
}

impl ApplicationHandler<()> for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        let attr = Window::default_attributes()
            .with_title("NNFractals Viewer")
            .with_inner_size(LogicalSize::new(800u32, 800u32));
        let win = Rc::new(el.create_window(attr).expect("create window"));
        let ctx = Context::new(win.clone()).expect("softbuffer context");
        let suf = Surface::new(&ctx, win.clone()).expect("softbuffer surface");
        self.window  = Some(win);
        self.sb_ctx  = Some(ctx);
        self.surface = Some(suf);
        // Go straight to the progressive full-res render: its first pass (8 iters)
        // is already ~20 ms, so a separate 1/4-res preview adds nothing but churn.
        self.request_render(false);
        if let Some(w) = &self.window { w.request_redraw(); }
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => el.exit(),

            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                // ── Resolution input mode ─────────────────────────────────────
                if matches!(self.mode, Mode::ResInput { .. }) {
                    let digits = if let Mode::ResInput { ref digits } = self.mode {
                        digits.clone()
                    } else { unreachable!() };

                    match &event.logical_key {
                        Key::Named(NamedKey::Escape) => {
                            self.mode = Mode::Normal;
                        }
                        Key::Named(NamedKey::Enter) => {
                            let res = digits.trim().parse::<u32>()
                                .ok().filter(|&r| r >= 64)
                                .unwrap_or(self.save_resolution);
                            self.save_resolution = res;
                            let genome = self.genome.clone();
                            let config = self.config.clone();
                            let view   = self.view.clone();
                            let path   = self.nn_path.clone();
                            self.mode  = Mode::Normal;
                            thread::spawn(move || {
                                eprintln!("Rendering {}×{} PNG…", res, res);
                                // Save at full requested resolution; use f64 when the
                                // zoom is deep enough that f32 would pixelate.
                                let save_f64 = needs_f64(&view, res);
                                let rgb = render_cpu(&genome, &config, &view, res, res, config.rendering.max_iter, save_f64);
                                let stem = path.file_stem()
                                    .and_then(|s| s.to_str()).unwrap_or("fractal");
                                let out = path.parent().unwrap_or(Path::new("."))
                                    .join(format!(
                                        "{stem}_cx{:.4}_cy{:.4}_z{:.2}_{res}px.png",
                                        view.cx, view.cy, view.zoom
                                    ));
                                match save_png(&rgb, res, res, &out) {
                                    Ok(_)  => eprintln!("Saved → {}", out.display()),
                                    Err(e) => eprintln!("Save error: {e}"),
                                }
                            });
                        }
                        Key::Named(NamedKey::Backspace) => {
                            let mut d = digits;
                            d.pop();
                            self.mode = Mode::ResInput { digits: d };
                        }
                        Key::Character(s) => {
                            let mut d = digits;
                            for ch in s.chars() {
                                if ch.is_ascii_digit() && d.len() < 6 { d.push(ch); }
                            }
                            self.mode = Mode::ResInput { digits: d };
                        }
                        _ => {}
                    }
                    if let Some(w) = &self.window { w.request_redraw(); }
                    return;
                }

                // ── Normal mode ───────────────────────────────────────────────
                match &event.logical_key {
                    Key::Character(s) if matches!(s.as_str(), "q" | "Q") => el.exit(),
                    Key::Named(NamedKey::Escape) => el.exit(),

                    Key::Character(s) if matches!(s.as_str(), "r" | "R") => {
                        self.view = self.default_view.clone();
                        self.view_stack.clear();
                        self.request_render(false);
                        if let Some(w) = &self.window { w.request_redraw(); }
                    }
                    Key::Character(s) if matches!(s.as_str(), "h" | "H") => {
                        self.show_help = !self.show_help;
                        if let Some(w) = &self.window { w.request_redraw(); }
                    }
                    Key::Character(s) if matches!(s.as_str(), "s" | "S") => {
                        self.mode = Mode::ResInput {
                            digits: self.save_resolution.to_string(),
                        };
                        if let Some(w) = &self.window { w.request_redraw(); }
                    }
                    Key::Named(NamedKey::Backspace) => {
                        self.undo_zoom();
                    }
                    _ => {}
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                self.mouse_pos = position;
                if matches!(self.mode, Mode::Selecting { .. }) {
                    if let Some(w) = &self.window { w.request_redraw(); }
                }
            }

            WindowEvent::MouseInput { button: MouseButton::Left, state, .. } => {
                match state {
                    ElementState::Pressed => {
                        if self.is_badge(self.mouse_pos.x, self.mouse_pos.y) {
                            self.show_help = !self.show_help;
                            if let Some(w) = &self.window { w.request_redraw(); }
                        } else {
                            self.mode = Mode::Selecting {
                                anchor: (self.mouse_pos.x, self.mouse_pos.y),
                            };
                        }
                    }
                    ElementState::Released => {
                        let anchor = match &self.mode {
                            Mode::Selecting { anchor } => Some(*anchor),
                            _ => None,
                        };
                        if let Some(anchor) = anchor {
                            self.mode = Mode::Normal;
                            self.commit_selection(anchor);
                            if let Some(w) = &self.window { w.request_redraw(); }
                        }
                    }
                }
            }

            WindowEvent::MouseInput {
                button: MouseButton::Right,
                state: ElementState::Released, ..
            } => {
                self.zoom_out();
                if let Some(w) = &self.window { w.request_redraw(); }
            }

            WindowEvent::Resized(_) => {
                self.request_render(false);
                if let Some(w) = &self.window { w.request_redraw(); }
            }

            WindowEvent::RedrawRequested => {
                self.poll_render();
                self.blit();
                // Keep requesting redraws while a render (or progressive pass) is in flight.
                if self.displayed_gen < self.render_gen || !self.render_complete {
                    if let Some(w) = &self.window { w.request_redraw(); }
                }
            }

            _ => {}
        }
    }

    fn user_event(&mut self, _: &ActiveEventLoop, _: ()) {
        // Render thread finished a pass — poll for the result and redraw.
        if self.poll_render() {
            if let Some(w) = &self.window { w.request_redraw(); }
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if self.poll_render() {
            if let Some(w) = &self.window { w.request_redraw(); }
        }
        // Auto-upgrade: only when a completed low-res preview has fully settled —
        // not while progressive full-quality passes are still arriving.
        if self.current_preview && self.render_complete && self.displayed_gen == self.render_gen {
            self.request_render(false);
        }
    }
}

// ── Fallback config ───────────────────────────────────────────────────────────

fn default_config() -> Config {
    use nnfractals::config::{DedupConfig, OptimizationConfig, OutputConfig, RenderingConfig};
    Config {
        dedup: DedupConfig::default(),
        rendering: RenderingConfig {
            default_width: 800, default_height: 800,
            max_iter: 256, bailout: 4.0,
            colormap: "turbo".into(),
            view_x_min: -2.0, view_x_max: 2.0,
            view_y_min: -2.0, view_y_max: 2.0,
        },
        optimization: OptimizationConfig {
            population_size: 40, elitism_count: 6,
            mutation_rate: 0.20, mutation_scale: 0.08,
            eval_width: 64, eval_height: 64, eval_max_iter: 128,
            restart_after_gens: 30, novelty_weight: 0.45,
            novelty_k: 5, archive_size: 150,
            self_replication_weight: 0.35,
            fractal_recursion_weight: 0.35,
        },
        output: OutputConfig {
            save_dir: "./fractals".into(),
            population_dir: "./populations".into(),
            min_entropy_prefilter: 0.42, max_entropy_prefilter: 0.65,
            min_clip_score: 0.512, min_laion_score: 5.30,
            min_beauty: 0.35, min_save_distance: 0.04,
        },
    }
}

// ── Entry ─────────────────────────────────────────────────────────────────────

fn main() -> anyhow::Result<()> {
    let nn_path = std::env::args().nth(1).map(PathBuf::from).ok_or_else(|| {
        anyhow::anyhow!("Usage: nnfractals-viewer <genome.nn>")
    })?;
    #[cfg(feature = "wgpu-backend")]
    {
        render_gpu::init_gpu();
        eprintln!("[viewer] Renderer: {}", if render_gpu::gpu_available() { "GPU (wgpu)" } else { "CPU (rayon fallback)" });
    }
    let el    = EventLoop::<()>::with_user_event().build()?;
    let proxy = el.create_proxy();
    let mut app = App::new(nn_path, proxy)?;
    el.run_app(&mut app)?;
    Ok(())
}
