//! NNFractals interactive viewer (egui/eframe).
//!
//! Keyboard shortcuts:
//!   W/A/S/D               Translate view (+Shift=2×, +Alt=½, +Ctrl+Shift=10 radii, +Ctrl+Alt=1/10 radius)
//!   Up/Down arrows        Zoom in/out (same modifiers as WASD)
//!   Left/Right arrows     Cycle palette
//!   Drag (left btn)       Zoom into selection (aspect-locked)
//!   Right-click           Zoom out ×2
//!   Backspace / Ctrl+Z    Undo zoom
//!   R                     Reset view
//!   H / ?                 Toggle help
//!   Ctrl+S                Save PNG
//!   Q / Esc               Quit
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread;

use eframe::egui::{self, Color32, ColorImage, Key, TextureHandle, TextureOptions};
use serde::{Deserialize, Serialize};

use nnfractals::colormap::apply_colormap;
use nnfractals::config::Config;
use nnfractals::formula::apply_formula;
use nnfractals::genome::Genome;
use nnfractals::io::{load_genome, save_png};
#[cfg(feature = "wgpu-backend")]
use nnfractals::render_gpu;

// ── Constants ─────────────────────────────────────────────────────────────────

const MAX_UNDO: usize = 20;
const MIN_SEL_PX: f32 = 12.0;

const RATIOS: &[(&str, f64, f64)] = &[
    ("1:1",  1.0, 1.0),
    ("4:3",  4.0, 3.0),
    ("3:2",  3.0, 2.0),
    ("16:9", 16.0, 9.0),
    ("2:1",  2.0, 1.0),
];

const COLORMAPS: &[&str] = &[
    "turbo", "inferno", "viridis", "plasma", "magma", "earth", "neon",
];

// ── View ──────────────────────────────────────────────────────────────────────

#[derive(Clone)]
// Center + zoom stored as f64 for precision at deep zoom. `aspect` = xrange/yrange
// so non-square ratios are first-class; 1.0 = square (legacy default).
struct View {
    cx:     f64,
    cy:     f64,
    zoom:   f64,   // vertical: half_y = 2.0 / zoom
    aspect: f64,   // xrange / yrange
}

impl View {
    fn new_square(cx: f64, cy: f64, zoom: f64) -> Self {
        View { cx, cy, zoom, aspect: 1.0 }
    }

    fn bounds(&self) -> (f64, f64, f64, f64) {
        let half_y = 2.0 / self.zoom;
        let half_x = half_y * self.aspect;
        (self.cx - half_x, self.cx + half_x, self.cy - half_y, self.cy + half_y)
    }

    fn pixel_to_fractal(&self, px: f64, py: f64, w: f64, h: f64) -> (f64, f64) {
        let (xmin, xmax, ymin, ymax) = self.bounds();
        (
            xmin + (px / w) * (xmax - xmin),
            ymin + (py / h) * (ymax - ymin),
        )
    }
}

fn needs_f64(view: &View, w: u32) -> bool {
    let (xmin, xmax, _, _) = view.bounds();
    let span       = xmax - xmin;
    let pixel_step = span / w.max(1) as f64;
    let coord_mag  = view.cx.abs().max(view.cy.abs()).max(1.0);
    let f32_ulp    = coord_mag * f32::EPSILON as f64;
    pixel_step < f32_ulp * 64.0
}

// ── Render helpers ────────────────────────────────────────────────────────────

fn render_cpu(
    genome: &Genome, config: &Config, view: &View,
    w: u32, h: u32, compute_iter: u32, use_f64: bool,
) -> Vec<u8> {
    let color_iter = config.rendering.max_iter;
    let dag = genome.uses_program();

    if use_f64 {
        use rayon::prelude::*;
        let (xmin, xmax, ymin, ymax) = view.bounds();
        let wf = (w.saturating_sub(1)).max(1) as f64;
        let hf = (h.saturating_sub(1)).max(1) as f64;
        let fw: Vec<(f64, f64)> = if dag { Vec::new() } else {
            genome.formula_weights().iter().map(|&(r, i)| (r as f64, i as f64)).collect()
        };
        let legacy_bsq = (config.rendering.bailout * config.rendering.bailout) as f64;
        let dag_bsq    = (genome.bailout_radius * genome.bailout_radius) as f64;
        let jc         = (genome.julia_cre as f64, genome.julia_cim as f64);
        let phoenix    = (genome.phoenix_re as f64, genome.phoenix_im as f64);
        let escape_times: Vec<f32> = (0..(w * h) as usize)
            .into_par_iter()
            .map(|idx| {
                let cx = xmin + (idx % w as usize) as f64 / wf * (xmax - xmin);
                let cy = ymin + (idx / w as usize) as f64 / hf * (ymax - ymin);
                if dag {
                    return nnfractals::fractal::dag_escape_pixel_f64(
                        &genome.program, &genome.warp, genome.julia_mode, jc, phoenix,
                        dag_bsq, cx, cy, compute_iter,
                    );
                }
                let (mut zx, mut zy) = (0.0f64, 0.0f64);
                for iter in 0..compute_iter {
                    let (nx, ny) = nnfractals::formula::f64_impl::apply_formula(&fw, zx, zy, cx, cy);
                    zx = nx; zy = ny;
                    let ms = zx * zx + zy * zy;
                    if ms > legacy_bsq {
                        return (iter as f64 + 1.0 - (ms.log2() * 0.5).log2()).max(0.0) as f32;
                    }
                    if !zx.is_finite() || !zy.is_finite() { return iter as f32; }
                }
                color_iter as f32
            })
            .collect();
        return apply_colormap(&escape_times, color_iter, &config.rendering.colormap);
    }

    let (bxmin, bxmax, bymin, bymax) = view.bounds();
    let (xmin, xmax, ymin, ymax) = (bxmin as f32, bxmax as f32, bymin as f32, bymax as f32);
    let bailout_sq = config.rendering.bailout * config.rendering.bailout;

    #[cfg(feature = "wgpu-backend")]
    if render_gpu::gpu_available() {
        let escape_times = if dag {
            let item = render_gpu::dag_item(genome);
            render_gpu::render_batch_dag(
                std::slice::from_ref(&item), &[(xmin, xmax, ymin, ymax)], w, h, compute_iter,
            ).into_iter().next().unwrap_or_default()
        } else {
            let fw = genome.formula_weights();
            render_gpu::render_fractal(&fw, w, h, compute_iter, xmin, xmax, ymin, ymax, bailout_sq)
        };
        return apply_colormap(&escape_times, color_iter, &config.rendering.colormap);
    }

    use rayon::prelude::*;
    let fw      = genome.formula_weights();
    let dag_bsq = genome.bailout_radius * genome.bailout_radius;
    let jc      = (genome.julia_cre, genome.julia_cim);
    let phoenix = (genome.phoenix_re, genome.phoenix_im);
    let wf = (w.saturating_sub(1)).max(1) as f32;
    let hf = (h.saturating_sub(1)).max(1) as f32;
    let escape_times: Vec<f32> = (0..(w * h) as usize)
        .into_par_iter()
        .map(|idx| {
            let cx = xmin + (idx % w as usize) as f32 / wf * (xmax - xmin);
            let cy = ymin + (idx / w as usize) as f32 / hf * (ymax - ymin);
            if dag {
                return nnfractals::fractal::dag_escape_pixel(
                    &genome.program, &genome.warp, genome.julia_mode, jc, phoenix,
                    dag_bsq, cx, cy, compute_iter,
                );
            }
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
            color_iter as f32
        })
        .collect();
    apply_colormap(&escape_times, color_iter, &config.rendering.colormap)
}

/// Render at W×H with letterboxing to preserve the view's coordinate aspect ratio.
fn render_save(genome: &Genome, config: &Config, view: &View, w: u32, h: u32) -> Vec<u8> {
    let (xmin, xmax, ymin, ymax) = view.bounds();
    let view_ratio = (xmax - xmin) / (ymax - ymin);
    let img_ratio  = w as f64 / h as f64;

    let (fw, fh) = if img_ratio >= view_ratio {
        let fw = (h as f64 * view_ratio).round() as u32;
        (fw.max(1), h.max(1))
    } else {
        let fh = (w as f64 / view_ratio).round() as u32;
        (w.max(1), fh.max(1))
    };

    let use_f64 = needs_f64(view, fw);
    let fractal = render_cpu(genome, config, view, fw, fh, config.rendering.max_iter, use_f64);

    let mut canvas = vec![0u8; (w * h * 3) as usize];
    let ox = (w - fw) / 2;
    let oy = (h - fh) / 2;
    for row in 0..fh {
        let src = (row * fw * 3) as usize;
        let dst = ((oy + row) * w * 3 + ox * 3) as usize;
        let len = (fw * 3) as usize;
        if dst + len <= canvas.len() && src + len <= fractal.len() {
            canvas[dst..dst + len].copy_from_slice(&fractal[src..src + len]);
        }
    }
    canvas
}

// ── Render channel ────────────────────────────────────────────────────────────

struct RenderRequest {
    view:       View,
    w:          u32,
    h:          u32,
    preview:    bool,
    generation: u64,
    colormap:   String,
}

struct RenderResult {
    pixels:     Vec<u8>,  // RGB flat (3 bytes/pixel)
    w:          u32,
    h:          u32,
    is_preview: bool,
    complete:   bool,
    generation: u64,
}

// ── Preferences ───────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
struct ViewerPrefs {
    last_save_width:  u32,
    last_save_height: u32,
    ratio_label:      String,
    colormap:         String,
    window_width:     u32,
    window_height:    u32,
}

impl Default for ViewerPrefs {
    fn default() -> Self {
        Self {
            last_save_width:  1920,
            last_save_height: 1080,
            ratio_label:      "1:1".into(),
            colormap:         "turbo".into(),
            window_width:     1024,
            window_height:    768,
        }
    }
}

impl ViewerPrefs {
    fn load(path: &Path) -> Self {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|s| toml::from_str(&s).ok())
            .unwrap_or_default()
    }

    fn save(&self, path: &Path) {
        if let Ok(s) = toml::to_string_pretty(self) {
            let _ = std::fs::write(path, s);
        }
    }
}

// ── Application ───────────────────────────────────────────────────────────────

struct App {
    genome:       Genome,
    config:       Config,
    nn_path:      PathBuf,

    view:         View,
    default_view: View,
    view_stack:   Vec<View>,

    req_tx:          mpsc::SyncSender<RenderRequest>,
    res_rx:          mpsc::Receiver<RenderResult>,
    render_gen:      u64,
    displayed_gen:   u64,
    render_complete: bool,

    texture: Option<TextureHandle>,
    // fractal display area within the window, updated each frame
    frac_rect:  egui::Rect,
    prev_frac_dims: (u32, u32),  // track size changes to avoid redundant re-renders

    drag_start: Option<egui::Pos2>,

    show_help: bool,
    show_save: bool,
    save_w_str: String,
    save_h_str: String,

    ratio_idx:    usize,
    colormap_idx: usize,

    // XY bound fields — stored so they survive across frames while being edited
    xmin_str:  String,
    xmax_str:  String,
    ymin_str:  String,
    ymax_str:  String,
    sync_xy:   bool,  // true = view changed externally, refresh strings next frame

    prefs:      ViewerPrefs,
    prefs_path: PathBuf,
}

impl App {
    fn new(cc: &eframe::CreationContext, nn_path: PathBuf) -> anyhow::Result<Self> {
        let config = Config::load(Path::new("config.toml"))
            .unwrap_or_else(|_| default_config());
        let genome = load_genome(&nn_path)?;

        let default_view = View::new_square(
            genome.view_cx as f64,
            genome.view_cy as f64,
            genome.view_zoom.max(0.1) as f64,
        );

        let prefs_path = nn_path.parent().unwrap_or(Path::new("."))
            .join("viewer_prefs.toml");
        let prefs = ViewerPrefs::load(&prefs_path);

        // Find colormap index from prefs
        let colormap_idx = COLORMAPS.iter().position(|&c| c == prefs.colormap)
            .unwrap_or(0);
        let ratio_idx = RATIOS.iter().position(|(label, _, _)| *label == prefs.ratio_label)
            .unwrap_or(0);

        // Sync config colormap from prefs
        let mut config = config;
        config.rendering.colormap = COLORMAPS[colormap_idx].to_string();

        let ctx = cc.egui_ctx.clone();
        let (req_tx, req_rx) = mpsc::sync_channel::<RenderRequest>(2);
        let (res_tx, res_rx) = mpsc::sync_channel::<RenderResult>(4);

        {
            let genome     = genome.clone();
            let base_config = config.clone();
            thread::spawn(move || {
                let full_iter  = base_config.rendering.max_iter;
                let full_steps: &[u32] = &[8, 24, 64, full_iter];
                let mut config = base_config; // mutable so colormap can be updated per-request
                let mut pending = req_rx.recv().ok();
                while let Some(req) = pending.take() {
                    let mut latest = req;
                    while let Ok(newer) = req_rx.try_recv() { latest = newer; }

                    // Apply the palette from this request (may have changed since startup)
                    config.rendering.colormap = latest.colormap.clone();

                    let use_f64 = !latest.preview && needs_f64(&latest.view, latest.w);

                    let (rw, rh) = if latest.preview {
                        ((latest.w / 4).max(1), (latest.h / 4).max(1))
                    } else if use_f64 {
                        // cap resolution for f64 CPU path; egui scales texture up
                        let long = latest.w.max(latest.h).max(1);
                        if long > 720 {
                            ((latest.w * 720 / long).max(1), (latest.h * 720 / long).max(1))
                        } else {
                            (latest.w, latest.h)
                        }
                    } else {
                        (latest.w, latest.h)
                    };

                    let steps: &[u32] = if latest.preview { &[full_iter] } else { full_steps };

                    for (i, &iter) in steps.iter().enumerate() {
                        let is_last = i == steps.len() - 1;
                        let pixels  = render_cpu(&genome, &config, &latest.view,
                                                 rw, rh, iter.min(full_iter), use_f64);
                        if res_tx.send(RenderResult {
                            pixels, w: rw, h: rh,
                            is_preview: latest.preview,
                            complete: is_last,
                            generation: latest.generation,
                        }).is_err() { return; }
                        ctx.request_repaint();
                        if is_last { break; }
                        if let Ok(newer) = req_rx.try_recv() {
                            pending = Some(newer);
                            break;
                        }
                    }
                    if pending.is_none() {
                        pending = req_rx.recv().ok();
                    }
                }
            });
        }

        let initial_rect = egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(prefs.window_width as f32, prefs.window_height as f32),
        );

        let (xmin, xmax, ymin, ymax) = default_view.bounds();
        let mut app = Self {
            genome, config, nn_path,
            view: default_view.clone(),
            default_view,
            view_stack: Vec::new(),
            req_tx, res_rx,
            render_gen: 0, displayed_gen: 0, render_complete: true,
            texture: None,
            frac_rect: initial_rect,
            prev_frac_dims: (0, 0),
            drag_start: None,
            show_help: false,
            show_save: false,
            save_w_str: prefs.last_save_width.to_string(),
            save_h_str: prefs.last_save_height.to_string(),
            ratio_idx, colormap_idx,
            xmin_str: format!("{:.6}", xmin),
            xmax_str: format!("{:.6}", xmax),
            ymin_str: format!("{:.6}", ymin),
            ymax_str: format!("{:.6}", ymax),
            sync_xy: false,
            prefs, prefs_path,
        };
        // Set initial aspect ratio from prefs
        app.apply_ratio(ratio_idx, false);
        app.request_render(false);
        Ok(app)
    }

    fn request_render(&mut self, preview: bool) {
        let w = self.frac_rect.width().round() as u32;
        let h = self.frac_rect.height().round() as u32;
        if w == 0 || h == 0 { return; }
        self.render_gen += 1;
        let _ = self.req_tx.try_send(RenderRequest {
            view: self.view.clone(), w, h, preview,
            generation: self.render_gen,
            colormap: self.config.rendering.colormap.clone(),
        });
    }

    fn push_view(&mut self) {
        let old = self.view.clone();
        if self.view_stack.len() >= MAX_UNDO { self.view_stack.remove(0); }
        self.view_stack.push(old);
    }

    fn undo_zoom(&mut self) {
        if let Some(prev) = self.view_stack.pop() {
            self.view = prev;
            self.sync_xy = true;
            self.request_render(true);
        }
    }

    fn zoom_out(&mut self) {
        self.push_view();
        self.view.zoom = (self.view.zoom * 0.5).max(0.05);
        self.sync_xy = true;
        self.request_render(true);
    }

    fn current_aspect(&self) -> f64 {
        let (_, rw, rh) = RATIOS[self.ratio_idx];
        rw / rh
    }

    // Change the display aspect ratio, keeping cy and the y range.
    fn apply_ratio(&mut self, idx: usize, save_prefs: bool) {
        self.ratio_idx = idx;
        let (_, rw, rh) = RATIOS[idx];
        let new_asp = rw / rh;
        self.view.aspect = new_asp;
        self.sync_xy = true;
        if save_prefs {
            self.prefs.ratio_label = RATIOS[idx].0.to_string();
            self.prefs.save(&self.prefs_path);
        }
    }

    fn set_colormap(&mut self, idx: usize) {
        self.colormap_idx = idx;
        self.config.rendering.colormap = COLORMAPS[idx].to_string();
        self.prefs.colormap = COLORMAPS[idx].to_string();
        self.prefs.save(&self.prefs_path);
        self.request_render(false);
    }

    fn update_view_from_bounds(&mut self, xmin: f64, xmax: f64, ymin: f64, ymax: f64) {
        if xmax <= xmin || ymax <= ymin { return; }
        self.view.cx     = (xmin + xmax) / 2.0;
        self.view.cy     = (ymin + ymax) / 2.0;
        let yrange       = ymax - ymin;
        let xrange       = xmax - xmin;
        self.view.zoom   = (4.0 / yrange).clamp(0.05, 1.0e13);
        self.view.aspect = xrange / yrange;
        self.sync_xy = true;
    }

    fn poll_render(&mut self, ctx: &egui::Context) -> bool {
        let mut got = false;
        while let Ok(res) = self.res_rx.try_recv() {
            if res.generation >= self.displayed_gen {
                let image = ColorImage::from_rgb([res.w as usize, res.h as usize], &res.pixels);
                self.texture = Some(ctx.load_texture("fractal", image, TextureOptions::LINEAR));
                self.render_complete = res.complete;
                self.displayed_gen   = res.generation;
                got = true;
            }
        }
        got
    }

    fn show_toolbar(&mut self, ui: &mut egui::Ui) {
        let win_h = ui.ctx().input(|i| i.viewport_rect().height());
        let toolbar_h = (win_h * 0.055).clamp(28.0, 58.0);
        let font_size = (toolbar_h * 0.55).clamp(12.0, 28.0);

        egui::Panel::top("toolbar")
            .exact_size(toolbar_h)
            .show(ui, |ui| {
                // Scale all button text to match toolbar height
                {
                    let style = ui.style_mut();
                    style.text_styles.insert(
                        egui::TextStyle::Button,
                        egui::FontId::proportional(font_size),
                    );
                    style.text_styles.insert(
                        egui::TextStyle::Body,
                        egui::FontId::proportional(font_size * 0.85),
                    );
                    style.text_styles.insert(
                        egui::TextStyle::Monospace,
                        egui::FontId::monospace(font_size * 0.80),
                    );
                }

                egui::ScrollArea::horizontal().show(ui, |ui| {
                    ui.horizontal_centered(|ui| {
                        // ── Translation arrows ──────────────────────────────
                        if ui.button("◀").on_hover_text("A — left (Shift=2×, Alt=½)").clicked() {
                            self.do_translate(-1.0, 0.0);
                        }
                        if ui.button("▲").on_hover_text("W — up").clicked() {
                            self.do_translate(0.0, -1.0);
                        }
                        if ui.button("▼").on_hover_text("S — down").clicked() {
                            self.do_translate(0.0, 1.0);
                        }
                        if ui.button("▶").on_hover_text("D — right").clicked() {
                            self.do_translate(1.0, 0.0);
                        }

                        ui.separator();

                        // ── Zoom / reset ────────────────────────────────────
                        if ui.button("⊕").on_hover_text("↑ — zoom in").clicked() {
                            self.do_zoom(true, 1.0);
                        }
                        if ui.button("⊖").on_hover_text("↓ — zoom out").clicked() {
                            self.do_zoom(false, 1.0);
                        }
                        if ui.button("⟳").on_hover_text("R — reset view").clicked() {
                            self.view = self.default_view.clone();
                            self.view.aspect = self.current_aspect();
                            self.view_stack.clear();
                            self.sync_xy = true;
                            self.request_render(false);
                        }

                        ui.separator();

                        // ── XY coordinate fields ────────────────────────────
                        let (xmin, xmax, ymin, ymax) = self.view.bounds();
                        if self.sync_xy {
                            self.xmin_str = format!("{:.6}", xmin);
                            self.xmax_str = format!("{:.6}", xmax);
                            self.ymin_str = format!("{:.6}", ymin);
                            self.ymax_str = format!("{:.6}", ymax);
                            self.sync_xy = false;
                        }

                        let field_w = font_size * 5.5;
                        ui.label("x:");
                        let rx = ui.add(egui::TextEdit::singleline(&mut self.xmin_str)
                            .desired_width(field_w).font(egui::TextStyle::Monospace));
                        if rx.lost_focus() {
                            if let Ok(v) = self.xmin_str.trim().parse::<f64>() {
                                let (_, cx, cy, cy2) = self.view.bounds();
                                self.push_view();
                                self.update_view_from_bounds(v, cx, cy, cy2);
                                self.request_render(false);
                            }
                        }
                        let rx = ui.add(egui::TextEdit::singleline(&mut self.xmax_str)
                            .desired_width(field_w).font(egui::TextStyle::Monospace));
                        if rx.lost_focus() {
                            if let Ok(v) = self.xmax_str.trim().parse::<f64>() {
                                let (cx, _, cy, cy2) = self.view.bounds();
                                self.push_view();
                                self.update_view_from_bounds(cx, v, cy, cy2);
                                self.request_render(false);
                            }
                        }

                        ui.label("y:");
                        let ry = ui.add(egui::TextEdit::singleline(&mut self.ymin_str)
                            .desired_width(field_w).font(egui::TextStyle::Monospace));
                        if ry.lost_focus() {
                            if let Ok(v) = self.ymin_str.trim().parse::<f64>() {
                                let (cx, cx2, _, cy2) = self.view.bounds();
                                self.push_view();
                                self.update_view_from_bounds(cx, cx2, v, cy2);
                                self.request_render(false);
                            }
                        }
                        let ry = ui.add(egui::TextEdit::singleline(&mut self.ymax_str)
                            .desired_width(field_w).font(egui::TextStyle::Monospace));
                        if ry.lost_focus() {
                            if let Ok(v) = self.ymax_str.trim().parse::<f64>() {
                                let (cx, cx2, cy, _) = self.view.bounds();
                                self.push_view();
                                self.update_view_from_bounds(cx, cx2, cy, v);
                                self.request_render(false);
                            }
                        }

                        ui.separator();

                        // ── Aspect ratio ────────────────────────────────────
                        let ratio_label = RATIOS[self.ratio_idx].0;
                        egui::ComboBox::from_id_salt("ratio")
                            .selected_text(ratio_label)
                            .show_ui(ui, |ui| {
                                for (i, (label, _, _)) in RATIOS.iter().enumerate() {
                                    if ui.selectable_label(i == self.ratio_idx, *label).clicked() {
                                        self.apply_ratio(i, true);
                                        self.request_render(false);
                                    }
                                }
                            });

                        ui.separator();

                        // ── Palette ─────────────────────────────────────────
                        if ui.button("◁").on_hover_text("← — previous palette").clicked() {
                            let n = COLORMAPS.len();
                            self.set_colormap((self.colormap_idx + n - 1) % n);
                        }
                        ui.label(COLORMAPS[self.colormap_idx]);
                        if ui.button("▷").on_hover_text("→ — next palette").clicked() {
                            self.set_colormap((self.colormap_idx + 1) % COLORMAPS.len());
                        }

                        ui.separator();

                        // ── Help / Save ─────────────────────────────────────
                        let help_label = if self.show_help { "✕ Help" } else { "? Help" };
                        if ui.button(help_label).clicked() {
                            self.show_help = !self.show_help;
                        }
                        if ui.button("💾 Save").on_hover_text("Ctrl+S").clicked() {
                            self.show_save = true;
                        }

                        // Status indicator (right-aligned)
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if !self.render_complete || self.displayed_gen < self.render_gen {
                                ui.colored_label(Color32::YELLOW, "⟳ rendering…");
                            } else {
                                let w = self.frac_rect.width() as u32;
                                if needs_f64(&self.view, w) {
                                    ui.colored_label(Color32::from_rgb(100, 200, 255), "f64");
                                }
                            }
                        });
                    });
                });
            });
    }

    fn show_fractal_panel(&mut self, ui: &mut egui::Ui) {
        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(Color32::BLACK))
            .show(ui, |ui| {
                let avail = ui.available_size();
                let asp   = self.current_aspect() as f32;
                let (fw, fh) = if avail.x / avail.y >= asp {
                    (avail.y * asp, avail.y)
                } else {
                    (avail.x, avail.x / asp)
                };

                let offset_x = (avail.x - fw) / 2.0;
                let offset_y = (avail.y - fh) / 2.0;
                let panel_min = ui.min_rect().min;
                let frac_min = egui::Pos2::new(panel_min.x + offset_x, panel_min.y + offset_y);
                let new_rect = egui::Rect::from_min_size(frac_min, egui::Vec2::new(fw, fh));

                // Trigger re-render if fractal area dimensions changed
                let new_dims = (fw.round() as u32, fh.round() as u32);
                if new_dims != self.prev_frac_dims && new_dims.0 > 0 && new_dims.1 > 0 {
                    self.frac_rect = new_rect;
                    self.prev_frac_dims = new_dims;
                    self.request_render(false);
                }
                self.frac_rect = new_rect;

                // Draw fractal texture
                if let Some(tex) = &self.texture {
                    let uv  = egui::Rect::from_min_max(egui::Pos2::ZERO, egui::Pos2::new(1.0, 1.0));
                    ui.painter().image(tex.id(), new_rect, uv, Color32::WHITE);
                }

                // Interaction area (drag + click)
                let resp = ui.allocate_rect(
                    new_rect,
                    egui::Sense::click_and_drag(),
                );

                // Right-click → zoom out
                if resp.secondary_clicked() {
                    self.zoom_out();
                }

                // Drag → selection rectangle
                if resp.drag_started() {
                    if let Some(p) = resp.interact_pointer_pos() {
                        self.drag_start = Some(p);
                    }
                }
                if resp.drag_stopped() {
                    if let (Some(start), Some(end)) = (self.drag_start.take(), resp.interact_pointer_pos()) {
                        self.commit_selection(start, end, fw, fh);
                    }
                    self.drag_start = None;
                }

                // Draw selection overlay
                if let (Some(start), Some(cur)) = (self.drag_start, ui.ctx().input(|i| i.pointer.latest_pos())) {
                    let (sel_rect, ok) = selection_rect(start, cur, fw / fh);
                    if ok {
                        let painter = ui.painter();
                        painter.rect_stroke(sel_rect, 0.0, egui::Stroke::new(2.0, Color32::WHITE), egui::StrokeKind::Middle);
                        painter.rect_stroke(
                            sel_rect.shrink(1.5),
                            0.0,
                            egui::Stroke::new(1.0, Color32::from_rgb(255, 200, 0)),
                            egui::StrokeKind::Middle,
                        );
                    }
                }

                // Auto-upgrade preview to full when settled
                if self.texture.is_some() {
                    let result_is_preview = self.displayed_gen == self.render_gen && self.render_complete;
                    if result_is_preview {
                        // already full — nothing to do
                    }
                }
            });
    }

    fn do_translate(&mut self, dx_sign: f64, dy_sign: f64) {
        self.do_translate_scaled(dx_sign, dy_sign, 1.0);
    }

    fn do_translate_scaled(&mut self, dx_sign: f64, dy_sign: f64, scale: f64) {
        let (xmin, xmax, ymin, ymax) = self.view.bounds();
        let radius = (xmax - xmin) / 2.0;
        let step   = radius / 3.0 * scale;
        self.push_view();
        self.view.cx += dx_sign * step;
        self.view.cy += dy_sign * step;
        self.sync_xy = true;
        self.request_render(true);
    }

    fn do_zoom(&mut self, zoom_in: bool, scale: f64) {
        // base factor: zoom by (3/2)^scale per press
        let factor = (1.5_f64).powf(scale);
        self.push_view();
        if zoom_in {
            self.view.zoom = (self.view.zoom * factor).clamp(0.05, 1.0e13);
        } else {
            self.view.zoom = (self.view.zoom / factor).clamp(0.05, 1.0e13);
        }
        self.sync_xy = true;
        self.request_render(true);
    }

    fn commit_selection(&mut self, start: egui::Pos2, end: egui::Pos2, fw: f32, fh: f32) {
        let (sel_rect, ok) = selection_rect(start, end, fw / fh);
        if !ok { return; }

        let (fx1, fy1) = self.view.pixel_to_fractal(
            (sel_rect.min.x - self.frac_rect.min.x) as f64,
            (sel_rect.min.y - self.frac_rect.min.y) as f64,
            fw as f64, fh as f64,
        );
        let (fx2, fy2) = self.view.pixel_to_fractal(
            (sel_rect.max.x - self.frac_rect.min.x) as f64,
            (sel_rect.max.y - self.frac_rect.min.y) as f64,
            fw as f64, fh as f64,
        );
        let new_cx = (fx1 + fx2) * 0.5;
        let new_cy = (fy1 + fy2) * 0.5;
        // zoom: sel_w / fw = fraction of view covered
        let sel_frac = sel_rect.width() / fw;
        let new_zoom = (self.view.zoom / sel_frac as f64).clamp(0.05, 1.0e13);

        self.push_view();
        self.view.cx   = new_cx;
        self.view.cy   = new_cy;
        self.view.zoom = new_zoom;
        self.sync_xy = true;
        self.request_render(true);
    }

    fn handle_keyboard(&mut self, ctx: &egui::Context) {
        // Don't steal keys while a text field has focus
        let any_focused = ctx.memory(|m| m.focused().is_some());
        if any_focused { return; }

        ctx.input(|i| {
            let mods  = i.modifiers;
            let scale = modifier_scale(&mods);

            // Q / Esc → quit
            if i.key_pressed(Key::Q) || i.key_pressed(Key::Escape) {
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }

            // WASD translation
            if i.key_pressed(Key::W) { self.do_translate_scaled( 0.0, -1.0, scale); }
            if i.key_pressed(Key::S) { self.do_translate_scaled( 0.0,  1.0, scale); }
            if i.key_pressed(Key::A) { self.do_translate_scaled(-1.0,  0.0, scale); }
            if i.key_pressed(Key::D) { self.do_translate_scaled( 1.0,  0.0, scale); }

            // Arrow up/down: zoom
            if i.key_pressed(Key::ArrowUp)   { self.do_zoom(true,  scale); }
            if i.key_pressed(Key::ArrowDown)  { self.do_zoom(false, scale); }

            // Arrow left/right: palette
            if i.key_pressed(Key::ArrowLeft) {
                let n = COLORMAPS.len();
                self.set_colormap((self.colormap_idx + n - 1) % n);
            }
            if i.key_pressed(Key::ArrowRight) {
                self.set_colormap((self.colormap_idx + 1) % COLORMAPS.len());
            }

            // R: reset
            if i.key_pressed(Key::R) {
                self.view = self.default_view.clone();
                self.view.aspect = self.current_aspect();
                self.view_stack.clear();
                self.sync_xy = true;
                self.request_render(false);
            }

            // H / ?: help
            if i.key_pressed(Key::H) {
                self.show_help = !self.show_help;
            }

            // Backspace / Ctrl+Z: undo
            if i.key_pressed(Key::Backspace)
                || (mods.ctrl && i.key_pressed(Key::Z))
            {
                self.undo_zoom();
            }

            // Ctrl+S: save
            if mods.ctrl && i.key_pressed(Key::S) {
                self.show_save = true;
            }
        });
    }

    fn show_help_window(&mut self, ctx: &egui::Context) {
        if !self.show_help { return; }
        egui::Window::new("Controls")
            .collapsible(false)
            .resizable(false)
            .show(ctx, |ui| {
                egui::Grid::new("help_grid").num_columns(2).spacing([20.0, 4.0]).show(ui, |ui| {
                    let rows: &[(&str, &str)] = &[
                        ("W/A/S/D",          "Translate (Shift=2×, Alt=½, Ctrl+Shift=10r, Ctrl+Alt=r/10)"),
                        ("↑ / ↓",            "Zoom in / out (same modifiers)"),
                        ("← / →",            "Previous / next palette"),
                        ("Drag (left btn)",  "Zoom into selection"),
                        ("Right-click",      "Zoom out ×2"),
                        ("Backspace/Ctrl+Z", "Undo zoom"),
                        ("R",                "Reset view"),
                        ("H / ?",            "Toggle this help"),
                        ("Ctrl+S",           "Save PNG"),
                        ("Q / Esc",          "Quit"),
                    ];
                    for (key, desc) in rows {
                        ui.monospace(*key);
                        ui.label(*desc);
                        ui.end_row();
                    }
                });
                if ui.button("Close").clicked() {
                    self.show_help = false;
                }
            });
    }

    fn show_save_window(&mut self, ctx: &egui::Context) {
        if !self.show_save { return; }
        let genome  = self.genome.clone();
        let config  = self.config.clone();
        let view    = self.view.clone();
        let nn_path = self.nn_path.clone();

        let mut do_save = false;
        let mut do_close = false;
        egui::Window::new("Save Fractal Image")
            .collapsible(false)
            .resizable(false)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Width:");
                    ui.add(egui::TextEdit::singleline(&mut self.save_w_str).desired_width(80.0));
                    ui.label("Height:");
                    ui.add(egui::TextEdit::singleline(&mut self.save_h_str).desired_width(80.0));
                });
                let sw: u32 = self.save_w_str.trim().parse().unwrap_or(1920);
                let sh: u32 = self.save_h_str.trim().parse().unwrap_or(1080);
                if sw > 0 && sh > 0 {
                    let r = sw as f64 / sh as f64;
                    ui.label(
                        egui::RichText::new(format!("→ ratio {sw}:{sh} = {r:.3}"))
                            .color(Color32::GRAY),
                    );
                }
                ui.horizontal(|ui| {
                    if ui.button("Save").clicked() { do_save = true; }
                    if ui.button("Cancel").clicked() { do_close = true; }
                });
            });

        if do_save {
            let sw: u32 = self.save_w_str.trim().parse().unwrap_or(1920);
            let sh: u32 = self.save_h_str.trim().parse().unwrap_or(1080);
            if sw >= 64 && sh >= 64 {
                self.prefs.last_save_width  = sw;
                self.prefs.last_save_height = sh;
                self.prefs.save(&self.prefs_path);
                thread::spawn(move || {
                    eprintln!("Rendering {sw}×{sh} PNG…");
                    let rgb  = render_save(&genome, &config, &view, sw, sh);
                    let stem = nn_path.file_stem()
                        .and_then(|s| s.to_str()).unwrap_or("fractal");
                    let out = nn_path.parent().unwrap_or(Path::new("."))
                        .join(format!(
                            "{stem}_cx{:.4}_cy{:.4}_z{:.2}_{sw}x{sh}.png",
                            view.cx, view.cy, view.zoom,
                        ));
                    match save_png(&rgb, sw, sh, &out) {
                        Ok(_)  => eprintln!("Saved → {}", out.display()),
                        Err(e) => eprintln!("Save error: {e}"),
                    }
                });
            }
            self.show_save = false;
        }
        if do_close {
            self.show_save = false;
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll_render(&ctx);
        self.handle_keyboard(&ctx);
        self.show_toolbar(ui);
        self.show_fractal_panel(ui);
        self.show_help_window(&ctx);
        self.show_save_window(&ctx);

        // Auto-upgrade preview to full quality when idle
        if self.render_complete && self.displayed_gen == self.render_gen {
            if let Some(tex) = &self.texture {
                let [tw, th] = tex.size();
                let fw = self.frac_rect.width().round() as usize;
                let fh = self.frac_rect.height().round() as usize;
                if tw < fw / 2 || th < fh / 2 {
                    self.request_render(false);
                }
            }
        }

        if !self.render_complete || self.displayed_gen < self.render_gen {
            ctx.request_repaint();
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Compute a selection rectangle constrained to `aspect` (w/h), returning (rect, valid).
fn selection_rect(start: egui::Pos2, cur: egui::Pos2, aspect: f32) -> (egui::Rect, bool) {
    let dx = cur.x - start.x;
    let dy = cur.y - start.y;
    // Constrain so sel_w / sel_h == aspect
    let (sw, sh) = if dx.abs() / aspect.max(0.001) < dy.abs() {
        (dy.abs() * aspect, dy.abs())
    } else {
        (dx.abs(), dx.abs() / aspect.max(0.001))
    };
    if sw < MIN_SEL_PX || sh < MIN_SEL_PX {
        return (egui::Rect::NOTHING, false);
    }
    let x1 = if dx >= 0.0 { start.x } else { start.x - sw };
    let y1 = if dy >= 0.0 { start.y } else { start.y - sh };
    (egui::Rect::from_min_size(egui::Pos2::new(x1, y1), egui::Vec2::new(sw, sh)), true)
}

/// Returns the WASD / arrow-key step multiplier for the given modifiers.
fn modifier_scale(mods: &egui::Modifiers) -> f64 {
    if mods.ctrl && mods.shift { 30.0 }
    else if mods.ctrl && mods.alt { 0.3 }
    else if mods.shift { 2.0 }
    else if mods.alt   { 0.5 }
    else               { 1.0 }
}

// ── Default config ────────────────────────────────────────────────────────────

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
            recursion_pred_weight: 0.60,
            formula_diversity_weight: 0.30,
            clip_pred_weight: 0.50,
            formula_system: "legacy".to_string(),
            max_nodes: 14, max_depth: 5,
            ood_weight: 0.0,
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

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() -> anyhow::Result<()> {
    let nn_path = std::env::args().nth(1).map(PathBuf::from).ok_or_else(|| {
        anyhow::anyhow!("Usage: nnfractals-viewer <genome.nn>")
    })?;

    #[cfg(feature = "wgpu-backend")]
    {
        render_gpu::init_gpu();
        eprintln!(
            "[viewer] Renderer: {}",
            if render_gpu::gpu_available() { "GPU (wgpu)" } else { "CPU (rayon fallback)" }
        );
    }

    let prefs_path = Path::new(&nn_path).parent().unwrap_or(Path::new("."))
        .join("viewer_prefs.toml");
    let prefs = ViewerPrefs::load(&prefs_path);

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("NNFractals Viewer")
            .with_inner_size([prefs.window_width as f32, prefs.window_height as f32]),
        ..Default::default()
    };

    eframe::run_native(
        "NNFractals Viewer",
        options,
        Box::new(move |cc| {
            Ok(Box::new(App::new(cc, nn_path).expect("Failed to load genome")))
        }),
    ).map_err(|e| anyhow::anyhow!("{e}"))?;

    Ok(())
}
