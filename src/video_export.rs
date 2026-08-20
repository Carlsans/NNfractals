//! Shared fractal rendering + video-export subsystem, used by both the
//! interactive viewer (`nnfractals-viewer`) and the batch export queue
//! (`nnfractals-queue`) — extracted so both binaries render frames
//! identically (same multi-precision-tier renderer, same interpolation
//! math) without duplicating it.
//!
//! Deliberately has no `egui`/`eframe` dependency (that would force the
//! headless GA binary to pull in GUI deps to link the shared library) —
//! `export_video`'s progress callback is a plain `Fn()`, and callers that
//! want an egui repaint pass `&|| ctx.request_repaint()`.

use std::path::{Path, PathBuf};
use std::sync::mpsc;

use serde::{Deserialize, Serialize};

use crate::colormap::{apply_colormap, apply_angle_colormap};
use crate::config::Config;
use crate::dd::Dd;
use crate::formula::apply_formula;
use crate::genome::Genome;
#[cfg(feature = "wgpu-backend")]
use crate::render_gpu;

// ── View ──────────────────────────────────────────────────────────────────

#[derive(Clone)]
// Center stored in double-double (hi + lo) so WASD translation and drag-zoom
// accumulate correctly beyond f64's ~10¹¹ limit.  `aspect` = xrange/yrange.
pub struct View {
    pub cx:     f64,  // hi part of double-double center x
    pub cx_lo:  f64,  // lo part (0.0 until zoom exceeds ~10¹¹)
    pub cy:     f64,
    pub cy_lo:  f64,
    pub zoom:   f64,   // vertical: half_y = 2.0 / zoom
    pub aspect: f64,   // xrange / yrange
}

impl View {
    pub fn new_square(cx: f64, cy: f64, zoom: f64) -> Self {
        View { cx, cx_lo: 0.0, cy, cy_lo: 0.0, zoom, aspect: 1.0 }
    }

    // f64-only bounds — used by f32/f64 render paths and toolbar display.
    pub fn bounds(&self) -> (f64, f64, f64, f64) {
        let half_y = 2.0 / self.zoom;
        let half_x = half_y * self.aspect;
        (self.cx - half_x, self.cx + half_x, self.cy - half_y, self.cy + half_y)
    }

    pub fn pixel_to_fractal(&self, px: f64, py: f64, w: f64, h: f64) -> (f64, f64) {
        let (xmin, xmax, ymin, ymax) = self.bounds();
        (
            xmin + (px / w) * (xmax - xmin),
            ymin + (py / h) * (ymax - ymin),
        )
    }

    // Double-double center accessors.
    pub fn cx_dd(&self) -> Dd { Dd { hi: self.cx, lo: self.cx_lo } }
    pub fn cy_dd(&self) -> Dd { Dd { hi: self.cy, lo: self.cy_lo } }
    pub fn set_cx_dd(&mut self, v: Dd) { self.cx = v.hi; self.cx_lo = v.lo; }
    pub fn set_cy_dd(&mut self, v: Dd) { self.cy = v.hi; self.cy_lo = v.lo; }
}

/// True when pixel coordinates need more precision than f32/GPU can give.
pub fn needs_f64(view: &View, w: u32) -> bool {
    let (xmin, xmax, _, _) = view.bounds();
    let span       = xmax - xmin;
    let pixel_step = span / w.max(1) as f64;
    let coord_mag  = view.cx.abs().max(view.cy.abs()).max(1.0);
    let f32_ulp    = coord_mag * f32::EPSILON as f64;
    pixel_step < f32_ulp * 64.0
}

/// Conservative default margin for [`needs_dd`], in f64 ULPs of pixel step.
/// At 4 ULPs there are still ~4 distinct representable doubles across a pixel,
/// so f64 output is visually smooth — DD escalation happens before any
/// quantisation is visible. This is the right call for the interactive viewer
/// and for stills, which CAN escalate to DD.
pub const DD_MARGIN_ULPS: f64 = 4.0;

/// The margin at which f64 quantisation actually becomes visible: one pixel
/// step per representable double, so adjacent pixels collapse onto the same
/// coordinate and the image visibly blocks up. Video zoom search deliberately runs
/// to here rather than stopping at [`DD_MARGIN_ULPS`], because the exporter
/// pins `allow_dd = false` (DD-tier rendering during camera movement produces
/// a visible shift artifact) — so for video the choice is not "f64 or DD", it
/// is "stop, or keep going in f64". Buys 4× more zoom (2 doublings).
pub const DD_MARGIN_ULPS_PIXELATE: f64 = 1.0;

/// Returns true when pixel coordinates need more precision than f64 can give,
/// at a caller-chosen safety margin in f64 ULPs. See [`DD_MARGIN_ULPS`] and
/// [`DD_MARGIN_ULPS_PIXELATE`] for the two meaningful values.
pub fn needs_dd_with_margin(view: &View, w: u32, margin_ulps: f64) -> bool {
    let (xmin, xmax, _, _) = view.bounds();
    let pixel_step = (xmax - xmin) / w.max(1) as f64;
    let coord_mag  = view.cx.abs().max(view.cy.abs()).max(1.0);
    pixel_step < coord_mag * f64::EPSILON * margin_ulps.max(f64::MIN_POSITIVE)
}

/// Returns true when pixel coordinates need more precision than f64 can give.
/// Threshold: pixel step < 4 × f64 ULP at this coordinate magnitude.
/// (f64 loses ~4 trailing bits before this; DD adds 31 more digits.)
pub fn needs_dd(view: &View, w: u32) -> bool {
    needs_dd_with_margin(view, w, DD_MARGIN_ULPS)
}

pub fn render_cpu(
    genome: &Genome, config: &Config, view: &View,
    w: u32, h: u32, compute_iter: u32, use_f64: bool,
    angle_coloring: bool, allow_dd: bool,
) -> Vec<u8> {
    let color_iter = config.rendering.max_iter;
    let dag = genome.uses_program();
    // Cosmetic angle-coloring is DAG-only, and supported on the f32 tier
    // (below) and the plain-f64 tier (`render_f64_with_angle`) — DD stays
    // unsupported (scope-limited: angle capture was never threaded through
    // the dd-precision iteration loop), so a render that actually needs DD
    // still falls back to the normal escape-time palette exactly as before.
    let will_use_dd = use_f64 && allow_dd && needs_dd(view, w);
    let want_angle = angle_coloring && dag && !will_use_dd;

    if want_angle && use_f64 {
        let (ets, angs) = render_f64_with_angle(genome, view, w, h, compute_iter);
        return apply_angle_colormap(&ets, &angs, color_iter);
    }

    if want_angle {
        let (bxmin, bxmax, bymin, bymax) = view.bounds();
        let (xmin, xmax, ymin, ymax) = (bxmin as f32, bxmax as f32, bymin as f32, bymax as f32);

        #[cfg(feature = "wgpu-backend")]
        if render_gpu::gpu_available() {
            let item = render_gpu::dag_item(genome);
            let (mut ets, mut angs) = render_gpu::render_batch_dag_angle(
                std::slice::from_ref(&item), &[(xmin, xmax, ymin, ymax)], w, h, compute_iter, true,
            );
            return apply_angle_colormap(&ets.pop().unwrap_or_default(),
                                        &angs.pop().unwrap_or_default(), color_iter);
        }

        use rayon::prelude::*;
        let dag_bsq = genome.bailout_radius * genome.bailout_radius;
        let jc      = (genome.julia_cre, genome.julia_cim);
        let phoenix = (genome.phoenix_re, genome.phoenix_im);
        let wf = (w.saturating_sub(1)).max(1) as f32;
        let hf = (h.saturating_sub(1)).max(1) as f32;
        let (ets, angs): (Vec<f32>, Vec<f32>) = (0..(w * h) as usize)
            .into_par_iter()
            .map(|idx| {
                let cx = xmin + (idx % w as usize) as f32 / wf * (xmax - xmin);
                let cy = ymin + (idx / w as usize) as f32 / hf * (ymax - ymin);
                crate::fractal::dag_escape_pixel(
                    &genome.program, &genome.warp, genome.julia_mode, jc, phoenix,
                    dag_bsq, cx, cy, compute_iter,
                )
            })
            .unzip();
        return apply_angle_colormap(&ets, &angs, color_iter);
    }

    let escape_times = render_escape_times(genome, config, view, w, h, compute_iter, use_f64, allow_dd);
    apply_colormap(&escape_times, color_iter, &config.rendering.colormap)
}

/// f64-tier counterpart to the f32 angle-capture loop above — DAG only
/// (matches the f32 tier's own restriction), never DD (see `render_cpu`'s
/// `want_angle`/`will_use_dd` gating; only called when DD is NOT needed,
/// same precondition `render_escape_times`'s own "Regular f64 path" relies
/// on for its `view.bounds()` call to be safe — see [[dd-bounds-invariant]]).
/// Otherwise mirrors that path's DAG branch exactly, just keeping the angle
/// `dag_escape_pixel_f64` was already computing and discarding.
fn render_f64_with_angle(genome: &Genome, view: &View, w: u32, h: u32, compute_iter: u32) -> (Vec<f32>, Vec<f32>) {
    use rayon::prelude::*;
    let (xmin, xmax, ymin, ymax) = view.bounds();
    let wf = (w.saturating_sub(1)).max(1) as f64;
    let hf = (h.saturating_sub(1)).max(1) as f64;
    let dag_bsq = (genome.bailout_radius * genome.bailout_radius) as f64;
    let jc      = (genome.julia_cre as f64, genome.julia_cim as f64);
    let phoenix = (genome.phoenix_re as f64, genome.phoenix_im as f64);
    (0..(w * h) as usize)
        .into_par_iter()
        .map(|idx| {
            let cx = xmin + (idx % w as usize) as f64 / wf * (xmax - xmin);
            let cy = ymin + (idx / w as usize) as f64 / hf * (ymax - ymin);
            crate::fractal::dag_escape_pixel_f64(
                &genome.program, &genome.warp, genome.julia_mode, jc, phoenix,
                dag_bsq, cx, cy, compute_iter,
            )
        })
        .unzip()
}

/// Raw escape times (no colormap) at every precision tier `render_cpu` uses
/// (DD / f64 / GPU f32 / CPU f32), DAG+legacy+warp+phoenix+julia aware.
/// Factored out of `render_cpu` so callers that need the underlying
/// structure rather than a colorized image — e.g. the wormhole self-
/// similarity matcher, which correlates raw escape-time shape and must
/// stay correct at whatever depth the caller is currently viewing — get
/// the exact same multi-tier precision handling `render_cpu` guarantees,
/// instead of a second, divergent implementation.
pub fn render_escape_times(
    genome: &Genome, config: &Config, view: &View,
    w: u32, h: u32, compute_iter: u32, use_f64: bool, allow_dd: bool,
) -> Vec<f32> {
    let dag = genome.uses_program();

    if use_f64 {
        // ── Double-double path — triggered when f64 pixel coordinates lose significance ──
        // Resolution capping is handled by the caller; (w, h) here are already capped.
        // Gated by `allow_dd`: past a certain zoom, whether a specific
        // direction still has resolvable detail depends on the formula's
        // own escape dynamics (how many iterations it takes to bail out
        // relative to how deep you are), not just on precision — DD can't
        // conjure detail out of a direction that escapes too fast for its
        // own chaos to have resolved that depth yet. Auto-escalating past
        // that point produces a flat render that looks like a bug but
        // isn't one, which is confusing enough that the caller may prefer
        // to cap at f64 and switch to DD manually only when they choose to.
        if allow_dd && needs_dd(view, w) {
            use rayon::prelude::*;

            let cx_dd   = view.cx_dd();
            let cy_dd   = view.cy_dd();
            let half_y  = Dd::from_f64(2.0 / view.zoom);
            let half_x  = half_y * view.aspect;
            let xmin_dd = cx_dd - half_x;
            let ymin_dd = cy_dd - half_y;
            // Step per pixel — multiply by pixel index (small integer) preserves DD precision
            let xs = (half_x + half_x) * (1.0 / (w.max(2) - 1) as f64);
            let ys = (half_y + half_y) * (1.0 / (h.max(2) - 1) as f64);

            let bsq     = (config.rendering.bailout * config.rendering.bailout) as f64;
            let dag_bsq = genome.bailout_radius as f64 * genome.bailout_radius as f64;
            let jc      = (genome.julia_cre as f64, genome.julia_cim as f64);
            let phoenix = (genome.phoenix_re as f64, genome.phoenix_im as f64);
            let fw_dd: Vec<(f64, f64)> = if dag { vec![] } else {
                genome.formula_weights().iter().map(|&(r, i)| (r as f64, i as f64)).collect()
            };

            return (0..(w * h) as usize)
                .into_par_iter()
                .map(|idx| {
                    let px_dd = xmin_dd + xs * (idx % w as usize) as f64;
                    let py_dd = ymin_dd + ys * (idx / w as usize) as f64;
                    if dag {
                        crate::fractal::dag_escape_pixel_dd(
                            &genome.program, &genome.warp, genome.julia_mode,
                            jc, phoenix, dag_bsq, px_dd, py_dd, compute_iter,
                        )
                    } else {
                        crate::fractal::legacy_escape_pixel_dd(
                            &fw_dd, bsq, px_dd, py_dd, compute_iter,
                        )
                    }
                })
                .collect();
        }

        // ── Regular f64 path ────────────────────────────────────────────────
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
        let color_iter = config.rendering.max_iter;
        return (0..(w * h) as usize)
            .into_par_iter()
            .map(|idx| {
                let cx = xmin + (idx % w as usize) as f64 / wf * (xmax - xmin);
                let cy = ymin + (idx / w as usize) as f64 / hf * (ymax - ymin);
                if dag {
                    return crate::fractal::dag_escape_pixel_f64(
                        &genome.program, &genome.warp, genome.julia_mode, jc, phoenix,
                        dag_bsq, cx, cy, compute_iter,
                    ).0;
                }
                let (mut zx, mut zy) = (0.0f64, 0.0f64);
                for iter in 0..compute_iter {
                    let (nx, ny) = crate::formula::f64_impl::apply_formula(&fw, zx, zy, cx, cy);
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
    }

    let (bxmin, bxmax, bymin, bymax) = view.bounds();
    let (xmin, xmax, ymin, ymax) = (bxmin as f32, bxmax as f32, bymin as f32, bymax as f32);
    let bailout_sq = config.rendering.bailout * config.rendering.bailout;

    #[cfg(feature = "wgpu-backend")]
    if render_gpu::gpu_available() {
        return if dag {
            let item = render_gpu::dag_item(genome);
            render_gpu::render_batch_dag(
                std::slice::from_ref(&item), &[(xmin, xmax, ymin, ymax)], w, h, compute_iter,
            ).into_iter().next().unwrap_or_default()
        } else {
            let fw = genome.formula_weights();
            render_gpu::render_fractal(&fw, w, h, compute_iter, xmin, xmax, ymin, ymax, bailout_sq)
        };
    }

    use rayon::prelude::*;
    let fw      = genome.formula_weights();
    let dag_bsq = genome.bailout_radius * genome.bailout_radius;
    let jc      = (genome.julia_cre, genome.julia_cim);
    let phoenix = (genome.phoenix_re, genome.phoenix_im);
    let wf = (w.saturating_sub(1)).max(1) as f32;
    let hf = (h.saturating_sub(1)).max(1) as f32;
    let color_iter = config.rendering.max_iter;

    (0..(w * h) as usize)
        .into_par_iter()
        .map(|idx| {
            let cx = xmin + (idx % w as usize) as f32 / wf * (xmax - xmin);
            let cy = ymin + (idx / w as usize) as f32 / hf * (ymax - ymin);
            if dag {
                return crate::fractal::dag_escape_pixel(
                    &genome.program, &genome.warp, genome.julia_mode, jc, phoenix,
                    dag_bsq, cx, cy, compute_iter,
                ).0;
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
        .collect()
}

/// Renders the raw complex z value at the moment of bailout, per pixel —
/// `(zx, zy)` — for exploring a complex-valued autoencoder (Carl's
/// request, 2026-08-07). Deliberately DAG-only and CPU-only (rayon,
/// mirroring `render_escape_times`'s f32/f64 CPU paths exactly): every
/// vae-explore genome is DAG-based (`build_genome` always sets
/// `program`), and adding a GPU output channel would mean touching the
/// shared WGSL dispatch shape for what's currently just a feasibility
/// probe — not worth that risk yet (see `render_gpu.rs`'s own doc comment
/// on the same tradeoff for the exit-angle feature). No DD tier either,
/// same scope limit `dag_escape_pixel_dd` already accepts for angle.
pub fn render_complex_field(
    genome: &Genome, view: &View,
    w: u32, h: u32, compute_iter: u32, use_f64: bool,
) -> Vec<(f32, f32)> {
    use rayon::prelude::*;
    let dag_bsq_f32 = genome.bailout_radius * genome.bailout_radius;
    let jc_f32      = (genome.julia_cre, genome.julia_cim);
    let phoenix_f32 = (genome.phoenix_re, genome.phoenix_im);

    if use_f64 {
        let (xmin, xmax, ymin, ymax) = view.bounds();
        let wf = (w.saturating_sub(1)).max(1) as f64;
        let hf = (h.saturating_sub(1)).max(1) as f64;
        let dag_bsq = (genome.bailout_radius * genome.bailout_radius) as f64;
        let jc      = (genome.julia_cre as f64, genome.julia_cim as f64);
        let phoenix = (genome.phoenix_re as f64, genome.phoenix_im as f64);
        return (0..(w * h) as usize)
            .into_par_iter()
            .map(|idx| {
                let cx = xmin + (idx % w as usize) as f64 / wf * (xmax - xmin);
                let cy = ymin + (idx / w as usize) as f64 / hf * (ymax - ymin);
                let (_, zx, zy) = crate::fractal::dag_escape_pixel_z_f64(
                    &genome.program, &genome.warp, genome.julia_mode, jc, phoenix,
                    dag_bsq, cx, cy, compute_iter,
                );
                (zx, zy)
            })
            .collect();
    }

    let (bxmin, bxmax, bymin, bymax) = view.bounds();
    let (xmin, xmax, ymin, ymax) = (bxmin as f32, bxmax as f32, bymin as f32, bymax as f32);
    let wf = (w.saturating_sub(1)).max(1) as f32;
    let hf = (h.saturating_sub(1)).max(1) as f32;
    (0..(w * h) as usize)
        .into_par_iter()
        .map(|idx| {
            let cx = xmin + (idx % w as usize) as f32 / wf * (xmax - xmin);
            let cy = ymin + (idx / w as usize) as f32 / hf * (ymax - ymin);
            let (_, zx, zy) = crate::fractal::dag_escape_pixel_z(
                &genome.program, &genome.warp, genome.julia_mode, jc_f32, phoenix_f32,
                dag_bsq_f32, cx, cy, compute_iter,
            );
            (zx, zy)
        })
        .collect()
}

/// Dedicated rayon pool for hi-res PNG/video-frame renders. `render_save` →
/// `render_cpu` uses `into_par_iter`, which otherwise runs on the GLOBAL
/// pool and — at deep zoom, where a save render is huge — monopolizes every
/// core, starving the interactive render (which shares that global pool).
/// Confining saves to their own half-sized pool guarantees the viewer keeps
/// cores and never freezes mid-zoom during a save/export.
/// Process-wide controls a long render checks BETWEEN frames: pause/resume
/// and how many cores to use, both changeable while a render is running.
///
/// Deliberately a global rather than a parameter threaded through every
/// export signature — the same reasoning as `NNFRACTALS_SAVE_THREADS`. A
/// render is a single long-lived operation per process here, and the GUI
/// that drives it needs to reach it without every call site growing a
/// control argument.
pub struct RenderControl {
    paused: std::sync::atomic::AtomicBool,
    /// Desired worker threads. 0 means "use the default policy".
    threads: std::sync::atomic::AtomicUsize,
}

pub static RENDER_CONTROL: RenderControl = RenderControl {
    paused: std::sync::atomic::AtomicBool::new(false),
    threads: std::sync::atomic::AtomicUsize::new(0),
};

impl RenderControl {
    pub fn is_paused(&self) -> bool {
        self.paused.load(std::sync::atomic::Ordering::Relaxed)
    }
    pub fn set_paused(&self, v: bool) {
        self.paused.store(v, std::sync::atomic::Ordering::Relaxed);
    }
    /// Configured thread count, or 0 when following the default policy.
    pub fn threads(&self) -> usize {
        self.threads.load(std::sync::atomic::Ordering::Relaxed)
    }
    pub fn set_threads(&self, n: usize) {
        self.threads.store(n, std::sync::atomic::Ordering::Relaxed);
    }

    /// Blocks while paused. Called between frames, never mid-frame, so a
    /// pause always lands on a clean frame boundary and the partially
    /// written video stays valid — ffmpeg simply waits on its stdin.
    pub fn wait_while_paused(&self) {
        while self.is_paused() {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }
}

/// Thread count the save pool should currently have.
fn desired_save_threads() -> usize {
    let n = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(4);
    let runtime = RENDER_CONTROL.threads();
    if runtime > 0 { return runtime.min(n.max(1)); }
    // `NNFRACTALS_SAVE_THREADS` overrides the default for headless batch
    // work, where there is no interactive session to protect and the
    // default would leave ~2x throughput unused.
    std::env::var("NNFRACTALS_SAVE_THREADS").ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&t| t > 0)
        // Half the cores by default: this pool runs behind the INTERACTIVE
        // viewer, which needs the rest to stay responsive.
        .unwrap_or((n / 2).max(1))
}

/// Rayon pool for final renders, REBUILT when the desired thread count
/// changes so the CPU budget can be dialled up or down while a render runs.
///
/// Rayon pools are fixed-size once built, so live adjustment means swapping
/// the pool. That is safe here because it only ever happens between frames:
/// `install` has returned and no tasks are outstanding when the swap occurs.
pub fn save_pool() -> std::sync::Arc<rayon::ThreadPool> {
    use std::sync::{Arc, Mutex, OnceLock};
    static POOL: OnceLock<Mutex<Option<(usize, Arc<rayon::ThreadPool>)>>> = OnceLock::new();
    let cell = POOL.get_or_init(|| Mutex::new(None));
    let want = desired_save_threads().max(1);
    let mut guard = cell.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((have, pool)) = guard.as_ref()
        && *have == want {
        return Arc::clone(pool);
    }
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(want)
            .build()
            .expect("build save thread pool"),
    );
    *guard = Some((want, Arc::clone(&pool)));
    pool
}

/// Extra iteration depth granted per decade of zoom, as a multiple of the
/// configured `rendering.max_iter`. `eff = base * (1 + GAIN * log10(zoom))`.
///
/// A FIXED iteration cap cannot render a deep zoom: past some depth every
/// pixel in the (tiny) viewport reaches the cap, gets the same escape time,
/// and the frame collapses to a single flat colour. That is not a subtle
/// quality loss — it is a dead image, and it was the cause of Carl's
/// "the last frames are always flat" zoom videos (2026-08-14).
///
/// 0.25 is measured, not guessed (`explorer verify-chain --iter-sweep` over
/// a real 13-leg chain, zoom 2.6e4 → 1.4e13, flood = fraction of the frame
/// taken by its single commonest colour):
///
/// ```text
///   zoom     iter=192  iter=384  iter=768   verdict
///   1.0e6      0.38      0.10      0.11     192 already marginal
///   7.1e9      0.74      0.06      0.04     192 failing
///   4.0e10     0.94      0.10      0.03     192 nearly dead
///   1.7e11     1.00      0.24      0.08     192 DEAD, 384 marginal
///   1.4e13     1.00      0.74      0.10     384 failing, 768 fine
/// ```
///
/// GAIN=0.25 yields 384 @1e4, 480 @1e6, 720 @1e11, 816 @1e13 — at or above
/// the measured requirement everywhere, while leaving zoom≈1 untouched at
/// exactly the configured value. 1536/3072/6144 measured no better than
/// 768, so this deliberately does not scale harder.
pub const ZOOM_ITER_GAIN: f64 = 0.25;

/// Ceiling on the multiplier, so a pathological zoom can't make a single
/// frame cost unbounded time.
pub const ZOOM_ITER_MAX_MULT: f64 = 16.0;

/// Iteration depth for rendering `view`, scaling the configured base up with
/// zoom depth (see [`ZOOM_ITER_GAIN`]). Never returns less than `base`, so
/// shallow renders behave exactly as before.
/// Absolute floor on the iteration budget for any FINAL render — video
/// frames, saved images, the viewer's settled render.
///
/// Matches `explore::explore_config`'s own `max_iter.max(1000)` floor, and
/// that agreement is the whole point. The search evaluates candidates
/// through `explore_config` (≥1000 iterations) while the exporter used
/// `config.rendering.max_iter` raw (192): at zoom 5e5 that is 2425 vs 466
/// iterations for the SAME frame. Too few iterations in a chaotic region
/// makes escape times effectively random, so the search saw resolved
/// structure and the exporter shipped speckle noise — and every noise gate
/// built on top agreed with whichever side computed it (Carl, 2026-08-15).
///
/// A fixed floor is not the theoretically ideal answer (the real
/// requirement is content-dependent), but it is the one that makes the
/// search and the exporter resolve identically, which is the property that
/// actually matters here.
pub const MIN_FINAL_RENDER_ITER: u32 = 1000;

pub fn effective_max_iter(view: &View, base: u32) -> u32 {
    // Floor FIRST, then scale — so the zoom scaling is applied to the same
    // base the search used, not to a lower one.
    let base = base.max(MIN_FINAL_RENDER_ITER);
    let decades = view.zoom.max(1.0).log10();
    let mult = (1.0 + ZOOM_ITER_GAIN * decades).clamp(1.0, ZOOM_ITER_MAX_MULT);
    ((base as f64 * mult).round() as u32).max(base)
}

/// Linear supersampling factor for final renders: each output pixel is the
/// mean of `factor²` sub-samples. 1 disables it (the default — cost scales
/// with the SQUARE of this, so 2x is 4x the render time).
///
/// This exists because fractal detail is routinely finer than the pixel
/// grid, and point-sampling it produces **aliasing that looks exactly like
/// noise** — Carl spotted the giveaway (2026-08-15): escape times recurring
/// on a diagonal ~3px lattice, with the 2D FFT peaking at offset (-1,+1) at
/// 40x mean magnitude. The values themselves are continuous and nearly all
/// distinct (39,909 unique in 40,000 px), so it is not a small palette
/// cycling — it is moiré between real structure and the sample grid.
///
/// Measured on a frame previously written off as chaotic: 4x supersampling
/// raised spatial coherence from **0.336 to 0.783**, turning speckle into
/// clean filaments with no change to zoom, view or iteration count. Several
/// genomes dismissed as "chaotic at depth" may simply be undersampled.
///
/// Env var rather than a parameter so it doesn't have to be threaded
/// through every render call site, matching `NNFRACTALS_SAVE_THREADS`.
pub fn supersample_factor() -> u32 {
    std::env::var("NNFRACTALS_SUPERSAMPLE").ok()
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|&f| (1..=4).contains(&f))
        .unwrap_or(1)
}

/// Box-average an RGB buffer down by `factor` in each axis.
fn box_downsample(rgb: &[u8], w: u32, h: u32, factor: u32) -> Vec<u8> {
    let (ow, oh) = (w / factor, h / factor);
    let mut out = vec![0u8; (ow * oh * 3) as usize];
    let n = (factor * factor) as u32;
    for y in 0..oh {
        for x in 0..ow {
            for c in 0..3usize {
                let mut sum = 0u32;
                for sy in 0..factor {
                    for sx in 0..factor {
                        let px = x * factor + sx;
                        let py = y * factor + sy;
                        sum += rgb[((py * w + px) * 3) as usize + c] as u32;
                    }
                }
                out[((y * ow + x) * 3) as usize + c] = (sum / n) as u8;
            }
        }
    }
    out
}

/// Render at W×H with letterboxing to preserve the view's coordinate aspect ratio.
///
/// Iteration depth scales with zoom via [`effective_max_iter`]. Both the
/// COMPUTE cap and the COLORMAP normalisation cap are raised together — they
/// must move as a pair. `apply_colormap` normalises `escape_time / max_iter`
/// and clamps to 1.0, so computing 768 iterations while still colouring
/// against 192 would map every escape time above 192 to the same clamped
/// colour and reproduce the exact flat frame this is meant to fix.
pub fn render_save(genome: &Genome, config: &Config, view: &View, w: u32, h: u32, angle_coloring: bool, allow_dd: bool) -> Vec<u8> {
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

    let eff_iter = effective_max_iter(view, config.rendering.max_iter);
    // `render_cpu` reads its colormap normalisation cap from
    // `config.rendering.max_iter`, so the raised depth has to go through the
    // config, not just the `compute_iter` argument — see this function's
    // doc comment for why the two must not diverge.
    let mut cfg = config.clone();
    cfg.rendering.max_iter = eff_iter;
    let ss = supersample_factor();
    let use_f64 = needs_f64(view, fw * ss);
    let fractal = if ss > 1 {
        let raw = render_cpu(genome, &cfg, view, fw * ss, fh * ss, eff_iter, use_f64, angle_coloring, allow_dd);
        box_downsample(&raw, fw * ss, fh * ss, ss)
    } else {
        render_cpu(genome, &cfg, view, fw, fh, eff_iter, use_f64, angle_coloring, allow_dd)
    };

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

// ── Zoom-video export ────────────────────────────────────────────────────────

/// A snapshot of `View` captured by "Set Start"/"Set End" for the video
/// exporter — same fields as `View` (double-double center + zoom + aspect)
/// but decoupled from it so capturing doesn't alias the live, still-changing
/// view the user keeps navigating with. Serializable so a queued export job
/// can be persisted to disk (`nnfractals-queue`'s `queue.json`).
#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct CapturedView {
    pub cx: f64, pub cx_lo: f64,
    pub cy: f64, pub cy_lo: f64,
    pub zoom: f64,
    pub aspect: f64,
}

impl CapturedView {
    pub fn from_view(v: &View) -> Self {
        CapturedView { cx: v.cx, cx_lo: v.cx_lo, cy: v.cy, cy_lo: v.cy_lo, zoom: v.zoom, aspect: v.aspect }
    }
    pub fn cx_dd(&self) -> Dd { Dd { hi: self.cx, lo: self.cx_lo } }
    pub fn cy_dd(&self) -> Dd { Dd { hi: self.cy, lo: self.cy_lo } }
}

/// Apply the two independent invert toggles: `invert_coords` swaps which
/// captured point's cx/cy feeds the start vs. end of the interpolation;
/// `invert_range` independently swaps which point's zoom/aspect does. Lets
/// you mix, e.g., one point's position with the other's zoom depth, or just
/// reverse playback direction.
pub fn video_endpoints(
    start: &CapturedView, end: &CapturedView, invert_coords: bool, invert_range: bool,
) -> (CapturedView, CapturedView) {
    let (coord_a, coord_b) = if invert_coords { (end, start) } else { (start, end) };
    let (range_a, range_b) = if invert_range { (end, start) } else { (start, end) };
    let a = CapturedView { cx: coord_a.cx, cx_lo: coord_a.cx_lo, cy: coord_a.cy, cy_lo: coord_a.cy_lo,
                            zoom: range_a.zoom, aspect: range_a.aspect };
    let b = CapturedView { cx: coord_b.cx, cx_lo: coord_b.cx_lo, cy: coord_b.cy, cy_lo: coord_b.cy_lo,
                            zoom: range_b.zoom, aspect: range_b.aspect };
    (a, b)
}

/// Interpolated `View` for frame `t ∈ [0,1]` between `start` and `end`.
/// zoom uses geometric/exponential lerp — a constant visual zoom RATE, the
/// standard technique for a smooth-looking fractal zoom; linear zoom-value
/// interpolation looks visually uneven. aspect is a plain linear lerp
/// (rarely differs between the two points). cx/cy use DD-precise arithmetic
/// throughout (`Dd`) — a zoom video's whole point is usually ending at a
/// deep zoom, where naive f64 would lose exactly the precision that matters
/// most there.
///
/// Position is deliberately NOT a plain linear-in-t lerp when zooming in:
/// that combination (linear position + geometric zoom) causes the pan to
/// visibly accelerate through the back half of the video ("slow then very
/// fast"), because a constant absolute drift covers an ever-larger FRACTION
/// of the view as it shrinks. Fix: decay the SCREEN-SPACE offset (position
/// offset measured in view-widths, i.e. offset·zoom) LINEARLY to zero
/// instead of decaying the absolute offset linearly. Since
/// zoom(t) = start_zoom·r^t, solving offset(t)·zoom(t) = offset(0)·zoom(0)·(1−t)
/// for offset(t) gives offset(t) = offset(0)·(1−t)·r^(−t) — this front-loads
/// the absolute position change into the early, zoomed-out (imperceptible)
/// frames and keeps the ON-SCREEN pan speed constant throughout. Only
/// applied when actually zooming IN (`r > 1`, the common case for this
/// feature): the same construction would require the ABSOLUTE offset to
/// overshoot hugely for a zoom-OUT video (screen space is expanding there,
/// not shrinking) — that case falls back to plain linear interpolation,
/// which has no such artifact since the view is only ever getting wider.
pub fn lerp_view(start: &CapturedView, end: &CapturedView, t: f64) -> View {
    let zoom = (start.zoom.ln() + (end.zoom.ln() - start.zoom.ln()) * t).exp();
    let aspect = start.aspect + (end.aspect - start.aspect) * t;

    let r = end.zoom / start.zoom;
    let (cx_dd, cy_dd) = if r > 1.0 + 1e-9 {
        let decay = (1.0 - t) * r.powf(-t);
        (end.cx_dd() - (end.cx_dd() - start.cx_dd()) * decay,
         end.cy_dd() - (end.cy_dd() - start.cy_dd()) * decay)
    } else {
        (start.cx_dd() + (end.cx_dd() - start.cx_dd()) * t,
         start.cy_dd() + (end.cy_dd() - start.cy_dd()) * t)
    };

    View { cx: cx_dd.hi, cx_lo: cx_dd.lo, cy: cy_dd.hi, cy_lo: cy_dd.lo, zoom, aspect }
}

/// Progress messages from a background video-export → the UI.
pub enum VideoMsg {
    /// Sent once, right after the `ffmpeg` child process is spawned —
    /// lets a caller that needs to cancel an in-progress export (e.g.
    /// `nnfractals-queue`'s "Cancel" button) kill it directly by PID,
    /// same `kill <pid>` pattern `viewer.rs`'s `cancel_explore_stage`
    /// already uses. Killing the process breaks its stdin pipe, which the
    /// per-frame `stdin.write_all` below already treats as a normal
    /// failure (`VideoMsg::Failed`) — no separate cancellation plumbing
    /// needed through the render loop itself.
    Started { pid: u32 },
    Progress { done: u32, total: u32 },
    Done(PathBuf),
    Failed(String),
}

/// Render `steps` frames interpolating start→end (after applying the invert
/// toggles) and pipe them as raw RGB24 into a single `ffmpeg` process.
/// Thin wrapper over `export_video_chain` (a 2-waypoint chain IS exactly
/// this) — kept as its own function so every existing caller (viewer,
/// queue) is unaffected by the chain support.
#[allow(clippy::too_many_arguments)]
pub fn export_video(
    genome: &Genome, config: &Config, angle_coloring: bool,
    start: CapturedView, end: CapturedView,
    steps: u32, fps: u32, w: u32, h: u32,
    invert_coords: bool, invert_range: bool,
    out_path: &Path, tx: &mpsc::Sender<VideoMsg>, on_progress: &(dyn Fn() + Sync),
) {
    export_video_chain(
        genome, config, angle_coloring, &[start, end],
        steps, fps, w, h, invert_coords, invert_range, out_path, tx, on_progress,
    );
}

/// (steps_per_leg, total_frames) for a `legs`-leg chain given a total
/// frame budget `steps` — split evenly across legs (at least 2 frames
/// each, so every leg's own endpoints both actually appear), with every
/// leg after the first NOT re-counting its shared boundary frame with the
/// previous leg (already rendered as that leg's own last frame). Pulled
/// out of `export_video_chain` so the arithmetic is directly testable
/// without needing a real ffmpeg subprocess.
/// The precision tier every exported video frame is rendered on. DD-tier
/// rendering combined with a panning camera produces a visible shift
/// artifact, so chain export deliberately caps at f64 (see
/// `export_video_chain`'s doc comment).
///
/// Named rather than inlined because ANY offline scorer/validator that
/// predicts how a frame will look MUST render on this same tier. Scoring a
/// frame with `allow_dd: true` while the exporter renders it with `false`
/// measures an image the exporter will never produce: past the f64 wall the
/// DD render stays rich while the real frame collapses to a flat colour, so
/// the validator passes a chain that exports as a dead video. That was a
/// real bug (Carl, 2026-08-14 — "the last frames are always flat").
pub const VIDEO_FRAME_ALLOW_DD: bool = false;

/// The EXACT sequence of frame `View`s `export_video_chain` will render, in
/// order. Extracted so offline validation replays the real frame sequence
/// instead of re-deriving it — a re-derivation that drifts from the
/// exporter (missed `out_aspect` override, wrong `video_endpoints`
/// handling, different `t` spacing) silently validates frames that are not
/// the ones shipped. Both prior "flat video" bugs were of exactly this
/// shape, so the exporter itself consumes this function: they cannot
/// diverge.
pub fn chain_frame_views(
    waypoints: &[CapturedView], steps: u32, w: u32, h: u32,
    invert_coords: bool, invert_range: bool,
) -> Vec<View> {
    let legs = waypoints.len().saturating_sub(1);
    if legs == 0 { return Vec::new(); }
    let (_, total_frames) = chain_frame_budget(steps, legs);
    if total_frames == 0 { return Vec::new(); }
    // The requested OUTPUT resolution's aspect wins, not whatever aspect the
    // waypoints happened to be captured at (typically square, from the
    // interactive viewer's default ratio) — otherwise render_save
    // letterboxes the captured (square) content into the requested canvas
    // instead of filling it, e.g. a 1080×1920 export showing a square frame
    // with black bars top/bottom.
    let out_aspect = w as f64 / h as f64;

    let ends: Vec<(CapturedView, CapturedView)> = (0..legs)
        .map(|l| video_endpoints(&waypoints[l], &waypoints[l + 1], invert_coords, invert_range))
        .collect();

    // ── Frame budget distributed by LOG-ZOOM span, not equally per leg ──
    // Zoom is geometric, so a leg covering 8x and a leg covering 1.5x given
    // the same frame count play at wildly different zoom speeds, and the
    // rate visibly jumps at every waypoint. Allocating frames in proportion
    // to each leg's log-zoom span makes the zoom rate CONSTANT across the
    // whole chain — the single biggest cause of "steppy" chain videos.
    let spans: Vec<f64> = ends.iter()
        .map(|(a, b)| (b.zoom.max(1e-300) / a.zoom.max(1e-300)).ln().abs().max(1e-9))
        .collect();
    let total_span: f64 = spans.iter().sum();
    // Cumulative normalised boundaries: bounds[l]..bounds[l+1] is leg l.
    let mut bounds = Vec::with_capacity(legs + 1);
    let mut acc = 0.0;
    bounds.push(0.0);
    for s in &spans { acc += s; bounds.push(acc / total_span); }

    // Half-width (in global u) of the cross-fade window centred on each
    // interior waypoint. Capped at WAYPOINT_BLEND_FRACTION of the SHORTER
    // adjacent leg so windows can never overlap.
    let half_window = |k: usize| -> f64 {
        WAYPOINT_BLEND_FRACTION * spans[k - 1].min(spans[k]) / total_span
    };

    let leg_t = |l: usize, u: f64| -> f64 {
        let (lo, hi) = (bounds[l], bounds[l + 1]);
        if hi - lo <= f64::EPSILON { 0.0 } else { (u - lo) / (hi - lo) }
    };

    let mut views = Vec::with_capacity(total_frames as usize);
    for i in 0..total_frames {
        let u = if total_frames <= 1 { 0.0 } else { i as f64 / (total_frames - 1) as f64 };

        // Which leg owns this frame.
        let mut l = 0usize;
        while l + 1 < legs && u > bounds[l + 1] { l += 1; }

        // Is it inside an interior waypoint's cross-fade window?
        let mut blend: Option<(usize, f64)> = None; // (waypoint k, weight of leg k)
        for k in 1..legs {
            let hw = half_window(k);
            if hw > 0.0 && (u - bounds[k]).abs() < hw {
                let x = (u - (bounds[k] - hw)) / (2.0 * hw);
                blend = Some((k, smoothstep01(x)));
                break;
            }
        }

        let mut frame_view = match blend {
            None => lerp_view(&ends[l].0, &ends[l].1, leg_t(l, u)),
            Some((k, w_next)) => {
                // Both legs are EXTRAPOLATED through the waypoint (lerp_view
                // is well-defined outside [0,1]) and cross-faded with a
                // smoothstep. At the waypoint itself both evaluate to the
                // same point and each carries weight 0.5, so the position is
                // exact there while the DERIVATIVE becomes continuous —
                // which is what removes the visible direction/speed kink.
                let w_prev = 1.0 - w_next;
                let prev = lerp_view(&ends[k - 1].0, &ends[k - 1].1, leg_t(k - 1, u));
                let next = lerp_view(&ends[k].0, &ends[k].1, leg_t(k, u));
                blend_views(&prev, &next, w_prev, w_next, &ends[k].0)
            }
        };
        frame_view.aspect = out_aspect;
        views.push(frame_view);
    }
    views
}

/// Fraction of the shorter adjacent leg over which two legs cross-fade
/// across a shared waypoint. Large enough that the direction change is
/// spread over a visible stretch rather than a couple of frames, small
/// enough that most of each leg still follows its own exact path.
const WAYPOINT_BLEND_FRACTION: f64 = 0.30;

#[inline]
fn smoothstep01(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/// Weighted blend of two frame views, anchored at `anchor` for precision.
///
/// Positions are combined as `anchor + Σ wᵢ·(Pᵢ − anchor)` rather than
/// `Σ wᵢ·Pᵢ`: at deep zoom the centres are enormous relative to the
/// distance between them, so blending them directly would round away the
/// very offsets being interpolated. Differencing against the shared
/// waypoint first keeps the arithmetic on small quantities, in double-double
/// throughout. Zoom blends in LOG space, matching `lerp_view`'s geometric
/// convention — a linear blend of zooms would dip below both inputs.
fn blend_views(prev: &View, next: &View, w_prev: f64, w_next: f64, anchor: &CapturedView) -> View {
    let (ax, ay) = (anchor.cx_dd(), anchor.cy_dd());
    let cx = ax + (prev.cx_dd() - ax) * w_prev + (next.cx_dd() - ax) * w_next;
    let cy = ay + (prev.cy_dd() - ay) * w_prev + (next.cy_dd() - ay) * w_next;
    let zoom = (prev.zoom.max(1e-300).ln() * w_prev + next.zoom.max(1e-300).ln() * w_next).exp();
    View {
        cx: cx.hi, cx_lo: cx.lo, cy: cy.hi, cy_lo: cy.lo,
        zoom,
        aspect: prev.aspect * w_prev + next.aspect * w_next,
    }
}

pub fn chain_frame_budget(steps: u32, legs: usize) -> (u32, u32) {
    let legs = legs.max(1) as u32;
    let steps_per_leg = (steps.max(2 * legs) / legs).max(2);
    let total_frames = steps_per_leg + (legs - 1) * (steps_per_leg - 1);
    (steps_per_leg, total_frames)
}

/// Render a continuous video across a SEQUENCE of waypoints (at least 2) —
/// each consecutive pair is one "leg", interpolated exactly like a normal
/// 2-point export via `lerp_view`, all piped into the SAME `ffmpeg`
/// process so a multi-leg video (e.g. a wormhole chain — see
/// `find_wormhole_chain`) is one continuous output file, not several clips
/// needing separate concatenation. Reuses `render_save` per frame — NOT
/// `fractal::render_bounds` (f32-only) — so `render_cpu`'s existing
/// f32→f64 precision auto-selection (`needs_f64`) gives every frame the
/// precision it needs, from the shallowest waypoint to the deepest, for
/// free — this matters more here than for a single 2-point export, since a
/// chain's legs can span wildly different absolute depths. Frames are
/// rendered with `allow_dd: false` — DD escalation (`needs_dd`) is
/// deliberately never reached here, even though `lerp_view`'s own
/// interpolation math IS still DD-precise internally (computing where the
/// camera sits across hundreds of frames needs that to not drift; it's the
/// per-frame RENDER tier this caps, not the position math). Confirmed a
/// real, reproducible visual bug otherwise: a video whose camera pans
/// (not just zooms straight in) while past the DD threshold shows a
/// visible shift/seam artifact — DD-tier rendering was validated for the
/// viewer's stationary, single-view case (an earlier session fixed
/// `View::bounds()`'s own DD-collapse bug there) but
/// never for a MOVING camera sweeping through many DD-tier frames in
/// sequence, and evidently doesn't hold up there. Capping at f64 avoids
/// the buggy path entirely rather than chasing it — a video zooming past
/// f64's own precision floor (~1e14-1e15x) will show f64 staircasing
/// artifacts at the deepest frames instead, a known, accepted tradeoff.
/// `steps` is the TOTAL frame budget across every leg combined (split
/// evenly, at least 2 frames per leg so each leg's own endpoints both
/// actually appear; a leg's shared boundary frame with the next leg is
/// rendered once, not duplicated). `on_progress` is called after each
/// frame (and on early failure) so the caller can wake up its UI — kept as
/// a plain closure rather than an `egui::Context` so this module has no
/// GUI dependency.
#[allow(clippy::too_many_arguments)]
/// One rendered keyframe, held with the view it was rendered at so
/// intermediate frames can be warped out of it.
struct Keyframe {
    view: View,
    w: u32,
    h: u32,
    rgb: Vec<u8>,
}

/// Renders a keyframe with a WIDER field of view than the output frame.
///
/// A keyframe has to serve the frames LEADING UP TO it, which are zoomed
/// out relative to it and therefore see a larger area. Rendering it at the
/// output framing would leave those frames sampling outside its edges.
/// `pad` is the zoom ratio back to the previous keyframe, so the view is
/// widened by exactly that much and the pixel count raised to match —
/// keeping the centre at the same detail density as a native render.
fn render_keyframe(
    genome: &Genome, config: &Config, base: &View, pad: f64, w: u32, h: u32, angle_coloring: bool,
) -> Keyframe {
    let pad = pad.max(1.0);
    let mut view = base.clone();
    view.zoom = base.zoom / pad;
    let kw = ((w as f64 * pad).round() as u32).max(1);
    let kh = ((h as f64 * pad).round() as u32).max(1);
    let rgb = render_save(genome, config, &view, kw, kh, angle_coloring, VIDEO_FRAME_ALLOW_DD);
    Keyframe { view, w: kw, h: kh, rgb }
}

/// Bilinearly resamples `kf` into `target`'s framing, accumulating into
/// `acc` with `weight`.
///
/// The mapping is a pure scale-and-translate: both views look at the same
/// plane, so a target pixel's fractal coordinate converts straight into a
/// keyframe pixel. Centres are differenced in double-double BEFORE dropping
/// to f64 — at deep zoom the centres are enormous next to the distance
/// between them, so subtracting first is what keeps the offset meaningful.
fn warp_accumulate(kf: &Keyframe, target: &View, w: u32, h: u32, weight: f32, acc: &mut [f32]) {
    let t_half_y = 2.0 / target.zoom;
    let t_half_x = t_half_y * target.aspect;
    let k_half_y = 2.0 / kf.view.zoom;
    let k_half_x = k_half_y * kf.view.aspect;
    let dcx = (target.cx_dd() - kf.view.cx_dd()).hi;
    let dcy = (target.cy_dd() - kf.view.cy_dd()).hi;
    let (kw, kh) = (kf.w as f64, kf.h as f64);

    for y in 0..h {
        let fy = (y as f64 + 0.5) / h as f64 - 0.5;
        let oy = fy * 2.0 * t_half_y + dcy;
        let sy = (oy / (2.0 * k_half_y) + 0.5) * kh - 0.5;
        let y0 = sy.floor();
        let wy = sy - y0;
        let y0i = (y0 as i64).clamp(0, kf.h as i64 - 1) as usize;
        let y1i = ((y0 as i64) + 1).clamp(0, kf.h as i64 - 1) as usize;
        for x in 0..w {
            let fx = (x as f64 + 0.5) / w as f64 - 0.5;
            let ox = fx * 2.0 * t_half_x + dcx;
            let sx = (ox / (2.0 * k_half_x) + 0.5) * kw - 0.5;
            let x0 = sx.floor();
            let wx = sx - x0;
            let x0i = (x0 as i64).clamp(0, kf.w as i64 - 1) as usize;
            let x1i = ((x0 as i64) + 1).clamp(0, kf.w as i64 - 1) as usize;

            let o = ((y * w + x) * 3) as usize;
            for c in 0..3 {
                let p00 = kf.rgb[(y0i * kf.w as usize + x0i) * 3 + c] as f64;
                let p10 = kf.rgb[(y0i * kf.w as usize + x1i) * 3 + c] as f64;
                let p01 = kf.rgb[(y1i * kf.w as usize + x0i) * 3 + c] as f64;
                let p11 = kf.rgb[(y1i * kf.w as usize + x1i) * 3 + c] as f64;
                let top = p00 + (p10 - p00) * wx;
                let bot = p01 + (p11 - p01) * wx;
                acc[o + c] += ((top + (bot - top) * wy) as f32) * weight;
            }
        }
    }
}

/// Renders a chain by computing only every `keyframe_stride`-th frame and
/// WARPING the rest out of the two keyframes that bracket them.
///
/// A zoom video's consecutive frames are related by a known scale-and-shift,
/// so an intermediate frame is almost entirely present in the keyframe
/// before it, just smaller. Rendering every frame from scratch re-derives
/// detail that is already known. Cost drops by roughly
/// `keyframe_stride / pad²`, where `pad` is the zoom ratio between
/// keyframes — for a typical chain at stride 16 that is ~12x less work.
///
/// The two bracketing keyframes are cross-faded by smoothstepped log-zoom
/// position, which is what hides the moment new detail resolves: the
/// outgoing keyframe is upscaled (losing sharpness) exactly as the incoming
/// one takes over at full sharpness.
#[allow(clippy::too_many_arguments)]
pub fn export_video_chain_interpolated(
    genome: &Genome, config: &Config, angle_coloring: bool,
    waypoints: &[CapturedView],
    steps: u32, fps: u32, w: u32, h: u32,
    invert_coords: bool, invert_range: bool,
    out_path: &Path, tx: &mpsc::Sender<VideoMsg>, on_progress: &(dyn Fn() + Sync),
    max_frames: Option<u32>, keyframe_stride: u32,
) {
    use std::process::{Command, Stdio};
    if waypoints.len() < 2 {
        let _ = tx.send(VideoMsg::Failed("need at least 2 waypoints to export a video".into()));
        on_progress();
        return;
    }
    // Stride 0/1 means "no interpolation" — fall back to the exact renderer
    // rather than paying warp cost for nothing.
    if keyframe_stride <= 1 {
        export_video_chain_limited(genome, config, angle_coloring, waypoints, steps, fps, w, h,
            invert_coords, invert_range, out_path, tx, on_progress, max_frames);
        return;
    }

    let views = chain_frame_views(waypoints, steps, w, h, invert_coords, invert_range);
    let total_frames = max_frames.map_or(views.len(), |m| (m as usize).min(views.len())).max(1);
    let views = &views[..total_frames];
    let fps = fps.max(1);

    let mut child = match Command::new("ffmpeg")
        .args(["-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", &format!("{w}x{h}"), "-r", &fps.to_string(), "-i", "-",
               "-c:v", "libx264", "-pix_fmt", "yuv420p"])
        .arg(out_path)
        .stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            let _ = tx.send(VideoMsg::Failed(format!("could not start ffmpeg: {e}")));
            on_progress();
            return;
        }
    };
    let mut stdin = match child.stdin.take() {
        Some(s) => s,
        None => {
            let _ = tx.send(VideoMsg::Failed("ffmpeg stdin unavailable".into()));
            on_progress();
            return;
        }
    };
    let _ = tx.send(VideoMsg::Started { pid: child.id() });

    let stride = keyframe_stride as usize;
    let kf_index = |i: usize| -> usize { (i / stride) * stride };
    let pad_for = |k: usize| -> f64 {
        if k == 0 { 1.0 } else {
            let prev = k.saturating_sub(stride);
            (views[k].zoom / views[prev].zoom).max(1.0)
        }
    };

    use std::io::Write;
    let mut cur_k = usize::MAX;
    let mut kf0: Option<Keyframe> = None;
    let mut kf1: Option<Keyframe> = None;
    let mut acc = vec![0f32; (w * h * 3) as usize];
    let mut out = vec![0u8; (w * h * 3) as usize];

    for (i, target) in views.iter().enumerate() {
        RENDER_CONTROL.wait_while_paused();
        let k0 = kf_index(i);
        let k1 = (k0 + stride).min(views.len() - 1);
        if k0 != cur_k {
            // Advance the window: the incoming keyframe becomes the outgoing
            // one when we step exactly one stride, so it is never re-rendered.
            kf0 = match kf1.take() {
                Some(prev) if cur_k != usize::MAX && k0 == cur_k + stride => Some(prev),
                _ => Some(render_keyframe(genome, config, &views[k0], pad_for(k0), w, h, angle_coloring)),
            };
            kf1 = if k1 > k0 {
                Some(render_keyframe(genome, config, &views[k1], pad_for(k1), w, h, angle_coloring))
            } else { None };
            cur_k = k0;
        }

        let (a, b) = (kf0.as_ref().expect("keyframe"), kf1.as_ref());
        acc.iter_mut().for_each(|v| *v = 0.0);
        match b {
            None => warp_accumulate(a, target, w, h, 1.0, &mut acc),
            Some(b) => {
                let (l0, l1) = (views[k0].zoom.ln(), views[k1].zoom.ln());
                let t = if (l1 - l0).abs() < 1e-12 { 0.0 } else { (target.zoom.ln() - l0) / (l1 - l0) };
                let wt = smoothstep01(t) as f32;
                warp_accumulate(a, target, w, h, 1.0 - wt, &mut acc);
                warp_accumulate(b, target, w, h, wt, &mut acc);
            }
        }
        for (o, v) in out.iter_mut().zip(acc.iter()) { *o = v.clamp(0.0, 255.0) as u8; }

        if let Err(e) = stdin.write_all(&out) {
            drop(stdin);
            let _ = child.kill();
            let _ = tx.send(VideoMsg::Failed(format!("ffmpeg pipe write failed: {e}")));
            on_progress();
            return;
        }
        let _ = tx.send(VideoMsg::Progress { done: i as u32 + 1, total: total_frames as u32 });
        on_progress();
    }
    drop(stdin);

    match child.wait_with_output() {
        Ok(o) if o.status.success() => { let _ = tx.send(VideoMsg::Done(out_path.to_path_buf())); }
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr);
            let tail: String = stderr.lines().rev().take(3).collect::<Vec<_>>().join(" | ");
            let _ = tx.send(VideoMsg::Failed(format!("ffmpeg exited with an error: {tail}")));
        }
        Err(e) => { let _ = tx.send(VideoMsg::Failed(format!("ffmpeg wait failed: {e}"))); }
    }
    on_progress();
}

pub fn export_video_chain(
    genome: &Genome, config: &Config, angle_coloring: bool,
    waypoints: &[CapturedView],
    steps: u32, fps: u32, w: u32, h: u32,
    invert_coords: bool, invert_range: bool,
    out_path: &Path, tx: &mpsc::Sender<VideoMsg>, on_progress: &(dyn Fn() + Sync),
) {
    export_video_chain_limited(genome, config, angle_coloring, waypoints, steps, fps, w, h,
        invert_coords, invert_range, out_path, tx, on_progress, None)
}

/// As [`export_video_chain`], but stops after `max_frames` frames.
///
/// Exists so a chain that is clean for most of its length and then degrades
/// can still ship the clean part. Rejecting it outright throws away the
/// whole render: a real case (2026-08-16) had a 2.5-hour search produce a
/// single surviving chain whose 2388 frames were good until frame 2220 —
/// 93% of a usable video, discarded for the last 7%. Capping the frame
/// count ends the video just before the bad stretch instead.
#[allow(clippy::too_many_arguments)]
pub fn export_video_chain_limited(
    genome: &Genome, config: &Config, angle_coloring: bool,
    waypoints: &[CapturedView],
    steps: u32, fps: u32, w: u32, h: u32,
    invert_coords: bool, invert_range: bool,
    out_path: &Path, tx: &mpsc::Sender<VideoMsg>, on_progress: &(dyn Fn() + Sync),
    max_frames: Option<u32>,
) {
    use std::process::{Command, Stdio};
    if waypoints.len() < 2 {
        let _ = tx.send(VideoMsg::Failed("need at least 2 waypoints to export a video".into()));
        on_progress();
        return;
    }
    let legs = waypoints.len() - 1;
    let (_, budget_frames) = chain_frame_budget(steps, legs);
    let total_frames = max_frames.map_or(budget_frames, |m| m.min(budget_frames).max(1));

    let mut child = match Command::new("ffmpeg")
        .args(["-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", &format!("{w}x{h}"), "-r", &fps.to_string(), "-i", "-",
               "-c:v", "libx264", "-pix_fmt", "yuv420p"])
        .arg(out_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            let _ = tx.send(VideoMsg::Failed("ffmpeg not found on PATH — install it to export videos.".into()));
            on_progress();
            return;
        }
        Err(e) => {
            let _ = tx.send(VideoMsg::Failed(format!("failed to spawn ffmpeg: {e}")));
            on_progress();
            return;
        }
    };

    let mut stdin = match child.stdin.take() {
        Some(s) => s,
        None => {
            let _ = tx.send(VideoMsg::Failed("ffmpeg stdin unavailable".into()));
            on_progress();
            return;
        }
    };
    let _ = tx.send(VideoMsg::Started { pid: child.id() });

    // The requested OUTPUT resolution's aspect wins, not whatever aspect the
    // waypoints happened to be captured at (typically square, from the
    // interactive viewer's default ratio) — otherwise render_save
    // letterboxes the captured (square) content into the requested canvas
    // instead of filling it, e.g. a 1080×1920 export showing a square frame
    // with black bars top/bottom.
    use std::io::Write;
    let mut frame_idx = 0u32;
    // Take only `total_frames` from the full sequence — the frames
    // themselves must still be generated against the FULL chain so the
    // camera path is identical; capping here shortens the video without
    // changing where the retained frames look.
    for frame_view in chain_frame_views(waypoints, steps, w, h, invert_coords, invert_range)
        .into_iter().take(total_frames as usize)
    {
        // Between frames only — a pause never splits a frame, so the
        // partially written video stays valid and ffmpeg just waits.
        RENDER_CONTROL.wait_while_paused();
        // allow_dd=false: see this function's doc comment — DD-tier
        // rendering combined with a panning camera produces a visible
        // shift artifact, never validated for a moving-camera video the
        // way it was for the viewer's stationary single-view case.
        // `VIDEO_FRAME_ALLOW_DD` names this so any offline validator can
        // render on exactly the same precision tier (see its doc comment).
        let rgb = save_pool().install(|| {
            render_save(genome, config, &frame_view, w, h, angle_coloring, VIDEO_FRAME_ALLOW_DD)
        });
        if let Err(e) = stdin.write_all(&rgb) {
            drop(stdin);
            let _ = child.kill();
            let _ = tx.send(VideoMsg::Failed(format!("ffmpeg pipe write failed: {e}")));
            on_progress();
            return;
        }
        frame_idx += 1;
        let _ = tx.send(VideoMsg::Progress { done: frame_idx, total: total_frames });
        on_progress();
    }
    drop(stdin);

    match child.wait_with_output() {
        Ok(out) if out.status.success() => {
            let _ = tx.send(VideoMsg::Done(out_path.to_path_buf()));
        }
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            let tail: String = stderr.lines().rev().take(3).collect::<Vec<_>>().join(" | ");
            let _ = tx.send(VideoMsg::Failed(format!("ffmpeg exited with an error: {tail}")));
        }
        Err(e) => {
            let _ = tx.send(VideoMsg::Failed(format!("ffmpeg wait failed: {e}")));
        }
    }
    on_progress();
}

/// Shared implementation for `probe_video_score`/`probe_video_score_keep`:
/// render `waypoints` to `out_path`, return a normalized
/// compressed-bytes / raw-bytes ratio — NOT a raw byte count, so probes
/// spanning different actual frame counts (e.g. a lookahead line truncated
/// early by a DD-boundary gate) stay comparable. This is the video analogue
/// of `fitness::png_compression_entropy`'s "compress it for real, use the
/// resulting ratio as a complexity proxy" recipe — x264 temporal+spatial
/// prediction standing in for PNG's spatial-only DEFLATE. Meaningful only
/// because `export_video_chain` never passes `-crf`/`-b:v`/`-qp` (confirmed
/// via `ffmpeg -h encoder=libx264`: unset here, so libx264's own
/// constant-QUALITY default applies) — a constant-BITRATE encode would make
/// this metric meaningless; if `export_video_chain` ever grows explicit
/// rate control for an unrelated reason, this function's whole premise
/// breaks silently. Returns `None` on any failure (ffmpeg missing, spawn
/// error, zero-length waypoints) — a single failed probe must never abort
/// a caller's search loop. Leaves whatever `export_video_chain` wrote (or
/// didn't) at `out_path` — callers decide whether that's a scratch file to
/// delete or a result worth keeping.
#[allow(clippy::too_many_arguments)]
fn probe_video_render(
    genome: &Genome, config: &Config, angle_coloring: bool,
    waypoints: &[CapturedView], steps: u32, fps: u32, w: u32, h: u32, out_path: &Path,
) -> Option<f64> {
    if waypoints.len() < 2 {
        return None;
    }
    let (tx, rx) = mpsc::channel();
    export_video_chain(
        genome, config, angle_coloring, waypoints, steps, fps, w, h, false, false, out_path, &tx, &|| {},
    );
    let bytes = rx.try_iter().find_map(|m| match m {
        VideoMsg::Done(p) => std::fs::metadata(&p).ok().map(|m| m.len()),
        _ => None,
    });
    let (_, total_frames) = chain_frame_budget(steps, waypoints.len() - 1);
    bytes.map(|b| b as f64 / (w as f64 * h as f64 * 3.0 * total_frames as f64))
}

/// Render a short, low-res probe video across `waypoints` (2+ points, same
/// contract as `export_video_chain`) to a scratch temp file, measure it via
/// `probe_video_render`, then delete the file. For the cheap, many-times-
/// per-search-node lookahead probes (`video_zoom_explore::zoom_level`),
/// where only the score is ever needed. See `probe_video_render`'s doc
/// comment for the ratio/precondition details.
#[allow(clippy::too_many_arguments)]
pub fn probe_video_score(
    genome: &Genome, config: &Config, angle_coloring: bool,
    waypoints: &[CapturedView], steps: u32, fps: u32, w: u32, h: u32,
) -> Option<f64> {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let tmp = std::env::temp_dir().join(format!("nnfractals_probe_{}_{nanos}.mp4", std::process::id()));
    let ratio = probe_video_render(genome, config, angle_coloring, waypoints, steps, fps, w, h, &tmp);
    let _ = std::fs::remove_file(&tmp);
    ratio
}

/// Like `probe_video_score`, but renders to a caller-chosen `out_path` and
/// keeps the file — for a surviving winner's final probe
/// (`video_zoom_explore::run`), which needs to stay on disk afterward for
/// Carl to spot-check/queue, not just contribute a score.
#[allow(clippy::too_many_arguments)]
pub fn probe_video_score_keep(
    genome: &Genome, config: &Config, angle_coloring: bool,
    waypoints: &[CapturedView], steps: u32, fps: u32, w: u32, h: u32, out_path: &Path,
) -> Option<f64> {
    probe_video_render(genome, config, angle_coloring, waypoints, steps, fps, w, h, out_path)
}

/// Builds a chain of waypoints by repeatedly wormhole-searching from each
/// found match: `[start, match1, match2, ...]`. Each leg is a bounded,
/// tractable zoom (one `wormhole_search` call); chaining several compounds
/// total depth far past what any single anchor's precision or iteration
/// budget could reach alone — the practical form of "wormhole past the
/// precision wall instead of growing precision forever." Stops early
/// (returns a shorter chain, always at least length 1) if a leg's search
/// finds nothing — a genuine "ran out of self-similar structure to
/// follow" is reported by the chain simply being shorter than requested,
/// not papered over with a fabricated jump.
pub fn find_wormhole_chain(
    genome: &Genome, config: &Config, start: &View, legs: usize,
) -> Vec<CapturedView> {
    let mut chain = vec![CapturedView::from_view(start)];
    let mut current = start.clone();
    for _ in 0..legs {
        let Some(m) = crate::fractal::wormhole_search(genome, config, &current) else { break };
        let new_cx = current.cx_dd() + Dd::from_f64(m.dx);
        let new_cy = current.cy_dd() + Dd::from_f64(m.dy);
        let next = View {
            cx: new_cx.hi, cx_lo: new_cx.lo, cy: new_cy.hi, cy_lo: new_cy.lo,
            zoom: m.zoom, aspect: current.aspect,
        };
        chain.push(CapturedView::from_view(&next));
        current = next;
    }
    chain
}

// ── Export queue — shared item format ────────────────────────────────────
//
// A `QueueItem` describes one video job persisted to `video_queue/queue.json`
// by the viewer (`Add to Queue`) and consumed by `nnfractals-queue`. Defined
// here (rather than duplicated in each binary) so a job written by one
// binary always deserializes exactly as the other expects.

/// `video_queue/` at the project root — `queue.json` plus one copied `<id>.nn`
/// per pending/processing item ("the queue runs the file from" this folder).
/// Resolved via `crate::project_root()`, NOT a bare relative path — this is
/// shared by both `viewer` (writes here) and `queue` (reads here), and a
/// GUI launch (desktop file / file-manager double-click) can have almost
/// any working directory, unlike a terminal-invoked CLI tool. Confirmed a
/// real bug, not theoretical (Carl, 2026-08-11): opening a `.nn` file via
/// the file manager and using the video-export feature failed with "No
/// such file or directory" before this.
pub fn queue_dir() -> PathBuf {
    crate::project_root().join("video_queue")
}

pub fn queue_json_path() -> PathBuf {
    queue_dir().join("queue.json")
}

#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueueStatus {
    Pending,
    Processing,
    Done,
    Failed,
}

/// One queued video-export job. `nn_filename` is relative to `queue_dir()`
/// (the copied genome the queue actually renders from — decoupled from
/// whatever the viewer currently has loaded). `output_dir` is captured at
/// add-time from the viewer's own save-location logic, so exports land
/// where they already land today, independent of `queue_dir()`.
#[derive(Clone, Serialize, Deserialize)]
pub struct QueueItem {
    pub id: String,
    pub nn_filename: String,
    pub genome_label: String,
    pub start: CapturedView,
    pub end: CapturedView,
    pub steps: u32,
    pub fps: u32,
    pub width: u32,
    pub height: u32,
    pub invert_coords: bool,
    pub invert_range: bool,
    pub colormap: String,
    pub angle_coloring: bool,
    pub output_dir: String,
    pub status: QueueStatus,
    pub output_path: Option<String>,
    pub error: Option<String>,
    pub created_at: u64,
    /// Wormhole-chain waypoints (see `find_wormhole_chain`) — when this has
    /// 2+ entries, the queue renders a multi-leg `export_video_chain`
    /// across ALL of them instead of the plain `start`→`end` two-point
    /// export; `start`/`end` still get set to the chain's first/last
    /// waypoint either way, purely so existing queue-list UI (which only
    /// ever displayed `start`/`end`) keeps showing something sensible.
    /// `#[serde(default)]` so every queue item written before this field
    /// existed still deserializes — empty means "plain two-point export,"
    /// unchanged behavior.
    #[serde(default)]
    pub waypoints: Vec<CapturedView>,
    /// Human-readable label for a multi-waypoint chain's origin (e.g.
    /// "wormhole chain", "zoom-explore chain") — `None` for a plain
    /// two-point export. `#[serde(default)]` for the same reason as
    /// `waypoints`: older queue items deserialize with `None`, and the
    /// queue UI falls back to a generic "chain" label in that case. Exists
    /// because the queue window used to hardcode "wormhole chain" for ANY
    /// item with 2+ waypoints — wrong once a second chain-producing feature
    /// (video-zoom-explore) exists.
    #[serde(default)]
    pub chain_label: Option<String>,
    /// Render only every Nth frame and warp the rest out of the two
    /// bracketing keyframes (see `export_video_chain_interpolated`). 0 or 1
    /// means every frame is rendered exactly, which is the behaviour of
    /// every queue item written before this field existed — hence
    /// `#[serde(default)]`, which yields 0 and therefore "off".
    #[serde(default)]
    pub keyframe_stride: u32,
}

/// Load the persisted queue, or an empty list if the file is absent/corrupt
/// (e.g. first run — `video_queue/` doesn't exist yet).
pub fn load_queue() -> Vec<QueueItem> {
    std::fs::read_to_string(queue_json_path())
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

/// Persist the queue. Best-effort (creates `queue_dir()` if missing);
/// callers that need to know about a write failure should check
/// `queue_json_path().exists()` themselves — nothing here currently does,
/// mirroring `ViewerPrefs::save`'s same best-effort contract.
pub fn save_queue(items: &[QueueItem]) {
    let _ = std::fs::create_dir_all(queue_dir());
    if let Ok(s) = serde_json::to_string_pretty(items) {
        let _ = std::fs::write(queue_json_path(), s);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cv(cx: f64, cy: f64, zoom: f64) -> CapturedView {
        CapturedView { cx, cx_lo: 0.0, cy, cy_lo: 0.0, zoom, aspect: 1.0 }
    }

    #[test]
    fn lerp_view_matches_endpoints_at_t0_and_t1() {
        let start = cv(-0.5, 0.25, 10.0);
        let end   = cv(0.75, -0.1, 1.0e6);
        let v0 = lerp_view(&start, &end, 0.0);
        let v1 = lerp_view(&start, &end, 1.0);
        assert!((v0.cx - start.cx).abs() < 1e-9 && (v0.cy - start.cy).abs() < 1e-9);
        assert!((v0.zoom - start.zoom).abs() < 1e-6);
        assert!((v1.cx - end.cx).abs() < 1e-9 && (v1.cy - end.cy).abs() < 1e-9);
        assert!((v1.zoom - end.zoom).abs() / end.zoom < 1e-6);
    }

    #[test]
    fn lerp_view_zoom_is_geometric_not_linear() {
        // Geometric midpoint of 1.0 and 100.0 is 10.0 (sqrt), not 50.5 (arithmetic
        // mean) — this is the whole point of using exp(lerp(ln,ln)) for zoom.
        let start = cv(0.0, 0.0, 1.0);
        let end   = cv(0.0, 0.0, 100.0);
        let mid = lerp_view(&start, &end, 0.5);
        assert!((mid.zoom - 10.0).abs() < 1e-6, "expected geometric midpoint 10.0, got {}", mid.zoom);
    }

    #[test]
    fn lerp_view_pan_speed_is_constant_in_screen_space_when_zooming_in() {
        // Regression test for a real reported bug: plain linear position lerp
        // + geometric zoom lerp made the pan visibly accelerate through the
        // back half of a zoom-in video ("slow then very fast"), because a
        // constant absolute drift covers an ever-larger fraction of the
        // shrinking view. Fixed by decaying the SCREEN-SPACE offset
        // (offset · zoom) linearly instead — verify that invariant directly.
        let start = cv(-1.0, 0.5, 1.0);
        let end   = cv(2.0, -0.3, 1.0e8);
        let screen_offset = |t: f64| {
            let v = lerp_view(&start, &end, t);
            let dx = end.cx - v.cx;
            let dy = end.cy - v.cy;
            (dx * dx + dy * dy).sqrt() * v.zoom
        };
        let s0 = screen_offset(0.0);
        let s_mid  = screen_offset(0.5);
        let s_late = screen_offset(0.9);
        assert!((s_mid  - s0 * 0.5).abs() < s0 * 1e-6, "s0={s0} s_mid={s_mid}");
        assert!((s_late - s0 * 0.1).abs() < s0 * 1e-6, "s0={s0} s_late={s_late}");
    }

    #[test]
    fn lerp_view_zoom_out_falls_back_to_linear_position() {
        // The screen-space-decay construction would force the ABSOLUTE
        // offset to overshoot hugely for a zoom-OUT video (screen space is
        // expanding, not shrinking) — confirm that case takes the plain
        // linear fallback instead, e.g. midpoint position is the arithmetic
        // mean, not something wildly outside [start,end].
        let start = cv(0.0, 0.0, 1.0e8);
        let end   = cv(1.0, 0.0, 1.0);
        let mid = lerp_view(&start, &end, 0.5);
        assert!((mid.cx - 0.5).abs() < 1e-6, "expected plain linear midpoint 0.5, got {}", mid.cx);
    }

    #[test]
    fn lerp_view_preserves_dd_precision_for_deep_endpoint() {
        // A cx whose lo-part actually matters (won't survive a naive f64-only
        // sum) must still land correctly at t=1.
        let start = cv(0.0, 0.0, 1.0);
        let mut end = cv(1.0, 0.0, 1.0);
        end.cx_lo = 1e-20; // far below f64 ULP at cx=1.0 — only DD arithmetic preserves this
        let v1 = lerp_view(&start, &end, 1.0);
        assert_eq!(v1.cx, end.cx);
        assert_eq!(v1.cx_lo, end.cx_lo);
    }

    #[test]
    fn video_endpoints_no_invert_is_identity() {
        let start = cv(-1.0, 0.0, 1.0);
        let end   = cv(1.0, 0.0, 100.0);
        let (a, b) = video_endpoints(&start, &end, false, false);
        assert_eq!(a.cx, start.cx); assert_eq!(a.zoom, start.zoom);
        assert_eq!(b.cx, end.cx);   assert_eq!(b.zoom, end.zoom);
    }

    #[test]
    fn video_endpoints_invert_coords_only_swaps_position_not_zoom() {
        let start = cv(-1.0, 0.0, 1.0);
        let end   = cv(1.0, 0.0, 100.0);
        let (a, b) = video_endpoints(&start, &end, true, false);
        assert_eq!(a.cx, end.cx);     // position swapped
        assert_eq!(a.zoom, start.zoom); // zoom NOT swapped
        assert_eq!(b.cx, start.cx);
        assert_eq!(b.zoom, end.zoom);
    }

    #[test]
    fn video_endpoints_invert_range_only_swaps_zoom_not_position() {
        let start = cv(-1.0, 0.0, 1.0);
        let end   = cv(1.0, 0.0, 100.0);
        let (a, b) = video_endpoints(&start, &end, false, true);
        assert_eq!(a.cx, start.cx);   // position NOT swapped
        assert_eq!(a.zoom, end.zoom); // zoom swapped
        assert_eq!(b.cx, end.cx);
        assert_eq!(b.zoom, start.zoom);
    }

    // ── wormhole-chain video ─────────────────────────────────────────────

    #[test]
    fn effective_max_iter_never_reduces_the_configured_base() {
        // Shallow renders must behave EXACTLY as before this existed.
        for zoom in [0.01, 0.5, 1.0] {
            let v = View { cx: 0.0, cx_lo: 0.0, cy: 0.0, cy_lo: 0.0, zoom, aspect: 1.0 };
            // No zoom SCALING at zoom<=1 — but never below the final-render
            // floor, which exists so the exporter resolves the same detail
            // the search validated against.
            assert_eq!(effective_max_iter(&v, 192), MIN_FINAL_RENDER_ITER);
            assert_eq!(effective_max_iter(&v, 4000), 4000, "a base above the floor is kept as-is");
        }
    }

    #[test]
    fn effective_max_iter_meets_the_measured_requirement_at_depth() {
        // The floor at each depth comes from a real `verify-chain
        // --iter-sweep` over a 13-leg chain (see ZOOM_ITER_GAIN's table):
        // below these the frame measurably collapses to one colour.
        let at = |zoom: f64| {
            let v = View { cx: 0.0, cx_lo: 0.0, cy: 0.0, cy_lo: 0.0, zoom, aspect: 1.0 };
            effective_max_iter(&v, 192)
        };
        assert!(at(1e6) >= 384, "1e6 needs >=384, got {}", at(1e6));
        assert!(at(1e11) >= 384, "1e11 needs >=384, got {}", at(1e11));
        assert!(at(1e13) >= 768, "1e13 needs >=768, got {}", at(1e13));
        // And never below the floor that keeps search and exporter in step.
        assert!(at(1.0) >= MIN_FINAL_RENDER_ITER);
        // Monotone in depth, and bounded so one frame can't cost unbounded time.
        assert!(at(1e6) <= at(1e11) && at(1e11) <= at(1e13));
        assert!(at(1e300) <= (MIN_FINAL_RENDER_ITER as f64 * ZOOM_ITER_MAX_MULT) as u32);
    }

    #[test]
    fn chain_frame_views_matches_the_budget_and_applies_output_aspect() {
        // The exporter itself consumes this function, so these properties
        // are the shipped frame sequence's properties by construction.
        let wps = vec![cv(-1.0, 0.0, 1.0), cv(-0.9, 0.1, 100.0), cv(-0.8, 0.2, 10000.0)];
        let (_, total) = chain_frame_budget(30, 2);
        let views = chain_frame_views(&wps, 30, 1080, 1920, false, false);
        assert_eq!(views.len() as u32, total, "must produce exactly the budgeted frame count");
        // Output aspect wins over the waypoints' captured (square) aspect.
        let out_aspect = 1080.0 / 1920.0;
        for v in &views {
            assert!((v.aspect - out_aspect).abs() < 1e-12, "every frame takes the OUTPUT aspect");
        }
        // Zoom advances monotonically from the first to the last waypoint.
        assert!((views[0].zoom - 1.0).abs() < 1e-9);
        assert!((views.last().unwrap().zoom - 10000.0).abs() < 1e-6);
        for pair in views.windows(2) {
            assert!(pair[1].zoom >= pair[0].zoom - 1e-9, "zoom must not go backwards");
        }
    }

    #[test]
    fn render_control_pause_and_thread_overrides_round_trip() {
        // Default state must be "not paused, follow default policy" — a
        // stale paused flag would silently wedge every future render.
        RENDER_CONTROL.set_paused(false);
        RENDER_CONTROL.set_threads(0);
        assert!(!RENDER_CONTROL.is_paused());
        assert_eq!(RENDER_CONTROL.threads(), 0);

        RENDER_CONTROL.set_paused(true);
        assert!(RENDER_CONTROL.is_paused());
        RENDER_CONTROL.set_paused(false);
        assert!(!RENDER_CONTROL.is_paused());

        // A runtime thread count overrides the default policy, and is
        // clamped to the machine's actual parallelism so a wild value can't
        // spawn an absurd pool.
        let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        RENDER_CONTROL.set_threads(1);
        assert_eq!(desired_save_threads(), 1);
        RENDER_CONTROL.set_threads(usize::MAX);
        assert_eq!(desired_save_threads(), cores, "must clamp to available cores");
        RENDER_CONTROL.set_threads(0);
        assert!(desired_save_threads() >= 1, "0 falls back to the default policy");
    }

    #[test]
    fn wait_while_paused_returns_immediately_when_not_paused() {
        // The frame loop calls this on EVERY frame, so the unpaused path
        // must cost effectively nothing.
        RENDER_CONTROL.set_paused(false);
        let t0 = std::time::Instant::now();
        for _ in 0..1000 { RENDER_CONTROL.wait_while_paused(); }
        assert!(t0.elapsed() < std::time::Duration::from_millis(50));
    }

    #[test]
    fn queue_item_without_keyframe_stride_still_deserializes() {
        // Every queue.json written before this field existed must keep
        // loading, and must mean "render every frame exactly".
        let json = r#"{
            "id":"abc","nn_filename":"abc.nn","genome_label":"g",
            "start":{"cx":0.0,"cx_lo":0.0,"cy":0.0,"cy_lo":0.0,"zoom":1.0,"aspect":1.0},
            "end":{"cx":0.1,"cx_lo":0.0,"cy":0.1,"cy_lo":0.0,"zoom":10.0,"aspect":1.0},
            "steps":60,"fps":30,"width":1280,"height":720,
            "invert_coords":false,"invert_range":false,"colormap":"turbo",
            "angle_coloring":false,"output_dir":"./out","status":"Pending",
            "output_path":null,"error":null,"created_at":0
        }"#;
        let item: QueueItem = serde_json::from_str(json).expect("legacy queue item must load");
        assert_eq!(item.keyframe_stride, 0, "absent field means off");
        assert!(item.waypoints.is_empty());
        assert!(item.chain_label.is_none());

        // And a round-trip preserves an explicitly set stride.
        let mut it = item;
        it.keyframe_stride = 16;
        let s = serde_json::to_string(&it).expect("serialize");
        let back: QueueItem = serde_json::from_str(&s).expect("round-trip");
        assert_eq!(back.keyframe_stride, 16);
    }

    #[test]
    fn box_downsample_averages_blocks_and_shrinks_dimensions() {
        // 4x2 image, factor 2 -> 2x1. Each output pixel is the mean of its
        // 2x2 block, per channel.
        let rgb: Vec<u8> = vec![
            0,0,0,   10,10,10,   100,100,100, 200,200,200,
            20,20,20, 30,30,30,  100,100,100, 200,200,200,
        ];
        let out = box_downsample(&rgb, 4, 2, 2);
        assert_eq!(out.len(), 2 * 1 * 3);
        assert_eq!(out[0], 15); // (0+10+20+30)/4
        assert_eq!(out[3], 150); // (100+200+100+200)/4
    }

    #[test]
    fn supersample_factor_defaults_to_one_and_rejects_absurd_values() {
        // Default (unset) must be 1 — supersampling costs the SQUARE of the
        // factor, so it must never turn itself on implicitly.
        unsafe { std::env::remove_var("NNFRACTALS_SUPERSAMPLE") };
        assert_eq!(supersample_factor(), 1);
        unsafe { std::env::set_var("NNFRACTALS_SUPERSAMPLE", "99") };
        assert_eq!(supersample_factor(), 1, "out-of-range factor must fall back to 1");
        unsafe { std::env::set_var("NNFRACTALS_SUPERSAMPLE", "2") };
        assert_eq!(supersample_factor(), 2);
        unsafe { std::env::remove_var("NNFRACTALS_SUPERSAMPLE") };
    }

    #[test]
    fn chain_frame_views_has_constant_zoom_rate_across_uneven_legs() {
        // Leg 0 spans 100x, leg 1 spans 2x. With frames split EQUALLY the
        // zoom rate would jump ~7x at the waypoint — the "steppy" look.
        // Proportional allocation keeps per-frame log-zoom step uniform.
        let wps = vec![cv(0.0, 0.0, 1.0), cv(0.1, 0.0, 100.0), cv(0.2, 0.0, 200.0)];
        let views = chain_frame_views(&wps, 240, 640, 640, false, false);
        assert!(views.len() > 50);
        let steps: Vec<f64> = views.windows(2)
            .map(|p| p[1].zoom.ln() - p[0].zoom.ln())
            .collect();
        let mean = steps.iter().sum::<f64>() / steps.len() as f64;
        let worst = steps.iter().map(|s| (s - mean).abs() / mean).fold(0.0f64, f64::max);
        assert!(worst < 0.10, "zoom rate must stay within 10% of constant, worst deviation {worst:.3}");
    }

    #[test]
    fn chain_frame_views_is_smooth_through_interior_waypoints() {
        // A chain that turns a corner: without cross-fading, the camera's
        // direction changes abruptly at the middle waypoint. Second
        // differences of the centre path spike at exactly that frame.
        let wps = vec![cv(0.0, 0.0, 1.0), cv(0.5, 0.0, 10.0), cv(0.5, 0.5, 100.0)];
        let views = chain_frame_views(&wps, 300, 640, 640, false, false);
        assert!(views.len() > 60);
        // Screen-space second difference — the quantity an eye reads as a jerk.
        let accel: Vec<f64> = views.windows(3).map(|t| {
            let z = t[1].zoom;
            let ax = (t[2].cx - 2.0 * t[1].cx + t[0].cx) * z;
            let ay = (t[2].cy - 2.0 * t[1].cy + t[0].cy) * z;
            (ax * ax + ay * ay).sqrt()
        }).collect();
        let mean = accel.iter().sum::<f64>() / accel.len() as f64;
        let peak = accel.iter().cloned().fold(0.0f64, f64::max);
        // A hard corner shows up as a peak orders of magnitude above the
        // mean; a cross-faded one stays within a small multiple.
        assert!(peak < mean * 25.0 + 1e-12,
            "waypoint transition is not smooth: peak {peak:.3e} vs mean {mean:.3e}");
    }

    #[test]
    fn chain_frame_views_endpoints_are_exact() {
        // Smoothing must not move where the video starts or finishes.
        let wps = vec![cv(-1.0, 0.0, 1.0), cv(-0.9, 0.1, 50.0), cv(-0.8, 0.2, 4000.0)];
        let views = chain_frame_views(&wps, 200, 800, 600, false, false);
        let first = views.first().expect("frames");
        let last = views.last().expect("frames");
        assert!((first.zoom - 1.0).abs() < 1e-9, "starts at the first waypoint's zoom");
        assert!((last.zoom - 4000.0).abs() < 1e-6, "ends at the last waypoint's zoom");
        assert!((first.cx - (-1.0)).abs() < 1e-12 && (first.cy - 0.0).abs() < 1e-12);
        assert!((last.cx - (-0.8)).abs() < 1e-12 && (last.cy - 0.2).abs() < 1e-12);
    }

    #[test]
    fn chain_frame_budget_two_waypoints_matches_old_single_leg_behavior() {
        // 1 leg must behave exactly like the pre-chain export_video did
        // (`steps.max(2)`, every requested frame counted) — existing
        // callers (viewer, queue) must see no change at all.
        let (per_leg, total) = chain_frame_budget(60, 1);
        assert_eq!(per_leg, 60);
        assert_eq!(total, 60);
        let (per_leg, total) = chain_frame_budget(1, 1);
        assert_eq!(per_leg, 2, "must floor to at least 2 frames even if 1 was requested");
        assert_eq!(total, 2);
    }

    #[test]
    fn chain_frame_budget_shares_boundary_frames_between_legs() {
        // 3 legs, 30 requested frames → 10/leg, but leg 2 and 3 each drop
        // their shared t=0 duplicate: 10 + 9 + 9 = 28, not 30.
        let (per_leg, total) = chain_frame_budget(30, 3);
        assert_eq!(per_leg, 10);
        assert_eq!(total, 28);
    }

    #[test]
    fn chain_frame_budget_never_drops_below_two_frames_per_leg() {
        // 5 legs but only 4 total frames requested — each leg still needs
        // its own start+end to exist as distinct frames.
        let (per_leg, total) = chain_frame_budget(4, 5);
        assert_eq!(per_leg, 2);
        assert_eq!(total, 2 + 4 * 1); // 6
    }

    #[test]
    fn needs_dd_threshold_moves_with_export_width() {
        // `needs_dd`'s zoom threshold is ~1/(w * coord_mag * EPSILON) — a
        // WIDER target render needs DD precision at a SHALLOWER zoom than a
        // narrow one, because each pixel covers less of the coordinate
        // space. This matters beyond documentation: video_zoom_explore's
        // whole "stop just before the DD zone" search gates candidates by
        // checking `needs_dd` against the REAL intended export width, not
        // the tiny probe width used for cheap search-time renders — using
        // the wrong width here would silently accept candidates that are
        // fine at probe resolution but already past the wall at the width
        // that actually matters. zoom=1e13 sits between the two
        // thresholds (~3.5e13 at w=128, ~3.5e12 at w=1280) with comfortable
        // margin on both sides.
        let view = View::new_square(0.0, 0.0, 1.0e13);
        assert!(!needs_dd(&view, 128), "should not need DD yet at a narrow probe width");
        assert!(needs_dd(&view, 1280), "should already need DD at a realistic export width");
    }

    fn mandelbrot_genome() -> Genome {
        use crate::formula::{op, OpNode};
        use crate::genome::ProgramBuilder;
        let mut b = ProgramBuilder::new();
        let z  = b.push(op::Z, 0, 0, 0.0, 0.0).unwrap();
        let c  = b.push(op::C, 0, 0, 0.0, 0.0).unwrap();
        let z2 = b.push(op::SQR, z, 0, 0.0, 0.0).unwrap();
        b.push(op::ADD, z2, c, 0.0, 0.0).unwrap();
        Genome { program: b.into_nodes(), bailout_radius: 4.0, view_zoom: 1.0, ..Default::default() }
    }

    fn chain_test_config() -> Config {
        use crate::config::{DedupConfig, MassExtinctionConfig, OptimizationConfig, OutputConfig, RenderingConfig};
        Config {
            dedup: DedupConfig::default(), mass_extinction: MassExtinctionConfig::default(),
            rendering: RenderingConfig { default_width: 800, default_height: 800, max_iter: 300, bailout: 4.0,
                colormap: "turbo".into(), view_x_min: -2.0, view_x_max: 2.0, view_y_min: -2.0, view_y_max: 2.0 },
            optimization: OptimizationConfig { population_size: 40, elitism_count: 6, mutation_rate: 0.20, mutation_scale: 0.08,
                eval_width: 64, eval_height: 64, eval_max_iter: 128, restart_after_gens: 30, novelty_weight: 0.45,
                novelty_k: 5, archive_size: 150, self_replication_weight: 0.35, fractal_recursion_weight: 0.35,
                recursion_pred_weight: 0.60, formula_diversity_weight: 0.30, clip_pred_weight: 0.50,
                formula_system: "dag".to_string(), max_nodes: 14, max_depth: 5, ood_weight: 0.0, pref_weight: 0.4,
                seed_pref_weight: 3.0, musiq_weight: 0.25, pref_elite_count: 4, archive_random_ratio: 0.30,
                duplicate_penalty_weight: 0.50, archive_seeding_enabled: false, angle_structure_weight: 0.0, img_novelty_weight: 0.0 },
            output: OutputConfig { save_dir: "./fractals".into(), population_dir: "./populations".into(),
                min_entropy_prefilter: 0.42, max_entropy_prefilter: 0.65, min_clip_score: 0.512, min_laion_score: 5.30,
                min_beauty: 0.35, min_save_distance: 0.04, min_ensemble: 4.6, min_musiq: 30.0, min_pref: 0.45 },
        }
    }

    #[test]
    fn wormhole_chain_gets_strictly_deeper_each_leg() {
        let genome = mandelbrot_genome();
        let config = chain_test_config();
        let start = View::new_square(-0.5, 0.0, 1.0);
        let chain = find_wormhole_chain(&genome, &config, &start, 2);
        assert!(chain.len() >= 2, "the classic Mandelbrot must yield at least one leg");
        assert_eq!(chain[0].zoom, start.zoom);
        for w in chain.windows(2) {
            assert!(w[1].zoom > w[0].zoom, "each waypoint must be a DEEPER zoom than the last");
        }
    }
}
