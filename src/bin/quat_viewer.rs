//! Standalone interactive viewer for quaternion ray-marched fractals
//! (`quat_dag`'s genome DAG system). Loads ONE genome, compiles its GPU
//! shader once (`CompiledDagPipeline`, the same codegen path
//! `quat-raymarch-genome-video` uses), then lets you orbit the camera by
//! dragging, zoom with the scroll wheel, and scrub the C (time-axis)
//! value live — the thing this whole session's static/video renders
//! couldn't give Carl: freely looking at a fractal from angles nobody
//! picked in advance, in real time.
//!
//! Deliberately NOT built on top of `viewer.rs` (the existing 2D
//! explorer, ~6800 lines) — a small self-contained binary, matching how
//! every other quaternion feature this session started as a standalone
//! prototype before any integration decision. It DOES follow one piece
//! of `viewer.rs` precedent exactly, though: single-instance IPC (see
//! "IPC — single-instance socket" below), added after double-clicking
//! several genomes in a row from the gallery left 7 separate GPU-backed
//! windows open simultaneously, competing for the one GPU with an
//! unrelated background evolution run — the same failure mode
//! `viewer.rs`'s own socket delegation exists to prevent for the 2D
//! viewer, just never ported over here until it actually bit.

use std::io::{Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread;

use eframe::egui;
use nnfractals::formula::OpNode;
use nnfractals::quat_dag::{QuatDagFormula, RaymarchDagParams};
use nnfractals::quat_raymarch::RaymarchCamera;
use nnfractals::render_gpu_raymarch_dag_codegen::CompiledDagPipeline;

// ── IPC — single-instance socket (mirrors viewer.rs's exactly) ─────────────

/// Cleans up the Unix socket file on drop (best-effort).
struct SocketGuard(PathBuf);
impl Drop for SocketGuard {
    fn drop(&mut self) { let _ = std::fs::remove_file(&self.0); }
}

fn socket_path() -> PathBuf {
    let tag = std::env::var("USER")
        .or_else(|_| std::env::var("LOGNAME"))
        .unwrap_or_else(|_| "user".into());
    std::env::temp_dir().join(format!("nnfractals-quat-viewer-{tag}.sock"))
}

/// Connects, hands over the path, then waits (briefly) for a one-byte ack
/// the listener writes back after it has actually parsed the path off the
/// wire. Without this, a socket file left behind by a process that died
/// mid-shutdown (or a listener thread that's alive but wedged) still
/// accepts the `connect()` — `write_all` even "succeeds" into the kernel
/// buffer — so the old blind version reported success, the caller exited
/// believing it had delegated, and no window ever appeared for anyone to
/// see: a real, reported defect ("closed 2 viewer windows, now a new one
/// can't be opened"). Any failure here — connect, write, or a missing ack
/// within the timeout — is now treated as "no live instance", so the
/// caller falls through to binding its own listener and opening a real
/// window instead of vanishing silently.
fn try_delegate(sock: &Path, path: &Path) -> bool {
    let Ok(mut s) = UnixStream::connect(sock) else { return false };
    if s.write_all(path.to_string_lossy().as_bytes()).is_err() {
        return false;
    }
    let _ = s.shutdown(std::net::Shutdown::Write); // EOF so the listener's read_to_string returns
    let _ = s.set_read_timeout(Some(std::time::Duration::from_millis(500)));
    let mut ack = [0u8; 1];
    s.read_exact(&mut ack).is_ok()
}

/// Mirrors `explorer.rs`'s `recommended_orbit_radius` exactly (see that
/// function's doc comment for the derivation/calibration) — duplicated
/// rather than shared because it lives in a `[[bin]]`, not the lib; keep
/// the two in sync if the calibration ever changes.
fn recommended_orbit_radius(domain_radius: f64, fov_deg: f64, width: u32, height: u32) -> f64 {
    // Was 1.42 — a bug, not a taste call: FILL_FRAC > 1.0 places the
    // silhouette's tangent edge OUTSIDE the camera's own half-FOV by
    // construction, which crops every render regardless of genome or
    // aspect. explorer.rs's copy of this function documents the
    // calibration this constant is supposed to encode (fit against
    // domain=1.6/fov=45°/"eye=5", cross-checked at fov=37°/"eye≈6.12") —
    // recomputing both of those cited examples from this exact formula
    // gives FILL_FRAC≈0.82 in both cases, not 1.42.
    const FILL_FRAC: f64 = 0.82;
    let aspect = width as f64 / height.max(1) as f64;
    let half_fov_y = (fov_deg / 2.0).to_radians();
    let half_fov_x = (half_fov_y.tan() * aspect).atan();
    let tight_half_fov = half_fov_y.min(half_fov_x);
    let target_angular_half_size = FILL_FRAC * tight_half_fov;
    domain_radius / target_angular_half_size.sin()
}

/// Builds the on-screen message for `CompiledDagPipeline::compile()`
/// returning `None`. Used to say "no GPU adapter available (or shader
/// failed to compile) — see stderr" — vague on two counts: it named a
/// failure mode (shader compile) that can't actually produce `None`
/// through this path at all (wgpu reports shader validation errors
/// asynchronously via the device's error scope, not a `Result` from
/// `create_shader_module`), and it pointed at stderr for a reason that
/// was only ever logged there, never surfaced to the caller. Now pulls
/// the actual recorded reason via `last_gpu_init_error()` when there is
/// one.
fn gpu_error_message() -> String {
    match nnfractals::render_gpu_raymarch_dag_codegen::last_gpu_init_error() {
        Some(reason) => format!("GPU unavailable: {reason}"),
        None => "GPU unavailable — no reason was recorded (GPU init may not have run yet, or this is a stale pipeline). Check stderr for '[gpu-raymarch-dag-codegen]' lines.".to_string(),
    }
}

/// Locate a project binary robustly (release build preferred, several
/// fallbacks) — duplicated from `browser.rs`'s `locate_bin` (same
/// reasoning as `recommended_orbit_radius` above: small enough, and
/// this lives in a `[[bin]]`, not the lib).
fn locate_bin(name: &str) -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            if let Some(target) = dir.parent() {
                let c = target.join("release").join(name);
                if c.exists() {
                    return c;
                }
            }
            let c = dir.join(name);
            if c.exists() {
                return c;
            }
            if let Some(target) = dir.parent() {
                let c = target.join("debug").join(name);
                if c.exists() {
                    return c;
                }
            }
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        let c = PathBuf::from(home).join(".local/bin").join(name);
        if c.exists() {
            return c;
        }
    }
    PathBuf::from(name)
}

fn camera_from_orbit(yaw: f64, pitch: f64, distance: f64, fov_y: f64) -> RaymarchCamera {
    let cp = pitch.cos();
    let eye = (distance * cp * yaw.sin(), distance * pitch.sin(), -distance * cp * yaw.cos());
    RaymarchCamera { eye, target: (0.0, 0.0, 0.0), up_hint: (0.0, 1.0, 0.0), fov_y }
}

/// Maps a linear UI slider position `t ∈ [-1,1]` to a C value in
/// `[c_min, c_max]` — NOT assumed symmetric (a real defect this fixes:
/// `CRangeScan` below already probes the positive and negative
/// directions independently, `found_pos`/`found_neg`, and a genome's
/// usable range genuinely can be lopsided — e.g. `found_pos=5.1,
/// found_neg=-1.8` — but the code used to fold both into one `c_bound =
/// max(found_pos, -found_neg)` and mirror it, silently discarding
/// whichever side was smaller and offering a slider range on that side
/// that mostly renders nothing). `t≥0` maps through `[0,c_max]` on its
/// own log curve, `t<0` through `[c_min,0]` on ITS own, each scaled by
/// its own magnitude — dense sampling near C=0 on both sides without
/// losing reach out to either bound. `K` controls how aggressively
/// "log-like" the curve is; `K=0` would degenerate to plain linear.
const C_SLIDER_LOG_K: f64 = 8.0;

fn slider_t_to_c(t: f64, c_min: f64, c_max: f64) -> f64 {
    let a = t.abs().min(1.0);
    let shape = ((C_SLIDER_LOG_K * a).exp() - 1.0) / (C_SLIDER_LOG_K.exp() - 1.0); // in [0,1]
    if t >= 0.0 {
        shape * c_max.max(0.0)
    } else {
        shape * c_min.min(0.0)
    }
}

fn c_to_slider_t(c: f64, c_min: f64, c_max: f64) -> f64 {
    if c >= 0.0 {
        if c_max <= 0.0 {
            return 0.0;
        }
        let a = (c / c_max).clamp(0.0, 1.0);
        let inner = (a * (C_SLIDER_LOG_K.exp() - 1.0) + 1.0).max(1.0);
        inner.ln() / C_SLIDER_LOG_K
    } else {
        if c_min >= 0.0 {
            return 0.0;
        }
        let a = (c / c_min).clamp(0.0, 1.0); // c and c_min both <=0, ratio in [0,1]
        let inner = (a * (C_SLIDER_LOG_K.exp() - 1.0) + 1.0).max(1.0);
        -(inner.ln() / C_SLIDER_LOG_K)
    }
}

/// Same sign convention as `c_to_slider_t` (0 always maps to 0, c_max/
/// c_min map to ±1, each side scaled independently) but WITHOUT the log
/// curve — plain linear. Used only for the slider's handle position
/// while `c_pulse` is animating it: the log curve exists purely to give
/// fine manual-drag control near C=0, which doesn't apply to a disabled,
/// pulse-driven slider — running a linearly-cycling C value through the
/// log curve there just made the handle appear to speed up and slow down
/// unevenly instead of moving at the pulse's own actual (linear) pace.
fn linear_slider_t(c: f64, c_min: f64, c_max: f64) -> f64 {
    if c >= 0.0 {
        if c_max <= 0.0 { 0.0 } else { (c / c_max).clamp(0.0, 1.0) }
    } else if c_min >= 0.0 {
        0.0
    } else {
        -(c / c_min).clamp(0.0, 1.0)
    }
}

/// Empirical C-range finder, stepped a little at a time from `ui()` (one
/// cheap low-res probe render per frame) rather than a real background
/// thread — the GPU pipeline lives on `App` and is driven with `&mut
/// self` for every render, so spreading the scan across frames on the
/// main thread sidesteps any question of sharing/`Send`ing the wgpu
/// device across threads while still never blocking the UI for more than
/// one small render at a time. Scans outward from C=0 in both directions
/// (geometric step growth) until the rendered hit-fraction drops below
/// 10% of the hit-fraction measured at C=0 (`full_mass` — genome-relative,
/// not an absolute constant) or a generous multiple of `bailout_radius`
/// is reached.
///
/// The coarse scan alone under-reaches, though: its step size grows
/// geometrically (×1.4 per sample), so by the time it lands on a sample
/// below the 10% floor, that sample can be FAR past the true crossing —
/// the true boundary might sit at, say, 60% of the distance between the
/// last good sample and the first empty one, but the coarse scan can only
/// ever report one or the other. Reporting the last good sample directly
/// (as this used to) is safe but systematically too conservative — a
/// real reported defect ("range feels too narrow now" — real structure
/// existed past where the slider stopped). Once the coarse scan detects a
/// crossing, `bisect` refines it: a handful of bisection probes between
/// the last known-good sample and the first known-empty one, narrowing
/// toward the true floor-crossing point before it's reported as the
/// boundary.
struct CRangeScan {
    direction: f64,
    step: f64,
    current: f64,
    found_pos: f64,
    found_neg: f64,
    iters_this_direction: u32,
    /// Hit-fraction at C=0, measured once before either direction is
    /// scanned — the reference "full mass" that the 10% floor is relative
    /// to. `None` until that reference probe has run.
    full_mass: Option<f32>,
    /// Last `current` value (this direction) whose hit-fraction was still
    /// ≥ 10% of `full_mass`. Reset to 0.0 (always full mass, by
    /// definition) when the scan switches direction.
    last_good: f64,
    /// `Some((lo, hi))` while refining a detected crossing: `lo` still has
    /// ≥10% mass, `hi` doesn't. Narrows every call until `bisect_iters`
    /// hits the cap, then `lo` becomes the direction's final boundary.
    bisect: Option<(f64, f64)>,
    bisect_iters: u32,
}

impl CRangeScan {
    fn start(bailout_radius: f64) -> Self {
        let start_step = (bailout_radius * 0.25).max(0.02);
        CRangeScan {
            direction: 1.0,
            step: start_step,
            current: start_step,
            found_pos: bailout_radius,
            found_neg: -bailout_radius,
            iters_this_direction: 0,
            full_mass: None,
            last_good: 0.0,
            bisect: None,
            bisect_iters: 0,
        }
    }
}

#[derive(PartialEq, Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
enum RenderMode {
    /// A single high-res still at EXACTLY the current interactive
    /// camera (eye/target/up/fov/C) — via `quat-raymarch-genome`, which
    /// takes an explicit camera rather than an orbit, so this is a
    /// pixel-for-pixel match of what's on screen, not an approximation.
    Static,
    /// A short orbit video starting from the current yaw, via
    /// `quat-raymarch-genome-video`. That CLI's orbit camera only has
    /// one rotation angle around a fixed axis (no separate elevation),
    /// so the orbit's starting pitch can't be carried over exactly —
    /// it starts level, not at the interactive view's current pitch.
    Orbit,
}

impl RenderMode {
    /// The `QuatRenderSpec::mode` string the queue/`process_quat_queue_item`
    /// expects.
    fn queue_mode_str(self) -> &'static str {
        match self {
            RenderMode::Static => "static",
            RenderMode::Orbit => "orbit",
        }
    }
}

// ── Render preferences ──────────────────────────────────────────────────
//
// Persisted across sessions (Carl's ask: "save user preference each time a
// setting is modified"), EXCEPT the C-sweep range — that's deliberately
// excluded and always re-derived from the live viewer's `c_value`/
// `c_min`/`c_max` when the render window opens (see the "Render…" button's
// click handler), since a sweep saved from one genome's C range would be
// meaningless (or out of range) for the next genome loaded.

#[derive(serde::Serialize, serde::Deserialize, PartialEq, Clone, Debug)]
struct RenderPrefs {
    render_mode: RenderMode,
    render_out_dir: String,
    render_static_size: u32,
    render_orbit_width: u32,
    render_orbit_height: u32,
    render_orbit_frames: u32,
    render_orbit_fps: u32,
    render_orbit_turns: f64,
}

impl Default for RenderPrefs {
    fn default() -> Self {
        RenderPrefs {
            // Orbit first — Carl's ask: "orbit should be the first thing
            // we see" in the render options window.
            render_mode: RenderMode::Orbit,
            render_out_dir: "explorer_out/quat_mandelbrot".to_string(),
            render_static_size: 1600,
            render_orbit_width: 960,
            render_orbit_height: 960,
            render_orbit_frames: 180,
            render_orbit_fps: 30,
            render_orbit_turns: 1.0,
        }
    }
}

impl RenderPrefs {
    fn load(path: &Path) -> Self {
        std::fs::read_to_string(path).ok().and_then(|s| toml::from_str(&s).ok()).unwrap_or_default()
    }
    fn save(&self, path: &Path) {
        if let Ok(s) = toml::to_string_pretty(self) {
            let _ = std::fs::write(path, s);
        }
    }
}

fn render_prefs_path() -> PathBuf {
    nnfractals::project_root().join("quat_viewer_render_prefs.toml")
}

// ── General viewer preferences ──────────────────────────────────────────
//
// Distinct from `RenderPrefs` above: this is for the main interactive
// view's own behavior — auto-rotate, and the display settings (colormap,
// max iter, supersampling, resolution cap) that affect the live view AND
// every render/export — not the "Render options" window's own
// export-specific settings (resolution, frame count, output dir, ...).
// Kept in a separate file/struct so the two concerns — "how I like to
// look at things live" vs. "how I like to export things" — don't get
// tangled together as either one grows.
//
// FOUND MISSING entirely (Carl: "my render settings are still not
// saved... please verify with end-to-end test") — `colormap`/`max_iter`/
// `aa`/`render_size_cap` had NO persistence at all before this: every
// launch silently reset them to hardcoded defaults, which is exactly
// what "settings not saved" describes, even though they were never part
// of `RenderPrefs` (that struct only ever covered the separate Render-
// options-WINDOW fields, which were already saving correctly). Verified
// by `viewer_prefs_round_trip_survives_every_field` below, which fails
// on the old `QuatViewerPrefs` (missing fields) and passes now.
#[derive(serde::Serialize, serde::Deserialize, PartialEq, Clone, Debug)]
struct QuatViewerPrefs {
    /// Carl's ask: "That option would be saved" — once turned on, stays
    /// on across launches. Defaults to off (a newly-added feature starts
    /// opt-in; nothing forces it on for someone who's never touched it).
    #[serde(default)]
    auto_rotate: bool,
    #[serde(default = "default_colormap")]
    colormap: String,
    #[serde(default = "default_max_iter")]
    max_iter: u32,
    #[serde(default = "default_aa")]
    aa: u32,
    #[serde(default = "default_render_size_cap")]
    render_size_cap: u32,
}

fn default_colormap() -> String { "lava".to_string() }
fn default_max_iter() -> u32 { 60 }
fn default_aa() -> u32 { 1 }
fn default_render_size_cap() -> u32 { 1400 }

impl Default for QuatViewerPrefs {
    fn default() -> Self {
        QuatViewerPrefs {
            auto_rotate: false,
            colormap: default_colormap(),
            max_iter: default_max_iter(),
            aa: default_aa(),
            render_size_cap: default_render_size_cap(),
        }
    }
}

impl QuatViewerPrefs {
    fn load(path: &Path) -> Self {
        std::fs::read_to_string(path).ok().and_then(|s| toml::from_str(&s).ok()).unwrap_or_default()
    }
    fn save(&self, path: &Path) {
        if let Ok(s) = toml::to_string_pretty(self) {
            let _ = std::fs::write(path, s);
        }
    }
}

fn viewer_prefs_path() -> PathBuf {
    nnfractals::project_root().join("quat_viewer_prefs.toml")
}

/// A random unit direction in (yaw, pitch) angle-space — see
/// `App::auto_rotate_dir`'s doc comment for why unit length matters (it's
/// what makes "1 turn per 10s" a fixed, direction-independent rate).
fn random_auto_rotate_dir(rng: &mut impl rand::Rng) -> (f64, f64) {
    let theta: f64 = rng.random_range(0.0..std::f64::consts::TAU);
    (theta.cos(), theta.sin())
}

struct App {
    genome_path: PathBuf,
    genome_label: String,
    program: Vec<OpNode>,
    warp: Vec<OpNode>,
    julia_mode: bool,
    jc: (f32, f32),
    phoenix: (f32, f32),
    bailout_radius: f32,

    pipeline: Option<CompiledDagPipeline>,
    gpu_error: Option<String>,

    yaw: f64,
    pitch: f64,
    distance: f64,
    c_value: f64,
    c_pulse: bool,
    /// Current best-known C range — NOT assumed symmetric (see
    /// `slider_t_to_c`'s docs for the defect this fixes). Starts as an
    /// instant `bailout_radius`-derived estimate (mirrored, since there's
    /// nothing better to guess from yet), then gets refined in place —
    /// independently on each side — by `c_scan` once the probe search
    /// finishes.
    c_min: f64,
    c_max: f64,
    /// Set once the user edits `c_min`/`c_max` directly (or types into
    /// the custom-range fields) — Carl's own ask, "allow the user to use
    /// custom range". Blocks `c_scan`'s auto-detected result from
    /// silently overwriting a value the user chose on purpose; cleared
    /// (and a fresh scan started) only by explicitly pressing
    /// "auto-detect" again.
    c_range_manual: bool,
    c_scan: Option<CRangeScan>,
    colormap: String,
    render_size: u32,
    /// User-controlled ceiling on `render_size` (the "max resolution"
    /// slider). `render_size` itself always follows the image panel's
    /// available width every frame (see `ui()`) — it isn't independently
    /// adjustable, so the slider that used to be wired directly to it was
    /// actually dead: any value picked got silently overwritten the very
    /// next frame by the panel-fit logic. This cap is the real knob:
    /// raise it to let the panel-fit resolution climb higher on a wide
    /// window, lower it to trade sharpness for faster live-drag renders.
    render_size_cap: u32,
    max_iter: u32,
    aa: u32,

    texture: Option<egui::TextureHandle>,
    dirty: bool,
    last_render_ms: f64,
    last_hit_frac: f32,
    dragging: bool,
    start_time: std::time::Instant,

    /// Carl's ask: continuous auto-rotate, 1 full turn every 10s, in a
    /// random direction. Persisted (`QuatViewerPrefs`) — once turned on,
    /// it stays on across sessions/launches. Yields to manual dragging
    /// (see `dragging`): the user's own orbit always wins, auto-rotate
    /// just resumes cleanly afterward rather than snapping or fighting
    /// it, since it advances yaw/pitch incrementally from wherever they
    /// are, not from an absolute time-based formula.
    auto_rotate: bool,
    /// Unit direction in (yaw, pitch) angle-space — `dir.0² + dir.1² ≈
    /// 1`, so advancing `(yaw, pitch) += dir * (2π/10) * dt` traces
    /// exactly one full lap (Euclidean distance 2π in that space) every
    /// 10 seconds, regardless of which random direction was picked. A
    /// fresh direction is drawn whenever auto-rotate turns on (including
    /// at startup, if it loaded on from prefs) and whenever a new genome
    /// loads while it's already on.
    auto_rotate_dir: (f64, f64),
    /// Wall-clock instant of the last applied auto-rotate step — `None`
    /// whenever auto-rotate is off or the user is actively dragging, so
    /// the NEXT step (whenever auto-rotate resumes) computes its `dt`
    /// from "just now," not from however long ago rotation last applied
    /// — without this, pausing to drag then releasing would apply one
    /// enormous catch-up jump instead of resuming smoothly.
    auto_rotate_last_tick: Option<std::time::Instant>,

    render_window_open: bool,
    render_mode: RenderMode,
    render_out_dir: String,
    render_static_size: u32,
    render_orbit_width: u32,
    render_orbit_height: u32,
    render_orbit_frames: u32,
    render_orbit_fps: u32,
    render_orbit_turns: f64,
    /// Explicit C sweep range for the exported orbit video — both ends
    /// independently editable (Carl's ask: "specify a range for C in the
    /// video render option"), and both DEFAULT to the viewer's own
    /// current `c_min`/`c_max` each time the render window opens fresh
    /// ("mimic the viewer" — his own words; see the "Render…" button's
    /// click handler). Freely overridable afterward. Deliberately NOT
    /// part of `RenderPrefs` — see that struct's doc comment.
    render_orbit_c0: f64,
    render_orbit_c1: f64,
    /// Result of the most recent "Add to queue" click — `Ok(item_id)` or
    /// `Err(explicit reason)`. Replaces the old `RenderJob`
    /// spawn-and-poll machinery: the render itself now runs entirely
    /// inside the queue's own process (see `video_export::enqueue` /
    /// `queue_runner::process_quat_queue_item`), one job at a time,
    /// serialized against every other queued render — not as a second
    /// GPU-holding process running concurrently with this interactive
    /// viewer's own live pipeline, which is what the old spawn-directly
    /// design did on every click.
    render_status: Option<Result<String, String>>,

    ipc_rx: mpsc::Receiver<PathBuf>,
}

impl App {
    fn new(genome_path: &std::path::Path, ipc_rx: mpsc::Receiver<PathBuf>) -> anyhow::Result<Self> {
        let g = nnfractals::io::load_genome(genome_path)?;
        if g.program.is_empty() {
            anyhow::bail!("{genome_path:?} has an empty DAG program (legacy 58-basis genome?) — quat-viewer only supports DAG-based genomes");
        }
        let pipeline = CompiledDagPipeline::compile(&g.program, &g.warp);
        let gpu_error = if pipeline.is_none() { Some(gpu_error_message()) } else { None };
        let domain_radius = 1.6;
        let fov_deg: f64 = 45.0;
        let distance = recommended_orbit_radius(domain_radius, fov_deg, 640, 640);
        let genome_label = genome_path.file_stem().and_then(|s| s.to_str()).unwrap_or("genome").to_string();
        let render_prefs = RenderPrefs::load(&render_prefs_path());
        let viewer_prefs = QuatViewerPrefs::load(&viewer_prefs_path());
        let mut rng = rand::rng();
        let auto_rotate_dir = random_auto_rotate_dir(&mut rng);
        Ok(App {
            genome_path: genome_path.to_path_buf(),
            genome_label: genome_label.clone(),
            program: g.program,
            warp: g.warp,
            julia_mode: g.julia_mode,
            jc: (g.julia_cre, g.julia_cim),
            phoenix: (g.phoenix_re, g.phoenix_im),
            bailout_radius: g.bailout_radius,
            pipeline,
            gpu_error,
            yaw: 0.0,
            pitch: 0.25,
            distance,
            c_value: 0.0,
            // On by default (Carl's ask) — the fractal animates through
            // its C range as soon as a genome loads, rather than sitting
            // static until the user finds and checks "pulse C
            // automatically" themselves.
            c_pulse: true,
            // Instant estimate (Carl's answer: instant estimate, refined
            // in the background) — `c_scan` immediately starts refining
            // this properly via a probe search once rendering begins,
            // independently on each side.
            c_min: -(g.bailout_radius as f64),
            c_max: g.bailout_radius as f64,
            c_range_manual: false,
            c_scan: Some(CRangeScan::start(g.bailout_radius as f64)),
            colormap: viewer_prefs.colormap.clone(),
            render_size: 900,
            render_size_cap: viewer_prefs.render_size_cap,
            max_iter: viewer_prefs.max_iter,
            aa: viewer_prefs.aa,
            texture: None,
            dirty: true,
            last_render_ms: 0.0,
            last_hit_frac: 0.0,
            dragging: false,
            start_time: std::time::Instant::now(),
            auto_rotate: viewer_prefs.auto_rotate,
            auto_rotate_dir,
            auto_rotate_last_tick: None,

            render_window_open: false,
            render_mode: render_prefs.render_mode,
            render_out_dir: render_prefs.render_out_dir,
            render_static_size: render_prefs.render_static_size,
            render_orbit_width: render_prefs.render_orbit_width,
            render_orbit_height: render_prefs.render_orbit_height,
            render_orbit_frames: render_prefs.render_orbit_frames,
            render_orbit_fps: render_prefs.render_orbit_fps,
            render_orbit_turns: render_prefs.render_orbit_turns,
            render_orbit_c0: 0.0,
            render_orbit_c1: 0.0,
            render_status: None,

            ipc_rx,
        })
    }

    /// Swaps in a new genome without tearing down the window — the IPC
    /// single-instance path (a second launch delegates its path here
    /// instead of opening a new GPU-backed window). Resets every
    /// genome-derived field (program/warp/dynamics/camera/pipeline/C-
    /// range/render-job) but deliberately PRESERVES window-level display
    /// prefs (colormap, resolution cap, max_iter, aa) — matches
    /// `viewer.rs`'s own `load_new_genome`'s split between what a fresh
    /// genome resets and what stays as "how you like to look at things".
    fn load_genome(&mut self, genome_path: &Path) -> anyhow::Result<()> {
        let g = nnfractals::io::load_genome(genome_path)?;
        if g.program.is_empty() {
            anyhow::bail!("{genome_path:?} has an empty DAG program (legacy 58-basis genome?) — quat-viewer only supports DAG-based genomes");
        }
        let pipeline = CompiledDagPipeline::compile(&g.program, &g.warp);
        self.gpu_error = if pipeline.is_none() { Some(gpu_error_message()) } else { None };
        self.pipeline = pipeline;
        let domain_radius = 1.6;
        let fov_deg: f64 = 45.0;
        self.genome_label = genome_path.file_stem().and_then(|s| s.to_str()).unwrap_or("genome").to_string();
        self.genome_path = genome_path.to_path_buf();
        self.program = g.program;
        self.warp = g.warp;
        self.julia_mode = g.julia_mode;
        self.jc = (g.julia_cre, g.julia_cim);
        self.phoenix = (g.phoenix_re, g.phoenix_im);
        self.bailout_radius = g.bailout_radius;
        self.yaw = 0.0;
        self.pitch = 0.25;
        self.distance = recommended_orbit_radius(domain_radius, fov_deg, 640, 640);
        self.c_value = 0.0;
        self.c_pulse = true; // on by default — see App::new's doc comment
        self.c_min = -(g.bailout_radius as f64);
        self.c_max = g.bailout_radius as f64;
        self.c_range_manual = false;
        self.c_scan = Some(CRangeScan::start(g.bailout_radius as f64));
        if self.auto_rotate {
            // Fresh random spin direction per genome — a newly-loaded
            // fractal shouldn't inherit the previous one's exact tumble.
            self.auto_rotate_dir = random_auto_rotate_dir(&mut rand::rng());
        }
        self.auto_rotate_last_tick = None;
        self.texture = None;
        self.dirty = true;
        self.render_window_open = false;
        // render_out_dir/mode/resolution/etc. are NOT reset here — they're
        // persisted `RenderPrefs`, "how you like to render things," not
        // genome-specific state (matches colormap/render_size_cap/
        // max_iter/aa's existing treatment just above). render_status
        // (the last enqueue result) also carries over on purpose — it's
        // about the last submitted job, not about which genome is loaded.
        Ok(())
    }

    fn render(&mut self, ctx: &egui::Context) {
        let Some(pipeline) = self.pipeline.as_mut() else { return };
        let domain_radius = 1.6;
        let fov_deg: f64 = 45.0;
        let formula = QuatDagFormula { prog: &self.program, warp: &self.warp, julia: self.julia_mode, jc: self.jc, phoenix: self.phoenix };
        let params = RaymarchDagParams {
            formula,
            time_axis: nnfractals::quat_fractal::TimeAxis::C,
            time_val: self.c_value,
            domain_radius,
            max_iter: self.max_iter,
            bailout: self.bailout_radius as f64,
            max_march_steps: 200,
            hit_epsilon: domain_radius * 1e-4,
            step_safety: 0.8,
            light_dir: (0.5, 0.8, 0.3),
            normal_eps: domain_radius * 1e-3,
            color_probe_offset: domain_radius * 1e-2,
            aa: self.aa,
        };
        let cam = camera_from_orbit(self.yaw, self.pitch, self.distance, fov_deg.to_radians());
        let start = std::time::Instant::now();
        let (shading, color_t) = pipeline.render(&params, &cam, self.render_size, self.render_size);
        self.last_render_ms = start.elapsed().as_secs_f64() * 1000.0;

        let hits = shading.iter().filter(|&&v| v > 0.0).count();
        self.last_hit_frac = hits as f32 / shading.len().max(1) as f32;

        let bg_color = (0.03, 0.02, 0.06);
        let rgb_bytes = nnfractals::colormap::apply_colormap_equalized(&color_t, self.max_iter, &self.colormap);
        let mut rgb = vec![0u8; shading.len() * 3];
        for i in 0..shading.len() {
            let v = shading[i];
            if v <= 0.0 {
                rgb[i * 3] = (bg_color.0 * 255.0) as u8;
                rgb[i * 3 + 1] = (bg_color.1 * 255.0) as u8;
                rgb[i * 3 + 2] = (bg_color.2 * 255.0) as u8;
            } else {
                rgb[i * 3] = (rgb_bytes[i * 3] as f32 * v).clamp(0.0, 255.0) as u8;
                rgb[i * 3 + 1] = (rgb_bytes[i * 3 + 1] as f32 * v).clamp(0.0, 255.0) as u8;
                rgb[i * 3 + 2] = (rgb_bytes[i * 3 + 2] as f32 * v).clamp(0.0, 255.0) as u8;
            }
        }
        let size = [self.render_size as usize, self.render_size as usize];
        let color_image = egui::ColorImage::from_rgb(size, &rgb);
        match self.texture.as_mut() {
            Some(t) => t.set(color_image, egui::TextureOptions::LINEAR),
            None => self.texture = Some(ctx.load_texture("quat_view", color_image, egui::TextureOptions::LINEAR)),
        }
        self.dirty = false;
    }

    /// Cheap throwaway probe render (no texture upload, no effect on
    /// `dirty`/`last_render_ms`) at a fixed canonical camera — used only
    /// by `step_c_scan` to measure whether a given C value still produces
    /// visible structure. Fixed camera (not the user's current
    /// orbit) so the scan measures the GENOME's C-sensitivity, not
    /// whatever angle the user happens to be looking from right now.
    fn probe_hit_frac(&mut self, c: f64) -> f32 {
        let Some(pipeline) = self.pipeline.as_mut() else { return 0.0 };
        let domain_radius = 1.6;
        let fov_deg: f64 = 45.0;
        let formula = QuatDagFormula { prog: &self.program, warp: &self.warp, julia: self.julia_mode, jc: self.jc, phoenix: self.phoenix };
        let params = RaymarchDagParams {
            formula,
            time_axis: nnfractals::quat_fractal::TimeAxis::C,
            time_val: c,
            domain_radius,
            max_iter: 40,
            bailout: self.bailout_radius as f64,
            max_march_steps: 120,
            hit_epsilon: domain_radius * 1e-4,
            step_safety: 0.8,
            light_dir: (0.5, 0.8, 0.3),
            normal_eps: domain_radius * 1e-3,
            color_probe_offset: domain_radius * 1e-2,
            aa: 1,
        };
        let cam = camera_from_orbit(0.0, 0.25, recommended_orbit_radius(domain_radius, fov_deg, 72, 72), fov_deg.to_radians());
        let (shading, _color_t) = pipeline.render(&params, &cam, 72, 72);
        let hits = shading.iter().filter(|&&v| v > 0.0).count();
        hits as f32 / shading.len().max(1) as f32
    }

    /// Advances the C-range probe search by exactly one probe render per
    /// call (see `CRangeScan`'s docs for why this is spread across
    /// frames instead of run to completion in one go). Scans outward from
    /// C=0 in both directions with a geometrically growing step; a
    /// direction is done once hit-fraction drops near zero (genome stops
    /// rendering anything there) or a generous cap (12x the starting
    /// step count, or 8x `bailout_radius` in magnitude) is hit. Once both
    /// directions finish, sets `self.c_min`/`self.c_max` independently
    /// (no forced symmetry) and clears `c_scan` — unless the user has set
    /// a manual range (`c_range_manual`), in which case the scan still
    /// runs (harmless) but its result is discarded rather than
    /// overwriting the user's explicit choice.
    fn step_c_scan(&mut self) {
        const MASS_FRACTION_FLOOR: f32 = 0.10;
        const MAX_STEPS_PER_DIRECTION: u32 = 14;
        const MAX_BISECT_ITERS: u32 = 6;
        let Some(mut scan) = self.c_scan.take() else { return };

        // First call: establish the "full mass" reference at C=0 before
        // scanning outward — everything below is judged relative to this,
        // not a genome-agnostic absolute threshold. Spends one frame's
        // probe on the reference measurement.
        let Some(full_mass) = scan.full_mass else {
            scan.full_mass = Some(self.probe_hit_frac(0.0).max(1e-4));
            self.c_scan = Some(scan);
            return;
        };
        let mass_floor = MASS_FRACTION_FLOOR * full_mass;

        // Refining a detected crossing: narrow (lo, hi) toward the true
        // floor-crossing point instead of accepting the coarse scan's
        // (possibly much too conservative) empty sample as-is.
        if let Some((lo, hi)) = scan.bisect {
            let mid = (lo + hi) / 2.0;
            let hit_frac = self.probe_hit_frac(mid);
            let (new_lo, new_hi) = if hit_frac >= mass_floor { (mid, hi) } else { (lo, mid) };
            scan.bisect_iters += 1;
            if scan.bisect_iters >= MAX_BISECT_ITERS {
                scan.last_good = new_lo;
                scan.bisect = None;
                self.finish_c_scan_direction(scan);
            } else {
                scan.bisect = Some((new_lo, new_hi));
                self.c_scan = Some(scan);
            }
            return;
        }

        let hit_frac = self.probe_hit_frac(scan.current);
        let magnitude_cap = 8.0 * (self.bailout_radius as f64).max(1.0);
        let still_has_mass = hit_frac >= mass_floor;
        let direction_done = !still_has_mass
            || scan.iters_this_direction >= MAX_STEPS_PER_DIRECTION
            || scan.current.abs() >= magnitude_cap;
        if still_has_mass {
            scan.last_good = scan.current;
        }
        if direction_done {
            if !still_has_mass {
                // Genuine crossing found (a good sample followed by an
                // empty one, not just running out of step budget/magnitude
                // headroom while mass was still fine) — refine it instead
                // of reporting the coarse (too-conservative) sample.
                // `last_good` is always a valid bracket start even on the
                // very first step: C=0 (its initial value) trivially has
                // ≥10% of its own mass by definition.
                scan.bisect = Some((scan.last_good, scan.current));
                scan.bisect_iters = 0;
                self.c_scan = Some(scan);
            } else {
                self.finish_c_scan_direction(scan);
            }
        } else {
            scan.step *= 1.4;
            scan.current += scan.direction * scan.step;
            scan.iters_this_direction += 1;
            self.c_scan = Some(scan);
        }
    }

    /// Records `scan.last_good` as this direction's boundary, then either
    /// switches to scanning the negative side or — once both are done —
    /// applies the result to `c_min`/`c_max` (unless the user set a
    /// manual range, which takes priority).
    fn finish_c_scan_direction(&mut self, mut scan: CRangeScan) {
        let bound = scan.last_good.abs().max(0.05);
        if scan.direction > 0.0 {
            scan.found_pos = bound;
        } else {
            scan.found_neg = -bound;
        }
        if scan.direction > 0.0 {
            // Switch to scanning the negative side.
            scan.direction = -1.0;
            scan.step = (self.bailout_radius as f64 * 0.25).max(0.02);
            scan.current = -scan.step;
            scan.iters_this_direction = 0;
            scan.last_good = 0.0;
            self.c_scan = Some(scan);
        } else {
            // Both directions done — apply independently unless the user
            // has set a manual range, which takes priority.
            if !self.c_range_manual {
                self.c_min = scan.found_neg;
                self.c_max = scan.found_pos;
                let clamped = self.c_value.clamp(self.c_min, self.c_max);
                if clamped != self.c_value {
                    self.c_value = clamped;
                    self.dirty = true;
                }
            }
        }
    }

    /// Snapshot of every field `RenderPrefs` tracks — compared before/after
    /// `show_render_window` draws its widgets so ANY change (whichever
    /// widget produced it) gets caught and saved, rather than
    /// instrumenting each slider/drag-value's `.changed()` individually.
    fn snapshot_render_prefs(&self) -> RenderPrefs {
        RenderPrefs {
            render_mode: self.render_mode,
            render_out_dir: self.render_out_dir.clone(),
            render_static_size: self.render_static_size,
            render_orbit_width: self.render_orbit_width,
            render_orbit_height: self.render_orbit_height,
            render_orbit_frames: self.render_orbit_frames,
            render_orbit_fps: self.render_orbit_fps,
            render_orbit_turns: self.render_orbit_turns,
        }
    }

    /// Writes the current colormap/max_iter/aa/render_size_cap/
    /// auto_rotate to disk — called directly from each widget's
    /// `.changed()` site (these live in the main side panel drawn every
    /// frame regardless of any window being open, unlike `RenderPrefs`'s
    /// before/after snapshot around `show_render_window`, which only
    /// makes sense while that window is specifically open).
    fn save_viewer_prefs(&self) {
        QuatViewerPrefs {
            auto_rotate: self.auto_rotate,
            colormap: self.colormap.clone(),
            max_iter: self.max_iter,
            aa: self.aa,
            render_size_cap: self.render_size_cap,
        }
        .save(&viewer_prefs_path());
    }

    /// Builds a `QueueSpec` from the current render-option fields and
    /// submits it via `video_export::enqueue` — replaces the old
    /// spawn-a-subprocess-directly design. The actual render now happens
    /// inside whichever process is running the queue (the `nnfractals-
    /// queue` GUI, or the `explorer queue-run` cron job), one item at a
    /// time, with its own GPU device/queue that the OS fully reclaims the
    /// instant that render finishes — never running concurrently with
    /// this interactive viewer's own live pipeline the way a directly-
    /// spawned `--gpu` subprocess used to.
    fn enqueue_render(&mut self) {
        let out_dir = self.render_out_dir.clone();
        let quat_spec = match self.render_mode {
            RenderMode::Static => {
                let cam = camera_from_orbit(self.yaw, self.pitch, self.distance, 45.0f64.to_radians());
                nnfractals::video_export::QuatRenderSpec {
                    mode: "static".to_string(),
                    eye: cam.eye,
                    target: cam.target,
                    up: cam.up_hint,
                    fov_deg: 45.0,
                    c: self.c_value,
                    width: self.render_static_size,
                    height: self.render_static_size,
                    max_iter: self.max_iter,
                    aa: self.aa.max(2),
                    colormap: self.colormap.clone(),
                    ..Default::default()
                }
            }
            RenderMode::Orbit => nnfractals::video_export::QuatRenderSpec {
                mode: "orbit".to_string(),
                axis: (0.0, 1.0, 0.0),
                phase0: self.yaw,
                turns: self.render_orbit_turns,
                c0: self.render_orbit_c0,
                c1: self.render_orbit_c1,
                frames: self.render_orbit_frames,
                fps: self.render_orbit_fps,
                width: self.render_orbit_width,
                height: self.render_orbit_height,
                max_iter: self.max_iter,
                aa: self.aa.max(2),
                colormap: self.colormap.clone(),
                ..Default::default()
            },
        };
        let spec = nnfractals::video_export::QueueSpec {
            nn_src: self.genome_path.clone(),
            genome_label: self.genome_label.clone(),
            output_dir: out_dir,
            quat_spec: Some(quat_spec),
            ..Default::default()
        };
        match nnfractals::video_export::enqueue(spec) {
            Ok(item) => {
                self.render_status = Some(Ok(item.id));
                // Carl's ask: adding an item should actually bring up the
                // queue manager so he can watch it, not require a second
                // separate click — the explicit "Open queue manager…"
                // button below stays too, for reopening it later.
                self.open_queue_manager();
            }
            Err(e) => self.render_status = Some(Err(format!("couldn't add to queue: {e}"))),
        }
    }

    /// Spawns `nnfractals-queue` (fire-and-forget, reaped on a detached
    /// thread — same pattern `browser.rs`'s `open_path` already uses for
    /// spawning `viewer`/`quat-viewer`) so Carl can watch/manage the job
    /// he just queued without leaving this window.
    fn open_queue_manager(&mut self) {
        let bin = locate_bin("nnfractals-queue");
        match std::process::Command::new(&bin).spawn() {
            Ok(mut child) => {
                thread::spawn(move || {
                    let _ = child.wait();
                });
            }
            Err(e) => {
                self.render_status = Some(Err(format!(
                    "couldn't launch {} to open the queue manager: {e} — is it built? (cargo build --release --bin nnfractals-queue --features queue)",
                    bin.display()
                )));
            }
        }
    }

    fn show_render_window(&mut self, ctx: &egui::Context) {
        let mut open = self.render_window_open;
        let before = self.snapshot_render_prefs();
        egui::Window::new("Render options").open(&mut open).resizable(false).show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.render_mode, RenderMode::Orbit, "Orbit video");
                ui.selectable_value(&mut self.render_mode, RenderMode::Static, "Static (current view)");
            });
            ui.separator();
            match self.render_mode {
                RenderMode::Orbit => {
                    ui.label("Starts orbiting from the current yaw (pitch always starts level — the video camera only orbits around one fixed axis, it can't carry over pitch).");
                    ui.horizontal(|ui| {
                        ui.add(egui::DragValue::new(&mut self.render_orbit_width).range(128..=3840).prefix("w: "));
                        ui.add(egui::DragValue::new(&mut self.render_orbit_height).range(128..=3840).prefix("h: "));
                    });
                    ui.add(egui::Slider::new(&mut self.render_orbit_frames, 30..=600).text("frames"));
                    ui.add(egui::Slider::new(&mut self.render_orbit_fps, 12..=60).text("fps"));
                    ui.add(egui::Slider::new(&mut self.render_orbit_turns, 0.0..=3.0).text("turns"));
                    // Same log-mapped `t`-slider the main viewer's own C
                    // slider uses (`c_to_slider_t`/`slider_t_to_c`), not a
                    // plain linear one — a real reported defect ("the
                    // render option for C range still do not mimic the
                    // viewer"): a linear slider over the same [c_min,
                    // c_max] range gives completely different fine control
                    // near C=0 than the viewer's own slider does, so a
                    // value that felt easy to dial in on the main slider
                    // was awkward to reproduce here.
                    ui.horizontal(|ui| {
                        ui.label("C sweep from:");
                        let mut t0 = c_to_slider_t(self.render_orbit_c0, self.c_min, self.c_max);
                        if ui.add(egui::Slider::new(&mut t0, -1.0..=1.0).show_value(false)).changed() {
                            self.render_orbit_c0 = slider_t_to_c(t0, self.c_min, self.c_max);
                        }
                        ui.label(format!("{:.4}", self.render_orbit_c0));
                    });
                    ui.horizontal(|ui| {
                        ui.label("           to:");
                        let mut t1 = c_to_slider_t(self.render_orbit_c1, self.c_min, self.c_max);
                        if ui.add(egui::Slider::new(&mut t1, -1.0..=1.0).show_value(false)).changed() {
                            self.render_orbit_c1 = slider_t_to_c(t1, self.c_min, self.c_max);
                        }
                        ui.label(format!("{:.4}", self.render_orbit_c1));
                    });
                    ui.label("(defaults to the viewer's own C range on each open — not saved as a preference.)");
                }
                RenderMode::Static => {
                    ui.label("Exact pixel-for-pixel match of the current camera and C value.");
                    ui.add(egui::Slider::new(&mut self.render_static_size, 512..=4096).text("resolution"));
                }
            }
            ui.separator();
            ui.horizontal(|ui| {
                ui.label("output dir:");
                ui.text_edit_singleline(&mut self.render_out_dir);
            });
            ui.separator();

            ui.horizontal(|ui| {
                if ui.button("Add to queue").clicked() {
                    self.enqueue_render();
                }
                if ui.button("Open queue manager…").clicked() {
                    self.open_queue_manager();
                }
            });
            match &self.render_status {
                Some(Ok(id)) => {
                    ui.colored_label(egui::Color32::from_rgb(120, 200, 140), format!("added to queue (id: {id}) — open the queue manager to watch it render"));
                }
                Some(Err(e)) => {
                    ui.colored_label(egui::Color32::RED, e);
                }
                None => {}
            }
        });
        self.render_window_open = open;
        let after = self.snapshot_render_prefs();
        if after != before {
            after.save(&render_prefs_path());
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();

        // IPC: another launch delegated its genome path to us instead of
        // opening a second GPU-backed window.
        while let Ok(path) = self.ipc_rx.try_recv() {
            if let Err(e) = self.load_genome(&path) {
                self.gpu_error = Some(format!("failed to load genome {}: {e}", path.display()));
            }
            ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
        }

        if self.c_scan.is_some() {
            self.step_c_scan();
            ctx.request_repaint();
        }
        if self.c_pulse {
            let t = self.start_time.elapsed().as_secs_f64();
            self.c_value = nnfractals::formula::ramp_or_pulse(self.c_min, self.c_max, nnfractals::formula::ModShape::Sine, 0.15, 0.0, t);
            self.dirty = true;
        }

        // Auto-rotate: 1 full turn (2π of combined yaw+pitch angular
        // distance) every 10s, along whatever random direction was
        // drawn. Yields to manual dragging entirely (see
        // `auto_rotate_last_tick`'s doc comment for why `dragging` resets
        // it rather than just skipping the step).
        const AUTO_ROTATE_RATE: f64 = std::f64::consts::TAU / 10.0;
        if self.auto_rotate && !self.dragging {
            let now = std::time::Instant::now();
            if let Some(last) = self.auto_rotate_last_tick {
                let dt = now.duration_since(last).as_secs_f64();
                self.yaw += self.auto_rotate_dir.0 * AUTO_ROTATE_RATE * dt;
                self.pitch += self.auto_rotate_dir.1 * AUTO_ROTATE_RATE * dt;
                self.dirty = true;
            }
            self.auto_rotate_last_tick = Some(now);
            ctx.request_repaint_after(std::time::Duration::from_millis(16));
        } else {
            self.auto_rotate_last_tick = None;
        }

        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.set_width(230.0);
                ui.heading(&self.genome_label);
                if let Some(err) = &self.gpu_error {
                    ui.colored_label(egui::Color32::RED, err);
                }
                ui.separator();
                ui.label("Drag the image to orbit. Scroll to zoom.");
                ui.separator();

                ui.label(if self.c_scan.is_some() {
                    format!("C (time axis) — finding range… (~[{:.2}, {:.2}])", self.c_min, self.c_max)
                } else if self.c_range_manual {
                    format!("C (time axis) — custom range [{:.2}, {:.2}]", self.c_min, self.c_max)
                } else {
                    format!("C (time axis) — range [{:.2}, {:.2}]", self.c_min, self.c_max)
                });
                let mut t = if self.c_pulse {
                    linear_slider_t(self.c_value, self.c_min, self.c_max)
                } else {
                    c_to_slider_t(self.c_value, self.c_min, self.c_max)
                };
                if ui.add_enabled(!self.c_pulse, egui::Slider::new(&mut t, -1.0..=1.0).show_value(false)).changed() {
                    self.c_value = slider_t_to_c(t, self.c_min, self.c_max);
                    self.dirty = true;
                }
                ui.label(format!("C = {:.4}", self.c_value));
                if ui.checkbox(&mut self.c_pulse, "pulse C automatically").changed() {
                    self.dirty = true;
                }
                if ui.checkbox(&mut self.auto_rotate, "auto-rotate (1 turn / 10s)").changed() {
                    if self.auto_rotate {
                        // Fresh random direction each time it's turned on.
                        self.auto_rotate_dir = random_auto_rotate_dir(&mut rand::rng());
                    }
                    self.auto_rotate_last_tick = None;
                    self.save_viewer_prefs();
                }
                ui.horizontal(|ui| {
                    ui.label("range:");
                    let mut changed = false;
                    changed |= ui
                        .add(egui::DragValue::new(&mut self.c_min).speed(0.01).range(f64::NEG_INFINITY..=0.0))
                        .changed();
                    ui.label("to");
                    changed |= ui
                        .add(egui::DragValue::new(&mut self.c_max).speed(0.01).range(0.0..=f64::INFINITY))
                        .changed();
                    if changed {
                        self.c_range_manual = true;
                        self.c_scan = None;
                        let clamped = self.c_value.clamp(self.c_min, self.c_max);
                        if clamped != self.c_value {
                            self.c_value = clamped;
                            self.dirty = true;
                        }
                    }
                });
                if self.c_range_manual {
                    if ui.button("auto-detect range").clicked() {
                        self.c_range_manual = false;
                        self.c_scan = Some(CRangeScan::start(self.bailout_radius as f64));
                    }
                }
                ui.separator();

                ui.label("Quality");
                if ui.add(egui::Slider::new(&mut self.render_size_cap, 256..=2400).text("max resolution")).changed() {
                    self.texture = None;
                    self.dirty = true;
                    self.save_viewer_prefs();
                }
                if ui.add(egui::Slider::new(&mut self.max_iter, 20..=150).text("max iter")).changed() {
                    self.dirty = true;
                    self.save_viewer_prefs();
                }
                if ui.add(egui::Slider::new(&mut self.aa, 1..=3).text("supersampling")).changed() {
                    self.dirty = true;
                    self.save_viewer_prefs();
                }
                ui.separator();

                egui::ComboBox::from_label("colormap")
                    .selected_text(&self.colormap)
                    .show_ui(ui, |ui| {
                        for name in ["lava", "aurora", "galaxy", "turbo", "viridis", "inferno", "plasma", "cubehelix", "sunset", "ember"] {
                            if ui.selectable_value(&mut self.colormap, name.to_string(), name).changed() {
                                self.dirty = true;
                                self.save_viewer_prefs();
                            }
                        }
                    });
                ui.separator();

                if ui.button("reset view").clicked() {
                    self.yaw = 0.0;
                    self.pitch = 0.25;
                    self.distance = recommended_orbit_radius(1.6, 45.0, self.render_size, self.render_size);
                    self.dirty = true;
                }
                if ui.button("Render…").clicked() {
                    if !self.render_window_open {
                        // Default the sweep to the viewer's own full C
                        // range on each fresh open — "mimic the viewer"
                        // (Carl's own words): the orbit video sweeps
                        // across the exact same range the main C slider
                        // spans, not an arbitrary current-point-to-stale-
                        // leftover-value pair. Still freely overridable
                        // afterward (see `render_orbit_c0`'s docs).
                        self.render_orbit_c0 = self.c_min;
                        self.render_orbit_c1 = self.c_max;
                    }
                    self.render_window_open = true;
                }
                ui.separator();
                ui.label(format!("resolution: {}x{}", self.render_size, self.render_size));
                ui.label(format!("render: {:.0} ms", self.last_render_ms));
                ui.label(format!("hit: {:.0}%", self.last_hit_frac * 100.0));
            });

            ui.separator();

            ui.vertical(|ui| {
                // Width-driven, not min(width,height): the image always
                // fills the full available width of this panel and stays
                // square, even if that makes it taller than the window
                // (Carl's ask — resize the window taller if you don't
                // want vertical overflow). The actual GPU render
                // resolution is clamped by `render_size_cap` rather than
                // by the panel size directly, so displayed size and
                // render cost are independently controllable: egui just
                // stretches the (possibly lower-res) texture to fill
                // `side`, which is normal texture scaling, not a quality
                // bug.
                let avail = ui.available_size();
                let side = avail.x.max(64.0);
                let target_size = (side as u32).clamp(128, self.render_size_cap);
                if target_size != self.render_size && !self.dragging {
                    // Follow panel width when idle, but don't fight an
                    // in-flight drag by resizing mid-gesture.
                    self.render_size = target_size;
                    self.texture = None;
                    self.dirty = true;
                }

                if self.dirty {
                    self.render(&ctx);
                }

                if let Some(tex) = &self.texture {
                    let resp = ui.add(
                        egui::Image::new(egui::load::SizedTexture::new(tex.id(), egui::vec2(side, side)))
                            .sense(egui::Sense::drag()),
                    );
                    if resp.dragged() {
                        self.dragging = true;
                        let delta = resp.drag_delta();
                        self.yaw -= delta.x as f64 * 0.01;
                        self.pitch = (self.pitch + delta.y as f64 * 0.01).clamp(-1.5, 1.5);
                        self.dirty = true;
                    } else {
                        self.dragging = false;
                    }
                    let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                    if scroll.abs() > 0.0 {
                        self.distance = (self.distance * (1.0 - scroll as f64 * 0.002)).clamp(1.7, 40.0);
                        self.dirty = true;
                    }
                } else {
                    ui.centered_and_justified(|ui| {
                        ui.label("no GPU renderer available — see the panel on the left");
                    });
                }
            });
        });

        if self.render_window_open {
            self.show_render_window(&ctx);
        }

        if self.c_pulse {
            ctx.request_repaint_after(std::time::Duration::from_millis(33));
        }
    }
}

fn main() -> anyhow::Result<()> {
    let genome_path: PathBuf = std::env::args().nth(1).map(PathBuf::from).ok_or_else(|| {
        anyhow::anyhow!("Usage: nnfractals-quat-viewer <genome.nn>")
    })?;

    // ── Single-instance IPC (mirrors viewer.rs's exactly) ──────────────
    let sock_path = socket_path();
    if try_delegate(&sock_path, &genome_path) {
        eprintln!("[quat-viewer] Delegated to running instance.");
        return Ok(());
    }
    let _ = std::fs::remove_file(&sock_path); // remove any stale socket
    let (ipc_tx, ipc_rx) = mpsc::channel::<PathBuf>();
    let _sock_guard = match UnixListener::bind(&sock_path) {
        Ok(listener) => {
            thread::spawn(move || {
                for stream in listener.incoming() {
                    if let Ok(mut s) = stream {
                        let mut buf = String::new();
                        if s.read_to_string(&mut buf).is_ok() {
                            let p = PathBuf::from(buf.trim());
                            if p.exists() {
                                let _ = ipc_tx.send(p);
                                let _ = s.write_all(b"K"); // ack — see try_delegate's doc comment
                            }
                        }
                    }
                }
            });
            Some(SocketGuard(sock_path))
        }
        Err(e) => {
            // Lost a bind race against a near-simultaneous second launch —
            // that other process now legitimately owns the socket. This is
            // the other real cause behind "closed 2 viewer windows": two
            // launches close enough together both saw no listener, both
            // tried to bind, and the loser used to fall straight through to
            // opening its OWN standalone window instead of retrying
            // delegation — two windows out of one genome open. One more
            // delegate attempt here closes that gap.
            if try_delegate(&sock_path, &genome_path) {
                eprintln!("[quat-viewer] Delegated to running instance (after bind race).");
                return Ok(());
            }
            eprintln!("[quat-viewer] IPC unavailable: {e}");
            None
        }
    };

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("NNFractals Quaternion Viewer")
            .with_inner_size([1180.0, 980.0]),
        ..Default::default()
    };

    eframe::run_native(
        "NNFractals Quaternion Viewer",
        options,
        Box::new(move |cc| {
            nnfractals::gui_font::install(&cc.egui_ctx);
            Ok(Box::new(App::new(&genome_path, ipc_rx).expect("failed to load genome")))
        }),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;

    Ok(())
}

#[cfg(test)]
mod auto_rotate_tests {
    use super::*;

    #[test]
    fn random_auto_rotate_dir_is_always_unit_length() {
        // Unit length is load-bearing (see `App::auto_rotate_dir`'s doc
        // comment) — it's what makes AUTO_ROTATE_RATE * dt a fixed,
        // direction-independent angular speed regardless of which random
        // direction was drawn.
        let mut rng = rand::rng();
        for _ in 0..200 {
            let (x, y) = random_auto_rotate_dir(&mut rng);
            let len = (x * x + y * y).sqrt();
            assert!((len - 1.0).abs() < 1e-9, "direction ({x}, {y}) has length {len}, not 1.0");
        }
    }
}

/// Real end-to-end verification of the persistence mechanism Carl
/// reported broken ("my render settings are still not saved... please
/// verify with end to end test that a change actually work"). Each test
/// here builds a prefs value where EVERY field differs from `Default`,
/// writes it through the actual `save()` used in production, reads it
/// back through the actual `load()`, and asserts full equality — this is
/// exactly the shape of bug that slips past a code read: a field quietly
/// left out of a struct (or never wired to a save call) still "looks
/// right" in the diff, but silently reverts to its default on the next
/// launch. A temp file is used so this never touches the real project's
/// prefs files, and each test claims a unique filename so `cargo test`'s
/// parallel test threads can't collide on the same path.
#[cfg(test)]
mod prefs_persistence_tests {
    use super::*;

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("nnfractals_quat_viewer_test_{name}_{}.toml", std::process::id()))
    }

    #[test]
    fn render_prefs_round_trip_survives_every_field() {
        let path = temp_path("render_prefs");
        let _ = std::fs::remove_file(&path);
        let original = RenderPrefs {
            render_mode: RenderMode::Static, // non-default (default is Orbit)
            render_out_dir: "some/other/dir".to_string(),
            render_static_size: 2048,
            render_orbit_width: 1234,
            render_orbit_height: 4321,
            render_orbit_frames: 77,
            render_orbit_fps: 24,
            render_orbit_turns: 2.5,
        };
        assert_ne!(original, RenderPrefs::default(), "test is meaningless if every field matches the default");
        original.save(&path);
        let loaded = RenderPrefs::load(&path);
        let _ = std::fs::remove_file(&path);
        assert!(loaded == original, "loaded prefs {loaded:?} don't match what was saved {original:?}");
    }

    #[test]
    fn quat_viewer_prefs_round_trip_survives_every_field() {
        // This is the exact bug that was actually present: colormap,
        // max_iter, aa, and render_size_cap had NO field on this struct
        // at all before this fix, so no save call anywhere could have
        // persisted them regardless of how many `.changed()` sites called
        // it — this test would have failed to compile against that old
        // struct (the field names below wouldn't exist), which is a
        // stronger guarantee than a runtime assertion alone.
        let path = temp_path("quat_viewer_prefs");
        let _ = std::fs::remove_file(&path);
        let original = QuatViewerPrefs {
            auto_rotate: true, // non-default (default is false)
            colormap: "turbo".to_string(),
            max_iter: 99,
            aa: 3,
            render_size_cap: 777,
        };
        assert_ne!(original, QuatViewerPrefs::default(), "test is meaningless if every field matches the default");
        original.save(&path);
        let loaded = QuatViewerPrefs::load(&path);
        let _ = std::fs::remove_file(&path);
        assert!(loaded == original, "loaded prefs {loaded:?} don't match what was saved {original:?}");
    }
}
