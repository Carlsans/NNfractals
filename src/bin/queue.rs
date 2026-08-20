//! NNFractals video-export queue (egui/eframe).
//!
//! Renders queued zoom videos one at a time in the background so the
//! interactive viewer never has to babysit an export. Single-instance,
//! same Unix-socket pattern as `nnfractals-viewer` — launching a second
//! copy just wakes and focuses the running one.
//!
//! Buttons: Process Queue / Stop Queue (graceful — finishes whatever's
//! currently rendering rather than killing an in-flight ffmpeg pipe) /
//! Clean Queue (drops Done/Failed entries only, never the output files).
//! The first item added to an empty queue starts automatically.

use std::io::Read as _;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

use eframe::egui::{self, Color32};

use nnfractals::config::Config;
use nnfractals::io::load_genome;
use nnfractals::video_export::{
    export_video_chain_interpolated, load_queue, queue_dir, save_queue, QueueItem, QueueStatus, VideoMsg,
};

// ── Small utilities (moved from viewer.rs — this window owns the "job
// finished" notifications now, the viewer no longer produces a video
// directly) ──────────────────────────────────────────────────────────────

/// Open a file with the OS's default handler (`xdg-open` on Linux). Spawned
/// detached and best-effort — a click on a status label has nowhere
/// meaningful to surface a launch failure, so errors are silently ignored.
fn open_file(path: &Path) {
    let _ = std::process::Command::new("xdg-open").arg(path).spawn();
}

/// Open the folder CONTAINING `path` — also via `xdg-open`, so it launches
/// whatever the user has set as their default file manager.
fn open_containing_folder(path: &Path) {
    if let Some(dir) = path.parent() {
        let _ = std::process::Command::new("xdg-open").arg(dir).spawn();
    }
}

/// Audible cue on every finished item (success or failure) — unlike the
/// interactive viewer's old 60s-gated sound, a queue's whole point is
/// unattended notification, so this fires every time regardless of how
/// long the render took.
fn play_completion_sound() {
    const FREEDESKTOP_SOUND: &str = "/usr/share/sounds/freedesktop/stereo/complete.oga";
    if Path::new(FREEDESKTOP_SOUND).exists() {
        if std::process::Command::new("paplay").arg(FREEDESKTOP_SOUND).spawn().is_ok() {
            return;
        }
        if std::process::Command::new("pw-play").arg(FREEDESKTOP_SOUND).spawn().is_ok() {
            return;
        }
    }
    let _ = std::process::Command::new("canberra-gtk-play").args(["--id", "complete"]).spawn();
}

// ── Single-instance IPC ──────────────────────────────────────────────────

struct SocketGuard(PathBuf);
impl Drop for SocketGuard {
    fn drop(&mut self) { let _ = std::fs::remove_file(&self.0); }
}

fn socket_path() -> PathBuf {
    let tag = std::env::var("USER")
        .or_else(|_| std::env::var("LOGNAME"))
        .unwrap_or_else(|_| "user".into());
    std::env::temp_dir().join(format!("nnfractals-queue-{tag}.sock"))
}

/// Try to connect to a running queue window. No payload needed (unlike the
/// viewer's delegate, which hands over a path) — connecting at all means
/// "wake up, reload queue.json, focus yourself." Returns true if delegated.
fn try_delegate(sock: &Path) -> bool {
    UnixStream::connect(sock).is_ok()
}

// ── Processing ────────────────────────────────────────────────────────────

/// Render one queued item, returning the output path or an error. Loads its
/// own fresh `Config` (colormap overridden from the item) rather than
/// sharing one with the viewer process — they're separate processes.
/// `progress` is updated live so the UI can show "frame X/Y" for whichever
/// item is currently rendering.
fn process_item(
    item: &QueueItem,
    progress: &Arc<Mutex<Option<(u32, u32)>>>,
    current_pid: &Arc<Mutex<Option<u32>>>,
    ctx: &egui::Context,
) -> Result<String, String> {
    let nn_path = queue_dir().join(&item.nn_filename);
    let genome = load_genome(&nn_path).map_err(|e| format!("failed to load {}: {e}", nn_path.display()))?;
    // Project-root-relative, NOT CWD-relative — see
    // `nnfractals::project_root`'s doc comment. This exact line was the
    // reported crash (Carl, 2026-08-11): "No such file or directory" when
    // this window was spawned by a viewer launched via the file manager,
    // whose working directory wasn't the project root.
    let config_path = nnfractals::project_root().join("config.toml");
    let mut config = Config::load(&config_path)
        .map_err(|e| format!("failed to load {}: {e}", config_path.display()))?;
    config.rendering.colormap = item.colormap.clone();

    let out_dir = PathBuf::from(&item.output_dir);
    std::fs::create_dir_all(&out_dir).map_err(|e| format!("cannot create {}: {e}", out_dir.display()))?;
    let base = format!("{}_zoom_{}s_{}fps", item.genome_label, item.steps, item.fps);
    let mut out_path = out_dir.join(format!("{base}.mp4"));
    let mut n = 2;
    while out_path.exists() {
        out_path = out_dir.join(format!("{base}_{n}.mp4"));
        n += 1;
    }

    let (tx, rx) = mpsc::channel::<VideoMsg>();
    let (g2, c2, start, end) = (genome, config, item.start, item.end);
    let waypoints = item.waypoints.clone();
    let (steps, fps, w, h, invc, invr, ang) = (
        item.steps, item.fps, item.width, item.height,
        item.invert_coords, item.invert_range, item.angle_coloring,
    );
    let kf_stride = item.keyframe_stride;
    let out_path2 = out_path.clone();
    let ctx2 = ctx.clone();
    let render_handle = thread::spawn(move || {
        // A wormhole-chain item carries its own waypoint sequence — render
        // ALL of it as one continuous multi-leg video, not just start→end
        // (which for a chain item are only the first/last waypoint, kept
        // solely for older UI that expects those two fields to exist).
        // A plain start→end item is just the 2-waypoint case of the same
        // thing (`export_video` is literally `export_video_chain(&[start,
        // end])`), so it goes through the SAME renderer rather than a
        // parallel one — otherwise the item's keyframe stride is carried
        // all the way here, badged "⚡kf/N" in the queue window, and then
        // silently ignored for exactly the export people run most.
        let waypoints = if waypoints.len() >= 2 { waypoints } else { vec![start, end] };
        // Honours the item's keyframe stride: 0/1 renders every frame
        // exactly (the pre-existing behaviour, and what older queue
        // items deserialize to), higher values render every Nth and
        // warp the rest.
        export_video_chain_interpolated(&g2, &c2, ang, &waypoints, steps, fps, w, h, invc, invr,
                            &out_path2, &tx, &|| ctx2.request_repaint(), None, kf_stride);
    });

    let mut result: Option<Result<String, String>> = None;
    for msg in rx {
        match msg {
            VideoMsg::Started { pid } => {
                *current_pid.lock().unwrap() = Some(pid);
            }
            VideoMsg::Progress { done, total } => {
                *progress.lock().unwrap() = Some((done, total));
                ctx.request_repaint();
            }
            VideoMsg::Done(p) => result = Some(Ok(p.to_string_lossy().into_owned())),
            VideoMsg::Failed(e) => result = Some(Err(e)),
        }
    }
    let _ = render_handle.join();
    *progress.lock().unwrap() = None;
    result.unwrap_or_else(|| Err("export thread ended without a result".to_string()))
}

/// One-time recovery for a previous ungraceful shutdown: an item still
/// marked `Processing` at startup was never actually finished (only the
/// processing thread itself sets that status, and it always clears it
/// before exiting normally) — reset it so it gets retried.
fn recover_stale_processing() {
    let mut items = load_queue();
    let mut changed = false;
    for it in items.iter_mut() {
        if it.status == QueueStatus::Processing {
            it.status = QueueStatus::Pending;
            changed = true;
        }
    }
    if changed { save_queue(&items); }
}

#[allow(clippy::too_many_arguments)]
fn spawn_processing_thread(
    running: Arc<AtomicBool>,
    progress: Arc<Mutex<Option<(u32, u32)>>>,
    current_id: Arc<Mutex<Option<String>>>,
    current_pid: Arc<Mutex<Option<u32>>>,
    cancelling: Arc<AtomicBool>,
    ctx: egui::Context,
) {
    thread::spawn(move || loop {
        if !running.load(Ordering::SeqCst) {
            thread::sleep(std::time::Duration::from_millis(300));
            continue;
        }
        let mut items = load_queue();
        let next_idx = items.iter()
            .enumerate()
            .filter(|(_, it)| it.status == QueueStatus::Pending)
            .min_by_key(|(_, it)| it.created_at)
            .map(|(i, _)| i);
        let Some(idx) = next_idx else {
            thread::sleep(std::time::Duration::from_millis(500));
            continue;
        };

        items[idx].status = QueueStatus::Processing;
        save_queue(&items);
        *current_id.lock().unwrap() = Some(items[idx].id.clone());
        ctx.request_repaint();

        let item = items[idx].clone();
        let result = process_item(&item, &progress, &current_pid, &ctx);

        // A cancelled render surfaces through the exact same path a real
        // ffmpeg failure would (killing the process breaks its stdin pipe
        // — see `VideoMsg::Started`'s doc comment). Originally this just
        // relabeled the resulting error "Cancelled by user" and left the
        // item sitting in the queue as Failed — but Carl reported (2026-
        // 08-13) that a killed item "still gets put on first position of
        // the queue": true in the sense that it stayed marked Failed
        // right where it was (near the front, since it was the oldest/
        // currently-processing item) instead of actually going away,
        // requiring a separate "Clean Queue" click to get rid of. Cancel
        // should mean "make it disappear," same as "✕ Remove" does for a
        // Pending item — so a cancelled item is now REMOVED outright
        // instead of landing on Failed.
        let was_cancelled = cancelling.swap(false, Ordering::SeqCst);

        let mut items = load_queue();
        if was_cancelled {
            items.retain(|it| it.id != item.id);
        } else if let Some(it) = items.iter_mut().find(|it| it.id == item.id) {
            match result {
                Ok(out) => { it.status = QueueStatus::Done; it.output_path = Some(out); it.error = None; }
                Err(e)  => { it.status = QueueStatus::Failed; it.error = Some(e); }
            }
        }
        save_queue(&items);
        let _ = std::fs::remove_file(queue_dir().join(&item.nn_filename));
        // The partial .mp4 ffmpeg was writing when killed is truncated/
        // invalid — clean it up too rather than leaving a useless file
        // behind, same reconstructed-path logic `process_item` used to
        // create it (numeric suffix only added if the exact base name was
        // already taken AT START, so this matches for the common case of
        // a job that was never retried under the same name).
        if was_cancelled {
            let base = format!("{}_zoom_{}s_{}fps", item.genome_label, item.steps, item.fps);
            let _ = std::fs::remove_file(PathBuf::from(&item.output_dir).join(format!("{base}.mp4")));
        }
        *current_id.lock().unwrap() = None;
        *current_pid.lock().unwrap() = None;
        play_completion_sound();
        ctx.request_repaint();
    });
}

// ── App ───────────────────────────────────────────────────────────────────

/// Per-item text-edit buffers, only meaningful while `status == Pending`.
struct EditBuf {
    steps: String,
    fps: String,
    width: String,
    height: String,
}

impl EditBuf {
    fn from_item(it: &QueueItem) -> Self {
        EditBuf {
            steps: it.steps.to_string(),
            fps: it.fps.to_string(),
            width: it.width.to_string(),
            height: it.height.to_string(),
        }
    }
}

struct App {
    items: Vec<QueueItem>,
    edit_bufs: std::collections::HashMap<String, EditBuf>,
    last_poll: std::time::Instant,
    /// Cores the render pool should use, adjustable while rendering. 0 =
    /// not yet initialised (set to the default on first paint).
    cpu_threads: usize,
    prev_active_count: usize,

    running: Arc<AtomicBool>,
    progress: Arc<Mutex<Option<(u32, u32)>>>,
    current_id: Arc<Mutex<Option<String>>>,
    // ffmpeg child PID for whatever's currently rendering, and whether a
    // cancel is in flight — added after Carl's report, 2026-08-13: "an
    // item being processed is still uncleanable." Previously there was NO
    // way to abort an in-progress render at all — "Stop Queue" only ever
    // stops the loop from picking up the NEXT item. Populated from
    // `VideoMsg::Started` (sent right after `export_video_chain` spawns
    // ffmpeg); the Cancel button just kills this PID directly, same
    // pattern `viewer.rs`'s `cancel_explore_stage` already uses.
    current_pid: Arc<Mutex<Option<u32>>>,
    cancelling: Arc<AtomicBool>,

    wake_rx: mpsc::Receiver<()>,

    /// This executable's path and mtime, both captured AT STARTUP — see
    /// `binary_is_stale`. `None` disables the check.
    exe_stamp: Option<(PathBuf, std::time::SystemTime)>,
    /// Cached so the banner doesn't stat the binary on every repaint.
    stale_checked: std::cell::Cell<Option<(std::time::Instant, bool)>>,
}

impl App {
    /// True when this binary has been rebuilt since the process started.
    ///
    /// The path MUST come from the stamp captured at startup, not from
    /// `current_exe()` here. Cargo replaces the executable rather than
    /// rewriting it, so once a rebuild has happened `/proc/self/exe` reads as
    /// `".../nnfractals-queue (deleted)"` — a path that cannot be stat'd, so
    /// asking the running process where it lives returns exactly nothing in
    /// the one case this check exists to catch. Verified against the real
    /// stale process (pid 555691) before trusting it.
    ///
    /// Rechecked at most once a second: it's a filesystem stat on the UI
    /// thread, and a rebuild does not need frame-rate latency.
    fn binary_is_stale(&self) -> bool {
        const RECHECK: std::time::Duration = std::time::Duration::from_secs(1);
        if let Some((at, verdict)) = self.stale_checked.get()
            && at.elapsed() < RECHECK {
            return verdict;
        }
        let verdict = self.exe_stamp.as_ref().is_some_and(|(path, at_launch)| {
            // A failed stat means the binary was deleted and not replaced —
            // nothing useful to warn about, and not worth a false alarm.
            std::fs::metadata(path).and_then(|m| m.modified())
                .is_ok_and(|now| now != *at_launch)
        });
        self.stale_checked.set(Some((std::time::Instant::now(), verdict)));
        verdict
    }
}

const POLL_INTERVAL: std::time::Duration = std::time::Duration::from_secs(2);

impl App {
    fn new(cc: &eframe::CreationContext, wake_rx: mpsc::Receiver<()>) -> Self {
        recover_stale_processing();
        let items = load_queue();
        let active = items.iter()
            .filter(|it| matches!(it.status, QueueStatus::Pending | QueueStatus::Processing))
            .count();

        let running = Arc::new(AtomicBool::new(active > 0));
        let progress = Arc::new(Mutex::new(None));
        let current_id = Arc::new(Mutex::new(None));
        let current_pid = Arc::new(Mutex::new(None));
        let cancelling = Arc::new(AtomicBool::new(false));
        spawn_processing_thread(
            running.clone(), progress.clone(), current_id.clone(),
            current_pid.clone(), cancelling.clone(), cc.egui_ctx.clone(),
        );

        App {
            items,
            edit_bufs: std::collections::HashMap::new(),
            last_poll: std::time::Instant::now(),
            cpu_threads: 0,
            prev_active_count: active,
            running,
            progress,
            current_id,
            current_pid,
            cancelling,
            wake_rx,
            // Captured now, while `/proc/self/exe` still resolves to a file
            // that exists — see `binary_is_stale`.
            exe_stamp: std::env::current_exe().ok().and_then(|p| {
                std::fs::metadata(&p).and_then(|m| m.modified()).ok().map(|t| (p, t))
            }),
            stale_checked: std::cell::Cell::new(None),
        }
    }

    /// Reload from disk and apply the auto-start rule: if the queue was
    /// empty (0 pending+processing) before this reload and now has at
    /// least one item, start running automatically — this is what makes
    /// "the first item added starts automatically" true regardless of
    /// whether it arrived via a fresh spawn or a wake ping to an
    /// already-open, idle window.
    fn reload(&mut self) {
        self.items = load_queue();
        let active = self.items.iter()
            .filter(|it| matches!(it.status, QueueStatus::Pending | QueueStatus::Processing))
            .count();
        if self.prev_active_count == 0 && active > 0 {
            self.running.store(true, Ordering::SeqCst);
        }
        self.prev_active_count = active;
        // Drop edit buffers for items that no longer exist or left Pending.
        let ids: std::collections::HashSet<&str> = self.items.iter()
            .filter(|it| it.status == QueueStatus::Pending)
            .map(|it| it.id.as_str())
            .collect();
        self.edit_bufs.retain(|id, _| ids.contains(id.as_str()));
    }

    fn save_edit(&mut self, id: &str) {
        let Some(buf) = self.edit_bufs.get(id) else { return };
        if let Some(it) = self.items.iter_mut().find(|it| it.id == id) {
            if let Ok(v) = buf.steps.trim().parse::<u32>() { it.steps = v.max(2); }
            if let Ok(v) = buf.fps.trim().parse::<u32>() { it.fps = v.max(1); }
            if let Ok(v) = buf.width.trim().parse::<u32>() { it.width = v.max(64); }
            if let Ok(v) = buf.height.trim().parse::<u32>() { it.height = v.max(64); }
        }
        save_queue(&self.items);
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        while self.wake_rx.try_recv().is_ok() {
            self.reload();
            ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
        }
        if self.last_poll.elapsed() >= POLL_INTERVAL {
            self.reload();
            self.last_poll = std::time::Instant::now();
        }

        let running = self.running.load(Ordering::SeqCst);
        let current_id = self.current_id.lock().unwrap().clone();
        let progress = *self.progress.lock().unwrap();

        // This window is typically left open for days while renders are
        // queued into it, so a rebuild lands while it is running and it keeps
        // executing the OLD code with nothing to say so. Real cost, 2026-08-20:
        // a colormap fix made the viewer render correctly and this process
        // silently produced a 3200-frame video with the pre-fix (near-black)
        // colouring — hours of render, indistinguishable from a code bug.
        if self.binary_is_stale() {
            egui::Panel::top("stale_binary").show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.colored_label(Color32::from_rgb(255, 170, 60), "⚠ REBUILT SINCE LAUNCH");
                    ui.label("— this window is still running the old binary. Renders started now \
                              will use the OLD code. Close and reopen it to pick up the changes.");
                });
            });
        }

        egui::Panel::top("controls").show(ui, |ui| {
            ui.horizontal(|ui| {
                if ui.add_enabled(!running, egui::Button::new("▶ Process Queue")).clicked() {
                    self.running.store(true, Ordering::SeqCst);
                }
                if ui.add_enabled(running, egui::Button::new("⏸ Stop Queue"))
                    .on_hover_text("Finishes whatever's currently rendering, then stops")
                    .clicked()
                {
                    self.running.store(false, Ordering::SeqCst);
                }
                if ui.button("🗑 Clean Queue")
                    .on_hover_text("Removes finished/failed entries only — never touches pending items or output files")
                    .clicked()
                {
                    self.items.retain(|it| matches!(it.status, QueueStatus::Pending | QueueStatus::Processing));
                    save_queue(&self.items);
                }
                ui.separator();

                // ── Live render controls ──────────────────────────────────
                // Both act on the CURRENT render immediately, unlike
                // "Stop Queue" which only takes effect after the in-flight
                // item finishes. Pause lands on a frame boundary, so the
                // partial video stays valid and can be resumed.
                let paused = nnfractals::video_export::RENDER_CONTROL.is_paused();
                let (label, hover) = if paused {
                    ("▶ Resume", "Resume the paused render")
                } else {
                    ("⏸ Pause", "Pause the render in progress — takes effect at the next frame boundary, and frees the CPU while paused")
                };
                if ui.add_enabled(running, egui::Button::new(label)).on_hover_text(hover).clicked() {
                    nnfractals::video_export::RENDER_CONTROL.set_paused(!paused);
                }

                ui.separator();
                ui.label("CPU:");
                let max_cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
                if self.cpu_threads == 0 { self.cpu_threads = max_cores.div_ceil(2); }
                let mut t = self.cpu_threads;
                if ui.add(egui::Slider::new(&mut t, 1..=max_cores).suffix(" cores"))
                    .on_hover_text("Cores used for rendering. Applies to the NEXT frame — the pool is rebuilt between frames, so you can dial this while a render is running.")
                    .changed()
                {
                    self.cpu_threads = t;
                    nnfractals::video_export::RENDER_CONTROL.set_threads(t);
                }

                ui.separator();
                let status = if paused && running {
                    "PAUSED".to_string()
                } else if running {
                    match &current_id {
                        Some(_) => match progress {
                            Some((d, t)) => format!("Processing — frame {d}/{t}"),
                            None => "Processing…".to_string(),
                        },
                        None => "Running (idle, waiting for items)".to_string(),
                    }
                } else {
                    "Stopped".to_string()
                };
                if paused && running {
                    ui.colored_label(Color32::from_rgb(255, 200, 60), status);
                } else {
                    ui.label(status);
                }
            });
        });

        egui::CentralPanel::default().show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                if self.items.is_empty() {
                    ui.label("Queue is empty — use \"Add to Queue\" in the viewer.");
                }
                let mut to_open: Option<PathBuf> = None;
                let mut to_open_folder: Option<PathBuf> = None;
                let mut save_edit_id: Option<String> = None;
                let mut to_remove_id: Option<String> = None;

                for it in &self.items {
                    ui.group(|ui| {
                        ui.horizontal(|ui| {
                            let (color, label) = match it.status {
                                QueueStatus::Pending    => (Color32::GRAY, "PENDING"),
                                QueueStatus::Processing => (Color32::from_rgb(100, 200, 255), "PROCESSING"),
                                QueueStatus::Done        => (Color32::LIGHT_GREEN, "DONE"),
                                QueueStatus::Failed       => (Color32::LIGHT_RED, "FAILED"),
                            };
                            ui.colored_label(color, label);
                            ui.strong(&it.genome_label);
                            if it.keyframe_stride > 1 {
        ui.colored_label(Color32::from_rgb(120, 200, 255), format!("⚡kf/{}", it.keyframe_stride))
            .on_hover_text("Keyframe-interpolated: only every Nth frame is rendered, the rest are warped from neighbours");
    }
    if it.waypoints.len() >= 2 {
                                let label = it.chain_label.as_deref().unwrap_or("chain");
                                ui.colored_label(Color32::from_rgb(230, 80, 230),
                                    format!("⟳ {label}, {} legs", it.waypoints.len() - 1));
                            }
                            ui.monospace(format!(
                                "({:.4},{:.4})@{:.2e}× → ({:.4},{:.4})@{:.2e}×",
                                it.start.cx, it.start.cy, it.start.zoom,
                                it.end.cx, it.end.cy, it.end.zoom,
                            ));
                        });

                        if it.status == QueueStatus::Pending {
                            let buf = self.edit_bufs.entry(it.id.clone())
                                .or_insert_with(|| EditBuf::from_item(it));
                            ui.horizontal(|ui| {
                                ui.label("Steps:");
                                let r1 = ui.add(egui::TextEdit::singleline(&mut buf.steps).desired_width(45.0));
                                ui.label("FPS:");
                                let r2 = ui.add(egui::TextEdit::singleline(&mut buf.fps).desired_width(35.0));
                                ui.label("Res:");
                                let r3 = ui.add(egui::TextEdit::singleline(&mut buf.width).desired_width(55.0));
                                ui.label("×");
                                let r4 = ui.add(egui::TextEdit::singleline(&mut buf.height).desired_width(55.0));
                                if r1.lost_focus() || r2.lost_focus() || r3.lost_focus() || r4.lost_focus() {
                                    save_edit_id = Some(it.id.clone());
                                }
                                ui.label(format!("(colormap: {}, invert coords={}, range={})",
                                                  it.colormap, it.invert_coords, it.invert_range));
                                // Only Pending is safely cancelable this way — a Processing
                                // item is being rendered by the worker thread from its own
                                // cloned copy, so removing it here wouldn't stop the render,
                                // just discard the record of it (see this button's PR notes /
                                // Carl's report, 2026-08-13: "Clean Queue" only ever drops
                                // Done/Failed, so there was previously NO way to cancel an
                                // unwanted queued item at all short of letting it fully render).
                                if ui.button("✕ Remove")
                                    .on_hover_text("Cancels this queued job — removes it and its copied genome file. Only available while Pending.")
                                    .clicked()
                                {
                                    to_remove_id = Some(it.id.clone());
                                }
                            });
                        } else if it.status == QueueStatus::Done {
                            if let Some(out) = &it.output_path {
                                let path = PathBuf::from(out);
                                let resp = ui.link(out)
                                    .on_hover_text("Click to open the video · right-click to open its folder");
                                if resp.clicked() { to_open = Some(path.clone()); }
                                if resp.secondary_clicked() { to_open_folder = Some(path); }
                            }
                        } else if it.status == QueueStatus::Failed {
                            if let Some(e) = &it.error {
                                ui.colored_label(Color32::LIGHT_RED, e);
                            }
                        } else if it.status == QueueStatus::Processing {
                            ui.horizontal(|ui| {
                                ui.label(format!("{} steps @ {} fps, {}×{}", it.steps, it.fps, it.width, it.height));
                                // `current_pid` is only `Some` once ffmpeg has
                                // actually spawned (see `VideoMsg::Started`) —
                                // there's a brief window right at the start of
                                // a render where the button isn't up yet, same
                                // "can't cancel what hasn't started" gap
                                // `cancel_explore_stage` in viewer.rs also has.
                                let pid = *self.current_pid.lock().unwrap();
                                if let Some(pid) = pid {
                                    let cancelling = self.cancelling.load(Ordering::SeqCst);
                                    if ui.add_enabled(!cancelling, egui::Button::new("✕ Cancel"))
                                        .on_hover_text(
                                            "Kills the in-progress ffmpeg render (`kill <pid>`) and removes \
                                             this item from the queue entirely, same as \"✕ Remove\" for a \
                                             Pending item. Previously there was no way to abort an \
                                             in-progress render at all."
                                        )
                                        .clicked()
                                    {
                                        self.cancelling.store(true, Ordering::SeqCst);
                                        let _ = std::process::Command::new("kill").arg(pid.to_string()).status();
                                    }
                                    if cancelling { ui.colored_label(Color32::YELLOW, "cancelling…"); }
                                }
                            });
                        } else {
                            ui.label(format!("{} steps @ {} fps, {}×{}", it.steps, it.fps, it.width, it.height));
                        }
                    });
                }

                if let Some(id) = save_edit_id { self.save_edit(&id); }
                if let Some(p) = to_open { open_file(&p); }
                if let Some(p) = to_open_folder { open_containing_folder(&p); }
                if let Some(id) = to_remove_id {
                    if let Some(it) = self.items.iter().find(|it| it.id == id) {
                        let _ = std::fs::remove_file(queue_dir().join(&it.nn_filename));
                    }
                    self.items.retain(|it| it.id != id);
                    self.edit_bufs.remove(&id);
                    save_queue(&self.items);
                }
            });
        });

        // Keep repainting while something is actively happening so the
        // progress readout and status stay live.
        if running { ctx.request_repaint_after(std::time::Duration::from_millis(200)); }
    }
}

fn main() -> anyhow::Result<()> {
    let sock_path = socket_path();
    if try_delegate(&sock_path) {
        eprintln!("[queue] Delegated to running instance.");
        return Ok(());
    }
    let _ = std::fs::remove_file(&sock_path);
    let (wake_tx, wake_rx) = mpsc::channel::<()>();
    let _sock_guard = match UnixListener::bind(&sock_path) {
        Ok(listener) => {
            thread::spawn(move || {
                for stream in listener.incoming().flatten() {
                    let mut s: UnixStream = stream;
                    let mut buf = [0u8; 1];
                    let _ = s.read(&mut buf); // don't care about content, just that a connection happened
                    let _ = wake_tx.send(());
                }
            });
            Some(SocketGuard(sock_path))
        }
        Err(e) => { eprintln!("[queue] IPC unavailable: {e}"); None }
    };

    #[cfg(feature = "wgpu-backend")]
    {
        nnfractals::render_gpu::init_gpu();
        eprintln!(
            "[queue] Renderer: {}",
            if nnfractals::render_gpu::gpu_available() { "GPU (wgpu)" } else { "CPU (rayon fallback)" }
        );
    }

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([720.0, 480.0]),
        ..Default::default()
    };
    eframe::run_native(
        "NNFractals Video Queue",
        options,
        Box::new(move |cc| {
            nnfractals::gui_font::install(&cc.egui_ctx);
            Ok(Box::new(App::new(cc, wake_rx)))
        }),
    ).map_err(|e| anyhow::anyhow!("{e}"))?;

    Ok(())
}
