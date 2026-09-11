//! NNFractals reel review (egui/eframe) — stage 2 of automated video
//! generation.
//!
//! `explorer auto-reel` runs unattended for hours and leaves a batch of
//! low-resolution previews behind. This window is where a human watches them
//! and makes the one decision the pipeline cannot: is this worth rendering
//! properly. Approving a reel writes a full-resolution job into the existing
//! video queue, which renders overnight.
//!
//! Playback is frames, not video: egui has no decoder, so a selected reel is
//! decoded once by ffmpeg (already a hard dependency of this project) straight
//! into memory as raw RGB, and the frames are flipped as textures. Decoding
//! happens on a worker thread so selecting a reel never blocks the UI.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};

use eframe::egui::{self, Color32};

use nnfractals::auto_reel::{self, ReelRecord, ReelStatus};
use nnfractals::video_export::{enqueue, QueueSpec};

/// Decoded preview frames for one reel.
struct Clip {
    id: String,
    frames: Vec<egui::ColorImage>,
}

enum Msg {
    Clip(Box<Clip>),
    Failed(String, String),
    /// A re-roll subprocess finished: reel id, and what to say about it.
    Redone(String, Result<(), String>),
    /// The batch subprocess finished.
    BatchDone(Result<(), String>),
}

/// Directories a batch can draw from: every top-level `fractals*` pool plus the
/// curated `Starred` folder, the same set the browser offers.
fn discover_pools() -> Vec<String> {
    let root = nnfractals::project_root();
    let mut out: Vec<String> = std::fs::read_dir(&root)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .filter_map(|p| p.file_name().and_then(|n| n.to_str()).map(str::to_owned))
        .filter(|n| n.starts_with("fractals"))
        .collect();
    out.sort();
    // First when it exists — it is the curated folder the browser stars into,
    // and the obvious place to point a first batch. Not offered when absent:
    // a pool with no .nn files just makes the run exit with an error.
    if root.join("Starred").is_dir() {
        out.insert(0, "Starred".into());
    }
    out
}

/// Width of the reel list down the left side.
const LIST_WIDTH: f32 = 240.0;

/// Fraction of the window width the player aims to fill.
pub const PLAYER_WIDTH_FRACTION: f32 = 0.66;

/// Height kept below the player for the transport row and the first line of the
/// recipe.
///
/// Deliberately small — the detail pane scrolls, so the Approve/Reject buttons
/// stay reachable either way, and reserving room for the whole recipe would cap
/// a square 512x512 preview at well under half the window. This is the "you can
/// always see that there IS more below" allowance, not room for all of it.
const DETAIL_HEIGHT: f32 = 132.0;

/// How large to draw the preview, given the whole panel and the clip's aspect.
///
/// Sized from the PANEL rather than from `available_size()` at the point of
/// drawing: inside the nested horizontal/vertical the latter reports what is
/// left of the current row, which made the player a fraction of the window.
///
/// Aims for [`PLAYER_WIDTH_FRACTION`] of the window width and shrinks only when
/// the height cannot take it — so the clip is as large as the window allows
/// without pushing the recipe and the Approve/Reject buttons out of reach.
pub fn player_size(panel: egui::Vec2, aspect: f32) -> (f32, f32) {
    let aspect = if aspect.is_finite() && aspect > 0.0 { aspect } else { 1.0 };
    let right = (panel.x - LIST_WIDTH - 16.0).max(120.0);
    let mut w = (panel.x * PLAYER_WIDTH_FRACTION).min(right).max(120.0);
    let mut h = w / aspect;
    let max_h = (panel.y - DETAIL_HEIGHT).max(120.0);
    if h > max_h {
        h = max_h;
        w = h * aspect;
    }
    (w, h)
}

/// Lines kept from a running batch. Enough to see what the last few reels did
/// without holding an unbounded log in memory for a run that lasts hours.
const LOG_LINES: usize = 400;

/// Frames held in memory for one preview.
///
/// A reel preview is deliberately short — 12 seconds at 5fps is 60 frames, about
/// 60MB as RGBA textures. The cap only matters if this window is ever pointed at
/// a full-length render, where 60s at 30fps would be nearly two gigabytes.
const MAX_PREVIEW_FRAMES: usize = 600;

/// Explode `mp4` into RGB frames via ffmpeg.
///
/// Reads raw `rgb24` off ffmpeg's stdout rather than writing PNGs to disk and
/// reading them back: it is one process instead of one-plus-N file operations,
/// and a preview is only a few hundred small frames.
fn decode(mp4: &Path, w: u32, h: u32) -> Result<Vec<egui::ColorImage>, String> {
    use std::io::Read;
    use std::process::{Command, Stdio};

    let mut child = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(mp4)
        .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("cannot run ffmpeg: {e}"))?;

    let mut buf = Vec::new();
    child.stdout.take().ok_or("no ffmpeg stdout")?
        .read_to_end(&mut buf)
        .map_err(|e| format!("reading frames: {e}"))?;
    let status = child.wait().map_err(|e| format!("ffmpeg: {e}"))?;
    if !status.success() && buf.is_empty() {
        return Err("ffmpeg produced no frames".into());
    }

    let stride = (w * h * 3) as usize;
    if stride == 0 || buf.len() < stride {
        return Err(format!("expected {stride} bytes per frame, got {} total", buf.len()));
    }
    Ok(buf.chunks_exact(stride)
        .take(MAX_PREVIEW_FRAMES)
        .map(|f| egui::ColorImage::from_rgb([w as usize, h as usize], f))
        .collect())
}

struct App {
    batches: Vec<PathBuf>,
    batch: usize,
    reels: Vec<ReelRecord>,
    selected: Option<usize>,
    /// Decoded frames for the selected reel, and the texture currently shown.
    clip: Option<Clip>,
    tex: Option<egui::TextureHandle>,
    frame_idx: usize,
    playing: bool,
    last_advance: std::time::Instant,
    loading: Option<String>,
    /// Id of the reel currently being re-rolled by a subprocess, if any.
    redoing: Option<String>,
    rx: mpsc::Receiver<Msg>,
    tx: mpsc::Sender<Msg>,
    message: String,
    /// Full-resolution settings applied when a reel is approved.
    out_w: String,
    out_h: String,
    out_fps: String,
    out_seconds: String,
    out_rife: String,
    show_rejected: bool,

    // ── Starting a batch ──────────────────────────────────────────────────
    pools: Vec<String>,
    pool: usize,
    batch_count: String,
    batch_seconds: String,
    batch_pop: String,
    batch_gens: String,
    batch_fit_time: bool,
    /// PID of the running batch, so it can be stopped. `None` = not running.
    batch_pid: Arc<Mutex<Option<u32>>>,
    /// Set when the user asks to stop, so the finish message can say so rather
    /// than reporting the kill as a failure.
    batch_stopping: Arc<AtomicBool>,
    batch_running: bool,
    batch_log: Arc<Mutex<Vec<String>>>,
    show_log: bool,
    last_batch_poll: std::time::Instant,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let _ = cc;
        let (tx, rx) = mpsc::channel();
        let batches = auto_reel::list_batches();
        let mut app = App {
            batches, batch: 0, reels: Vec::new(), selected: None,
            clip: None, tex: None, frame_idx: 0, playing: true,
            last_advance: std::time::Instant::now(), loading: None, redoing: None,
            rx, tx, message: String::new(),
            out_w: "1920".into(), out_h: "1080".into(), out_fps: "30".into(),
            out_seconds: "45".into(), out_rife: String::new(),
            show_rejected: false,
            pools: discover_pools(),
            pool: 0,
            batch_count: "5".into(),
            batch_seconds: "12".into(),
            batch_pop: "24".into(),
            batch_gens: "8".into(),
            batch_fit_time: false,
            batch_pid: Arc::new(Mutex::new(None)),
            batch_stopping: Arc::new(AtomicBool::new(false)),
            batch_running: false,
            batch_log: Arc::new(Mutex::new(Vec::new())),
            // Off until a batch starts (`start_batch` turns it on): the log
            // panel costs ~110px of height, which is height the player wants.
            show_log: false,
            last_batch_poll: std::time::Instant::now(),
        };
        app.reload();
        app
    }

    fn batch_dir(&self) -> Option<&PathBuf> {
        self.batches.get(self.batch)
    }

    /// Re-read the batch list and the current batch from disk.
    ///
    /// Keeps the selection AND the decoded clip when the selected reel is still
    /// there. A batch writes reels one at a time, so this runs every few seconds
    /// while one is going — dropping the clip each time would make a preview
    /// unwatchable exactly while you are trying to watch it.
    fn reload(&mut self) {
        let was = self.selected.and_then(|i| self.reels.get(i).map(|r| r.id.clone()));
        self.batches = auto_reel::list_batches();
        self.batch = self.batch.min(self.batches.len().saturating_sub(1));
        self.reels = match self.batch_dir() {
            Some(d) => auto_reel::load_batch(d),
            None => Vec::new(),
        };
        self.selected = was.and_then(|id| self.reels.iter().position(|r| r.id == id));
        if self.selected.is_none() {
            self.clip = None;
            self.tex = None;
        }
    }

    fn visible(&self) -> Vec<usize> {
        (0..self.reels.len())
            .filter(|&i| self.show_rejected || self.reels[i].status != ReelStatus::Rejected)
            .collect()
    }

    /// Start decoding the selected reel's preview on a worker thread.
    fn select(&mut self, i: usize) {
        self.selected = Some(i);
        self.clip = None;
        self.tex = None;
        self.frame_idx = 0;
        let Some(rec) = self.reels.get(i) else { return };
        let Some(dir) = self.batch_dir() else { return };
        let mp4 = dir.join(&rec.preview_file);
        let (id, w, h) = (rec.id.clone(), rec.preview_w, rec.preview_h);
        self.loading = Some(id.clone());
        let tx = self.tx.clone();
        std::thread::spawn(move || {
            let msg = match decode(&mp4, w, h) {
                Ok(frames) if !frames.is_empty() => Msg::Clip(Box::new(Clip { id, frames })),
                Ok(_) => Msg::Failed(id, "the preview decoded to zero frames".into()),
                Err(e) => Msg::Failed(id, e),
            };
            let _ = tx.send(msg);
        });
    }

    fn set_status(&mut self, i: usize, status: ReelStatus) {
        let Some(dir) = self.batch_dir().cloned() else { return };
        if let Some(rec) = self.reels.get_mut(i) {
            rec.status = status;
            if let Err(e) = auto_reel::save_record(&dir, rec) {
                self.message = format!("cannot save decision: {e}");
            }
        }
    }

    /// Start an `explorer auto-reel` batch in a subprocess.
    ///
    /// A subprocess for the same reason the re-roll is one: a batch renders for
    /// hours, and this window has to stay responsive enough to watch the reels
    /// it is producing as they land. Both pipes are drained on their own
    /// threads — a child that fills one and blocks would stall the whole run.
    fn start_batch(&mut self) {
        use std::io::{BufRead, BufReader};
        use std::process::{Command, Stdio};

        if self.batch_running {
            self.message = "a batch is already running".into();
            return;
        }
        let pool = self.pools.get(self.pool).cloned().unwrap_or_else(|| "fractals_1".into());
        let mut args: Vec<String> = vec![
            "auto-reel".into(),
            "--pool".into(), pool.clone(),
            "--count".into(), self.batch_count.trim().to_string(),
            "--seconds".into(), self.batch_seconds.trim().to_string(),
            "--pop".into(), self.batch_pop.trim().to_string(),
            "--gens".into(), self.batch_gens.trim().to_string(),
        ];
        if self.batch_fit_time {
            args.push("--fit-time".into());
        }

        let child = Command::new(nnfractals::locate_bin("explorer"))
            .current_dir(nnfractals::project_root())
            .args(&args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn();
        let mut child = match child {
            Ok(c) => c,
            Err(e) => {
                self.message = format!("cannot run explorer: {e}");
                return;
            }
        };

        self.batch_log.lock().unwrap().clear();
        *self.batch_pid.lock().unwrap() = Some(child.id());
        self.batch_stopping.store(false, Ordering::SeqCst);
        self.batch_running = true;
        self.show_log = true;
        self.message = format!("batch started on {pool}");

        let log = self.batch_log.clone();
        let pid = self.batch_pid.clone();
        let stopping = self.batch_stopping.clone();
        let tx = self.tx.clone();

        // stderr carries the `[gpu]` banner and any panic; keep it in the same
        // log so a failure is visible here rather than only in a terminal
        // nobody is watching.
        if let Some(err) = child.stderr.take() {
            let log = log.clone();
            std::thread::spawn(move || {
                for line in BufReader::new(err).lines().map_while(Result::ok) {
                    let mut l = log.lock().unwrap();
                    l.push(line);
                    let n = l.len();
                    if n > LOG_LINES { l.drain(..n - LOG_LINES); }
                }
            });
        }
        std::thread::spawn(move || {
            if let Some(out) = child.stdout.take() {
                for line in BufReader::new(out).lines().map_while(Result::ok) {
                    let mut l = log.lock().unwrap();
                    l.push(line);
                    let n = l.len();
                    if n > LOG_LINES { l.drain(..n - LOG_LINES); }
                }
            }
            let status = child.wait();
            *pid.lock().unwrap() = None;
            let result = match status {
                _ if stopping.load(Ordering::SeqCst) => Err("stopped".to_string()),
                Ok(s) if s.success() => Ok(()),
                Ok(s) => Err(format!("explorer exited with {s}")),
                Err(e) => Err(e.to_string()),
            };
            let _ = tx.send(Msg::BatchDone(result));
        });
    }

    /// Ask the running batch to stop.
    ///
    /// SIGTERM rather than SIGKILL: a reel's record and clip are written before
    /// the next one starts, so a clean exit keeps everything already finished.
    fn stop_batch(&mut self) {
        let Some(pid) = *self.batch_pid.lock().unwrap() else { return };
        self.batch_stopping.store(true, Ordering::SeqCst);
        let _ = std::process::Command::new("kill").arg(pid.to_string()).status();
        self.message = "stopping after the current reel…".into();
    }

    /// Re-run only the time-formula search for one reel, in a subprocess.
    ///
    /// A subprocess rather than a thread because re-rolling re-renders the
    /// preview, and this window must stay responsive — `explorer auto-reel
    /// --redo` already does exactly this work for the batch.
    fn redo(&mut self, i: usize) {
        if self.redoing.is_some() {
            self.message = "a re-roll is already running".into();
            return;
        }
        let Some(dir) = self.batch_dir().cloned() else { return };
        let Some(rec) = self.reels.get(i).cloned() else { return };
        let json = dir.join(format!("{}.json", rec.id));
        let exe = nnfractals::locate_bin("explorer");
        let cwd = nnfractals::project_root();
        let id = rec.id.clone();
        let tx = self.tx.clone();
        self.redoing = Some(id.clone());
        self.message = format!("re-rolling {}… (renders at several depths)", rec.label);
        std::thread::spawn(move || {
            let out = std::process::Command::new(exe)
                .current_dir(cwd)
                .arg("auto-reel")
                .arg("--redo")
                .arg(&json)
                .output();
            let result = match out {
                Ok(o) if o.status.success() => Ok(()),
                Ok(o) => Err(String::from_utf8_lossy(&o.stderr).lines().last()
                    .unwrap_or("re-roll failed").to_string()),
                Err(e) => Err(format!("cannot run explorer: {e}")),
            };
            let _ = tx.send(Msg::Redone(id, result));
        });
    }

    /// Queue one reel for a full-resolution render.
    fn approve(&mut self, i: usize) {
        let Some(dir) = self.batch_dir().cloned() else { return };
        let Some(rec) = self.reels.get(i).cloned() else { return };

        let w: u32 = self.out_w.trim().parse().unwrap_or(1920).max(64);
        let h: u32 = self.out_h.trim().parse().unwrap_or(1080).max(64);
        let fps: u32 = self.out_fps.trim().parse().unwrap_or(30).max(1);
        let seconds: f32 = self.out_seconds.trim().parse::<f32>().unwrap_or(45.0).max(1.0);
        let rife: u32 = self.out_rife.trim().parse().unwrap_or(0);
        let steps = ((seconds * fps as f32).round() as u32).max(2);

        let spec = QueueSpec {
            nn_src: dir.join(&rec.nn_file),
            genome_label: rec.label.clone(),
            waypoints: vec![rec.start, rec.end],
            chain_label: Some("auto-reel".into()),
            steps,
            fps,
            width: w,
            height: h,
            output_dir: nnfractals::project_root().join("viewer_output/reels")
                .to_string_lossy().into_owned(),
            // A time formula lives on the genome, so it travels in the copied
            // `.nn` — but the queue item must ALSO carry it, because that is
            // what routes the job away from keyframe warping.
            time_prog: rec.time_prog.clone(),
            keyframe_stride: 1,
            rife_fps: rife,
            ..Default::default()
        };
        match enqueue(spec) {
            Ok(_) => {
                self.set_status(i, ReelStatus::Approved);
                self.message = format!(
                    "{} queued at {w}x{h} @{fps}fps, {steps} frames", rec.label);
            }
            Err(e) => self.message = format!("cannot queue {}: {e}", rec.label),
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();

        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                Msg::Clip(c) => {
                    if self.loading.as_deref() == Some(c.id.as_str()) {
                        self.loading = None;
                        self.clip = Some(*c);
                        self.frame_idx = 0;
                    }
                }
                Msg::Failed(id, why) => {
                    if self.loading.as_deref() == Some(id.as_str()) {
                        self.loading = None;
                        self.message = format!("cannot play preview: {why}");
                    }
                }
                Msg::BatchDone(result) => {
                    self.batch_running = false;
                    self.reload();
                    self.message = match result {
                        Ok(()) => "batch finished ✓".into(),
                        Err(why) if why == "stopped" => "batch stopped — finished reels kept".into(),
                        Err(why) => format!("cannot finish batch: {why}"),
                    };
                }
                Msg::Redone(id, result) => {
                    self.redoing = None;
                    match result {
                        Ok(()) => {
                            self.message = "re-rolled ✓".into();
                            let keep = self.selected;
                            self.reload();
                            // Re-select the same reel and re-decode: the clip on
                            // disk is a different one now.
                            if let Some(i) = self.reels.iter().position(|r| r.id == id) {
                                self.select(i);
                            } else {
                                self.selected = keep;
                            }
                        }
                        Err(why) => self.message = format!("cannot re-roll: {why}"),
                    }
                }
            }
        }

        let mut do_select: Option<usize> = None;
        let mut do_approve: Option<usize> = None;
        let mut do_reject: Option<usize> = None;
        let mut do_reset: Option<usize> = None;
        let mut do_redo: Option<usize> = None;
        let mut do_reload = false;
        let mut do_start_batch = false;
        let mut do_stop_batch = false;

        egui::Panel::top("bar").show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.heading("Reels");
                let label = self.batch_dir()
                    .and_then(|d| d.file_name().and_then(|s| s.to_str()))
                    .unwrap_or("no batches")
                    .to_string();
                egui::ComboBox::from_id_salt("batch")
                    .selected_text(label)
                    .show_ui(ui, |ui| {
                        for (i, b) in self.batches.clone().iter().enumerate() {
                            let name = b.file_name().and_then(|s| s.to_str()).unwrap_or("?");
                            if ui.selectable_label(self.batch == i, name).clicked() {
                                self.batch = i;
                                // A different batch has different reels, so the
                                // current selection and clip are meaningless.
                                self.selected = None;
                                self.clip = None;
                                self.tex = None;
                                do_reload = true;
                            }
                        }
                    });
                if ui.button("↻").on_hover_text("Re-read the batch from disk — a run in \
                                                 progress writes reels as it finishes them")
                    .clicked()
                { do_reload = true; }
                ui.checkbox(&mut self.show_rejected, "show rejected");
                ui.separator();
                ui.label("Full render:");
                ui.add(egui::TextEdit::singleline(&mut self.out_w).desired_width(50.0))
                    .on_hover_text("Width of the approved render");
                ui.label("×");
                ui.add(egui::TextEdit::singleline(&mut self.out_h).desired_width(50.0));
                ui.label("fps");
                ui.add(egui::TextEdit::singleline(&mut self.out_fps).desired_width(35.0));
                ui.label("secs");
                ui.add(egui::TextEdit::singleline(&mut self.out_seconds).desired_width(35.0))
                    .on_hover_text("Length of the approved render. The preview is short so a \
                                    batch stays cheap; the real one can breathe.");
                ui.label("rife");
                ui.add(egui::TextEdit::singleline(&mut self.out_rife).desired_width(35.0))
                    .on_hover_text("Optional RIFE interpolation target frame rate. Blank = off.");
            });

            // ── Start a batch ─────────────────────────────────────────────
            ui.horizontal_wrapped(|ui| {
                ui.label("New batch:");
                egui::ComboBox::from_id_salt("pool")
                    .selected_text(self.pools.get(self.pool).cloned().unwrap_or_default())
                    .width(130.0)
                    .show_ui(ui, |ui| {
                        for (i, p) in self.pools.clone().iter().enumerate() {
                            if ui.selectable_label(self.pool == i, p).clicked() {
                                self.pool = i;
                            }
                        }
                    });
                ui.label("count");
                ui.add(egui::TextEdit::singleline(&mut self.batch_count).desired_width(35.0))
                    .on_hover_text("How many fractals to plan. They are taken from the top of the \
                                    pool by aesthetic/pref/novelty, skipping anything an earlier \
                                    batch already used.");
                ui.label("secs");
                ui.add(egui::TextEdit::singleline(&mut self.batch_seconds).desired_width(30.0))
                    .on_hover_text("Length of each PREVIEW clip, at 512x512 and 5fps. Short keeps \
                                    a batch cheap; the approved render can be any length.");
                ui.label("pop");
                ui.add(egui::TextEdit::singleline(&mut self.batch_pop).desired_width(30.0))
                    .on_hover_text("Time-formula search population. Smaller is much faster and \
                                    finds less.");
                ui.label("gens");
                ui.add(egui::TextEdit::singleline(&mut self.batch_gens).desired_width(30.0));
                ui.checkbox(&mut self.batch_fit_time, "⏱ fit time")
                    .on_hover_text("Pull the end zoom back so the formula animates across the \
                                    WHOLE shot. Past zoom ~2e5 a genome scalar is an f32 with no \
                                    step small enough to move smoothly, so without this a deep \
                                    reel animates for its first stretch and is a plain zoom after. \
                                    Costs depth: a 41-doubling shot becomes about 20.");

                if !self.batch_running {
                    if ui.button("▶  Start batch")
                        .on_hover_text("Runs `explorer auto-reel` in the background. Reels appear \
                                        in the list as they finish, so you can start watching \
                                        before the batch is done. Budget about a minute each.")
                        .clicked()
                    { do_start_batch = true; }
                } else {
                    if ui.button("■  Stop").on_hover_text(
                        "Stops after the reel in progress. Everything already finished is kept.")
                        .clicked()
                    { do_stop_batch = true; }
                    ui.spinner();
                    let last = self.batch_log.lock().unwrap().last().cloned().unwrap_or_default();
                    ui.label(egui::RichText::new(last).monospace().small());
                }
                ui.checkbox(&mut self.show_log, "log");
            });

            if self.show_log && (self.batch_running || !self.batch_log.lock().unwrap().is_empty()) {
                let lines = self.batch_log.lock().unwrap().clone();
                egui::ScrollArea::vertical()
                    .max_height(110.0)
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        for l in &lines {
                            ui.label(egui::RichText::new(l).monospace().small());
                        }
                    });
            }
        });

        // This build's eframe hands `ui`, not a `Context`, and exposes only
        // `Panel::top` / `CentralPanel` — no `SidePanel`. A two-column split
        // inside the central panel gives the same layout.
        egui::CentralPanel::default().show(ui, |ui| {
        // Measured ONCE, on the panel, before any nesting. `available_size()`
        // inside the horizontal/vertical below reports what is left of the
        // current row, not the window — which is what made the player come out
        // a fraction of its intended size.
        let panel = ui.available_size();
        ui.horizontal(|ui| {
        ui.allocate_ui_with_layout(
            egui::Vec2::new(LIST_WIDTH, panel.y),
            egui::Layout::top_down(egui::Align::Min),
            |ui| {
            let visible = self.visible();
            let pending = self.reels.iter().filter(|r| r.status == ReelStatus::Pending).count();
            ui.label(egui::RichText::new(format!("{pending} pending of {}", self.reels.len()))
                .color(Color32::GRAY).small());
            egui::ScrollArea::vertical().show(ui, |ui| {
                for i in visible {
                    let r = &self.reels[i];
                    let (mark, col) = match r.status {
                        ReelStatus::Pending => ("•", Color32::GRAY),
                        ReelStatus::Approved => ("✓", Color32::from_rgb(120, 255, 180)),
                        ReelStatus::Rejected => ("✗", Color32::from_rgb(230, 120, 110)),
                    };
                    ui.horizontal(|ui| {
                        ui.colored_label(col, mark);
                        let text = format!("{}  {:.0}d", r.label, r.doublings());
                        if ui.selectable_label(self.selected == Some(i), text).clicked() {
                            do_select = Some(i);
                        }
                    });
                }
            });
        });
        ui.separator();
        // Scrollable: at a large player the recipe and the Approve/Reject
        // buttons sit below the fold, and content taller than the window with
        // no way to reach it is exactly the bug that hid the launcher's Stop
        // button.
        egui::ScrollArea::vertical().id_salt("detail").show(ui, |ui| {
            let Some(i) = self.selected else {
                ui.centered_and_justified(|ui| {
                    ui.label(egui::RichText::new(
                        "Pick a reel on the left.\n\nEach one is a shot the pipeline chose by \
                         itself: where the fractal ends, how deep it can go before f64 runs out, \
                         and a time formula evolved against that zoom."
                    ).color(Color32::GRAY));
                });
                return;
            };
            let Some(rec) = self.reels.get(i).cloned() else { return };

            // ── Player ────────────────────────────────────────────────────
            if let Some(clip) = &self.clip {
                if self.playing && self.last_advance.elapsed()
                    >= std::time::Duration::from_secs_f32(1.0 / rec.preview_fps.max(1) as f32)
                {
                    self.frame_idx = (self.frame_idx + 1) % clip.frames.len();
                    self.last_advance = std::time::Instant::now();
                    self.tex = None;
                }
                if self.tex.is_none() {
                    let img = clip.frames[self.frame_idx.min(clip.frames.len() - 1)].clone();
                    self.tex = Some(ctx.load_texture("reel", img, egui::TextureOptions::LINEAR));
                }
                if let Some(tex) = &self.tex {
                    let (w, h) = player_size(
                        panel, rec.preview_w as f32 / rec.preview_h.max(1) as f32);
                    ui.vertical_centered(|ui| {
                        ui.add(egui::Image::new(egui::load::SizedTexture::new(
                            tex.id(), egui::Vec2::new(w, h))));
                    });
                }
                ui.horizontal(|ui| {
                    if ui.button(if self.playing { "⏸" } else { "▶" }).clicked() {
                        self.playing = !self.playing;
                    }
                    let n = clip.frames.len();
                    let mut idx = self.frame_idx;
                    if ui.add(egui::Slider::new(&mut idx, 0..=n.saturating_sub(1)).text("frame"))
                        .changed()
                    {
                        self.frame_idx = idx;
                        self.playing = false;
                        self.tex = None;
                    }
                });
                ctx.request_repaint_after(std::time::Duration::from_millis(40));
            } else {
                // A fixed box: inside a scroll area `centered_and_justified`
                // has no height to centre within and would stretch.
                let (w, h) = player_size(panel, 1.0);
                let _ = w;
                ui.allocate_ui(egui::Vec2::new(ui.available_width(), h), |ui| {
                    ui.centered_and_justified(|ui| {
                        if self.loading.is_some() {
                            ui.spinner();
                        } else {
                            ui.label(egui::RichText::new("no preview loaded").color(Color32::GRAY));
                        }
                    });
                });
                if self.loading.is_some() {
                    ctx.request_repaint_after(std::time::Duration::from_millis(100));
                }
            }

            ui.separator();

            // ── The recipe, and the measurements behind it ───────────────
            ui.horizontal_wrapped(|ui| {
                ui.monospace(format!("{:.3e}x → {:.3e}x", rec.start.zoom, rec.end.zoom));
                ui.label(format!("({:.1} doublings)", rec.doublings()));
                ui.separator();
                ui.label(format!("{:.1} short of the f64 wall", rec.destination.doublings_short()))
                    .on_hover_text("A shot that stops far short ran out of STRUCTURE; one that \
                                    stops just short ran out of PRECISION. Opposite fixes.");
                if rec.destination.trimmed_to.is_some() {
                    ui.colored_label(Color32::from_rgb(230, 200, 120), "trimmed");
                }
            });
            ui.horizontal_wrapped(|ui| {
                ui.label(format!("frame: body {:.1}% at scan {:.3}x",
                                 rec.frame.body_fraction * 100.0, rec.frame.scan_zoom));
                ui.separator();
                ui.label(format!("line: {}/{} rich, worst {:.3}",
                                 rec.destination.passed, rec.destination.checked,
                                 rec.destination.worst_richness));
                if rec.destination.sparse_opening > 0 {
                    ui.label(egui::RichText::new(
                        format!("{} sparse opening frames", rec.destination.sparse_opening))
                        .color(Color32::GRAY).small())
                      .on_hover_text("Expected: the establishing shot is a wide view of the whole \
                                      fractal, so it is mostly background.");
                }
            });
            if rec.animated_fraction < 0.999 {
                ui.horizontal_wrapped(|ui| {
                    ui.colored_label(Color32::from_rgb(150, 170, 255),
                        format!("⏱ animatable over the first {:.0}% of the shot (to {:.2e}x)",
                                rec.animated_fraction * 100.0, rec.animatable_to));
                }).response.on_hover_text(
                    "Every animatable scalar on a genome is an f32, so there are only so many \
                     representable values inside one frame at a given zoom. A clip needs one per \
                     frame to move smoothly, and past this depth there are none left — the camera \
                     keeps descending and the formula holds still. A precision limit, not a defect.");
            }
            if rec.time_prog.is_empty() {
                ui.colored_label(Color32::GRAY, "⏱ plain zoom — no time formula survived");
            } else {
                ui.horizontal_wrapped(|ui| {
                    ui.label(format!("⏱ score {:.4}", rec.time_score));
                    if rec.time_loops {
                        ui.colored_label(Color32::from_rgb(120, 255, 180), "loops");
                    } else {
                        ui.colored_label(Color32::from_rgb(230, 200, 120), "one-shot");
                    }
                });
                for tp in &rec.time_prog {
                    ui.label(egui::RichText::new(tp.label()).monospace().small());
                }
            }
            if !rec.time_summary.is_empty() {
                ui.label(egui::RichText::new(&rec.time_summary).color(Color32::GRAY).small());
            }
            ui.label(egui::RichText::new(format!(
                "planned in {:.0}s (frame {:.0} · aim {:.0} · evolve {:.0} · render {:.0})",
                rec.total_secs(), rec.secs_frame, rec.secs_aim, rec.secs_evolve, rec.secs_render
            )).color(Color32::GRAY).small());

            ui.separator();
            ui.horizontal(|ui| {
                if ui.add(egui::Button::new("✓ Approve"))
                    .on_hover_text("Queue a full-resolution render at the settings above. The \
                                    video queue renders it in the background.")
                    .clicked()
                { do_approve = Some(i); }
                if ui.button("✗ Reject").clicked() { do_reject = Some(i); }
                if ui.add_enabled(self.redoing.is_none(), egui::Button::new("↻ Re-roll"))
                    .on_hover_text("Search for a different time formula and re-render the \
                                    preview. Keeps the framing and the zoom — those are the \
                                    expensive, deterministic half of planning a shot, and \
                                    re-deriving them would give the same answer.")
                    .clicked()
                { do_redo = Some(i); }
                if self.redoing.is_some() { ui.spinner(); }
                if rec.status != ReelStatus::Pending && ui.button("↺ Undecided").clicked() {
                    do_reset = Some(i);
                }
                if ui.button("Open preview").clicked() {
                    if let Some(d) = self.batch_dir() {
                        let p = d.join(&rec.preview_file);
                        let _ = std::process::Command::new("xdg-open").arg(p).spawn();
                    }
                }
            });
            if !self.message.is_empty() {
                let col = if self.message.starts_with("cannot") {
                    Color32::LIGHT_RED
                } else {
                    Color32::LIGHT_GREEN
                };
                ui.colored_label(col, &self.message);
            }
        });
        });
        });

        if do_reload { self.reload(); }
        if let Some(i) = do_select { self.select(i); }
        if let Some(i) = do_approve { self.approve(i); }
        if let Some(i) = do_reject {
            self.set_status(i, ReelStatus::Rejected);
            self.message = "rejected".into();
        }
        if let Some(i) = do_reset {
            self.set_status(i, ReelStatus::Pending);
            self.message.clear();
        }
        if do_start_batch { self.start_batch(); }
        if do_stop_batch { self.stop_batch(); }
        if let Some(i) = do_redo { self.redo(i); }
        if self.batch_running {
            // Reels are written one at a time, so re-read periodically and the
            // list fills in while the batch is still going.
            if self.last_batch_poll.elapsed() >= std::time::Duration::from_secs(5) {
                self.last_batch_poll = std::time::Instant::now();
                self.reload();
            }
            ctx.request_repaint_after(std::time::Duration::from_millis(400));
        }
        if self.redoing.is_some() {
            ctx.request_repaint_after(std::time::Duration::from_millis(500));
        }
    }
}

fn main() -> anyhow::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1200.0, 860.0]),
        ..Default::default()
    };
    eframe::run_native(
        "NNFractals Reels",
        options,
        Box::new(move |cc| {
            nnfractals::gui_font::install(&cc.egui_ctx);
            Ok(Box::new(App::new(cc)))
        }),
    ).map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_player_fills_two_thirds_of_the_window_when_it_can() {
        // Carl's report: the preview came out "very small". It was being sized
        // from `available_size()` inside a nested layout, which reports what is
        // left of the current row rather than the window.
        //
        // A tall enough window: width is what binds, and it is two thirds.
        let panel = egui::Vec2::new(1400.0, 1200.0);
        let (w, h) = player_size(panel, 1.0);
        assert!((w / panel.x - PLAYER_WIDTH_FRACTION).abs() < 0.01,
                "{w} of {} is not two thirds", panel.x);
        assert!((w - h).abs() < 0.01, "a square clip must stay square");
    }

    #[test]
    fn a_square_preview_gets_most_of_a_realistic_window() {
        // The shape that actually ships: 512x512 previews in the default
        // window, minus the two toolbar rows. Whatever binds, the result has to
        // be a clip worth looking at rather than a thumbnail.
        let panel = egui::Vec2::new(1200.0 - 16.0, 860.0 - 90.0);
        let (w, h) = player_size(panel, 1.0);
        assert!(w >= 500.0, "a 512px preview drawn at {w}px is still small");
        assert!((w - h).abs() < 0.01);
    }

    #[test]
    fn the_player_gives_way_before_the_buttons_go_out_of_reach() {
        // A short window must shrink the clip rather than push Approve/Reject
        // off the bottom.
        let panel = egui::Vec2::new(1400.0, 480.0);
        let (w, h) = player_size(panel, 1.0);
        assert!(h <= panel.y - DETAIL_HEIGHT + 0.01, "player {h} leaves no room in {}", panel.y);
        assert!(w < panel.x * PLAYER_WIDTH_FRACTION, "it should have shrunk, got {w}");
    }

    #[test]
    fn the_player_never_overlaps_the_list_or_inverts() {
        for (pw, ph, asp) in [(1400.0, 900.0, 1.0), (600.0, 400.0, 1.0),
                              (900.0, 900.0, 16.0 / 9.0), (900.0, 900.0, 9.0 / 16.0),
                              (300.0, 200.0, 1.0)] {
            let (w, h) = player_size(egui::Vec2::new(pw, ph), asp);
            assert!(w > 0.0 && h > 0.0, "{pw}x{ph} @{asp} gave {w}x{h}");
            assert!(w <= (pw - LIST_WIDTH).max(120.0) + 0.01,
                    "{w} overlaps the {LIST_WIDTH}px list in a {pw}px panel");
            assert!((w / h - asp).abs() < 0.01, "aspect drifted: {w}x{h} vs {asp}");
        }
        // A degenerate aspect must not produce NaN.
        let (w, h) = player_size(egui::Vec2::new(1400.0, 900.0), 0.0);
        assert!(w.is_finite() && h.is_finite() && w > 0.0 && h > 0.0);
    }

    #[test]
    fn the_log_stays_bounded_however_long_a_batch_runs() {
        // A batch runs for hours and prints per generation; an unbounded log
        // would grow without limit in a window meant to be left open.
        let log: Vec<String> = Vec::new();
        let log = std::sync::Mutex::new(log);
        for i in 0..LOG_LINES * 3 {
            let mut l = log.lock().unwrap();
            l.push(format!("line {i}"));
            let n = l.len();
            if n > LOG_LINES {
                l.drain(..n - LOG_LINES);
            }
        }
        let l = log.lock().unwrap();
        assert_eq!(l.len(), LOG_LINES);
        // And it keeps the NEWEST lines, which are the ones worth seeing.
        assert_eq!(l.last().unwrap(), &format!("line {}", LOG_LINES * 3 - 1));
    }

    #[test]
    fn discovered_pools_are_real_directories() {
        // The combo must never offer a pool that would make the run exit with
        // "no .nn files" — `Starred` in particular is only there when it is.
        let root = nnfractals::project_root();
        for p in discover_pools() {
            assert!(root.join(&p).is_dir(), "offered {p}, which is not a directory");
        }
    }
}
