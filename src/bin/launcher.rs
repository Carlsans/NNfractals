//! nnfractals-launcher — one front door to every part of the project.
//!
//! A small GUI that lets you: browse the gallery, open a fractal in the viewer,
//! start/stop/monitor the evolution loop (N instances, via `run.sh`), and install
//! a desktop entry so NNFractals shows up in your application menu.
//!
//! It also generates its own `.desktop` file — either from the "Install to app
//! menu" button, or headless via `nnfractals-launcher --install-desktop`.
//!
//! Run:  cargo run --features launcher --bin nnfractals-launcher

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use clap::Parser;
use eframe::egui::{self, Color32};
use sysinfo::{ProcessesToUpdate, System};

#[derive(Parser)]
#[command(name = "nnfractals-launcher", about = "Front door to NNFractals")]
struct Args {
    /// Write the .desktop entry to ~/.local/share/applications and exit
    #[arg(long)]
    install_desktop: bool,
}

// ── Locating things ────────────────────────────────────────────────────────────

/// Directory containing the sibling binaries (nnfractals, -viewer, -browser).
fn bin_dir() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(Path::to_path_buf))
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Locate a project binary robustly: sibling of this exe, the other build profile
/// (target/debug ↔ target/release), then ~/.local/bin, then the bare name (PATH).
fn sibling(name: &str) -> PathBuf {
    let dir = bin_dir();
    let c = dir.join(name);
    if c.exists() {
        return c;
    }
    if let Some(target) = dir.parent() {
        for prof in ["release", "debug"] {
            let c = target.join(prof).join(name);
            if c.exists() {
                return c;
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

/// The project root — the dir holding `config.toml` (needed as cwd for evolution,
/// which reads config.toml and writes ./fractals*). Search: cwd, then up from the
/// executable (covers target/{debug,release}/ layouts).
fn project_root() -> PathBuf {
    if Path::new("config.toml").exists() {
        return std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    }
    let mut dir = bin_dir();
    for _ in 0..6 {
        if dir.join("config.toml").exists() {
            return dir;
        }
        match dir.parent() {
            Some(p) => dir = p.to_path_buf(),
            None => break,
        }
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

// ── Desktop entry generation ───────────────────────────────────────────────────

fn desktop_entry_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".local/share/applications/nnfractals.desktop")
}

/// Write ~/.local/share/applications/nnfractals.desktop pointing at this launcher.
/// Returns the path written (or an error string).
fn install_desktop_entry() -> Result<PathBuf, String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let root = project_root();
    // Use a gallery-agnostic icon only if the user dropped one at the root.
    let icon = root.join("icon.png");
    let icon_line = if icon.exists() {
        format!("Icon={}\n", icon.display())
    } else {
        String::new()
    };
    let contents = format!(
        "[Desktop Entry]\n\
         Type=Application\n\
         Name=NNFractals\n\
         GenericName=Fractal Evolver\n\
         Comment=Evolve, browse and view neural-network fractals\n\
         Exec={exe}\n\
         Path={root}\n\
         {icon}\
         Terminal=false\n\
         Categories=Graphics;2DGraphics;\n\
         Keywords=fractal;evolution;art;\n",
        exe = exe.display(),
        root = root.display(),
        icon = icon_line,
    );
    let path = desktop_entry_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(&path, contents).map_err(|e| e.to_string())?;
    // Best-effort: refresh the menu database (ignored if the tool is absent).
    if let Some(parent) = path.parent() {
        let _ = Command::new("update-desktop-database").arg(parent).status();
    }
    Ok(path)
}

// ── Process / resource monitoring ────────────────────────────────────────────

/// One running evolution or scorer process with its resource usage.
struct ProcRow {
    pid: u32,
    kind: &'static str, // "evolution" | "scorer"
    cpu: f32,           // percent (can exceed 100 across cores)
    ram_mb: u64,        // resident set size
    vram_mb: u64,       // GPU memory (from nvidia-smi), 0 if none/unknown
}

/// Per-PID GPU memory (MiB) from `nvidia-smi`. Empty if nvidia-smi is absent.
fn gpu_mem_by_pid() -> HashMap<u32, u64> {
    let mut m = HashMap::new();
    if let Ok(out) = Command::new("nvidia-smi")
        .args(["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"])
        .output()
    {
        for line in String::from_utf8_lossy(&out.stdout).lines() {
            let mut it = line.split(',');
            if let (Some(p), Some(mem)) = (it.next(), it.next()) {
                if let (Ok(pid), Ok(mb)) = (p.trim().parse::<u32>(), mem.trim().parse::<u64>()) {
                    *m.entry(pid).or_insert(0) += mb;
                }
            }
        }
    }
    m
}

/// One-line overall GPU utilisation + VRAM summary from `nvidia-smi`.
fn gpu_overall() -> Option<String> {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let parts: Vec<String> = text.lines().next()?.split(',').map(|s| s.trim().to_string()).collect();
    if parts.len() >= 3 {
        Some(format!("GPU {}% · VRAM {}/{} MiB", parts[0], parts[1], parts[2]))
    } else {
        None
    }
}

// ── Background job tracking (train / rescore) ────────────────────────────────

/// A message streamed from a running python job (train_pref.py) to the UI.
enum JobMsg {
    Progress { phase: String, done: usize, total: usize },
    Log(String),
    Done(String),
    Failed(String),
}

/// Live state of the currently-running (or last) train/rescore job.
#[derive(Default)]
struct JobState {
    running: bool,
    name: String,   // e.g. "Train taste model" / "Rescore fractals_1"
    phase: String,  // "load" | "embed" | "write"
    done: usize,
    total: usize,
    message: String,
}

/// Parse one stdout/stderr line from the python job and forward it to the UI.
fn parse_job_line(tx: &mpsc::Sender<JobMsg>, line: &str) {
    if let Some(rest) = line.strip_prefix("PROGRESS ") {
        let mut it = rest.split_whitespace();
        if let (Some(phase), Some(d), Some(t)) = (it.next(), it.next(), it.next()) {
            if let (Ok(done), Ok(total)) = (d.parse::<usize>(), t.parse::<usize>()) {
                let _ = tx.send(JobMsg::Progress { phase: phase.to_string(), done, total });
                return;
            }
        }
    }
    if let Some(rest) = line.strip_prefix("DONE ") {
        let _ = tx.send(JobMsg::Done(rest.to_string()));
        return;
    }
    let _ = tx.send(JobMsg::Log(line.to_string()));
}

/// Fractal folders offering a rescore target: the live pools, Starred, corpus,
/// and any legacy `fractals*` dirs that still exist.
fn discover_pools(root: &Path) -> Vec<String> {
    let mut dirs: Vec<String> = std::fs::read_dir(root)
        .map(|entries| {
            entries
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.is_dir())
                .filter_map(|p| p.file_name().and_then(|n| n.to_str()).map(str::to_owned))
                .filter(|n| n.starts_with("fractals") || n == "Starred" || n == "train_corpus")
                .collect()
        })
        .unwrap_or_default();
    dirs.sort();
    dirs.dedup();
    dirs
}

// ── Application ────────────────────────────────────────────────────────────────

struct App {
    root: PathBuf,
    instances: u32,
    viewer_path: String,
    status: String,

    // Live process/resource monitor (refreshed ~every 60s and on demand).
    sys: System,
    procs: Vec<ProcRow>,
    gpu_line: String,
    last_refresh: Option<Instant>,

    // Background train/rescore job + live progress.
    job: JobState,
    job_rx: Option<mpsc::Receiver<JobMsg>>,
    known_folders: Vec<String>,
    rescore_folder: String,

    // Dedup preview/run controls.
    dedup_folder: String,
    dedup_threshold: f32,
    dedup_confirm: bool,
}

impl App {
    fn new() -> Self {
        let root = project_root();
        let known_folders = discover_pools(&root);
        let rescore_folder = known_folders
            .iter()
            .find(|d| d.as_str() == "fractals_1")
            .or_else(|| known_folders.first())
            .cloned()
            .unwrap_or_else(|| "fractals_1".into());
        App {
            root,
            instances: 2,
            viewer_path: String::new(),
            status: String::new(),
            sys: System::new(),
            procs: Vec::new(),
            gpu_line: String::new(),
            last_refresh: None,
            job: JobState::default(),
            job_rx: None,
            known_folders: known_folders.clone(),
            rescore_folder,
            dedup_folder: known_folders
                .iter()
                .find(|d| d.as_str() == "fractals_1")
                .or_else(|| known_folders.first())
                .cloned()
                .unwrap_or_else(|| "fractals_1".into()),
            dedup_threshold: 0.94,
            dedup_confirm: false,
        }
    }

    /// Spawn `python3 <script> <args>` with piped output, streaming its progress
    /// into `self.job` via background reader threads (non-blocking, live-tracked).
    fn spawn_tracked(&mut self, name: String, script: &str, args: Vec<String>) {
        if self.job.running {
            self.status = "a background job is already running".into();
            return;
        }
        let mut cmd = Command::new("python3");
        cmd.arg(script)
            .args(&args)
            .current_dir(&self.root)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        // Mirror to a log file too (best-effort), matching the old behaviour.
        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                self.status = format!("could not start job: {e}");
                return;
            }
        };
        let (tx, rx) = mpsc::channel();
        if let Some(out) = child.stdout.take() {
            let tx = tx.clone();
            thread::spawn(move || {
                for line in BufReader::new(out).lines().map_while(Result::ok) {
                    parse_job_line(&tx, &line);
                }
            });
        }
        if let Some(err) = child.stderr.take() {
            let tx = tx.clone();
            thread::spawn(move || {
                for line in BufReader::new(err).lines().map_while(Result::ok) {
                    parse_job_line(&tx, &line);
                }
            });
        }
        // Waiter thread: emit a terminal Done/Failed when the process exits.
        thread::spawn(move || {
            let msg = match child.wait() {
                Ok(s) if s.success() => JobMsg::Done("completed".into()),
                Ok(s) => JobMsg::Failed(format!("python exited with {s}")),
                Err(e) => JobMsg::Failed(format!("wait failed: {e}")),
            };
            let _ = tx.send(msg);
        });
        self.job = JobState { running: true, name: name.clone(), ..Default::default() };
        self.job_rx = Some(rx);
        self.status = format!("{name}: started");
    }

    /// Drain any queued job messages into `self.job` (called each frame).
    fn poll_job(&mut self) {
        let mut msgs = Vec::new();
        if let Some(rx) = &self.job_rx {
            while let Ok(m) = rx.try_recv() {
                msgs.push(m);
            }
        }
        for m in msgs {
            match m {
                JobMsg::Progress { phase, done, total } => {
                    self.job.phase = phase;
                    self.job.done = done;
                    self.job.total = total;
                }
                JobMsg::Log(l) => self.job.message = l,
                JobMsg::Done(m) => {
                    self.job.running = false;
                    self.job.message = m;
                }
                JobMsg::Failed(e) => {
                    self.job.running = false;
                    self.job.message = format!("FAILED: {e}");
                }
            }
        }
        if !self.job.running {
            self.job_rx = None;
        }
    }

    /// Re-score every fractal in the chosen folder with the current saved model.
    fn rescore_folder(&mut self) {
        if !self.root.join("pref_model.npz").exists() {
            self.status = "no pref_model.npz — train the taste model first".into();
            return;
        }
        let folder = self.rescore_folder.clone();
        if !self.root.join(&folder).is_dir() {
            self.status = format!("folder '{folder}' not found");
            return;
        }
        self.spawn_tracked(
            format!("Rescore {folder}"),
            "scripts/train_pref.py",
            vec!["--score-only".into(), "--dirs".into(), folder],
        );
    }

    /// Scan the process table for evolution (`nnfractals`) + aesthetic scorer
    /// (`aesthetic_scorer.py`) processes — regardless of who started them — and
    /// read their CPU/RAM (sysinfo) + VRAM (nvidia-smi).
    fn refresh_procs(&mut self) {
        self.sys.refresh_processes(ProcessesToUpdate::All, true);
        let vram = gpu_mem_by_pid();
        let mut rows: Vec<ProcRow> = Vec::new();
        for (pid, p) in self.sys.processes() {
            let name = p.name().to_string_lossy();
            let is_evo = name == "nnfractals";
            let is_scorer = !is_evo
                && p.cmd().iter().any(|a| a.to_string_lossy().contains("aesthetic_scorer.py"));
            if !(is_evo || is_scorer) {
                continue;
            }
            let id = pid.as_u32();
            rows.push(ProcRow {
                pid: id,
                kind: if is_evo { "evolution" } else { "scorer" },
                cpu: p.cpu_usage(),
                ram_mb: p.memory() / 1024 / 1024,
                vram_mb: vram.get(&id).copied().unwrap_or(0),
            });
        }
        rows.sort_by(|a, b| (a.kind, a.pid).cmp(&(b.kind, b.pid)));
        self.procs = rows;
        self.gpu_line = gpu_overall().unwrap_or_else(|| "GPU: nvidia-smi unavailable".into());
        self.known_folders = discover_pools(&self.root);
        self.last_refresh = Some(Instant::now());
    }

    /// Report how many evolution/scorer processes are running (works even for
    /// instances started from the CLI, not just via run.sh).
    fn status_evolution(&mut self) {
        self.refresh_procs();
        if self.procs.is_empty() {
            self.status = "no evolution / scorer processes running".into();
        } else {
            let evo = self.procs.iter().filter(|r| r.kind == "evolution").count();
            let sc = self.procs.iter().filter(|r| r.kind == "scorer").count();
            self.status = format!("running: {evo} evolution + {sc} scorer process(es)");
        }
    }

    /// Stop ALL evolution + scorer processes by PID (SIGTERM, then SIGKILL any
    /// stragglers), independent of run.sh's .run_pids. Also runs `run.sh stop`
    /// for its bookkeeping.
    fn stop_evolution(&mut self) {
        self.refresh_procs();
        let pids: Vec<u32> = self.procs.iter().map(|r| r.pid).collect();
        if pids.is_empty() {
            self.status = "no evolution / scorer processes to stop".into();
            return;
        }
        for pid in &pids {
            let _ = Command::new("kill").arg(pid.to_string()).status();
        }
        std::thread::sleep(Duration::from_millis(1200));
        self.refresh_procs();
        for r in &self.procs {
            let _ = Command::new("kill").arg("-9").arg(r.pid.to_string()).status();
        }
        // Best-effort run.sh cleanup (clears .run_pids, pkills scorer stragglers).
        let script = self.root.join("run.sh");
        if script.exists() {
            let _ = Command::new("bash").arg(&script).arg("stop").current_dir(&self.root).output();
        }
        self.refresh_procs();
        self.status = format!("stopped {} process(es)", pids.len());
    }

    /// Spawn a sibling binary with the project root as cwd.
    fn spawn(&mut self, bin: PathBuf, args: &[&str], what: &str) {
        match Command::new(&bin).args(args).current_dir(&self.root).spawn() {
            Ok(_) => self.status = format!("Launched {what}"),
            Err(e) => self.status = format!("Could not launch {what}: {e}"),
        }
    }

    /// Run `run.sh <args>` in the project root and capture its output.
    fn run_sh(&mut self, args: &[&str]) {
        let script = self.root.join("run.sh");
        if !script.exists() {
            self.status = format!("run.sh not found in {}", self.root.display());
            return;
        }
        match Command::new("bash")
            .arg(&script)
            .args(args)
            .current_dir(&self.root)
            .output()
        {
            Ok(out) => {
                let s = String::from_utf8_lossy(&out.stdout);
                let e = String::from_utf8_lossy(&out.stderr);
                let text = format!("{s}{e}");
                self.status = text.trim().to_string();
            }
            Err(err) => self.status = format!("run.sh failed: {err}"),
        }
    }

    /// Train the human-preference model from the browser's ratings, then score
    /// the galleries. Runs in the background with live progress tracking.
    fn train_pref(&mut self) {
        // Prefer the central accumulating corpus store (browser writes rated
        // images + comparisons there so they survive dedup); fall back to legacy.
        let ratings = [
            "train_corpus/ratings.jsonl",
            "fractals_dag/ratings.jsonl",
            "fractals/ratings.jsonl",
            "ratings.jsonl",
        ]
        .iter()
        .map(|r| self.root.join(r))
        .find(|p| p.exists());
        let Some(ratings) = ratings else {
            self.status = "no ratings.jsonl found — rate fractals in the browser (⚖ Rate) first".into();
            return;
        };
        // Score the live pools + curated Starred (plus the corpus itself, which
        // holds the rated images even after evolution deleted the originals).
        let dirs: Vec<String> = ["fractals_1", "fractals_2", "Starred", "train_corpus", "fractals_dag", "fractals"]
            .iter()
            .filter(|d| self.root.join(d).is_dir())
            .map(|d| d.to_string())
            .collect();
        if dirs.is_empty() {
            self.status = "no gallery folders found to score".into();
            return;
        }
        let mut args = vec![
            "--ratings".to_string(),
            ratings.to_string_lossy().into_owned(),
            "--dirs".to_string(),
        ];
        args.extend(dirs);
        self.spawn_tracked("Train taste model".into(), "scripts/train_pref.py", args);
    }

    /// Preview (dry-run) or actually run the near-duplicate cleaner on the
    /// chosen folder at the chosen threshold. Progress/result stream through
    /// the same job-tracking machinery as train/rescore.
    fn dedup(&mut self, dry_run: bool) {
        let folder = self.dedup_folder.clone();
        if !self.root.join(&folder).is_dir() {
            self.status = format!("folder '{folder}' not found");
            return;
        }
        let mut args = vec![
            "--run".to_string(),
            "--dir".to_string(),
            folder.clone(),
            "--threshold".to_string(),
            format!("{:.3}", self.dedup_threshold),
        ];
        if dry_run {
            args.push("--dry-run".to_string());
        } else {
            let binary = sibling("nnfractals");
            args.push("--binary".to_string());
            args.push(binary.to_string_lossy().into_owned());
        }
        let name = if dry_run {
            format!("Preview dedup {folder}")
        } else {
            format!("Dedup {folder}")
        };
        self.spawn_tracked(name, "scripts/dedup.py", args);
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // Drain live progress from any running train/rescore job.
        self.poll_job();

        // Refresh the resource monitor at most every 60s; schedule a wake-up so
        // it keeps updating even when the window is idle.
        let stale = self.last_refresh.map_or(true, |t| t.elapsed() >= Duration::from_secs(60));
        if stale {
            self.refresh_procs();
        }
        // While a job runs, repaint frequently for smooth progress; otherwise
        // just keep the resource monitor ticking.
        if self.job.running {
            ui.ctx().request_repaint_after(Duration::from_millis(250));
        } else {
            ui.ctx().request_repaint_after(Duration::from_secs(60));
        }

        egui::CentralPanel::default().show(ui, |ui| {
            ui.add_space(6.0);
            ui.heading("NNFractals");
            ui.label(
                egui::RichText::new(format!("project: {}", self.root.display()))
                    .weak()
                    .monospace(),
            );
            ui.separator();

            // ── Explore ──
            ui.label(egui::RichText::new("Explore").strong());
            ui.horizontal(|ui| {
                if ui
                    .button("🖼  Browse gallery")
                    .on_hover_text("Open the fractal browser (sort, curate, pick breeding stock)")
                    .clicked()
                {
                    self.spawn(sibling("nnfractals-browser"), &[], "gallery browser");
                }
            });
            ui.horizontal(|ui| {
                ui.label("Viewer:");
                ui.add(
                    egui::TextEdit::singleline(&mut self.viewer_path)
                        .desired_width(300.0)
                        .hint_text("path to a .nn file (or use the browser)"),
                );
                let has = !self.viewer_path.trim().is_empty();
                if ui
                    .add_enabled(has, egui::Button::new("🔬  Open"))
                    .clicked()
                {
                    let p = self.viewer_path.trim().to_string();
                    self.spawn(sibling("nnfractals-viewer"), &[p.as_str()], "viewer");
                }
            });

            ui.add_space(8.0);
            ui.separator();

            // ── Evolve ──
            ui.label(egui::RichText::new("Evolve").strong());
            ui.horizontal(|ui| {
                ui.label("Instances:");
                ui.add(egui::DragValue::new(&mut self.instances).range(1..=32));
                if ui
                    .button("▶  Start")
                    .on_hover_text("Launch N background evolution instances (run.sh).\n\
                                     Each instance auto-starts its aesthetic scorer.")
                    .clicked()
                {
                    let n = self.instances.to_string();
                    self.run_sh(&[n.as_str()]);
                    // Give the instances a moment to fork their scorer children,
                    // then reflect them in the monitor.
                    std::thread::sleep(Duration::from_millis(600));
                    self.refresh_procs();
                }
                if ui
                    .button("■  Stop")
                    .on_hover_text("Stop every evolution + scorer process (by PID scan),\n\
                                     even ones started outside the launcher.")
                    .clicked()
                {
                    self.stop_evolution();
                }
                if ui
                    .button("↻  Status")
                    .on_hover_text("List running evolution + scorer processes.")
                    .clicked()
                {
                    self.status_evolution();
                }
            });
            ui.label(
                egui::RichText::new("Instances share one gallery and log to evolution.log.")
                    .weak(),
            );

            ui.horizontal(|ui| {
                if ui
                    .add_enabled(!self.job.running, egui::Button::new("🎓  Train taste model"))
                    .on_hover_text("Train the preference model from your browser ratings\n\
                                     (ratings.jsonl) and score every fractal → pref_score.\n\
                                     Evolution then selects on your taste (optimization.pref_weight).")
                    .clicked()
                {
                    self.train_pref();
                }
            });

            // ── Rescore a folder with the current (already-trained) model ──
            ui.horizontal(|ui| {
                ui.label("Rescore folder:");
                egui::ComboBox::from_id_salt("rescore_folder")
                    .selected_text(&self.rescore_folder)
                    .show_ui(ui, |ui| {
                        for d in &self.known_folders {
                            ui.selectable_value(&mut self.rescore_folder, d.clone(), d);
                        }
                    });
                if ui
                    .add_enabled(!self.job.running, egui::Button::new("↺  Rescore with model"))
                    .on_hover_text("Iterate every .nn in the folder and update pref_score\n\
                                     using the current saved model (pref_model.npz).\n\
                                     Run this after training to refresh a whole gallery.")
                    .clicked()
                {
                    self.rescore_folder();
                }
            });

            ui.add_space(8.0);
            ui.separator();

            // ── Dedup: find/remove near-duplicate fractals in a folder ──
            ui.label(egui::RichText::new("Dedup").strong());
            ui.horizontal(|ui| {
                ui.label("Folder:");
                egui::ComboBox::from_id_salt("dedup_folder")
                    .selected_text(&self.dedup_folder)
                    .show_ui(ui, |ui| {
                        for d in &self.known_folders {
                            ui.selectable_value(&mut self.dedup_folder, d.clone(), d);
                        }
                    });
                ui.label("Threshold:");
                ui.add(
                    egui::DragValue::new(&mut self.dedup_threshold)
                        .range(0.80..=0.999)
                        .speed(0.001)
                        .fixed_decimals(3),
                )
                .on_hover_text("Cosine-similarity cutoff for \"same fractal\" (0–1).\n\
                                 Higher = stricter (fewer matches). Default 0.94.");
            });
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(!self.job.running, egui::Button::new("🔍  Preview (dry run)"))
                    .on_hover_text("Scan the folder and report how many fractals would be\n\
                                     deleted at this threshold. Deletes and re-renders nothing.")
                    .clicked()
                {
                    self.dedup_confirm = false;
                    self.dedup(true);
                }
                ui.checkbox(&mut self.dedup_confirm, "confirm");
                if ui
                    .add_enabled(
                        !self.job.running && self.dedup_confirm,
                        egui::Button::new("🗑  Delete duplicates"),
                    )
                    .on_hover_text("Actually delete the lower-scored fractal of each\n\
                                     near-duplicate pair. Preview first — tick \"confirm\" to enable.")
                    .clicked()
                {
                    self.dedup_confirm = false;
                    self.dedup(false);
                }
            });

            // ── Live job progress (train / rescore / dedup) ──
            if self.job.running || !self.job.message.is_empty() {
                ui.add_space(4.0);
                let phase_label = match self.job.phase.as_str() {
                    "load" => "loading model",
                    "embed" => "embedding images",
                    "write" => "writing pref_score",
                    "round" => "scanning",
                    "vectorize" => "comparing images",
                    "" => "starting…",
                    other => other,
                };
                let header = if self.job.running {
                    format!("{} · {}", self.job.name, phase_label)
                } else {
                    format!("{} · done", self.job.name)
                };
                ui.label(egui::RichText::new(header).strong());
                let frac = if self.job.total > 0 {
                    self.job.done as f32 / self.job.total as f32
                } else if self.job.running {
                    0.0
                } else {
                    1.0
                };
                let text = if self.job.total > 0 {
                    format!("{}/{}", self.job.done, self.job.total)
                } else {
                    String::new()
                };
                ui.add(egui::ProgressBar::new(frac).text(text).desired_width(360.0));
                if !self.job.message.is_empty() {
                    let col = if self.job.message.starts_with("FAILED") {
                        Color32::LIGHT_RED
                    } else {
                        Color32::GRAY
                    };
                    ui.label(egui::RichText::new(&self.job.message).color(col).monospace());
                }
            }

            ui.add_space(8.0);
            ui.separator();

            // ── Processes (live resource monitor) ──
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Processes").strong());
                if ui.button("↻  Refresh").on_hover_text("Auto-refreshes every 60s").clicked() {
                    self.refresh_procs();
                }
                if let Some(t) = self.last_refresh {
                    ui.label(
                        egui::RichText::new(format!("updated {}s ago", t.elapsed().as_secs()))
                            .weak(),
                    );
                }
            });
            ui.label(egui::RichText::new(&self.gpu_line).monospace());
            if self.procs.is_empty() {
                ui.label(egui::RichText::new("no evolution / scorer processes running").weak());
            } else {
                egui::Grid::new("proc_grid")
                    .striped(true)
                    .num_columns(5)
                    .show(ui, |ui| {
                        ui.strong("pid");
                        ui.strong("kind");
                        ui.strong("cpu%");
                        ui.strong("ram");
                        ui.strong("vram");
                        ui.end_row();
                        for r in &self.procs {
                            ui.monospace(r.pid.to_string());
                            ui.label(r.kind);
                            ui.monospace(format!("{:.0}", r.cpu));
                            ui.monospace(format!("{} MB", r.ram_mb));
                            ui.monospace(if r.vram_mb > 0 {
                                format!("{} MiB", r.vram_mb)
                            } else {
                                "—".into()
                            });
                            ui.end_row();
                        }
                    });
            }

            ui.add_space(8.0);
            ui.separator();

            // ── System ──
            ui.label(egui::RichText::new("System").strong());
            if ui
                .button("📥  Install to app menu")
                .on_hover_text("Write a .desktop entry so NNFractals appears in your launcher")
                .clicked()
            {
                match install_desktop_entry() {
                    Ok(p) => self.status = format!("Installed desktop entry → {}", p.display()),
                    Err(e) => self.status = format!("Install failed: {e}"),
                }
            }

            ui.add_space(10.0);
            if !self.status.is_empty() {
                ui.separator();
                ui.label(egui::RichText::new(&self.status).color(Color32::LIGHT_GREEN).monospace());
            }
        });
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    if args.install_desktop {
        match install_desktop_entry() {
            Ok(p) => {
                println!("Installed desktop entry → {}", p.display());
                return Ok(());
            }
            Err(e) => anyhow::bail!("install failed: {e}"),
        }
    }

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("NNFractals Launcher")
            .with_inner_size([560.0, 440.0]),
        ..Default::default()
    };
    eframe::run_native(
        "NNFractals Launcher",
        options,
        Box::new(|cc| {
            nnfractals::gui_font::install(&cc.egui_ctx);
            Ok(Box::new(App::new()))
        }),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(())
}
