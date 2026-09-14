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
use serde::{Deserialize, Serialize};
use sysinfo::{ProcessesToUpdate, System};

use nnfractals::config::Config;
use nnfractals::fitness_profile::{self, FieldSpec, FitnessProfile, Group, Sign, FIELDS};

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

/// Locate a project binary robustly: target/release/<name> FIRST — even if
/// this exe is itself a debug build sitting right next to a debug sibling —
/// then sibling of this exe, then target/debug/<name>, then ~/.local/bin,
/// then the bare name (PATH). Same fix as `browser.rs::locate_bin` /
/// `viewer.rs::locate_sibling_bin` (identical bug, independently
/// duplicated here) — confirmed in practice: a desktop entry hardcoding
/// `target/debug/nnfractals-launcher` cascaded into debug browser/viewer
/// siblings indefinitely, because the old check-my-own-dir-first order
/// never got as far as preferring release.
fn sibling(name: &str) -> PathBuf {
    let dir = bin_dir();
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
    /// The script's own authoritative "DONE ..." summary line — wins over
    /// any later terminal message once received.
    Summary(String),
    /// Process termination, reported by the waiter thread. On success this
    /// is only a fallback message (used if the script never printed its own
    /// summary); on failure it always overrides, since a crash matters more
    /// than whatever was last logged.
    Exited { success: bool, detail: String },
}

/// Live state of the currently-running (or last) train/rescore/dedup job.
#[derive(Default)]
struct JobState {
    running: bool,
    name: String,   // e.g. "Train taste model" / "Rescore fractals_1"
    phase: String,  // "load" | "embed" | "write" | "round" | "vectorize"
    done: usize,
    total: usize,
    message: String,
    /// Set once the script's own "DONE ..." summary has been received, so
    /// the process-exit fallback message doesn't clobber it (see JobMsg::Exited).
    got_summary: bool,
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
        let _ = tx.send(JobMsg::Summary(rest.to_string()));
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

/// Classify a process as one this launcher manages, or not at all.
///
/// Stop kills exactly what this matches, so it is worth being able to test:
/// the evolution binary is matched on its process NAME (comm), which is stable
/// no matter what arguments it was launched with — notably `--profile`, added
/// when fitness profiles landed. Matching on the command line instead would
/// have quietly stopped finding profile-launched runs.
fn classify_proc(name: &str, cmd: &[String]) -> Option<&'static str> {
    if name == "nnfractals" {
        return Some("evolution");
    }
    if cmd.iter().any(|a| a.contains("aesthetic_scorer.py")) {
        return Some("scorer");
    }
    None
}

/// Every `config*.toml` at the project root, sorted. Mirrors `discover_pools`:
/// scan rather than hardcode, so a fifth instance config shows up on its own.
fn discover_configs(root: &Path) -> Vec<String> {
    let mut files: Vec<String> = std::fs::read_dir(root)
        .map(|entries| {
            entries
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.is_file())
                .filter_map(|p| p.file_name().and_then(|n| n.to_str()).map(str::to_owned))
                .filter(|n| n.starts_with("config") && n.ends_with(".toml"))
                .collect()
        })
        .unwrap_or_default();
    files.sort();
    files.dedup();
    if files.is_empty() {
        files.push("config.toml".to_string());
    }
    files
}

/// Rebuild a profile from the editor buffers. An empty buffer means "no
/// override — inherit the config's value". So does a buffer that doesn't parse:
/// silently turning a typo into 0 would quietly change what the run selects on,
/// and a zero weight is a real, meaningful choice that must be typed
/// deliberately. `bad_fields` surfaces those so they aren't invisible.
fn profile_from_bufs(
    name: &str, notes: &str, bufs: &HashMap<&'static str, String>,
) -> FitnessProfile {
    let mut p = FitnessProfile {
        name: name.trim().to_string(),
        notes: notes.to_string(),
        ..Default::default()
    };
    for f in FIELDS {
        let raw = bufs.get(f.key).map(|s| s.trim()).unwrap_or("");
        if raw.is_empty() { continue; }
        if let Ok(v) = raw.parse::<f32>() {
            p.set(f.key, Some(v));
        }
    }
    p
}

/// Labels of buffers that hold something that isn't a number.
fn bad_fields(bufs: &HashMap<&'static str, String>) -> Vec<&'static str> {
    FIELDS.iter()
        .filter(|f| {
            let raw = bufs.get(f.key).map(|s| s.trim()).unwrap_or("");
            !raw.is_empty() && raw.parse::<f32>().is_err()
        })
        .map(|f| f.label)
        .collect()
}

/// Render a weight for an edit box: short and round-trippable, without the
/// `0.6000000238` an f32 would otherwise print.
fn format_weight(v: f32) -> String {
    // Early-out on zero so -0.0 (which compares equal to 0.0) can't render as
    // "-0" after the trailing-zero trim.
    if v == 0.0 { return "0".to_string(); }
    let s = format!("{v:.4}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s.is_empty() || s == "-" { "0".to_string() } else { s.to_string() }
}

// ── Preferences (small TOML round-trip, mirrors BrowserPrefs) ──────────────────

const PREFS_FILE: &str = "launcher_prefs.toml";

#[derive(Serialize, Deserialize)]
struct LauncherPrefs {
    #[serde(default = "default_dedup_threshold")]
    dedup_threshold: f32,
}

fn default_dedup_threshold() -> f32 {
    0.94
}

impl Default for LauncherPrefs {
    fn default() -> Self {
        Self { dedup_threshold: default_dedup_threshold() }
    }
}

impl LauncherPrefs {
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
    prefs_path: PathBuf,

    // ── Fitness profiles ──────────────────────────────────────────────────
    show_fitness: bool,
    /// (display name, path) for every profiles/*.toml, rescanned when the
    /// window opens so files added by hand show up without a restart.
    fp_profiles: Vec<(String, PathBuf)>,
    /// Index into `fp_profiles`; `None` = editing an unsaved profile.
    fp_selected: Option<usize>,
    fp_name: String,
    fp_notes: String,
    /// One String buffer per knob, keyed by `FieldSpec::key`. EMPTY means "no
    /// override — inherit the config file's value", which is exactly the
    /// profile's sparse semantics made visible.
    fp_bufs: HashMap<&'static str, String>,
    fp_dirty: bool,
    /// The config the percentage readout and the "inherited" hints are computed
    /// against. `None` if the selected config file failed to parse.
    fp_base: Option<Config>,
    fp_config_file: String,
    fp_config_files: Vec<String>,
    fp_delete_confirm: bool,
}

impl App {
    fn new() -> Self {
        let root = project_root();
        let prefs_path = root.join(PREFS_FILE);
        let prefs = LauncherPrefs::load(&prefs_path);
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
            dedup_threshold: prefs.dedup_threshold,
            dedup_confirm: false,
            prefs_path,

            show_fitness: false,
            fp_profiles: Vec::new(),
            fp_selected: None,
            fp_name: String::new(),
            fp_notes: String::new(),
            fp_bufs: HashMap::new(),
            fp_dirty: false,
            fp_base: None,
            fp_config_file: "config.toml".to_string(),
            fp_config_files: Vec::new(),
            fp_delete_confirm: false,
        }
    }

    // ── Fitness profiles ──────────────────────────────────────────────────

    /// Rescan `profiles/` and the config files, and reload the base config the
    /// readouts are computed against. Called every time the window is opened so
    /// hand-edited files are picked up without restarting.
    fn fp_refresh(&mut self) {
        self.fp_profiles = fitness_profile::list_profiles(&self.root);
        self.fp_config_files = discover_configs(&self.root);
        if !self.fp_config_files.iter().any(|c| *c == self.fp_config_file) {
            if let Some(first) = self.fp_config_files.first() {
                self.fp_config_file = first.clone();
            }
        }
        self.fp_reload_base();
        if self.fp_bufs.is_empty() {
            self.fp_set_editor(&FitnessProfile::default());
        }
    }

    fn fp_reload_base(&mut self) {
        let path = self.root.join(&self.fp_config_file);
        match Config::load(&path) {
            Ok(c) => self.fp_base = Some(c),
            Err(e) => {
                self.fp_base = None;
                self.status = format!("could not read {}: {e}", path.display());
            }
        }
    }

    /// Load a profile's values into the editor buffers. A knob the profile does
    /// not set gets an empty buffer, i.e. "inherit".
    fn fp_set_editor(&mut self, p: &FitnessProfile) {
        self.fp_name = p.name.clone();
        self.fp_notes = p.notes.clone();
        self.fp_bufs.clear();
        for f in FIELDS {
            let text = p.get(f.key).map(|v| format_weight(v)).unwrap_or_default();
            self.fp_bufs.insert(f.key, text);
        }
        self.fp_dirty = false;
    }

    fn fp_current(&self) -> FitnessProfile {
        profile_from_bufs(&self.fp_name, &self.fp_notes, &self.fp_bufs)
    }

    fn fp_bad_fields(&self) -> Vec<&'static str> {
        bad_fields(&self.fp_bufs)
    }

    fn fp_load_selected(&mut self) {
        let Some(idx) = self.fp_selected else { return };
        let Some((name, path)) = self.fp_profiles.get(idx).cloned() else { return };
        match FitnessProfile::load(&path) {
            Ok(p) => {
                self.fp_set_editor(&p);
                self.status = format!("loaded profile '{name}'");
            }
            Err(e) => self.status = format!("could not load {}: {e}", path.display()),
        }
    }

    fn fp_save(&mut self) {
        let p = self.fp_current();
        if p.name.is_empty() {
            self.status = "give the profile a name before saving".into();
            return;
        }
        let bad = self.fp_bad_fields();
        if !bad.is_empty() {
            self.status = format!("not saved — these fields aren't numbers: {}", bad.join(", "));
            return;
        }
        let path = fitness_profile::profiles_dir(&self.root).join(format!("{}.toml", p.name));
        match p.save(&path) {
            Ok(()) => {
                self.status = format!("saved → {}", path.display());
                self.fp_dirty = false;
                self.fp_profiles = fitness_profile::list_profiles(&self.root);
                self.fp_selected = self.fp_profiles.iter().position(|(n, _)| *n == p.name);
            }
            Err(e) => self.status = format!("could not save {}: {e}", path.display()),
        }
    }

    fn fp_delete_selected(&mut self) {
        let Some(idx) = self.fp_selected else { return };
        let Some((name, path)) = self.fp_profiles.get(idx).cloned() else { return };
        match std::fs::remove_file(&path) {
            Ok(()) => {
                self.status = format!("deleted profile '{name}'");
                self.fp_profiles = fitness_profile::list_profiles(&self.root);
                self.fp_selected = None;
            }
            Err(e) => self.status = format!("could not delete {}: {e}", path.display()),
        }
    }

    /// Launch evolution with the selected config and (if one is selected and
    /// saved) the selected profile.
    fn fp_start(&mut self) {
        let n = self.instances.to_string();
        let cfg = self.fp_config_file.clone();
        let profile = self.fp_selected
            .and_then(|i| self.fp_profiles.get(i))
            .map(|(name, _)| name.clone());
        match profile {
            Some(name) => self.run_sh(&[n.as_str(), "--config", cfg.as_str(), "--profile", name.as_str()]),
            None => self.run_sh(&[n.as_str(), "--config", cfg.as_str()]),
        }
        std::thread::sleep(Duration::from_millis(600));
        self.refresh_procs();
    }

    /// Spawn `python3 <script> <args>` with piped output, streaming its progress
    /// into `self.job` via background reader threads (non-blocking, live-tracked).
    fn spawn_tracked(&mut self, name: String, script: &str, args: Vec<String>) {
        if self.job.running {
            self.status = "a background job is already running".into();
            return;
        }
        let mut cmd = Command::new(nnfractals::python_bin(&self.root));
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
        // Waiter thread: emit a terminal Exited when the process ends. On
        // success this is only a fallback (see JobMsg::Exited) so it doesn't
        // race with — and clobber — the script's own DONE summary line.
        thread::spawn(move || {
            let msg = match child.wait() {
                Ok(s) if s.success() => JobMsg::Exited { success: true, detail: "completed".into() },
                Ok(s) => JobMsg::Exited { success: false, detail: format!("python exited with {s}") },
                Err(e) => JobMsg::Exited { success: false, detail: format!("wait failed: {e}") },
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
                JobMsg::Log(l) => {
                    // Routine progress chatter — don't let it clobber the
                    // script's own final summary once one has arrived.
                    if !self.job.got_summary {
                        self.job.message = l;
                    }
                }
                JobMsg::Summary(m) => {
                    self.job.running = false;
                    self.job.got_summary = true;
                    self.job.message = m;
                }
                JobMsg::Exited { success, detail } => {
                    self.job.running = false;
                    if success {
                        // Fallback only: keep the script's own summary if we
                        // already have one (this message races it and is
                        // usually a beat behind, so it must never win).
                        if !self.job.got_summary {
                            self.job.message = detail;
                        }
                    } else {
                        self.job.message = format!("FAILED: {detail}");
                        self.job.got_summary = true;
                    }
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
            let cmd: Vec<String> = p.cmd().iter().map(|a| a.to_string_lossy().into_owned()).collect();
            let Some(kind) = classify_proc(&name, &cmd) else { continue };

            let id = pid.as_u32();
            rows.push(ProcRow {
                pid: id,
                kind,
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
            Ok(child) => {
                // Reap it — dropping a `Child` never waits, so every launched
                // binary that exits (immediately, in the viewer's
                // single-instance delegate case) would stay a zombie for the
                // lifetime of this launcher. Same fix as `browser.rs`'s
                // `open_path`, where three had accumulated in one session.
                std::thread::spawn(move || {
                    let mut child = child;
                    let _ = child.wait();
                });
                self.status = format!("Launched {what}");
            }
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
            // Confirmed deletions are a deliberate choice of threshold —
            // remember it as the default for next time.
            LauncherPrefs { dedup_threshold: self.dedup_threshold }.save(&self.prefs_path);
        }
        let name = if dry_run {
            format!("Preview dedup {folder}")
        } else {
            format!("Dedup {folder}")
        };
        self.spawn_tracked(name, "scripts/dedup.py", args);
    }

    /// The fitness-profile editor. Follows the viewer's `show_explore_options_window`
    /// shape: a bool on `App`, an early return, deferred action flags so `&mut self`
    /// is only touched after the closure's borrow ends, and hover text on
    /// everything.
    fn show_fitness_window(&mut self, ctx: &egui::Context, avail_h: f32) {
        if !self.show_fitness { return; }

        let mut do_load = false;
        let mut do_save = false;
        let mut do_delete = false;
        let mut do_start = false;
        let mut do_stop = false;
        let mut do_close = false;
        let mut reload_base = false;

        // Cloned up front: the combo boxes below need to read these while the
        // closure already holds `&mut self`.
        let profile_names: Vec<String> = self.fp_profiles.iter().map(|(n, _)| n.clone()).collect();
        let running = self.procs.iter().filter(|p| p.kind == "evolution").count();
        let config_files = self.fp_config_files.clone();
        let working = self.fp_current();
        let bad = self.fp_bad_fields();

        // Bounded by the actual app window: this used to be able to grow taller
        // than the launcher itself, putting its own Close button off-screen and
        // covering the main panel's controls with no way back.
        let screen_h = avail_h.max(320.0);
        let grid_h = (screen_h - 340.0).clamp(120.0, 420.0);
        egui::Window::new("Fitness Profiles")
            .collapsible(false)
            .resizable(true)
            .default_width(600.0)
            .max_height(screen_h - 20.0)
            .show(ctx, |ui| {
                // ── Profile picker ────────────────────────────────────────
                ui.horizontal(|ui| {
                    ui.label("Profile:").on_hover_text(
                        "Named fitness mixes stored as profiles/<name>.toml. A profile carries \
                         ONLY the weights it wants to change and is overlaid on top of the config \
                         file at launch, so config.toml keeps its own values — and its tuning \
                         comments — for everything else."
                    );
                    let selected_label = self.fp_selected
                        .and_then(|i| profile_names.get(i).cloned())
                        .unwrap_or_else(|| "(unsaved)".to_string());
                    egui::ComboBox::from_id_salt("fp_profile")
                        .selected_text(selected_label)
                        .width(200.0)
                        .show_ui(ui, |ui| {
                            for (i, name) in profile_names.iter().enumerate() {
                                ui.selectable_value(&mut self.fp_selected, Some(i), name);
                            }
                        });
                    if ui.add_enabled(self.fp_selected.is_some(), egui::Button::new("Load"))
                        .on_hover_text("Replace the editor below with this profile's values. \
                                        Unsaved edits are lost.")
                        .clicked()
                    { do_load = true; }
                    if ui.button("Save")
                        .on_hover_text("Write the editor's values to profiles/<name>.toml, \
                                        creating it if needed.")
                        .clicked()
                    { do_save = true; }
                    if ui.add_enabled(self.fp_selected.is_some(), egui::Button::new("Delete"))
                        .on_hover_text("Delete the selected profile file.")
                        .clicked()
                    { self.fp_delete_confirm = true; }
                });

                if self.fp_delete_confirm {
                    ui.horizontal(|ui| {
                        let name = self.fp_selected
                            .and_then(|i| profile_names.get(i).cloned())
                            .unwrap_or_default();
                        ui.colored_label(Color32::from_rgb(240, 160, 80),
                            format!("Delete profile '{name}'?"));
                        if ui.button("Yes, delete").clicked() {
                            do_delete = true;
                            self.fp_delete_confirm = false;
                        }
                        if ui.button("Cancel").clicked() { self.fp_delete_confirm = false; }
                    });
                }

                ui.horizontal(|ui| {
                    ui.label("Name:").on_hover_text("Saved as profiles/<name>.toml, and the name \
                                                     passed to --profile at launch.");
                    if ui.add(egui::TextEdit::singleline(&mut self.fp_name).desired_width(200.0))
                        .changed()
                    { self.fp_dirty = true; }
                });
                ui.horizontal_top(|ui| {
                    ui.label("Notes:").on_hover_text(
                        "Why this mix exists and what it is for. This is where the rationale that \
                         lives in config.toml's line comments should go for a profile — the \
                         numbers alone never explain themselves later."
                    );
                    if ui.add(egui::TextEdit::multiline(&mut self.fp_notes)
                            .desired_width(500.0).desired_rows(3))
                        .changed()
                    { self.fp_dirty = true; }
                });

                ui.separator();
                ui.label(egui::RichText::new(
                    "An EMPTY box means 'no override' — that knob keeps whatever the config file \
                     says (shown greyed in the box as a hint). Type a number to override it."
                ).color(Color32::GRAY).small());

                let Some(base) = self.fp_base.clone() else {
                    ui.colored_label(Color32::from_rgb(240, 120, 120),
                        "The selected config file could not be read — values and percentages \
                         are unavailable.");
                    if ui.button("Close").clicked() { do_close = true; }
                    return;
                };

                let shares = working.selection_shares(&base);

                egui::ScrollArea::vertical().max_height(grid_h).show(ui, |ui| {
                    for (group, title, blurb) in [
                        (Group::Selection, "1. Selection fitness",
                         "Summed every generation to rank the population. The percentages are each \
                          positive term's share of the mix — the stored values stay the raw numbers \
                          the code uses."),
                        (Group::Blend, "2. Saved fitness & seeding",
                         "Blended into the fitness stored on a saved genome, and into the ranking \
                          that picks which archive genomes seed a population. The 'Seed:' rows only \
                          do anything when archive_seeding_enabled = true in the config, which is \
                          currently false everywhere."),
                        (Group::Gate, "3. Save gate",
                         "Pass/fail thresholds, not weights — a genome must clear all of them to be \
                          written to disk. These change how MUCH gets saved as much as what."),
                    ] {
                        egui::CollapsingHeader::new(title)
                            .default_open(group == Group::Selection)
                            .show(ui, |ui| {
                                ui.label(egui::RichText::new(blurb).color(Color32::GRAY).small());
                                egui::Grid::new(format!("fp_grid_{title}"))
                                    .num_columns(4)
                                    .spacing([10.0, 4.0])
                                    .striped(true)
                                    .show(ui, |ui| {
                                        for f in FIELDS.iter().filter(|f| f.group == group) {
                                            fp_row(ui, f, &base, &working, &shares,
                                                   &mut self.fp_bufs, &mut self.fp_dirty);
                                        }
                                    });
                            });
                    }
                });

                // ── Resulting expression ──────────────────────────────────
                ui.separator();
                ui.label(egui::RichText::new("Resulting per-generation fitness").strong());
                ui.label(
                    egui::RichText::new(working.fitness_expression(&base))
                        .monospace()
                        .color(Color32::LIGHT_BLUE)
                ).on_hover_text(
                    "Exactly what Optimizer::step will compute, with zero-weighted terms omitted. \
                     This is the fastest way to catch a knob you meant to set and didn't."
                );

                if !bad.is_empty() {
                    ui.colored_label(Color32::from_rgb(240, 120, 120),
                        format!("Not a number: {} — these will be treated as 'inherit', not 0.",
                                bad.join(", ")));
                }

                // ── Launch ────────────────────────────────────────────────
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label("Config:").on_hover_text(
                        "The base config file. It decides the pool directory, population size and \
                         everything else the profile does not touch. Launching a second instance \
                         on the same config means both write to the same pool."
                    );
                    egui::ComboBox::from_id_salt("fp_config")
                        .selected_text(&self.fp_config_file)
                        .width(140.0)
                        .show_ui(ui, |ui| {
                            for c in &config_files {
                                if ui.selectable_value(&mut self.fp_config_file, c.clone(), c).clicked() {
                                    reload_base = true;
                                }
                            }
                        });
                    ui.label("Instances:");
                    ui.add(egui::DragValue::new(&mut self.instances).range(1..=32));
                    if ui.button("▶  Start with this profile")
                        .on_hover_text("run.sh N --config <config> --profile <name>. The child \
                                        process reads the profile FILE, so save your edits first.")
                        .clicked()
                    { do_start = true; }
                    // Deliberately here and not only on the main panel: this
                    // window floats over it, so without a Stop of its own the
                    // only way to end a run you just started is to close this
                    // window first — or kill the process by hand.
                    if ui.add_enabled(running > 0, egui::Button::new("■  Stop"))
                        .on_hover_text("Stop every evolution + scorer process, the same as the \
                                        Stop on the main panel.")
                        .clicked()
                    { do_stop = true; }
                    if running > 0 {
                        ui.colored_label(Color32::LIGHT_GREEN, format!("{running} running"));
                    }
                });

                if self.fp_dirty {
                    ui.colored_label(Color32::from_rgb(240, 200, 100),
                        "Unsaved edits — Start launches from the saved file, not from what is on \
                         screen. Save first.");
                }
                if self.fp_selected.is_none() {
                    ui.label(egui::RichText::new(
                        "No profile selected — Start will use the config file's own weights."
                    ).color(Color32::GRAY).small());
                }

                ui.separator();
                if ui.button("Close").clicked() { do_close = true; }
            });

        if reload_base { self.fp_reload_base(); }
        if do_load   { self.fp_load_selected(); }
        if do_save   { self.fp_save(); }
        if do_delete { self.fp_delete_selected(); }
        if do_start  { self.fp_start(); }
        if do_stop   { self.stop_evolution(); }
        if do_close  { self.show_fitness = false; }
    }
}

/// One editable knob: label, override box, live share, inherited value.
/// Free function rather than a method so it can borrow `fp_bufs` mutably while
/// the caller still holds the profile and base config immutably.
#[allow(clippy::too_many_arguments)]
fn fp_row(
    ui: &mut egui::Ui,
    f: &FieldSpec,
    base: &Config,
    working: &FitnessProfile,
    shares: &[(&'static str, f32, f32)],
    bufs: &mut HashMap<&'static str, String>,
    dirty: &mut bool,
) {
    let inherited = fitness_profile::FitnessProfile::default().effective(f.key, base);
    ui.label(f.label).on_hover_text(f.help);

    let buf = bufs.entry(f.key).or_default();
    if ui.add(
        egui::TextEdit::singleline(buf)
            .desired_width(70.0)
            .hint_text(format_weight(inherited)),
    ).on_hover_text(f.help).changed()
    {
        *dirty = true;
    }

    // Share column — only meaningful for the positive selection terms.
    match shares.iter().find(|(k, _, _)| *k == f.key) {
        Some((_, _, share)) => {
            let c = if *share >= 25.0 { Color32::LIGHT_BLUE }
                    else if *share > 0.0 { Color32::GRAY }
                    else { Color32::DARK_GRAY };
            ui.colored_label(c, format!("{share:5.1}%"))
                .on_hover_text("This term's share of the positive selection mix.");
        }
        None => {
            let note = match f.sign {
                Sign::Penalty => "penalty",
                Sign::Threshold => "gate",
                Sign::Positive => "",
            };
            ui.colored_label(Color32::DARK_GRAY, note);
        }
    }

    // Effective value, so an inherited row still shows what it will actually be.
    let effective = working.effective(f.key, base);
    let overridden = working.get(f.key).is_some();
    ui.colored_label(
        if overridden { Color32::GRAY } else { Color32::DARK_GRAY },
        format!("→ {}", format_weight(effective)),
    ).on_hover_text(if overridden {
        "Overridden by this profile."
    } else {
        "Inherited from the config file — this profile does not set it."
    });
    ui.end_row();
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
        // Secondary windows need a Context, not the panel's Ui — but this
        // eframe's App::ui hands us a Ui, and its Context exposes no screen
        // rect, so the height the window must fit inside comes from here.
        let avail_h = ui.available_height();
        let ctx = ui.ctx().clone();
        self.show_fitness_window(&ctx, avail_h);

        // While a job runs, repaint frequently for smooth progress; otherwise
        // just keep the resource monitor ticking.
        if self.job.running {
            ui.ctx().request_repaint_after(Duration::from_millis(250));
        } else {
            ui.ctx().request_repaint_after(Duration::from_secs(60));
        }

        // Scrollable: the panel's content (Explore / Evolve / Dedup /
        // Processes / System) is taller than the default window, and without
        // this everything below the fold is unreachable — which is how the
        // Stop button became impossible to press.
        egui::CentralPanel::default().show(ui, |ui| {
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
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
                if ui
                    .button("🎬  Review reels")
                    .on_hover_text("Watch the previews `explorer auto-reel` produced and approve \
                                    the ones worth rendering properly. Approving queues a \
                                    full-resolution render in the video queue.")
                    .clicked()
                {
                    self.spawn(sibling("nnfractals-reels"), &[], "reel review");
                }
                if ui
                    .button("📼  Video queue")
                    .on_hover_text("Open the video export queue: see what's Pending/Processing/\
                                    Done, and (🌙) hold new renders to a wall-clock window. A \
                                    cron job can drive the same queue headlessly overnight via \
                                    `explorer queue-run --until HH:MM`.")
                    .clicked()
                {
                    self.spawn(sibling("nnfractals-queue"), &[], "video queue");
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
                if ui
                    .button("⚖  Fitness…")
                    .on_hover_text("Edit, save and load named fitness profiles — the mix of \
                                    entropy / novelty / diversity / taste that evolution selects \
                                    on, plus the save-gate thresholds. Launch a run with one from \
                                    inside that window.")
                    .clicked()
                {
                    self.fp_refresh();
                    self.show_fitness = true;
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
            .with_inner_size([620.0, 620.0]),
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

#[cfg(test)]
mod fitness_ui_tests {
    use super::*;

    fn bufs(pairs: &[(&'static str, &str)]) -> HashMap<&'static str, String> {
        pairs.iter().map(|(k, v)| (*k, v.to_string())).collect()
    }

    #[test]
    fn an_empty_buffer_means_inherit_not_zero() {
        let p = profile_from_bufs("x", "", &bufs(&[("novelty_weight", "")]));
        assert_eq!(p.novelty_weight, None, "empty must not pin the weight to 0");
    }

    #[test]
    fn a_typo_means_inherit_and_is_reported() {
        let b = bufs(&[("novelty_weight", "0.6o"), ("ood_weight", "0.25")]);
        let p = profile_from_bufs("x", "", &b);
        assert_eq!(p.novelty_weight, None, "an unparseable value must not become 0");
        assert_eq!(p.ood_weight, Some(0.25), "a good value alongside it must still apply");
        assert_eq!(bad_fields(&b), vec!["Behavioural novelty"]);
    }

    #[test]
    fn an_explicit_zero_is_kept() {
        // Zeroing a term is a real choice — it must survive, unlike a blank.
        let p = profile_from_bufs("x", "", &bufs(&[("novelty_weight", "0")]));
        assert_eq!(p.novelty_weight, Some(0.0));
        assert!(bad_fields(&bufs(&[("novelty_weight", "0")])).is_empty());
    }

    #[test]
    fn whitespace_is_tolerated() {
        let p = profile_from_bufs("  spaced  ", "", &bufs(&[("ood_weight", "  0.5 ")]));
        assert_eq!(p.ood_weight, Some(0.5));
        assert_eq!(p.name, "spaced");
    }

    #[test]
    fn format_weight_is_short_and_round_trips() {
        assert_eq!(format_weight(0.6), "0.6");
        assert_eq!(format_weight(1.0), "1");
        assert_eq!(format_weight(0.012), "0.012");
        assert_eq!(format_weight(0.0), "0");
        assert_eq!(format_weight(-0.0), "0", "negative zero must not render as \"-\"");
        for v in [0.6f32, 1.0, 0.012, 3.0, 0.15, 58.0] {
            assert_eq!(format_weight(v).parse::<f32>().unwrap(), v, "{v} did not round-trip");
        }
    }

    #[test]
    fn discover_configs_never_returns_empty() {
        // A missing root must still give the caller something selectable.
        assert_eq!(discover_configs(Path::new("/nonexistent")), vec!["config.toml".to_string()]);
    }

    #[test]
    fn discover_configs_finds_the_real_config_files() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let found = discover_configs(root);
        assert!(found.contains(&"config.toml".to_string()), "got {found:?}");
        assert!(found.windows(2).all(|w| w[0] <= w[1]), "must be sorted: {found:?}");
    }
}

#[cfg(test)]
mod proc_matching_tests {
    use super::classify_proc;

    fn cmd(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn a_profile_launched_run_is_still_recognised() {
        // Carl, 2026-09-10: an evolution run had to be killed by hand. The
        // matching itself is fine — it keys on comm, so the --profile argument
        // added by the fitness feature cannot break it. Pinned so a future
        // change to cmdline-based matching fails here instead of in the field.
        assert_eq!(
            classify_proc("nnfractals", &cmd(&[
                "./target/release/nnfractals", "--config", "config.toml",
                "--profile", "Entropy only",
            ])),
            Some("evolution")
        );
    }

    #[test]
    fn a_plain_run_is_recognised() {
        assert_eq!(
            classify_proc("nnfractals", &cmd(&["./target/release/nnfractals"])),
            Some("evolution")
        );
    }

    #[test]
    fn the_scorer_sidecar_is_recognised_by_its_script() {
        assert_eq!(
            classify_proc("python3", &cmd(&["python3", "aesthetic_scorer.py"])),
            Some("scorer")
        );
    }

    #[test]
    fn the_gui_binaries_are_not_stopped() {
        // Stop must never take out the launcher, viewer, queue or browser —
        // their names all start with "nnfractals-".
        for n in ["nnfractals-launcher", "nnfractals-viewer", "nnfractals-queue", "nnfractals-browser",
                  "nnfractals-reels"] {
            assert_eq!(classify_proc(n, &cmd(&[n])), None, "{n} must be left alone");
        }
        assert_eq!(classify_proc("explorer", &cmd(&["./target/release/explorer"])), None);
    }
}
