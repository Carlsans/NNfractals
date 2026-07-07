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
use std::path::{Path, PathBuf};
use std::process::Command;
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
}

impl App {
    fn new() -> Self {
        App {
            root: project_root(),
            instances: 2,
            viewer_path: String::new(),
            status: String::new(),
            sys: System::new(),
            procs: Vec::new(),
            gpu_line: String::new(),
            last_refresh: None,
        }
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

    /// Train the human-preference model from the browser's ratings, then score the
    /// galleries. Long-running, so it's spawned detached with output to train_pref.log.
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
        let dirs: Vec<&str> = ["fractals_1", "fractals_2", "Starred", "train_corpus", "fractals_dag", "fractals"]
            .iter()
            .copied()
            .filter(|d| self.root.join(d).is_dir())
            .collect();
        if dirs.is_empty() {
            self.status = "no gallery folders found to score".into();
            return;
        }
        let log_path = self.root.join("train_pref.log");
        let log = match std::fs::File::create(&log_path) {
            Ok(f) => f,
            Err(e) => { self.status = format!("cannot write train_pref.log: {e}"); return; }
        };
        let errlog = log.try_clone().ok();
        let mut cmd = Command::new("python3");
        cmd.arg("scripts/train_pref.py")
            .arg("--ratings").arg(&ratings)
            .arg("--dirs").args(&dirs)
            .current_dir(&self.root)
            .stdout(std::process::Stdio::from(log));
        if let Some(e) = errlog {
            cmd.stderr(std::process::Stdio::from(e));
        }
        match cmd.spawn() {
            Ok(_) => self.status = format!(
                "training taste model in background (embeds all fractals; a few min)\n\
                 watch: {}", log_path.display()),
            Err(e) => self.status = format!("could not start training: {e}"),
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // Refresh the resource monitor at most every 60s; schedule a wake-up so
        // it keeps updating even when the window is idle.
        let stale = self.last_refresh.map_or(true, |t| t.elapsed() >= Duration::from_secs(60));
        if stale {
            self.refresh_procs();
        }
        ui.ctx().request_repaint_after(Duration::from_secs(60));

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
                    .button("🎓  Train taste model")
                    .on_hover_text("Train the preference model from your browser ratings\n\
                                     (ratings.jsonl) and score every fractal → pref_score.\n\
                                     Evolution then selects on your taste (optimization.pref_weight).")
                    .clicked()
                {
                    self.train_pref();
                }
            });

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
