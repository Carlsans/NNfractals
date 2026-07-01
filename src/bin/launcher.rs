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

use std::path::{Path, PathBuf};
use std::process::Command;

use clap::Parser;
use eframe::egui::{self, Color32};

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

// ── Application ────────────────────────────────────────────────────────────────

struct App {
    root: PathBuf,
    instances: u32,
    viewer_path: String,
    status: String,
}

impl App {
    fn new() -> Self {
        App {
            root: project_root(),
            instances: 2,
            viewer_path: String::new(),
            status: String::new(),
        }
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
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
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
                    .on_hover_text("Launch N background evolution instances (run.sh)")
                    .clicked()
                {
                    let n = self.instances.to_string();
                    self.run_sh(&[n.as_str()]);
                }
                if ui.button("■  Stop").clicked() {
                    self.run_sh(&["stop"]);
                }
                if ui.button("↻  Status").clicked() {
                    self.run_sh(&["status"]);
                }
            });
            ui.label(
                egui::RichText::new("Instances share one gallery and log to evolution.log.")
                    .weak(),
            );

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
        Box::new(|_cc| Ok(Box::new(App::new()))),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(())
}
