//! Anim viewer ✎ Plot window: the pen-plotter "anatomy" of the slice the ✂ Slice window
//! shows. A quick preview (seconds) and the full SVG + G-code export both run `explorer
//! quat-anatomy` on the viewer's own state (the genome with any unsaved formula edits,
//! the timeline and the playhead), so the plot is exactly this slice. It can also show
//! what any G-code file draws (`explorer gcode-preview`).

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::Instant;

use eframe::egui;

/// What the window needs from the viewer when it starts a job.
pub struct PlotInputs {
    pub genome: nnfractals::genome::Genome,
    pub timeline: nnfractals::anim_timeline::AnimationTimeline,
    pub at_s: f64,
    pub spec: nnfractals::view_slice::SliceSpec,
    pub max_iter: u32,
    pub label: String,
    /// The genome's file (where the gallery looks for its plot thumbnail).
    pub genome_path: PathBuf,
}

/// Something the window asks the viewer to do.
pub enum PlotAction {
    None,
    OpenSlice,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Compose {
    /// Self-subdivided mosaic: equal-area patches, each with one layer.
    Mosaic,
    /// One small tile per layer, the same view in each.
    Atlas,
}

/// Mosaic densities: label, outside bands × sectors, inside bands × sectors.
const DENSITY: [(&str, usize, usize, usize, usize); 4] = [
    ("6200 patches (up to ~3000 layers)", 64, 48, 56, 56),
    ("2000 patches (~1500 layers)", 32, 32, 32, 32),
    ("1000 patches (~700 layers)", 24, 20, 24, 24),
    ("250 patches (~200 layers)", 12, 10, 12, 12),
];
/// Atlas grids (columns × rows).
const ATLAS: [(usize, usize); 4] = [(8, 8), (12, 12), (16, 16), (23, 23)];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum JobKind { Preview, Export, Gcode }

struct Job {
    kind: JobKind,
    child: Child,
    lines: mpsc::Receiver<String>,
    started: Instant,
    /// The PNG it will show: the preview image, or the export's preview.png.
    png: PathBuf,
    /// The export's folder.
    dir: Option<PathBuf>,
    progress: f32,
    last: String,
    log: Vec<String>,
}

pub struct PlotPanel {
    pub open: bool,
    compose: Compose,
    follow: f64,
    density: usize,
    atlas: usize,
    draft: bool,
    machine_carl: bool,
    auto: bool,
    tex: Option<egui::TextureHandle>,
    /// The PNG behind the picture shown.
    shown_png: Option<PathBuf>,
    shown: String,
    note: String,
    /// What the shown preview was made from (viewer state + plot settings).
    key: String,
    changed_at: Option<Instant>,
    job: Option<Job>,
    status: Option<Result<String, String>>,
    last_export: Option<PathBuf>,
    gcode_path: String,
}

impl Default for PlotPanel {
    fn default() -> Self {
        PlotPanel {
            open: false, compose: Compose::Mosaic, follow: 0.5, density: 0, atlas: 1, draft: false, machine_carl: true, auto: false,
            tex: None, shown_png: None, shown: String::new(), note: String::new(), key: String::new(), changed_at: None, job: None, status: None,
            last_export: None, gcode_path: String::new(),
        }
    }
}

fn root() -> PathBuf { nnfractals::project_root() }

fn timestamp() -> String {
    let s = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    format!("{s}")
}

/// A file name piece from a label (genome names can be long and odd).
fn short(label: &str) -> String {
    let s: String = label.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' }).collect();
    s.chars().take(48).collect()
}

/// Runs `explorer` with `args` (from the project root, at low priority, the export
/// under a 12 GB memory cap when systemd is there), its output lines on a channel.
fn spawn_explorer(args: &[String], cap_memory: bool) -> Result<(Child, mpsc::Receiver<String>), String> {
    let bin = nnfractals::locate_bin("explorer");
    let mut cmd = if cap_memory && Path::new("/usr/bin/systemd-run").exists() {
        let mut c = Command::new("systemd-run");
        c.args(["--user", "--scope", "-q", "-p", "MemoryMax=12G", "-p", "MemorySwapMax=2G", "nice", "-n", "10"]).arg(&bin);
        c
    } else {
        let mut c = Command::new("nice");
        c.args(["-n", "10"]).arg(&bin);
        c
    };
    cmd.args(args).current_dir(root()).stdout(Stdio::piped()).stderr(Stdio::piped()).stdin(Stdio::null());
    let mut child = cmd.spawn().map_err(|e| format!("couldn't start {}: {e}", bin.display()))?;
    let (tx, rx) = mpsc::channel();
    for stream in [child.stdout.take().map(|s| Box::new(s) as Box<dyn std::io::Read + Send>), child.stderr.take().map(|s| Box::new(s) as Box<dyn std::io::Read + Send>)].into_iter().flatten() {
        let tx = tx.clone();
        std::thread::spawn(move || {
            // Keep reading after the window lets go (an export left running in the
            // background): a closed pipe would kill the explorer mid-write.
            for line in BufReader::new(stream).lines().map_while(Result::ok) {
                let _ = tx.send(line);
            }
        });
    }
    Ok((child, rx))
}

/// A genome's plot thumbnail in the gallery (nnfractals-browser): `<folder>/_plots/<name>.png`.
fn plot_thumb_path(nn: &Path) -> PathBuf {
    let dir = nn.parent().unwrap_or(Path::new("."));
    dir.join("_plots").join(format!("{}.png", nn.file_stem().and_then(|s| s.to_str()).unwrap_or("genome")))
}

/// `src` scaled to the gallery's largest thumbnail (320 px) and saved at `dst`.
fn save_thumb(src: &Path, dst: &Path) -> Result<(), String> {
    let img = image::open(src).map_err(|e| format!("{}: {e}", src.display()))?;
    let small = img.resize(320, 320, image::imageops::FilterType::Triangle);
    if let Some(d) = dst.parent() { std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?; }
    let tmp = dst.with_extension("tmp.png");
    small.save(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, dst).map_err(|e| format!("{}: {e}", dst.display()))
}

fn load_png(ctx: &egui::Context, path: &Path) -> Result<egui::TextureHandle, String> {
    let img = image::open(path).map_err(|e| format!("{}: {e}", path.display()))?.to_rgb8();
    let size = [img.width() as usize, img.height() as usize];
    Ok(ctx.load_texture("plot_preview", egui::ColorImage::from_rgb(size, img.as_raw()), egui::TextureOptions::LINEAR))
}

impl PlotPanel {
    /// The settings part of the preview key.
    fn settings_key(&self) -> String {
        format!("{:?}|{}|{}|{}|{}", self.compose, self.follow, self.density, self.atlas, self.draft)
    }

    /// `quat-anatomy` arguments for the viewer's state (written to files under
    /// explorer_out/plots/viewer/) and this window's settings.
    fn anatomy_args(&self, inp: &PlotInputs) -> Result<Vec<String>, String> {
        let dir = root().join("explorer_out/plots/viewer");
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let gpath = dir.join(format!("state_{}.nn", std::process::id()));
        let tpath = dir.join(format!("state_{}_timeline.json", std::process::id()));
        nnfractals::io::save_genome(&inp.genome, &gpath).map_err(|e| format!("{}: {e}", gpath.display()))?;
        let tl = serde_json::to_string(&inp.timeline).map_err(|e| e.to_string())?;
        std::fs::write(&tpath, tl).map_err(|e| format!("{}: {e}", tpath.display()))?;
        let sp = &inp.spec;
        let mut a: Vec<String> = vec![
            "quat-anatomy".into(), "--genome".into(), gpath.display().to_string(), "--timeline".into(), tpath.display().to_string(),
            "--at-s".into(), format!("{}", inp.at_s), "--plane".into(), sp.base.label().to_string(),
            "--tilt-right".into(), format!("{}", sp.tilt_right), "--tilt-up".into(), format!("{}", sp.tilt_up),
            "--spin".into(), format!("{}", sp.spin), "--offset".into(), format!("{}", sp.offset), "--zoom".into(), format!("{}", sp.zoom),
            "--max-iter".into(), format!("{}", inp.max_iter.max(20)),
        ];
        match self.compose {
            Compose::Mosaic => {
                let (_, b, s, ib, is) = DENSITY[self.density];
                a.extend(["--compose", "mosaic", "--levels", "2048", "--min-layer-m", "0.05", "--min-fill", "0.03"].map(String::from));
                a.extend(["--bands".into(), b.to_string(), "--sectors".into(), s.to_string(), "--inner-bands".into(), ib.to_string(), "--inner-sectors".into(), is.to_string()]);
                a.extend(["--follow".into(), format!("{}", self.follow)]);
            }
            Compose::Atlas => {
                let (c, r) = ATLAS[self.atlas];
                a.extend(["--compose".into(), "plate".into(), "--tiles".into(), format!("{c}x{r}"), "--gutter".into(), "3".into()]);
            }
        }
        Ok(a)
    }

    fn start_preview(&mut self, inp: &PlotInputs) {
        let png = root().join(format!("explorer_out/plots/viewer/preview_{}.png", std::process::id()));
        let mut args = match self.anatomy_args(inp) { Ok(a) => a, Err(e) => { self.status = Some(Err(e)); return; } };
        args.extend(["--preview".into(), png.display().to_string(), "--preview-px".into(), "1000".into()]);
        if self.draft { args.extend(["--cell".into(), "2".into()]); }
        self.run(JobKind::Preview, args, png, None, false);
    }

    fn start_export(&mut self, inp: &PlotInputs) {
        let sp = &inp.spec;
        let dir = root().join(format!("explorer_out/plots/{}_{}_{}_{}", short(&inp.label), sp.base.label(),
            match self.compose { Compose::Mosaic => "mosaic", Compose::Atlas => "atlas" }, timestamp()));
        let mut args = match self.anatomy_args(inp) { Ok(a) => a, Err(e) => { self.status = Some(Err(e)); return; } };
        args.extend(["--out-dir".into(), dir.display().to_string()]);
        if self.machine_carl { args.extend(["--machine".into(), "carl".into()]); }
        self.run(JobKind::Export, args, dir.join("preview.png"), Some(dir), true);
    }

    fn start_gcode(&mut self, file: &str) {
        let png = root().join(format!("explorer_out/plots/viewer/gcode_{}.png", std::process::id()));
        if let Some(d) = png.parent() { let _ = std::fs::create_dir_all(d); }
        let args = vec!["gcode-preview".into(), "--gcode".into(), file.to_string(), "--out".into(), png.display().to_string(), "--px".into(), "1000".into()];
        self.run(JobKind::Gcode, args, png, None, false);
    }

    fn run(&mut self, kind: JobKind, args: Vec<String>, png: PathBuf, dir: Option<PathBuf>, cap: bool) {
        if let Some(mut j) = self.job.take() { let _ = j.child.kill(); let _ = j.child.wait(); }
        match spawn_explorer(&args, cap) {
            Ok((child, lines)) => {
                self.job = Some(Job { kind, child, lines, started: Instant::now(), png, dir, progress: 0.0, last: "starting…".into(), log: Vec::new() });
                self.status = None;
            }
            Err(e) => self.status = Some(Err(e)),
        }
    }

    /// Reads the running job's output; when it ends, shows its picture.
    fn poll(&mut self, ctx: &egui::Context) {
        let Some(job) = &mut self.job else { return };
        while let Ok(line) = job.lines.try_recv() {
            if let Some(rest) = line.trim().strip_prefix("scoring block ") {
                // "scoring block 3/11: …"
                if let Some((a, b)) = rest.split(':').next().and_then(|f| f.split_once('/')) {
                    if let (Ok(a), Ok(b)) = (a.trim().parse::<f32>(), b.trim().parse::<f32>()) { job.progress = 0.75 * a / b.max(1.0); }
                }
            } else if line.contains("orbits at") || line.contains("orbits (") { job.progress = job.progress.max(0.8); }
            else if line.contains("traced") { job.progress = job.progress.max(0.9); }
            if !line.trim().is_empty() && !line.starts_with("[gpu]") {
                job.last = line.trim().chars().take(140).collect();
                job.log.push(line);
                if job.log.len() > 400 { job.log.drain(..100); }
            }
        }
        let Ok(Some(exit)) = job.child.try_wait() else { return };
        // Let the reader threads hand over the last lines.
        std::thread::sleep(std::time::Duration::from_millis(30));
        while let Ok(line) = job.lines.try_recv() { if !line.trim().is_empty() { job.log.push(line); } }
        let job = self.job.take().unwrap();
        let secs = job.started.elapsed().as_secs_f64();
        if !exit.success() {
            let tail: Vec<&str> = job.log.iter().rev().take(4).map(|s| s.as_str()).collect::<Vec<_>>().into_iter().rev().collect();
            self.status = Some(Err(format!("{} failed ({exit}) after {secs:.0} s: {}", match job.kind { JobKind::Preview => "preview", JobKind::Export => "export", JobKind::Gcode => "G-code preview" }, tail.join(" · "))));
            return;
        }
        match load_png(ctx, &job.png) {
            Ok(t) => { self.tex = Some(t); self.shown_png = Some(job.png.clone()); }
            Err(e) => { self.status = Some(Err(e)); return; }
        }
        let find = |pat: &str| job.log.iter().rev().find(|l| l.contains(pat)).cloned();
        match job.kind {
            JobKind::Preview => {
                // "PREVIEW <png> | 433 m of line | 34.9% inside | 12 s"
                let info = find("PREVIEW ").and_then(|l| l.split_once(" | ").map(|(_, r)| r.to_string())).unwrap_or_default();
                self.shown = "quick preview".into();
                self.note = format!("{info} · one family in six scored, so the layer count is not the final one");
            }
            JobKind::Export => {
                let layers = find("layers with at least").unwrap_or_default();
                let draw = find(" paths, ").and_then(|l| l.split_once("; draw ").map(|(_, r)| r.split(" → ").next().unwrap_or("").to_string())).unwrap_or_default();
                let dir = job.dir.clone().unwrap_or_default();
                self.shown = format!("export: {}", dir.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default());
                let ink = find("ink: ").map(|l| l.trim().to_string()).unwrap_or_default();
                self.note = format!("{} · draw {draw} · {ink}", layers.trim());
                self.status = Some(Ok(format!("saved in {} ({secs:.0} s)", dir.display())));
                self.last_export = Some(dir);
            }
            JobKind::Gcode => {
                let pens: Vec<String> = job.log.iter().filter(|l| l.starts_with("pen ")).map(|l| l.split_whitespace().take(5).collect::<Vec<_>>().join(" ")).collect();
                self.shown = format!("G-code: {}", Path::new(&self.gcode_path).file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default());
                self.note = pens.join(" · ");
            }
        }
    }

    /// The window. `state_key` changes whenever the viewer's slice changes; `inputs`
    /// is only called when a job starts.
    pub fn show(&mut self, ctx: &egui::Context, state_key: &str, inputs: &dyn Fn() -> PlotInputs) -> PlotAction {
        let mut action = PlotAction::None;
        if !self.open {
            return action;
        }
        self.poll(ctx);
        let key = format!("{state_key}|{}", self.settings_key());
        if key != self.key {
            self.key = key;
            self.changed_at = Some(Instant::now());
        }
        // A first look when the window opens, and (with "auto") a new one a moment
        // after the slice or the settings stop changing.
        let busy = self.job.is_some();
        let settled = self.changed_at.is_some_and(|t| t.elapsed().as_secs_f64() > 1.2);
        if !busy && settled && (self.auto || self.tex.is_none()) && self.status.as_ref().is_none_or(|s| s.is_ok()) {
            self.changed_at = None;
            self.start_preview(&inputs());
        }
        let mut open = self.open;
        egui::Window::new("\u{270e} Plot").open(&mut open).resizable(true).default_width(560.0).show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label("Plots the slice of the");
                if ui.link("\u{2702} Slice window").on_hover_text("Plane, tilt, spin, position and zoom of the plot are the Slice window's").clicked() {
                    action = PlotAction::OpenSlice;
                }
                ui.label("on Carl's 894 × 865 mm machine, 6 pens.");
            });
            ui.horizontal(|ui| {
                ui.label("layout");
                ui.selectable_value(&mut self.compose, Compose::Mosaic, "mosaic").on_hover_text("The fractal cuts itself into equal-area patches (escape-time bands × angle sectors outside, closest-approach bands × cycle-angle sectors inside); every patch shows one of 4470 orbit layers");
                ui.selectable_value(&mut self.compose, Compose::Atlas, "atlas").on_hover_text("One small tile per layer, the same view in each, the last tile a mosaic");
            });
            match self.compose {
                Compose::Mosaic => {
                    egui::ComboBox::from_label("density").selected_text(DENSITY[self.density].0).show_ui(ui, |ui| {
                        for (i, d) in DENSITY.iter().enumerate() { ui.selectable_value(&mut self.density, i, d.0); }
                    });
                    ui.add(egui::Slider::new(&mut self.follow, 0.0..=1.5).text("follow the outline").custom_formatter(|v, _| if v == 0.0 { "off".into() } else { format!("{v:.2}") }))
                        .on_hover_text("> 0: every tile's lines run along the fractal's own bands (equipotentials outside, closest-approach layers inside), bent by up to this many band widths by the tile's layer. Off: each tile draws its layer's own contours.");
                }
                Compose::Atlas => {
                    egui::ComboBox::from_label("tiles").selected_text(format!("{} × {}", ATLAS[self.atlas].0, ATLAS[self.atlas].1)).show_ui(ui, |ui| {
                        for (i, (c, r)) in ATLAS.iter().enumerate() { ui.selectable_value(&mut self.atlas, i, format!("{c} × {r} ({} layers)", c * r - 1)); }
                    });
                }
            }
            ui.separator();
            ui.horizontal(|ui| {
                let busy = self.job.is_some();
                if ui.add_enabled(!busy, egui::Button::new("\u{1f441} Preview")).on_hover_text("A quick look: 1 mm sampling (2 mm with draft), one layer in six scored; about 10 s").clicked() {
                    self.start_preview(&inputs());
                }
                ui.checkbox(&mut self.draft, "draft").on_hover_text("2 mm sampling: about twice as fast, misses the finest fragments");
                ui.checkbox(&mut self.auto, "auto").on_hover_text("Preview again a moment after the slice or the settings change");
                ui.separator();
                if ui.add_enabled(!busy, egui::Button::new("\u{1f4be} Export SVG + G-code")).on_hover_text("The full plot: 0.35 mm sampling, every layer; 7–15 min and up to ~6 GB, under a 12 GB cap. SVG, Marlin G-code (all pens, and one file per pen), a summary and preview.png in explorer_out/plots/").clicked() {
                    self.start_export(&inputs());
                }
                ui.checkbox(&mut self.machine_carl, "Carl's machine").on_hover_text("G-code in the gcodefractal plotter's dialect: Z0 draws, lifts 2.5 / 7 mm, pen change at the centre on M226, no Y flip");
            });
            let mut cancel = false;
            if let Some(job) = &self.job {
                ui.horizontal(|ui| {
                    let what = match job.kind { JobKind::Preview => "preview", JobKind::Export => "export", JobKind::Gcode => "G-code" };
                    ui.add(egui::ProgressBar::new(job.progress).desired_width(220.0).text(format!("{what} {:.0} s", job.started.elapsed().as_secs_f64())));
                    cancel = ui.button("cancel").clicked();
                });
                ui.label(egui::RichText::new(&job.last).small().weak());
            }
            if cancel {
                if let Some(mut j) = self.job.take() { let _ = j.child.kill(); let _ = j.child.wait(); }
            }
            match &self.status {
                Some(Ok(m)) => { ui.colored_label(egui::Color32::from_rgb(120, 200, 120), m); }
                Some(Err(e)) => { ui.colored_label(egui::Color32::RED, e); }
                None => {}
            }
            // The picture.
            if let Some(t) = &self.tex {
                ui.label(egui::RichText::new(format!("{} — {}", self.shown, self.note)).small());
                let w = ui.available_width().clamp(240.0, 900.0);
                let sz = t.size_vec2();
                let h = w * sz.y / sz.x.max(1.0);
                ui.add(egui::Image::new(egui::load::SizedTexture::new(t.id(), egui::vec2(w, h))));
            } else if self.job.is_some() {
                ui.weak("the first preview is on its way…");
            }
            ui.separator();
            ui.horizontal(|ui| {
                if let Some(png) = self.shown_png.clone() {
                    if ui.add_enabled(self.tex.is_some(), egui::Button::new("\u{1f5bc} Use as gallery thumbnail"))
                        .on_hover_text("Saves the picture shown as this genome's plot thumbnail (_plots/ next to its .nn), which the gallery shows in its ✎ plot view").clicked() {
                        let dst = plot_thumb_path(&inputs().genome_path);
                        self.status = Some(save_thumb(&png, &dst).map(|_| format!("gallery thumbnail saved: {}", dst.display())));
                    }
                }
                if let Some(dir) = self.last_export.clone() {
                    if ui.button("\u{1f4c2} Open export folder").clicked() { let _ = Command::new("xdg-open").arg(&dir).spawn(); }
                    if ui.add_enabled(self.job.is_none(), egui::Button::new("check its G-code")).on_hover_text("Draws the exported G-code itself (all pens), exactly as the machine will").clicked() {
                        let gc = std::fs::read_dir(&dir).ok().and_then(|rd| rd.filter_map(|e| e.ok()).map(|e| e.path())
                            .find(|p| p.to_string_lossy().ends_with("_g5.gcode")));
                        if let Some(p) = gc { self.gcode_path = p.display().to_string(); let s = self.gcode_path.clone(); self.start_gcode(&s); }
                        else { self.status = Some(Err(format!("no *_g5.gcode in {}", dir.display()))); }
                    }
                }
            });
            ui.horizontal(|ui| {
                ui.label("G-code file");
                ui.add(egui::TextEdit::singleline(&mut self.gcode_path).desired_width(300.0).hint_text("path to any .gcode (gcodefractal's too)"));
                if ui.add_enabled(self.job.is_none() && !self.gcode_path.trim().is_empty(), egui::Button::new("show")).on_hover_text("Draws what the file draws: pen-down moves, one colour per pen").clicked() {
                    let s = self.gcode_path.trim().to_string();
                    self.start_gcode(&s);
                }
            });
        });
        self.open = open;
        if !open {
            if let Some(mut j) = self.job.take() {
                // An export keeps going in the background; a preview is dropped.
                if j.kind == JobKind::Export { std::thread::spawn(move || { let _ = j.child.wait(); }); } else { let _ = j.child.kill(); let _ = j.child.wait(); }
            }
        }
        if self.job.is_some() || self.changed_at.is_some() {
            ctx.request_repaint_after(std::time::Duration::from_millis(150));
        }
        action
    }
}
