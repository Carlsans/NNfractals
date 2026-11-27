//! Queue manager (nnfractals-queue): upload a finished render to YouTube.
//!
//! The upload itself is `scripts/youtube_upload.py` (YouTube Data API v3, resumable),
//! run with its own Python at `~/.local/share/nnfractals/youtube-venv` (installed from
//! here on first use). Google needs an OAuth client of type "Desktop app" from Google
//! Cloud Console (YouTube Data API v3 enabled), saved as
//! `~/.config/nnfractals/youtube_client_secret.json`; the first upload opens the
//! browser for the consent. An API project YouTube has not audited uploads as private,
//! whatever is asked. Uploaded videos are recorded on their queue item and in
//! `<queue dir>/youtube_uploads.jsonl` (kept after "Clean Queue").

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc;

use eframe::egui::{self, Color32};
use nnfractals::video_export::{queue_dir, QueueItem, YoutubeUpload};

const PRIVACY: [&str; 3] = ["private", "unlisted", "public"];
/// YouTube categories that fit: Film & Animation, Science & Technology, Education, Entertainment.
const CATEGORIES: [(&str, &str); 4] = [("1", "Film & Animation"), ("28", "Science & Technology"), ("27", "Education"), ("24", "Entertainment")];
const LIBS: [&str; 3] = ["google-api-python-client", "google-auth-oauthlib", "google-auth-httplib2"];

fn home() -> PathBuf { PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into())) }
fn venv_dir() -> PathBuf { home().join(".local/share/nnfractals/youtube-venv") }
fn venv_python() -> PathBuf { venv_dir().join("bin/python") }
fn secret_path() -> PathBuf { home().join(".config/nnfractals/youtube_client_secret.json") }
fn token_path() -> PathBuf { home().join(".config/nnfractals/youtube_token.json") }
fn settings_path() -> PathBuf { nnfractals::project_root().join("youtube.toml") }
fn history_path() -> PathBuf { queue_dir().join("youtube_uploads.jsonl") }
/// The upload script, next to the binary's project (or, for test binaries that run
/// one folder deeper, the source tree's).
fn script_path() -> PathBuf {
    let p = nnfractals::project_root().join("scripts/youtube_upload.py");
    if p.exists() { p } else { PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts/youtube_upload.py") }
}

/// The defaults the upload dialog starts from (`youtube.toml` in the project root),
/// updated with what was used at each upload.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct Settings {
    privacy: String,
    category: String,
    tags: String,
    /// `{label}` `{formula}` `{width}` `{height}` `{fps}` `{date}` are filled in.
    title: String,
    description: String,
    thumbnail: bool,
    /// Tick the 9:16 reel (as a Short) by default.
    short: bool,
    /// Tick the full-size video by default.
    #[serde(default = "yes")]
    full: bool,
}

fn yes() -> bool { true }

impl Default for Settings {
    fn default() -> Self {
        Settings {
            privacy: "private".into(),
            category: "1".into(),
            tags: "fractal, quaternion fractal, 4D, generative art, mathematics, NNfractals".into(),
            title: "{label}".into(),
            description: "A four-dimensional (quaternion) fractal, rendered with NNfractals.\n\nFormula: {formula}\n{width}×{height}, {fps} fps\n\n#fractal #generativeart #math".into(),
            thumbnail: true,
            short: true,
            full: true,
        }
    }
}

impl Settings {
    fn load() -> Self {
        std::fs::read_to_string(settings_path()).ok().and_then(|t| toml::from_str(&t).ok()).unwrap_or_default()
    }
    fn save(&self) {
        if let Ok(t) = toml::to_string(self) { let _ = std::fs::write(settings_path(), t); }
    }
}

/// An item's two videos, either of which may be missing: (the full-size render, its 9:16
/// reel). With a reel crop the queue records the REEL as the item's output and keeps the
/// full render beside it as `<name>.<ext>`; otherwise it records the full render, and a
/// reel, if any, is `<name>_reel.<ext>` beside it.
fn videos_of(output: &Path) -> (Option<PathBuf>, Option<PathBuf>) {
    let stem = output.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let ext = output.extension().and_then(|e| e.to_str()).unwrap_or("mp4");
    match stem.strip_suffix("_reel") {
        Some(base) => {
            let full = output.with_file_name(format!("{base}.{ext}"));
            (full.exists().then_some(full), output.exists().then(|| output.to_path_buf()))
        }
        None => (output.exists().then(|| output.to_path_buf()), reel_of(output)),
    }
}

/// Is this upload record a reel (by its flag, or by its file: before 2026-11-27 every
/// upload sent the item's recorded output, the reel, as a plain video)?
fn is_reel_upload(u: &YoutubeUpload) -> bool {
    u.short || Path::new(&u.file).file_stem().and_then(|s| s.to_str()).is_some_and(|s| s.ends_with("_reel"))
}

/// The 9:16 centre crop the queue writes beside a render (`<name>_reel.<ext>`), if any.
fn reel_of(video: &Path) -> Option<PathBuf> {
    let stem = video.file_stem()?.to_str()?;
    let ext = video.extension().and_then(|e| e.to_str()).unwrap_or("mp4");
    let p = video.with_file_name(format!("{stem}_reel.{ext}"));
    p.exists().then_some(p)
}

/// The item's formula, from the genome the queue kept.
fn formula_of(it: &QueueItem) -> String {
    nnfractals::io::load_genome(&queue_dir().join(&it.nn_filename)).ok()
        .map(|g| if g.formula_readable.is_empty() { g.formula_expr() } else { g.formula_readable.clone() })
        .unwrap_or_default()
}

fn fill(template: &str, it: &QueueItem, formula: &str) -> String {
    let (w, h, fps) = match &it.quat { Some(q) => (q.width, q.height, q.fps), None => (it.width, it.height, it.fps) };
    let date = chrono_like_date();
    template.replace("{label}", &it.genome_label).replace("{formula}", formula).replace("{width}", &w.to_string())
        .replace("{height}", &h.to_string()).replace("{fps}", &fps.to_string()).replace("{date}", &date)
}

/// Today as YYYY-MM-DD (from `date`, no extra dependency).
fn chrono_like_date() -> String {
    Command::new("date").arg("+%Y-%m-%d").output().ok().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default()
}

fn now_s() -> u64 { std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) }

/// A still from the middle of the video, 1280 wide, for the thumbnail.
fn middle_frame(video: &Path) -> Option<PathBuf> {
    let out = Command::new("ffprobe").args(["-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0"]).arg(video).output().ok()?;
    let dur: f64 = String::from_utf8_lossy(&out.stdout).trim().parse().ok()?;
    let jpg = std::env::temp_dir().join(format!("nnfractals_yt_thumb_{}.jpg", std::process::id()));
    let ok = Command::new("ffmpeg").args(["-y", "-v", "error", "-ss", &format!("{:.2}", dur * 0.5)]).arg("-i").arg(video)
        .args(["-frames:v", "1", "-vf", "scale=1280:-2", "-q:v", "3"]).arg(&jpg).status().ok()?.success();
    ok.then_some(jpg)
}

enum Event {
    Line(String),
    Uploaded(YoutubeUpload),
    Finished(Result<String, String>),
}

/// What the background thread does: install the uploader, or upload a list of files.
enum Work {
    Install,
    Upload { files: Vec<(PathBuf, bool, String)>, description: String, tags: String, privacy: String, category: String, thumbnail: bool },
}

struct Job {
    item_id: Option<String>,
    events: mpsc::Receiver<Event>,
    progress: f32,
    line: String,
    what: String,
}

struct Dialog {
    item_id: String,
    /// The full-size video and the reel, each sent on its own (either may be missing).
    full: Option<PathBuf>,
    reel: Option<PathBuf>,
    send_full: bool,
    send_reel: bool,
    /// Their YouTube videos already, if any.
    full_on_yt: Vec<YoutubeUpload>,
    reel_on_yt: Vec<YoutubeUpload>,
    title: String,
    reel_title: String,
    description: String,
    tags: String,
    privacy: String,
    category: String,
    thumbnail: bool,
}

#[derive(Default)]
pub struct YoutubePanel {
    settings: Option<Settings>,
    open_for: Option<String>,
    dialog: Option<Dialog>,
    job: Option<Job>,
    status: Option<Result<String, String>>,
}

fn run(work: Work, tx: mpsc::Sender<Event>) {
    let send_lines = |cmd: &mut Command, tx: &mpsc::Sender<Event>| -> Result<Vec<String>, String> {
        let mut child = cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).stdin(Stdio::null()).spawn().map_err(|e| e.to_string())?;
        let err = child.stderr.take().map(|e| std::thread::spawn(move || {
            BufReader::new(e).lines().map_while(Result::ok).collect::<Vec<_>>()
        }));
        let mut lines = Vec::new();
        if let Some(out) = child.stdout.take() {
            for l in BufReader::new(out).lines().map_while(Result::ok) {
                let _ = tx.send(Event::Line(l.clone()));
                lines.push(l);
            }
        }
        let status = child.wait().map_err(|e| e.to_string())?;
        let errs = err.and_then(|h| h.join().ok()).unwrap_or_default();
        if !status.success() && !lines.iter().any(|l| l.starts_with("ERROR")) {
            let tail: Vec<&String> = errs.iter().rev().take(3).collect();
            return Err(format!("exit {status}: {}", tail.into_iter().rev().cloned().collect::<Vec<_>>().join(" · ")));
        }
        Ok(lines)
    };
    let result = match work {
        Work::Install => {
            let _ = tx.send(Event::Line("STATUS creating the uploader's Python and installing Google's API client…".into()));
            let mut c = Command::new("bash");
            c.arg("-c").arg(format!("set -e; mkdir -p \"$(dirname '{v}')\"; [ -x '{v}/bin/python' ] || python3 -m venv '{v}'; '{v}/bin/pip' install -q --disable-pip-version-check {libs}; echo DONE installed",
                v = venv_dir().display(), libs = LIBS.join(" ")));
            send_lines(&mut c, &tx).and_then(|l| if l.iter().any(|x| x.starts_with("DONE")) { Ok("uploader installed".into()) } else { Err("pip did not finish".into()) })
        }
        Work::Upload { files, description, tags, privacy, category, thumbnail } => {
            let script = script_path();
            let mut done = Vec::new();
            let mut err = None;
            for (file, short, title) in files {
                let thumb = if thumbnail && !short { middle_frame(&file) } else { None };
                let mut c = Command::new(venv_python());
                c.arg(&script).arg("--file").arg(&file).args(["--title", &title, "--description", &description, "--tags", &tags, "--privacy", &privacy, "--category", &category]);
                if let Some(t) = &thumb { c.arg("--thumbnail").arg(t); }
                match send_lines(&mut c, &tx) {
                    Ok(lines) => {
                        if let Some(id) = lines.iter().rev().find_map(|l| l.strip_prefix("DONE ")) {
                            let up = YoutubeUpload { file: file.display().to_string(), video_id: id.trim().to_string(), privacy: privacy.clone(), uploaded_at: now_s(), short };
                            let _ = tx.send(Event::Uploaded(up.clone()));
                            done.push(up);
                        } else {
                            err = Some(lines.iter().rev().find_map(|l| l.strip_prefix("ERROR ")).unwrap_or("no video id came back").to_string());
                        }
                    }
                    Err(e) => err = Some(e),
                }
                if let Some(t) = thumb { let _ = std::fs::remove_file(t); }
                if err.is_some() { break; }
            }
            match err {
                Some(e) => Err(e),
                None => Ok(format!("uploaded: {}", done.iter().map(|u| u.url()).collect::<Vec<_>>().join("  "))),
            }
        }
    };
    let _ = tx.send(Event::Finished(result));
}

impl YoutubePanel {
    fn settings(&mut self) -> &mut Settings {
        self.settings.get_or_insert_with(Settings::load)
    }

    fn start(&mut self, work: Work, item_id: Option<String>, what: &str) {
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || run(work, tx));
        self.job = Some(Job { item_id, events: rx, progress: 0.0, line: "starting…".into(), what: what.into() });
        self.status = None;
    }

    /// The YouTube part of a finished item's row: its uploaded videos, the upload in
    /// progress, or the button.
    pub fn row(&mut self, ui: &mut egui::Ui, it: &QueueItem) {
        let Some(out) = it.output_path.as_ref().map(PathBuf::from) else { return };
        // The recorded output is the reel when the queue made one: link the full video too.
        let (full, reel) = videos_of(&out);
        if let (Some(f), Some(r)) = (&full, &reel) {
            if *r == out && ui.link("full 16:9").on_hover_text(format!("{} — the full-size render (the link before it is the 9:16 reel)", f.display())).clicked() {
                let _ = Command::new("xdg-open").arg(f).spawn();
            }
        }
        for u in &it.youtube {
            let label = format!("▶ {}{}", if is_reel_upload(u) { "Short " } else { "" }, u.video_id);
            if ui.link(label).on_hover_text(format!("{} ({}) — open on YouTube", u.url(), u.privacy)).clicked() {
                let _ = Command::new("xdg-open").arg(u.url()).spawn();
            }
        }
        if let Some(j) = self.job.as_ref().filter(|j| j.item_id.as_deref() == Some(it.id.as_str())) {
            ui.add(egui::ProgressBar::new(j.progress).desired_width(140.0).text(format!("YouTube {:.0}%", j.progress * 100.0)));
            return;
        }
        let label = if it.youtube.is_empty() { "⬆ YouTube" } else { "⬆ again" };
        if ui.add_enabled((full.is_some() || reel.is_some()) && self.job.is_none(), egui::Button::new(label))
            .on_hover_text("Upload the full-size video, the 9:16 reel as a Short, or both: title, description, tags and privacy first, then it all runs by itself").clicked() {
            self.open_for = Some(it.id.clone());
        }
    }

    /// The upload dialog and the running job. Returns true when `items` changed (an
    /// upload finished), so the caller saves the queue.
    pub fn show(&mut self, ctx: &egui::Context, items: &mut [QueueItem]) -> bool {
        let mut changed = false;
        // Open the dialog asked for by a row.
        if let Some(id) = self.open_for.take() {
            if let Some(it) = items.iter().find(|it| it.id == id) {
                let s = self.settings().clone();
                let formula = formula_of(it);
                let (full, reel) = videos_of(Path::new(&it.output_path.clone().unwrap_or_default()));
                let on_yt = |f: &Option<PathBuf>, reel_kind: bool| -> Vec<YoutubeUpload> {
                    let Some(f) = f else { return Vec::new() };
                    it.youtube.iter().filter(|u| Path::new(&u.file) == f.as_path() || (Path::new(&u.file).file_name() == f.file_name() && is_reel_upload(u) == reel_kind)).cloned().collect()
                };
                let (full_on_yt, reel_on_yt) = (on_yt(&full, false), on_yt(&reel, true));
                let title = fill(&s.title, it, &formula);
                self.dialog = Some(Dialog {
                    item_id: id,
                    // Ticked: what is there and not on YouTube yet (and wanted by default).
                    send_full: full.is_some() && full_on_yt.is_empty() && s.full,
                    send_reel: reel.is_some() && reel_on_yt.is_empty() && s.short,
                    full, reel, full_on_yt, reel_on_yt,
                    reel_title: format!("{title} #Shorts"), title,
                    description: fill(&s.description, it, &formula),
                    tags: s.tags, privacy: s.privacy, category: s.category, thumbnail: s.thumbnail,
                });
            }
        }
        // The running job.
        let mut finished = None;
        if let Some(job) = &mut self.job {
            while let Ok(ev) = job.events.try_recv() {
                match ev {
                    Event::Line(l) => {
                        if let Some(p) = l.strip_prefix("PROGRESS ") { job.progress = p.trim().parse::<f32>().unwrap_or(0.0) / 100.0; }
                        else if let Some(t) = l.strip_prefix("STATUS ").or_else(|| l.strip_prefix("WARN ")) { job.line = t.to_string(); }
                    }
                    Event::Uploaded(up) => {
                        if let Some(it) = job.item_id.as_ref().and_then(|id| items.iter_mut().find(|it| &it.id == id)) {
                            it.youtube.push(up.clone());
                            changed = true;
                        }
                        let rec = serde_json::json!({ "item": job.item_id, "file": up.file, "video_id": up.video_id, "url": up.url(), "privacy": up.privacy, "short": up.short, "uploaded_at": up.uploaded_at });
                        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(history_path()) {
                            use std::io::Write as _;
                            let _ = writeln!(f, "{rec}");
                        }
                    }
                    Event::Finished(r) => finished = Some(r),
                }
            }
            ctx.request_repaint_after(std::time::Duration::from_millis(300));
        }
        if let Some(r) = finished {
            self.status = Some(r);
            self.job = None;
        }
        self.dialog_window(ctx);
        changed
    }

    fn dialog_window(&mut self, ctx: &egui::Context) {
        let Some(d) = &mut self.dialog else { return };
        let mut open = true;
        let mut start = false;
        let mut install = false;
        let venv_ok = venv_python().exists();
        let secret_ok = secret_path().exists();
        let authorised = token_path().exists();
        let busy = self.job.is_some();
        egui::Window::new("⬆ Upload to YouTube").open(&mut open).resizable(true).default_width(560.0).show(ctx, |ui| {
            // Set-up, once.
            if !venv_ok {
                ui.horizontal_wrapped(|ui| {
                    ui.colored_label(Color32::from_rgb(230, 180, 80), "The uploader is not installed yet.");
                    install = ui.add_enabled(!busy, egui::Button::new("Install it")).on_hover_text(format!("A small Python of its own at {} with {}", venv_dir().display(), LIBS.join(", "))).clicked();
                });
            }
            if !secret_ok {
                ui.colored_label(Color32::from_rgb(230, 180, 80), "Google needs an OAuth client (once):");
                ui.label("1. console.cloud.google.com: a project, then \"YouTube Data API v3\" → Enable.");
                ui.label("2. Google Auth Platform: the consent screen (External), with your Google account as a test user; publish the app, or the login lapses every 7 days.");
                ui.label("3. Clients → Create client → Desktop app → download the JSON.");
                ui.horizontal(|ui| { ui.label("4. Save it as"); ui.monospace(secret_path().display().to_string()); });
                if ui.link("open Google Cloud Console").clicked() { let _ = Command::new("xdg-open").arg("https://console.cloud.google.com/apis/library/youtube.googleapis.com").spawn(); }
                ui.separator();
            } else if !authorised {
                ui.weak("The first upload opens the browser to sign in to Google and allow the upload.");
            }
            // Each video on its own: the full-size render, the 9:16 reel as a Short, or both.
            let name = |p: &Option<PathBuf>| p.as_ref().and_then(|p| p.file_name()).map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
            let already = |ui: &mut egui::Ui, ups: &[YoutubeUpload]| {
                for u in ups {
                    ui.colored_label(Color32::from_rgb(230, 180, 80), "already on YouTube:");
                    if ui.link(&u.video_id).on_hover_text(u.url()).clicked() { let _ = Command::new("xdg-open").arg(u.url()).spawn(); }
                }
            };
            ui.add_enabled_ui(d.full.is_some(), |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.checkbox(&mut d.send_full, "full-size video (16:9)");
                    ui.monospace(if d.full.is_some() { name(&d.full) } else { "(no full-size file)".into() });
                    already(ui, &d.full_on_yt);
                });
                if d.send_full {
                    ui.horizontal(|ui| { ui.label("    title"); ui.add(egui::TextEdit::singleline(&mut d.title).desired_width(f32::INFINITY).char_limit(100)); });
                }
            });
            ui.add_enabled_ui(d.reel.is_some(), |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.checkbox(&mut d.send_reel, "9:16 reel, as a Short");
                    ui.monospace(if d.reel.is_some() { name(&d.reel) } else { "(no reel: render with the reel crop to get one)".into() });
                    already(ui, &d.reel_on_yt);
                });
                if d.send_reel {
                    ui.horizontal(|ui| { ui.label("    title"); ui.add(egui::TextEdit::singleline(&mut d.reel_title).desired_width(f32::INFINITY).char_limit(100)); });
                }
            });
            ui.separator();
            ui.label("description");
            ui.add(egui::TextEdit::multiline(&mut d.description).desired_rows(6).desired_width(f32::INFINITY));
            ui.horizontal(|ui| { ui.label("tags"); ui.add(egui::TextEdit::singleline(&mut d.tags).desired_width(f32::INFINITY)).on_hover_text("comma-separated"); });
            ui.horizontal(|ui| {
                ui.label("privacy");
                for p in PRIVACY { ui.selectable_value(&mut d.privacy, p.to_string(), p); }
                ui.separator();
                egui::ComboBox::from_label("category").selected_text(CATEGORIES.iter().find(|c| c.0 == d.category).map_or(d.category.as_str(), |c| c.1)).show_ui(ui, |ui| {
                    for (id, name) in CATEGORIES { ui.selectable_value(&mut d.category, id.to_string(), name); }
                });
            });
            if d.privacy != "private" {
                ui.colored_label(Color32::from_rgb(230, 180, 80), "Until YouTube audits your API project, its uploads stay private whatever is chosen here.")
                    .on_hover_text("YouTube's rule for API projects created since July 2020: request the audit at support.google.com/youtube/contact/yt_api_form");
            }
            ui.add_enabled_ui(d.send_full, |ui| {
                ui.checkbox(&mut d.thumbnail, "full video: thumbnail from its middle").on_hover_text("Custom thumbnails need a phone-verified channel; without it the upload still works. Shorts take theirs from the video.");
            });
            ui.separator();
            let ready = (d.send_full && !d.title.trim().is_empty()) || (d.send_reel && !d.reel_title.trim().is_empty());
            let what = match (d.send_full, d.send_reel) { (true, true) => "⬆ Upload both", (true, false) => "⬆ Upload the full video", (false, true) => "⬆ Upload the reel", _ => "⬆ Upload" };
            ui.horizontal(|ui| {
                start = ui.add_enabled(venv_ok && secret_ok && !busy && ready, egui::Button::new(what)).clicked();
                if busy { ui.spinner(); }
            });
            if let Some(j) = &self.job {
                ui.add(egui::ProgressBar::new(j.progress).text(format!("{} {:.0}%", j.what, j.progress * 100.0)));
                ui.label(egui::RichText::new(&j.line).small().weak());
            }
            match &self.status {
                Some(Ok(m)) => { ui.colored_label(Color32::LIGHT_GREEN, m); }
                Some(Err(e)) => { ui.colored_label(Color32::LIGHT_RED, e); }
                None => {}
            }
        });
        if install {
            self.start(Work::Install, None, "installing");
            return;
        }
        if start {
            let d = self.dialog.as_ref().unwrap();
            let mut files = Vec::new();
            if d.send_full { if let Some(f) = &d.full { files.push((f.clone(), false, d.title.trim().to_string())); } }
            if d.send_reel { if let Some(r) = &d.reel { files.push((r.clone(), true, d.reel_title.trim().to_string())); } }
            let (privacy, category, tags, thumbnail, id) = (d.privacy.clone(), d.category.clone(), d.tags.clone(), d.thumbnail, d.item_id.clone());
            let (full, short) = (d.send_full, d.send_reel);
            let work = Work::Upload { files, description: d.description.clone(), tags: tags.clone(), privacy: privacy.clone(), category: category.clone(), thumbnail };
            // What was used becomes the next default (the title and description templates stay).
            let s = self.settings();
            s.privacy = privacy;
            s.category = category;
            s.tags = tags;
            s.thumbnail = thumbnail;
            s.short = short;
            s.full = full;
            s.save();
            self.start(work, Some(id), "uploading");
        }
        // Closing the dialog never stops an upload: its row in the queue shows the progress.
        if !open {
            self.dialog = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(label: &str) -> QueueItem {
        let v = nnfractals::video_export::CapturedView { cx: 0.0, cx_lo: 0.0, cy: 0.0, cy_lo: 0.0, zoom: 1.0, aspect: 1.0 };
        QueueItem {
            id: "t".into(), nn_filename: "missing.nn".into(), genome_label: label.into(), start: v.clone(), end: v,
            steps: 10, fps: 30, width: 3840, height: 2160, invert_coords: false, invert_range: false,
            colormap: String::new(), angle_coloring: false, output_dir: String::new(),
            status: nnfractals::video_export::QueueStatus::Done, output_path: None, error: None, created_at: 0,
            waypoints: Vec::new(), chain_label: None, time_mod: Vec::new(), time_frames: 0, blend_nn_filename: None,
            blend_shape: String::new(), blend_amp: 0.0, time_prog: Vec::new(), rife_fps: 0, keyframe_stride: 0, quat: None,
            youtube: Vec::new(),
        }
    }

    #[test]
    fn an_items_two_videos_are_found_whichever_is_recorded() {
        let dir = std::env::temp_dir().join(format!("nnf_yt_videos_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (full, reel) = (dir.join("clip.mp4"), dir.join("clip_reel.mp4"));
        std::fs::write(&full, b"x").unwrap();
        // Only the full render.
        assert_eq!(videos_of(&full), (Some(full.clone()), None));
        std::fs::write(&reel, b"x").unwrap();
        // The queue records the reel when it made one; the full render is beside it.
        assert_eq!(videos_of(&reel), (Some(full.clone()), Some(reel.clone())));
        assert_eq!(videos_of(&full), (Some(full.clone()), Some(reel.clone())));
        let up = |f: &Path, short: bool| YoutubeUpload { file: f.display().to_string(), video_id: "x".into(), privacy: "private".into(), uploaded_at: 0, short };
        assert!(is_reel_upload(&up(&reel, false)) && !is_reel_upload(&up(&full, false)) && is_reel_upload(&up(&full, true)));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn templates_fill_and_youtube_refuses_angle_brackets_downstream() {
        let it = item("trial17");
        let t = fill("{label} {width}x{height}@{fps} {formula}", &it, "z^2+c");
        assert_eq!(t, "trial17 3840x2160@30 z^2+c");
    }

    #[test]
    fn a_dry_run_upload_records_the_video_and_its_short() {
        if !venv_python().exists() { return; } // the uploader is not installed on this machine
        let dir = std::env::temp_dir().join(format!("nnf_yt_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let video = dir.join("clip.mp4");
        // One second of a test pattern, and its "reel" beside it.
        let ok = Command::new("ffmpeg").args(["-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=10", "-t", "1", "-pix_fmt", "yuv420p"]).arg(&video).status().map(|s| s.success()).unwrap_or(false);
        if !ok { return; } // no ffmpeg
        std::fs::copy(&video, dir.join("clip_reel.mp4")).unwrap();
        assert_eq!(reel_of(&video), Some(dir.join("clip_reel.mp4")));
        let thumb = middle_frame(&video).expect("a still from the middle of the video");
        assert!(std::fs::metadata(&thumb).unwrap().len() > 0);
        let _ = std::fs::remove_file(thumb);
        // SAFETY: tests in this binary do not read this variable concurrently.
        unsafe { std::env::set_var("NNFRACTALS_YOUTUBE_DRY_RUN", "1"); }
        let (tx, rx) = mpsc::channel();
        run(Work::Upload { files: vec![(video.clone(), false, "t".into()), (dir.join("clip_reel.mp4"), true, "t #Shorts".into())],
                           description: "d".into(), tags: "a, b".into(), privacy: "unlisted".into(), category: "1".into(), thumbnail: true }, tx);
        let events: Vec<Event> = rx.try_iter().collect();
        let ups: Vec<&YoutubeUpload> = events.iter().filter_map(|e| if let Event::Uploaded(u) = e { Some(u) } else { None }).collect();
        let said: Vec<String> = events.iter().map(|e| match e { Event::Line(l) => l.clone(), Event::Uploaded(u) => format!("uploaded {}", u.video_id), Event::Finished(r) => format!("finished {r:?}") }).collect();
        assert_eq!(ups.len(), 2, "{said:?}");
        assert!(!ups[0].short && ups[1].short && ups[0].privacy == "unlisted" && ups[0].video_id.starts_with("dryrun"));
        assert!(ups[1].url().contains("/shorts/"));
        assert!(matches!(events.last(), Some(Event::Finished(Ok(_)))));
        let _ = std::fs::remove_dir_all(dir);
    }
}
