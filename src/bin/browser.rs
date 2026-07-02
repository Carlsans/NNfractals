//! nnfractals-browser — a sortable, thumbnailed gallery for evolved `.nn` fractals.
//!
//! Browse a folder of genome files by their metrics (Excel-style column sort),
//! cull junk (permanent delete), organise (move/copy), hand-pick breeding stock
//! (copy into a breed folder that a config's `save_dir` can later point at), and
//! open any fractal in the single-instance viewer.
//!
//! Columns are NOT hardcoded: each `.nn` is parsed as generic JSON and every
//! scalar key becomes a sortable column (arrays contribute a `<key>.len` count),
//! so new `Genome` metric fields appear automatically with no code change here.
//!
//! Run:  cargo run --features browser --bin nnfractals-browser -- [--dir <folder>]

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread;

use clap::Parser;
use eframe::egui::{self, Color32, ColorImage, TextureHandle, TextureOptions};
use egui_extras::{Column, TableBuilder};
use rand::Rng;
use serde::{Deserialize, Serialize};

use nnfractals::config::Config;

const PREFS_FILE: &str = "browser_prefs.toml";
const MIN_THUMB: u32 = 40;
const MAX_THUMB: u32 = 320;
const ZOOM_STEP: i32 = 16;

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "nnfractals-browser", about = "Browse & curate evolved fractals")]
struct Args {
    /// Folder of `.nn` files to browse (default: last used, else config save_dir)
    #[arg(long)]
    dir: Option<PathBuf>,
}

// ── A single metric value, schema-agnostic ─────────────────────────────────────

#[derive(Clone, Debug)]
enum Cell {
    Num(f64),
    Bool(bool),
    Text(String),
}

impl Cell {
    fn display(&self) -> String {
        match self {
            Cell::Num(n) => {
                if n.fract() == 0.0 && n.abs() < 1e15 {
                    format!("{}", *n as i64)
                } else {
                    format!("{n:.4}")
                }
            }
            Cell::Bool(b) => b.to_string(),
            Cell::Text(s) => s.clone(),
        }
    }
    fn type_rank(&self) -> u8 {
        match self {
            Cell::Num(_) => 0,
            Cell::Bool(_) => 1,
            Cell::Text(_) => 2,
        }
    }
}

/// Compare two present cells (numeric by value, text lexically, bool false<true;
/// mismatched types fall back to a stable type ordering).
fn cmp_present(x: &Cell, y: &Cell) -> Ordering {
    match (x, y) {
        (Cell::Num(a), Cell::Num(b)) => a.partial_cmp(b).unwrap_or(Ordering::Equal),
        (Cell::Bool(a), Cell::Bool(b)) => a.cmp(b),
        (Cell::Text(a), Cell::Text(b)) => a.cmp(b),
        _ => x.type_rank().cmp(&y.type_rank()),
    }
}

/// Full comparison with missing-last semantics; `desc` flips only present-vs-present.
fn cmp_cells(a: Option<&Cell>, b: Option<&Cell>, desc: bool) -> Ordering {
    match (a, b) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Greater, // missing sorts last
        (Some(_), None) => Ordering::Less,
        (Some(x), Some(y)) => {
            let o = cmp_present(x, y);
            if desc { o.reverse() } else { o }
        }
    }
}

// ── A row = one genome file ────────────────────────────────────────────────────

struct Row {
    nn_path: PathBuf,
    png_path: PathBuf,
    cells: BTreeMap<String, Cell>,
}

enum LoadMsg {
    Row(Row),
    Done,
}

/// Read `dir` for `*.nn`, parse each as generic JSON, stream `Row`s back.
fn spawn_loader(dir: PathBuf) -> mpsc::Receiver<LoadMsg> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for e in entries.flatten() {
                let path = e.path();
                if path.extension().and_then(|x| x.to_str()) != Some("nn") {
                    continue;
                }
                let Ok(text) = std::fs::read_to_string(&path) else { continue };
                let Ok(val) = serde_json::from_str::<serde_json::Value>(&text) else { continue };
                let serde_json::Value::Object(map) = val else { continue };

                let mut cells: BTreeMap<String, Cell> = BTreeMap::new();
                for (k, v) in &map {
                    match v {
                        serde_json::Value::Number(n) => {
                            if let Some(f) = n.as_f64() {
                                cells.insert(k.clone(), Cell::Num(f));
                            }
                        }
                        serde_json::Value::Bool(b) => {
                            cells.insert(k.clone(), Cell::Bool(*b));
                        }
                        serde_json::Value::String(s) => {
                            cells.insert(k.clone(), Cell::Text(s.clone()));
                        }
                        serde_json::Value::Array(a) => {
                            cells.insert(format!("{k}.len"), Cell::Num(a.len() as f64));
                        }
                        _ => {}
                    }
                }
                // Always-available file columns.
                let fname = path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_string();
                let fsize = e.metadata().map(|m| m.len()).unwrap_or(0);
                cells.insert("file".into(), Cell::Text(fname));
                cells.insert("bytes".into(), Cell::Num(fsize as f64));

                let png_path = path.with_extension("png");
                let row = Row { nn_path: path, png_path, cells };
                if tx.send(LoadMsg::Row(row)).is_err() {
                    return;
                }
            }
        }
        let _ = tx.send(LoadMsg::Done);
    });
    rx
}

// ── Preferences (TOML round-trip, mirrors ViewerPrefs) ─────────────────────────

#[derive(Serialize, Deserialize)]
struct BrowserPrefs {
    folder: String,
    columns: Vec<String>,
    sort_column: String,
    sort_desc: bool,
    thumb_size: u32,
    breed_dir: String,
    move_dir: String,
    copy_dir: String,
    window_width: u32,
    window_height: u32,
}

impl Default for BrowserPrefs {
    fn default() -> Self {
        Self {
            folder: String::new(),
            columns: ["file", "beauty", "clip_score", "laion_score", "fitness"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            sort_column: "beauty".into(),
            sort_desc: true,
            thumb_size: 96,
            breed_dir: "breeding".into(),
            move_dir: String::new(),
            copy_dir: String::new(),
            window_width: 1400,
            window_height: 900,
        }
    }
}

impl BrowserPrefs {
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

/// Startup folder precedence: --dir > prefs.folder > config.toml save_dir > ./fractals_dag.
fn resolve_folder(cli: Option<PathBuf>, prefs: &BrowserPrefs) -> PathBuf {
    if let Some(d) = cli {
        return d;
    }
    if !prefs.folder.is_empty() {
        return PathBuf::from(&prefs.folder);
    }
    if let Ok(cfg) = Config::load(Path::new("config.toml")) {
        return cfg.output.save_dir;
    }
    PathBuf::from("fractals_dag")
}

/// Locate a project binary robustly: sibling of this exe, the other build profile
/// (target/debug ↔ target/release — the viewer is often only built in release),
/// then ~/.local/bin, and finally the bare name (OS PATH lookup).
fn locate_bin(name: &str) -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let c = dir.join(name);
            if c.exists() {
                return c;
            }
            // …/target/{debug,release}/<exe> → also try the sibling profile dir.
            if let Some(target) = dir.parent() {
                for prof in ["release", "debug"] {
                    let c = target.join(prof).join(name);
                    if c.exists() {
                        return c;
                    }
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

fn viewer_path() -> PathBuf {
    locate_bin("nnfractals-viewer")
}

fn load_thumb(ctx: &egui::Context, path: &Path, sz: u32) -> Option<TextureHandle> {
    let img = image::open(path).ok()?.thumbnail(sz, sz).to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let color = ColorImage::from_rgb([w, h], img.as_raw());
    Some(ctx.load_texture(path.to_string_lossy(), color, TextureOptions::LINEAR))
}

// ── Destination dialog ─────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum DestKind {
    Move,
    Copy,
    Breed,
}

struct DestDialog {
    kind: DestKind,
    path: String,
}

// ── Application ────────────────────────────────────────────────────────────────

struct App {
    prefs: BrowserPrefs,
    prefs_path: PathBuf,
    folder: PathBuf,

    rows: Vec<Row>,
    catalog: BTreeSet<String>,
    loader: Option<mpsc::Receiver<LoadMsg>>,
    loading: bool,
    load_count: usize,

    thumb_cache: HashMap<PathBuf, Option<TextureHandle>>,
    selection: HashSet<PathBuf>,
    anchor: Option<usize>,
    sort_dirty: bool,
    scroll_offset: f32,       // last known table scroll offset (for PageUp/Down/Home/End)
    scroll_to: Option<f32>,   // target offset to apply next frame

    show_columns: bool,
    confirm_delete: bool,
    dest: Option<DestDialog>,
    status: String,

    // Pairwise rating mode (collect human "which is nicer?" judgments).
    rating_mode: bool,
    pair: Option<(usize, usize)>,
    rating_cache: HashMap<PathBuf, Option<TextureHandle>>,
    ratings_logged: usize,
}

impl App {
    fn new(prefs: BrowserPrefs, prefs_path: PathBuf, folder: PathBuf) -> Self {
        let loader = Some(spawn_loader(folder.clone()));
        App {
            prefs,
            prefs_path,
            folder,
            rows: Vec::new(),
            catalog: BTreeSet::new(),
            loader,
            loading: true,
            load_count: 0,
            thumb_cache: HashMap::new(),
            selection: HashSet::new(),
            anchor: None,
            sort_dirty: false,
            scroll_offset: 0.0,
            scroll_to: None,
            show_columns: false,
            confirm_delete: false,
            dest: None,
            status: String::new(),
            rating_mode: false,
            pair: None,
            rating_cache: HashMap::new(),
            ratings_logged: 0,
        }
    }

    fn save_prefs(&mut self) {
        self.prefs.folder = self.folder.to_string_lossy().into_owned();
        self.prefs.save(&self.prefs_path);
    }

    fn reload(&mut self) {
        self.rows.clear();
        self.catalog.clear();
        self.thumb_cache.clear();
        self.selection.clear();
        self.anchor = None;
        self.load_count = 0;
        self.loading = true;
        self.loader = Some(spawn_loader(self.folder.clone()));
    }

    fn drain_loader(&mut self) {
        let mut done = false;
        if let Some(rx) = &self.loader {
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    LoadMsg::Row(r) => {
                        for k in r.cells.keys() {
                            self.catalog.insert(k.clone());
                        }
                        self.rows.push(r);
                        self.load_count += 1;
                    }
                    LoadMsg::Done => {
                        done = true;
                        break;
                    }
                }
            }
        }
        if done {
            self.loader = None;
            self.loading = false;
            self.sort_dirty = true;
        }
    }

    fn sort_rows(&mut self) {
        let col = self.prefs.sort_column.clone();
        let desc = self.prefs.sort_desc;
        self.rows
            .sort_by(|a, b| cmp_cells(a.cells.get(&col), b.cells.get(&col), desc));
    }

    // ── Actions ────────────────────────────────────────────────────────────────

    /// Selected genome paths, in current display order.
    fn selected_paths(&self) -> Vec<PathBuf> {
        self.rows
            .iter()
            .filter(|r| self.selection.contains(&r.nn_path))
            .map(|r| r.nn_path.clone())
            .collect()
    }

    fn open_path(&mut self, path: &Path) {
        match std::process::Command::new(viewer_path()).arg(path).spawn() {
            Ok(_) => self.status = format!("Opened {}", path.display()),
            Err(e) => self.status = format!("Could not launch viewer: {e}"),
        }
    }

    fn open_selected(&mut self) {
        if let Some(p) = self.selected_paths().first().cloned() {
            self.open_path(&p);
        }
    }

    fn delete_selected(&mut self) {
        let paths = self.selected_paths();
        for p in &paths {
            let _ = std::fs::remove_file(p);
            let png = p.with_extension("png");
            let _ = std::fs::remove_file(&png);
            self.thumb_cache.remove(&png);
        }
        self.rows.retain(|r| !self.selection.contains(&r.nn_path));
        let n = paths.len();
        self.selection.clear();
        self.anchor = None;
        self.status = format!("Deleted {n} fractal(s)");
    }

    /// Copy (and optionally remove originals for a move) selected `.nn`+`.png` to `dest`.
    fn transfer_selected(&mut self, dest: &Path, remove: bool) {
        if let Err(e) = std::fs::create_dir_all(dest) {
            self.status = format!("Cannot create {}: {e}", dest.display());
            return;
        }
        let paths = self.selected_paths();
        let mut done: Vec<PathBuf> = Vec::new();
        for p in &paths {
            let png = p.with_extension("png");
            let Some(nn_name) = p.file_name() else { continue };
            let dnn = dest.join(nn_name);
            let dpng = png.file_name().map(|n| dest.join(n));

            let nn_ok = if remove {
                move_file(p, &dnn).is_ok()
            } else {
                std::fs::copy(p, &dnn).is_ok()
            };
            if !nn_ok {
                continue;
            }
            if let Some(dpng) = dpng {
                if png.exists() {
                    if remove {
                        let _ = move_file(&png, &dpng);
                    } else {
                        let _ = std::fs::copy(&png, &dpng);
                    }
                }
            }
            done.push(p.clone());
        }
        let verb = if remove { "Moved" } else { "Copied" };
        if remove {
            let moved: HashSet<PathBuf> = done.iter().cloned().collect();
            self.rows.retain(|r| !moved.contains(&r.nn_path));
            for m in &moved {
                self.selection.remove(m);
            }
        }
        self.status = format!("{verb} {} fractal(s) → {}", done.len(), dest.display());
    }

    // ── UI ───────────────────────────────────────────────────────────────────────

    /// Adjust the thumbnail size by `delta` px (clamped); re-decode thumbnails at the
    /// new size and persist. Used by Ctrl+/Ctrl- and Ctrl+mousewheel, and the toolbar.
    fn zoom_thumbs(&mut self, delta: i32) {
        let sz = (self.prefs.thumb_size as i32 + delta).clamp(MIN_THUMB as i32, MAX_THUMB as i32) as u32;
        if sz != self.prefs.thumb_size {
            self.prefs.thumb_size = sz;
            self.thumb_cache.clear(); // re-decode at the new resolution so it stays crisp
            self.save_prefs();
        }
    }

    /// Ctrl + '+' / '-' and Ctrl + mousewheel zoom the thumbnails in/out.
    /// (Ctrl+scroll is surfaced by egui as `zoom_delta()`; it does not also scroll
    /// the table. We disable egui's own keyboard zoom in `ui()` so Ctrl+/- is ours.)
    fn handle_zoom(&mut self, ctx: &egui::Context) {
        let (ctrl, plus, minus, zoom_delta) = ctx.input(|i| {
            (
                i.modifiers.ctrl || i.modifiers.command,
                i.key_pressed(egui::Key::Plus) || i.key_pressed(egui::Key::Equals),
                i.key_pressed(egui::Key::Minus),
                i.zoom_delta(),
            )
        });
        if ctrl && plus {
            self.zoom_thumbs(ZOOM_STEP);
        }
        if ctrl && minus {
            self.zoom_thumbs(-ZOOM_STEP);
        }
        if zoom_delta > 1.001 {
            self.zoom_thumbs(ZOOM_STEP);
        } else if zoom_delta < 0.999 {
            self.zoom_thumbs(-ZOOM_STEP);
        }
    }

    fn toolbar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("📁");
            ui.monospace(self.folder.to_string_lossy());
            if ui.button("Reload").clicked() {
                self.reload();
            }
            ui.separator();
            if ui.button("Columns ▾").clicked() {
                self.show_columns = !self.show_columns;
            }
            if ui.selectable_label(self.rating_mode, "⚖ Rate")
                .on_hover_text("Pairwise rating: pick the nicer of two fractals to build a\n\
                                 human-preference dataset (ratings.jsonl) for training.")
                .clicked()
            {
                self.rating_mode = !self.rating_mode;
                if self.rating_mode {
                    self.pick_pair();
                }
            }
            ui.separator();

            let n = self.selection.len();
            let has_sel = n > 0;
            ui.add_enabled_ui(n == 1 || (has_sel), |ui| {
                if ui
                    .add_enabled(has_sel, egui::Button::new("Open ▶"))
                    .on_hover_text("Open the (first) selected fractal in the viewer")
                    .clicked()
                {
                    self.open_selected();
                }
            });
            if ui
                .add_enabled(has_sel, egui::Button::new("★ Good"))
                .on_hover_text("Mark selected as favorite/good (F). Writes favorite=true\n\
                                 into the .nn — a sortable column and a training signal.")
                .clicked()
            {
                self.toggle_favorite();
            }
            if ui
                .add_enabled(has_sel, egui::Button::new("Delete"))
                .on_hover_text("Permanently delete selected .nn + .png")
                .clicked()
            {
                self.confirm_delete = true;
            }
            if ui.add_enabled(has_sel, egui::Button::new("Move…")).clicked() {
                self.dest = Some(DestDialog {
                    kind: DestKind::Move,
                    path: self.prefs.move_dir.clone(),
                });
            }
            if ui.add_enabled(has_sel, egui::Button::new("Copy…")).clicked() {
                self.dest = Some(DestDialog {
                    kind: DestKind::Copy,
                    path: self.prefs.copy_dir.clone(),
                });
            }
            if ui
                .add_enabled(has_sel, egui::Button::new("Add to breeding ▶"))
                .on_hover_text(
                    "Copy selected genomes into the breeding folder.\n\
                     Point a config's save_dir at that folder to warm-start\n\
                     a GA run from these picks (load_archive_seeds reads it).",
                )
                .clicked()
            {
                self.dest = Some(DestDialog {
                    kind: DestKind::Breed,
                    path: self.prefs.breed_dir.clone(),
                });
            }

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if self.loading {
                    ui.label(format!("loading… {}", self.load_count));
                } else {
                    ui.label(format!("{} fractals · {} selected", self.rows.len(), n));
                }
                ui.separator();
                // Thumbnail zoom (also Ctrl +/- and Ctrl+mousewheel).
                if ui.small_button("＋").on_hover_text("Zoom in (Ctrl+ / Ctrl+wheel)").clicked() {
                    self.zoom_thumbs(ZOOM_STEP);
                }
                ui.label(format!("{}px", self.prefs.thumb_size));
                if ui.small_button("－").on_hover_text("Zoom out (Ctrl- / Ctrl+wheel)").clicked() {
                    self.zoom_thumbs(-ZOOM_STEP);
                }
                ui.label("🔍");
            });
        });
        if !self.status.is_empty() {
            ui.label(egui::RichText::new(&self.status).color(Color32::LIGHT_GREEN));
        }
    }

    /// Indices (in current display order) of the selected rows.
    fn selected_indices(&self) -> Vec<usize> {
        (0..self.rows.len())
            .filter(|&i| self.selection.contains(&self.rows[i].nn_path))
            .collect()
    }

    /// Toggle a persistent `favorite` flag on the selected fractals: writes
    /// "favorite": true/false into each .nn (a sortable browser column, and a
    /// positive-example signal we can later feed into the preference model).
    fn toggle_favorite(&mut self) {
        let idxs = self.selected_indices();
        if idxs.is_empty() {
            self.status = "select fractal(s) first, then ★ / F to mark as good".into();
            return;
        }
        // New state = opposite of the first selected row's current flag.
        let currently = matches!(self.rows[idxs[0]].cells.get("favorite"), Some(Cell::Bool(true)));
        let value = !currently;
        let mut n = 0;
        for i in idxs {
            let p = self.rows[i].nn_path.clone();
            let Ok(txt) = std::fs::read_to_string(&p) else { continue };
            let Ok(mut v) = serde_json::from_str::<serde_json::Value>(&txt) else { continue };
            if let Some(obj) = v.as_object_mut() {
                obj.insert("favorite".into(), serde_json::Value::Bool(value));
                let out = serde_json::to_string_pretty(&v).unwrap_or(txt);
                if std::fs::write(&p, out).is_ok() {
                    self.rows[i].cells.insert("favorite".into(), Cell::Bool(value));
                    self.catalog.insert("favorite".into());
                    n += 1;
                }
            }
        }
        self.status = format!("marked {n} fractal(s) favorite = {value}");
    }

    // ── Pairwise rating mode ─────────────────────────────────────────────────────

    fn pick_pair(&mut self) {
        if self.rows.len() < 2 {
            self.pair = None;
            return;
        }
        let mut r = rand::rng();
        let a = r.random_range(0..self.rows.len());
        let mut b = r.random_range(0..self.rows.len());
        while b == a {
            b = r.random_range(0..self.rows.len());
        }
        self.pair = Some((a, b));
    }

    fn log_rating(&mut self, winner: usize, loser: usize) {
        let wp = self.rows[winner].nn_path.to_string_lossy().into_owned();
        let lp = self.rows[loser].nn_path.to_string_lossy().into_owned();
        let line = format!("{{\"winner\": {:?}, \"loser\": {:?}}}\n", wp, lp);
        let path = self.folder.join("ratings.jsonl");
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
            let _ = f.write_all(line.as_bytes());
            self.ratings_logged += 1;
            self.status = format!("{} ratings → {}", self.ratings_logged, path.display());
        } else {
            self.status = format!("could not write {}", path.display());
        }
    }

    /// Draw one fractal image as a big clickable button; returns true if picked.
    fn rating_cell(&mut self, ui: &mut egui::Ui, idx: usize, side: f32) -> bool {
        let path = self.rows[idx].png_path.clone();
        let tex = self
            .rating_cache
            .entry(path.clone())
            .or_insert_with(|| load_thumb(ui.ctx(), &path, 768));
        let mut clicked = false;
        ui.vertical_centered(|ui| {
            if let Some(t) = tex {
                let ts = t.size_vec2();
                let scale = (side / ts.x).min(side / ts.y);
                let img = egui::Image::new(egui::load::SizedTexture::new(t.id(), ts * scale))
                    .sense(egui::Sense::click());
                if ui.add(img).clicked() {
                    clicked = true;
                }
            } else {
                ui.label("(no image)");
            }
        });
        clicked
    }

    fn show_rating_panel(&mut self, ui: &mut egui::Ui) {
        if self.rows.len() < 2 {
            ui.centered_and_justified(|ui| ui.label("Need at least 2 fractals to rate."));
            return;
        }
        if self.pair.map_or(true, |(a, b)| a >= self.rows.len() || b >= self.rows.len()) {
            self.pick_pair();
        }
        let (li, ri) = self.pair.unwrap();

        let (pick_l, pick_r, skip) = ui.input(|i| {
            (
                i.key_pressed(egui::Key::Num1) || i.key_pressed(egui::Key::ArrowLeft),
                i.key_pressed(egui::Key::Num2) || i.key_pressed(egui::Key::ArrowRight),
                i.key_pressed(egui::Key::Space),
            )
        });

        ui.horizontal(|ui| {
            ui.heading("Which fractal is nicer?");
            ui.label(
                egui::RichText::new("click an image  ·  1/←  ·  2/→  ·  Space = skip")
                    .weak(),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(format!("{} rated", self.ratings_logged));
            });
        });

        let avail = ui.available_size();
        let side = ((avail.x / 2.0) - 24.0).min(avail.y - 12.0).max(64.0);

        let mut chosen: Option<bool> = None; // Some(true) = left wins
        ui.columns(2, |cols| {
            if self.rating_cell(&mut cols[0], li, side) {
                chosen = Some(true);
            }
            if self.rating_cell(&mut cols[1], ri, side) {
                chosen = Some(false);
            }
        });

        if pick_l {
            chosen = Some(true);
        } else if pick_r {
            chosen = Some(false);
        }

        if let Some(left_wins) = chosen {
            let (w, l) = if left_wins { (li, ri) } else { (ri, li) };
            self.log_rating(w, l);
            self.pick_pair();
        } else if skip {
            self.pick_pair();
        }
    }

    fn show_table(&mut self, ui: &mut egui::Ui) {
        let columns = self.prefs.columns.clone();
        let thumb = self.prefs.thumb_size as f32;
        let sort_col = self.prefs.sort_column.clone();
        let sort_desc = self.prefs.sort_desc;
        let mods = ui.input(|i| i.modifiers);
        let fav_key = ui.input(|i| i.key_pressed(egui::Key::F));

        let mut clicked_col: Option<String> = None;
        let mut clicked_row: Option<usize> = None;
        let mut open_row: Option<usize> = None;

        // ── Page Up/Down/Home/End scrolling ──
        let row_height = thumb + 6.0 + ui.spacing().item_spacing.y;
        let page = (ui.available_height() - row_height).max(row_height);
        let content_h = self.rows.len() as f32 * row_height;
        let (pgdn, pgup, home, end) = ui.input(|i| {
            (
                i.key_pressed(egui::Key::PageDown),
                i.key_pressed(egui::Key::PageUp),
                i.key_pressed(egui::Key::Home),
                i.key_pressed(egui::Key::End),
            )
        });
        if pgdn {
            self.scroll_to = Some(self.scroll_offset + page);
        } else if pgup {
            self.scroll_to = Some((self.scroll_offset - page).max(0.0));
        } else if home {
            self.scroll_to = Some(0.0);
        } else if end {
            self.scroll_to = Some(content_h);
        }

        // Disjoint borrows so the row closure can read rows/selection and fill the cache.
        let scroll_to = self.scroll_to.take();
        let rows = &self.rows;
        let selection = &self.selection;
        let thumb_cache = &mut self.thumb_cache;

        let mut builder = TableBuilder::new(ui)
            .striped(true)
            .resizable(true)
            .sense(egui::Sense::click())
            .cell_layout(egui::Layout::left_to_right(egui::Align::Center))
            .column(Column::exact(thumb + 4.0));
        if let Some(off) = scroll_to {
            builder = builder.vertical_scroll_offset(off);
        }
        for _ in &columns {
            builder = builder.column(Column::initial(100.0).at_least(40.0).clip(true));
        }

        let table_out = builder
            .header(24.0, |mut header| {
                header.col(|ui| {
                    ui.strong("img");
                });
                for c in &columns {
                    header.col(|ui| {
                        let arrow = if *c == sort_col {
                            if sort_desc { " ▼" } else { " ▲" }
                        } else {
                            ""
                        };
                        if ui
                            .button(egui::RichText::new(format!("{c}{arrow}")).strong())
                            .clicked()
                        {
                            clicked_col = Some(c.clone());
                        }
                    });
                }
            })
            .body(|body| {
                body.rows(thumb + 6.0, rows.len(), |mut row| {
                    let idx = row.index();
                    let r = &rows[idx];
                    row.set_selected(selection.contains(&r.nn_path));

                    row.col(|ui| {
                        if matches!(r.cells.get("favorite"), Some(Cell::Bool(true))) {
                            ui.colored_label(Color32::from_rgb(255, 205, 60), "★");
                        }
                        let tex = thumb_cache
                            .entry(r.png_path.clone())
                            .or_insert_with(|| load_thumb(ui.ctx(), &r.png_path, thumb as u32));
                        if let Some(t) = tex {
                            let ts = t.size_vec2();
                            let scale = (thumb / ts.x).min(thumb / ts.y).min(1.0);
                            let size = ts * scale;
                            ui.add(egui::Image::new(egui::load::SizedTexture::new(t.id(), size)));
                        } else {
                            ui.weak("—");
                        }
                    });
                    for c in &columns {
                        row.col(|ui| {
                            if let Some(cell) = r.cells.get(c) {
                                let txt = cell.display();
                                ui.label(txt).on_hover_text(
                                    r.cells.get(c).map(|v| v.display()).unwrap_or_default(),
                                );
                            }
                        });
                    }
                    let resp = row.response();
                    if resp.clicked() {
                        clicked_row = Some(idx);
                    }
                    if resp.double_clicked() {
                        open_row = Some(idx);
                    }
                });
            });
        // Remember where the table ended up so the next Page key steps from here.
        self.scroll_offset = table_out.state.offset.y;

        // ── Apply interactions (borrows above have ended) ──
        if let Some(c) = clicked_col {
            if self.prefs.sort_column == c {
                self.prefs.sort_desc = !self.prefs.sort_desc;
            } else {
                self.prefs.sort_column = c;
                self.prefs.sort_desc = true;
            }
            self.sort_dirty = true;
            self.save_prefs();
        }
        if let Some(idx) = clicked_row {
            self.apply_row_click(idx, mods);
        }
        // Double-click opens that fractal in the viewer (and selects it).
        if let Some(idx) = open_row {
            let path = self.rows[idx].nn_path.clone();
            self.selection.clear();
            self.selection.insert(path.clone());
            self.anchor = Some(idx);
            self.open_path(&path);
        }
        // F toggles the "favorite / good" flag on the selection.
        if fav_key {
            self.toggle_favorite();
        }
    }

    fn apply_row_click(&mut self, idx: usize, mods: egui::Modifiers) {
        let path = self.rows[idx].nn_path.clone();
        if mods.ctrl || mods.command {
            if !self.selection.remove(&path) {
                self.selection.insert(path);
            }
            self.anchor = Some(idx);
        } else if mods.shift {
            if let Some(a) = self.anchor {
                let (lo, hi) = if a <= idx { (a, idx) } else { (idx, a) };
                self.selection.clear();
                for i in lo..=hi {
                    self.selection.insert(self.rows[i].nn_path.clone());
                }
            } else {
                self.selection.insert(path);
                self.anchor = Some(idx);
            }
        } else {
            self.selection.clear();
            self.selection.insert(path);
            self.anchor = Some(idx);
        }
    }

    fn show_columns_window(&mut self, ctx: &egui::Context) {
        let mut open = self.show_columns;
        // Union of discovered keys plus any persisted column not currently present.
        let mut keys: BTreeSet<String> = self.catalog.clone();
        for c in &self.prefs.columns {
            keys.insert(c.clone());
        }
        egui::Window::new("Columns")
            .open(&mut open)
            .resizable(true)
            .show(ctx, |ui| {
                ui.label("Shown columns (also the sort keys):");
                egui::ScrollArea::vertical().max_height(400.0).show(ui, |ui| {
                    for key in &keys {
                        let mut on = self.prefs.columns.contains(key);
                        let present = self.catalog.contains(key);
                        let label = if present {
                            egui::RichText::new(key)
                        } else {
                            egui::RichText::new(key).weak().italics()
                        };
                        if ui.checkbox(&mut on, label).changed() {
                            if on {
                                self.prefs.columns.push(key.clone());
                            } else {
                                self.prefs.columns.retain(|c| c != key);
                            }
                            self.save_prefs();
                        }
                    }
                });
            });
        self.show_columns = open;
    }

    fn show_delete_window(&mut self, ctx: &egui::Context) {
        if !self.confirm_delete {
            return;
        }
        let n = self.selection.len();
        let mut close = false;
        egui::Window::new("Confirm delete")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                ui.label(
                    egui::RichText::new(format!("Permanently delete {n} fractal(s)?"))
                        .color(Color32::from_rgb(255, 120, 120)),
                );
                ui.label("Both the .nn and its .png are removed from disk. No undo.");
                ui.horizontal(|ui| {
                    if ui.button("Delete").clicked() {
                        self.delete_selected();
                        close = true;
                    }
                    if ui.button("Cancel").clicked() {
                        close = true;
                    }
                });
            });
        if close {
            self.confirm_delete = false;
        }
    }

    fn show_dest_window(&mut self, ctx: &egui::Context) {
        let Some(dlg) = &self.dest else { return };
        let kind = dlg.kind;
        let mut path = dlg.path.clone();
        let (title, verb) = match kind {
            DestKind::Move => ("Move to folder", "Move"),
            DestKind::Copy => ("Copy to folder", "Copy"),
            DestKind::Breed => ("Add to breeding folder", "Add"),
        };
        let n = self.selection.len();
        let mut action: Option<bool> = None; // Some(confirm?) — true=confirm, false=cancel

        egui::Window::new(title)
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                ui.label(format!("{verb} {n} fractal(s) to:"));
                ui.add(
                    egui::TextEdit::singleline(&mut path)
                        .desired_width(360.0)
                        .hint_text("destination folder path"),
                );
                if kind == DestKind::Breed {
                    ui.label(
                        egui::RichText::new(
                            "Point a config's save_dir here, then run nnfractals to breed from these.",
                        )
                        .weak(),
                    );
                }
                ui.horizontal(|ui| {
                    let ok = !path.trim().is_empty();
                    if ui.add_enabled(ok, egui::Button::new(verb)).clicked() {
                        action = Some(true);
                    }
                    if ui.button("Cancel").clicked() {
                        action = Some(false);
                    }
                });
            });

        // reflect edits back into the live dialog
        if let Some(d) = &mut self.dest {
            d.path = path.clone();
        }
        match action {
            Some(true) => {
                let dest = PathBuf::from(path.trim());
                match kind {
                    DestKind::Move => {
                        self.prefs.move_dir = dest.to_string_lossy().into_owned();
                        self.transfer_selected(&dest, true);
                    }
                    DestKind::Copy => {
                        self.prefs.copy_dir = dest.to_string_lossy().into_owned();
                        self.transfer_selected(&dest, false);
                    }
                    DestKind::Breed => {
                        self.prefs.breed_dir = dest.to_string_lossy().into_owned();
                        self.transfer_selected(&dest, false);
                    }
                }
                self.save_prefs();
                self.dest = None;
            }
            Some(false) => self.dest = None,
            None => {}
        }
    }
}

/// Move a file, falling back to copy+remove across filesystem boundaries.
fn move_file(src: &Path, dst: &Path) -> std::io::Result<()> {
    if std::fs::rename(src, dst).is_ok() {
        return Ok(());
    }
    std::fs::copy(src, dst)?;
    std::fs::remove_file(src)?;
    Ok(())
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();

        // Track window size for persistence.
        let vr = ctx.viewport_rect();
        self.prefs.window_width = vr.width() as u32;
        self.prefs.window_height = vr.height() as u32;

        // We own Ctrl+/- for thumbnail zoom, so stop egui rescaling the whole UI.
        ctx.options_mut(|o| o.zoom_with_keyboard = false);

        self.drain_loader();
        self.handle_zoom(&ctx);
        if self.sort_dirty {
            self.sort_rows();
            self.sort_dirty = false;
        }

        egui::Panel::top("toolbar").show(ui, |ui| self.toolbar(ui));
        egui::CentralPanel::default().show(ui, |ui| {
            if self.rating_mode {
                self.show_rating_panel(ui);
            } else {
                self.show_table(ui);
            }
        });

        self.show_columns_window(&ctx);
        self.show_delete_window(&ctx);
        self.show_dest_window(&ctx);

        if self.loading {
            ctx.request_repaint();
        }
    }

    fn on_exit(&mut self) {
        self.save_prefs();
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let prefs_path = PathBuf::from(PREFS_FILE);
    let prefs = BrowserPrefs::load(&prefs_path);
    let folder = resolve_folder(args.dir, &prefs);

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("NNFractals Browser")
            .with_inner_size([prefs.window_width as f32, prefs.window_height as f32]),
        ..Default::default()
    };

    eframe::run_native(
        "NNFractals Browser",
        options,
        Box::new(move |_cc| Ok(Box::new(App::new(prefs, prefs_path, folder)))),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(())
}
