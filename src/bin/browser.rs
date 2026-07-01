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
use serde::{Deserialize, Serialize};

use nnfractals::config::Config;

const PREFS_FILE: &str = "browser_prefs.toml";

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

/// Path to the sibling viewer binary (falls back to bare name on PATH).
fn viewer_path() -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let cand = dir.join("nnfractals-viewer");
            if cand.exists() {
                return cand;
            }
        }
    }
    PathBuf::from("nnfractals-viewer")
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

    show_columns: bool,
    confirm_delete: bool,
    dest: Option<DestDialog>,
    status: String,
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
            show_columns: false,
            confirm_delete: false,
            dest: None,
            status: String::new(),
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
            });
        });
        if !self.status.is_empty() {
            ui.label(egui::RichText::new(&self.status).color(Color32::LIGHT_GREEN));
        }
    }

    fn show_table(&mut self, ui: &mut egui::Ui) {
        let columns = self.prefs.columns.clone();
        let thumb = self.prefs.thumb_size as f32;
        let sort_col = self.prefs.sort_column.clone();
        let sort_desc = self.prefs.sort_desc;
        let mods = ui.input(|i| i.modifiers);

        let mut clicked_col: Option<String> = None;
        let mut clicked_row: Option<usize> = None;
        let mut open_row: Option<usize> = None;

        // Disjoint borrows so the row closure can read rows/selection and fill the cache.
        let rows = &self.rows;
        let selection = &self.selection;
        let thumb_cache = &mut self.thumb_cache;

        let mut builder = TableBuilder::new(ui)
            .striped(true)
            .resizable(true)
            .sense(egui::Sense::click())
            .cell_layout(egui::Layout::left_to_right(egui::Align::Center))
            .column(Column::exact(thumb + 4.0));
        for _ in &columns {
            builder = builder.column(Column::initial(100.0).at_least(40.0).clip(true));
        }

        builder
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

        self.drain_loader();
        if self.sort_dirty {
            self.sort_rows();
            self.sort_dirty = false;
        }

        egui::Panel::top("toolbar").show(ui, |ui| self.toolbar(ui));
        egui::CentralPanel::default().show(ui, |ui| self.show_table(ui));

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
