//! Launcher window: the fitness-recipe builder (data model and term catalog in
//! `nnfractals::fitness_recipe`). A recipe is what the GA rewards: weighted terms plus the
//! scoring settings that belong with them. The window lists every term with what it
//! measures, what it costs and what your ratings say about it, warns about the failure
//! modes past runs ran into, shows which parts of the program a run will start, and can
//! preview the ranking a recipe gives on a folder of genomes.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use eframe::egui::{self, Color32, RichText};
use nnfractals::fitness_recipe::{self as fr, Group, Level, Recipe, RunContext, Term, CATALOG};

use super::{sibling, QuatEvoMode, QuatEvoPrefs};

// ── Recipe ↔ run settings ───────────────────────────────────────────────────────

pub fn run_context(prefs: &QuatEvoPrefs) -> RunContext {
    RunContext { gpu: prefs.gpu, map_elites: prefs.mode == QuatEvoMode::MapElites, probe_size: prefs.probe_size, workers: prefs.workers }
}

/// Put `r` into the run settings. Returns what changed beyond the terms.
pub fn apply(r: &Recipe, prefs: &mut QuatEvoPrefs, root: &Path) -> String {
    prefs.fitness_metric = r.spec();
    prefs.collapse = r.collapse.clone();
    prefs.eval_phases = r.eval_phases;
    prefs.smooth_weight = r.smooth_weight;
    prefs.embed_diversity = r.embed_diversity;
    prefs.auto_range = r.auto_range;
    prefs.recipe = r.name.clone();
    let mut note = String::new();
    if !r.pool_dir.trim().is_empty() && prefs.pool_dir.trim() != r.pool_dir.trim() {
        if root.join(r.pool_dir.trim()).is_dir() {
            note = format!(" · seed pool set to {}", r.pool_dir.trim());
            prefs.pool_dir = r.pool_dir.trim().to_string();
        } else {
            note = format!(" · suggested pool {} not found, pool unchanged", r.pool_dir.trim());
        }
    }
    format!("using recipe {}{note}", r.name)
}

/// Every known recipe (built-ins, then `recipes/*.json` keyed by file stem), read
/// once and refreshed every few seconds or after a save/delete, so the pickers
/// drawn on every frame do not touch the disk.
pub struct Recipes {
    list: Vec<(String, Recipe)>,
    read_at: Instant,
}

impl Recipes {
    pub fn read(root: &Path) -> Recipes {
        let mut list: Vec<(String, Recipe)> = fr::builtins().into_iter().map(|r| (r.name.clone(), r)).collect();
        let dir = root.join(fr::RECIPES_DIR);
        for stem in Recipe::list(root) {
            if let Some(mut r) = Recipe::load_file(&dir.join(format!("{stem}.json"))) {
                if r.name.trim().is_empty() { r.name = stem.clone(); }
                list.push((stem, r));
            }
        }
        Recipes { list, read_at: Instant::now() }
    }

    /// Re-read when the last read is a few seconds old (files saved elsewhere).
    pub fn refresh_if_old(&mut self, root: &Path) {
        if self.read_at.elapsed() > Duration::from_secs(3) { *self = Recipes::read(root); }
    }

    /// A recipe by its shown name or its file key.
    pub fn find(&self, name: &str) -> Option<&Recipe> {
        if name.trim().is_empty() { return None; }
        let key = nnfractals::tuning::sanitize(name);
        self.list.iter().find(|(k, r)| r.name == name || (!r.builtin && *k == key)).map(|(_, r)| r)
    }

    pub fn all(&self) -> impl Iterator<Item = &Recipe> { self.list.iter().map(|(_, r)| r) }
}

/// The named recipe the settings came from, else the first recipe with the same terms
/// (settings saved before recipes existed carry no name).
fn known_for<'a>(prefs: &QuatEvoPrefs, rec: &'a Recipes) -> Option<&'a Recipe> {
    if let Some(r) = rec.find(&prefs.recipe) { return Some(r); }
    if prefs.fitness_metric.trim().is_empty() { return None; }
    rec.all().find(|r| fr::same_spec(&r.spec(), &prefs.fitness_metric))
}

/// The recipe the run settings amount to right now.
pub fn from_prefs(prefs: &QuatEvoPrefs, rec: &Recipes) -> Recipe {
    let mut r = known_for(prefs, rec).cloned().unwrap_or_else(|| Recipe::empty(if prefs.recipe.is_empty() { "my recipe" } else { &prefs.recipe }));
    r.terms = fr::parse_spec(&prefs.fitness_metric).unwrap_or_else(|_| {
        // Keep unknown names visible (shown in red) instead of dropping them.
        prefs.fitness_metric.split(',').map(str::trim).filter(|p| !p.is_empty()).map(|p| match p.split_once(':') {
            Some((n, w)) => Term { name: n.trim().into(), weight: w.trim().parse().unwrap_or(1.0) },
            None => Term { name: p.into(), weight: 1.0 },
        }).collect()
    });
    r.collapse = prefs.collapse.clone();
    r.eval_phases = prefs.eval_phases;
    r.smooth_weight = prefs.smooth_weight;
    r.embed_diversity = prefs.embed_diversity;
    r.auto_range = prefs.auto_range;
    r
}

/// Do the run settings still equal recipe `r` (everything that changes a score)?
pub fn matches(r: &Recipe, prefs: &QuatEvoPrefs) -> bool {
    let taste = r.uses_taste();
    fr::same_spec(&r.spec(), &prefs.fitness_metric)
        && r.collapse == prefs.collapse
        && r.effective_auto_range() == (prefs.auto_range && r.needs_render())
        && (!taste || (r.eval_phases == prefs.eval_phases && (r.smooth_weight - prefs.smooth_weight).abs() < 1e-6 && (r.embed_diversity - prefs.embed_diversity).abs() < 1e-6))
}

/// "name", "name (modified)", "custom", or what a run without a recipe scores with.
pub fn status(prefs: &QuatEvoPrefs, rec: &Recipes) -> (String, bool) {
    if prefs.fitness_metric.trim().is_empty() {
        // Without a recipe, MAP-Elites and every projection but slice score with the
        // taste model alone (`build_quat_evolve_args` adds `--fitness-metric taste`).
        let taste_only = prefs.mode == QuatEvoMode::MapElites || prefs.collapse != "slice";
        return (if taste_only { "none — taste model only" } else { "none — default blend" }.into(), false);
    }
    match known_for(prefs, rec) {
        Some(r) if matches(r, prefs) => (r.name.clone(), false),
        Some(r) => (format!("{} (modified)", r.name), true),
        None => ("custom".into(), true),
    }
}

fn group_color(g: Group) -> Color32 {
    match g {
        Group::Picture => Color32::from_rgb(110, 170, 240),
        Group::Motion => Color32::from_rgb(130, 210, 130),
        Group::Sheet => Color32::from_rgb(235, 170, 70),
        Group::Shape => Color32::from_rgb(180, 150, 230),
        Group::Field => Color32::from_rgb(90, 200, 200),
        Group::Planes => Color32::from_rgb(220, 120, 140),
        Group::Structure => Color32::from_gray(150),
        Group::Taste => Color32::from_rgb(240, 210, 90),
    }
}

fn level_color(l: Level) -> Color32 {
    match l { Level::Error => Color32::from_rgb(240, 90, 90), Level::Warn => Color32::from_rgb(240, 170, 60), Level::Info => Color32::from_gray(175) }
}

/// The compact picker shown next to the run settings: choosing a recipe applies it.
/// Returns true when the settings changed.
pub fn picker(ui: &mut egui::Ui, id: &str, root: &Path, prefs: &mut QuatEvoPrefs, rui: &mut RecipeUi) -> bool {
    let mut changed = false;
    rui.recipes.refresh_if_old(root);
    ui.horizontal_wrapped(|ui| {
        ui.label("Fitness recipe:");
        let (label, modified) = status(prefs, &rui.recipes);
        let mut pick: Option<Recipe> = None;
        egui::ComboBox::from_id_salt(id).selected_text(RichText::new(&label).color(if modified { Color32::from_rgb(240, 170, 60) } else { ui.visuals().text_color() }))
            .width(300.0).show_ui(ui, |ui| {
                for r in rui.recipes.all() {
                    let resp = ui.selectable_label(prefs.recipe == r.name, if r.builtin { r.name.clone() } else { format!("  {}", r.name) });
                    let resp = if r.builtin { resp.on_hover_text("Built-in: a recipe that worked in past runs.") } else { resp };
                    if resp.clicked() { pick = Some(r.clone()); }
                }
            }).response.on_hover_text("What the GA rewards. ★ = recipes that worked in past runs. Picking one applies its terms, projection and scoring settings (and its suggested seed pool).");
        if let Some(r) = pick {
            rui.msg = apply(&r, prefs, root);
            rui.load(r);
            changed = true;
        }
        if ui.button("✏ Recipe builder…").on_hover_text("Build, check, save and load recipes: every term explained, with what it costs and what your ratings say about it.").clicked() {
            let r = from_prefs(prefs, &rui.recipes);
            rui.open_with(r, root);
        }
    });
    let r = from_prefs(prefs, &rui.recipes);
    if !r.terms.is_empty() {
        let ctx = run_context(prefs);
        let mods = r.modules(ctx);
        let on: Vec<&str> = mods.iter().filter(|m| m.on).map(|m| m.name.split(" (").next().unwrap_or(m.name)).collect();
        let worst = r.advice(ctx).first().map(|a| a.level);
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new(format!("{} terms · ~{:.2} Mpx rendered per genome · starts: {}", r.terms.iter().filter(|t| t.weight != 0.0).count(), r.render_mpx(prefs.probe_size), on.join(", "))).small().weak());
            match worst {
                Some(Level::Error) => { ui.label(RichText::new("✖ has errors — open the builder").small().color(level_color(Level::Error))); }
                Some(Level::Warn) => { ui.label(RichText::new("⚠ warnings in the builder").small().color(level_color(Level::Warn))); }
                _ => {}
            }
        });
    }
    changed
}

// ── Preview: rank a folder of genomes with the recipe ───────────────────────────

struct PreviewRow { file: String, gate: bool, vals: HashMap<String, f64> }

#[derive(Default)]
struct Preview {
    dir: String,
    limit: usize,
    child: Option<(Child, Instant, PathBuf, PathBuf)>,
    rows: Vec<PreviewRow>,
    /// Terms the last run computed.
    terms: Vec<String>,
    collapse: String,
    msg: String,
    textures: HashMap<String, Option<egui::TextureHandle>>,
}

impl Preview {
    fn start(&mut self, root: &Path, r: &Recipe, prefs: &QuatEvoPrefs) {
        let names: Vec<String> = {
            let mut v: Vec<String> = r.terms.iter().filter(|t| t.weight != 0.0 && fr::info(&t.name).is_some()).map(|t| t.name.clone()).collect();
            v.dedup();
            v
        };
        if names.is_empty() { self.msg = "add a term first".into(); return; }
        let dir = root.join(self.dir.trim());
        if !dir.is_dir() { self.msg = format!("{} is not a folder", dir.display()); return; }
        let tmp = std::env::temp_dir().join(format!("nnf_recipe_preview_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&tmp);
        let (csv, log) = (tmp.join("terms.csv"), tmp.join("terms.log"));
        let _ = std::fs::remove_file(&csv);
        let mut cmd = Command::new(sibling("explorer"));
        cmd.args(["quat-metric-terms", "--dir"]).arg(&dir)
            .args(["--terms", &names.join(","), "--collapse", &r.collapse, "--phases", &r.eval_phases.to_string(),
                   "--limit", &self.limit.max(1).to_string(), "--probe-size", &prefs.probe_size.to_string(), "--ga-gate", "--out"]).arg(&csv);
        if prefs.gpu { cmd.arg("--gpu"); }
        let Ok(logf) = std::fs::File::create(&log) else { self.msg = "cannot write the preview log".into(); return };
        match cmd.current_dir(root).stdout(Stdio::null()).stderr(logf).spawn() {
            Ok(c) => {
                self.child = Some((c, Instant::now(), csv, log));
                self.terms = names;
                self.collapse = r.collapse.clone();
                self.msg = String::new();
            }
            Err(e) => self.msg = format!("could not start explorer: {e}"),
        }
    }

    /// Kill the scoring run and reap it (no zombie left behind).
    fn stop(&mut self) {
        if let Some((mut c, ..)) = self.child.take() {
            let _ = c.kill();
            let _ = c.wait();
            self.msg = "stopped".into();
        }
    }

    fn poll(&mut self) {
        let Some((child, started, csv, log)) = &mut self.child else { return };
        match child.try_wait() {
            Ok(None) => {
                let tail = std::fs::read_to_string(&*log).unwrap_or_default();
                let prog = tail.rsplit('\r').next().unwrap_or("").trim().to_string();
                self.msg = format!("scoring… {prog} ({:.0}s)", started.elapsed().as_secs_f64());
            }
            Ok(Some(st)) => {
                let text = std::fs::read_to_string(&*csv).unwrap_or_default();
                let mut lines = text.lines();
                let header: Vec<String> = lines.next().unwrap_or("").split(',').map(String::from).collect();
                self.rows = lines.filter_map(|l| {
                    let cells: Vec<&str> = l.split(',').collect();
                    if cells.len() != header.len() || cells.len() < 2 { return None; }
                    let vals = header[2..].iter().zip(&cells[2..]).filter_map(|(h, c)| Some((h.clone(), c.parse::<f64>().ok()?))).collect();
                    Some(PreviewRow { file: cells[0].to_string(), gate: cells[1] == "1", vals })
                }).collect();
                self.msg = if self.rows.is_empty() {
                    let tail = std::fs::read_to_string(&*log).unwrap_or_default();
                    format!("no genome scored ({st}): {}", tail.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or(""))
                } else {
                    format!("{} genomes scored in {:.0}s — change the weights and the ranking updates at once", self.rows.len(), started.elapsed().as_secs_f64())
                };
                self.child = None;
            }
            Err(e) => { self.msg = e.to_string(); self.child = None; }
        }
    }

    fn total(&self, row: &PreviewRow, r: &Recipe) -> f64 {
        if r.gated() && !row.gate { return -5.0; }
        r.terms.iter().map(|t| t.weight * row.vals.get(&t.name).copied().unwrap_or(0.0)).sum()
    }

    fn texture(&mut self, ctx: &egui::Context, root: &Path, file: &str) -> Option<egui::TextureHandle> {
        let path = root.join(self.dir.trim()).join(Path::new(file).with_extension("png"));
        let key = path.to_string_lossy().to_string();
        if let Some(t) = self.textures.get(&key) { return t.clone(); }
        let tex = image::open(&path).ok().map(|img| {
            let img = img.thumbnail(150, 200).to_rgba8();
            let ci = egui::ColorImage::from_rgba_unmultiplied([img.width() as usize, img.height() as usize], img.as_raw());
            ctx.load_texture(key.clone(), ci, egui::TextureOptions::LINEAR)
        });
        self.textures.insert(key, tex.clone());
        tex
    }
}

// ── The window ──────────────────────────────────────────────────────────────────

pub struct RecipeUi {
    pub open: bool,
    work: Recipe,
    name_buf: String,
    search: String,
    spec_buf: String,
    spec_err: String,
    pub msg: String,
    delete_confirm: bool,
    preview: Preview,
    recipes: Recipes,
}

impl RecipeUi {
    pub fn new(prefs: &QuatEvoPrefs, root: &Path) -> Self {
        let recipes = Recipes::read(root);
        let work = from_prefs(prefs, &recipes);
        let mut s = RecipeUi {
            open: false, name_buf: String::new(), search: String::new(), spec_buf: String::new(), spec_err: String::new(),
            msg: String::new(), delete_confirm: false, work: Recipe::empty(""),
            preview: Preview { dir: prefs.pool_dir.clone(), limit: 24, ..Default::default() },
            recipes,
        };
        s.load(work);
        s
    }

    fn load(&mut self, r: Recipe) {
        // A built-in cannot be overwritten: leave the box empty for the copy's name.
        self.name_buf = if r.builtin { String::new() } else { r.name.clone() };
        self.spec_buf = r.spec();
        self.spec_err.clear();
        self.delete_confirm = false;
        self.work = r;
    }

    pub fn open_with(&mut self, r: Recipe, _root: &Path) {
        self.load(r);
        self.open = true;
    }

    pub fn busy(&self) -> bool { self.preview.child.is_some() }

    /// Draw the builder (its own native window). Returns true when it changed the run settings.
    pub fn show(&mut self, ctx: &egui::Context, root: &Path, prefs: &mut QuatEvoPrefs) -> bool {
        if !self.open { return false; }
        self.preview.poll();
        self.recipes.refresh_if_old(root);
        let mut changed = false;
        let mut open = true;
        let builder = egui::ViewportBuilder::default().with_title("Fitness recipes").with_inner_size([1180.0, 900.0]);
        ctx.show_viewport_immediate(egui::ViewportId::from_hash_of("recipe_window"), builder, |ui, _class| {
            if ui.input(|i| i.viewport().close_requested()) { open = false; }
            egui::CentralPanel::default().show(ui, |ui| {
                changed |= self.top_bar(ui, root, prefs);
                ui.separator();
                let run = run_context(prefs);
                ui.columns(2, |cols| {
                    egui::ScrollArea::vertical().id_salt("recipe_left").auto_shrink([false, false]).show(&mut cols[0], |ui| {
                        self.settings(ui, root);
                        ui.separator();
                        self.terms_table(ui);
                        ui.separator();
                        self.modules_and_advice(ui, run);
                        ui.separator();
                        self.preview_ui(ui, root, prefs);
                    });
                    egui::ScrollArea::vertical().id_salt("recipe_right").auto_shrink([false, false]).show(&mut cols[1], |ui| {
                        self.catalog(ui);
                    });
                });
            });
        });
        if self.preview.child.is_some() { ctx.request_repaint_after(std::time::Duration::from_millis(300)); }
        if !open {
            self.preview.stop();
        }
        self.open = open;
        changed
    }

    fn top_bar(&mut self, ui: &mut egui::Ui, root: &Path, prefs: &mut QuatEvoPrefs) -> bool {
        let mut changed = false;
        ui.horizontal_wrapped(|ui| {
            ui.label("Recipe:");
            let mut pick: Option<Recipe> = None;
            egui::ComboBox::from_id_salt("recipe_pick").selected_text(self.work.name.clone()).width(300.0).show_ui(ui, |ui| {
                for r in self.recipes.all() {
                    let resp = ui.selectable_label(self.work.name == r.name, if r.builtin { r.name.clone() } else { format!("  {}", r.name) });
                    if resp.clicked() { pick = Some(r.clone()); }
                }
            }).response.on_hover_text("Load a recipe into the builder (★ = built-in, from past runs that worked). Nothing changes for the runs until you press «Use for runs».");
            if let Some(r) = pick { self.load(r); }
            ui.add(egui::TextEdit::singleline(&mut self.name_buf).desired_width(170.0).hint_text(if self.work.builtin { "name for your copy" } else { "name to save under" }));
            if ui.button("💾 Save").on_hover_text("Save under the name in the box, in recipes/<name>.json (overwrites a saved recipe of that name).").clicked() {
                let name = self.name_buf.trim().to_string();
                if name.is_empty() { self.msg = "type a name first".into(); }
                else if fr::builtins().iter().any(|b| b.name == name) { self.msg = "built-in recipes are read-only: type a new name to save your version".into(); }
                else {
                    let mut r = self.work.clone();
                    r.name = name;
                    r.builtin = false;
                    match r.save(root) {
                        Ok(p) => { self.msg = format!("saved {}", p.strip_prefix(root).unwrap_or(&p).display()); self.work = r; self.recipes = Recipes::read(root); }
                        Err(e) => self.msg = e.to_string(),
                    }
                }
            }
            if ui.button("✨ New").on_hover_text("An empty recipe (slice, no view-range scan).").clicked() {
                self.load(Recipe::empty("new recipe"));
            }
            if ui.button("⟲ From run settings").on_hover_text("Load what the run settings use right now.").clicked() {
                self.load(from_prefs(prefs, &self.recipes));
            }
            ui.separator();
            let errors = self.work.advice(run_context(prefs)).iter().any(|a| a.level == Level::Error);
            if ui.add_enabled(!errors, egui::Button::new(RichText::new("✔ Use for runs").strong())).on_hover_text("Apply this recipe to the run settings (single runs, generate forever and campaigns).").clicked() {
                let mut r = self.work.clone();
                if !r.builtin && !self.name_buf.trim().is_empty() { r.name = self.name_buf.trim().to_string(); }
                self.msg = apply(&r, prefs, root);
                changed = true;
            }
            // Last, so saving (which makes the button appear) never shifts the others.
            ui.separator();
            let key = nnfractals::tuning::sanitize(&self.work.name);
            if !self.work.builtin && self.recipes.list.iter().any(|(k, r)| !r.builtin && *k == key) {
                if !self.delete_confirm {
                    if ui.button("🗑 Delete").clicked() { self.delete_confirm = true; }
                } else if ui.button("Really delete?").clicked() {
                    let _ = std::fs::remove_file(Recipe::path(&root.join(fr::RECIPES_DIR), &self.work.name));
                    self.msg = format!("deleted {}", self.work.name);
                    self.delete_confirm = false;
                    self.recipes = Recipes::read(root);
                }
            }
        });
        let (label, modified) = status(prefs, &self.recipes);
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new(format!("Runs use: {label}")).color(if modified { Color32::from_rgb(240, 170, 60) } else { Color32::LIGHT_GREEN }));
            if !self.msg.is_empty() { ui.label(RichText::new(&self.msg).small()); }
        });
        changed
    }

    fn settings(&mut self, ui: &mut egui::Ui, root: &Path) {
        let r = &mut self.work;
        if r.builtin {
            ui.label(RichText::new("Built-in recipe (read-only file): edit freely here, then save under a new name.").small().color(Color32::from_rgb(240, 210, 90)));
        }
        ui.label(RichText::new("What it is for").strong());
        ui.add(egui::TextEdit::multiline(&mut r.description).desired_rows(2).desired_width(f32::INFINITY).hint_text("a sentence about what this recipe should evolve"));
        ui.label(RichText::new("Evidence / notes").strong());
        ui.add(egui::TextEdit::multiline(&mut r.evidence).desired_rows(2).desired_width(f32::INFINITY).hint_text("where it comes from, how it did"));
        ui.horizontal_wrapped(|ui| {
            ui.label("Projection:");
            for m in nnfractals::quat_collapse::OFFERED {
                ui.selectable_value(&mut r.collapse, m.name().to_string(), m.name());
            }
            ui.label(RichText::new("(weights and the rating evidence below depend on it)").small().weak());
        });
        let renders = r.needs_render();
        ui.horizontal_wrapped(|ui| {
            ui.add_enabled(renders, egui::Checkbox::new(&mut r.auto_range, "auto view range"))
                .on_hover_text("Scan each new genome for the best distance to look at it from. It STARTS THE TASTE MODEL to judge the framings, even when the recipe has no taste term. Skipped when no term renders. The recipes that worked ran without it.")
                .on_disabled_hover_text("No term renders, so the view range cannot change a score: skipped.");
            ui.label("Seed pool:");
            ui.add(egui::TextEdit::singleline(&mut r.pool_dir).desired_width(170.0).hint_text("keep the current pool"))
                .on_hover_text("Suggested --pool-dir, set when the recipe is used (left alone when empty or missing).");
            super::launcher_tuning::folder_picker_button(ui, "recipe_pool", &mut r.pool_dir, root);
        });
        let taste = r.uses_taste();
        ui.horizontal_wrapped(|ui| {
            ui.add_enabled_ui(taste, |ui| {
                ui.label("Taste moments:");
                ui.add(egui::DragValue::new(&mut r.eval_phases).range(1..=16)).on_hover_text("T moments the taste term scores (half mean, half worst) under a time-sweeping projection.");
                ui.label("Flash penalty:");
                ui.add(egui::DragValue::new(&mut r.smooth_weight).range(0.0..=2.0).speed(0.02)).on_hover_text("Penalty for one big jump between consecutive T moments.");
                ui.label("Look diversity:");
                ui.add(egui::DragValue::new(&mut r.embed_diversity).range(0.0..=5.0).speed(0.05)).on_hover_text("Keep survivors and saved genomes visually different (taste-model embedding similarity). 1.0–1.5 gave 60 distinct looks.");
            });
            if !taste { ui.label(RichText::new("(taste-only settings: no taste term)").small().weak()); }
        });
    }

    fn terms_table(&mut self, ui: &mut egui::Ui) {
        ui.heading("Terms — what the GA rewards");
        ui.label(RichText::new("Fitness = Σ weight × term. Every term is about 0…1, so the weight is how much it counts; a − weight rewards a LOW value. Hover a name for details.").small().weak());
        let s3 = self.work.collapse == "s3";
        let shares = self.work.shares();
        let mut remove = None;
        let mut edited = false;
        egui::Grid::new("recipe_terms").num_columns(5).striped(true).spacing([8.0, 4.0]).show(ui, |ui| {
            ui.label(RichText::new("term").small().weak());
            ui.label(RichText::new("weight").small().weak());
            ui.label(RichText::new("share of the score").small().weak());
            ui.label(RichText::new(if s3 { "your S3 ratings" } else { "your slice ratings" }).small().weak());
            ui.label("");
            ui.end_row();
            for (i, t) in self.work.terms.iter_mut().enumerate() {
                let info = fr::info(&t.name);
                let col = info.map(|x| group_color(x.group)).unwrap_or(Color32::from_rgb(240, 90, 90));
                let label = ui.label(RichText::new(format!("■ {}", t.name)).color(col));
                match info {
                    Some(x) => { label.on_hover_text(term_hover(x)); }
                    None => { label.on_hover_text("Unknown term: the GA would stop at once. Remove it."); }
                }
                edited |= ui.add(egui::DragValue::new(&mut t.weight).speed(0.01).range(-20.0..=20.0).max_decimals(4)).changed();
                share_cell(ui, shares.get(i).copied().unwrap_or(0.0), t.weight);
                evidence_cell(ui, info.and_then(|x| if s3 { x.wins.1 } else { x.wins.0 }), Some(t.weight));
                if ui.small_button("✕").on_hover_text("Remove").clicked() { remove = Some(i); }
                ui.end_row();
            }
        });
        if let Some(i) = remove { self.work.terms.remove(i); edited = true; }
        if self.work.terms.is_empty() { ui.label(RichText::new("No terms yet: add some from the list on the right.").italics()); }
        group_bar(ui, &self.work);
        let (lo, hi) = self.work.score_range();
        ui.label(RichText::new(format!("possible score: {lo:.2} … {hi:.2}   (trial values of different recipes are not comparable)")).small().weak());
        // Raw text, two-way.
        ui.horizontal(|ui| {
            ui.label("As text:");
            if edited { self.spec_buf = self.work.spec(); }
            let r = ui.add(egui::TextEdit::singleline(&mut self.spec_buf).desired_width(f32::INFINITY).font(egui::TextStyle::Monospace))
                .on_hover_text("The --fitness-metric value. Paste one here to import it.");
            if r.changed() {
                match fr::parse_spec(&self.spec_buf) {
                    Ok(t) => { self.work.terms = t; self.spec_err.clear(); }
                    Err(e) => self.spec_err = e,
                }
            }
        });
        if !self.spec_err.is_empty() { ui.colored_label(level_color(Level::Error), &self.spec_err); }
    }

    fn modules_and_advice(&mut self, ui: &mut egui::Ui, run: RunContext) {
        ui.heading("What a run with it starts");
        for m in self.work.modules(run) {
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new(if m.on { "●" } else { "○" }).color(if m.on { Color32::LIGHT_GREEN } else { Color32::from_gray(110) }));
                ui.label(RichText::new(m.name).color(if m.on { ui.visuals().text_color() } else { Color32::from_gray(130) }));
                ui.label(RichText::new(format!("— {}", m.why)).small().weak());
            });
        }
        let passes = self.work.passes(run.probe_size);
        if !passes.is_empty() {
            egui::CollapsingHeader::new("Per-genome work").default_open(false).show(ui, |ui| {
                for (p, mpx, names) in passes {
                    ui.label(RichText::new(format!("{} — {}{}", p.label(), names.join(", "), if mpx > 0.0 { format!("  ({mpx:.2} Mpx)") } else { String::new() })).small());
                }
                ui.label(RichText::new("Terms sharing a computation cost it once. image_motion reuses the stills of image_entropy_band. Saved genomes always get the full metric sweep and thumbnails.").small().weak());
            });
        }
        let advice = self.work.advice(run);
        if !advice.is_empty() {
            ui.heading("Advice");
            for a in advice {
                let icon = match a.level { Level::Error => "✖", Level::Warn => "⚠", Level::Info => "ℹ" };
                ui.horizontal_wrapped(|ui| {
                    ui.label(RichText::new(icon).color(level_color(a.level)));
                    ui.label(RichText::new(a.text).color(level_color(a.level)));
                });
            }
        }
    }

    fn catalog(&mut self, ui: &mut egui::Ui) {
        ui.heading("Add a term");
        ui.horizontal(|ui| {
            ui.label("🔍");
            ui.add(egui::TextEdit::singleline(&mut self.search).desired_width(220.0).hint_text("search names and descriptions"));
            if !self.search.is_empty() && ui.small_button("✕").clicked() { self.search.clear(); }
        });
        let s3 = self.work.collapse == "s3";
        ui.label(RichText::new(format!("Bar = what your {} ratings say: right/green = you pick the HIGHER value, left/red = the LOWER one (share of rated pairs, 2026-10-17). + adds the term with a weight of that sign.", if s3 { "S3" } else { "slice" })).small().weak());
        let q = self.search.to_lowercase();
        let mut add: Option<(&'static str, f64)> = None;
        for g in Group::ALL {
            let items: Vec<&fr::TermInfo> = CATALOG.iter().filter(|t| t.group == g)
                .filter(|t| q.is_empty() || t.name.contains(&q) || t.what.to_lowercase().contains(&q) || t.notes.to_lowercase().contains(&q)).collect();
            if items.is_empty() { continue; }
            let used = items.iter().filter(|t| self.work.has(t.name)).count();
            let title = RichText::new(format!("{}  ({}{})", g.label(), items.len(), if used > 0 { format!(", {used} in use") } else { String::new() })).color(group_color(g));
            egui::CollapsingHeader::new(title).id_salt(g.short()).default_open(!q.is_empty() || matches!(g, Group::Picture | Group::Motion | Group::Sheet)).show(ui, |ui| {
                ui.label(RichText::new(g.blurb()).small().weak());
                for t in items {
                    let present = self.work.has(t.name);
                    ui.horizontal(|ui| {
                        if ui.add_enabled(!present, egui::Button::new("+").small()).on_hover_text(format!("Add with weight {}", t.suggest)).clicked() {
                            add = Some((t.name, t.suggest));
                        }
                        let name = ui.label(RichText::new(t.name).strong().color(if present { Color32::from_gray(120) } else { group_color(g) }));
                        name.on_hover_text(term_hover(t));
                        cost_badge(ui, t);
                        evidence_cell(ui, if s3 { t.wins.1 } else { t.wins.0 }, None);
                        if !t.caution.is_empty() { ui.label(RichText::new("⚠").color(level_color(Level::Warn))).on_hover_text(t.caution); }
                    });
                    ui.label(RichText::new(t.what).small());
                    ui.add_space(3.0);
                }
            });
        }
        if let Some((n, w)) = add {
            self.work.terms.push(Term { name: n.into(), weight: w });
            self.spec_buf = self.work.spec();
        }
    }

    fn preview_ui(&mut self, ui: &mut egui::Ui, root: &Path, prefs: &QuatEvoPrefs) {
        ui.heading("Try it on a folder");
        ui.label(RichText::new("Scores a random sample of a folder's genomes with this recipe's terms and ranks them, so you can see what it rewards before spending hours on a GA. Then move the weights: the ranking updates at once. (A GA explores far beyond any folder, so a good ranking here is necessary, not sufficient.)").small().weak());
        ui.horizontal_wrapped(|ui| {
            ui.label("Folder:");
            ui.add(egui::TextEdit::singleline(&mut self.preview.dir).desired_width(200.0));
            super::launcher_tuning::folder_picker_button(ui, "recipe_preview_dir", &mut self.preview.dir, root);
            ui.label("genomes:");
            ui.add(egui::DragValue::new(&mut self.preview.limit).range(2..=400));
            if self.preview.child.is_none() {
                if ui.button("▶ Score them").on_hover_text(if prefs.gpu { "Uses the GPU (a running GA will slow down a little)." } else { "GPU is off in the run settings: rendering terms will be slow." }).clicked() {
                    let r = self.work.clone();
                    self.preview.start(root, &r, prefs);
                }
            } else if ui.button("■ Stop").clicked() {
                self.preview.stop();
            }
        });
        if !self.preview.msg.is_empty() { ui.label(RichText::new(&self.preview.msg).small()); }
        if self.preview.rows.is_empty() { return; }
        let missing: Vec<&str> = self.work.terms.iter().filter(|t| t.weight != 0.0 && !self.preview.terms.contains(&t.name)).map(|t| t.name.as_str()).collect();
        if !missing.is_empty() {
            ui.colored_label(level_color(Level::Warn), format!("not scored in this preview (count as 0): {} — score again", missing.join(", ")));
        }
        if self.preview.collapse != self.work.collapse {
            ui.colored_label(level_color(Level::Warn), format!("scored with the {} projection — score again for {}", self.preview.collapse, self.work.collapse));
        }
        let mut order: Vec<(usize, f64)> = self.preview.rows.iter().enumerate().map(|(i, r)| (i, self.preview.total(r, &self.work))).collect();
        order.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let ctx = ui.ctx().clone();
        let per_row = ((ui.available_width() / 98.0).floor() as usize).max(1);
        egui::Grid::new("recipe_preview_grid").spacing([6.0, 6.0]).show(ui, |ui| {
            for (rank, (i, total)) in order.iter().enumerate() {
                let file = self.preview.rows[*i].file.clone();
                let tex = self.preview.texture(&ctx, root, &file);
                let row = &self.preview.rows[*i];
                let mut tip = format!("#{} {file}\ntotal {total:.3}{}\n", rank + 1, if !row.gate && self.work.gated() { "  (fails the image gate: −5)" } else { "" });
                let mut parts: Vec<(f64, String)> = self.work.terms.iter().map(|t| {
                    let v = row.vals.get(&t.name).copied().unwrap_or(0.0);
                    (t.weight * v, format!("{:+.3}  {} = {v:.3} × {:.3}", t.weight * v, t.name, t.weight))
                }).collect();
                parts.sort_by(|a, b| b.0.abs().partial_cmp(&a.0.abs()).unwrap_or(std::cmp::Ordering::Equal));
                for (_, p) in parts { tip.push_str(&p); tip.push('\n'); }
                ui.vertical(|ui| {
                    match &tex {
                        Some(t) => { ui.add(egui::Image::new(t).fit_to_exact_size(egui::vec2(90.0, 120.0))).on_hover_text(tip.trim_end()); }
                        None => { ui.add_sized([90.0, 120.0], egui::Label::new(RichText::new("no thumbnail").small().weak())).on_hover_text(tip.trim_end()); }
                    }
                    ui.label(RichText::new(format!("#{} {total:.2}", rank + 1)).small().monospace().color(if *total <= -5.0 { level_color(Level::Error) } else { ui.visuals().text_color() }));
                });
                if (rank + 1) % per_row == 0 { ui.end_row(); }
            }
        });
        ui.label(RichText::new("Hover a picture for each term's value × weight.").small().weak());
    }
}

fn term_hover(t: &fr::TermInfo) -> String {
    let ev = |w: Option<f32>, c: &str| w.map(|w| format!("{c}: the higher value wins {:.0}% of your rated pairs\n", w * 100.0)).unwrap_or_default();
    format!("{}\n\n{}{}\n{}{}cost: {}{}",
        t.what,
        if t.notes.is_empty() { String::new() } else { format!("{}\n", t.notes) },
        if t.caution.is_empty() { String::new() } else { format!("⚠ {}\n", t.caution) },
        ev(t.wins.0, "slice"), ev(t.wins.1, "S3"),
        t.pass.label(),
        if t.gated { "\nstarts the low-res image gate (blank / noise / T-inert genomes get −5 without the expensive scoring)" } else { "" })
}

fn cost_badge(ui: &mut egui::Ui, t: &fr::TermInfo) {
    let (text, col) = match t.pass {
        fr::Pass::Free => ("free".to_string(), Color32::from_gray(140)),
        fr::Pass::Field | fr::Pass::Planes => ("CPU".to_string(), Color32::from_rgb(90, 200, 200)),
        fr::Pass::Taste => ("taste model".to_string(), Color32::from_rgb(240, 210, 90)),
        p => (format!("{:.2} Mpx", p.mpx(128, 3)), Color32::from_rgb(110, 170, 240)),
    };
    ui.label(RichText::new(text).small().color(col)).on_hover_text(format!("{} (probe size 128, 3 taste moments)", t.pass.label()));
}

/// A small diverging bar: the share of rated pairs the higher value wins (50% = centre).
fn evidence_cell(ui: &mut egui::Ui, win: Option<f32>, weight: Option<f64>) {
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(70.0, 14.0), egui::Sense::hover());
    let p = ui.painter_at(rect);
    p.rect_filled(rect, 2.0, Color32::from_gray(30));
    let cx = rect.center().x;
    p.line_segment([egui::pos2(cx, rect.top()), egui::pos2(cx, rect.bottom())], egui::Stroke::new(1.0, Color32::from_gray(80)));
    let Some(w) = win else {
        p.text(rect.center(), egui::Align2::CENTER_CENTER, "no data", egui::FontId::proportional(9.0), Color32::from_gray(100));
        resp.on_hover_text("No rating comparison for this term.");
        return;
    };
    let d = ((w - 0.5) / 0.25).clamp(-1.0, 1.0) * rect.width() / 2.0;
    let bar = egui::Rect::from_min_max(egui::pos2(cx.min(cx + d), rect.top() + 2.0), egui::pos2(cx.max(cx + d), rect.bottom() - 2.0));
    p.rect_filled(bar, 1.0, if d >= 0.0 { Color32::from_rgb(110, 200, 120) } else { Color32::from_rgb(220, 110, 110) });
    p.text(rect.center(), egui::Align2::CENTER_CENTER, format!("{:.0}%", w * 100.0), egui::FontId::monospace(9.0), Color32::WHITE);
    let against = weight.is_some_and(|wt| (wt > 0.0 && w < 0.42) || (wt < 0.0 && w > 0.58));
    if against { p.rect_stroke(rect, 2.0, egui::Stroke::new(1.5, Color32::from_rgb(240, 170, 60)), egui::StrokeKind::Inside); }
    resp.on_hover_text(format!("The genome with the HIGHER value won {:.0}% of your rated pairs.{}", w * 100.0,
        if against { "\nOrange frame: this weight's sign goes against your ratings." } else { "" }));
}

fn share_cell(ui: &mut egui::Ui, share: f64, weight: f64) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(140.0, 14.0), egui::Sense::hover());
    let p = ui.painter_at(rect);
    p.rect_filled(rect, 2.0, Color32::from_gray(30));
    let w = rect.width() * share.clamp(0.0, 1.0) as f32;
    let bar = egui::Rect::from_min_size(rect.min, egui::vec2(w, rect.height()));
    p.rect_filled(bar, 2.0, if weight >= 0.0 { Color32::from_rgb(110, 170, 240) } else { Color32::from_rgb(220, 120, 140) });
    p.text(egui::pos2(rect.right() - 3.0, rect.center().y), egui::Align2::RIGHT_CENTER, format!("{}{:.0}%", if weight < 0.0 { "−" } else { "" }, share * 100.0), egui::FontId::monospace(9.0), Color32::WHITE);
}

/// The recipe's weight split by group, as one bar.
fn group_bar(ui: &mut egui::Ui, r: &Recipe) {
    let total: f64 = r.terms.iter().map(|t| t.weight.abs()).sum();
    if total <= 0.0 { return; }
    let mut by: Vec<(Group, f64)> = Vec::new();
    for t in &r.terms {
        let Some(i) = fr::info(&t.name) else { continue };
        match by.iter_mut().find(|(g, _)| *g == i.group) { Some(e) => e.1 += t.weight.abs(), None => by.push((i.group, t.weight.abs())) }
    }
    by.sort_by_key(|(g, _)| *g);
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(ui.available_width().min(520.0), 18.0), egui::Sense::hover());
    let p = ui.painter_at(rect);
    let mut x = rect.left();
    let mut tip = String::new();
    for (g, w) in &by {
        let seg = egui::Rect::from_min_max(egui::pos2(x, rect.top()), egui::pos2(x + rect.width() * (*w / total) as f32, rect.bottom()));
        x = seg.right();
        p.rect_filled(seg.shrink2(egui::vec2(0.5, 0.0)), 0.0, group_color(*g));
        if seg.width() > 60.0 { p.text(seg.center(), egui::Align2::CENTER_CENTER, format!("{} {:.0}%", g.short(), 100.0 * w / total), egui::FontId::monospace(10.0), Color32::BLACK); }
        tip.push_str(&format!("{}: {:.0}%\n", g.short(), 100.0 * w / total));
    }
    resp.on_hover_text(format!("Share of the total |weight| per kind of term:\n{}", tip.trim_end()));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build_quat_evolve_args;

    fn args_for(r: &Recipe, mode: QuatEvoMode) -> Vec<String> {
        let mut p = QuatEvoPrefs::default();
        p.mode = mode;
        apply(r, &mut p, Path::new("/nonexistent"));
        build_quat_evolve_args(&p, "/tmp/out", 1, 2)
    }

    #[test]
    fn flags_a_recipe_cannot_use_stay_off_the_command_line() {
        let mut r = Recipe::empty("planes");
        r.terms = vec![Term { name: "planes_smooth".into(), weight: 1.0 }, Term { name: "sphericity".into(), weight: -0.5 }];
        r.auto_range = true;
        r.embed_diversity = 1.0;
        r.collapse = "s3".into();
        r.smooth_weight = 0.3;
        let a = args_for(&r, QuatEvoMode::Blend);
        for f in ["--auto-range", "--embed-diversity", "--pref-model", "--smooth-weight"] {
            assert!(!a.iter().any(|x| x == f), "{f} passed for a recipe that cannot use it: {a:?}");
        }
        assert!(a.windows(2).any(|w| w[0] == "--fitness-metric" && w[1] == "planes_smooth:1,sphericity:-0.5"));

        let taste = fr::builtins().into_iter().find(|r| r.uses_taste()).unwrap();
        let a = args_for(&taste, QuatEvoMode::Blend);
        assert!(a.iter().any(|x| x == "--embed-diversity"), "taste recipes keep look diversity");
        assert!(!a.iter().any(|x| x == "--pref-model"));

        let campaign = &fr::builtins()[0];
        let mut with_scan = campaign.clone();
        with_scan.auto_range = true;
        assert!(args_for(&with_scan, QuatEvoMode::Blend).iter().any(|x| x == "--auto-range"), "a rendering recipe may scan");
        assert!(!args_for(campaign, QuatEvoMode::Blend).iter().any(|x| x == "--auto-range"));

        // No recipe: the default blend still gets its preference model.
        let mut p = QuatEvoPrefs::default();
        p.mode = QuatEvoMode::Blend;
        assert!(build_quat_evolve_args(&p, "/tmp/out", 1, 2).iter().any(|x| x == "--pref-model"));
    }

    #[test]
    fn applied_recipes_read_back_and_edits_show_as_modified() {
        let root = Path::new("/nonexistent");
        let rec = Recipes::read(root);
        for r in fr::builtins() {
            let mut p = QuatEvoPrefs::default();
            apply(&r, &mut p, root);
            assert!(matches(&r, &p), "{}", r.name);
            assert_eq!(status(&p, &rec), (r.name.clone(), false));
            let back = from_prefs(&p, &rec);
            assert!(fr::same_spec(&back.spec(), &r.spec()));
            p.fitness_metric.push_str(",node_count:0.1");
            assert!(status(&p, &rec).1, "an added term is a modification");
        }
        let mut p = QuatEvoPrefs::default();
        p.mode = QuatEvoMode::MapElites;
        assert_eq!(status(&p, &rec).0, "none — taste model only", "MAP-Elites without a recipe scores with taste");
        p.mode = QuatEvoMode::Blend;
        assert_eq!(status(&p, &rec).0, "none — default blend");
        p.collapse = "s3".into();
        assert_eq!(status(&p, &rec).0, "none — taste model only", "S3 without a recipe: the launcher adds the taste term");
        p.collapse = "slice".into();
        p.fitness_metric = "taste:0.5,grey_band:1".into();
        p.recipe.clear();
        assert_eq!(status(&p, &rec).0, "custom");
        p.fitness_metric = "taste".into();
        assert!(status(&p, &rec).0.starts_with("★ Taste model"), "same terms as a built-in: recognised");
    }

    #[test]
    fn settings_without_a_recipe_name_are_recognised_by_their_terms() {
        let rec = Recipes::read(Path::new("/nonexistent"));
        let camp = &fr::builtins()[0];
        let mut p = QuatEvoPrefs::default();
        // What the old "campaign recipe" button wrote: same terms, alphabetical order.
        let mut terms = camp.terms.clone();
        terms.sort_by(|a, b| a.name.cmp(&b.name));
        p.fitness_metric = terms.iter().map(|t| format!("{}:{}", t.name, t.weight)).collect::<Vec<_>>().join(",");
        p.auto_range = false;
        assert_eq!(status(&p, &rec), (camp.name.clone(), false));
        p.auto_range = true;
        assert_eq!(status(&p, &rec), (format!("{} (modified)", camp.name), true), "auto range starts the taste model: a real difference");
    }
}
