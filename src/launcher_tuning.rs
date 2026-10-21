//! Launcher window for GA hyperparameter tunings and Optuna campaigns (compiled into
//! `nnfractals-launcher` via `#[path]`; see `crate::tuning` for the data model and
//! `scripts/optuna_tuner.py` for the optimizer bridge).
//!
//! A *tuning* is edited as range bars: the full track is the parameter's hard limits, the
//! blue band is the range the optimizer may search, the dot is the value the next run
//! uses, and the orange tick is the best finished run's value. A *campaign* is an endless
//! loop of finite GA runs: ask Optuna for a tuning → (you may edit it) → run the GA with a
//! wall-clock limit and the stagnation stop → score → tell Optuna → repeat.

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

use eframe::egui::{self, Color32};
use nnfractals::tuning::{Campaign, Kind, Optimizer, Param, RunRecord, Tuning, TUNINGS_DIR};

use super::{build_quat_evolve_args, sibling, QuatEvoMode, QuatEvoPrefs};

const DEFAULT_MAX_MINUTES: f32 = 30.0;
/// A run still alive this long after its final contact sheet was written is hung at exit.
const HANG_GRACE: Duration = Duration::from_secs(60);

// ── Python bridge ───────────────────────────────────────────────────────────────

fn py_call(root: &Path, args: &[&str], stdin: Option<&serde_json::Value>) -> Result<serde_json::Value, String> {
    let mut child = Command::new(nnfractals::python_bin(root))
        .arg(root.join("scripts/optuna_tuner.py"))
        .args(args)
        .current_dir(root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("could not start python: {e}"))?;
    if let Some(mut si) = child.stdin.take() {
        if let Some(v) = stdin {
            let _ = si.write_all(v.to_string().as_bytes());
        }
    }
    let out = child.wait_with_output().map_err(|e| e.to_string())?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(err.lines().last().unwrap_or("optuna_tuner.py failed").to_string());
    }
    serde_json::from_slice(&out.stdout).map_err(|e| format!("bad optuna_tuner.py output: {e}"))
}

// ── Command line for one run ────────────────────────────────────────────────────

/// Drop every occurrence of `flags` (each followed by its value) from `args`.
fn strip_flags(args: Vec<String>, flags: &[&str]) -> Vec<String> {
    let mut out = Vec::with_capacity(args.len());
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        if flags.contains(&a.as_str()) {
            it.next();
        } else {
            out.push(a);
        }
    }
    out
}

/// `explorer quat-dag-evolve` arguments for `prefs` with `tuning` on top: the tuning wins for
/// every flag it controls, and `--max-seconds` comes from `prefs.max_minutes`. MAP-Elites
/// has no `--stop-flat`/`--save-top` path, so a tuned run always uses Blend or PredatorPrey.
pub fn run_args(prefs: &QuatEvoPrefs, tuning: &Tuning, out_dir: &str, seed: u64, generations: usize) -> Vec<String> {
    let mut p = prefs.clone();
    if p.mode == QuatEvoMode::MapElites {
        p.mode = QuatEvoMode::Blend;
    }
    p.generate_forever = false;
    let mut args = strip_flags(
        build_quat_evolve_args(&p, out_dir, seed, generations),
        &["--survivors", "--immigrant-frac", "--crossover-frac", "--crossover-mode", "--max-seconds", "--op-weights", "--random-immigrant-frac"],
    );
    if p.pool_dir.trim().is_empty() {
        args = strip_flags(args, &["--pool-dir"]);
    }
    args.extend(tuning.to_cli_args());
    if p.max_minutes > 0.0 {
        args.push("--max-seconds".into());
        args.push(format!("{:.0}", p.max_minutes * 60.0));
    }
    args
}

/// `{"workers": N}` from a campaign's control file.
fn read_workers(path: &Path) -> Option<usize> {
    serde_json::from_str::<serde_json::Value>(&std::fs::read_to_string(path).ok()?).ok()?.get("workers")?.as_u64().map(|n| n as usize)
}

/// `{"auto": bool}` from a campaign's control file.
fn read_auto(path: &Path) -> Option<bool> {
    serde_json::from_str::<serde_json::Value>(&std::fs::read_to_string(path).ok()?).ok()?.get("auto")?.as_bool()
}

/// The fitness recipe the Optuna campaigns have used so far: H1 with `image_motion` replaced
/// by the T-continuity, front-cover and grey bands (same construction as
/// `scripts/optuna_campaigns.py`).
pub fn default_campaign_recipe(root: &Path) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(root.join("taste_analysis/combo_specs.json")).ok()?).ok()?;
    let h1 = v.get("H1_s3_rho25")?.as_object()?;
    let band = h1.get("image_motion")?.as_f64()?;
    let mut parts: Vec<String> = h1.iter().filter(|(k, _)| k.as_str() != "image_motion")
        .filter_map(|(k, w)| Some(format!("{k}:{}", (w.as_f64()? * 1e4).round() / 1e4))).collect();
    for n in ["t_cont_band", "front_cover_band", "grey_band"] {
        parts.push(format!("{n}:{}", (band * 1e4).round() / 1e4));
    }
    Some(parts.join(","))
}

// ── Campaign runner (own thread) ────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Phase { Asking, ReadyForEdit, Running, Scoring, Stopped }

#[derive(Default)]
pub struct Shared {
    pub phase: Mutex<Option<Phase>>,
    pub status: Mutex<String>,
    /// (trial number, proposed values) waiting for the user's edit / go.
    pub proposal: Mutex<Option<(usize, serde_json::Map<String, serde_json::Value>)>>,
    /// Best-so-far fitness per generation of the run in progress.
    pub live: Mutex<Vec<f64>>,
    pub importances: Mutex<HashMap<String, f64>>,
    /// Bumped whenever `campaign.json` changed on disk.
    pub version: AtomicUsize,
    /// Finish the run in progress, then stop.
    pub stop: AtomicBool,
    /// Kill the run in progress and stop now.
    pub kill: AtomicBool,
    /// Skip the edit card: run each proposal as it comes.
    pub auto: AtomicBool,
}

impl Shared {
    fn set(&self, phase: Phase, status: impl Into<String>) {
        *self.phase.lock().unwrap() = Some(phase);
        *self.status.lock().unwrap() = status.into();
    }
}

/// Latest editor state (ranges, locks, run settings) and, for `go_trial`, the values to run.
pub struct Cmd {
    pub tuning: Tuning,
    pub prefs: QuatEvoPrefs,
    pub go_trial: Option<usize>,
}

fn same_values(a: &Tuning, b: &serde_json::Map<String, serde_json::Value>) -> bool {
    let mut t = a.clone();
    t.apply_values(b);
    t.params.iter().zip(&a.params).all(|(x, y)| (x.value - y.value).abs() < 1e-9)
}

/// Read the generation log of a run into best-so-far values.
fn read_live(out_dir: &Path) -> Vec<f64> {
    std::fs::read_to_string(out_dir.join("evolve_stats.jsonl")).unwrap_or_default().lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok()?.get("best_total")?.as_f64()).collect()
}

fn campaign_thread(root: PathBuf, name: String, mut tuning: Tuning, mut prefs: QuatEvoPrefs, rx: mpsc::Receiver<Cmd>, sh: Arc<Shared>) {
    let mut camp = Campaign::load(&root, &name).unwrap_or_else(|| Campaign::new(&name, tuning.clone()));
    let cdir = Campaign::dir(&root, &name);
    let camp_arg = cdir.to_string_lossy().to_string();
    let _ = std::fs::create_dir_all(cdir.join("runs"));
    let fail = |msg: String| sh.set(Phase::Stopped, msg);

    'campaign: loop {
        if sh.stop.load(Ordering::Relaxed) || sh.kill.load(Ordering::Relaxed) {
            break;
        }
        // ── ask ──
        while let Ok(c) = rx.try_recv() { tuning = c.tuning; prefs = c.prefs; }
        sh.set(Phase::Asking, "asking Optuna for the next tuning…");
        let req = serde_json::json!({ "optimizer": tuning.optimizer.key(), "params": tuning.params });
        let ask = match py_call(&root, &["ask", "--campaign", &camp_arg], Some(&req)) {
            Ok(v) => v,
            Err(e) => return fail(format!("Optuna ask failed: {e}")),
        };
        let mut trial = ask["trial"].as_u64().unwrap_or(0) as usize;
        let mut values = ask["params"].as_object().cloned().unwrap_or_default();
        *sh.proposal.lock().unwrap() = Some((trial, values.clone()));

        // ── wait for the user (or auto-continue) ──
        sh.set(Phase::ReadyForEdit, format!("trial {trial} proposed — edit it or press Run"));
        loop {
            if sh.stop.load(Ordering::Relaxed) || sh.kill.load(Ordering::Relaxed) {
                let _ = py_call(&root, &["tell", "--campaign", &camp_arg], Some(&serde_json::json!({ "trial": trial, "value": null })));
                break 'campaign;
            }
            let mut go = sh.auto.load(Ordering::Relaxed);
            match rx.recv_timeout(Duration::from_millis(400)) {
                Ok(c) => {
                    tuning = c.tuning.clone();
                    prefs = c.prefs.clone();
                    if c.go_trial == Some(trial) {
                        go = true;
                        let edited = c.tuning.values_json().as_object().cloned().unwrap_or_default();
                        if !same_values(&c.tuning, &values) {
                            // The study must record what actually runs: drop the proposal and
                            // enqueue the edited values as the trial.
                            let _ = py_call(&root, &["tell", "--campaign", &camp_arg], Some(&serde_json::json!({ "trial": trial, "value": null })));
                            let req = serde_json::json!({ "optimizer": tuning.optimizer.key(), "params": tuning.params, "override": edited });
                            match py_call(&root, &["ask", "--campaign", &camp_arg], Some(&req)) {
                                Ok(v) => {
                                    trial = v["trial"].as_u64().unwrap_or(0) as usize;
                                    values = v["params"].as_object().cloned().unwrap_or(edited);
                                }
                                Err(e) => return fail(format!("Optuna ask (edited) failed: {e}")),
                            }
                        }
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => return fail("launcher window closed".into()),
            }
            if go {
                break;
            }
        }
        *sh.proposal.lock().unwrap() = None;
        tuning.apply_values(&values);

        // ── run ──
        let work = cdir.join(format!("work_{trial:03}"));
        let _ = std::fs::remove_dir_all(&work);
        let seed: u64 = rand::random();
        if prefs.max_minutes <= 0.0 {
            prefs.max_minutes = DEFAULT_MAX_MINUTES;
        }
        let mut args = run_args(&prefs, &tuning, &work.to_string_lossy(), seed, prefs.generations);
        // Scoring threads can be changed while the run goes: the GUI writes `control.json` in the
        // campaign folder and the explorer re-reads it every second. The file wins over the prefs.
        let ctl = cdir.join("control.json");
        let workers = read_workers(&ctl).unwrap_or(prefs.workers).max(1);
        args = strip_flags(args, &["--workers"]);
        args.retain(|a| a != "--auto-workers");
        if read_auto(&ctl).unwrap_or(prefs.auto_workers) { args.push("--auto-workers".into()); }
        args.extend(["--workers".into(), workers.to_string(), "--control-file".into(), ctl.to_string_lossy().into_owned()]);
        let log_path = cdir.join(format!("run_{trial:03}.log"));
        let log = std::fs::File::create(&log_path).and_then(|f| Ok((f.try_clone()?, f)));
        let Ok((out_f, err_f)) = log else { return fail(format!("cannot write {}", log_path.display())) };
        let started = Instant::now();
        sh.live.lock().unwrap().clear();
        sh.set(Phase::Running, format!("trial {trial} running"));
        let mut child = match Command::new(sibling("explorer")).args(&args).current_dir(&root).stdout(out_f).stderr(err_f).spawn() {
            Ok(c) => c,
            Err(e) => return fail(format!("could not start explorer: {e}")),
        };
        let limit = Duration::from_secs_f64(prefs.max_minutes as f64 * 60.0 + 180.0);
        let (mut done_at, mut watchdog) = (None::<Instant>, false);
        loop {
            match child.try_wait() {
                Ok(Some(_)) | Err(_) => break,
                Ok(None) => {}
            }
            std::thread::sleep(Duration::from_secs(1));
            let live = read_live(&work);
            if let Some(b) = live.last() {
                *sh.status.lock().unwrap() = format!("trial {trial} running — gen {} best {b:.3} — {:.0}s", live.len(), started.elapsed().as_secs_f64());
            }
            *sh.live.lock().unwrap() = live;
            if done_at.is_none() {
                let tail = std::fs::read_to_string(&log_path).unwrap_or_default();
                if tail.contains("wrote contact sheet") { done_at = Some(Instant::now()); }
            }
            let hung = done_at.is_some_and(|t| t.elapsed() > HANG_GRACE) || started.elapsed() > limit;
            if hung || sh.kill.load(Ordering::Relaxed) {
                watchdog = hung;
                let _ = child.kill();
                let _ = child.wait();
                break;
            }
        }
        let killed = sh.kill.load(Ordering::Relaxed);
        let log_text = std::fs::read_to_string(&log_path).unwrap_or_default();
        let stop_reason = if killed { "killed" } else if watchdog { "watchdog" } else if log_text.contains("[max-seconds]") { "max-seconds" }
            else if log_text.contains("[stop-flat]") { "stop-flat" } else { "generations" };

        // ── score + tell ──
        sh.set(Phase::Scoring, format!("trial {trial} scoring"));
        let prefix = format!("trial{trial:03}");
        let scored = py_call(&root, &["score", "--dir", &work.to_string_lossy(), "--keep", &cdir.join("runs").to_string_lossy(), "--prefix", &prefix], None);
        let (value, stem) = match &scored {
            Ok(v) if !killed => (v["value"].as_f64(), v["stem"].as_str().map(String::from)),
            _ => (None, None),
        };
        let _ = py_call(&root, &["tell", "--campaign", &camp_arg], Some(&serde_json::json!({ "trial": trial, "value": value })));
        camp.tuning = tuning.clone();
        camp.runs.push(RunRecord { trial, value, params: tuning.values_json(), seed, secs: started.elapsed().as_secs_f64(), stop_reason: stop_reason.into(), stem });
        let _ = camp.save(&root);
        let _ = std::fs::remove_dir_all(&work);
        if let Ok(serde_json::Value::Object(m)) = py_call(&root, &["importances", "--campaign", &camp_arg], None) {
            *sh.importances.lock().unwrap() = m.into_iter().filter_map(|(k, v)| Some((k, v.as_f64()?))).collect();
        }
        sh.version.fetch_add(1, Ordering::Relaxed);
        if killed { break; }
        // Runs that die at once (bad folder, missing binary) would otherwise loop forever.
        if camp.runs.len() >= 3 && camp.runs.iter().rev().take(3).all(|r| r.value.is_none() && r.secs < 20.0) {
            sh.set(Phase::Stopped, "campaign stopped: 3 runs in a row failed at once (see the run_NNN.log files)");
            return;
        }
    }
    sh.set(Phase::Stopped, "campaign stopped");
}

// ── Widgets ─────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
enum Grab { Min, Max, Value }

/// One tuning row: name, range bar, value text, lock. Returns true if anything changed.
fn param_row(ui: &mut egui::Ui, p: &mut Param, ghost: Option<f64>, importance: Option<f64>, share: Option<f64>) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        let label = p.name.strip_prefix(nnfractals::tuning::OP_PREFIX).unwrap_or(&p.name);
        let hint = explain(&p.name);
        let r = ui.add_sized([112.0, 18.0], egui::Label::new(label));
        if !hint.is_empty() { r.on_hover_text(hint.clone()); }
        if p.kind == Kind::Choice {
            for (i, c) in p.choices.clone().iter().enumerate() {
                if ui.selectable_label(p.value as usize == i, c).clicked() {
                    p.value = i as f64;
                    changed = true;
                }
            }
            ui.add_space(300.0 - 90.0);
        } else {
            changed |= range_bar(ui, p, ghost);
            let text = match share {
                Some(s) => format!("{:>5.1}%  {}  [{:.2}…{:.2}]", s * 100.0, p.text(), p.min, p.max),
                None => format!("{}  [{:.3} … {:.3}]", p.text(), p.min, p.max),
            };
            let r = ui.add_sized([170.0, 18.0], egui::Label::new(egui::RichText::new(text).monospace().small()));
            if !hint.is_empty() { r.on_hover_text(hint); }
        }
        if ui.checkbox(&mut p.locked, "lock").on_hover_text("Locked: the optimizer keeps this value fixed.").changed() {
            changed = true;
        }
        let (r, _) = ui.allocate_exact_size(egui::vec2(46.0, 10.0), egui::Sense::hover());
        let w = importance.unwrap_or(0.0).clamp(0.0, 1.0) as f32;
        ui.painter().rect_filled(r, 2.0, Color32::from_gray(40));
        ui.painter().rect_filled(egui::Rect::from_min_size(r.min, egui::vec2(r.width() * w, r.height())), 2.0, Color32::from_rgb(120, 190, 120));
    });
    changed
}

fn range_bar(ui: &mut egui::Ui, p: &mut Param, ghost: Option<f64>) -> bool {
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(300.0, 22.0), egui::Sense::click_and_drag());
    let track = rect.shrink2(egui::vec2(8.0, 0.0));
    let x_of = |v: f64| track.left() + track.width() * p.track_of(v) as f32;
    let cy = rect.center().y;
    let painter = ui.painter_at(rect);
    painter.rect_filled(egui::Rect::from_center_size(egui::pos2(track.center().x, cy), egui::vec2(track.width(), 4.0)), 2.0, Color32::from_gray(55));
    let band = egui::Rect::from_min_max(egui::pos2(x_of(p.min), cy - 7.0), egui::pos2(x_of(p.max).max(x_of(p.min) + 1.0), cy + 7.0));
    painter.rect_filled(band, 3.0, Color32::from_rgba_unmultiplied(80, 140, 220, 110));
    for x in [band.left(), band.right()] {
        painter.line_segment([egui::pos2(x, cy - 9.0), egui::pos2(x, cy + 9.0)], egui::Stroke::new(2.0, Color32::from_rgb(110, 170, 240)));
    }
    if let Some(g) = ghost {
        let x = x_of(g);
        painter.line_segment([egui::pos2(x, cy - 10.0), egui::pos2(x, cy + 10.0)], egui::Stroke::new(2.0, Color32::from_rgb(240, 150, 60)));
    }
    let handle = if p.locked { Color32::from_gray(150) } else { Color32::from_rgb(235, 205, 90) };
    painter.circle_filled(egui::pos2(x_of(p.value), cy), 6.0, handle);

    let id = resp.id;
    let mut changed = false;
    if resp.drag_started() || resp.clicked() {
        // The press position, not the current one: a fast drag has already moved on.
        let press = ui.input(|i| i.pointer.press_origin()).or(resp.interact_pointer_pos());
        if let Some(pos) = press {
            let d = |v: f64| (x_of(v) - pos.x).abs();
            let grab = if d(p.value) <= d(p.min).min(d(p.max)) + 3.0 { Grab::Value }
                else if d(p.min) <= d(p.max) { Grab::Min } else { Grab::Max };
            ui.memory_mut(|m| m.data.insert_temp(id, match grab { Grab::Min => 0u8, Grab::Max => 1, Grab::Value => 2 }));
        }
    }
    if resp.dragged() || resp.clicked() {
        if let Some(pos) = resp.interact_pointer_pos() {
            let v = p.from_track(((pos.x - track.left()) / track.width().max(1.0)) as f64);
            let v = if p.kind == Kind::Float { v } else { v.round() };
            let grab = ui.memory(|m| m.data.get_temp::<u8>(id)).unwrap_or(2);
            match grab {
                0 => { p.min = v.min(p.max); if p.value < p.min { p.value = p.min; } }
                1 => { p.max = v.max(p.min); if p.value > p.max { p.value = p.max; } }
                _ => {
                    p.value = v;
                    p.min = p.min.min(v);   // dragging the value out of the band widens it
                    p.max = p.max.max(v);
                }
            }
            p.clamp();
            changed = true;
        }
    }
    changed
}

/// The run in progress: best fitness of the population per generation on its own scale
/// (it includes unchanged seeds, unlike the trial value, so the two are not comparable).
fn draw_live(painter: &egui::Painter, area: egui::Rect, _a: f64, _b: f64, live: &[f64]) {
    if live.len() < 2 { return; }
    let (lo, hi) = live.iter().fold((f64::MAX, f64::MIN), |(l, h), v| (l.min(*v), h.max(*v)));
    let span = (hi - lo).max(1e-9);
    let n = live.len() as f32;
    let pts: Vec<egui::Pos2> = live.iter().enumerate().map(|(i, v)| {
        egui::pos2(area.left() + area.width() * i as f32 / n, area.bottom() - area.height() * ((v - lo) / span) as f32)
    }).collect();
    painter.add(egui::Shape::line(pts, egui::Stroke::new(1.5, Color32::from_rgba_unmultiplied(120, 220, 120, 190))));
}

/// Dots = each run's value, line = best so far; the live curve of the run in progress is
/// drawn faintly at the right. Returns the index of a clicked run.
fn run_chart(ui: &mut egui::Ui, runs: &[RunRecord], live: &[f64]) -> Option<usize> {
    let size = egui::vec2(ui.available_width().clamp(300.0, 700.0), 110.0);
    let (rect, resp) = ui.allocate_exact_size(size, egui::Sense::click());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 3.0, Color32::from_gray(22));
    let vals: Vec<f64> = runs.iter().filter_map(|r| r.value).collect();
    let (lo, hi) = vals.iter().fold((f64::MAX, f64::MIN), |(l, h), v| (l.min(*v), h.max(*v)));
    // No scored run yet: neutral axis, not f64::MAX labels.
    let (lo, hi) = if vals.is_empty() { (0.0, 1.0) } else { (lo, hi) };
    let span = (hi - lo).max(1e-6);
    let inner = rect.shrink(8.0);
    let slots = (runs.len() + 1).max(2) as f32;
    let pt = |i: usize, v: f64| egui::pos2(inner.left() + inner.width() * (i as f32 + 0.5) / slots, inner.bottom() - inner.height() * ((v - lo) / span) as f32);
    let mut best = f64::MIN;
    let mut prev: Option<egui::Pos2> = None;
    let mut nearest: Option<(f32, usize)> = None;
    let click = resp.interact_pointer_pos().filter(|_| resp.clicked());
    for (i, r) in runs.iter().enumerate() {
        let Some(v) = r.value else { continue };
        best = best.max(v);
        let p = pt(i, v);
        painter.circle_filled(p, 3.5, Color32::from_rgb(110, 170, 240));
        let b = pt(i, best);
        if let Some(q) = prev { painter.line_segment([q, b], egui::Stroke::new(1.5, Color32::from_rgb(240, 150, 60))); }
        prev = Some(b);
        if let Some(c) = click {
            let d = (p - c).length();
            if nearest.is_none_or(|(bd, _)| d < bd) { nearest = Some((d, i)); }
        }
    }
    let right = egui::Rect::from_min_max(egui::pos2(inner.left() + inner.width() * (runs.len() as f32) / slots, inner.top()), inner.right_bottom());
    draw_live(&painter, right, 0.0, 1.0, live);
    painter.text(rect.left_top() + egui::vec2(6.0, 3.0), egui::Align2::LEFT_TOP, format!("{hi:.3}"), egui::FontId::monospace(10.0), Color32::GRAY);
    painter.text(rect.left_bottom() + egui::vec2(6.0, -3.0), egui::Align2::LEFT_BOTTOM, format!("{lo:.3}"), egui::FontId::monospace(10.0), Color32::GRAY);
    nearest.filter(|(d, _)| *d < 14.0).map(|(_, i)| i)
}

/// (op, share) in the tuning's display order.
fn shares_in_order(t: &Tuning) -> Vec<(String, f64)> { t.op_shares() }

/// One stacked bar: each op's share of the total weight, a distinct hue per op, the bigger
/// ones labelled. Hover a segment for its name and percentage.
fn share_bar(ui: &mut egui::Ui, shares: &[(String, f64)]) {
    let w = ui.available_width().clamp(300.0, 700.0);
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(w, 26.0), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 3.0, Color32::from_gray(22));
    let hover = resp.hover_pos();
    let mut x = rect.left();
    let n = shares.len().max(1) as f32;
    let mut tip = None;
    for (i, (name, s)) in shares.iter().enumerate() {
        let seg = egui::Rect::from_min_max(egui::pos2(x, rect.top()), egui::pos2((x + rect.width() * *s as f32).min(rect.right()), rect.bottom()));
        x = seg.right();
        if seg.width() < 0.5 { continue; }
        let hue = i as f32 / n;
        let col = egui::ecolor::Hsva::new(hue, if name == "bulb" || name == "kbc" { 0.85 } else { 0.45 }, 0.75, 1.0);
        painter.rect_filled(seg.shrink2(egui::vec2(0.5, 0.0)), 0.0, Color32::from(col));
        if seg.width() > 30.0 {
            painter.text(seg.center(), egui::Align2::CENTER_CENTER, name, egui::FontId::monospace(10.0), Color32::BLACK);
        }
        if hover.is_some_and(|h| seg.contains(h)) { tip = Some(format!("{name}: {:.1}%", s * 100.0)); }
    }
    if let Some(t) = tip { resp.on_hover_text(t); }
}

/// The hover text for a tuning row: what the setting does and which way to push it. Operation
/// rows (`op_*`) describe the operation itself and what raising its weight does.
fn explain(name: &str) -> String {
    if let Some(o) = name.strip_prefix(nnfractals::tuning::OP_PREFIX) {
        return format!(
            "OPERATION «{o}»: {}\\n\\nWEIGHT = how often a mutation or a random immigrant picks this operation, relative to all the others (the % beside the bar is its share of the total). 0 = never used; doubling it doubles its share.",
            op_explain(o));
    }
    match name {
        "survivors" => "How many of the best genomes of a generation are kept as parents of the next one (and survive unchanged). Few = greedy, converges fast but loses variety; many = slower, more diverse.",
        "immigrant_frac" => "Share of every new generation made of IMMIGRANTS: genomes that do not descend from the current population. They keep fresh ideas coming in and fight stagnation. The rest of the generation is crossover (crossover_frac) and mutation.",
        "random_immigrants" => "Of the immigrants, the share that are brand-new random programs built from the operation weights below. The rest are taken from the pool folder (earlier results). 0 = pool only, 1 = random only. This is how the operation weights shape fresh blood, not just mutations.",
        "random_size" => "Longest random immigrant in program nodes. Each one gets a length between 3 and this value (3 = z, c and one op is the smallest possible, so 1 and 2 behave like 3). Short = simple formulas, long = intricate ones. The format maximum is 24.",
        "crossover_frac" => "Share of every generation produced by crossover: two good parents are combined into one child. What is left after crossover and immigrants is made by mutation.",
        "crossover_mode" => "legacy: glue both whole parent programs under one new operation (add, sub, mul or div, see the combiner weights). The child is always bigger. subtree: cut a real piece out of one parent and graft it into the other. Finer and keeps size in check.",
        "min_edits" => "Fewest changes one mutation applies to a genome. A mutation makes between min_edits and min_edits + extra_edits changes, drawn at random.",
        "extra_edits" => "Extra changes a mutation may add on top of min_edits (0 = always exactly min_edits). More edits = bigger jumps, less like the parent.",
        "const_scale" => "How far a «perturb» edit moves a constant: each part is shifted by a random amount up to ± this. Small = fine tuning of a shape, large = new shapes.",
        "max_depth" => "Deepest chain of operations allowed (the longest path from an input to the result). Mutation will not grow past it, and random immigrants respect it. Deeper = more intricate but slower and more chaotic.",
        "w_perturb" => "EDIT KIND «perturb»: nudge a constant by ±const_scale (or turn a z or c leaf into a constant). Relative weight against the other edit kinds.",
        "w_swap" => "EDIT KIND «swap op»: replace one operation with a different one, chosen by the operation weights below. Relative weight against the other edit kinds.",
        "w_rewire" => "EDIT KIND «rewire»: re-point an input of an operation at a different earlier node, changing the structure but not the operations. Relative weight against the other edit kinds.",
        "w_grow" => "EDIT KIND «grow»: put a new operation on top of the result, combining it with another node (the program gets one node bigger). Relative weight against the other edit kinds.",
        "w_prune" => "EDIT KIND «prune»: delete the last node, making the program smaller (never below gen_prune_min). Relative weight against the other edit kinds.",
        "gen_prune_min" => "The prune edit never shrinks a program below this many nodes. Raise it to stop formulas collapsing to trivial ones.",
        "gen_max_nodes" => "Largest program (in nodes) that mutation, crossover and random immigrants may produce. The formula format allows at most 24. Lower = simpler, faster formulas.",
        "gen_comb_add" | "gen_comb_sub" | "gen_comb_mul" | "gen_comb_div" => "LEGACY CROSSOVER ONLY: chance, relative to the other three, that this operation (a+b, a−b, a×b, a÷b) is the one joining the two parent programs. Ignored by subtree crossover.",
        "gen_cut_pref_attempts" => "Subtree crossover tries up to 6 cut points. The first this many keep redrawing when the cut lands on a leaf (z, c or a constant), hoping for an operation, because swapping a lone leaf changes little. 0 = take the first point drawn.",
        "gen_cut_leaf_retries" => "How many redraws each of those leaf-avoiding tries may make before accepting a leaf cut point.",
        "gen_bulb_pow_min" => "Lowest power of a newly made bulb node. The power is drawn evenly between this and the maximum (swapped if reversed). Power 2 is a plain square, 8 is the classic Mandelbulb; higher = more lobes.",
        "gen_bulb_pow_max" => "Highest power of a newly made bulb node (see the minimum).",
        "gen_bulb_twist" => "Chance that a new bulb gets a different angular power than radial power. 0 = both the same, the classic Mandelbulb. Above 0 gives twisted, spiralling bulbs.",
        "gen_kbc_scale" => "Size of a new kbc constant: its B and C parts are each uniform in ± this. Larger = shifts the shape further off the R-A plane.",
        "gen_const_range" => "Fresh random constants (new constant leaves, leaves turned into constants) are drawn uniformly in ± this, real and imaginary part. Larger = stronger off-centre shifts and more varied shapes.",
        "gen_const_start" => "Chance that a random immigrant program starts with a constant node.",
        "gen_const_node" => "At each step of building a random immigrant, the chance of adding a constant instead of an operation.",
        "gen_julia_prob" => "Chance a random immigrant is a Julia set: c is fixed and the pixel becomes the starting z, instead of the Mandelbrot way.",
        "gen_julia_scale" => "Size of a random immigrant's fixed Julia constant: each part uniform in ± this.",
        "gen_phoenix_prob" => "Chance a random immigrant has a phoenix term (a memory of the previous iterate added in), which gives layered, feathery structure.",
        "gen_phoenix_scale" => "Size of the phoenix constants of a random immigrant: each part uniform in ± this.",
        "gen_warp_prob" => "Chance a random immigrant gets a small warp program that bends the input plane before iterating.",
        "gen_bailout_min" => "Lowest bailout radius for random immigrants: the distance at which an orbit counts as escaped. Small = coarse, blobby edges; large = finer detail.",
        "gen_bailout_max" => "Highest bailout radius for random immigrants (see the minimum).",
        _ => "",
    }.to_string()
}

fn op_explain(op: &str) -> &'static str {
    match op {
        "sqr" => "a² — squares the value. The classic Mandelbrot step; the base of most fractals.",
        "cube" => "a³ — cubes the value. Three-fold symmetry and more lobes than a square.",
        "quart" => "a⁴ — fourth power. Four-fold symmetry; escapes faster.",
        "recip" => "1/a — inversion. Swaps inside and outside and makes poles near zero: sharp, unstable structure.",
        "sin" => "sin(a) — periodic and bounded: repeating bands and tiled, lattice-like shapes.",
        "cos" => "cos(a) — like sin, shifted: periodic, bounded, tiled shapes.",
        "exp" => "e^a — exponential. Explosive growth (orbits escape quickly), which makes sharp edges and spirals.",
        "log" => "log(a), guarded — compresses large values and tames growth; makes swirling folds.",
        "tanh" => "tanh(a) — squashes the value into a bounded range, so orbits rarely escape: stable, filled-in blobs.",
        "conj" => "Conjugate — flips the sign of the imaginary parts, a mirror image. Adds mirror symmetry.",
        "absfold" => "(|re|, |im|) — absolute value of both parts, the Burning Ship fold: jagged ship-like shapes with mirror symmetry.",
        "absre" => "(|re|, im) — folds only the real part: one-sided mirror, half a burning ship.",
        "absim" => "(re, |im|) — folds only the imaginary part: the other half-fold.",
        "normz" => "a/|a| — keeps only the direction, length becomes 1. Orbits stay bounded, behaving like pure rotations.",
        "add" => "a + b — adds two earlier results. The basic way to combine pieces.",
        "sub" => "a − b — subtracts. Like add, but gives lopsided, non-mirrored combinations.",
        "mul" => "a × b — multiplies. Couples two results together; for quaternions it does not commute, so order twists the shape.",
        "div" => "a ÷ b — divides. Poles where b is near zero give sharp, spiky structure.",
        "kbcmul" => "x × kbc: multiplies a value by a constant in the B and C parts (the multiplying twin of kbc, which adds it). Multiplying by a B/C constant rotates and mixes the parts of the value, so it twists the shape instead of shifting it. Made like kbc: a multiply node with its own fresh constant, spliced in after some node, with the constant on either side at random (quaternion products depend on order). Size set by gen_kbc_scale.",
        "bulb" => "Generalised Mandelbulb power: raises a to a radial power and multiplies its angles, parameters taken from b. The classic 3D Mandelbulb is power 8. A new bulb is created with its own constant node (power range set in the Bulb group).",
        "kbc" => "A constant in the B and C parts, added as x + kbc. Plain constants only fill R and A, which leaves the B and C directions interchangeable and produces ring-like balls; kbc breaks that symmetry and shifts the shape off-axis.",
        _ => "",
    }
}

// ── Grouped sections with a picture each ────────────────────────────────────────

#[derive(Clone, Copy)]
enum Graphic { Breeding, Crossover, Mutation, Bulb, Constants, Immigrants }

const GROUPS: &[(&str, &[&str], Graphic)] = &[
    ("Breeding — who the next generation is made of", &["survivors", "immigrant_frac", "random_immigrants", "gen_max_nodes"], Graphic::Breeding),
    ("Crossover — how two parents are joined", &["crossover_frac", "crossover_mode", "gen_comb_add", "gen_comb_sub", "gen_comb_mul", "gen_comb_div", "gen_cut_pref_attempts", "gen_cut_leaf_retries"], Graphic::Crossover),
    ("Mutation — how many edits, and which kind", &["min_edits", "extra_edits", "max_depth", "gen_prune_min", "const_scale", "w_perturb", "w_swap", "w_rewire", "w_grow", "w_prune"], Graphic::Mutation),
    ("Bulb — power of a new bulb node", &["gen_bulb_pow_min", "gen_bulb_pow_max", "gen_bulb_twist"], Graphic::Bulb),
    ("Constants — size of fresh constants and kbc", &["gen_kbc_scale", "gen_const_range", "gen_const_start", "gen_const_node"], Graphic::Constants),
    ("Random immigrants — size and dynamics", &["random_size", "gen_julia_prob", "gen_julia_scale", "gen_phoenix_prob", "gen_phoenix_scale", "gen_warp_prob", "gen_bailout_min", "gen_bailout_max"], Graphic::Immigrants),
];

fn tv(t: &Tuning, name: &str) -> f64 { t.get(name).map(|p| p.value).unwrap_or(0.0) }

/// A horizontal bar split into labelled coloured segments (fractions summing to ~1).
fn segment_bar(ui: &mut egui::Ui, title: &str, segs: &[(&str, f64, Color32)]) {
    ui.horizontal(|ui| {
        ui.add_sized([112.0, 18.0], egui::Label::new(egui::RichText::new(title).small().weak()));
        let (rect, resp) = ui.allocate_exact_size(egui::vec2(420.0, 18.0), egui::Sense::hover());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 3.0, Color32::from_gray(22));
        let mut x = rect.left();
        let mut tip = String::new();
        for (name, f, col) in segs {
            let w = rect.width() * (*f as f32).clamp(0.0, 1.0);
            let seg = egui::Rect::from_min_max(egui::pos2(x, rect.top()), egui::pos2((x + w).min(rect.right()), rect.bottom()));
            x = seg.right();
            if seg.width() < 0.5 { continue; }
            painter.rect_filled(seg.shrink2(egui::vec2(0.5, 0.0)), 0.0, *col);
            if seg.width() > 56.0 {
                painter.text(seg.center(), egui::Align2::CENTER_CENTER, format!("{name} {:.0}%", f * 100.0), egui::FontId::monospace(10.0), Color32::BLACK);
            }
            tip.push_str(&format!("{name}: {:.1}%\n", f * 100.0));
        }
        resp.on_hover_text(tip.trim_end());
    });
}

/// A number line with the interval `[lo, hi]` highlighted.
fn interval_bar(ui: &mut egui::Ui, title: &str, axis: (f64, f64), lo: f64, hi: f64) {
    let (lo, hi) = (lo.min(hi), lo.max(hi));
    ui.horizontal(|ui| {
        ui.add_sized([112.0, 18.0], egui::Label::new(egui::RichText::new(title).small().weak()));
        let (rect, _) = ui.allocate_exact_size(egui::vec2(420.0, 22.0), egui::Sense::hover());
        let painter = ui.painter_at(rect);
        let x_of = |v: f64| rect.left() + 6.0 + (rect.width() - 12.0) * (((v - axis.0) / (axis.1 - axis.0)).clamp(0.0, 1.0)) as f32;
        let cy = rect.center().y;
        painter.line_segment([egui::pos2(x_of(axis.0), cy), egui::pos2(x_of(axis.1), cy)], egui::Stroke::new(2.0, Color32::from_gray(70)));
        let band = egui::Rect::from_min_max(egui::pos2(x_of(lo), cy - 6.0), egui::pos2(x_of(hi).max(x_of(lo) + 2.0), cy + 6.0));
        painter.rect_filled(band, 3.0, Color32::from_rgb(235, 170, 70));
        painter.text(egui::pos2(rect.left() + 6.0, rect.bottom()), egui::Align2::LEFT_BOTTOM, format!("{}", axis.0), egui::FontId::monospace(9.0), Color32::GRAY);
        painter.text(egui::pos2(rect.right() - 6.0, rect.bottom()), egui::Align2::RIGHT_BOTTOM, format!("{}", axis.1), egui::FontId::monospace(9.0), Color32::GRAY);
        painter.text(band.center(), egui::Align2::CENTER_CENTER, format!("{lo:.1}…{hi:.1}"), egui::FontId::monospace(9.0), Color32::BLACK);
    });
}

/// Squares in the complex plane, all on one scale, showing how big each kind of constant is.
fn constants_plane(ui: &mut egui::Ui, t: &Tuning) {
    let sizes = [
        ("fresh constants", tv(t, "gen_const_range"), Color32::from_rgb(110, 170, 240)),
        ("kbc", tv(t, "gen_kbc_scale"), Color32::from_rgb(235, 170, 70)),
        ("mutation nudge", tv(t, "const_scale"), Color32::from_rgb(130, 210, 130)),
    ];
    let top = sizes.iter().fold(0.1f64, |m, s| m.max(s.1));
    ui.horizontal(|ui| {
        ui.add_sized([112.0, 18.0], egui::Label::new(egui::RichText::new("constants (re × im)").small().weak()));
        let (rect, resp) = ui.allocate_exact_size(egui::vec2(120.0, 120.0), egui::Sense::hover());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 3.0, Color32::from_gray(22));
        let c = rect.center();
        painter.line_segment([egui::pos2(rect.left(), c.y), egui::pos2(rect.right(), c.y)], egui::Stroke::new(1.0, Color32::from_gray(60)));
        painter.line_segment([egui::pos2(c.x, rect.top()), egui::pos2(c.x, rect.bottom())], egui::Stroke::new(1.0, Color32::from_gray(60)));
        for (_, v, col) in sizes.iter().rev() {
            let h = 56.0 * (*v / top) as f32;
            painter.rect_stroke(egui::Rect::from_center_size(c, egui::vec2(2.0 * h, 2.0 * h)), 0.0, egui::Stroke::new(2.0, *col), egui::StrokeKind::Inside);
        }
        resp.on_hover_text("Each square is the ± range a constant is drawn from (same scale for all).");
        ui.vertical(|ui| {
            for (name, v, col) in sizes { ui.label(egui::RichText::new(format!("■ {name}  ±{v:.2}")).color(col).small()); }
            ui.label(egui::RichText::new(format!("scale: outer edge = ±{top:.2}")).small().weak());
        });
    });
}

fn draw_group_graphic(ui: &mut egui::Ui, g: Graphic, t: &Tuning) {
    let desc = match g {
        Graphic::Breeding => "Each generation is built from survivors' offspring (crossover and mutation) plus immigrants. These set the proportions.",
        Graphic::Crossover => "Crossover combines two good parents into one child. Hover any row for what it does.",
        Graphic::Mutation => "Mutation changes one parent a little. Each mutation makes a few edits of five kinds; the weights pick which kind.",
        Graphic::Bulb => "A bulb node raises a value to a power while multiplying its angles: the 3D Mandelbulb. These set the power of each new bulb.",
        Graphic::Constants => "Constants are numbers baked into a formula. These set how big freshly drawn ones are, and how big a mutation's nudge is.",
        Graphic::Immigrants => "Random immigrants are fresh formulas built from the operation weights. These set their size and extra dynamics (Julia, phoenix, warp, bailout).",
    };
    ui.label(egui::RichText::new(desc).small().weak());
    let blue = Color32::from_rgb(110, 170, 240);
    let amber = Color32::from_rgb(235, 170, 70);
    let green = Color32::from_rgb(130, 210, 130);
    let rose = Color32::from_rgb(220, 120, 140);
    match g {
        Graphic::Breeding => {
            let imm = tv(t, "immigrant_frac").clamp(0.0, 1.0);
            let cross = tv(t, "crossover_frac").clamp(0.0, 1.0 - imm);
            let rnd = tv(t, "random_immigrants").clamp(0.0, 1.0);
            segment_bar(ui, "next generation", &[("crossover", cross, blue), ("mutation", (1.0 - imm - cross).max(0.0), green), ("random imm.", imm * rnd, rose), ("pool imm.", imm * (1.0 - rnd), amber)]);
        }
        Graphic::Crossover => {
            let w = [tv(t, "gen_comb_add").max(0.0), tv(t, "gen_comb_sub").max(0.0), tv(t, "gen_comb_mul").max(0.0), tv(t, "gen_comb_div").max(0.0)];
            let total = w.iter().sum::<f64>().max(1e-9);
            segment_bar(ui, "legacy combiner", &[("add", w[0] / total, blue), ("sub", w[1] / total, amber), ("mul", w[2] / total, green), ("div", w[3] / total, rose)]);
        }
        Graphic::Mutation => {
            let w: Vec<(&str, f64)> = [("perturb", "w_perturb"), ("swap op", "w_swap"), ("rewire", "w_rewire"), ("grow", "w_grow"), ("prune", "w_prune")]
                .iter().map(|(l, n)| (*l, tv(t, n).max(0.0))).collect();
            let total: f64 = w.iter().map(|x| x.1).sum::<f64>().max(1e-9);
            let cols = [blue, amber, green, rose, Color32::from_rgb(180, 150, 230)];
            let segs: Vec<(&str, f64, Color32)> = w.iter().zip(cols).map(|((l, v), c)| (*l, v / total, c)).collect();
            segment_bar(ui, "edit kinds", &segs);
            let lo = tv(t, "min_edits");
            interval_bar(ui, "edits per child", (0.0, 8.0), lo, lo + tv(t, "extra_edits"));
        }
        Graphic::Bulb => {
            interval_bar(ui, "power of a new bulb", (0.0, 16.0), tv(t, "gen_bulb_pow_min"), tv(t, "gen_bulb_pow_max"));
            let tw = tv(t, "gen_bulb_twist").clamp(0.0, 1.0);
            segment_bar(ui, "angular power", &[("same as radial", 1.0 - tw, blue), ("independent", tw, amber)]);
        }
        Graphic::Constants => constants_plane(ui, t),
        Graphic::Immigrants => {
            interval_bar(ui, "program length", (0.0, 24.0), 3.0, tv(t, "random_size"));
            interval_bar(ui, "bailout radius", (0.0, 50.0), tv(t, "gen_bailout_min"), tv(t, "gen_bailout_max"));
            segment_bar(ui, "chance of", &[("julia", tv(t, "gen_julia_prob") / 3.0, blue), ("phoenix", tv(t, "gen_phoenix_prob") / 3.0, green), ("warp", tv(t, "gen_warp_prob") / 3.0, rose)]);
            ui.label(egui::RichText::new("(chance of: each third of the bar = 100% for that feature)").small().weak());
        }
    }
}

// ── Window state ────────────────────────────────────────────────────────────────

/// What the launcher should do after a frame of the window.
pub enum Action {
    /// Run one GA now with these args (no campaign).
    RunOnce(Vec<String>),
}

pub struct TuningUi {
    pub open: bool,
    tuning: Tuning,
    names: Vec<String>,
    name_buf: String,
    camp: Option<Campaign>,
    camp_names: Vec<String>,
    camp_buf: String,
    /// Last read of the running explorer's thread report (when, text).
    workers_status: (Instant, Option<String>),
    shared: Option<Arc<Shared>>,
    tx: Option<mpsc::Sender<Cmd>>,
    last_sent: String,
    seen_version: usize,
    applied_trial: Option<usize>,
    delete_confirm: bool,
    msg: String,
}

impl TuningUi {
    pub fn new(root: &Path) -> Self {
        let dir = root.join(TUNINGS_DIR);
        let mut names = Tuning::list(&dir);
        if names.is_empty() {
            let _ = Tuning::defaults("default").save(&dir);
            names = Tuning::list(&dir);
        }
        let first = names.first().cloned().unwrap_or_else(|| "default".into());
        let tuning = Tuning::load(&Tuning::path(&dir, &first)).unwrap_or_else(|| Tuning::defaults("default"));
        TuningUi {
            open: false, name_buf: tuning.name.clone(), tuning, names, camp: None, camp_names: Campaign::list(root), camp_buf: String::new(), workers_status: (Instant::now(), None),
            shared: None, tx: None, last_sent: String::new(), seen_version: 0, applied_trial: None, delete_confirm: false, msg: String::new(),
        }
    }

    fn refresh_names(&mut self, root: &Path) {
        self.names = Tuning::list(&root.join(TUNINGS_DIR));
        self.camp_names = Campaign::list(root);
    }

    fn running(&self) -> bool {
        self.shared.as_ref().is_some_and(|s| !matches!(*s.phase.lock().unwrap(), None | Some(Phase::Stopped)))
    }

    fn start_campaign(&mut self, root: &Path, prefs: &mut QuatEvoPrefs) {
        let name = nnfractals::tuning::sanitize(self.camp_buf.trim());
        if name.is_empty() || self.camp_buf.trim().is_empty() {
            self.msg = "name the campaign first".into();
            return;
        }
        if prefs.fitness_metric.trim().is_empty() {
            self.msg = "a campaign needs a fitness metric (run settings below) — scores come from it".into();
            return;
        }
        if prefs.max_minutes <= 0.0 { prefs.max_minutes = DEFAULT_MAX_MINUTES; }
        if prefs.save_top < 5 { prefs.save_top = 5; }
        if prefs.mode == QuatEvoMode::MapElites { prefs.mode = QuatEvoMode::Blend; }
        let (tx, rx) = mpsc::channel();
        let shared = Arc::new(Shared::default());
        shared.set(Phase::Asking, "starting…");
        let (r, t, p, s) = (root.to_path_buf(), self.tuning.clone(), prefs.clone(), shared.clone());
        let n = name.clone();
        std::thread::spawn(move || campaign_thread(r, n, t, p, rx, s));
        self.camp = Campaign::load(root, &name);
        self.shared = Some(shared);
        self.tx = Some(tx);
        self.applied_trial = None;
        self.last_sent.clear();
        self.seen_version = 0;
        self.msg = format!("campaign {name} started");
    }

    /// Draw the window. `prefs` are the launcher's own run settings (edited here too).
    pub fn show(&mut self, ctx: &egui::Context, root: &Path, prefs: &mut QuatEvoPrefs, ga_running: bool) -> Option<Action> {
        if !self.open { return None; }
        let mut action = None;
        let mut open = self.open;

        // Pull state from the campaign thread.
        let mut phase = None;
        let (mut status, mut live, mut importances) = (String::new(), Vec::new(), HashMap::new());
        if let Some(sh) = &self.shared {
            phase = *sh.phase.lock().unwrap();
            status = sh.status.lock().unwrap().clone();
            if phase == Some(Phase::Running) { live = sh.live.lock().unwrap().clone(); }
            importances = sh.importances.lock().unwrap().clone();
            if let Some((trial, vals)) = sh.proposal.lock().unwrap().clone() {
                if self.applied_trial != Some(trial) {
                    self.tuning.apply_values(&vals);
                    self.applied_trial = Some(trial);
                }
            }
            let v = sh.version.load(Ordering::Relaxed);
            if v != self.seen_version {
                self.seen_version = v;
                if !self.camp_buf.trim().is_empty() {
                    self.camp = Campaign::load(root, &nnfractals::tuning::sanitize(self.camp_buf.trim()));
                }
            }
            if phase == Some(Phase::Stopped) && self.tx.is_some() {
                self.tx = None;
                self.msg = status.clone();
            }
        }
        let camp_active = self.tx.is_some();
        let ready = phase == Some(Phase::ReadyForEdit);

        // Its own native window: the launcher window is too small for the bars.
        let builder = egui::ViewportBuilder::default().with_title("Tuning & campaigns").with_inner_size([880.0, 940.0]);
        ctx.show_viewport_immediate(egui::ViewportId::from_hash_of("tuning_window"), builder, |ui, _class| {
            if ui.input(|i| i.viewport().close_requested()) {
                open = false;
            }
            egui::CentralPanel::default().show(ui, |ui| {
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                // ── tuning picker ──
                ui.horizontal_wrapped(|ui| {
                    ui.label("Tuning:");
                    let mut pick: Option<String> = None;
                    egui::ComboBox::from_id_salt("tuning_pick").selected_text(self.tuning.name.clone()).show_ui(ui, |ui| {
                        for n in &self.names { if ui.selectable_label(*n == self.tuning.name, n).clicked() { pick = Some(n.clone()); } }
                    });
                    if let Some(n) = pick {
                        if let Some(t) = Tuning::load(&Tuning::path(&root.join(TUNINGS_DIR), &n)) { self.name_buf = t.name.clone(); self.tuning = t; }
                    }
                    ui.add(egui::TextEdit::singleline(&mut self.name_buf).desired_width(130.0)).on_hover_text("Name to save under.");
                    if ui.button("💾 Save").on_hover_text("Save as the name in the box (overwrites).").clicked() {
                        self.tuning.name = self.name_buf.trim().to_string();
                        if self.tuning.name.is_empty() { self.tuning.name = "tuning".into(); }
                        self.msg = match self.tuning.save(&root.join(TUNINGS_DIR)) { Ok(()) => format!("saved tunings/{}.json", nnfractals::tuning::sanitize(&self.tuning.name)), Err(e) => e.to_string() };
                        self.refresh_names(root);
                    }
                    if ui.button("✨ New").on_hover_text("Default values and ranges.").clicked() {
                        let opt = self.tuning.optimizer;
                        self.tuning = Tuning::defaults(self.name_buf.trim()); self.tuning.optimizer = opt;
                    }
                    if ui.button("🎲 Random").on_hover_text("Draw every unlocked value at random inside its range. Then edit what you like.").clicked() {
                        self.tuning.randomize(&mut rand::rng());
                    }
                    if ui.button("↺ Reset ranges").on_hover_text("Default ranges, keep the current values where they fit.").clicked() {
                        let d = Tuning::defaults("d");
                        for p in &mut self.tuning.params { if let Some(q) = d.get(&p.name) { p.min = q.min; p.max = q.max; p.lo = q.lo; p.hi = q.hi; p.clamp(); } }
                    }
                    if !self.delete_confirm {
                        if ui.button("🗑 Delete").clicked() { self.delete_confirm = true; }
                    } else if ui.button("Really delete?").clicked() {
                        let _ = std::fs::remove_file(Tuning::path(&root.join(TUNINGS_DIR), &self.tuning.name));
                        self.delete_confirm = false;
                        self.refresh_names(root);
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Optimizer:");
                    egui::ComboBox::from_id_salt("optimizer").selected_text(self.tuning.optimizer.label()).show_ui(ui, |ui| {
                        for o in Optimizer::ALL { ui.selectable_value(&mut self.tuning.optimizer, o, o.label()); }
                    }).response.on_hover_text("TPE: default, good with ~30+ runs. CMA-ES: continuous search (needs `pip install cmaes`). Random/QMC: baselines and exploration. GP: few runs, slower to suggest.");
                    ui.label(egui::RichText::new("track = hard limits · blue band = search range · dot = value · orange tick = best run · green bar = importance").small().weak());
                });
                ui.separator();

                // ── the bars ──
                let ghost: Option<serde_json::Value> = self.camp.as_ref().and_then(|c| c.best()).map(|r| r.params.clone());
                let max_imp = importances.values().cloned().fold(1e-9, f64::max);
                let mut edited = false;
                let shares: HashMap<String, f64> = self.tuning.op_shares().into_iter().collect();
                for (title, names, graphic) in GROUPS {
                    egui::CollapsingHeader::new(*title).default_open(true).show(ui, |ui| {
                        draw_group_graphic(ui, *graphic, &self.tuning);
                        for p in self.tuning.params.iter_mut().filter(|p| names.contains(&p.name.as_str())) {
                            let g = ghost.as_ref().and_then(|v| v.get(&p.name)).and_then(|v| v.as_f64());
                            let imp = importances.get(&p.name).map(|v| v / max_imp);
                            edited |= param_row(ui, p, g, imp, None);
                        }
                    });
                }
                // Anything a future tuning adds that no group lists still gets its row.
                for p in self.tuning.params.iter_mut().filter(|p| !p.name.starts_with(nnfractals::tuning::OP_PREFIX) && !GROUPS.iter().any(|(_, n, _)| n.contains(&p.name.as_str()))) {
                    let g = ghost.as_ref().and_then(|v| v.get(&p.name)).and_then(|v| v.as_f64());
                    let imp = importances.get(&p.name).map(|v| v / max_imp);
                    edited |= param_row(ui, p, g, imp, None);
                }
                egui::CollapsingHeader::new("Operation weights — chance each op is picked when a mutation swaps in or grows an op")
                    .default_open(true).show(ui, |ui| {
                    ui.label(egui::RichText::new("Each operation is one building block of a formula. Hover an operation's name for what it does; its weight is how often it is picked (the % is its share of the total).").small().weak());
                    share_bar(ui, &shares_in_order(&self.tuning));
                    for p in self.tuning.params.iter_mut().filter(|p| p.name.starts_with(nnfractals::tuning::OP_PREFIX)) {
                        let g = ghost.as_ref().and_then(|v| v.get(&p.name)).and_then(|v| v.as_f64());
                        let imp = importances.get(&p.name).map(|v| v / max_imp);
                        let share = shares.get(&p.name[nnfractals::tuning::OP_PREFIX.len()..]).copied();
                        edited |= param_row(ui, p, g, imp, share);
                    }
                });
                let _ = edited;
                ui.separator();

                // ── run settings ──
                egui::CollapsingHeader::new("Run settings").default_open(true).show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Fitness metric:");
                        ui.add(egui::TextEdit::singleline(&mut prefs.fitness_metric).desired_width(360.0));
                        if ui.button("campaign recipe").on_hover_text("The recipe the previous campaigns used (H1 + continuity, front-cover and grey bands).").clicked() {
                            match default_campaign_recipe(root) { Some(r) => prefs.fitness_metric = r, None => self.msg = "taste_analysis/combo_specs.json not readable".into() }
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Pool dir:"); ui.add(egui::TextEdit::singleline(&mut prefs.pool_dir).desired_width(190.0));
                        ui.label("Collapse:");
                        egui::ComboBox::from_id_salt("tun_collapse").selected_text(prefs.collapse.clone()).show_ui(ui, |ui| {
                            for m in nnfractals::quat_collapse::OFFERED { ui.selectable_value(&mut prefs.collapse, m.name().to_string(), m.name()); }
                        });
                        ui.checkbox(&mut prefs.gpu, "GPU");
                    });
                    ui.horizontal(|ui| {
                        ui.label("Population:"); ui.add(egui::DragValue::new(&mut prefs.population).range(4..=2000));
                        ui.label("Generations:"); ui.add(egui::DragValue::new(&mut prefs.generations).range(0..=5000)).on_hover_text("0 = until the time limit or the stagnation stop.");
                        ui.label("Eval phases:"); ui.add(egui::DragValue::new(&mut prefs.eval_phases).range(1..=16));
                        ui.label("Save top:"); ui.add(egui::DragValue::new(&mut prefs.save_top).range(0..=500)).on_hover_text("Genomes kept at the end of a run. Campaigns score the best evolved one of these, so they use at least 5.");
                    });
                    ui.horizontal(|ui| {
                        let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
                        ui.label("Workers:");
                        let r = ui.add(egui::Slider::new(&mut prefs.workers, 1..=cores)).on_hover_text("Threads scoring genomes at once. 1 = the old one-at-a-time loop (uses ~1 core and leaves the GPU idle). Changing it takes effect within a second on a running campaign run, at the next genome. The GPU renders take turns, so more workers help until it is the bottleneck.");
                        let a = ui.checkbox(&mut prefs.auto_workers, "auto").on_hover_text("Automatic throttle: the run tries one or two threads more and fewer while it works, keeps what makes genomes score fastest, and prefers fewer threads when more no longer help (the GPU is then the bottleneck). The slider is its starting point; moving the slider restarts the search from there.");
                        if (r.changed() || a.changed()) && !self.camp_buf.trim().is_empty() {
                            let dir = Campaign::dir(root, self.camp_buf.trim());
                            let _ = std::fs::create_dir_all(&dir);
                            let _ = std::fs::write(dir.join("control.json"), serde_json::json!({ "workers": prefs.workers, "auto": prefs.auto_workers }).to_string());
                        }
                        // What the running explorer reports (written by it once a second).
                        if !self.camp_buf.trim().is_empty() {
                            let f = Campaign::dir(root, self.camp_buf.trim()).join("control.json.now");
                            if self.workers_status.0.elapsed() > Duration::from_secs(1) {
                                let fresh = std::fs::metadata(&f).and_then(|m| m.modified()).ok().and_then(|t| t.elapsed().ok()).is_some_and(|e| e < Duration::from_secs(5));
                                self.workers_status = (Instant::now(), std::fs::read_to_string(&f).ok().filter(|_| fresh).and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok()).map(|v| {
                                    let now = v["now"].as_u64().unwrap_or(0);
                                    match v["settled"].as_u64() { Some(s) => format!("auto: running {now}, settled at {s}"), None => format!("running {now}") }
                                }));
                            }
                            if let Some(t) = &self.workers_status.1 { ui.label(egui::RichText::new(t).small().weak()); }
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Time limit (min):"); ui.add(egui::DragValue::new(&mut prefs.max_minutes).range(0.0..=1440.0).speed(0.5)).on_hover_text("Each run ends after this long (saved normally). 0 = none for a single run; campaigns use 30.");
                        ui.label("Stop when flat (gens):"); ui.add(egui::DragValue::new(&mut prefs.stop_flat).range(0..=500)).on_hover_text("Stagnation detector: end the run when the elite mean has not risen for this many generations. 0 = off.");
                        ui.label("Mode:");
                        egui::ComboBox::from_id_salt("tun_mode").selected_text(match prefs.mode { QuatEvoMode::PredatorPrey => "predator-prey", QuatEvoMode::MapElites => "MAP-Elites", QuatEvoMode::Blend => "blend" }).show_ui(ui, |ui| {
                            ui.selectable_value(&mut prefs.mode, QuatEvoMode::MapElites, "MAP-Elites (not for tuned runs)");
                            ui.selectable_value(&mut prefs.mode, QuatEvoMode::Blend, "blend");
                            ui.selectable_value(&mut prefs.mode, QuatEvoMode::PredatorPrey, "predator-prey");
                        });
                    });
                    if prefs.mode == QuatEvoMode::MapElites {
                        ui.colored_label(Color32::YELLOW, "MAP-Elites has no time limit / stagnation stop; tuned runs use blend mode.");
                    }
                });
                ui.separator();

                // ── single run ──
                ui.horizontal(|ui| {
                    if ui.add_enabled(!camp_active, egui::Button::new("▶ Run once with this tuning")).on_hover_text("One GA run with the values above (no Optuna).").clicked() {
                        let seed: u64 = rand::random();
                        action = Some(Action::RunOnce(run_args(prefs, &self.tuning, &prefs.out_dir, seed, prefs.generations)));
                    }
                    if ga_running && !camp_active { ui.weak("another GA is running (they share the GPU)"); }
                });
                ui.separator();

                // ── campaigns ──
                ui.heading("Campaign");
                ui.horizontal_wrapped(|ui| {
                    ui.label("Name:");
                    ui.add_enabled(!camp_active, egui::TextEdit::singleline(&mut self.camp_buf).desired_width(150.0));
                    let mut pick = None;
                    ui.add_enabled_ui(!camp_active, |ui| {
                        egui::ComboBox::from_id_salt("camp_pick").selected_text("open…").show_ui(ui, |ui| {
                            for n in &self.camp_names { if ui.selectable_label(false, n).clicked() { pick = Some(n.clone()); } }
                        });
                    });
                    if let Some(n) = pick {
                        self.camp = Campaign::load(root, &n);
                        if let Some(c) = &self.camp { self.tuning = c.tuning.clone(); self.name_buf = c.tuning.name.clone(); }
                        self.camp_buf = n;
                    }
                    if !camp_active {
                        if ui.add(egui::Button::new("▶ Start / resume campaign")).on_hover_text("Endless: each run asks Optuna for a tuning, lets you edit it, runs one finite GA, scores it and tells Optuna.").clicked() {
                            self.start_campaign(root, prefs);
                        }
                        if ga_running { ui.weak("another GA is running (they share the GPU)"); }
                    } else if let Some(sh) = &self.shared {
                        let mut auto = sh.auto.load(Ordering::Relaxed);
                        if ui.checkbox(&mut auto, "auto-continue").on_hover_text("Run every proposal as it comes (overnight). Off: pause on the editable card after each run.").changed() { sh.auto.store(auto, Ordering::Relaxed); }
                        if ui.button("■ Stop after this run").clicked() { sh.stop.store(true, Ordering::Relaxed); }
                        if ui.button("✖ Kill now").clicked() { sh.kill.store(true, Ordering::Relaxed); }
                    }
                });
                if !self.camp_buf.trim().is_empty() {
                    let runs = Campaign::dir(root, self.camp_buf.trim()).join("runs");
                    if ui.button("🖼 Open winners in browser").on_hover_text("Open the gallery browser on this campaign's runs/ folder (the best genome of every scored run).").clicked() {
                        let _ = std::fs::create_dir_all(&runs);
                        let _ = Command::new(sibling("nnfractals-browser")).arg("--dir").arg(&runs).current_dir(root).spawn();
                    }
                }
                if camp_active || !status.is_empty() {
                    let color = if camp_active { Color32::LIGHT_GREEN } else { Color32::GRAY };
                    ui.label(egui::RichText::new(&status).color(color));
                }
                if ready {
                    egui::Frame::group(ui.style()).show(ui, |ui| {
                        ui.label("The bars above hold the proposed tuning. Edit values, ranges or locks, then:");
                        if ui.button("▶ Run this").clicked() {
                            if let (Some(tx), Some(t)) = (&self.tx, self.applied_trial) {
                                let _ = tx.send(Cmd { tuning: self.tuning.clone(), prefs: prefs.clone(), go_trial: Some(t) });
                            }
                        }
                    });
                }
                if !self.msg.is_empty() { ui.label(egui::RichText::new(&self.msg).small()); }
                if let Some(c) = &self.camp {
                    ui.label(format!("{} runs · best {}", c.runs.len(), c.best().and_then(|r| r.value).map(|v| format!("{v:.3}")).unwrap_or_else(|| "—".into())));
                    if let Some(i) = run_chart(ui, &c.runs, &live) {
                        if let Some(v) = c.runs[i].params.as_object() {
                            self.tuning.apply_values(v);
                            self.msg = format!("loaded the tuning of run {} into the bars", c.runs[i].trial);
                        }
                    }
                    ui.label(egui::RichText::new("click a dot to load that run's values · blue = run value, orange = best so far, green = current run").small().weak());
                    egui::CollapsingHeader::new("History").show(ui, |ui| {
                        for r in c.runs.iter().rev().take(40) {
                            ui.label(egui::RichText::new(format!("#{:<3} {:>8}  {:>6.0}s  {}", r.trial, r.value.map(|v| format!("{v:.3}")).unwrap_or_else(|| "failed".into()), r.secs, r.stop_reason)).monospace().small());
                        }
                    });
                } else {
                    run_chart(ui, &[], &live);
                }
            });
            });
        });

        // Push the latest editor state to the campaign thread (go is only sent by the button).
        if let Some(tx) = &self.tx {
            let snap = serde_json::to_string(&(&self.tuning, &*prefs)).unwrap_or_default();
            if snap != self.last_sent {
                self.last_sent = snap;
                let _ = tx.send(Cmd { tuning: self.tuning.clone(), prefs: prefs.clone(), go_trial: None });
            }
        }
        self.open = open;
        action
    }

    /// Repaint cadence hint: true while a campaign is live.
    pub fn busy(&self) -> bool { self.running() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tuning_overrides_the_plain_run_flags() {
        let mut prefs = QuatEvoPrefs::default();
        prefs.crossover_subtree = false;
        prefs.survivors = 99;
        prefs.max_minutes = 2.0;
        let mut t = Tuning::defaults("x");
        t.get_mut("survivors").unwrap().value = 7.0;
        let a = run_args(&prefs, &t, "/tmp/o", 5, 10);
        let count = |f: &str| a.iter().filter(|x| *x == f).count();
        assert_eq!(count("--survivors"), 1);
        assert_eq!(a[a.iter().position(|x| x == "--survivors").unwrap() + 1], "7");
        assert_eq!(a[a.iter().position(|x| x == "--max-seconds").unwrap() + 1], "120");
        assert_eq!(count("--crossover-mode"), 1);
        assert!(!a.contains(&"--map-elites".to_string()), "tuned runs never use MAP-Elites");
    }

    #[test]
    fn edited_values_are_detected() {
        let t = Tuning::defaults("x");
        let same = t.values_json().as_object().cloned().unwrap();
        assert!(same_values(&t, &same));
        let mut other = same.clone();
        other.insert("survivors".into(), serde_json::json!(12));
        assert!(!same_values(&t, &other));
    }
}
