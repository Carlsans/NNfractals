//! The launcher's "Idle manager" window: what `nnfractals-idle` is doing now, its service,
//! its settings, and what the computer did while nobody used it (the last 24 h by default).

use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

use eframe::egui::{self, Color32};
use nnfractals::idle::{self, Config, Minute, Stats, Status};

pub struct IdlePanel {
    pub open: bool,
    status: Status,
    status_at: Option<Instant>,
    /// systemctl --user is-active / is-enabled.
    service: (String, String),
    service_at: Option<Instant>,
    stats: Option<Stats>,
    stats_at: Option<Instant>,
    hours: f64,
    log_tail: Vec<String>,
    cfg: Config,
    cfg_saved: Config,
    cfg_err: Option<String>,
    cfg_loaded: bool,
    new_env: (String, String),
    message: Option<Result<String, String>>,
}

impl Default for IdlePanel {
    fn default() -> Self {
        IdlePanel {
            open: false, status: Status::default(), status_at: None, service: (String::new(), String::new()), service_at: None,
            stats: None, stats_at: None, hours: 24.0, log_tail: Vec::new(), cfg: Config::default(), cfg_saved: Config::default(),
            cfg_err: None, cfg_loaded: false, new_env: (String::new(), String::new()), message: None,
        }
    }
}

fn systemctl(args: &[&str]) -> Result<String, String> {
    let out = Command::new("systemctl").arg("--user").args(args).output().map_err(|e| e.to_string())?;
    let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if out.status.success() { Ok(text) } else {
        let err = String::from_utf8_lossy(&out.stderr).trim().to_string();
        Err(if err.is_empty() { text } else { err })
    }
}

fn now_ms() -> u64 { idle::now_s() * 1000 }

fn ago(ms: u64) -> String {
    if ms == 0 { return "never".into(); }
    let s = now_ms().saturating_sub(ms) / 1000;
    if s < 120 { format!("{s} s ago") } else if s < 7200 { format!("{} min ago", s / 60) } else { format!("{:.1} h ago", s as f64 / 3600.0) }
}

/// Local "HH:MM" of a unix time.
fn local_hm(t: u64) -> String {
    // SAFETY: localtime_r on stack values.
    unsafe {
        let tt = t as libc::time_t;
        let mut tm: libc::tm = std::mem::zeroed();
        libc::localtime_r(&tt, &mut tm);
        format!("{:02}:{:02}", tm.tm_hour, tm.tm_min)
    }
}

fn minute_color(m: Minute) -> Color32 {
    match m {
        Minute::Off => Color32::from_gray(35),
        Minute::InUse => Color32::from_gray(85),
        Minute::InUsePaused => Color32::from_rgb(200, 150, 60),
        Minute::Idle => Color32::from_gray(170),
        Minute::Render => Color32::from_rgb(80, 140, 235),
        Minute::Campaign => Color32::from_rgb(90, 190, 110),
    }
}

impl IdlePanel {
    fn refresh(&mut self, root: &Path) {
        if self.status_at.is_none_or(|t| t.elapsed() > Duration::from_secs(2)) {
            self.status = Status::read(root);
            self.status_at = Some(Instant::now());
            self.log_tail = std::fs::read_to_string(root.join(idle::LOG)).unwrap_or_default()
                .lines().rev().take(80).map(str::to_string).collect::<Vec<_>>().into_iter().rev().collect();
        }
        if self.service_at.is_none_or(|t| t.elapsed() > Duration::from_secs(5)) {
            let active = systemctl(&["is-active", idle::SERVICE]).unwrap_or_else(|e| e);
            let enabled = systemctl(&["is-enabled", idle::SERVICE]).unwrap_or_else(|e| e);
            self.service = (active, enabled);
            self.service_at = Some(Instant::now());
        }
        if self.open && self.stats_at.is_none_or(|t| t.elapsed() > Duration::from_secs(60)) {
            self.stats = Some(idle::stats(root, self.hours));
            self.stats_at = Some(Instant::now());
        }
        if !self.cfg_loaded {
            let (c, e) = Config::load(root);
            self.cfg = c.clone();
            self.cfg_saved = c;
            self.cfg_err = e;
            self.cfg_loaded = true;
        }
    }

    /// One line for the launcher's main panel.
    pub fn summary(&mut self, root: &Path) -> (String, Color32) {
        self.refresh(root);
        let s = &self.status;
        let live = now_ms().saturating_sub(s.updated_ms) < 30_000;
        if !live || self.service.0 != "active" {
            return (format!("idle manager not running ({})", self.service.0), Color32::from_rgb(220, 120, 100));
        }
        let what = |j: &Option<idle::Job>, name: &str| j.as_ref().map(|j| format!("{name} {}", if j.paused { "paused" } else { "running" }));
        let jobs: Vec<String> = [what(&s.render, "render"), what(&s.campaign, "campaign")].into_iter().flatten().collect();
        let jobs = if jobs.is_empty() { "nothing running".to_string() } else { jobs.join(", ") };
        if s.in_use {
            (format!("computer in use — {jobs}"), Color32::from_gray(170))
        } else {
            (format!("computer not in use since {} — {jobs}", ago(s.since_ms)), Color32::from_rgb(110, 200, 120))
        }
    }

    pub fn busy(&self) -> bool { self.open }

    pub fn show(&mut self, ctx: &egui::Context, root: &Path) {
        if !self.open { return; }
        self.refresh(root);
        let mut open = self.open;
        let builder = egui::ViewportBuilder::default().with_title("Idle manager").with_inner_size([980.0, 900.0]);
        ctx.show_viewport_immediate(egui::ViewportId::from_hash_of("idle_manager_window"), builder, |ui, _class| {
            if ui.input(|i| i.viewport().close_requested()) { open = false; }
            egui::CentralPanel::default().show(ui, |ui| {
                egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                    self.ui_now(ui, root);
                    ui.separator();
                    self.ui_stats(ui, root);
                    ui.separator();
                    self.ui_settings(ui, root);
                    ui.separator();
                    egui::CollapsingHeader::new("Log").id_salt("idle_log").show(ui, |ui| {
                        egui::ScrollArea::vertical().id_salt("idle_log_scroll").max_height(260.0).stick_to_bottom(true).show(ui, |ui| {
                            for l in &self.log_tail { ui.label(egui::RichText::new(l).monospace().size(11.0)); }
                        });
                    });
                });
            });
            ui.ctx().request_repaint_after(Duration::from_secs(2));
        });
        self.open = open;
    }

    fn ui_now(&mut self, ui: &mut egui::Ui, root: &Path) {
        ui.heading("Now");
        let s = self.status.clone();
        let live = now_ms().saturating_sub(s.updated_ms) < 30_000;
        ui.horizontal(|ui| {
            let (active, enabled) = self.service.clone();
            let c = if active == "active" { Color32::from_rgb(110, 200, 120) } else { Color32::from_rgb(220, 120, 100) };
            ui.colored_label(c, format!("service {active}"));
            ui.label(format!("({}at login)", if enabled == "enabled" { "" } else { "not started " }));
            if active == "active" {
                if ui.button("■ Stop").on_hover_text("Stops the manager; what it started goes on running (unpaused). Start it again to hand the work back.").clicked() {
                    self.message = Some(systemctl(&["stop", idle::SERVICE]).map(|_| "stopped".into()));
                    self.service_at = None;
                }
                if ui.button("⟳ Restart").on_hover_text("Restarts the manager (it takes back what it had started)").clicked() {
                    self.message = Some(systemctl(&["restart", idle::SERVICE]).map(|_| "restarted".into()));
                    self.service_at = None;
                }
            } else if ui.button("▶ Start").clicked() {
                self.message = Some(systemctl(&["start", idle::SERVICE]).map(|_| "started".into()));
                self.service_at = None;
            }
            let mut at_login = enabled == "enabled";
            if ui.checkbox(&mut at_login, "start at login").changed() {
                self.message = Some(systemctl(&[if at_login { "enable" } else { "disable" }, idle::SERVICE]).map(|_| if at_login { "starts at login".into() } else { "no longer starts at login".into() }));
                self.service_at = None;
            }
        });
        if !live {
            ui.colored_label(Color32::from_rgb(220, 120, 100), "No fresh status from the manager (it is not running, or has not written for 30 s).");
            return;
        }
        let state = if s.in_use { format!("Computer in use since {}", ago(s.since_ms)) } else { format!("Computer NOT in use since {}", ago(s.since_ms)) };
        ui.label(egui::RichText::new(state).strong());
        let input = if s.last_device.is_empty() { format!("no input since the manager started ({})", ago(s.last_input_ms)) } else { format!("last input {} ({})", ago(s.last_input_ms), s.last_device) };
        ui.label(format!("{input}{}", if s.sound_playing { "; sound playing" } else { "" }));
        ui.label(&s.note);
        for (j, what) in [(&s.render, "render queue"), (&s.campaign, "campaign")] {
            if let Some(j) = j {
                ui.label(format!("{what}: started {}, {}", ago(j.started_ms), if j.paused { "PAUSED" } else { "running" }));
            }
        }
        ui.label(format!("{:.1} GB free{}", s.mem_free_gb, s.workers.map(|w| format!(", {w} GA workers")).unwrap_or_default()));
        let stop = root.join(idle::STOP_FILE);
        if stop.exists() {
            ui.horizontal(|ui| {
                ui.colored_label(Color32::from_rgb(230, 180, 80), format!("{} exists: no campaign starts.", idle::STOP_FILE));
                if ui.button("remove it").clicked() { let _ = std::fs::remove_file(&stop); }
            });
        }
        egui::CollapsingHeader::new(format!("input devices watched ({})", s.devices.len())).id_salt("idle_devices").show(ui, |ui| {
            egui::Grid::new("idle_devices_grid").striped(true).show(ui, |ui| {
                for (name, t) in &s.devices { ui.label(name); ui.label(format!("last event {}", ago(*t))); ui.end_row(); }
            });
        });
        match &self.message {
            Some(Ok(m)) => { ui.colored_label(Color32::LIGHT_GREEN, m); }
            Some(Err(e)) => { ui.colored_label(Color32::LIGHT_RED, e); }
            None => {}
        }
    }

    fn ui_stats(&mut self, ui: &mut egui::Ui, root: &Path) {
        ui.horizontal(|ui| {
            ui.heading("While nobody used the computer");
            let mut changed = false;
            for h in [6.0, 24.0, 72.0, 168.0] {
                let label = if h < 48.0 { format!("{h:.0} h") } else { format!("{:.0} days", h / 24.0) };
                changed |= ui.selectable_value(&mut self.hours, h, label).changed();
            }
            if ui.button("⟳").on_hover_text("Recompute now (it updates every minute)").clicked() || changed {
                self.stats = Some(idle::stats(root, self.hours));
                self.stats_at = Some(Instant::now());
            }
        });
        let Some(st) = self.stats.clone() else { return };
        // The timeline: one stripe per minute, oldest on the left.
        let h = 34.0;
        let (rect, resp) = ui.allocate_exact_size(egui::vec2(ui.available_width(), h + 16.0), egui::Sense::hover());
        let bar = egui::Rect::from_min_size(rect.min, egui::vec2(rect.width(), h));
        let p = ui.painter_at(rect);
        p.rect_filled(bar, 2.0, Color32::from_gray(25));
        let n = st.minutes.len().max(1);
        let px = bar.width() / n as f32;
        let mut i = 0;
        while i < st.minutes.len() {
            let m = st.minutes[i];
            let mut j = i;
            while j < st.minutes.len() && st.minutes[j] == m { j += 1; }
            let r = egui::Rect::from_min_max(egui::pos2(bar.left() + i as f32 * px, bar.top()), egui::pos2(bar.left() + j as f32 * px, bar.bottom()));
            p.rect_filled(r, 0.0, minute_color(m));
            i = j;
        }
        // Hour ticks.
        let step_h = if self.hours <= 6.0 { 1 } else if self.hours <= 24.0 { 3 } else if self.hours <= 72.0 { 12 } else { 24 };
        let first = st.start.div_ceil(3600) * 3600;
        let mut t = first;
        while t < st.end {
            let hour = ((t % 86_400) / 3600) as i64;
            // Local hour boundaries (the tick label is local time).
            let lbl = local_hm(t);
            if lbl.ends_with(":00") && (lbl[..2].parse::<i64>().unwrap_or(hour) % step_h == 0) {
                let x = bar.left() + ((t - st.start) as f32 / 60.0) * px;
                p.line_segment([egui::pos2(x, bar.bottom()), egui::pos2(x, bar.bottom() + 4.0)], egui::Stroke::new(1.0, Color32::from_gray(140)));
                p.text(egui::pos2(x, bar.bottom() + 4.0), egui::Align2::CENTER_TOP, lbl, egui::FontId::proportional(10.0), Color32::from_gray(160));
            }
            t += 3600;
        }
        if let Some(pos) = resp.hover_pos() {
            if bar.contains(pos) {
                let k = (((pos.x - bar.left()) / px) as usize).min(n - 1);
                let when = st.start + k as u64 * 60;
                resp.on_hover_text(format!("{} — {}", local_hm(when), st.minutes[k].label()));
            }
        }
        // Legend.
        ui.horizontal_wrapped(|ui| {
            for m in [Minute::Render, Minute::Campaign, Minute::Idle, Minute::InUsePaused, Minute::InUse, Minute::Off] {
                let (r, _) = ui.allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
                ui.painter().rect_filled(r, 2.0, minute_color(m));
                ui.label(egui::RichText::new(m.label()).small());
                ui.add_space(6.0);
            }
        });
        ui.add_space(4.0);
        egui::Grid::new("idle_totals").num_columns(2).spacing([16.0, 4.0]).show(ui, |ui| {
            let hrs = |x: f64| if x < 1.0 { format!("{:.0} min", x * 60.0) } else { format!("{x:.1} h") };
            ui.label("not in use"); ui.label(format!("{} ({} idle periods)", hrs(st.not_in_use_h), st.times_idle)); ui.end_row();
            ui.label("   rendering"); ui.label(format!("{} ({} queue runs)", hrs(st.render_h), st.render_runs)); ui.end_row();
            ui.label("   campaign"); ui.label(format!("{}{}", hrs(st.campaign_h), st.workers_mean.map(|w| format!(", {w:.1} GA workers on average")).unwrap_or_default())); ui.end_row();
            ui.label("   nothing to run"); ui.label(hrs(st.idle_h)); ui.end_row();
            ui.label("in use, work paused"); ui.label(format!("{} ({} pauses, {} campaign stops to free memory)", hrs(st.paused_h), st.pauses, st.clean_stops)); ui.end_row();
            ui.label("manager off"); ui.label(hrs(st.off_h)); ui.end_row();
        });
        ui.add_space(6.0);
        let ours = st.trials_of(true);
        ui.label(egui::RichText::new(format!("Campaign: {} trials finished while the computer was not in use, {} winners kept", ours.len(), st.winners.iter().filter(|w| w.2).count())).strong());
        if let Some(b) = st.best(true) {
            ui.horizontal(|ui| {
                ui.label(format!("best {:.3} — {} trial {} at {}", b.value, b.campaign, b.n, local_hm(b.t)));
                if let Some(f) = &b.file {
                    if ui.small_button("open").on_hover_text(f.display().to_string()).clicked() {
                        let _ = Command::new(super::sibling("nnfractals-anim-viewer")).arg(f).spawn();
                    }
                }
            });
        }
        if !ours.is_empty() {
            egui::CollapsingHeader::new("trials").id_salt("idle_trials").show(ui, |ui| {
                egui::Grid::new("idle_trials_grid").striped(true).show(ui, |ui| {
                    for t in ours.iter().rev() {
                        ui.label(local_hm(t.t));
                        ui.label(format!("{} #{}", t.campaign, t.n));
                        ui.label(format!("{:.3}", t.value));
                        ui.end_row();
                    }
                });
            });
        }
        let others = st.trials.len() - ours.len();
        if others > 0 {
            ui.label(egui::RichText::new(format!("({others} more trials in the window came from campaigns started by hand; best {:.3})", st.best(false).map(|b| b.value).unwrap_or(f64::NAN))).small().weak());
        }
        ui.add_space(6.0);
        ui.label(egui::RichText::new(format!("Renders finished: {}", st.renders_done.len())).strong());
        for (l, t) in st.renders_done.iter().rev().take(20) { ui.label(format!("{}  {l}", local_hm(*t))); }
    }

    fn ui_settings(&mut self, ui: &mut egui::Ui, root: &Path) {
        egui::CollapsingHeader::new(egui::RichText::new("Settings").heading()).id_salt("idle_settings").show(ui, |ui| {
            if let Some(e) = &self.cfg_err { ui.colored_label(Color32::LIGHT_RED, format!("{e} (showing the defaults)")); }
            let c = &mut self.cfg;
            egui::Grid::new("idle_settings_grid").num_columns(2).spacing([16.0, 6.0]).show(ui, |ui| {
                ui.label("not in use after");
                ui.add(egui::DragValue::new(&mut c.idle_after_min).range(1.0..=240.0).speed(0.5).suffix(" min without input")); ui.end_row();
                ui.label("sound");
                ui.checkbox(&mut c.sound_counts_as_use, "sound coming out counts as use (a film, a call)"); ui.end_row();
                ui.label("what may run");
                ui.horizontal(|ui| { ui.checkbox(&mut c.renders, "video render queue (first)"); ui.checkbox(&mut c.campaign, "then the campaign"); }); ui.end_row();
                ui.label("free the campaign's memory");
                ui.add(egui::DragValue::new(&mut c.free_memory_after_min).range(1.0..=600.0).speed(0.5).suffix(" min after you are back")); ui.end_row();
                ui.label("GA workers");
                ui.add(egui::DragValue::new(&mut c.max_workers).range(1..=16).suffix(" at most")).on_hover_text("Fewer when memory is short: 14 GB free up to 10, 9: 6, 6: 4, 4: 2, else 1"); ui.end_row();
                ui.label("pause the campaign below");
                ui.add(egui::DragValue::new(&mut c.min_free_gb).range(0.5..=16.0).speed(0.1).suffix(" GB free")); ui.end_row();
                ui.label("campaign log");
                ui.text_edit_singleline(&mut c.campaign_log); ui.end_row();
            });
            ui.label(egui::RichText::new("Campaign settings (scripts/optuna_campaigns.py)").strong());
            let mut remove: Option<String> = None;
            egui::Grid::new("idle_env_grid").num_columns(3).striped(true).show(ui, |ui| {
                for (k, v) in c.campaign_env.iter_mut() {
                    ui.monospace(k);
                    ui.add_sized([620.0, 20.0], egui::TextEdit::singleline(v));
                    if ui.small_button("✕").on_hover_text("remove").clicked() { remove = Some(k.clone()); }
                    ui.end_row();
                }
            });
            if let Some(k) = remove { c.campaign_env.remove(&k); }
            ui.horizontal(|ui| {
                ui.add(egui::TextEdit::singleline(&mut self.new_env.0).hint_text("NAME").desired_width(120.0));
                ui.add(egui::TextEdit::singleline(&mut self.new_env.1).hint_text("value").desired_width(300.0));
                if ui.add_enabled(!self.new_env.0.trim().is_empty(), egui::Button::new("add")).clicked() {
                    c.campaign_env.insert(self.new_env.0.trim().to_uppercase(), self.new_env.1.trim().to_string());
                    self.new_env = (String::new(), String::new());
                }
            });
            ui.horizontal(|ui| {
                let dirty = self.cfg != self.cfg_saved;
                if ui.add_enabled(dirty, egui::Button::new("💾 Save and restart the manager")).clicked() {
                    match self.cfg.save(root) {
                        Ok(()) => {
                            self.cfg_saved = self.cfg.clone();
                            self.cfg_err = None;
                            self.message = Some(systemctl(&["restart", idle::SERVICE]).map(|_| "settings saved; manager restarted".into()));
                            self.service_at = None;
                        }
                        Err(e) => self.message = Some(Err(format!("{}: {e}", idle::CONFIG))),
                    }
                }
                if ui.add_enabled(dirty, egui::Button::new("undo")).clicked() { self.cfg = self.cfg_saved.clone(); }
                if ui.button("defaults").on_hover_text("Fill in the defaults (save to keep them)").clicked() { self.cfg = Config::default(); }
            });
        });
    }
}
