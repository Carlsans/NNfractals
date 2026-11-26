//! What runs while nobody uses the computer (`nnfractals-idle`): its settings, the events it
//! records and the statistics of the last hours, shared by the manager, its `stats`
//! command and the launcher's Idle manager window.
//!
//! Events (`explorer_out/idle_manager_events.jsonl`, one JSON object per line) are state
//! changes: the manager starting or stopping, the computer in use or not, a render or the
//! campaign started, paused, resumed, ended, the GA worker count. The statistics replay
//! them minute by minute; results come from the video queue (renders finished) and the
//! campaign's log and files (trials, scores, winners kept).

use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

pub const CONFIG: &str = "idle_manager.toml";
pub const STATUS: &str = "explorer_out/idle_manager_status.json";
pub const LOG: &str = "explorer_out/idle_manager.log";
pub const EVENTS: &str = "explorer_out/idle_manager_events.jsonl";
pub const STOP_FILE: &str = "STOP_campaigns";
pub const SERVICE: &str = "nnfractals-idle.service";

pub fn now_s() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

// ---------------------------------------------------------------- settings

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub idle_after_min: f64,
    pub sound_counts_as_use: bool,
    pub renders: bool,
    pub campaign: bool,
    pub free_memory_after_min: f64,
    pub max_workers: usize,
    pub min_free_gb: f64,
    pub campaign_log: String,
    pub ignore_devices: Vec<String>,
    pub campaign_env: BTreeMap<String, String>,
}

impl Default for Config {
    fn default() -> Self {
        let env = [
            ("OUT", "fractals_dag_orgcx_loop"), ("COLLAPSE", "slice"), ("EXTRA", "orgcx_rel2d:4"), ("RUN_LIMIT", "3600"),
            ("WORKERS", "6"), ("CONTROL_FILE", "explorer_out/ga_control.json"), ("ORGCX_SIDECARS", "4"), ("ORGCX_THREADS", "4"),
            ("LABEL", "v10: optuna campaign (30 trials, reset), H1_cont + front_cover_band + grey_band, evolved-only trial score, slice T linear, run limit 3600 s, parallel scoring under a memory guard"),
        ];
        Config {
            idle_after_min: 10.0,
            sound_counts_as_use: true,
            renders: true,
            campaign: true,
            free_memory_after_min: 15.0,
            max_workers: 6,
            min_free_gb: 2.0,
            campaign_log: "optuna_campaign_orgcx.log".into(),
            ignore_devices: vec!["PC Speaker".into()],
            campaign_env: env.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
        }
    }
}

pub const CONFIG_HEADER: &str = "\
# nnfractals-idle: what runs while nobody uses the computer (see src/bin/idle_manager.rs).
# Read at start: `systemctl --user restart nnfractals-idle` after editing (the launcher's
# Idle manager window saves and restarts it for you).
#
# idle_after_min        minutes without keyboard/mouse/remote input before work starts
# sound_counts_as_use   sound coming out of the speakers (a film, a call) also means the computer is in use
# renders / campaign    what may run: the video render queue first, then the GA campaign
# free_memory_after_min back at the computer this long: the paused campaign ends its trial
#                       cleanly so its memory is freed (renders stay paused, keeping progress)
# max_workers           most GA workers; fewer when memory is short (14 GB free: up to 10, 9: 6, 6: 4, 4: 2, else 1)
# min_free_gb           the campaign pauses below this much free memory, even while idle
# campaign_log          where the campaign script writes (appended)
# ignore_devices        input devices whose events are not use (names as in /proc/bus/input/devices)
# [campaign_env]        the campaign's settings (scripts/optuna_campaigns.py reads them);
#                       these are the orgcx campaign's, as in explorer_out/resume_campaign.sh
";

impl Config {
    /// The settings file, or the defaults (and `None` for the error) when it is missing or bad.
    pub fn load(root: &Path) -> (Config, Option<String>) {
        match fs::read_to_string(root.join(CONFIG)) {
            Ok(t) => match toml::from_str(&t) {
                Ok(c) => (c, None),
                Err(e) => (Config::default(), Some(format!("{CONFIG}: {e}"))),
            },
            Err(_) => (Config::default(), None),
        }
    }

    pub fn save(&self, root: &Path) -> std::io::Result<()> {
        let body = toml::to_string(self).map_err(std::io::Error::other)?;
        fs::write(root.join(CONFIG), format!("{CONFIG_HEADER}\n{body}"))
    }
}

// ---------------------------------------------------------------- status (written by the manager)

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Job {
    /// Leader pid = process group id (each job is started in a group of its own).
    pub pgid: i32,
    pub paused: bool,
    pub started_ms: u64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Status {
    pub in_use: bool,
    pub since_ms: u64,
    pub last_input_ms: u64,
    pub last_device: String,
    pub sound_playing: bool,
    pub mem_free_gb: f64,
    pub workers: Option<usize>,
    pub render: Option<Job>,
    pub campaign: Option<Job>,
    pub note: String,
    pub devices: Vec<(String, u64)>,
    pub updated_ms: u64,
}

impl Status {
    pub fn read(root: &Path) -> Status {
        fs::read_to_string(root.join(STATUS)).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default()
    }

    pub fn write(&self, root: &Path) {
        let p = root.join(STATUS);
        let tmp = p.with_extension("json.tmp");
        if fs::write(&tmp, serde_json::to_string_pretty(self).unwrap_or_default()).is_ok() {
            let _ = fs::rename(tmp, p);
        }
    }
}

// ---------------------------------------------------------------- events

/// One state change. `ev`: start, stop, use, idle, render_start, render_pause,
/// render_resume, render_end, camp_start, camp_adopt, camp_pause, camp_resume,
/// camp_stop_clean, camp_end, render_adopt, workers.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Event {
    pub t: u64,
    pub ev: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub why: Option<String>,
}

pub fn record(root: &Path, ev: &str, n: Option<f64>, why: Option<&str>) {
    let e = Event { t: now_s(), ev: ev.to_string(), n, why: why.map(str::to_string) };
    if let (Ok(line), Ok(mut f)) = (serde_json::to_string(&e), OpenOptions::new().create(true).append(true).open(root.join(EVENTS))) {
        let _ = writeln!(f, "{line}");
    }
}

pub fn load_events(root: &Path) -> Vec<Event> {
    let mut v: Vec<Event> = fs::read_to_string(root.join(EVENTS)).unwrap_or_default().lines()
        .filter_map(|l| serde_json::from_str(l).ok()).collect();
    v.sort_by_key(|e| e.t);
    v
}

/// Keeps the last `days` days of events (the manager calls it once a day).
pub fn prune_events(root: &Path, days: u64) {
    let cut = now_s().saturating_sub(days * 86_400);
    let ev = load_events(root);
    if ev.first().is_some_and(|e| e.t < cut) {
        let kept: Vec<String> = ev.iter().filter(|e| e.t >= cut).filter_map(|e| serde_json::to_string(e).ok()).collect();
        let _ = fs::write(root.join(EVENTS), kept.join("\n") + "\n");
    }
}

// ---------------------------------------------------------------- statistics

/// What the computer was doing in one minute.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Minute {
    /// No record: the manager was not running.
    Off,
    /// In use, nothing of the manager's waiting.
    InUse,
    /// In use, a render or the campaign paused until the next idle time.
    InUsePaused,
    /// Not in use, nothing to run (or memory too short).
    Idle,
    /// Not in use, rendering the video queue.
    Render,
    /// Not in use, running the campaign.
    Campaign,
}

impl Minute {
    pub fn label(self) -> &'static str {
        match self {
            Minute::Off => "manager off",
            Minute::InUse => "in use",
            Minute::InUsePaused => "in use, work paused",
            Minute::Idle => "not in use, nothing to run",
            Minute::Render => "rendering",
            Minute::Campaign => "campaign",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Trial {
    pub campaign: String,
    pub n: usize,
    pub value: f64,
    /// When it finished (its winner's file).
    pub t: u64,
    /// The trial's winner (.nn), if still there.
    pub file: Option<PathBuf>,
    /// Finished while the manager's campaign ran (rather than one started by hand).
    pub by_manager: bool,
}

#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub start: u64,
    pub end: u64,
    /// One per minute from `start`.
    pub minutes: Vec<Minute>,
    pub not_in_use_h: f64,
    pub render_h: f64,
    pub campaign_h: f64,
    pub idle_h: f64,
    pub paused_h: f64,
    pub off_h: f64,
    pub times_idle: usize,
    pub pauses: usize,
    pub clean_stops: usize,
    pub campaigns_started: usize,
    pub render_runs: usize,
    /// Mean GA workers over the minutes the campaign ran.
    pub workers_mean: Option<f64>,
    /// (label, finished at) of the renders done in the window.
    pub renders_done: Vec<(String, u64)>,
    pub trials: Vec<Trial>,
    /// Winners the campaign kept (copied into its folder) in the window, and whether the
    /// manager's campaign kept them.
    pub winners: Vec<(String, f64, bool)>,
}

impl Stats {
    /// The trials the manager's campaign finished (`by_manager`) or all of them.
    pub fn trials_of(&self, by_manager: bool) -> Vec<&Trial> {
        self.trials.iter().filter(|t| !by_manager || t.by_manager).collect()
    }

    pub fn best(&self, by_manager: bool) -> Option<&Trial> {
        self.trials_of(by_manager).into_iter().max_by(|a, b| a.value.total_cmp(&b.value))
    }

    /// What the computer did at time `t` (a minute of the window).
    pub fn minute_at(&self, t: u64) -> Option<Minute> {
        if t < self.start { return None; }
        self.minutes.get(((t - self.start) / 60) as usize).copied()
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Job3 { None, Run, Paused }

/// Replays the events over the last `hours` (ending now) minute by minute, and gathers the
/// results: renders finished, the campaign's trials and kept winners.
pub fn stats(root: &Path, hours: f64) -> Stats {
    let end = now_s();
    let start = end.saturating_sub((hours * 3600.0) as u64);
    let n_min = ((end - start) / 60) as usize;
    let events = load_events(root);
    // Replay: state before `start`, then minute by minute.
    let (mut on, mut in_use, mut render, mut camp, mut workers) = (false, true, Job3::None, Job3::None, None::<f64>);
    let apply = |e: &Event, on: &mut bool, in_use: &mut bool, render: &mut Job3, camp: &mut Job3, workers: &mut Option<f64>| {
        match e.ev.as_str() {
            "start" => { *on = true; *in_use = true; *render = Job3::None; *camp = Job3::None; }
            "stop" => { *on = false; }
            "use" => *in_use = true,
            "idle" => *in_use = false,
            "render_start" | "render_resume" => *render = Job3::Run,
            "render_adopt" => *render = if e.n == Some(1.0) { Job3::Paused } else { Job3::Run },
            "render_pause" => *render = Job3::Paused,
            "render_end" => *render = Job3::None,
            "camp_start" | "camp_resume" => *camp = Job3::Run,
            "camp_adopt" => *camp = if e.n == Some(1.0) { Job3::Paused } else { Job3::Run },
            "camp_pause" => *camp = Job3::Paused,
            "camp_end" | "camp_stop_clean" => { *camp = Job3::None; *workers = None; }
            "workers" => *workers = e.n,
            _ => {}
        }
    };
    let mut i = 0;
    while i < events.len() && events[i].t < start {
        apply(&events[i], &mut on, &mut in_use, &mut render, &mut camp, &mut workers);
        i += 1;
    }
    let mut s = Stats { start, end, ..Default::default() };
    let (mut w_sum, mut w_n) = (0.0, 0usize);
    for m in 0..n_min {
        let t_end = start + (m as u64 + 1) * 60;
        while i < events.len() && events[i].t < t_end {
            let e = &events[i];
            match e.ev.as_str() {
                "idle" => s.times_idle += 1,
                "render_pause" | "camp_pause" => s.pauses += 1,
                "camp_stop_clean" => s.clean_stops += 1,
                "camp_start" => s.campaigns_started += 1,
                "render_start" => s.render_runs += 1,
                _ => {}
            }
            apply(e, &mut on, &mut in_use, &mut render, &mut camp, &mut workers);
            i += 1;
        }
        let state = if !on {
            Minute::Off
        } else if in_use {
            if render != Job3::None || camp != Job3::None { Minute::InUsePaused } else { Minute::InUse }
        } else if render == Job3::Run {
            Minute::Render
        } else if camp == Job3::Run {
            if let Some(w) = workers { w_sum += w; w_n += 1; }
            Minute::Campaign
        } else {
            Minute::Idle
        };
        s.minutes.push(state);
    }
    let h = |k: Minute| s.minutes.iter().filter(|&&m| m == k).count() as f64 / 60.0;
    s.render_h = h(Minute::Render);
    s.campaign_h = h(Minute::Campaign);
    s.idle_h = h(Minute::Idle);
    s.paused_h = h(Minute::InUsePaused);
    s.off_h = h(Minute::Off);
    s.not_in_use_h = s.render_h + s.campaign_h + s.idle_h;
    s.workers_mean = (w_n > 0).then(|| w_sum / w_n as f64);

    // Renders finished in the window (their output file's time).
    for it in crate::video_export::load_queue() {
        if it.status != crate::video_export::QueueStatus::Done { continue; }
        let Some(out) = &it.output_path else { continue };
        let p = if Path::new(out).is_absolute() { PathBuf::from(out) } else { root.join(out) };
        if let Some(t) = mtime(&p) {
            if t >= start { s.renders_done.push((it.genome_label.clone(), t)); }
        }
    }
    s.renders_done.sort_by_key(|r| r.1);

    // The campaign's trials and winners: its log gives the values, its files the times.
    let (cfg, _) = Config::load(root);
    let out = root.join(cfg.campaign_env.get("OUT").map(String::as_str).unwrap_or("fractals_dag_orgcx_loop"));
    let log = fs::read_to_string(root.join(&cfg.campaign_log)).unwrap_or_default();
    let mut camp_name = String::new();
    for line in log.lines() {
        if let Some(rest) = line.strip_prefix("CAMPAIGN ") {
            if rest.contains(" start") { camp_name = rest.split_whitespace().next().unwrap_or("").to_string(); }
        } else if let Some(rest) = line.strip_prefix("TRIAL ") {
            let mut it = rest.split_whitespace();
            let (Some(n), Some("value"), Some(v)) = (it.next().and_then(|x| x.parse::<usize>().ok()), it.next(), it.next().and_then(|x| x.parse::<f64>().ok())) else { continue };
            let dir = out.join("_campaigns").join(&camp_name);
            let file = fs::read_dir(&dir).ok().and_then(|d| d.flatten().map(|e| e.path())
                .find(|p| p.extension().is_some_and(|x| x == "nn") && p.file_name().is_some_and(|f| f.to_string_lossy().starts_with(&format!("trial{n:02}_")))));
            if let Some(t) = file.as_deref().and_then(mtime) {
                if t >= start { s.trials.push(Trial { campaign: camp_name.clone(), n, value: v, t, file, by_manager: false }); }
            }
        } else if let Some(rest) = line.strip_prefix("WINNER ") {
            // WINNER <campaign> trial <n> value <v> <path without .nn>
            let parts: Vec<&str> = rest.split_whitespace().collect();
            if parts.len() >= 6 {
                let p = root.join(format!("{}.nn", parts[5]));
                if let Some(t) = mtime(&p).filter(|&t| t >= start) {
                    let ours = matches!(s.minute_at(t), Some(Minute::Campaign | Minute::InUsePaused));
                    s.winners.push((parts[5].to_string(), parts[4].parse().unwrap_or(f64::NAN), ours));
                }
            }
        }
    }
    s.trials.sort_by_key(|t| t.t);
    // A trial is the manager's when its campaign was alive (running or paused) as it finished.
    let flags: Vec<bool> = s.trials.iter().map(|t| matches!(s.minute_at(t.t), Some(Minute::Campaign | Minute::InUsePaused))).collect();
    for (t, f) in s.trials.iter_mut().zip(flags) { t.by_manager = f; }
    s
}

fn mtime(p: &Path) -> Option<u64> {
    fs::metadata(p).ok()?.modified().ok()?.duration_since(UNIX_EPOCH).ok().map(|d| d.as_secs())
}

/// The statistics as text (the `nnfractals-idle stats` command): totals, a timeline (one
/// character per `step` minutes, newest on the right) and the results.
pub fn stats_text(s: &Stats) -> String {
    let hours = (s.end - s.start) as f64 / 3600.0;
    let mut t = format!("last {hours:.0} h: not in use {:.1} h (rendering {:.1} h, campaign {:.1} h, nothing to run {:.1} h); in use with work paused {:.1} h; manager off {:.1} h\n",
                        s.not_in_use_h, s.render_h, s.campaign_h, s.idle_h, s.paused_h, s.off_h);
    t += &format!("  {} idle periods, {} pauses, {} clean campaign stops to free memory, {} campaign start(s), {} render run(s){}\n",
                  s.times_idle, s.pauses, s.clean_stops, s.campaigns_started, s.render_runs,
                  s.workers_mean.map(|w| format!(", {w:.1} GA workers on average")).unwrap_or_default());
    let step = 10;
    let line: String = s.minutes.chunks(step).map(|c| {
        let count = |k: Minute| c.iter().filter(|&&m| m == k).count();
        let top = [Minute::Render, Minute::Campaign, Minute::Idle, Minute::InUsePaused, Minute::InUse, Minute::Off]
            .into_iter().max_by_key(|&k| count(k)).unwrap_or(Minute::Off);
        match top { Minute::Render => 'R', Minute::Campaign => 'C', Minute::Idle => '.', Minute::InUsePaused => 'p', Minute::InUse => 'u', Minute::Off => ' ' }
    }).collect();
    t += &format!("  timeline ({step} min per character, now on the right): R render, C campaign, . idle with nothing to run, p in use (work paused), u in use, blank: manager off\n  |{line}|\n");
    t += &format!("  renders finished: {}\n", s.renders_done.len());
    for (l, _) in &s.renders_done { t += &format!("    {l}\n"); }
    let ours = s.trials_of(true);
    t += &format!("  campaign trials finished while the computer was not in use: {}", ours.len());
    if let Some(b) = s.best(true) { t += &format!(", best {:.3} ({} trial {})", b.value, b.campaign, b.n); }
    t += &format!("; winners kept: {}\n", s.winners.iter().filter(|w| w.2).count());
    t += &format!("  (all campaign trials in the window, including campaigns started by hand: {}", s.trials.len());
    if let Some(b) = s.best(false) { t += &format!(", best {:.3}", b.value); }
    t += ")\n";
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(t: u64, e: &str) -> Event { Event { t, ev: e.into(), n: None, why: None } }

    #[test]
    fn minutes_replay_the_events() {
        let dir = std::env::temp_dir().join(format!("nnf_idle_stats_{}", std::process::id()));
        fs::create_dir_all(dir.join("explorer_out")).unwrap();
        let now = now_s();
        let t0 = now - 3600; // one hour ago
        let evs = [ev(t0 - 600, "start"), ev(t0 + 600, "idle"), ev(t0 + 600, "camp_start"), ev(t0 + 1800, "use"), ev(t0 + 1800, "camp_pause"), ev(t0 + 2400, "camp_stop_clean")];
        let text: Vec<String> = evs.iter().map(|e| serde_json::to_string(e).unwrap()).collect();
        fs::write(dir.join(EVENTS), text.join("\n")).unwrap();
        let s = stats(&dir, 1.0);
        assert_eq!(s.minutes.len(), 60);
        assert_eq!(s.minutes[5], Minute::InUse);
        assert_eq!(s.minutes[15], Minute::Campaign);
        assert_eq!(s.minutes[35], Minute::InUsePaused);
        assert_eq!(s.minutes[50], Minute::InUse);
        assert!((s.campaign_h - 20.0 / 60.0).abs() < 0.02, "{}", s.campaign_h);
        assert_eq!((s.times_idle, s.pauses, s.clean_stops, s.campaigns_started), (1, 1, 1, 1));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn settings_round_trip() {
        let c = Config::default();
        let back: Config = toml::from_str(&format!("{CONFIG_HEADER}\n{}", toml::to_string(&c).unwrap())).unwrap();
        assert_eq!(back, c);
        let part: Config = toml::from_str("idle_after_min = 3").unwrap();
        assert_eq!(part.idle_after_min, 3.0);
        assert!(part.campaign_env.contains_key("EXTRA"));
    }
}
