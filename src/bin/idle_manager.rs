//! `nnfractals-idle`: puts the computer to work while nobody uses it.
//!
//! Not in use = no keyboard, mouse, touch or remote-desktop input for `idle_after_min`
//! minutes, read straight from /dev/input (Carl is in the `input` group; swayidle did not
//! behave on this machine), and by default no sound coming out (a film without touching
//! anything is still use; measured on the output, as Firefox keeps silent streams open). While not in use:
//!   1. the video render queue first (`explorer queue-run`), and new renders go ahead of
//!      the campaign;
//!   2. then the GA campaign (`scripts/optuna_campaigns.py` with `campaign_env`), its GA
//!      workers scaled to the free memory through the campaign's control file.
//! Back at the computer: everything it started is paused at once (SIGSTOP, nothing lost).
//! After `free_memory_after_min` of use the campaign's trial is ended cleanly (the script
//! writes its summary and exits; its memory and GPU memory are freed); renders stay paused
//! with their progress. It only ever touches what it started itself; a campaign or render
//! started by hand is left alone (and a second campaign is not started next to it).
//!
//! Settings: `idle_manager.toml` in the project root (written with comments on first run).
//! State: `explorer_out/idle_manager_status.json`, shown by `nnfractals-idle status`.
//! Log: stdout (the journal of the `nnfractals-idle` user service) and
//! `explorer_out/idle_manager.log`.

use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

const STOP_FILE: &str = "STOP_campaigns";
const STATUS: &str = "explorer_out/idle_manager_status.json";
const LOG: &str = "explorer_out/idle_manager.log";
const CONFIG: &str = "idle_manager.toml";

#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
struct Config {
    idle_after_min: f64,
    sound_counts_as_use: bool,
    renders: bool,
    campaign: bool,
    free_memory_after_min: f64,
    max_workers: usize,
    min_free_gb: f64,
    campaign_log: String,
    ignore_devices: Vec<String>,
    campaign_env: BTreeMap<String, String>,
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

const CONFIG_HEADER: &str = "\
# nnfractals-idle: what runs while nobody uses the computer (see src/bin/idle_manager.rs).
# Read at start: `systemctl --user restart nnfractals-idle` after editing.
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

fn load_config(root: &Path) -> Config {
    let p = root.join(CONFIG);
    match fs::read_to_string(&p) {
        Ok(t) => toml::from_str(&t).unwrap_or_else(|e| {
            log(&format!("{CONFIG}: {e}; using the defaults"));
            Config::default()
        }),
        Err(_) => {
            let c = Config::default();
            let body = toml::to_string(&c).unwrap_or_default();
            let _ = fs::write(&p, format!("{CONFIG_HEADER}\n{body}"));
            log(&format!("wrote the default settings to {}", p.display()));
            c
        }
    }
}

// ---------------------------------------------------------------- small helpers

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

fn stamp() -> String {
    // SAFETY: plain libc time calls on stack values.
    unsafe {
        let t = libc::time(std::ptr::null_mut());
        let mut tm: libc::tm = std::mem::zeroed();
        libc::localtime_r(&t, &mut tm);
        format!("{:04}-{:02}-{:02} {:02}:{:02}:{:02}", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec)
    }
}

fn log(msg: &str) {
    let line = format!("{} {msg}", stamp());
    println!("{line}");
    if cfg!(test) { return; }
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(LOG) {
        let _ = writeln!(f, "{line}");
    }
}

fn mem_available_gb() -> f64 {
    fs::read_to_string("/proc/meminfo").ok()
        .and_then(|t| t.lines().find(|l| l.starts_with("MemAvailable:")).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse::<f64>().ok()))
        .map(|kb| kb / 1048576.0)
        .unwrap_or(0.0)
}

/// Sound actually coming out of the speakers: one second of the default output's monitor
/// (parec), its peak above about -44 dBFS. Whether a stream is open says nothing: Firefox
/// tabs (Gmail, Docs...) keep silent streams open all day, so the sound card always reads
/// RUNNING.
fn sound_playing() -> bool {
    let Ok(mut child) = Command::new("timeout")
        .args(["3", "parec", "-d", "@DEFAULT_MONITOR@", "--raw", "--format=s16le", "--rate=8000", "--channels=1", "--latency-msec=100"])
        .stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::null()).spawn() else { return false };
    let mut buf = vec![0u8; 16000];
    let mut got = 0;
    if let Some(mut out) = child.stdout.take() {
        while got < buf.len() {
            match out.read(&mut buf[got..]) {
                Ok(0) | Err(_) => break,
                Ok(n) => got += n,
            }
        }
    }
    let _ = child.kill();
    let _ = child.wait();
    let peak = buf[..got & !1].chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]]).unsigned_abs()).max().unwrap_or(0);
    peak > 200
}

/// (pid, ppid, pgrp, state) from /proc/<pid>/stat.
fn proc_stat(pid: i32) -> Option<(i32, i32, i32, char)> {
    let s = fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let rest = &s[s.rfind(')')? + 2..];
    let mut it = rest.split_whitespace();
    let state = it.next()?.chars().next()?;
    let ppid = it.next()?.parse().ok()?;
    let pgrp = it.next()?.parse().ok()?;
    Some((pid, ppid, pgrp, state))
}

fn all_pids() -> Vec<i32> {
    fs::read_dir("/proc").map(|d| d.flatten().filter_map(|e| e.file_name().to_str()?.parse().ok()).collect()).unwrap_or_default()
}

fn cmdline(pid: i32) -> String {
    fs::read(format!("/proc/{pid}/cmdline")).map(|b| String::from_utf8_lossy(&b).replace('\0', " ")).unwrap_or_default()
}

fn alive(pid: i32) -> bool {
    proc_stat(pid).is_some_and(|s| s.3 != 'Z' && s.3 != 'X')
}

fn group_members(pgid: i32) -> Vec<i32> {
    all_pids().into_iter().filter(|&p| proc_stat(p).is_some_and(|s| s.2 == pgid && s.3 != 'Z')).collect()
}

fn signal_group(pgid: i32, sig: i32) {
    // SAFETY: kill(2) on a process group we created.
    unsafe { libc::kill(-pgid, sig); }
}

// ---------------------------------------------------------------- input activity

/// Last input from any keyboard, mouse, touchpad, tablet or virtual (remote) device.
struct Activity {
    last_ms: AtomicU64,
    last_dev: Mutex<String>,
    /// /dev/input/eventN -> (device name, last event ms)
    devices: Mutex<BTreeMap<String, (String, u64)>>,
}

fn device_usable(ev: &str, ignore: &[String]) -> Option<String> {
    let base = Path::new("/sys/class/input").join(ev).join("device");
    let name = fs::read_to_string(base.join("name")).ok()?.trim().to_string();
    if ignore.iter().any(|i| i == &name) { return None; }
    let caps = fs::read_to_string(base.join("capabilities/ev")).ok()?;
    let bits = u64::from_str_radix(caps.trim(), 16).ok()?;
    // EV_KEY (1), EV_REL (2), EV_ABS (3): keys, pointer motion, touch/tablet.
    (bits & 0b1110 != 0).then_some(name)
}

/// Starts a reader thread for every input device not watched yet (call periodically: hot-plug).
fn watch_new_devices(act: &Arc<Activity>, ignore: &[String]) {
    let Ok(dir) = fs::read_dir("/dev/input") else { return };
    for e in dir.flatten() {
        let ev = e.file_name().to_string_lossy().to_string();
        if !ev.starts_with("event") { continue; }
        let path = e.path().display().to_string();
        if act.devices.lock().unwrap().contains_key(&path) { continue; }
        let Some(name) = device_usable(&ev, ignore) else { continue };
        let Ok(mut f) = File::open(&path) else { continue };
        act.devices.lock().unwrap().insert(path.clone(), (name.clone(), 0));
        let act = Arc::clone(act);
        std::thread::spawn(move || {
            // struct input_event on 64-bit Linux: timeval (16 bytes), type u16, code u16, value i32.
            let mut buf = [0u8; 24 * 64];
            loop {
                match f.read(&mut buf) {
                    Ok(n) if n >= 24 => {
                        let real = buf[..n].chunks_exact(24).any(|ev| matches!(u16::from_ne_bytes([ev[16], ev[17]]), 1..=3));
                        if real {
                            let t = now_ms();
                            act.last_ms.store(t, Ordering::Relaxed);
                            *act.last_dev.lock().unwrap() = name.clone();
                            if let Some(d) = act.devices.lock().unwrap().get_mut(&path) { d.1 = t; }
                        }
                    }
                    _ => break, // unplugged
                }
            }
            act.devices.lock().unwrap().remove(&path);
        });
    }
}

// ---------------------------------------------------------------- jobs and status

#[derive(Clone, Serialize, Deserialize)]
struct Job {
    /// Leader pid = process group id (each job is started in a group of its own).
    pgid: i32,
    paused: bool,
    started_ms: u64,
}

#[derive(Default, Serialize, Deserialize)]
struct Status {
    in_use: bool,
    since_ms: u64,
    last_input_ms: u64,
    last_device: String,
    sound_playing: bool,
    mem_free_gb: f64,
    workers: Option<usize>,
    render: Option<Job>,
    campaign: Option<Job>,
    note: String,
    devices: Vec<(String, u64)>,
    updated_ms: u64,
}

fn read_status(root: &Path) -> Status {
    fs::read_to_string(root.join(STATUS)).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default()
}

fn write_status(root: &Path, s: &Status) {
    let p = root.join(STATUS);
    let tmp = p.with_extension("json.tmp");
    if fs::write(&tmp, serde_json::to_string_pretty(s).unwrap_or_default()).is_ok() { let _ = fs::rename(tmp, p); }
}

/// A job from an earlier run of the manager, still alive and still what it was.
fn adopt(job: Option<Job>, what: &str) -> Option<Job> {
    job.filter(|j| alive(j.pgid) && cmdline(j.pgid).contains(what))
}

fn pause(job: &mut Job, what: &str, why: &str) {
    if !job.paused {
        signal_group(job.pgid, libc::SIGSTOP);
        job.paused = true;
        log(&format!("paused the {what} ({why})"));
    }
}

fn resume(job: &mut Job, what: &str) {
    if job.paused {
        signal_group(job.pgid, libc::SIGCONT);
        job.paused = false;
        log(&format!("resumed the {what}"));
    }
}

fn spawn(root: &Path, program: &Path, args: &[&str], env: &BTreeMap<String, String>, log_file: &str) -> std::io::Result<Child> {
    let out = OpenOptions::new().create(true).append(true).open(root.join(log_file))?;
    Command::new(program).args(args).envs(env).current_dir(root)
        .stdin(Stdio::null()).stdout(out.try_clone()?).stderr(out)
        .process_group(0) // its own group: pause/resume/stop reach every child (ffmpeg, scorers)
        .spawn()
}

fn pending_renders() -> usize {
    nnfractals::video_export::load_queue().iter().filter(|i| i.status == nnfractals::video_export::QueueStatus::Pending).count()
}

/// A campaign script running that this manager did not start.
fn foreign_campaign(ours: Option<i32>) -> Option<i32> {
    all_pids().into_iter().find(|&p| Some(p) != ours && proc_stat(p).is_some_and(|s| Some(s.2) != ours && s.3 != 'Z')
        && cmdline(p).contains("optuna_campaigns.py") && !cmdline(p).contains("nnfractals-idle"))
}

/// Ends a (paused) campaign cleanly: the stop file, then its GA run killed, so the script
/// records the trials it finished (trials.json, manifest) and exits. Up to 3 minutes.
fn stop_campaign_cleanly(root: &Path, job: &Job) {
    let stop = root.join(STOP_FILE);
    let had_stop = stop.exists();
    let _ = fs::write(&stop, "nnfractals-idle: freeing memory while the computer is in use\n");
    let t0 = std::time::Instant::now();
    while alive(job.pgid) && t0.elapsed() < Duration::from_secs(180) {
        for p in group_members(job.pgid) {
            if p != job.pgid {
                // SAFETY: kill(2) on members of our own process group.
                unsafe { libc::kill(p, libc::SIGKILL); }
            }
        }
        // SAFETY: as above, the group's leader (the campaign script).
        unsafe { libc::kill(job.pgid, libc::SIGCONT); }
        std::thread::sleep(Duration::from_secs(2));
    }
    if alive(job.pgid) {
        log("the campaign script did not exit within 3 min; killing its group");
        signal_group(job.pgid, libc::SIGKILL);
    }
    if !had_stop { let _ = fs::remove_file(&stop); }
}

fn workers_for(mem_gb: f64, max: usize) -> usize {
    let w = if mem_gb >= 14.0 { 10 } else if mem_gb >= 9.0 { 6 } else if mem_gb >= 6.0 { 4 } else if mem_gb >= 4.0 { 2 } else { 1 };
    w.min(max.max(1))
}

// ---------------------------------------------------------------- main loop

static TERMINATE: AtomicBool = AtomicBool::new(false);
extern "C" fn on_terminate(_: libc::c_int) { TERMINATE.store(true, Ordering::Relaxed); }

fn run(root: &Path) {
    // One manager at a time.
    let lock_path = std::env::var("XDG_RUNTIME_DIR").map(PathBuf::from).unwrap_or_else(|_| std::env::temp_dir()).join("nnfractals-idle.lock");
    let lock = File::create(&lock_path).expect("lock file");
    // SAFETY: flock(2) on a file we own; held until exit.
    if unsafe { libc::flock(std::os::fd::AsRawFd::as_raw_fd(&lock), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        eprintln!("nnfractals-idle is already running ({})", lock_path.display());
        std::process::exit(1);
    }
    // SAFETY: the handler only stores into an atomic.
    unsafe {
        libc::signal(libc::SIGTERM, on_terminate as *const () as libc::sighandler_t);
        libc::signal(libc::SIGINT, on_terminate as *const () as libc::sighandler_t);
    }
    let cfg = load_config(root);
    let idle_after = (cfg.idle_after_min * 60_000.0) as u64;
    let free_after = (cfg.free_memory_after_min * 60_000.0) as u64;
    let act = Arc::new(Activity { last_ms: AtomicU64::new(now_ms()), last_dev: Mutex::new("(start)".into()), devices: Mutex::new(BTreeMap::new()) });
    watch_new_devices(&act, &cfg.ignore_devices);
    let names: Vec<String> = act.devices.lock().unwrap().values().map(|d| d.0.clone()).collect();
    if names.is_empty() {
        log("no readable input device in /dev/input (is this user in the `input` group?): never considered idle");
    }
    log(&format!("started: idle after {} min without input from {} device(s): {}{}", cfg.idle_after_min, names.len(), names.join(", "),
                 if cfg.sound_counts_as_use { "; sound playing counts as use" } else { "" }));

    let old = read_status(root);
    let mut render = adopt(old.render, "queue-run");
    let mut campaign = adopt(old.campaign, "optuna_campaigns.py");
    for (j, what) in [(&render, "render"), (&campaign, "campaign")] {
        if let Some(j) = j { log(&format!("took back the {what} it had started (group {}{})", j.pgid, if j.paused { ", paused" } else { "" })); }
    }
    let explorer = root.join("target/release/explorer");
    let mut children: Vec<Child> = Vec::new();
    let mut in_use = true;
    let mut since = now_ms();
    let mut render_retry_at = 0u64;
    let mut render_pending_at_start = 0usize;
    let mut campaign_retry_at = 0u64;
    let mut last_workers: Option<usize> = None;
    let mut note = String::new();
    let mut tick = 0u64;
    let (mut sound, mut sound_checked) = (false, 0u64);
    loop {
        if TERMINATE.load(Ordering::Relaxed) {
            // Stopping the manager hands its work back: nothing it paused stays frozen.
            // A new instance takes the jobs back (status file) and pauses them again if
            // the computer is in use.
            if let Some(j) = render.as_mut() { resume(j, "render"); }
            if let Some(j) = campaign.as_mut() { resume(j, "campaign"); }
            let mut st = read_status(root);
            st.render = render.clone();
            st.campaign = campaign.clone();
            st.note = "manager stopped; its jobs were left running".into();
            write_status(root, &st);
            log("stopping");
            return;
        }
        tick += 1;
        if tick % 6 == 0 { watch_new_devices(&act, &cfg.ignore_devices); }
        children.retain_mut(|c| !matches!(c.try_wait(), Ok(Some(_))));
        let now = now_ms();
        // Jobs that ended.
        if render.as_ref().is_some_and(|j| !alive(j.pgid)) {
            let left = pending_renders();
            log(&format!("render queue run ended ({} pending left)", left));
            if left > 0 && left >= render_pending_at_start { render_retry_at = now + 10 * 60_000; } // no progress: the window holds the queue, or an item keeps failing
            render = None;
        }
        if campaign.as_ref().is_some_and(|j| !alive(j.pgid)) {
            log("the campaign script ended");
            campaign = None;
            campaign_retry_at = now + 5 * 60_000;
        }

        let last_input = act.last_ms.load(Ordering::Relaxed);
        let input_recent = now.saturating_sub(last_input) < idle_after;
        // Sound is only measured once input has stopped, every 30 s at most.
        if !cfg.sound_counts_as_use || input_recent {
            sound = false;
        } else if now.saturating_sub(sound_checked) >= 30_000 {
            sound = sound_playing();
            sound_checked = now;
        }
        let use_now = input_recent || sound;
        if use_now != in_use {
            in_use = use_now;
            since = now;
            if in_use {
                let why = if input_recent { format!("input from {}", act.last_dev.lock().unwrap()) } else { "sound playing".into() };
                log(&format!("computer in use ({why})"));
            } else {
                log(&format!("computer not in use for {} min: work may run", cfg.idle_after_min));
            }
        }
        let mem = mem_available_gb();
        if in_use {
            if let Some(j) = render.as_mut() { pause(j, "render", "computer in use"); }
            if let Some(j) = campaign.as_mut() { pause(j, "campaign", "computer in use"); }
            if let Some(j) = campaign.as_ref() {
                if j.paused && now.saturating_sub(since) >= free_after {
                    log(&format!("in use for {} min: ending the campaign's trial to free its memory", cfg.free_memory_after_min));
                    stop_campaign_cleanly(root, j);
                    campaign = None;
                    log(&format!("campaign stopped; {:.1} GB free now", mem_available_gb()));
                }
            }
            note = "in use: work paused".into();
        } else {
            // 1) renders first.
            let pending = if cfg.renders { pending_renders() } else { 0 };
            if let Some(j) = render.as_mut() {
                if let Some(c) = campaign.as_mut() { pause(c, "campaign", "a render goes first"); }
                resume(j, "render");
                note = "rendering the video queue".into();
            } else if pending > 0 && now >= render_retry_at {
                if let Some(c) = campaign.as_mut() { pause(c, "campaign", "a render goes first"); }
                match spawn(root, &explorer, &["queue-run"], &BTreeMap::new(), "explorer_out/idle_manager_render.log") {
                    Ok(child) => {
                        let pgid = child.id() as i32;
                        log(&format!("{pending} render(s) pending: started the render queue (group {pgid})"));
                        render = Some(Job { pgid, paused: false, started_ms: now });
                        render_pending_at_start = pending;
                        children.push(child);
                    }
                    Err(e) => { log(&format!("could not start the render queue: {e}")); render_retry_at = now + 10 * 60_000; }
                }
                note = "rendering the video queue".into();
            } else if cfg.campaign {
                // 2) then the campaign.
                if let Some(c) = campaign.as_mut() {
                    if mem < cfg.min_free_gb {
                        pause(c, "campaign", &format!("only {mem:.1} GB free"));
                        note = format!("campaign paused: only {mem:.1} GB free");
                    } else if c.paused && mem >= cfg.min_free_gb + 1.0 {
                        resume(c, "campaign");
                    }
                    if !c.paused { note = "running the campaign".into(); }
                } else if let Some(p) = foreign_campaign(None) {
                    note = format!("a campaign started by hand is running (pid {p}); not starting another");
                } else if root.join(STOP_FILE).exists() {
                    note = format!("{STOP_FILE} exists: campaigns are stopped");
                } else if mem < cfg.min_free_gb + 2.0 {
                    note = format!("not starting the campaign: only {mem:.1} GB free");
                } else if now >= campaign_retry_at {
                    let mut env = cfg.campaign_env.clone();
                    env.insert("WORKERS".into(), cfg.max_workers.to_string());
                    match spawn(root, Path::new("python3"), &["-u", "scripts/optuna_campaigns.py"], &env, &cfg.campaign_log) {
                        Ok(child) => {
                            let pgid = child.id() as i32;
                            log(&format!("started the campaign (group {pgid}, {} → {})", env.get("OUT").map(String::as_str).unwrap_or("?"), cfg.campaign_log));
                            campaign = Some(Job { pgid, paused: false, started_ms: now });
                            children.push(child);
                            note = "running the campaign".into();
                        }
                        Err(e) => { log(&format!("could not start the campaign: {e}")); campaign_retry_at = now + 10 * 60_000; }
                    }
                }
            } else {
                note = "idle; nothing to do".into();
            }
        }
        // GA workers follow the free memory while the campaign runs.
        let workers = campaign.as_ref().filter(|c| !c.paused).map(|_| workers_for(mem, cfg.max_workers));
        if let (Some(w), Some(cf)) = (workers, cfg.campaign_env.get("CONTROL_FILE")) {
            if last_workers != Some(w) {
                let _ = fs::write(root.join(cf), format!("{{\"workers\": {w}}}\n"));
                log(&format!("GA workers: {w} ({mem:.1} GB free)"));
                last_workers = Some(w);
            }
        }
        if campaign.is_none() { last_workers = None; }

        let devices: Vec<(String, u64)> = act.devices.lock().unwrap().values().cloned().collect();
        write_status(root, &Status {
            in_use, since_ms: since, last_input_ms: last_input, last_device: act.last_dev.lock().unwrap().clone(),
            sound_playing: sound, mem_free_gb: mem, workers, render: render.clone(), campaign: campaign.clone(),
            note: note.clone(), devices, updated_ms: now,
        });
        std::thread::sleep(Duration::from_secs(5));
    }
}

fn ago(ms: u64) -> String {
    if ms == 0 { return "never".into(); }
    let s = now_ms().saturating_sub(ms) / 1000;
    if s < 120 { format!("{s} s ago") } else if s < 7200 { format!("{} min ago", s / 60) } else { format!("{:.1} h ago", s as f64 / 3600.0) }
}

fn show_status(root: &Path) {
    let s = read_status(root);
    if s.updated_ms == 0 { println!("nnfractals-idle has not run yet (no {STATUS})"); return; }
    let stale = now_ms().saturating_sub(s.updated_ms) > 30_000;
    println!("{}{}", if s.in_use { "computer in use" } else { "computer NOT in use" }, if stale { "  (status is old: is the service running?)" } else { "" });
    println!("  since {}, last input {} ({}){}", ago(s.since_ms), ago(s.last_input_ms), s.last_device, if s.sound_playing { ", sound playing" } else { "" });
    println!("  {}", s.note);
    for (j, what) in [(&s.render, "render"), (&s.campaign, "campaign")] {
        if let Some(j) = j { println!("  {what}: group {}, started {}{}", j.pgid, ago(j.started_ms), if j.paused { ", PAUSED" } else { ", running" }); }
    }
    if let Some(w) = s.workers { println!("  GA workers {w}, {:.1} GB free", s.mem_free_gb); } else { println!("  {:.1} GB free", s.mem_free_gb); }
    println!("  input devices watched:");
    for (name, t) in &s.devices { println!("    {name:50} last event {}", ago(*t)); }
}

fn main() {
    let root = nnfractals::project_root();
    std::env::set_current_dir(&root).expect("project root");
    match std::env::args().nth(1).as_deref() {
        None | Some("run") => run(&root),
        Some("status") => show_status(&root),
        Some(o) => {
            eprintln!("nnfractals-idle [run|status]: unknown {o}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workers_follow_memory_and_the_cap() {
        assert_eq!(workers_for(20.0, 6), 6);
        assert_eq!(workers_for(20.0, 12), 10);
        assert_eq!(workers_for(10.0, 6), 6);
        assert_eq!(workers_for(7.0, 6), 4);
        assert_eq!(workers_for(5.0, 6), 2);
        assert_eq!(workers_for(1.0, 6), 1);
    }

    #[test]
    fn default_settings_round_trip_through_the_file() {
        let c = Config::default();
        let text = format!("{CONFIG_HEADER}\n{}", toml::to_string(&c).unwrap());
        let back: Config = toml::from_str(&text).unwrap();
        assert_eq!(back.campaign_env.get("OUT").map(String::as_str), Some("fractals_dag_orgcx_loop"));
        assert_eq!(back.idle_after_min, 10.0);
        // A file with only one setting keeps the defaults for the rest.
        let part: Config = toml::from_str("idle_after_min = 3").unwrap();
        assert_eq!(part.idle_after_min, 3.0);
        assert!(part.renders && part.campaign_env.contains_key("EXTRA"));
    }

    #[test]
    fn proc_stat_reads_this_process() {
        let me = std::process::id() as i32;
        let (pid, _ppid, pgrp, state) = proc_stat(me).unwrap();
        assert_eq!(pid, me);
        assert!(pgrp > 0 && state != 'Z');
        assert!(alive(me));
    }

    #[test]
    fn a_paused_group_stops_and_continues() {
        // A child in a group of its own, paused and resumed like a render.
        let mut child = Command::new("sleep").arg("30").process_group(0).spawn().unwrap();
        let mut job = Job { pgid: child.id() as i32, paused: false, started_ms: 0 };
        pause(&mut job, "test", "test");
        std::thread::sleep(Duration::from_millis(200));
        assert_eq!(proc_stat(job.pgid).unwrap().3, 'T');
        resume(&mut job, "test");
        std::thread::sleep(Duration::from_millis(200));
        assert_ne!(proc_stat(job.pgid).unwrap().3, 'T');
        assert_eq!(group_members(job.pgid), vec![job.pgid]);
        signal_group(job.pgid, libc::SIGKILL);
        let _ = child.wait();
    }
}

#[cfg(test)]
mod sound_tests {
    #[test]
    fn measuring_the_output_never_hangs() {
        // Whatever the machine plays, the measurement returns within the 3 s timeout.
        let t0 = std::time::Instant::now();
        let _ = super::sound_playing();
        assert!(t0.elapsed() < std::time::Duration::from_secs(5));
    }
}
