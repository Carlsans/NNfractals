//! Queue processing logic shared by the interactive `nnfractals-queue` GUI
//! and the headless `explorer queue-run` subcommand (what a cron job fires at
//! night for unattended rendering).
//!
//! [`process_queue_item`] used to live inside `queue.rs` itself, wired
//! directly to an `egui::Context` for repaint requests. Pulled out here, with
//! the repaint calls replaced by a plain `&dyn Fn(QueueProgress)` callback, so
//! a CLI process with no GUI at all can drive the exact same render path the
//! interactive window does — not a reimplementation that could drift from it.

use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread;

use crate::config::Config;
use crate::formula::ModShape;
use crate::io::load_genome;
use crate::video_export::{
    export_blend_video, export_chain_time_video, export_time_video,
    export_video_chain_interpolated, interpolate_with_rife, queue_dir, QueueItem, VideoMsg,
    DEFAULT_TIME_FRAMES,
};

// ── Wall-clock hold window ──────────────────────────────────────────────

/// Wall-clock window during which the queue is allowed to start a new item.
///
/// Stored as minutes past midnight so it survives a restart (or a `--until`
/// CLI flag) as a plain number. `None` means no restriction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HoldWindow {
    pub start_min: u32,
    pub end_min: u32,
}

impl HoldWindow {
    /// Whether `minute_of_day` falls inside the window.
    ///
    /// Handles a window that wraps midnight, which is the normal case here —
    /// "render overnight" means something like 23:00 to 07:00, and a naive
    /// `start <= t && t < end` would make that window empty.
    pub fn contains(&self, minute_of_day: u32) -> bool {
        if self.start_min <= self.end_min {
            minute_of_day >= self.start_min && minute_of_day < self.end_min
        } else {
            minute_of_day >= self.start_min || minute_of_day < self.end_min
        }
    }
}

/// Parse "HH:MM" into minutes past midnight.
pub fn parse_hhmm(s: &str) -> Option<u32> {
    let (h, m) = s.trim().split_once(':')?;
    let h: u32 = h.trim().parse().ok()?;
    let m: u32 = m.trim().parse().ok()?;
    (h < 24 && m < 60).then_some(h * 60 + m)
}

/// Local minute of day, from the system clock.
pub fn now_minute_of_day() -> u32 {
    // `date +%H:%M` rather than a timezone crate: this project has no chrono
    // dependency, and `SystemTime` is UTC — which for a "render overnight"
    // setting would silently be wrong by the local offset.
    let out = std::process::Command::new("date").arg("+%H:%M").output().ok();
    out.and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| parse_hhmm(&s))
        .unwrap_or(0)
}

// ── Mutual exclusion between the GUI and a headless run ────────────────

/// A simple PID lock file so the interactive queue window and a headless
/// `queue-run` invocation never process the SAME queue.json concurrently —
/// each independently picks "the oldest Pending item," so if both started at
/// once they could pick the same one and race writing its status/output.
///
/// Held for the duration of processing ONE item, not the whole session: a
/// long-running headless run releases it between items so the interactive
/// window (opened to check on things, or to Approve more reels) can still
/// take a turn rather than being locked out all night.
pub struct ProcessorLock(PathBuf);

impl ProcessorLock {
    fn path() -> PathBuf {
        queue_dir().join(".processing.lock")
    }

    /// Try to take the lock. `None` if another live process already holds it.
    ///
    /// A lock file left behind by a process that crashed or was killed (`kill
    /// -9`, a power loss) would otherwise wedge every future run forever, so
    /// a lock whose PID is no longer alive (`/proc/<pid>` absent — Linux
    /// only, matching the rest of this project) is treated as stale and
    /// reclaimed.
    pub fn acquire() -> Option<Self> {
        let path = Self::path();
        if let Ok(existing) = std::fs::read_to_string(&path) {
            if let Ok(pid) = existing.trim().parse::<u32>() {
                if pid != std::process::id() && Path::new(&format!("/proc/{pid}")).exists() {
                    return None;
                }
            }
        }
        // Best-effort: another process could win a race here between the
        // read above and this write, in which case whichever writes LAST
        // holds the lock and the other's item will simply be picked up on
        // its next poll — a missed turn, not a corrupted queue.
        std::fs::write(&path, std::process::id().to_string()).ok()?;
        Some(ProcessorLock(path))
    }
}

impl Drop for ProcessorLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

// ── CPU load monitoring ─────────────────────────────────────────────────

/// System-wide busy core-equivalents and our own process's, over a short
/// sampling window — `(system_busy_cores, own_cores)`. `None` off Linux, or
/// if `/proc` is unreadable (e.g. a sandboxed environment).
fn cpu_load_sample(window: std::time::Duration) -> Option<(f64, f64)> {
    fn cpu_snapshot() -> Option<(u64, u64)> {
        let s = std::fs::read_to_string("/proc/stat").ok()?;
        let line = s.lines().next()?; // "cpu  user nice system idle iowait irq softirq steal ..."
        let fields: Vec<u64> = line.split_whitespace().skip(1)
            .filter_map(|f| f.parse().ok()).collect();
        if fields.len() < 4 { return None; }
        let idle = fields[3] + fields.get(4).copied().unwrap_or(0); // idle + iowait
        let total: u64 = fields.iter().sum();
        Some((total.saturating_sub(idle), total))
    }
    // utime (field 14) + stime (field 15) of THIS process, which on Linux is
    // already summed across every thread we own (rayon workers included) —
    // no need to walk /proc/self/task separately.
    fn own_ticks() -> Option<u64> {
        let s = std::fs::read_to_string("/proc/self/stat").ok()?;
        // The comm field (2nd, in parens) can itself contain spaces or
        // parens, so split on the LAST ')' rather than whitespace.
        let after = s.rfind(')')?;
        let rest = s.get(after + 2..)?;
        let fields: Vec<&str> = rest.split_whitespace().collect();
        // `rest` starts at field 3 (state), so utime/stime (fields 14/15
        // overall) are indices 11/12 here.
        let utime: u64 = fields.get(11)?.parse().ok()?;
        let stime: u64 = fields.get(12)?.parse().ok()?;
        Some(utime + stime)
    }

    let (busy0, total0) = cpu_snapshot()?;
    let own0 = own_ticks()?;
    thread::sleep(window);
    let (busy1, total1) = cpu_snapshot()?;
    let own1 = own_ticks()?;

    let d_total = total1.saturating_sub(total0).max(1) as f64;
    let d_busy = busy1.saturating_sub(busy0) as f64;
    let d_own = own1.saturating_sub(own0) as f64;
    let ncpu = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4) as f64;
    Some((d_busy / d_total * ncpu, d_own / d_total * ncpu))
}

/// The actual policy, isolated from `/proc` so it can be tested without a
/// real system to sample: given the machine's core count and how many of
/// them something ELSE is using right now, how many should rendering take?
///
/// Always leaves `min_free_cores` free even when nothing else is running —
/// this is what stops a night run from ever claiming literally every core —
/// and never recommends zero: a stalled render makes no progress at all,
/// where one core alongside a heavy foreground process is still forward
/// motion, just slower. `other_cores` is rounded UP before reserving it: a
/// process using 1.2 cores still needs a full 2nd core of room, not 1.8.
pub fn recommend_thread_count_for(total_cores: usize, other_cores: f64, min_free_cores: usize) -> usize {
    let reserved = other_cores.max(0.0).ceil() as usize + min_free_cores;
    total_cores.saturating_sub(reserved).max(1)
}

/// [`recommend_thread_count_for`], sampling the real system over `window`.
/// Falls back to half the cores (the same default the interactive queue
/// starts with) if `/proc` can't be read.
pub fn recommend_thread_count(min_free_cores: usize, window: std::time::Duration) -> usize {
    let total = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    match cpu_load_sample(window) {
        Some((busy, own)) => {
            let other = (busy - own).max(0.0);
            recommend_thread_count_for(total, other, min_free_cores)
        }
        None => (total / 2).max(1),
    }
}

// ── Progress reporting (replaces direct egui::Context repaint calls) ────

/// What `process_queue_item` reports as it goes, for a caller to relay
/// however it likes — updating GUI state and repainting, or just printing.
pub enum QueueProgress {
    /// The render subprocess's PID, once it's known (lets a caller offer a
    /// "cancel" button, or in a headless run, just log it).
    Pid(u32),
    Frame(u32, u32),
    /// Status text for the optional RIFE interpolation pass.
    Rife(String),
}

/// Frame count of a rendered clip, straight from the container.
fn rendered_frame_count(path: &Path) -> Option<u32> {
    let out = std::process::Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0",
               "-count_frames", "-show_entries", "stream=nb_read_frames",
               "-of", "csv=p=0"])
        .arg(path)
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout).trim().parse().ok()
}

/// Render one queue item end to end: load the genome, dispatch to whichever
/// exporter its fields call for, run the optional RIFE pass, and return the
/// output path — or an error, for the caller to record on the item.
///
/// This IS the queue's actual work; everything else (finding the next
/// Pending item, writing its status, the hold window, the processor lock) is
/// the caller's job, because the GUI and the headless runner each have a
/// different idea of what "the next item" and "record the result" mean (a
/// `Mutex`-backed field with a repaint vs. a plain `load_queue`/`save_queue`
/// round-trip).
pub fn process_queue_item(
    item: &QueueItem, on_progress: &(dyn Fn(QueueProgress) + Sync),
) -> Result<String, String> {
    let rife_fps = item.rife_fps;
    let nn_path = queue_dir().join(&item.nn_filename);
    let genome = load_genome(&nn_path).map_err(|e| format!("failed to load {}: {e}", nn_path.display()))?;
    // Project-root-relative, NOT CWD-relative — see `crate::project_root`'s
    // doc comment. This exact line was the reported crash (Carl, 2026-08-11):
    // "No such file or directory" when this window was spawned by a viewer
    // launched via the file manager, whose working directory wasn't the
    // project root — and it matters just as much for a cron-launched
    // headless run, whose working directory is whatever cron happened to use.
    let config_path = crate::project_root().join("config.toml");
    let mut config = Config::load(&config_path)
        .map_err(|e| format!("failed to load {}: {e}", config_path.display()))?;
    config.rendering.colormap = item.colormap.clone();

    let out_dir = PathBuf::from(&item.output_dir);
    std::fs::create_dir_all(&out_dir).map_err(|e| format!("cannot create {}: {e}", out_dir.display()))?;
    let base = format!("{}_zoom_{}s_{}fps", item.genome_label, item.steps, item.fps);
    let mut out_path = out_dir.join(format!("{base}.mp4"));
    let mut n = 2;
    while out_path.exists() {
        out_path = out_dir.join(format!("{base}_{n}.mp4"));
        n += 1;
    }

    let (tx, rx) = mpsc::channel::<VideoMsg>();
    let (g2, c2, start, end) = (genome, config, item.start, item.end);
    let waypoints = item.waypoints.clone();
    let (steps, fps, w, h, invc, invr, ang) = (
        item.steps, item.fps, item.width, item.height,
        item.invert_coords, item.invert_range, item.angle_coloring,
    );
    let kf_stride = item.keyframe_stride;
    let time_mod = item.time_mod.clone();
    let time_prog = item.time_prog.clone();
    // The morph partner travels as its own .nn beside the item's, same as the
    // main genome — a whole genome inside the queue JSON would be unreadable.
    let blend_partner = item.blend_nn_filename.as_ref()
        .and_then(|f| load_genome(&queue_dir().join(f)).ok());
    let blend_shape = ModShape::parse(&item.blend_shape).unwrap_or(ModShape::Sine);
    let blend_amp = if item.blend_amp > 0.0 { item.blend_amp } else { 0.15 };
    let camera_moves = item.camera_moves();
    let animates_formula = item.animates_formula();
    let time_frames = if item.time_frames >= 2 { item.time_frames } else { DEFAULT_TIME_FRAMES };
    let out_path2 = out_path.clone();
    let render_handle = thread::spawn(move || {
        // A time item animates the FORMULA with the camera held still, so it
        // must not go anywhere near the camera-path exporters — and in
        // particular not near the interpolated one, which warps intermediate
        // frames on the assumption that only the camera moved.
        if animates_formula {
            let mut g_anim = g2.clone();
            g_anim.time_mod = time_mod;
            g_anim.time_prog = time_prog;
            if camera_moves {
                // Both axes at once: the camera travels the chain exactly as a
                // normal zoom does while the formula animates across the clip.
                let wp = if waypoints.len() >= 2 { waypoints.clone() } else { vec![start, end] };
                export_chain_time_video(
                    &g_anim, blend_partner.as_ref(), &c2, ang, &wp, steps, fps, w, h,
                    invc, invr, blend_shape, 1.0, 0.0, blend_amp,
                    &out_path2, &tx, &|| {});
            } else {
                let view = start.to_view();
                match blend_partner {
                    Some(partner) => export_blend_video(
                        &g_anim, &partner, &c2, ang, &view, time_frames, fps, w, h,
                        blend_shape, 1.0, 0.0, blend_amp,
                        &out_path2, &tx, &|| {}),
                    None => export_time_video(&g_anim, &c2, ang, &view, time_frames, fps, w, h,
                                              &out_path2, &tx, &|| {}),
                }
            }
            return;
        }
        // A wormhole-chain item carries its own waypoint sequence — render
        // ALL of it as one continuous multi-leg video, not just start→end
        // (which for a chain item are only the first/last waypoint, kept
        // solely for older UI that expects those two fields to exist). A
        // plain start→end item is just the 2-waypoint case of the same thing.
        let waypoints = if waypoints.len() >= 2 { waypoints } else { vec![start, end] };
        export_video_chain_interpolated(&g2, &c2, ang, &waypoints, steps, fps, w, h, invc, invr,
                            &out_path2, &tx, &|| {}, None, kf_stride);
    });

    let mut result: Option<Result<String, String>> = None;
    for msg in rx {
        match msg {
            VideoMsg::Started { pid } => on_progress(QueueProgress::Pid(pid)),
            VideoMsg::Progress { done, total } => on_progress(QueueProgress::Frame(done, total)),
            VideoMsg::Done(p) => result = Some(Ok(p.to_string_lossy().into_owned())),
            VideoMsg::Failed(e) => result = Some(Err(e)),
        }
    }
    let _ = render_handle.join();

    // Optional RIFE pass. A failure here does NOT fail the item: the render
    // succeeded and that file is real and kept. Interpolation is a bonus pass
    // over it, so the worst case is a note saying why there is no smoothed
    // version — losing a finished render because a post-process could not
    // find its binary would be indefensible.
    if let Some(Ok(path)) = &result {
        if rife_fps > 0 {
            let src = PathBuf::from(path);
            let frames = rendered_frame_count(&src).unwrap_or(0);
            on_progress(QueueProgress::Rife("interpolating…".to_string()));
            match interpolate_with_rife(&src, frames, fps, rife_fps,
                &|stage| on_progress(QueueProgress::Rife(format!("RIFE: {stage}")))) {
                Ok(out) => on_progress(QueueProgress::Rife(
                    format!("interpolated → {}", out.file_name().unwrap_or_default().to_string_lossy()))),
                Err(e) => on_progress(QueueProgress::Rife(format!("RIFE skipped: {e}"))),
            }
        }
    }

    result.unwrap_or_else(|| Err("export thread ended without a result".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hold_window_handles_the_midnight_wrap() {
        let overnight = HoldWindow { start_min: 23 * 60, end_min: 7 * 60 };
        assert!(overnight.contains(23 * 60 + 30), "23:30 is inside 23:00-07:00");
        assert!(overnight.contains(3 * 60), "03:00 is inside 23:00-07:00");
        assert!(!overnight.contains(12 * 60), "noon is outside 23:00-07:00");

        let daytime = HoldWindow { start_min: 9 * 60, end_min: 17 * 60 };
        assert!(daytime.contains(12 * 60));
        assert!(!daytime.contains(20 * 60));
    }

    #[test]
    fn parse_hhmm_rejects_garbage() {
        assert_eq!(parse_hhmm("23:00"), Some(23 * 60));
        assert_eq!(parse_hhmm(" 7:05 "), Some(7 * 60 + 5));
        assert_eq!(parse_hhmm("24:00"), None, "hour must be < 24");
        assert_eq!(parse_hhmm("7:60"), None, "minute must be < 60");
        assert_eq!(parse_hhmm("garbage"), None);
        assert_eq!(parse_hhmm(""), None);
    }

    #[test]
    fn thread_recommendation_leaves_room_for_other_work() {
        // Nothing else running: use everything except the headroom.
        assert_eq!(recommend_thread_count_for(8, 0.0, 1), 7);
        // Something using a couple of cores: leave room for it AND headroom.
        assert_eq!(recommend_thread_count_for(8, 2.0, 1), 5);
        // A fractional core still reserves a WHOLE core of room (2, not 1),
        // on top of the 1 core of headroom that's always reserved.
        assert_eq!(recommend_thread_count_for(8, 1.2, 1), 5, "1.2 must round up to 2 reserved");
        // Never zero, however saturated the system looks — a stalled render
        // makes no progress, one slow core still does.
        assert_eq!(recommend_thread_count_for(8, 20.0, 1), 1);
        assert_eq!(recommend_thread_count_for(2, 0.0, 1), 1);
        // Headroom is never skipped even when nothing else is measured.
        assert_eq!(recommend_thread_count_for(4, 0.0, 2), 2);
    }

    #[test]
    fn thread_recommendation_never_exceeds_the_machine() {
        assert_eq!(recommend_thread_count_for(4, -5.0, 0),
                   4, "a negative sample (measurement noise) must not hand out MORE than exists");
    }
}
