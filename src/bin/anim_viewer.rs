//! Quaternion animation viewer — the primary quaternion-fractal animation
//! authoring GUI (`~/.claude/plans/majestic-sauteeing-nova.md`; the original
//! 9-phase build plan and the follow-up "Timeline UI Corrective Pass" that
//! replaced the form-based track editors with a kdenlive-style visual
//! timeline). Replaces `quat_viewer.rs` as the entry point for quaternion
//! genomes (`quat_viewer.rs` itself is left buildable-but-unlinked, not
//! deleted).
//!
//! Naming note: this file must never name anything `Quat` for 3D camera
//! rotation — `Quat`/`quaternion` already means the fractal's own 4D
//! iteration parameter throughout this codebase
//! (`nnfractals::quaternion::Quat`). See `anim_timeline`'s module doc.

use std::io::{Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread;

use eframe::egui;
use nnfractals::formula::OpNode;
use nnfractals::quat_dag::QuatDagFormula;
use nnfractals::render_gpu_raymarch_dag_codegen::CompiledDagPipeline;
use nnfractals::anim_timeline::{
    AnimParam, AnimationTimeline, AudioClip, AxisTimeline, BoundingBox, Drive, Effect, EffectClip,
    QuatPart, Ramp, RampStyle, ViewerAxis, WaveShape,
};

// ── IPC — single-instance socket (mirrors quat_viewer.rs's exactly, but a
//    DIFFERENT socket path/tag — this is a separate running instance from
//    quat_viewer.rs, not a drop-in replacement yet) ─────────────────────

struct SocketGuard(PathBuf);
impl Drop for SocketGuard {
    fn drop(&mut self) { let _ = std::fs::remove_file(&self.0); }
}

fn socket_path() -> PathBuf {
    let tag = std::env::var("USER")
        .or_else(|_| std::env::var("LOGNAME"))
        .unwrap_or_else(|_| "user".into());
    std::env::temp_dir().join(format!("nnfractals-anim-viewer-{tag}.sock"))
}

/// See `quat_viewer.rs`'s `try_delegate` for the full rationale (the ack
/// byte matters — without it a wedged/dead listener's socket file makes
/// delegation silently vanish instead of falling back to a real window).
fn try_delegate(sock: &Path, path: &Path) -> bool {
    let Ok(mut s) = UnixStream::connect(sock) else { return false };
    if s.write_all(path.to_string_lossy().as_bytes()).is_err() {
        return false;
    }
    let _ = s.shutdown(std::net::Shutdown::Write);
    let _ = s.set_read_timeout(Some(std::time::Duration::from_millis(500)));
    let mut ack = [0u8; 1];
    s.read_exact(&mut ack).is_ok()
}

// `recommended_orbit_radius`/`characteristic_scale` live in
// `nnfractals::anim_eval` — the batch CLI renderer needs them too.
use nnfractals::anim_eval::{characteristic_scale, recommended_orbit_radius, manual_orbit_basis, FOV_DEG};

fn gpu_error_message() -> String {
    match nnfractals::render_gpu_raymarch_dag_codegen::last_gpu_init_error() {
        Some(reason) => format!("GPU unavailable: {reason}"),
        None => "GPU unavailable — no reason was recorded. Check stderr for '[gpu-raymarch-dag-codegen]' lines.".to_string(),
    }
}

/// `AudioClip.peaks` is `#[serde(skip)]` (never persisted), so right after
/// loading a timeline from disk every clip has `peaks: None`. This decodes
/// them fresh from each clip's `file_path` so the waveform actually draws
/// after a reopen, without panicking if a referenced file has since moved
/// (logs and leaves `peaks: None` for that clip instead).
fn ensure_audio_peaks(timeline: &mut AnimationTimeline) {
    for clip in timeline.audio_clips.iter_mut() {
        if clip.peaks.is_none() {
            match nnfractals::video_export::decode_audio_peaks(&clip.file_path, 400) {
                Ok(peaks) => clip.peaks = Some(std::sync::Arc::new(peaks)),
                Err(e) => eprintln!("[anim-viewer] failed to decode waveform for {}: {e}", clip.file_path.display()),
            }
        }
    }
}

// ── Timeline panel: shared time<->pixel mapping ──────────────────────────
//
// Every row (ruler, trim bar, every lane) is drawn inside the SAME
// `egui::ScrollArea::horizontal()`, so a row's own allocated `Rect` is
// already in final screen-space position reflecting however far the user
// has scrolled — there is no separate "scroll offset in seconds" to track
// by hand. `rect` below is always that row's own full-content-width rect
// (which may be far wider than the visible viewport; the ScrollArea clips
// drawing/interaction to what's actually visible).

const LANE_HEIGHT: f32 = 26.0;
const RULER_HEIGHT: f32 = 20.0;
const TRIM_BAR_HEIGHT: f32 = 16.0;

fn time_to_x(rect: &egui::Rect, px_per_s: f32, t: f64) -> f32 {
    rect.left() + (t as f32) * px_per_s
}
fn x_to_time(rect: &egui::Rect, px_per_s: f32, x: f32) -> f64 {
    ((x - rect.left()) / px_per_s) as f64
}

/// A "nice" tick step (1/2/5 * 10^n) so labeled ruler ticks land at
/// human-friendly seconds regardless of zoom, rather than at an arbitrary
/// fraction.
fn nice_step(raw: f64) -> f64 {
    if raw <= 0.0 { return 1.0; }
    let mag = 10f64.powf(raw.log10().floor());
    let norm = raw / mag;
    let step = if norm < 1.5 { 1.0 } else if norm < 3.5 { 2.0 } else if norm < 7.5 { 5.0 } else { 10.0 };
    step * mag
}

fn fixed_height_label(ui: &mut egui::Ui, text: &str, height: f32) {
    let width = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::hover());
    ui.painter().text(rect.left_center(), egui::Align2::LEFT_CENTER, text, egui::FontId::proportional(11.0), ui.visuals().text_color());
}

/// Generalizes `viewer.rs`'s `time_prog_plot` Painter recipe
/// (`allocate_exact_size` -> `painter_at` -> filled background -> shapes)
/// to the timeline's own ruler: tick marks at "nice" seconds, a marker at
/// the authored duration, and the playhead position.
fn draw_ruler(ui: &mut egui::Ui, content_w: f32, px_per_s: f32, duration_s: f64, playhead_s: f64) -> egui::Rect {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(content_w, RULER_HEIGHT), egui::Sense::hover());
    let p = ui.painter_at(rect);
    p.rect_filled(rect, 0.0, egui::Color32::from_gray(26));
    let target_px = 70.0;
    let step = nice_step((target_px / px_per_s.max(0.01)) as f64);
    let visible_end_s = ((rect.width() / px_per_s.max(0.01)) as f64).max(duration_s);
    let mut t = 0.0;
    while t <= visible_end_s + step {
        let x = time_to_x(&rect, px_per_s, t);
        p.line_segment([egui::pos2(x, rect.top()), egui::pos2(x, rect.top() + 6.0)], egui::Stroke::new(1.0, egui::Color32::from_gray(120)));
        p.text(egui::pos2(x + 2.0, rect.top() + 6.0), egui::Align2::LEFT_TOP, format!("{t:.1}s"), egui::FontId::proportional(9.0), egui::Color32::from_gray(160));
        t += step;
    }
    let end_x = time_to_x(&rect, px_per_s, duration_s);
    p.line_segment([egui::pos2(end_x, rect.top()), egui::pos2(end_x, rect.bottom())], egui::Stroke::new(1.0, egui::Color32::from_gray(90)));
    let px = time_to_x(&rect, px_per_s, playhead_s);
    p.line_segment([egui::pos2(px, rect.top()), egui::pos2(px, rect.bottom())], egui::Stroke::new(2.0, egui::Color32::from_rgb(230, 200, 90)));
    rect
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TrimHandle { Start, End }

/// The dual-handle trim/render range bar — sets `trim_start_s`/`trim_end_s`
/// directly on the timeline ruler (replacing the old Render-modal-only
/// DragValues). Live-updates while dragging; caller commits on release
/// (mirrors the orbit-drag / clip-drag convention throughout this file).
/// Returns `(dirty, committed)`: `dirty` means the value changed this frame
/// and the preview should re-render, `committed` means a drag just ended
/// and the change should be persisted/pushed to the undo stack. Kept
/// separate (not one bool) so a live drag only re-renders every frame —
/// mirroring the orbit-drag's own "commit once, on release" convention —
/// instead of pushing a fresh undo entry and disk write on every single
/// frame of the drag.
fn draw_trim_bar(
    ui: &mut egui::Ui, content_w: f32, px_per_s: f32, duration_s: f64,
    trim_start_s: &mut f64, trim_end_s: &mut f64, dragging: &mut Option<TrimHandle>,
) -> (bool, bool) {
    let mut dirty = false;
    let mut committed = false;
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(content_w, TRIM_BAR_HEIGHT), egui::Sense::click_and_drag());
    let p = ui.painter_at(rect);
    p.rect_filled(rect, 0.0, egui::Color32::from_gray(22));
    let x0 = time_to_x(&rect, px_per_s, *trim_start_s);
    let x1 = time_to_x(&rect, px_per_s, *trim_end_s);
    let span = egui::Rect::from_min_max(egui::pos2(x0, rect.top() + 1.0), egui::pos2(x1, rect.bottom() - 1.0));
    p.rect_filled(span, 2.0, egui::Color32::from_rgba_unmultiplied(90, 160, 220, 100));
    const HANDLE_W: f32 = 6.0;
    for x in [x0, x1] {
        let h = egui::Rect::from_center_size(egui::pos2(x, rect.center().y), egui::vec2(HANDLE_W, rect.height()));
        p.rect_filled(h, 2.0, egui::Color32::from_rgb(230, 200, 90));
    }

    if resp.drag_started() {
        if let Some(pos) = resp.interact_pointer_pos() {
            *dragging = Some(if (pos.x - x0).abs() <= (pos.x - x1).abs() { TrimHandle::Start } else { TrimHandle::End });
        }
    }
    if resp.dragged() {
        if let (Some(pos), Some(handle)) = (resp.interact_pointer_pos(), *dragging) {
            let t = x_to_time(&rect, px_per_s, pos.x).clamp(0.0, duration_s);
            match handle {
                TrimHandle::Start => *trim_start_s = t.min(*trim_end_s - 0.01),
                TrimHandle::End => *trim_end_s = t.max(*trim_start_s + 0.01),
            }
            dirty = true;
        }
    }
    if resp.drag_stopped() {
        *dragging = None;
        committed = true;
    }
    (dirty, committed)
}

// ── Clip identity (any lane) ──────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum LaneRef { X, Y, Z, T, Effects(usize), Audio }

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ClipRef { lane: LaneRef, index: usize }

fn lane_label(lane: LaneRef) -> String {
    match lane {
        LaneRef::X => "X".to_string(),
        LaneRef::Y => "Y".to_string(),
        LaneRef::Z => "Z".to_string(),
        LaneRef::T => "T".to_string(),
        LaneRef::Effects(i) => format!("Effects lane {}", i + 1),
        LaneRef::Audio => "Audio".to_string(),
    }
}

fn effect_kind_label(effect: &Effect) -> &'static str {
    match effect {
        Effect::Rotation { .. } => "rotation",
        Effect::Translation { .. } => "translation",
        Effect::Scale { .. } => "scale",
        Effect::Difference { .. } => "difference",
        Effect::FixedDifference { .. } => "fixed difference",
        Effect::SlideIteration { .. } => "slide iteration",
    }
}

fn effect_color(effect: &Effect) -> egui::Color32 {
    match effect {
        Effect::Rotation { .. } => egui::Color32::from_rgb(160, 120, 220),
        Effect::Translation { .. } => egui::Color32::from_rgb(90, 160, 220),
        Effect::Scale { .. } => egui::Color32::from_rgb(220, 160, 90),
        Effect::Difference { .. } => egui::Color32::from_rgb(220, 90, 120),
        Effect::FixedDifference { .. } => egui::Color32::from_rgb(220, 90, 200),
        Effect::SlideIteration { .. } => egui::Color32::from_rgb(120, 220, 160),
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum ClipDragKind { ResizeLeft, ResizeRight, Move { grab_offset_s: f64 } }

#[derive(Clone, Copy, Debug, PartialEq)]
enum AudioDragKind {
    /// Anchors captured at drag-start so the source-relative shift computes
    /// without drift across many frames (unlike a plain EffectClip resize,
    /// which has no "source" to stay aligned with and can just set
    /// `start_s` directly from the cursor position each frame).
    ResizeLeft { orig_start_s: f64, orig_offset_s: f64 },
    ResizeRight,
    Move { grab_offset_s: f64 },
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum ClipEdgeHit { Left, Right, Body }

/// Edge tolerance is deliberately wider than egui's own 6pt click-vs-drag
/// threshold (`max_click_dist`): by the time a gesture is actually
/// recognized as a drag, the reported pointer position can already have
/// drifted a comparable distance from the press origin, so a tolerance
/// equal to that threshold would miss edges it should catch. The
/// containment check is against a slightly EXPANDED rect (not the drawn
/// rect exactly) for the same reason — a real edge-grab whose recognized
/// position has drifted just outside the clip's visual bounds should still
/// register as an edge hit, not fall through to "empty space."
fn hit_test_clip_rect(rects: &[egui::Rect], pos: egui::Pos2) -> Option<(usize, ClipEdgeHit)> {
    const EDGE_PX: f32 = 10.0;
    for (i, r) in rects.iter().enumerate().rev() {
        if r.expand(EDGE_PX * 0.6).contains(pos) {
            if (pos.x - r.left()).abs() < EDGE_PX { return Some((i, ClipEdgeHit::Left)); }
            if (pos.x - r.right()).abs() < EDGE_PX { return Some((i, ClipEdgeHit::Right)); }
            if r.contains(pos) { return Some((i, ClipEdgeHit::Body)); }
        }
    }
    None
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PendingAdd { lane: LaneRef, at_s: f64 }

/// One X/Y/Z/T axis track or effects lane, drawn as a row of draggable/
/// resizable clip blocks. Shared by axis tracks and effects lanes — both
/// are backed by the same `AxisTimeline`/`EffectClip` shape (audio is a
/// separate row type, `audio_lane_row`, since `AudioClip`'s span is derived
/// from offset/trim-out rather than being its own `start_s`/`end_s` pair).
/// Returns `(dirty, committed)` — see `draw_trim_bar`'s doc comment for why
/// they're kept separate rather than one bool.
fn axis_lane_row(
    ui: &mut egui::Ui, lane_ref: LaneRef, content_w: f32, px_per_s: f32, duration_s: f64,
    track: &mut AxisTimeline,
    selected: &mut Option<ClipRef>, dragging: &mut Option<(ClipRef, ClipDragKind)>, pending_add: &mut Option<PendingAdd>,
) -> (bool, bool) {
    let mut dirty = false;
    let mut committed = false;
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(content_w, LANE_HEIGHT), egui::Sense::click_and_drag());
    let p = ui.painter_at(rect);
    p.rect_filled(rect, 0.0, egui::Color32::from_gray(18));

    let clip_rects: Vec<egui::Rect> = track.clips.iter().map(|c| {
        let x0 = time_to_x(&rect, px_per_s, c.start_s);
        let x1 = time_to_x(&rect, px_per_s, c.end_s).max(x0 + 2.0);
        egui::Rect::from_min_max(egui::pos2(x0, rect.top() + 2.0), egui::pos2(x1, rect.bottom() - 2.0))
    }).collect();

    for (i, cr) in clip_rects.iter().enumerate() {
        let is_selected = *selected == Some(ClipRef { lane: lane_ref, index: i });
        p.rect_filled(*cr, 3.0, effect_color(&track.clips[i].effect));
        let stroke = if is_selected { egui::Stroke::new(2.0, egui::Color32::WHITE) } else { egui::Stroke::new(1.0, egui::Color32::from_gray(60)) };
        p.rect_stroke(*cr, 3.0, stroke, egui::StrokeKind::Inside);
        p.text(cr.left_top() + egui::vec2(3.0, 1.0), egui::Align2::LEFT_TOP, effect_kind_label(&track.clips[i].effect), egui::FontId::proportional(10.0), egui::Color32::WHITE);
    }

    if resp.drag_started() {
        if let Some(pos) = resp.interact_pointer_pos() {
            if let Some((i, hit)) = hit_test_clip_rect(&clip_rects, pos) {
                let t = x_to_time(&rect, px_per_s, pos.x);
                let kind = match hit {
                    ClipEdgeHit::Left => ClipDragKind::ResizeLeft,
                    ClipEdgeHit::Right => ClipDragKind::ResizeRight,
                    ClipEdgeHit::Body => ClipDragKind::Move { grab_offset_s: t - track.clips[i].start_s },
                };
                let r = ClipRef { lane: lane_ref, index: i };
                *dragging = Some((r, kind));
                *selected = Some(r);
            }
        }
    }
    if resp.dragged() {
        if let Some((r, kind)) = *dragging {
            if r.lane == lane_ref {
                if let (Some(pos), Some(clip)) = (resp.interact_pointer_pos(), track.clips.get_mut(r.index)) {
                    let t = x_to_time(&rect, px_per_s, pos.x);
                    match kind {
                        ClipDragKind::ResizeLeft => clip.start_s = t.clamp(0.0, clip.end_s - 0.01),
                        ClipDragKind::ResizeRight => clip.end_s = t.clamp(clip.start_s + 0.01, duration_s),
                        ClipDragKind::Move { grab_offset_s } => {
                            let len = clip.end_s - clip.start_s;
                            let new_start = (t - grab_offset_s).clamp(0.0, (duration_s - len).max(0.0));
                            clip.start_s = new_start;
                            clip.end_s = new_start + len;
                        }
                    }
                    dirty = true;
                }
            }
        }
    }
    if resp.drag_stopped() {
        if let Some((r, _)) = *dragging {
            if r.lane == lane_ref { committed = true; }
        }
        *dragging = None;
    }
    if resp.clicked() {
        if let Some(pos) = resp.interact_pointer_pos() {
            if let Some((i, _)) = hit_test_clip_rect(&clip_rects, pos) {
                *selected = Some(ClipRef { lane: lane_ref, index: i });
            } else {
                let t = x_to_time(&rect, px_per_s, pos.x).max(0.0);
                *pending_add = Some(PendingAdd { lane: lane_ref, at_s: t });
            }
        }
    }
    (dirty, committed)
}

/// The audio lane — one row of positioned, independently trimmable clips.
/// Left-edge drag adjusts `offset_s` (trim-in) AND `start_s` together, so
/// the timeline position of the trimmed-in point tracks the cursor;
/// right-edge drag adjusts `trim_out_s` only; body-drag moves `start_s`.
fn audio_lane_row(
    ui: &mut egui::Ui, content_w: f32, px_per_s: f32,
    clips: &mut Vec<AudioClip>,
    selected: &mut Option<ClipRef>, dragging: &mut Option<(ClipRef, AudioDragKind)>, pending_add: &mut Option<PendingAdd>,
) -> (bool, bool) {
    let mut dirty = false;
    let mut committed = false;
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(content_w, LANE_HEIGHT), egui::Sense::click_and_drag());
    let p = ui.painter_at(rect);
    p.rect_filled(rect, 0.0, egui::Color32::from_gray(18));

    let clip_end_s = |c: &AudioClip| -> f64 {
        let max_out = if c.source_duration_s > 0.0 { c.trim_out_s.min(c.source_duration_s) } else { c.trim_out_s };
        c.start_s + (max_out - c.offset_s).max(0.0)
    };
    let clip_rects: Vec<egui::Rect> = clips.iter().map(|c| {
        let x0 = time_to_x(&rect, px_per_s, c.start_s);
        let x1 = time_to_x(&rect, px_per_s, clip_end_s(c)).max(x0 + 2.0);
        egui::Rect::from_min_max(egui::pos2(x0, rect.top() + 2.0), egui::pos2(x1, rect.bottom() - 2.0))
    }).collect();

    for (i, cr) in clip_rects.iter().enumerate() {
        let is_selected = *selected == Some(ClipRef { lane: LaneRef::Audio, index: i });
        p.rect_filled(*cr, 3.0, egui::Color32::from_rgb(120, 200, 160));
        let stroke = if is_selected { egui::Stroke::new(2.0, egui::Color32::WHITE) } else { egui::Stroke::new(1.0, egui::Color32::from_gray(60)) };
        p.rect_stroke(*cr, 3.0, stroke, egui::StrokeKind::Inside);
        let name = clips[i].file_path.file_name().and_then(|s| s.to_str()).unwrap_or("audio");
        p.text(cr.left_top() + egui::vec2(3.0, 1.0), egui::Align2::LEFT_TOP, name, egui::FontId::proportional(10.0), egui::Color32::BLACK);
    }

    if resp.drag_started() {
        if let Some(pos) = resp.interact_pointer_pos() {
            if let Some((i, hit)) = hit_test_clip_rect(&clip_rects, pos) {
                let t = x_to_time(&rect, px_per_s, pos.x);
                let clip = &clips[i];
                let kind = match hit {
                    ClipEdgeHit::Left => AudioDragKind::ResizeLeft { orig_start_s: clip.start_s, orig_offset_s: clip.offset_s },
                    ClipEdgeHit::Right => AudioDragKind::ResizeRight,
                    ClipEdgeHit::Body => AudioDragKind::Move { grab_offset_s: t - clip.start_s },
                };
                let r = ClipRef { lane: LaneRef::Audio, index: i };
                *dragging = Some((r, kind));
                *selected = Some(r);
            }
        }
    }
    if resp.dragged() {
        if let Some((r, kind)) = *dragging {
            if r.lane == LaneRef::Audio {
                if let (Some(pos), Some(clip)) = (resp.interact_pointer_pos(), clips.get_mut(r.index)) {
                    let t = x_to_time(&rect, px_per_s, pos.x);
                    match kind {
                        AudioDragKind::ResizeLeft { orig_start_s, orig_offset_s } => {
                            let max_trim_out = if clip.source_duration_s > 0.0 { clip.source_duration_s } else { clip.trim_out_s };
                            let desired_start = t.max(0.0);
                            let delta = desired_start - orig_start_s;
                            let new_offset = (orig_offset_s + delta).clamp(0.0, (max_trim_out - 0.01).max(0.0));
                            let actual_delta = new_offset - orig_offset_s;
                            clip.offset_s = new_offset;
                            clip.start_s = (orig_start_s + actual_delta).max(0.0);
                        }
                        AudioDragKind::ResizeRight => {
                            let max_trim_out = if clip.source_duration_s > 0.0 { clip.source_duration_s } else { f64::INFINITY };
                            let len = (t - clip.start_s).max(0.01);
                            clip.trim_out_s = (clip.offset_s + len).clamp(clip.offset_s + 0.01, max_trim_out);
                        }
                        AudioDragKind::Move { grab_offset_s } => {
                            clip.start_s = (t - grab_offset_s).max(0.0);
                        }
                    }
                    dirty = true;
                }
            }
        }
    }
    if resp.drag_stopped() {
        if let Some((r, _)) = *dragging {
            if r.lane == LaneRef::Audio { committed = true; }
        }
        *dragging = None;
    }
    if resp.clicked() {
        if let Some(pos) = resp.interact_pointer_pos() {
            if let Some((i, _)) = hit_test_clip_rect(&clip_rects, pos) {
                *selected = Some(ClipRef { lane: LaneRef::Audio, index: i });
            } else {
                let t = x_to_time(&rect, px_per_s, pos.x).max(0.0);
                *pending_add = Some(PendingAdd { lane: LaneRef::Audio, at_s: t });
            }
        }
    }
    (dirty, committed)
}

/// One `AnimParam` as a mode selector (Constant / Wave / Ramp, mutually
/// exclusive) plus that mode's own controls. Ramp is the timeline UI
/// corrective pass's new addition: a clip-relative "from -> to" transition
/// across the clip's own span, "Rolled" (smoothstep-eased) or "Teleport"
/// (an instant, non-eased jump at a chosen fraction of the clip).
fn anim_param_mode_editor(ui: &mut egui::Ui, label: &str, param: &mut AnimParam, id_seed: impl std::hash::Hash + std::fmt::Debug) -> bool {
    let mut changed = false;
    #[derive(Copy, Clone, PartialEq)]
    enum Mode { Constant, Wave, Ramp }
    ui.horizontal(|ui| {
        ui.label(label);
        changed |= ui.add(egui::DragValue::new(&mut param.constant).speed(0.01)).changed();
        let mut mode = if param.ramp.is_some() { Mode::Ramp } else if param.drive.is_some() { Mode::Wave } else { Mode::Constant };
        let prev_mode = mode;
        egui::ComboBox::from_id_salt(("anim_param_mode", &id_seed))
            .selected_text(match mode { Mode::Constant => "constant", Mode::Wave => "wave", Mode::Ramp => "ramp" })
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut mode, Mode::Constant, "constant");
                ui.selectable_value(&mut mode, Mode::Wave, "wave");
                ui.selectable_value(&mut mode, Mode::Ramp, "ramp");
            });
        if mode != prev_mode {
            match mode {
                Mode::Constant => { param.drive = None; param.ramp = None; }
                Mode::Wave => { param.drive = Some(Drive { shape: WaveShape::Sine, amp: 0.1, freq: 1.0, phase: 0.0 }); param.ramp = None; }
                Mode::Ramp => { param.ramp = Some(Ramp { target: param.constant, style: RampStyle::Rolled }); param.drive = None; }
            }
            changed = true;
        }
    });
    if let Some(drive) = param.drive.as_mut() {
        ui.horizontal(|ui| {
            egui::ComboBox::from_id_salt(("anim_param_wave_shape", &id_seed))
                .selected_text(format!("{:?}", drive.shape))
                .show_ui(ui, |ui| {
                    for shape in [WaveShape::Sine, WaveShape::Triangle, WaveShape::Sawtooth, WaveShape::Orbit] {
                        changed |= ui.selectable_value(&mut drive.shape, shape, format!("{shape:?}")).changed();
                    }
                });
            changed |= ui.add(egui::DragValue::new(&mut drive.amp).speed(0.01).prefix("amp ")).changed();
            changed |= ui.add(egui::DragValue::new(&mut drive.freq).speed(0.01).prefix("freq ")).changed();
            changed |= ui.add(egui::DragValue::new(&mut drive.phase).speed(0.01).prefix("phase ")).changed();
        });
    }
    if let Some(ramp) = param.ramp.as_mut() {
        ui.horizontal(|ui| {
            ui.label("target:");
            changed |= ui.add(egui::DragValue::new(&mut ramp.target).speed(0.01)).changed();
            let mut is_teleport = matches!(ramp.style, RampStyle::Teleport { .. });
            if ui.selectable_label(!is_teleport, "rolled").clicked() && is_teleport {
                ramp.style = RampStyle::Rolled;
                is_teleport = false;
                changed = true;
            }
            if ui.selectable_label(is_teleport, "teleport").clicked() && !is_teleport {
                ramp.style = RampStyle::Teleport { teleport_at: 0.5 };
                is_teleport = true;
                changed = true;
            }
            if is_teleport {
                if let RampStyle::Teleport { teleport_at } = &mut ramp.style {
                    changed |= ui.add(egui::Slider::new(teleport_at, 0.0..=1.0).text("at")).changed();
                }
            }
        });
    }
    changed
}

/// Draws (min,max) peak pairs as a waveform strip — same Painter recipe
/// `viewer.rs`'s `time_prog_plot` uses (`allocate_exact_size` ->
/// `painter_at` -> filled background -> shapes), just min/max bars instead
/// of a continuous curve.
fn draw_waveform(ui: &mut egui::Ui, peaks: &[(f32, f32)]) {
    let width = ui.available_width().max(64.0);
    let height = 40.0;
    let (rect, _resp) = ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::hover());
    let p = ui.painter_at(rect);
    p.rect_filled(rect, 2.0, egui::Color32::from_gray(20));
    if peaks.is_empty() {
        return;
    }
    let mid_y = rect.center().y;
    let half_h = rect.height() * 0.5 - 2.0;
    let n = peaks.len() as f32;
    for (i, &(lo, hi)) in peaks.iter().enumerate() {
        let x = rect.left() + (i as f32 + 0.5) / n * rect.width();
        let y0 = mid_y - hi.clamp(-1.0, 1.0) * half_h;
        let y1 = mid_y - lo.clamp(-1.0, 1.0) * half_h;
        p.line_segment([egui::pos2(x, y0), egui::pos2(x, y1)], egui::Stroke::new(1.0, egui::Color32::from_rgb(120, 200, 160)));
    }
    p.line_segment([egui::pos2(rect.left(), mid_y), egui::pos2(rect.right(), mid_y)], egui::Stroke::new(1.0, egui::Color32::from_gray(60)));
}

// ── Global (non-per-fractal) prefs ────────────────────────────────────────

fn default_colormap() -> String { "lava".to_string() }
fn default_max_iter() -> u32 { 60 }
fn default_aa() -> u32 { 1 }
fn default_render_size_cap() -> u32 { 1400 }
/// Deliberately lower than `default_max_iter`/`default_aa` — used only
/// while actively scrubbing/playing/orbit-dragging (`App::is_interacting`),
/// per the confirmed plan answer: preview may drop quality live for
/// responsiveness; motion/content/timing always match the final render,
/// and a settled/paused preview sharpens back to full quality.
fn default_preview_max_iter() -> u32 { 20 }
fn default_preview_aa() -> u32 { 1 }
fn default_undo_stack_depth() -> usize { 50 }
fn default_render_out_dir() -> String { "explorer_out/quat_mandelbrot".to_string() }
fn default_render_width() -> u32 { 1080 }
fn default_render_height() -> u32 { 1080 }
fn default_render_fps() -> u32 { 30 }
fn default_timeline_px_per_s() -> f32 { 60.0 }

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
struct AnimViewerPrefs {
    #[serde(default = "default_colormap")]
    colormap: String,
    #[serde(default = "default_max_iter")]
    max_iter: u32,
    #[serde(default = "default_aa")]
    aa: u32,
    #[serde(default = "default_render_size_cap")]
    render_size_cap: u32,
    #[serde(default = "default_preview_max_iter")]
    preview_max_iter: u32,
    #[serde(default = "default_preview_aa")]
    preview_aa: u32,
    #[serde(default = "default_undo_stack_depth")]
    undo_stack_depth: usize,
    // Render/export settings — NOT per-fractal (Carl's own spec: "settings
    // not associated with a transform or a specific fractal... are saved
    // globally"), unlike trim_start_s/trim_end_s/delta_t which live on
    // AnimationTimeline because they ARE authored, per-fractal content.
    #[serde(default = "default_render_out_dir")]
    render_out_dir: String,
    #[serde(default = "default_render_width")]
    render_width: u32,
    #[serde(default = "default_render_height")]
    render_height: u32,
    #[serde(default = "default_render_fps")]
    render_fps: u32,
    #[serde(default = "default_timeline_px_per_s")]
    timeline_px_per_s: f32,
}

impl Default for AnimViewerPrefs {
    fn default() -> Self {
        AnimViewerPrefs {
            colormap: default_colormap(), max_iter: default_max_iter(), aa: default_aa(), render_size_cap: default_render_size_cap(),
            preview_max_iter: default_preview_max_iter(), preview_aa: default_preview_aa(),
            undo_stack_depth: default_undo_stack_depth(),
            render_out_dir: default_render_out_dir(),
            render_width: default_render_width(), render_height: default_render_height(), render_fps: default_render_fps(),
            timeline_px_per_s: default_timeline_px_per_s(),
        }
    }
}

impl AnimViewerPrefs {
    fn load(path: &Path) -> Self {
        std::fs::read_to_string(path).ok().and_then(|s| toml::from_str(&s).ok()).unwrap_or_default()
    }
    fn save(&self, path: &Path) {
        if let Ok(s) = toml::to_string_pretty(self) {
            let _ = std::fs::write(path, s);
        }
    }
}

fn viewer_prefs_path() -> PathBuf {
    nnfractals::project_root().join("anim_viewer_prefs.toml")
}

// ── App ───────────────────────────────────────────────────────────────

/// How the currently-loaded fractal's `AnimationTimeline` reached memory —
/// shown in the side panel so a persistence checkpoint ("close, reopen,
/// confirm the same settings load") can be visually confirmed without
/// inspecting the JSON file directly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TimelineOrigin { LoadedFromDisk, NewlyCreated }

struct App {
    genome_path: PathBuf,
    genome_label: String,
    program: Vec<OpNode>,
    warp: Vec<OpNode>,
    julia_mode: bool,
    jc: (f32, f32),
    phoenix: (f32, f32),
    bailout_radius: f32,

    timeline: AnimationTimeline,
    timeline_origin: TimelineOrigin,

    pipeline: Option<CompiledDagPipeline>,
    gpu_error: Option<String>,

    // Camera orbit (yaw/pitch/distance) and T's base value live on
    // `self.timeline` (camera_yaw/camera_pitch/camera_distance/t_value) —
    // persisted so a batch render reproduces exactly what the interactive
    // view was showing, not a hardcoded default. Updated live every frame
    // during a drag/scroll/slider-move for rendering; `commit_timeline()`
    // (called on gesture end) is what actually saves to disk and pushes an
    // undo entry.
    colormap: String,
    render_size: u32,
    render_size_cap: u32,
    max_iter: u32,
    aa: u32,
    preview_max_iter: u32,
    preview_aa: u32,

    /// Authored-seconds position along the timeline (0..=timeline.duration_s).
    /// Session-only — NOT part of AnimationTimeline (matches how orbit
    /// yaw/pitch aren't persisted per-fractal either).
    playhead_s: f64,
    playing: bool,
    /// `None` whenever not playing (mirrors `quat_viewer.rs`'s
    /// `auto_rotate_last_tick`: prevents a paused-then-resumed playhead
    /// from applying one giant catch-up jump).
    last_frame_instant: Option<std::time::Instant>,
    /// True only while the transport scrub slider is actively being
    /// dragged this frame — part of `is_interacting()`.
    scrub_dragging: bool,

    /// Bounded stacks of full `AnimationTimeline` snapshots — chosen over
    /// a command/diff pattern because the struct is small and cheap to
    /// clone (mostly small Vecs; each `AudioClip.peaks` is an `Arc`), so a
    /// full-snapshot stack can't have an inverse-operation bug, unlike a
    /// hand-rolled command stack, for no real cost. `timeline_before_this_frame`
    /// is captured once at the top of every `ui()` call and is what a
    /// commit this frame pushes as the undo entry — correctly "the state
    /// before any of this frame's edits" regardless of which control
    /// fired the commit. Known rough edge: a long continuous drag commits
    /// every frame (not just on release), so undoing it may take a few
    /// presses rather than exactly one — acceptable, not a correctness
    /// issue (every pushed entry is still a real prior state).
    undo_stack: std::collections::VecDeque<AnimationTimeline>,
    redo_stack: std::collections::VecDeque<AnimationTimeline>,
    undo_stack_depth: usize,
    timeline_before_this_frame: AnimationTimeline,

    texture: Option<egui::TextureHandle>,
    dirty: bool,
    last_render_ms: f64,
    last_hit_frac: f32,
    dragging: bool,
    /// The camera `render()` most recently used — stashed so the
    /// orbit/pan-drag overlay (pivot marker + camera-position readout,
    /// MeshLab-style) can project into the SAME frame the user is actually
    /// looking at, without recomputing `build_frame_params` a second time
    /// just for a screen-space projection. One frame behind the live
    /// cursor during a drag, same as the rendered image itself already is.
    last_cam: Option<nnfractals::quat_raymarch::RaymarchCamera>,

    // Render/export modal — width/height/fps/out_dir are global prefs (see
    // AnimViewerPrefs); trim now lives permanently on the timeline ruler
    // (`draw_trim_bar`) — the modal only shows the resolved length.
    render_window_open: bool,
    render_out_dir: String,
    render_width: u32,
    render_height: u32,
    render_fps: u32,
    render_length_s: f64,
    render_status: Option<Result<String, String>>,

    // Audio import (reference + mux only) — a plain text field for the
    // path (no native file-dialog dependency anywhere in this codebase).
    audio_path_input: String,
    audio_import_error: Option<String>,

    // ── Timeline panel view state (session-only, never persisted/undone —
    //    same category as playhead_s) ──────────────────────────────────
    timeline_px_per_s: f32,
    selected_clip: Option<ClipRef>,
    dragging_clip: Option<(ClipRef, ClipDragKind)>,
    dragging_audio_clip: Option<(ClipRef, AudioDragKind)>,
    dragging_trim: Option<TrimHandle>,
    pending_add: Option<PendingAdd>,
    /// Sticky checkbox in the "Add to <lane>" dialog: when set, the clip
    /// being inserted spans the whole trim window instead of a fixed ~2s
    /// starting at the clicked position. Carl, 2026-09-22: "add a button
    /// that stretch the time period to the trim period" when inserting an
    /// effect/transform.
    pending_add_stretch_to_trim: bool,
    /// The timeline snapshot from the frame a lane-row drag (clip
    /// resize/move or trim-handle drag) FIRST became active — NOT the same
    /// as `timeline_before_this_frame`, which is re-captured every single
    /// frame and would already match `self.timeline` by the time a
    /// multi-frame drag's deferred commit runs, making the change
    /// invisible to `commit_timeline()`'s own comparison. Refreshed
    /// whenever no lane-row drag is active; consumed (and the undo entry
    /// pushed against IT specifically) when one concludes.
    timeline_before_drag: Option<AnimationTimeline>,

    // ── Bounding-box quick-set control ──────────────────────────────────
    bbox_dialog_open: bool,
    bbox_dialog_value: f64,
    bbox_drag_snapshot: Option<BoundingBox>,
    /// Result of the last "go to bounds" click — which axes actually had a
    /// discovered raycasting limit to jump to, and which didn't.
    bounds_status: Option<String>,

    // ── Profiles: named, genome-agnostic timeline presets ────────────────
    // (Carl's ask: save all the settings of the current fractal and reuse
    // them on another one — create new, save over, delete.) Cached list,
    // refreshed after any save/delete so the dropdown never drifts from
    // disk. `selected_profile` is session-only, same category as
    // `selected_clip`.
    profile_names: Vec<String>,
    selected_profile: Option<String>,
    profile_save_as_open: bool,
    profile_save_as_name: String,
    profile_status: Option<Result<String, String>>,

    ipc_rx: mpsc::Receiver<PathBuf>,
}

impl App {
    /// True while scrubbing the transport slider, actively playing, or
    /// orbit-dragging the camera — the preview-quality carve-out (confirmed
    /// plan answer #7) drops to `preview_max_iter`/`preview_aa` during any
    /// of these for responsiveness; a settled view always renders at full
    /// `max_iter`/`aa`.
    fn is_interacting(&self) -> bool {
        self.dragging || self.scrub_dragging || self.playing
    }

    fn undo(&mut self) {
        if let Some(prev) = self.undo_stack.pop_back() {
            self.redo_stack.push_back(std::mem::replace(&mut self.timeline, prev));
            self.dirty = true;
            self.texture = None;
            self.playhead_s = self.playhead_s.min(self.timeline.duration_s);
            self.selected_clip = None;
            let _ = nnfractals::anim_persist::save_timeline(&self.timeline);
        }
    }

    fn redo(&mut self) {
        if let Some(next) = self.redo_stack.pop_back() {
            self.undo_stack.push_back(std::mem::replace(&mut self.timeline, next));
            self.dirty = true;
            self.texture = None;
            self.playhead_s = self.playhead_s.min(self.timeline.duration_s);
            self.selected_clip = None;
            let _ = nnfractals::anim_persist::save_timeline(&self.timeline);
        }
    }
}

impl App {
    fn new(genome_path: &Path, ipc_rx: mpsc::Receiver<PathBuf>) -> anyhow::Result<Self> {
        let g = nnfractals::io::load_genome(genome_path)?;
        if g.program.is_empty() {
            anyhow::bail!("{genome_path:?} has an empty DAG program (legacy 58-basis genome?) — the animation viewer only supports DAG-based genomes, per the plan's confirmed scope");
        }
        let pipeline = CompiledDagPipeline::compile(&g.program, &g.warp);
        let gpu_error = if pipeline.is_none() { Some(gpu_error_message()) } else { None };
        let fov_deg: f64 = 45.0;
        let genome_label = genome_path.file_stem().and_then(|s| s.to_str()).unwrap_or("genome").to_string();
        let prefs = AnimViewerPrefs::load(&viewer_prefs_path());

        let content_hash = g.content_hash();
        let (mut timeline, timeline_origin) = match nnfractals::anim_persist::load_timeline(content_hash) {
            Some(tl) => (tl, TimelineOrigin::LoadedFromDisk),
            None => {
                let mut tl = AnimationTimeline::new(content_hash);
                // A box-aware initial distance beats AnimationTimeline::new's
                // generic constant — but only for a genuinely fresh timeline.
                tl.camera_distance = recommended_orbit_radius(characteristic_scale(&tl.bounds), fov_deg, 640, 640);
                // Save immediately so anim_settings/<hash>.json exists after
                // the very first launch, before the user has touched anything.
                if let Err(e) = nnfractals::anim_persist::save_timeline(&tl) {
                    eprintln!("[anim-viewer] failed to save initial timeline: {e}");
                }
                (tl, TimelineOrigin::NewlyCreated)
            }
        };
        timeline.t_value = timeline.t_value.clamp(timeline.bounds.t.min, timeline.bounds.t.max);
        timeline.clamp_pivot_offset_to_bounds();
        ensure_audio_peaks(&mut timeline);
        // Computed BEFORE `timeline,` shorthand moves it into the literal
        // below (struct-literal field-move ordering — see this project's
        // own established gotcha).
        let timeline_before_this_frame = timeline.clone();
        let render_length_s = nnfractals::anim_timeline::wallclock_span(timeline.trim_end_s - timeline.trim_start_s, timeline.delta_t);

        Ok(App {
            genome_path: genome_path.to_path_buf(),
            genome_label,
            program: g.program,
            warp: g.warp,
            julia_mode: g.julia_mode,
            jc: (g.julia_cre, g.julia_cim),
            phoenix: (g.phoenix_re, g.phoenix_im),
            bailout_radius: g.bailout_radius,
            timeline,
            timeline_origin,
            pipeline,
            gpu_error,
            colormap: prefs.colormap.clone(),
            render_size: 900,
            render_size_cap: prefs.render_size_cap,
            max_iter: prefs.max_iter,
            aa: prefs.aa,
            preview_max_iter: prefs.preview_max_iter,
            preview_aa: prefs.preview_aa,
            playhead_s: 0.0,
            playing: false,
            last_frame_instant: None,
            scrub_dragging: false,
            undo_stack: std::collections::VecDeque::new(),
            redo_stack: std::collections::VecDeque::new(),
            undo_stack_depth: prefs.undo_stack_depth,
            timeline_before_this_frame,
            texture: None,
            dirty: true,
            last_render_ms: 0.0,
            last_hit_frac: 0.0,
            dragging: false,
            last_cam: None,
            render_window_open: false,
            render_out_dir: prefs.render_out_dir.clone(),
            render_width: prefs.render_width,
            render_height: prefs.render_height,
            render_fps: prefs.render_fps,
            render_length_s,
            render_status: None,
            audio_path_input: String::new(),
            audio_import_error: None,
            timeline_px_per_s: prefs.timeline_px_per_s,
            selected_clip: None,
            dragging_clip: None,
            dragging_audio_clip: None,
            dragging_trim: None,
            pending_add: None,
            pending_add_stretch_to_trim: false,
            timeline_before_drag: None,
            bbox_dialog_open: false,
            bbox_dialog_value: 1.6,
            bbox_drag_snapshot: None,
            bounds_status: None,
            profile_names: nnfractals::anim_profile::list_profiles(),
            selected_profile: None,
            profile_save_as_open: false,
            profile_save_as_name: String::new(),
            profile_status: None,
            ipc_rx,
        })
    }

    /// Swaps in a new genome without tearing down the window — the IPC
    /// single-instance path. Mirrors `quat_viewer.rs::load_genome`'s split
    /// between what resets (genome-derived state) and what's preserved
    /// (display prefs) — plus loading/creating THIS genome's own timeline.
    fn load_genome(&mut self, genome_path: &Path) -> anyhow::Result<()> {
        let g = nnfractals::io::load_genome(genome_path)?;
        if g.program.is_empty() {
            anyhow::bail!("{genome_path:?} has an empty DAG program (legacy 58-basis genome?) — the animation viewer only supports DAG-based genomes");
        }
        let pipeline = CompiledDagPipeline::compile(&g.program, &g.warp);
        self.gpu_error = if pipeline.is_none() { Some(gpu_error_message()) } else { None };
        self.pipeline = pipeline;
        // Computed BEFORE the field-by-field moves below (content_hash()
        // reads g.program/g.warp, which those moves would otherwise take
        // away from `g` first).
        let content_hash = g.content_hash();
        let fov_deg: f64 = 45.0;
        self.genome_label = genome_path.file_stem().and_then(|s| s.to_str()).unwrap_or("genome").to_string();
        self.genome_path = genome_path.to_path_buf();
        self.program = g.program;
        self.warp = g.warp;
        self.julia_mode = g.julia_mode;
        self.jc = (g.julia_cre, g.julia_cim);
        self.phoenix = (g.phoenix_re, g.phoenix_im);
        self.bailout_radius = g.bailout_radius;

        let (mut timeline, timeline_origin) = match nnfractals::anim_persist::load_timeline(content_hash) {
            Some(tl) => (tl, TimelineOrigin::LoadedFromDisk),
            None => {
                let mut tl = AnimationTimeline::new(content_hash);
                tl.camera_distance = recommended_orbit_radius(characteristic_scale(&tl.bounds), fov_deg, 640, 640);
                if let Err(e) = nnfractals::anim_persist::save_timeline(&tl) {
                    eprintln!("[anim-viewer] failed to save initial timeline: {e}");
                }
                (tl, TimelineOrigin::NewlyCreated)
            }
        };
        timeline.t_value = timeline.t_value.clamp(timeline.bounds.t.min, timeline.bounds.t.max);
        timeline.clamp_pivot_offset_to_bounds();
        ensure_audio_peaks(&mut timeline);
        self.timeline = timeline;
        self.timeline_origin = timeline_origin;
        self.timeline_before_this_frame = self.timeline.clone();
        self.undo_stack.clear();
        self.redo_stack.clear();
        self.playhead_s = 0.0;
        self.playing = false;
        self.last_frame_instant = None;
        self.render_window_open = false;
        self.render_length_s = nnfractals::anim_timeline::wallclock_span(self.timeline.trim_end_s - self.timeline.trim_start_s, self.timeline.delta_t);
        self.render_status = None;
        self.selected_clip = None;
        self.pending_add = None;
        self.dragging_clip = None;
        self.dragging_audio_clip = None;
        self.dragging_trim = None;
        self.timeline_before_drag = None;
        self.selected_profile = None;
        self.profile_status = None;
        self.bounds_status = None;

        self.texture = None;
        self.dirty = true;
        Ok(())
    }

    fn save_prefs(&self) {
        let prefs = AnimViewerPrefs {
            colormap: self.colormap.clone(),
            max_iter: self.max_iter,
            aa: self.aa,
            render_size_cap: self.render_size_cap,
            preview_max_iter: self.preview_max_iter,
            preview_aa: self.preview_aa,
            undo_stack_depth: self.undo_stack_depth,
            render_out_dir: self.render_out_dir.clone(),
            render_width: self.render_width,
            render_height: self.render_height,
            render_fps: self.render_fps,
            timeline_px_per_s: self.timeline_px_per_s,
        };
        prefs.save(&viewer_prefs_path());
    }

    /// Persists a per-fractal timeline edit (axis assignment, bounding
    /// box, a single-frame click/drag-value edit, ...) and marks the render
    /// dirty. Called on every COMMITTED change — matches `save_prefs`'s
    /// immediate-save convention, plus an undo-stack push against the
    /// state from the TOP of this frame (correct for anything that both
    /// mutates and commits within one frame; a gesture that defers its
    /// commit across multiple frames, like a lane-row clip drag, must use
    /// `commit_timeline_from` with a snapshot captured at the gesture's
    /// own start instead — see `timeline_before_drag`).
    fn commit_timeline(&mut self) {
        let before = self.timeline_before_this_frame.clone();
        self.commit_timeline_from(before);
    }

    /// Same as `commit_timeline`, but compares/pushes against an explicitly
    /// supplied "before" state rather than `timeline_before_this_frame`.
    fn commit_timeline_from(&mut self, before: AnimationTimeline) {
        self.dirty = true;
        self.texture = None;
        if self.timeline != before {
            self.undo_stack.push_back(before);
            while self.undo_stack.len() > self.undo_stack_depth {
                self.undo_stack.pop_front();
            }
            self.redo_stack.clear();
        }
        if let Err(e) = nnfractals::anim_persist::save_timeline(&self.timeline) {
            eprintln!("[anim-viewer] failed to save timeline: {e}");
        }
    }

    /// Decodes and appends `self.audio_path_input` as a new positioned clip
    /// on the audio lane — reference + mux only (confirmed plan scope), so
    /// this is the only place audio content is ever read besides
    /// `ensure_audio_peaks` re-decoding it after a reopen.
    fn import_audio_at(&mut self, start_s: f64) {
        let trimmed = self.audio_path_input.trim();
        if trimmed.is_empty() {
            self.audio_import_error = Some("enter a file path first".to_string());
            return;
        }
        let path = PathBuf::from(trimmed);
        if !path.exists() {
            self.audio_import_error = Some(format!("file not found: {}", path.display()));
            return;
        }
        let source_duration_s = nnfractals::video_export::probe_audio_duration_s(&path).unwrap_or(0.0);
        match nnfractals::video_export::decode_audio_peaks(&path, 400) {
            Ok(peaks) => {
                let trim_out_s = if source_duration_s > 0.0 { source_duration_s } else { f64::INFINITY };
                self.timeline.audio_clips.push(AudioClip {
                    file_path: path, start_s, offset_s: 0.0, trim_out_s, source_duration_s,
                    gain_db: 0.0, peaks: Some(std::sync::Arc::new(peaks)),
                });
                self.selected_clip = Some(ClipRef { lane: LaneRef::Audio, index: self.timeline.audio_clips.len() - 1 });
                self.audio_path_input.clear();
                self.audio_import_error = None;
                self.commit_timeline();
            }
            Err(e) => self.audio_import_error = Some(format!("couldn't decode audio: {e}")),
        }
    }

    fn push_effect_clip(&mut self, lane: LaneRef, effect: Effect, start_s: f64, end_s: f64) {
        let clip = EffectClip { effect, start_s, end_s, label: String::new() };
        let index = match lane {
            LaneRef::X => { self.timeline.x_track.clips.push(clip); self.timeline.x_track.clips.len() - 1 }
            LaneRef::Y => { self.timeline.y_track.clips.push(clip); self.timeline.y_track.clips.len() - 1 }
            LaneRef::Z => { self.timeline.z_track.clips.push(clip); self.timeline.z_track.clips.len() - 1 }
            LaneRef::T => { self.timeline.t_track.clips.push(clip); self.timeline.t_track.clips.len() - 1 }
            LaneRef::Effects(i) => {
                let lane_tl = self.timeline.effects_lanes.get_mut(i).expect("pending_add references a live effects lane");
                lane_tl.clips.push(clip);
                lane_tl.clips.len() - 1
            }
            LaneRef::Audio => unreachable!("push_effect_clip is never called for the audio lane"),
        };
        self.selected_clip = Some(ClipRef { lane, index });
        self.commit_timeline();
    }

    /// The (start_s, end_s) of whichever clip is presently selected (shown
    /// in the Clip Inspector right now) — `None` if nothing is selected,
    /// or the selection is the audio lane (audio clips don't carry an
    /// `end_s` the same way — see `show_pending_add_window`'s doc comment).
    fn selected_clip_span(&self) -> Option<(f64, f64)> {
        let r = self.selected_clip?;
        match r.lane {
            LaneRef::X => self.timeline.x_track.clips.get(r.index),
            LaneRef::Y => self.timeline.y_track.clips.get(r.index),
            LaneRef::Z => self.timeline.z_track.clips.get(r.index),
            LaneRef::T => self.timeline.t_track.clips.get(r.index),
            LaneRef::Effects(i) => self.timeline.effects_lanes.get(i)?.clips.get(r.index),
            LaneRef::Audio => None,
        }.map(|c| (c.start_s, c.end_s))
    }

    /// The small floating chooser opened by clicking empty lane space —
    /// filtered per lane type (Rotation/Translation/Scale on X/Y/Z,
    /// Translation/Scale only on T; a single "add difference" on an
    /// effects lane; a file-path + import flow on the audio lane).
    fn show_pending_add_window(&mut self, ctx: &egui::Context) {
        let Some(pa) = self.pending_add else { return };
        let duration_s = self.timeline.duration_s;
        let default_len = 2.0f64.min(duration_s.max(0.1));
        let trim_start_s = self.timeline.trim_start_s;
        let trim_end_s = self.timeline.trim_end_s.max(trim_start_s + 0.01);
        // Where the new clip lands does NOT depend on where in the lane the
        // user happened to click — Carl, 2026-09-23: "Ignore where the user
        // click on the lane for setting start value. Use only current clip
        // value, the user can always adjust it with the lane clip sides."
        // "stretch to trim period" (explicit opt-in, 2026-09-22) wins when
        // checked; otherwise default to the SAME span as whichever clip is
        // currently selected (so a new effect naturally lines up with the
        // one you're already working on), falling back to a fixed
        // [0, default_len) window only when nothing is selected at all —
        // never the click position.
        let (start_s, end_s) = if self.pending_add_stretch_to_trim {
            (trim_start_s, trim_end_s)
        } else if let Some((s, e)) = self.selected_clip_span() {
            (s, e)
        } else {
            (0.0, default_len.max(0.01))
        };
        let mut open = true;
        let mut close = false;
        egui::Window::new(format!("Add to {} @ {:.2}s", lane_label(pa.lane), pa.at_s))
            .collapsible(false)
            .open(&mut open)
            .show(ctx, |ui| {
                if pa.lane != LaneRef::Audio {
                    ui.checkbox(&mut self.pending_add_stretch_to_trim,
                        format!("stretch to trim period ({trim_start_s:.2}s\u{2013}{trim_end_s:.2}s)"));
                }
                match pa.lane {
                    LaneRef::X | LaneRef::Y | LaneRef::Z | LaneRef::T => {
                        if ui.button("+ translation").clicked() {
                            self.push_effect_clip(pa.lane, Effect::Translation { offset: AnimParam::constant(0.0) }, start_s, end_s);
                            close = true;
                        }
                        if ui.button("+ scale").clicked() {
                            self.push_effect_clip(pa.lane, Effect::Scale { factor: AnimParam::constant(1.0) }, start_s, end_s);
                            close = true;
                        }
                        if pa.lane != LaneRef::T && ui.button("+ rotation").clicked() {
                            self.push_effect_clip(pa.lane, Effect::Rotation { degrees_per_second: AnimParam::constant(30.0) }, start_s, end_s);
                            close = true;
                        }
                    }
                    LaneRef::Effects(_) => {
                        if ui.button("+ difference (cutaway)").clicked() {
                            let b = &self.timeline.bounds;
                            let default_pos = ((b.x.min + b.x.max) * 0.5, (b.y.min + b.y.max) * 0.5, (b.z.min + b.z.max) * 0.5);
                            self.push_effect_clip(pa.lane, Effect::Difference {
                                pos: [AnimParam::constant(default_pos.0), AnimParam::constant(default_pos.1), AnimParam::constant(default_pos.2)],
                                normal: [AnimParam::constant(1.0), AnimParam::constant(0.0), AnimParam::constant(0.0)],
                            }, start_s, end_s);
                            close = true;
                        }
                        if ui.button("+ fixed difference (screen-locked cutaway)").clicked() {
                            // Anchored partway between the camera and the
                            // pivot by default (half the current orbit
                            // distance) so it starts as a visible partial
                            // cut, not degenerate all-or-nothing — see
                            // `Effect::FixedDifference`'s own doc comment
                            // for the camera-local (right,up,forward) axes.
                            let half_dist = self.timeline.camera_distance * 0.5;
                            self.push_effect_clip(pa.lane, Effect::FixedDifference {
                                camera_offset: [AnimParam::constant(0.0), AnimParam::constant(0.0), AnimParam::constant(half_dist)],
                                normal: [AnimParam::constant(0.0), AnimParam::constant(0.0), AnimParam::constant(1.0)],
                            }, start_s, end_s);
                            close = true;
                        }
                        if ui.button("+ slide iteration (iteration cutaway)").clicked() {
                            // min==max is a no-op (never gates anything) —
                            // a freshly-added clip shouldn't drastically
                            // change the render until the band is opened.
                            self.push_effect_clip(pa.lane, Effect::SlideIteration {
                                min_iter: AnimParam::constant(0.0),
                                max_iter: AnimParam::constant(0.0),
                            }, start_s, end_s);
                            close = true;
                        }
                    }
                    LaneRef::Audio => {
                        ui.horizontal(|ui| {
                            ui.label("file:");
                            ui.text_edit_singleline(&mut self.audio_path_input);
                        });
                        if let Some(err) = &self.audio_import_error {
                            ui.colored_label(egui::Color32::RED, err);
                        }
                        if ui.button("import").clicked() {
                            self.import_audio_at(pa.at_s);
                            close = true;
                        }
                    }
                }
                if ui.button("cancel").clicked() {
                    close = true;
                }
            });
        if !open || close {
            self.pending_add = None;
        }
    }

    /// Selecting a clip in the timeline below opens this: the clip's full
    /// edit form (kind, span, value/wave/ramp, delete) — the one thing a
    /// drag gesture alone can't express (picking a wave shape, a ramp
    /// target, ...), so some form-based surface here is unavoidable even
    /// with the rest of the timeline now being a visual, draggable strip.
    fn show_clip_inspector(&mut self, ui: &mut egui::Ui) {
        let Some(r) = self.selected_clip else {
            ui.label("(nothing selected)");
            ui.label("Click a clip in the timeline below to edit it, or click empty space in a lane to add one.");
            return;
        };
        let duration_s = self.timeline.duration_s;
        let b = &self.timeline.bounds;
        let default_pos = ((b.x.min + b.x.max) * 0.5, (b.y.min + b.y.max) * 0.5, (b.z.min + b.z.max) * 0.5);
        let mut changed = false;
        let mut deleted = false;

        match r.lane {
            LaneRef::X | LaneRef::Y | LaneRef::Z | LaneRef::T => {
                let track = match r.lane {
                    LaneRef::X => &mut self.timeline.x_track,
                    LaneRef::Y => &mut self.timeline.y_track,
                    LaneRef::Z => &mut self.timeline.z_track,
                    LaneRef::T => &mut self.timeline.t_track,
                    _ => unreachable!(),
                };
                let Some(clip) = track.clips.get_mut(r.index) else { self.selected_clip = None; return; };
                ui.label(format!("{} — {}", lane_label(r.lane), effect_kind_label(&clip.effect)));
                ui.horizontal(|ui| {
                    changed |= ui.add(egui::DragValue::new(&mut clip.start_s).speed(0.05).range(0.0..=(clip.end_s - 0.01).max(0.0)).prefix("start ").suffix("s")).changed();
                    changed |= ui.add(egui::DragValue::new(&mut clip.end_s).speed(0.05).range((clip.start_s + 0.01)..=duration_s).prefix("end ").suffix("s")).changed();
                });
                match &mut clip.effect {
                    Effect::Rotation { degrees_per_second } => {
                        ui.horizontal(|ui| {
                            ui.label("deg/s:");
                            changed |= ui.add(egui::DragValue::new(&mut degrees_per_second.constant).speed(0.1)).changed();
                        });
                    }
                    Effect::Translation { offset } => { changed |= anim_param_mode_editor(ui, "offset", offset, (r, "translation")); }
                    Effect::Scale { factor } => { changed |= anim_param_mode_editor(ui, "factor", factor, (r, "scale")); }
                    Effect::Difference { .. } => { ui.colored_label(egui::Color32::RED, "internal error: Difference on an axis track"); }
                    Effect::FixedDifference { .. } => { ui.colored_label(egui::Color32::RED, "internal error: FixedDifference on an axis track"); }
                    Effect::SlideIteration { .. } => { ui.colored_label(egui::Color32::RED, "internal error: SlideIteration on an axis track"); }
                }
                if ui.button("delete clip").clicked() { deleted = true; }
            }
            LaneRef::Effects(lane_i) => {
                let Some(lane) = self.timeline.effects_lanes.get_mut(lane_i) else { self.selected_clip = None; return; };
                let Some(clip) = lane.clips.get_mut(r.index) else { self.selected_clip = None; return; };
                ui.label(format!("{} — {}", lane_label(r.lane), effect_kind_label(&clip.effect)));
                ui.horizontal(|ui| {
                    changed |= ui.add(egui::DragValue::new(&mut clip.start_s).speed(0.05).range(0.0..=(clip.end_s - 0.01).max(0.0)).prefix("start ").suffix("s")).changed();
                    changed |= ui.add(egui::DragValue::new(&mut clip.end_s).speed(0.05).range((clip.start_s + 0.01)..=duration_s).prefix("end ").suffix("s")).changed();
                });
                if let Effect::Difference { pos, normal } = &mut clip.effect {
                    ui.label("position:");
                    changed |= anim_param_mode_editor(ui, "x", &mut pos[0], (r, "pos_x"));
                    changed |= anim_param_mode_editor(ui, "y", &mut pos[1], (r, "pos_y"));
                    changed |= anim_param_mode_editor(ui, "z", &mut pos[2], (r, "pos_z"));
                    ui.label("normal (which side is removed):");
                    changed |= anim_param_mode_editor(ui, "x", &mut normal[0], (r, "norm_x"));
                    changed |= anim_param_mode_editor(ui, "y", &mut normal[1], (r, "norm_y"));
                    changed |= anim_param_mode_editor(ui, "z", &mut normal[2], (r, "norm_z"));
                    if ui.button("reset to axis-aligned half-cut").clicked() {
                        *pos = [AnimParam::constant(default_pos.0), AnimParam::constant(default_pos.1), AnimParam::constant(default_pos.2)];
                        *normal = [AnimParam::constant(1.0), AnimParam::constant(0.0), AnimParam::constant(0.0)];
                        changed = true;
                    }
                }
                if let Effect::FixedDifference { camera_offset, normal } = &mut clip.effect {
                    ui.label("camera-relative offset (right, up, forward):");
                    changed |= anim_param_mode_editor(ui, "right", &mut camera_offset[0], (r, "cam_off_r"));
                    changed |= anim_param_mode_editor(ui, "up", &mut camera_offset[1], (r, "cam_off_u"));
                    changed |= anim_param_mode_editor(ui, "fwd", &mut camera_offset[2], (r, "cam_off_f"));
                    ui.label("normal, camera-relative (which side is removed):");
                    changed |= anim_param_mode_editor(ui, "right", &mut normal[0], (r, "cam_norm_r"));
                    changed |= anim_param_mode_editor(ui, "up", &mut normal[1], (r, "cam_norm_u"));
                    changed |= anim_param_mode_editor(ui, "fwd", &mut normal[2], (r, "cam_norm_f"));
                    ui.label("Stays fixed on screen as the camera orbits — doesn't sweep across the frame like a plain difference does.");
                    if ui.button("reset (half-distance in front of the camera)").clicked() {
                        let half_dist = self.timeline.camera_distance * 0.5;
                        *camera_offset = [AnimParam::constant(0.0), AnimParam::constant(0.0), AnimParam::constant(half_dist)];
                        *normal = [AnimParam::constant(0.0), AnimParam::constant(0.0), AnimParam::constant(1.0)];
                        changed = true;
                    }
                }
                if let Effect::SlideIteration { min_iter, max_iter } = &mut clip.effect {
                    ui.label("hidden escape-iteration band (surfaces in this range become see-through, letting the ray continue further into the object):");
                    changed |= anim_param_mode_editor(ui, "min", min_iter, (r, "slide_min"));
                    changed |= anim_param_mode_editor(ui, "max", max_iter, (r, "slide_max"));
                }
                if ui.button("delete clip").clicked() { deleted = true; }
            }
            LaneRef::Audio => {
                let Some(clip) = self.timeline.audio_clips.get_mut(r.index) else { self.selected_clip = None; return; };
                let name = clip.file_path.file_name().and_then(|s| s.to_str()).unwrap_or("?").to_string();
                ui.label(format!("audio — {name}"));
                changed |= ui.add(egui::DragValue::new(&mut clip.start_s).speed(0.05).range(0.0..=f64::INFINITY).prefix("start ").suffix("s")).changed();
                let max_trim_out = if clip.source_duration_s > 0.0 { clip.source_duration_s } else { f64::INFINITY };
                ui.horizontal(|ui| {
                    changed |= ui.add(egui::DragValue::new(&mut clip.offset_s).speed(0.05).range(0.0..=(clip.trim_out_s - 0.01).max(0.0)).prefix("trim-in ").suffix("s")).changed();
                    changed |= ui.add(egui::DragValue::new(&mut clip.trim_out_s).speed(0.05).range((clip.offset_s + 0.01)..=max_trim_out).prefix("trim-out ").suffix("s")).changed();
                });
                changed |= ui.add(egui::DragValue::new(&mut clip.gain_db).speed(0.5).suffix("dB")).changed();
                if let Some(peaks) = &clip.peaks {
                    draw_waveform(ui, peaks);
                }
                if ui.button("remove clip").clicked() { deleted = true; }
            }
        }

        if deleted {
            match r.lane {
                LaneRef::X => { self.timeline.x_track.clips.remove(r.index); }
                LaneRef::Y => { self.timeline.y_track.clips.remove(r.index); }
                LaneRef::Z => { self.timeline.z_track.clips.remove(r.index); }
                LaneRef::T => { self.timeline.t_track.clips.remove(r.index); }
                LaneRef::Effects(i) => { if let Some(lane) = self.timeline.effects_lanes.get_mut(i) { lane.clips.remove(r.index); } }
                LaneRef::Audio => { self.timeline.audio_clips.remove(r.index); }
            }
            self.selected_clip = None;
            self.commit_timeline();
        } else if changed {
            self.commit_timeline();
        }
    }

    /// Sets ONE viewer axis's bounds (or, if `axis` is `None`, all four —
    /// X, Y, Z, AND T) to the actual discovered raycasting limit for
    /// whichever quaternion part `axis_assignment` currently maps it to —
    /// Carl, 2026-09-23: "add a go to bounds option in the animator. For
    /// all axis or for one only." Also reframes the camera distance (and
    /// clears any manual pan) to fit the new bounds — "please move the
    /// camera so I can see th fractal as a whole."
    ///
    /// Computed LIVE against the currently loaded genome, not read from
    /// the persisted `quat_bound_*` fields `quat-bounded-scan` writes —
    /// those can be stale (genome edited since the last batch scan) or
    /// simply absent (a genome outside any scanned `fractals_dag_quat*`
    /// folder, or scanned before this metric existed). The scan itself is
    /// cheap enough that this is a non-issue: confirmed scanning the
    /// entire ~2500-genome archive takes 1.6s total, well under 1ms/genome,
    /// so recomputing once per click is imperceptible.
    ///
    /// An axis whose quaternion part has no discovered limit in one or
    /// both directions (unbounded, or bounded past `SEARCH_MAX` — see
    /// `quat_boundedness`'s own doc comment) is left untouched rather than
    /// guessed at; `bounds_status` reports which axes were actually
    /// applied so that's never silent.
    fn go_to_bounds(&mut self, axis: Option<ViewerAxis>) {
        let formula = QuatDagFormula { prog: &self.program, warp: &self.warp, julia: self.julia_mode, jc: self.jc, phoenix: self.phoenix };
        let bailout_sq = (self.bailout_radius as f64) * (self.bailout_radius as f64);
        let report = nnfractals::quat_boundedness::compute_boundedness(&formula, 200, bailout_sq);
        let limits_for = |part: QuatPart| match part {
            QuatPart::R => report.r,
            QuatPart::A => report.a,
            QuatPart::B => report.b,
            QuatPart::C => report.c,
        };
        let targets = match axis {
            Some(a) => vec![a],
            None => vec![ViewerAxis::X, ViewerAxis::Y, ViewerAxis::Z, ViewerAxis::T],
        };
        let mut applied = Vec::new();
        let mut skipped = Vec::new();
        for va in targets {
            let part = self.timeline.axis_assignment.part_of(va);
            let lim = limits_for(part);
            match (lim.neg, lim.pos) {
                (Some(neg), Some(pos)) => {
                    let bound = match va {
                        ViewerAxis::X => &mut self.timeline.bounds.x,
                        ViewerAxis::Y => &mut self.timeline.bounds.y,
                        ViewerAxis::Z => &mut self.timeline.bounds.z,
                        ViewerAxis::T => &mut self.timeline.bounds.t,
                    };
                    bound.min = -neg;
                    bound.max = pos;
                    applied.push(va.label());
                }
                _ => skipped.push(va.label()),
            }
        }
        self.bounds_status = Some(match (applied.is_empty(), skipped.is_empty()) {
            (true, _) => format!("no discovered limit for {} \u{2014} left unchanged", skipped.join(", ")),
            (false, true) => format!("went to bounds on {}", applied.join(", ")),
            (false, false) => format!("went to bounds on {} \u{2014} {} has no discovered limit, left unchanged", applied.join(", "), skipped.join(", ")),
        });
        if !applied.is_empty() {
            self.timeline.t_value = self.timeline.t_value.clamp(self.timeline.bounds.t.min, self.timeline.bounds.t.max);
            // Carl, 2026-09-23: "when I go to bound, please move the
            // camera so I can see th fractal as a whole" — the bounds can
            // grow (or shrink) by orders of magnitude relative to
            // whatever the camera was framed for before, so jumping the
            // box alone can leave the object a speck in the middle of an
            // empty frame or badly clipped. Reuses "reset view"'s own
            // framing formula exactly (`characteristic_scale` + FOV-aware
            // `recommended_orbit_radius`), recomputed against the NEW
            // bounds — same distance a fresh "reset view" click would
            // land on for this box. Yaw/pitch are left alone (the chosen
            // viewing ANGLE isn't what "see it as a whole" is about), but
            // the pan offset resets to zero — a manual pan tuned for the
            // OLD box's size/shape would otherwise point the recentered
            // camera at empty space next to the new one.
            self.reframe_camera_to_bounds();
            self.commit_timeline();
        }
    }

    /// Recomputes `camera_distance` from the CURRENT bounds and resets the
    /// manual pan (`camera_pivot_offset`) to zero — the same framing
    /// "reset view" has always used, factored out so every control that can
    /// change the bounds' overall scale reframes the camera the same way,
    /// not just "reset view" and "go to bounds" — Carl, 2026-09-23: "it
    /// seem better but still does not work". Root cause: bounds changed via
    /// "scale box" (drag-rescale), the ± dialog, "reset to ±1.6", or a
    /// direct min/max edit never touched `camera_distance` at all, so a
    /// bounds change of orders of magnitude (exactly what these controls
    /// are FOR) could leave the camera sitting at a now-wildly-wrong
    /// distance relative to the new box — e.g. `camera_distance` stuck at a
    /// small leftover value against a box that just grew to ±7850, putting
    /// the camera essentially inside/against the structure with nothing
    /// recognizable in frame. Only the box's overall SCALE drives this
    /// (`characteristic_scale`), so calling it after every bounds commit is
    /// safe even for a small nudge — yaw/pitch (the chosen viewing angle)
    /// are deliberately left alone.
    fn reframe_camera_to_bounds(&mut self) {
        self.timeline.camera_distance = recommended_orbit_radius(characteristic_scale(&self.timeline.bounds), 45.0, self.render_size, self.render_size);
        self.timeline.camera_pivot_offset = (0.0, 0.0, 0.0);
    }

    /// Bounding-box quick-set: plain click opens a dialog to set X/Y/Z
    /// bounds directly to a chosen ±value; click-and-drag live-rescales the
    /// CURRENT bounds by a log/exponential factor of the drag distance
    /// (~150px per doubling, so small nudges are fine-grained and long
    /// drags cover a wide range fast), shown near the cursor while
    /// dragging, applied once on release. Never touches T.
    fn show_bbox_quick_set(&mut self, ui: &mut egui::Ui) {
        let resp = ui.add(egui::Button::new("\u{29c9} scale box").sense(egui::Sense::click_and_drag()));
        if resp.drag_started() {
            self.bbox_drag_snapshot = Some(self.timeline.bounds);
        }
        if resp.dragged() {
            if let (Some(snapshot), Some(total)) = (self.bbox_drag_snapshot, resp.total_drag_delta()) {
                const PX_PER_DOUBLING: f32 = 150.0;
                let factor = 2.0f64.powf((total.x / PX_PER_DOUBLING) as f64);
                let mut b = snapshot;
                b.scale_xyz_around_center(factor);
                self.timeline.bounds = b;
                self.timeline.t_value = self.timeline.t_value.clamp(self.timeline.bounds.t.min, self.timeline.bounds.t.max);
                self.dirty = true;
                if let Some(pos) = ui.ctx().pointer_latest_pos() {
                    ui.ctx().debug_painter().text(
                        pos + egui::vec2(14.0, 0.0), egui::Align2::LEFT_CENTER,
                        format!("\u{d7}{factor:.2}"), egui::FontId::proportional(14.0), egui::Color32::YELLOW,
                    );
                }
            }
        }
        if resp.drag_stopped() {
            self.bbox_drag_snapshot = None;
            self.reframe_camera_to_bounds();
            self.commit_timeline();
        }
        if resp.clicked() {
            self.bbox_dialog_value = (self.timeline.bounds.x.max - self.timeline.bounds.x.min).abs() / 2.0;
            self.bbox_dialog_open = true;
        }
    }

    fn show_bbox_dialog(&mut self, ctx: &egui::Context) {
        let mut open = self.bbox_dialog_open;
        egui::Window::new("Set bounds").collapsible(false).open(&mut open).show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("\u{b1}");
                ui.add(egui::DragValue::new(&mut self.bbox_dialog_value).speed(0.05).range(0.001..=f64::INFINITY));
                ui.label("on X, Y, Z (T unchanged)");
            });
            if ui.button("apply").clicked() {
                self.timeline.bounds.set_xyz_symmetric(self.bbox_dialog_value);
                self.timeline.t_value = self.timeline.t_value.clamp(self.timeline.bounds.t.min, self.timeline.bounds.t.max);
                self.reframe_camera_to_bounds();
                self.commit_timeline();
                self.bbox_dialog_open = false;
            }
        });
        if !open {
            self.bbox_dialog_open = false;
        }
    }

    /// Replaces `self.timeline` with the named profile's saved settings —
    /// everything except which genome it belongs to, which must stay the
    /// CURRENTLY LOADED genome's own hash (a profile's own stored hash is
    /// always 0, see `anim_profile::save_profile` — applying it verbatim
    /// would silently corrupt a different genome's persisted settings the
    /// next time anything saves, since `genome_content_hash` is the on-disk
    /// key `anim_persist::save_timeline` writes under). Routed through
    /// `commit_timeline()` like any other edit, so applying a profile is
    /// itself undoable.
    fn apply_selected_profile(&mut self) {
        let Some(name) = self.selected_profile.clone() else { return; };
        let Some(profile) = nnfractals::anim_profile::load_profile(&name) else {
            self.profile_status = Some(Err(format!("profile {name:?} could not be loaded")));
            return;
        };
        let mut tl = profile.timeline;
        tl.genome_content_hash = self.timeline.genome_content_hash;
        tl.t_value = tl.t_value.clamp(tl.bounds.t.min, tl.bounds.t.max);
        self.timeline = tl;
        ensure_audio_peaks(&mut self.timeline);
        self.selected_clip = None;
        self.playhead_s = self.playhead_s.min(self.timeline.duration_s);
        self.commit_timeline();
        self.profile_status = Some(Ok(format!("applied {name:?}")));
    }

    /// Saves the CURRENT timeline over the currently-selected profile —
    /// "save over an existing profile." Distinct from
    /// `save_as_new_profile`, which prompts for a (possibly new) name.
    fn save_over_selected_profile(&mut self) {
        let Some(name) = self.selected_profile.clone() else { return; };
        match nnfractals::anim_profile::save_profile(&name, &self.timeline) {
            Ok(()) => {
                self.profile_names = nnfractals::anim_profile::list_profiles();
                self.profile_status = Some(Ok(format!("saved over {name:?}")));
            }
            Err(e) => self.profile_status = Some(Err(format!("couldn't save {name:?}: {e}"))),
        }
    }

    fn delete_selected_profile(&mut self) {
        let Some(name) = self.selected_profile.take() else { return; };
        match nnfractals::anim_profile::delete_profile(&name) {
            Ok(()) => {
                self.profile_names = nnfractals::anim_profile::list_profiles();
                self.profile_status = Some(Ok(format!("deleted {name:?}")));
            }
            Err(e) => self.profile_status = Some(Err(format!("couldn't delete {name:?}: {e}"))),
        }
    }

    /// The "Save as new…" dialog — also how you overwrite an EXISTING
    /// profile by (re)typing its exact name, since `save_profile` itself
    /// always overwrites; "new" vs. "over" is purely a matter of whether
    /// the typed name happens to already exist, reflected here only in the
    /// warning label, never a hard block.
    fn show_profile_save_as_dialog(&mut self, ctx: &egui::Context) {
        let mut open = self.profile_save_as_open;
        let mut close = false;
        egui::Window::new("Save profile as").collapsible(false).open(&mut open).show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("name:");
                ui.text_edit_singleline(&mut self.profile_save_as_name);
            });
            let trimmed = self.profile_save_as_name.trim().to_string();
            if !trimmed.is_empty() && self.profile_names.iter().any(|n| n == &trimmed) {
                ui.colored_label(egui::Color32::from_rgb(210, 180, 90), format!("{trimmed:?} already exists — saving will overwrite it"));
            }
            ui.horizontal(|ui| {
                if ui.add_enabled(!trimmed.is_empty(), egui::Button::new("save")).clicked() {
                    match nnfractals::anim_profile::save_profile(&trimmed, &self.timeline) {
                        Ok(()) => {
                            self.profile_names = nnfractals::anim_profile::list_profiles();
                            self.selected_profile = Some(trimmed.clone());
                            self.profile_status = Some(Ok(format!("saved {trimmed:?}")));
                            close = true;
                        }
                        Err(e) => self.profile_status = Some(Err(format!("couldn't save {trimmed:?}: {e}"))),
                    }
                }
                if ui.button("cancel").clicked() {
                    close = true;
                }
            });
        });
        if !open || close {
            self.profile_save_as_open = false;
        }
    }

    /// Spawns `nnfractals-queue` (fire-and-forget, reaped on a detached
    /// thread) so a queued render can be watched/managed without leaving
    /// this window. Ported from `quat_viewer.rs::open_queue_manager` — Carl
    /// asked (back on `quat_viewer.rs`) that adding a queue item actually
    /// bring up the queue manager rather than requiring a second click;
    /// that call was dropped when `anim_viewer.rs` replaced it as the
    /// entry point, which is what made "Add to queue" look like it "did
    /// nothing." Restored here, called automatically after a successful
    /// enqueue, plus kept as its own button for reopening later.
    fn open_queue_manager(&mut self) {
        let bin = nnfractals::locate_bin("nnfractals-queue");
        match std::process::Command::new(&bin).spawn() {
            Ok(mut child) => {
                thread::spawn(move || {
                    let _ = child.wait();
                });
            }
            Err(e) => {
                self.render_status = Some(Err(format!(
                    "couldn't launch {} to open the queue manager: {e} — is it built? (cargo build --release --bin nnfractals-queue --features queue)",
                    bin.display()
                )));
            }
        }
    }

    /// The resolution/fps/length render-export modal. Trim now lives
    /// permanently on the timeline ruler (`draw_trim_bar`) — this window
    /// only shows the resolved length (read-only) plus a convenience
    /// "length" field that back-solves the trim END (trim START and
    /// delta_t stay fixed), so length/trim/fps never disagree.
    fn show_render_window(&mut self, ctx: &egui::Context) {
        let mut open = self.render_window_open;
        egui::Window::new("Render animation").open(&mut open).show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("resolution:");
                ui.add(egui::DragValue::new(&mut self.render_width).range(64..=7680));
                ui.label("x");
                ui.add(egui::DragValue::new(&mut self.render_height).range(64..=7680));
            });
            ui.horizontal(|ui| {
                ui.label("fps:");
                ui.add(egui::DragValue::new(&mut self.render_fps).range(1..=240));
            });
            ui.label(format!(
                "trim: {:.2}s to {:.2}s (drag the handles on the timeline ruler to change)",
                self.timeline.trim_start_s, self.timeline.trim_end_s
            ));
            ui.horizontal(|ui| {
                ui.label("length:");
                if ui.add(egui::DragValue::new(&mut self.render_length_s).speed(0.1).range(0.01..=f64::INFINITY).suffix("s")).changed() {
                    // Back-solve the trim END so length/trim/delta_t never
                    // disagree — trim START and delta_t stay fixed.
                    let authored_span = self.render_length_s * self.timeline.delta_t;
                    self.timeline.trim_end_s = (self.timeline.trim_start_s + authored_span).min(self.timeline.duration_s);
                    self.commit_timeline();
                }
                let frame_count = ((self.render_length_s * self.render_fps as f64).round() as u32).max(2);
                ui.label(format!("({frame_count} frames)"));
            });
            ui.horizontal(|ui| {
                ui.label("output dir:");
                if ui.text_edit_singleline(&mut self.render_out_dir).changed() {
                    self.save_prefs();
                }
            });
            ui.horizontal(|ui| {
                if ui.button("Add to queue").clicked() {
                    let quat_spec = nnfractals::video_export::QuatRenderSpec {
                        mode: "timeline".to_string(),
                        width: self.render_width,
                        height: self.render_height,
                        fps: self.render_fps,
                        max_iter: self.max_iter,
                        aa: self.aa.max(2),
                        colormap: self.colormap.clone(),
                        timeline: Some(self.timeline.clone()),
                        trim_start_s: self.timeline.trim_start_s,
                        trim_end_s: self.timeline.trim_end_s,
                        ..Default::default()
                    };
                    let spec = nnfractals::video_export::QueueSpec {
                        nn_src: self.genome_path.clone(),
                        genome_label: self.genome_label.clone(),
                        output_dir: self.render_out_dir.clone(),
                        quat_spec: Some(quat_spec),
                        ..Default::default()
                    };
                    match nnfractals::video_export::enqueue(spec) {
                        Ok(item) => {
                            self.render_status = Some(Ok(item.id));
                            self.open_queue_manager();
                        }
                        Err(e) => self.render_status = Some(Err(format!("couldn't add to queue: {e}"))),
                    }
                    self.save_prefs();
                }
                if ui.button("Open queue manager\u{2026}").clicked() {
                    self.open_queue_manager();
                }
            });
            match &self.render_status {
                Some(Ok(id)) => { ui.colored_label(egui::Color32::from_rgb(120, 200, 120), format!("queued (id {id})")); }
                Some(Err(e)) => { ui.colored_label(egui::Color32::RED, e); }
                None => {}
            }
        });
        self.render_window_open = open;
    }

    /// The single function both the live preview and the batch CLI renderer
    /// call to go from timeline state to rendered pixels
    /// (`anim_eval::build_frame_params`) — never constructed by hand
    /// anywhere else, so preview and render can't silently drift apart.
    fn render(&mut self, ctx: &egui::Context) {
        // Preview-quality carve-out (confirmed plan answer #7): drop
        // quality while actively interacting, full quality once settled.
        let (max_iter, aa) = if self.is_interacting() { (self.preview_max_iter, self.preview_aa) } else { (self.max_iter, self.aa) };
        let Some(pipeline) = self.pipeline.as_mut() else { return };
        let formula = QuatDagFormula { prog: &self.program, warp: &self.warp, julia: self.julia_mode, jc: self.jc, phoenix: self.phoenix };
        let (params, cam) = nnfractals::anim_eval::build_frame_params(
            &self.timeline, formula, self.playhead_s, self.bailout_radius as f64,
            nnfractals::anim_eval::RenderQuality { max_iter, aa },
        );
        self.last_cam = Some(cam);
        let start = std::time::Instant::now();
        let (mut shading, mut color_t) = pipeline.render(&params, &cam, self.render_size, self.render_size);
        // "hide top N%": the cutoff can only be known AFTER seeing this
        // frame's own escape-time distribution, so this is a genuine
        // second render — see `percentile_escape_time_cutoff`'s doc
        // comment for why a fixed absolute value can't do this instead.
        if self.timeline.hide_top_percentile > 0.0 {
            let fraction = self.timeline.hide_top_percentile / 100.0;
            if let Some(cutoff) = nnfractals::anim_eval::percentile_escape_time_cutoff(&shading, &color_t, max_iter, fraction) {
                let mut params_hidden = params;
                params_hidden.hide_above = Some(cutoff);
                let (shading2, color_t2) = pipeline.render(&params_hidden, &cam, self.render_size, self.render_size);
                shading = shading2;
                color_t = color_t2;
            }
        }
        self.last_render_ms = start.elapsed().as_secs_f64() * 1000.0;

        let hits = shading.iter().filter(|&&v| v > 0.0).count();
        self.last_hit_frac = hits as f32 / shading.len().max(1) as f32;

        let bg_color = (0.03, 0.02, 0.06);
        let rgb_bytes = nnfractals::colormap::apply_colormap_equalized(&color_t, max_iter, &self.colormap);
        let mut rgb = vec![0u8; shading.len() * 3];
        for i in 0..shading.len() {
            let v = shading[i];
            if v <= 0.0 {
                rgb[i * 3] = (bg_color.0 * 255.0) as u8;
                rgb[i * 3 + 1] = (bg_color.1 * 255.0) as u8;
                rgb[i * 3 + 2] = (bg_color.2 * 255.0) as u8;
            } else {
                rgb[i * 3] = (rgb_bytes[i * 3] as f32 * v).clamp(0.0, 255.0) as u8;
                rgb[i * 3 + 1] = (rgb_bytes[i * 3 + 1] as f32 * v).clamp(0.0, 255.0) as u8;
                rgb[i * 3 + 2] = (rgb_bytes[i * 3 + 2] as f32 * v).clamp(0.0, 255.0) as u8;
            }
        }
        let size = [self.render_size as usize, self.render_size as usize];
        let color_image = egui::ColorImage::from_rgb(size, &rgb);
        match self.texture.as_mut() {
            Some(t) => t.set(color_image, egui::TextureOptions::LINEAR),
            None => self.texture = Some(ctx.load_texture("anim_view", color_image, egui::TextureOptions::LINEAR)),
        }
        self.dirty = false;
    }

    // ── Timeline panel top-level draw (ruler + trim bar + every lane) ────

    fn show_timeline_panel(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui.button("zoom to fit").clicked() {
                let w = (ui.available_width() - 60.0).max(50.0);
                self.timeline_px_per_s = (w as f64 / self.timeline.duration_s.max(0.1)) as f32;
                self.save_prefs();
            }
            if ui.add(egui::Slider::new(&mut self.timeline_px_per_s, 4.0..=800.0).logarithmic(true).text("zoom (px/s)")).changed() {
                self.save_prefs();
            }
            ui.separator();
            if ui.button("+ effects lane").clicked() {
                self.timeline.effects_lanes.push(AxisTimeline::default());
                self.commit_timeline();
            }
            if !self.timeline.effects_lanes.is_empty() && ui.button("\u{2212} effects lane").clicked() {
                if let Some(popped_i) = Some(self.timeline.effects_lanes.len() - 1) {
                    self.timeline.effects_lanes.pop();
                    if self.selected_clip.map(|r| r.lane) == Some(LaneRef::Effects(popped_i)) {
                        self.selected_clip = None;
                    }
                }
                self.commit_timeline();
            }
        });

        let n_effects_lanes = self.timeline.effects_lanes.len();
        let row_count = 2 + 4 + n_effects_lanes + 1; // ruler + trim bar + X/Y/Z/T + effects lanes + audio
        let _ = row_count;

        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.set_width(60.0);
                fixed_height_label(ui, "", RULER_HEIGHT);
                fixed_height_label(ui, "trim", TRIM_BAR_HEIGHT);
                fixed_height_label(ui, "X", LANE_HEIGHT);
                fixed_height_label(ui, "Y", LANE_HEIGHT);
                fixed_height_label(ui, "Z", LANE_HEIGHT);
                fixed_height_label(ui, "T", LANE_HEIGHT);
                for i in 0..n_effects_lanes {
                    fixed_height_label(ui, &format!("eff {}", i + 1), LANE_HEIGHT);
                }
                fixed_height_label(ui, "audio", LANE_HEIGHT);
            });

            egui::ScrollArea::horizontal().id_salt("anim_timeline_hscroll").show(ui, |ui| {
                let content_w = ((self.timeline.duration_s * self.timeline_px_per_s as f64) as f32).max(200.0);
                ui.vertical(|ui| {
                    draw_ruler(ui, content_w, self.timeline_px_per_s, self.timeline.duration_s, self.playhead_s);

                    // Every row below returns (dirty, committed): dirty means
                    // "re-render this frame" (set during a live drag, exactly
                    // like the orbit-drag's own per-frame update), committed
                    // means "a drag just ended, persist it now" — kept
                    // separate so a long drag pushes exactly ONE undo entry
                    // (on release) instead of one per frame.
                    //
                    // `timeline_before_drag` must be refreshed HERE, before
                    // any row runs, whenever no lane-row drag is currently
                    // active — NOT via the generic `timeline_before_this_frame`
                    // (captured at the top of `ui()` every frame), which by
                    // the time a multi-frame drag's deferred commit finally
                    // runs would already equal `self.timeline` and make the
                    // whole gesture invisible to change-detection.
                    let drag_active_before = self.dragging_clip.is_some() || self.dragging_audio_clip.is_some() || self.dragging_trim.is_some();
                    if !drag_active_before {
                        self.timeline_before_drag = Some(self.timeline_before_this_frame.clone());
                    }

                    let mut any_dirty = false;
                    let mut any_committed = false;

                    let mut trim_start = self.timeline.trim_start_s;
                    let mut trim_end = self.timeline.trim_end_s;
                    let (trim_dirty, trim_committed) = draw_trim_bar(ui, content_w, self.timeline_px_per_s, self.timeline.duration_s, &mut trim_start, &mut trim_end, &mut self.dragging_trim);
                    if trim_dirty || trim_committed {
                        self.timeline.trim_start_s = trim_start;
                        self.timeline.trim_end_s = trim_end;
                        self.render_length_s = nnfractals::anim_timeline::wallclock_span(trim_end - trim_start, self.timeline.delta_t);
                    }
                    any_dirty |= trim_dirty;
                    any_committed |= trim_committed;

                    let duration_s = self.timeline.duration_s;
                    let px_per_s = self.timeline_px_per_s;
                    let (dx, cx) = axis_lane_row(ui, LaneRef::X, content_w, px_per_s, duration_s, &mut self.timeline.x_track, &mut self.selected_clip, &mut self.dragging_clip, &mut self.pending_add);
                    let (dy, cy) = axis_lane_row(ui, LaneRef::Y, content_w, px_per_s, duration_s, &mut self.timeline.y_track, &mut self.selected_clip, &mut self.dragging_clip, &mut self.pending_add);
                    let (dz, cz) = axis_lane_row(ui, LaneRef::Z, content_w, px_per_s, duration_s, &mut self.timeline.z_track, &mut self.selected_clip, &mut self.dragging_clip, &mut self.pending_add);
                    let (dt, ct) = axis_lane_row(ui, LaneRef::T, content_w, px_per_s, duration_s, &mut self.timeline.t_track, &mut self.selected_clip, &mut self.dragging_clip, &mut self.pending_add);
                    any_dirty |= dx || dy || dz || dt;
                    any_committed |= cx || cy || cz || ct;
                    for i in 0..n_effects_lanes {
                        let (de, ce) = axis_lane_row(ui, LaneRef::Effects(i), content_w, px_per_s, duration_s, &mut self.timeline.effects_lanes[i], &mut self.selected_clip, &mut self.dragging_clip, &mut self.pending_add);
                        any_dirty |= de;
                        any_committed |= ce;
                    }
                    let (da, ca) = audio_lane_row(ui, content_w, px_per_s, &mut self.timeline.audio_clips, &mut self.selected_clip, &mut self.dragging_audio_clip, &mut self.pending_add);
                    any_dirty |= da;
                    any_committed |= ca;

                    if any_committed {
                        let before = self.timeline_before_drag.take().unwrap_or_else(|| self.timeline_before_this_frame.clone());
                        self.commit_timeline_from(before);
                    } else if any_dirty {
                        self.dirty = true;
                    }
                });
            });
        });
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();

        // IPC: another launch delegated its genome path to us.
        while let Ok(path) = self.ipc_rx.try_recv() {
            if let Err(e) = self.load_genome(&path) {
                self.gpu_error = Some(format!("failed to load genome {}: {e}", path.display()));
            }
            ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
        }

        // The state BEFORE any of this frame's edits — whichever control
        // calls commit_timeline() this frame pushes this exact snapshot as
        // the undo entry (see commit_timeline's doc comment).
        self.timeline_before_this_frame = self.timeline.clone();

        let (undo_pressed, redo_pressed) = ui.input(|i| {
            let z_pressed = i.key_pressed(egui::Key::Z);
            let undo = z_pressed && i.modifiers.ctrl && !i.modifiers.shift;
            let redo = z_pressed && i.modifiers.ctrl && i.modifiers.shift;
            (undo, redo)
        });
        if undo_pressed {
            self.undo();
        }
        if redo_pressed {
            self.redo();
        }
        // Delete/Backspace removes the selected clip — guarded so typing
        // Backspace into a text field (e.g. the audio path box) never
        // deletes a clip out from under the user.
        if self.selected_clip.is_some() && !ctx.egui_wants_keyboard_input() {
            let delete_pressed = ui.input(|i| i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace));
            if delete_pressed {
                if let Some(r) = self.selected_clip {
                    match r.lane {
                        LaneRef::X => { self.timeline.x_track.clips.remove(r.index); }
                        LaneRef::Y => { self.timeline.y_track.clips.remove(r.index); }
                        LaneRef::Z => { self.timeline.z_track.clips.remove(r.index); }
                        LaneRef::T => { self.timeline.t_track.clips.remove(r.index); }
                        LaneRef::Effects(i) => { if let Some(lane) = self.timeline.effects_lanes.get_mut(i) { lane.clips.remove(r.index); } }
                        LaneRef::Audio => { self.timeline.audio_clips.remove(r.index); }
                    }
                    self.selected_clip = None;
                    self.commit_timeline();
                }
            }
        }

        // Transport: advance the playhead by real elapsed time * delta_t
        // (delta_t is a pure playback-rate multiplier, never applied to
        // authored-seconds content itself). Mirrors quat_viewer.rs's own
        // auto_rotate_last_tick pattern: last_frame_instant resets to None
        // whenever not playing, so resuming never applies one giant
        // catch-up jump.
        //
        // Gated on window focus — Carl, 2026-09-23: "Please make sure the
        // animator is not working in background when not viewed." Without
        // this, leaving playback running and alt-tabbing away kept driving
        // a GPU render every ~16ms indefinitely, competing with a queued
        // render for the GPU exactly when nobody's even watching this one.
        // `last_frame_instant` resets to `None` here too (not just on
        // pause), so losing and regaining focus mid-playback doesn't apply
        // one giant catch-up jump the moment the window is focused again —
        // same reasoning the plain-pause branch already relied on.
        let focused = ctx.input(|i| i.focused);
        if self.playing && focused {
            let now = std::time::Instant::now();
            if let Some(last) = self.last_frame_instant {
                let real_dt = now.duration_since(last).as_secs_f64();
                self.playhead_s = nnfractals::anim_timeline::advance_playhead(
                    self.playhead_s, real_dt, self.timeline.delta_t, self.timeline.duration_s, true,
                );
                self.dirty = true;
            }
            self.last_frame_instant = Some(now);
            ctx.request_repaint_after(std::time::Duration::from_millis(16));
        } else {
            self.last_frame_instant = None;
        }

        // Keep repainting at a real frame rate for the duration of ANY held
        // pointer press, not just video playback — otherwise, whenever this
        // app's host environment doesn't wake the event loop promptly on
        // its own for plain mouse motion (observed: idle repaints falling
        // back to a ~1Hz cadence under X11-forwarded/virtual-display
        // testing), a fast drag's intermediate positions never reach a
        // processed frame at all, and a gesture that pressed, moved, and
        // released can be coalesced into a single do-nothing frame. A real
        // display normally wakes on every motion event regardless, so this
        // is a defensive floor, not a behavior change for the common case.
        if ui.input(|i| i.pointer.any_down()) {
            ctx.request_repaint_after(std::time::Duration::from_millis(16));
        }

        egui::Panel::top("anim_top_bar").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.heading(&self.genome_label);
                if ui.add_enabled(!self.undo_stack.is_empty(), egui::Button::new("\u{21b6} undo")).clicked() {
                    self.undo();
                }
                if ui.add_enabled(!self.redo_stack.is_empty(), egui::Button::new("\u{21b7} redo")).clicked() {
                    self.redo();
                }
                ui.label("(Ctrl+Z / Ctrl+Shift+Z, Delete to remove a selected clip)");
                if let Some(err) = &self.gpu_error {
                    ui.colored_label(egui::Color32::RED, err);
                }
            });
        });

        egui::Panel::bottom("anim_quality_bar").exact_size(34.0).show(ui, |ui| {
            ui.horizontal(|ui| {
                if ui.add_sized([150.0, 20.0], egui::Slider::new(&mut self.render_size_cap, 256..=2400).text("max res")).changed() {
                    self.texture = None;
                    self.dirty = true;
                    self.save_prefs();
                }
                if ui.add_sized([120.0, 20.0], egui::Slider::new(&mut self.max_iter, 20..=150).text("iter")).changed() {
                    self.dirty = true;
                    self.save_prefs();
                }
                if ui.add_sized([110.0, 20.0], egui::Slider::new(&mut self.aa, 1..=3).text("AA")).changed() {
                    self.dirty = true;
                    self.save_prefs();
                }
                ui.separator();
                ui.label("preview:");
                if ui.add_sized([130.0, 20.0], egui::Slider::new(&mut self.preview_max_iter, 5..=150).text("iter")).changed() {
                    self.save_prefs();
                }
                if ui.add_sized([110.0, 20.0], egui::Slider::new(&mut self.preview_aa, 1..=3).text("AA")).changed() {
                    self.save_prefs();
                }
                ui.separator();
                egui::ComboBox::from_label("colormap")
                    .selected_text(&self.colormap)
                    .show_ui(ui, |ui| {
                        for name in ["lava", "aurora", "galaxy", "turbo", "viridis", "inferno", "plasma", "cubehelix", "sunset", "ember"] {
                            if ui.selectable_value(&mut self.colormap, name.to_string(), name).changed() {
                                self.dirty = true;
                                self.save_prefs();
                            }
                        }
                    });
                ui.separator();
                // Carl, 2026-09-23: "The render button could be placed
                // aside the colormap at the bottom" — moved from the top
                // bar down next to the colormap picker it exports with.
                if ui.button("Render\u{2026}").clicked() {
                    self.render_length_s = nnfractals::anim_timeline::wallclock_span(
                        self.timeline.trim_end_s - self.timeline.trim_start_s, self.timeline.delta_t,
                    );
                    self.render_status = None;
                    self.render_window_open = true;
                }
                ui.separator();
                ui.label(format!("{}x{}  {:.0}ms  hit {:.0}%", self.render_size, self.render_size, self.last_render_ms, self.last_hit_frac * 100.0));
            });
        });

        egui::Panel::bottom("anim_timeline_panel").resizable(true).default_size(300.0).min_size(160.0).show(ui, |ui| {
            self.show_timeline_panel(ui);
        });

        egui::Panel::right("anim_inspector_panel").default_size(300.0).min_size(220.0).resizable(true).show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Clip inspector");
                self.show_clip_inspector(ui);
            });
        });

        egui::Panel::left("anim_left_panel").default_size(240.0).min_size(200.0).resizable(true).show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.label("Drag to orbit. Scroll to zoom. Middle-drag to pan the origin.");
                ui.separator();

                let t_part = self.timeline.axis_assignment.part_of(ViewerAxis::T);
                ui.label(format!("T ({} axis) = {:.4}", t_part.label(), self.timeline.t_value));
                let range = self.timeline.bounds.t.min..=self.timeline.bounds.t.max;
                if ui.add(egui::Slider::new(&mut self.timeline.t_value, range).show_value(false)).changed() {
                    self.commit_timeline();
                }
                ui.separator();

                ui.label("Axis assignment — drag a part onto an axis");
                ui.horizontal(|ui| {
                    for part in [QuatPart::R, QuatPart::A, QuatPart::B, QuatPart::C] {
                        ui.dnd_drag_source(egui::Id::new(("anim_quatpart_badge", part)), part, |ui| {
                            ui.add_sized([26.0, 26.0], egui::Button::new(egui::RichText::new(part.label()).strong()));
                        });
                    }
                });
                let mut axis_changed = false;
                for axis in [ViewerAxis::X, ViewerAxis::Y, ViewerAxis::Z, ViewerAxis::T] {
                    let current_part = self.timeline.axis_assignment.part_of(axis);
                    ui.horizontal(|ui| {
                        ui.label(format!("{}:", axis.label()));
                        let frame = egui::Frame::group(ui.style());
                        let (_, dropped) = ui.dnd_drop_zone::<QuatPart, _>(frame, |ui| {
                            ui.set_min_width(48.0);
                            ui.label(current_part.label());
                        });
                        if let Some(part) = dropped {
                            self.timeline.axis_assignment.assign(axis, *part);
                            axis_changed = true;
                        }
                    });
                }
                if axis_changed {
                    self.commit_timeline();
                }
                ui.separator();

                ui.label("Bounding box");
                let mut bounds_changed = false;
                let mut go_to_bounds_axis: Option<ViewerAxis> = None;
                ui.horizontal(|ui| {
                    ui.label("X:");
                    bounds_changed |= ui.add(egui::DragValue::new(&mut self.timeline.bounds.x.min).speed(0.01)).changed();
                    ui.label("to");
                    bounds_changed |= ui.add(egui::DragValue::new(&mut self.timeline.bounds.x.max).speed(0.01)).changed();
                    if ui.small_button("\u{2192}bounds").on_hover_text("Go to bounds: jump X to the actual discovered raycasting limit").clicked() {
                        go_to_bounds_axis = Some(ViewerAxis::X);
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Y:");
                    bounds_changed |= ui.add(egui::DragValue::new(&mut self.timeline.bounds.y.min).speed(0.01)).changed();
                    ui.label("to");
                    bounds_changed |= ui.add(egui::DragValue::new(&mut self.timeline.bounds.y.max).speed(0.01)).changed();
                    if ui.small_button("\u{2192}bounds").on_hover_text("Go to bounds: jump Y to the actual discovered raycasting limit").clicked() {
                        go_to_bounds_axis = Some(ViewerAxis::Y);
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Z:");
                    bounds_changed |= ui.add(egui::DragValue::new(&mut self.timeline.bounds.z.min).speed(0.01)).changed();
                    ui.label("to");
                    bounds_changed |= ui.add(egui::DragValue::new(&mut self.timeline.bounds.z.max).speed(0.01)).changed();
                    if ui.small_button("\u{2192}bounds").on_hover_text("Go to bounds: jump Z to the actual discovered raycasting limit").clicked() {
                        go_to_bounds_axis = Some(ViewerAxis::Z);
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("T:");
                    bounds_changed |= ui.add(egui::DragValue::new(&mut self.timeline.bounds.t.min).speed(0.01)).changed();
                    ui.label("to");
                    bounds_changed |= ui.add(egui::DragValue::new(&mut self.timeline.bounds.t.max).speed(0.01)).changed();
                    if ui.small_button("\u{2192}bounds").on_hover_text("Go to bounds: jump T to the actual discovered raycasting limit").clicked() {
                        go_to_bounds_axis = Some(ViewerAxis::T);
                    }
                });
                ui.horizontal(|ui| {
                    if ui.button("reset to \u{b1}1.6").clicked() {
                        self.timeline.bounds = BoundingBox::default();
                        bounds_changed = true;
                    }
                    self.show_bbox_quick_set(ui);
                    // Carl, 2026-09-23: "add a go to bounds option in the
                    // animator. For all axis or for one only" — this is
                    // the "all axis" affordance; the per-row "→bounds"
                    // buttons above are "one only".
                    if ui.button("\u{2192} go to bounds (all axes)")
                        .on_hover_text("Set every axis to the actual raycasting-discovered limit for whatever quaternion part it's currently assigned to")
                        .clicked()
                    {
                        self.go_to_bounds(None);
                    }
                });
                if let Some(axis) = go_to_bounds_axis {
                    self.go_to_bounds(Some(axis));
                }
                ui.label(egui::RichText::new("click: set X/Y/Z to \u{b1}value  \u{2022}  drag: rescale current bounds").small());
                if let Some(status) = &self.bounds_status {
                    ui.colored_label(egui::Color32::from_rgb(150, 190, 255), status);
                }
                if bounds_changed {
                    self.timeline.t_value = self.timeline.t_value.clamp(self.timeline.bounds.t.min, self.timeline.bounds.t.max);
                    self.reframe_camera_to_bounds();
                    self.commit_timeline();
                }
                ui.separator();

                ui.label("Transport");
                ui.horizontal(|ui| {
                    let play_label = if self.playing { "\u{23f8} pause" } else { "\u{25b6} play" };
                    if ui.button(play_label).clicked() {
                        self.playing = !self.playing;
                        if !self.playing {
                            self.dirty = true; // one final full-quality render on pause
                        }
                    }
                    let mut duration = self.timeline.duration_s;
                    if ui.add(egui::DragValue::new(&mut duration).speed(0.1).range(0.1..=f64::INFINITY).suffix("s")).changed() {
                        self.timeline.duration_s = duration.max(0.1);
                        self.timeline.trim_end_s = self.timeline.trim_end_s.min(self.timeline.duration_s);
                        self.playhead_s = self.playhead_s.min(self.timeline.duration_s);
                        self.commit_timeline();
                    }
                });
                let playhead_label = format!("playhead {:.2}s", self.playhead_s);
                let scrub_resp = ui.add(
                    egui::Slider::new(&mut self.playhead_s, 0.0..=self.timeline.duration_s.max(0.1))
                        .text(playhead_label),
                );
                if scrub_resp.changed() {
                    self.dirty = true;
                }
                self.scrub_dragging = scrub_resp.dragged();
                if scrub_resp.drag_stopped() {
                    self.dirty = true; // one final full-quality render on release
                }
                let mut delta_t = self.timeline.delta_t;
                if ui.add(egui::Slider::new(&mut delta_t, 0.1..=4.0).logarithmic(true).text("speed (delta T)")).changed() {
                    self.timeline.delta_t = delta_t.max(1e-3);
                    self.commit_timeline();
                }
                ui.separator();

                if ui.button("reset view").clicked() {
                    self.timeline.camera_yaw = 0.0;
                    self.timeline.camera_pitch = 0.25;
                    self.timeline.camera_distance = recommended_orbit_radius(characteristic_scale(&self.timeline.bounds), 45.0, self.render_size, self.render_size);
                    self.timeline.camera_pivot_offset = (0.0, 0.0, 0.0);
                    self.commit_timeline();
                }
                let mut hide_pct = self.timeline.hide_top_percentile;
                if ui.add(egui::Slider::new(&mut hide_pct, 0.0..=50.0).text("hide top % (the white/gray shell)")).changed() {
                    self.timeline.hide_top_percentile = hide_pct;
                    self.commit_timeline();
                }
                ui.separator();

                // Profiles: named, genome-agnostic snapshots of the whole
                // timeline — save the current fractal's settings once,
                // re-apply them to any other fractal later.
                ui.label("Profiles");
                egui::ComboBox::from_id_salt("profile_select")
                    .selected_text(self.selected_profile.as_deref().unwrap_or("(none selected)"))
                    .show_ui(ui, |ui| {
                        for name in self.profile_names.clone() {
                            let selected = self.selected_profile.as_deref() == Some(name.as_str());
                            if ui.selectable_label(selected, &name).clicked() {
                                self.selected_profile = Some(name);
                            }
                        }
                    });
                ui.horizontal(|ui| {
                    let has_selection = self.selected_profile.is_some();
                    if ui.add_enabled(has_selection, egui::Button::new("apply")).clicked() {
                        self.apply_selected_profile();
                    }
                    if ui.add_enabled(has_selection, egui::Button::new("save over")).clicked() {
                        self.save_over_selected_profile();
                    }
                    if ui.add_enabled(has_selection, egui::Button::new("delete")).clicked() {
                        self.delete_selected_profile();
                    }
                });
                if ui.button("save as new\u{2026}").clicked() {
                    self.profile_save_as_name = self.selected_profile.clone().unwrap_or_default();
                    self.profile_save_as_open = true;
                }
                match &self.profile_status {
                    Some(Ok(msg)) => { ui.colored_label(egui::Color32::from_rgb(120, 200, 120), msg); }
                    Some(Err(e)) => { ui.colored_label(egui::Color32::RED, e); }
                    None => {}
                }
                ui.separator();

                // Persistence origin, shown so a reload can be visually
                // confirmed without inspecting the JSON file directly.
                ui.label("Animation settings");
                ui.monospace(format!("hash: {:016x}", self.timeline.genome_content_hash));
                match self.timeline_origin {
                    TimelineOrigin::LoadedFromDisk => ui.colored_label(egui::Color32::from_rgb(120, 200, 120), "loaded from anim_settings/"),
                    TimelineOrigin::NewlyCreated => ui.colored_label(egui::Color32::from_rgb(200, 180, 100), "new — saved to anim_settings/"),
                };
            });
        });

        egui::CentralPanel::default().show(ui, |ui| {
            let avail = ui.available_size();
            let side = avail.x.min(avail.y).max(64.0);
            let target_size = (side as u32).clamp(128, self.render_size_cap);
            if target_size != self.render_size && !self.dragging {
                self.render_size = target_size;
                self.texture = None;
                self.dirty = true;
            }

            if self.dirty {
                self.render(&ctx);
            }

            if let Some(tex_id) = self.texture.as_ref().map(|t| t.id()) {
                ui.centered_and_justified(|ui| {
                    let resp = ui.add(
                        egui::Image::new(egui::load::SizedTexture::new(tex_id, egui::vec2(side, side)))
                            .sense(egui::Sense::drag()),
                    );
                    if resp.dragged() {
                        self.dragging = true;
                        let delta = resp.drag_delta();
                        // Live per-frame update (for rendering) — NOT a
                        // commit_timeline() call here, which would push an
                        // undo entry + disk save every single frame of the
                        // drag. Committed once below, only when the drag
                        // actually ends.
                        if resp.dragged_by(egui::PointerButton::Middle) {
                            // MeshLab-style pan: middle-drag moves the orbit's
                            // pivot (Carl, 2026-09-22: "the mouse wheel button
                            // allow to move the origin like in meshlab"), not
                            // the viewing angle. Screen delta -> world offset
                            // via the manual orbit's own (unrotated) basis, so
                            // panning follows the cursor regardless of the
                            // current yaw/pitch. Scale matches this frame's
                            // actual projection (world units per pixel at the
                            // pivot's distance) so a drag tracks the point
                            // under the cursor at any zoom level.
                            let (_, right, up) = manual_orbit_basis(self.timeline.camera_yaw, self.timeline.camera_pitch);
                            let half_fov = (FOV_DEG * 0.5).to_radians();
                            let world_per_px = 2.0 * self.timeline.camera_distance * half_fov.tan() / (side as f64).max(1.0);
                            let dx = -delta.x as f64 * world_per_px;
                            let dy = delta.y as f64 * world_per_px;
                            let o = &mut self.timeline.camera_pivot_offset;
                            o.0 += right.0 * dx + up.0 * dy;
                            o.1 += right.1 * dx + up.1 * dy;
                            o.2 += right.2 * dx + up.2 * dy;
                            // See `AnimationTimeline::clamp_pivot_offset_to_bounds`'s
                            // doc comment — Carl, 2026-09-23: "the fractal vanish".
                            self.timeline.clamp_pivot_offset_to_bounds();
                        } else {
                            self.timeline.camera_yaw -= delta.x as f64 * 0.01;
                            self.timeline.camera_pitch = (self.timeline.camera_pitch + delta.y as f64 * 0.01).clamp(-1.5, 1.5);
                        }
                        self.dirty = true;
                    } else {
                        self.dragging = false;
                    }
                    if resp.drag_stopped() {
                        self.commit_timeline();
                    }
                    // Orbit-pivot overlay: WHILE dragging (orbit or pan),
                    // mark where the rotation center actually is and how
                    // far the camera currently sits from it — Carl,
                    // 2026-09-23: "when changing the origin, it often
                    // become very hard to monitor rotation, the center of
                    // rotation become unknown or infered at best... let the
                    // user see where is the origin and where he is
                    // situated next to it (also like meshlab browsing)."
                    // Gone the instant the drag ends, same as MeshLab's own
                    // pivot marker — this is an interaction aid, not a
                    // permanent HUD.
                    if self.dragging {
                        if let Some(cam) = self.last_cam {
                            let painter = ui.painter_at(resp.rect);
                            let col = egui::Color32::from_rgb(255, 210, 60);
                            let label = format!(
                                "origin \u{b7} {:.2}u away \u{b7} yaw {:.0}\u{b0} pitch {:.0}\u{b0}",
                                self.timeline.camera_distance,
                                self.timeline.camera_yaw.to_degrees(),
                                self.timeline.camera_pitch.to_degrees(),
                            );
                            match nnfractals::anim_eval::project_to_screen_ndc(&cam, cam.target, 1.0) {
                                Some((ndc_x, ndc_y)) => {
                                    let mag = (ndc_x.abs()).max(ndc_y.abs());
                                    let on_screen = mag <= 1.0;
                                    let (nx, ny) = if on_screen || mag < 1e-9 { (ndc_x, ndc_y) } else { (ndc_x / mag, ndc_y / mag) };
                                    let marker = egui::pos2(
                                        resp.rect.center().x + nx as f32 * resp.rect.width() * 0.5,
                                        resp.rect.center().y - ny as f32 * resp.rect.height() * 0.5,
                                    );
                                    if on_screen {
                                        // MeshLab's own trackball convention (Carl,
                                        // 2026-09-23, after seeing the plain crosshair:
                                        // "I really would like to see the axis
                                        // following the same convention as meshlab
                                        // (only when using mouse controls unlike
                                        // meshlab)") — R/G/B lines through the pivot
                                        // along world X/Y/Z, each extending both ways,
                                        // plus a faint reference ring sized to match
                                        // them. Kept to WHILE-dragging-only, unlike
                                        // MeshLab's own always-on gizmo — that
                                        // restriction is deliberate, not a gap to fix.
                                        // Fixed SCREEN-space length, not a world-space
                                        // offset re-projected — projecting a real world
                                        // offset let perspective foreshortening (which
                                        // differs per axis direction) make the lines
                                        // wildly different lengths and sprawl clear off
                                        // the visible frame at some camera distances
                                        // (confirmed live, not just reasoned about — a
                                        // screenshot showed exactly that). A fixed
                                        // on-screen length keeps the gizmo a small,
                                        // uniform, contained indicator near the pivot
                                        // at any zoom, matching MeshLab's own look.
                                        let gizmo_len: f32 = 70.0;
                                        let probe_world = self.timeline.camera_distance * 0.01;
                                        let axes: [((f64, f64, f64), egui::Color32); 3] = [
                                            ((1.0, 0.0, 0.0), egui::Color32::from_rgb(230, 70, 70)),
                                            ((0.0, 1.0, 0.0), egui::Color32::from_rgb(90, 210, 90)),
                                            ((0.0, 0.0, 1.0), egui::Color32::from_rgb(90, 150, 235)),
                                        ];
                                        let to_px = |ndc: (f64, f64)| egui::pos2(
                                            resp.rect.center().x + ndc.0 as f32 * resp.rect.width() * 0.5,
                                            resp.rect.center().y - ndc.1 as f32 * resp.rect.height() * 0.5,
                                        );
                                        for (axis, acol) in axes {
                                            // A small nearby probe point gives the axis's
                                            // on-screen DIRECTION (a local linearization
                                            // of the projection) without the probe's own
                                            // length leaking into the drawn length.
                                            let probe = (cam.target.0 + axis.0 * probe_world, cam.target.1 + axis.1 * probe_world, cam.target.2 + axis.2 * probe_world);
                                            if let Some(ndc) = nnfractals::anim_eval::project_to_screen_ndc(&cam, probe, 1.0) {
                                                let dir = (to_px(ndc) - marker).normalized();
                                                if dir.is_finite() && dir != egui::Vec2::ZERO {
                                                    painter.line_segment([marker - dir * gizmo_len, marker + dir * gizmo_len], (2.0, acol));
                                                }
                                            }
                                        }
                                        painter.circle_stroke(marker, gizmo_len, (1.0, egui::Color32::from_rgba_unmultiplied(255, 255, 255, 100)));
                                        painter.circle_filled(marker, 3.5, egui::Color32::WHITE);
                                        painter.text(marker + egui::vec2(gizmo_len + 6.0, -gizmo_len), egui::Align2::LEFT_BOTTOM, label, egui::FontId::monospace(12.0), col);
                                    } else {
                                        // Panned the pivot out of frame — an
                                        // edge-clamped pointer beats no
                                        // indication of which way it went.
                                        painter.circle_filled(marker, 5.0, col);
                                        painter.text(marker, egui::Align2::CENTER_CENTER, "\u{2192}", egui::FontId::monospace(16.0), col);
                                        painter.text(resp.rect.left_top() + egui::vec2(8.0, 8.0), egui::Align2::LEFT_TOP, format!("origin off-screen \u{b7} {label}"), egui::FontId::monospace(12.0), col);
                                    }
                                }
                                None => {
                                    painter.text(resp.rect.left_top() + egui::vec2(8.0, 8.0), egui::Align2::LEFT_TOP, format!("origin behind camera \u{b7} {label}"), egui::FontId::monospace(12.0), col);
                                }
                            }
                        }
                    }
                    let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                    if scroll.abs() > 0.0 {
                        self.timeline.camera_distance = (self.timeline.camera_distance * (1.0 - scroll as f64 * 0.002)).clamp(1.7, 40.0);
                        self.commit_timeline();
                    }
                });
            } else {
                ui.centered_and_justified(|ui| {
                    ui.label("no GPU renderer available — see the top bar");
                });
            }
        });

        if self.render_window_open {
            self.show_render_window(&ctx);
        }
        if self.bbox_dialog_open {
            self.show_bbox_dialog(&ctx);
        }
        if self.pending_add.is_some() {
            self.show_pending_add_window(&ctx);
        }
        if self.profile_save_as_open {
            self.show_profile_save_as_dialog(&ctx);
        }
    }
}

fn main() -> anyhow::Result<()> {
    let genome_path: PathBuf = std::env::args().nth(1).map(PathBuf::from).ok_or_else(|| {
        anyhow::anyhow!("Usage: nnfractals-anim-viewer <genome.nn>")
    })?;

    // ── Single-instance IPC (mirrors quat_viewer.rs's exactly) ─────────
    let sock_path = socket_path();
    if try_delegate(&sock_path, &genome_path) {
        eprintln!("[anim-viewer] Delegated to running instance.");
        return Ok(());
    }
    let _ = std::fs::remove_file(&sock_path);
    let (ipc_tx, ipc_rx) = mpsc::channel::<PathBuf>();
    let _sock_guard = match UnixListener::bind(&sock_path) {
        Ok(listener) => {
            thread::spawn(move || {
                for stream in listener.incoming() {
                    if let Ok(mut s) = stream {
                        let mut buf = String::new();
                        if s.read_to_string(&mut buf).is_ok() {
                            let p = PathBuf::from(buf.trim());
                            if p.exists() {
                                let _ = ipc_tx.send(p);
                                let _ = s.write_all(b"K");
                            }
                        }
                    }
                }
            });
            Some(SocketGuard(sock_path))
        }
        Err(e) => {
            if try_delegate(&sock_path, &genome_path) {
                eprintln!("[anim-viewer] Delegated to running instance (after bind race).");
                return Ok(());
            }
            eprintln!("[anim-viewer] IPC unavailable: {e}");
            None
        }
    };

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("NNFractals Animation Viewer")
            .with_inner_size([1400.0, 1000.0]),
        ..Default::default()
    };

    eframe::run_native(
        "NNFractals Animation Viewer",
        options,
        Box::new(move |cc| {
            nnfractals::gui_font::install(&cc.egui_ctx);
            Ok(Box::new(App::new(&genome_path, ipc_rx).expect("failed to load genome")))
        }),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;

    Ok(())
}
