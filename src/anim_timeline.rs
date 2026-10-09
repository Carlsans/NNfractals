//! Data model for the quaternion animation viewer (Phase 0 of the
//! animation-viewer plan). Pure Rust, no egui/wgpu dependency — this module
//! is evaluated identically by the live preview and the batch renderer
//! (see `anim_eval::build_frame_params`, added in a later phase).
//!
//! Naming note: `Quat`/`quaternion` already means exactly one thing in this
//! codebase — the fractal's own 4D iteration parameter
//! (`crate::quaternion::Quat{r,a,b,c}`, mirrored in every `.wgsl` shader).
//! Nothing in this file reuses that name for anything else. 3D camera
//! rotation (added in a later phase) lives in `crate::orient::OrientQuat`,
//! never here and never called `Quat`.

use std::path::PathBuf;
use std::sync::Arc;
use serde::{Deserialize, Serialize};

use crate::quaternion::Quat;

// ── Axis / part identity ─────────────────────────────────────────────────

/// One of the fractal's own 4 quaternion components — Carl's naming
/// (`crate::quaternion::Quat`'s own doc comment): R + A·i + B·j + C·k.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuatPart { R, A, B, C }

impl QuatPart {
    /// R=0, A=1, B=2, C=3 — the fixed on-disk/on-GPU-uniform ordering.
    pub fn index(self) -> usize {
        match self { QuatPart::R => 0, QuatPart::A => 1, QuatPart::B => 2, QuatPart::C => 3 }
    }
    pub fn label(self) -> &'static str {
        match self { QuatPart::R => "R", QuatPart::A => "A", QuatPart::B => "B", QuatPart::C => "C" }
    }
}

/// One of the 4 axes the viewer's timeline is organised around: 3 spatial
/// (X, Y, Z) plus time (T).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViewerAxis { X, Y, Z, T }

impl ViewerAxis {
    pub fn label(self) -> &'static str {
        match self { ViewerAxis::X => "X", ViewerAxis::Y => "Y", ViewerAxis::Z => "Z", ViewerAxis::T => "T" }
    }
}

/// A bijection QuatPart -> ViewerAxis: which quaternion component currently
/// drives which viewer axis. Generalizes `quat_fractal::TimeAxis` (which
/// lets the user pick only ONE part as "time" and hardcodes the other three
/// into a fixed x/y/z order inside `TimeAxis::assemble`) into the full
/// 4! = 24-way permutation "each part associated with only one axis,
/// swappable" requires. `quat_fractal::TimeAxis`/`assemble` are untouched —
/// this is new, additive, parallel infrastructure, not a replacement.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AxisAssignment { pub x: QuatPart, pub y: QuatPart, pub z: QuatPart, pub t: QuatPart }

impl Default for AxisAssignment {
    /// A->X, B->Y, R->Z, C->T — Carl's requested default for a freshly
    /// opened fractal (2026-09-28), superseding an earlier default
    /// (A->X, B->Y, C->Z, R->T). C plays the role of "time" here, matching
    /// the rest of this codebase's quaternion tooling (`quat_viewer.rs`'s
    /// hardcoded `TimeAxis::C`); all 24 permutations are equally supported,
    /// this is just the starting point a new timeline opens with.
    fn default() -> Self {
        AxisAssignment { x: QuatPart::A, y: QuatPart::B, z: QuatPart::R, t: QuatPart::C }
    }
}

impl AxisAssignment {
    /// Place 4 axis-space scalars into a `Quat` according to this assignment.
    pub fn assemble(&self, x: f64, y: f64, z: f64, t: f64) -> Quat {
        let mut parts = [0.0f64; 4];
        parts[self.x.index()] = x;
        parts[self.y.index()] = y;
        parts[self.z.index()] = z;
        parts[self.t.index()] = t;
        Quat { r: parts[0], a: parts[1], b: parts[2], c: parts[3] }
    }

    /// Which axis currently carries `part`.
    pub fn axis_of(&self, part: QuatPart) -> ViewerAxis {
        if self.x == part { ViewerAxis::X }
        else if self.y == part { ViewerAxis::Y }
        else if self.z == part { ViewerAxis::Z }
        else { ViewerAxis::T }
    }

    /// Which part currently sits on `axis`.
    pub fn part_of(&self, axis: ViewerAxis) -> QuatPart {
        match axis { ViewerAxis::X => self.x, ViewerAxis::Y => self.y, ViewerAxis::Z => self.z, ViewerAxis::T => self.t }
    }

    fn set(&mut self, axis: ViewerAxis, part: QuatPart) {
        match axis { ViewerAxis::X => self.x = part, ViewerAxis::Y => self.y = part, ViewerAxis::Z => self.z = part, ViewerAxis::T => self.t = part }
    }

    /// Move `part` onto `target`, swapping it with whatever was already
    /// there so the assignment always stays a valid bijection (every part
    /// on exactly one axis) — this is the ONLY way callers should mutate an
    /// `AxisAssignment`; never write to `.x`/`.y`/`.z`/`.t` directly.
    pub fn assign(&mut self, target: ViewerAxis, part: QuatPart) {
        let src = self.axis_of(part);
        if src == target { return; }
        let displaced = self.part_of(target);
        self.set(src, displaced);
        self.set(target, part);
    }
}

// ── Bounding box ─────────────────────────────────────────────────────────

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AxisBound { pub min: f64, pub max: f64 }

impl Default for AxisBound {
    /// ±1.6 mirrors today's hardcoded spherical `domain_radius` default
    /// (`quat_viewer.rs:552` etc.) as a generic, genome-agnostic fallback.
    fn default() -> Self { AxisBound { min: -1.6, max: 1.6 } }
}

/// A true axis-aligned box, independent bounds per axis including time.
/// x/y/z bound the ray-march volume (feeds `quat_raymarch::ray_box`, added
/// in a later phase). `t` bounds the TIME AXIS'S OWN VALUE RANGE
/// (generalizes today's `c_min`/`c_max`, `quat_viewer.rs:457-465`) — it
/// never feeds ray intersection, since time isn't a ray dimension.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BoundingBox { pub x: AxisBound, pub y: AxisBound, pub z: AxisBound, pub t: AxisBound }

impl Default for BoundingBox {
    fn default() -> Self {
        BoundingBox { x: AxisBound::default(), y: AxisBound::default(), z: AxisBound::default(), t: AxisBound::default() }
    }
}

impl BoundingBox {
    /// Scales X, Y, Z (never T — T is the time axis's own value range, not
    /// a ray-march spatial dimension) around each axis's OWN current
    /// center. An off-center box scales in place; this deliberately does
    /// NOT recenter on the origin, so a box the user has already nudged
    /// off-center keeps its position when rescaled.
    pub fn scale_xyz_around_center(&mut self, factor: f64) {
        for axis in [&mut self.x, &mut self.y, &mut self.z] {
            let center = (axis.min + axis.max) * 0.5;
            let half = (axis.max - axis.min) * 0.5 * factor;
            axis.min = center - half;
            axis.max = center + half;
        }
    }

    /// Sets X, Y, Z bounds directly to ±value (never T) — generalizes the
    /// GUI's "reset bounds to ±1.6" convenience to an arbitrary
    /// user-chosen value instead of the hardcoded default.
    pub fn set_xyz_symmetric(&mut self, value: f64) {
        let v = value.abs();
        self.x = AxisBound { min: -v, max: v };
        self.y = AxisBound { min: -v, max: v };
        self.z = AxisBound { min: -v, max: v };
    }
}

// ── Animatable scalar ────────────────────────────────────────────────────

/// Mirrors `formula::ModShape` restricted to the 4 shapes Carl named: Sine,
/// Triangle, Sawtooth, Orbit. `Orbit` on this single-channel `AnimParam`
/// degenerates to cosine — the same documented degenerate behavior
/// `formula::ModShape::Orbit` already has ("on a single-channel target it
/// degenerates to its real part, i.e. a cosine") — v1 has no 2-channel
/// `AnimParam` pairing, so every use here is this single-channel case.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum WaveShape { Sine, Triangle, Sawtooth, Orbit }

/// `freq` = cycles per AUTHORED second (not per-clip-normalized like
/// `formula::TimeMod`'s `t ∈ [0,1)`, since an effect clip's `t` here is
/// absolute authored-timeline seconds). `phase` in turns (0..1).
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Drive { pub shape: WaveShape, pub amp: f64, pub freq: f64, pub phase: f64 }

/// The complex offset for `d` at absolute authored time `t` (seconds).
/// `amp = 0` is always exactly 0.0 regardless of shape — the same no-op
/// invariant `formula::mod_offset` guarantees for `TimeMod`.
pub fn drive_offset(d: &Drive, t: f64) -> f64 {
    use std::f64::consts::TAU;
    let theta = TAU * (d.freq * t + d.phase);
    match d.shape {
        WaveShape::Sine => d.amp * theta.sin(),
        WaveShape::Triangle => {
            // 4|x - round(x)| - 1 over one period, same recipe as
            // formula::mod_offset's ModShape::Triangle arm.
            let x = d.freq * t + d.phase;
            let tri = 4.0 * (x - (x + 0.5).floor()).abs() - 1.0;
            d.amp * tri
        }
        WaveShape::Sawtooth => {
            let x = d.freq * t + d.phase;
            d.amp * (2.0 * (x - x.floor()) - 1.0)
        }
        WaveShape::Orbit => d.amp * theta.cos(),
    }
}

/// A clip-relative "from `constant` to `target`" transition across the
/// clip's own `[start_s, end_s)` span — a genuinely new third mode for
/// `AnimParam`, distinct from `drive`'s absolute-time periodic wave.
/// Mutually exclusive with `drive` at the GUI layer (a clip is in
/// "constant", "wave", or "ramp" mode, never two at once); if a hand-edited
/// file somehow sets both, `eval_anim_param` prefers `ramp`.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Ramp { pub target: f64, pub style: RampStyle }

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum RampStyle {
    /// Smoothstep-eased interpolation across the whole clip span.
    Rolled,
    /// Instant, non-eased jump from `constant` to `target` at
    /// `u = (t - start_s) / (end_s - start_s) == teleport_at` (clamped to
    /// [0,1]) — no interpolation at all, unlike `Rolled`.
    Teleport { teleport_at: f64 },
}

/// One typed, animatable scalar. `constant` alone (`drive = None`,
/// `ramp = None`) is the amp=0 no-op case, mirroring `TimeMod`'s "amp=0 is
/// always exactly the original" invariant.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnimParam {
    pub constant: f64,
    #[serde(default)] pub drive: Option<Drive>,
    #[serde(default)] pub ramp: Option<Ramp>,
}

impl AnimParam {
    pub fn constant(v: f64) -> Self { AnimParam { constant: v, drive: None, ramp: None } }
}

/// The value of `p` at absolute authored time `t` (seconds), given the
/// enclosing clip's own `[clip_start_s, clip_end_s)` span (needed only for
/// `ramp`, which is clip-relative — `drive` stays absolute-time as before).
/// `ramp` takes priority over `drive` when both are set (never produced by
/// the GUI, which keeps them mutually exclusive; documented here so a
/// hand-edited file's behavior isn't a surprise).
pub fn eval_anim_param(p: &AnimParam, t: f64, clip_start_s: f64, clip_end_s: f64) -> f64 {
    if let Some(ramp) = &p.ramp {
        let span = (clip_end_s - clip_start_s).max(1e-9);
        let u = ((t - clip_start_s) / span).clamp(0.0, 1.0);
        return match ramp.style {
            RampStyle::Rolled => {
                let eased = u * u * (3.0 - 2.0 * u); // smoothstep
                p.constant + (ramp.target - p.constant) * eased
            }
            RampStyle::Teleport { teleport_at } => {
                if u < teleport_at.clamp(0.0, 1.0) { p.constant } else { ramp.target }
            }
        };
    }
    p.constant + p.drive.map(|d| drive_offset(&d, t)).unwrap_or(0.0)
}

// ── Effect catalog (fixed, per Carl's confirmed scope — no user-authored
//    effect types) ──────────────────────────────────────────────────────

/// Rotation's axis is IMPLICIT = whichever axis track (X/Y/Z) the clip sits
/// on (never offered on T — the GUI hides it there). `degrees_per_second`
/// is constant-only for v1 (the GUI forces `drive = None`) so the
/// accumulated angle has a closed form — see `accumulated_angle`.
/// `Difference` lives only on an effects lane (its `pos`/`normal` are full
/// 3-vectors, not scoped to one axis) — GUI-enforced.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Effect {
    Rotation { degrees_per_second: AnimParam },
    Translation { offset: AnimParam },
    /// 1.0 = identity. Overlapping Scale clips on one track MULTIPLY, not
    /// sum — see `evaluate_axis_track`.
    Scale { factor: AnimParam },
    /// A cutaway plane fixed in WORLD space — as the camera orbits (manual
    /// drag or an animating Rotation-track clip), this plane stays glued to
    /// the fractal's own domain, so its cut visibly sweeps across the frame
    /// as the view moves around the object.
    Difference { pos: [AnimParam; 3], normal: [AnimParam; 3] },
    /// A cutaway plane fixed relative to the CAMERA instead — `camera_offset`
    /// (right, up, forward) and `normal` (also right/up/forward) are in the
    /// camera's own local axes, re-resolved into world space fresh every
    /// frame from that frame's actual eye/target/up_hint
    /// (`anim_eval::build_frame_params`). The cut therefore stays at the
    /// same place ON SCREEN throughout an orbit, instead of sweeping across
    /// it — Carl, 2026-09-22: "add to effects a fixed difference that
    /// doesn't rotate with the object" (the existing plain `Difference`,
    /// being world-fixed, visibly rotates WITH the object's silhouette as
    /// the camera orbits around it; this is the screen-locked alternative).
    FixedDifference { camera_offset: [AnimParam; 3], normal: [AnimParam; 3] },
    /// A band of escape-iteration values that becomes invisible to the
    /// raymarcher — surfaces whose escape-time falls in
    /// `[min_iter, max_iter)` are skipped (never treated as a hit),
    /// letting the ray continue further into the object and reveal
    /// deeper iteration shells. Carl, 2026-09-28: "This allow the
    /// raycasting to continue further into the object by ignoring some
    /// iteration values. A range is used to do so (min iter, max iter)."
    /// An empty band (`min_iter == max_iter`) is a no-op — the identity
    /// default, like Scale's `1.0` or Translation's `0.0`. Effects-lane
    /// only, like `Difference`/`FixedDifference` (GUI-enforced) — an
    /// axis track has no meaningful reading of "iteration count."
    SlideIteration { min_iter: AnimParam, max_iter: AnimParam },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EffectClip {
    pub effect: Effect,
    pub start_s: f64,
    pub end_s: f64,
    #[serde(default)] pub label: String,
}

impl EffectClip {
    fn active_at(&self, t: f64) -> bool { t >= self.start_s && t < self.end_s }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AxisTimeline { pub clips: Vec<EffectClip> }

// ── Audio (reference + mux only) ─────────────────────────────────────────

/// DEPRECATED — superseded by `AudioClip`/`AnimationTimeline::audio_clips`
/// (the timeline UI corrective pass's multi-clip audio lane). Kept only so
/// `AnimationTimeline::migrate_audio_track` can convert an old file's single
/// audio reference into the new shape; never written by current code.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AudioTrack {
    /// Referenced in place, never copied into the project.
    pub file_path: PathBuf,
    /// Clamped >= 0 at the GUI layer (ffmpeg's `-itsoffset` can't go negative).
    pub offset_s: f64,
    pub gain_db: f64,
    /// (min,max) peak pairs for waveform drawing. Rebuilt from `file_path`
    /// on load (never persisted) — `Arc` so cloning an `AnimationTimeline`
    /// for an undo snapshot is a refcount bump, not a waveform copy.
    #[serde(skip)]
    pub peaks: Option<Arc<Vec<(f32, f32)>>>,
}

fn default_trim_out() -> f64 { f64::INFINITY } // "play to the source's natural end" until probed

/// One positioned audio clip on the (now multi-clip) audio lane. Referenced
/// in place, never copied into the project — same "reference + mux only"
/// philosophy `AudioTrack` had. `offset_s`/`trim_out_s` are a trim-IN/trim-OUT
/// pair measured INTO THE SOURCE FILE (not timeline positions) — `start_s`
/// is the only timeline-position field; this mirrors how every other clip
/// type keeps "where on the timeline" (`start_s`/`end_s`) separate from
/// "how much of the underlying thing to use."
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AudioClip {
    pub file_path: PathBuf,
    /// Timeline position (authored seconds) where this clip starts playing.
    pub start_s: f64,
    /// Trim-IN: seconds into the source file where playback starts.
    pub offset_s: f64,
    /// Trim-OUT: seconds into the source file where playback stops.
    /// Defaults to "play to the end" until `source_duration_s` is probed
    /// (see `video_export::probe_audio_duration_s`) and the GUI clamps it.
    #[serde(default = "default_trim_out")]
    pub trim_out_s: f64,
    /// Probed once at import; 0.0 means "not probed yet." Used only to
    /// clamp `trim_out_s` in the GUI — never re-probed automatically if the
    /// referenced file changes on disk (same "reference, not a live sync"
    /// limitation `AudioTrack` already had).
    #[serde(default)]
    pub source_duration_s: f64,
    pub gain_db: f64,
    /// (min,max) peak pairs for waveform drawing. Rebuilt from `file_path`
    /// on load (never persisted) — `Arc` so cloning an `AnimationTimeline`
    /// for an undo snapshot is a refcount bump, not a waveform copy.
    #[serde(skip)]
    pub peaks: Option<Arc<Vec<(f32, f32)>>>,
}

// ── The timeline itself ──────────────────────────────────────────────────

fn schema_v1() -> u32 { 1 }
fn one() -> f64 { 1.0 }

/// Everything about how one fractal genome animates. Persisted keyed by
/// `genome_content_hash` (see `anim_persist`), never by file path, so
/// renaming/copying the `.nn` file doesn't lose it.
///
/// Every field carries `#[serde(default)]` from day one — matches
/// `Genome`'s own convention (~95% of its ~230 fields use it), which is
/// what has let that struct survive years of additions with no version
/// bump.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnimationTimeline {
    #[serde(default = "schema_v1")]
    pub schema_version: u32,
    pub genome_content_hash: u64,
    #[serde(default)]
    pub axis_assignment: AxisAssignment,
    #[serde(default)]
    pub bounds: BoundingBox,
    #[serde(default)]
    pub x_track: AxisTimeline,
    #[serde(default)]
    pub y_track: AxisTimeline,
    #[serde(default)]
    pub z_track: AxisTimeline,
    #[serde(default)]
    pub t_track: AxisTimeline,
    /// "Another timeline corresponds to the effects. This timeline can be
    /// multiple timelines" (Carl). Index order is bottom-to-top like a
    /// layers panel: later entries are drawn/composited on top, so when
    /// more than one lane has an active `Difference` clip at the same
    /// time, the LAST (highest-index) lane with an active clip wins — see
    /// `evaluate_difference`.
    #[serde(default)]
    pub effects_lanes: Vec<AxisTimeline>,
    /// DEPRECATED — see `AudioTrack`'s own doc comment and
    /// `migrate_audio_track`. Never populated by current code; kept only so
    /// an old on-disk timeline still deserializes, then gets migrated into
    /// `audio_clips` on load.
    #[serde(default)]
    pub audio: Option<AudioTrack>,
    /// The audio lane: zero or more independently positioned/trimmed clips,
    /// all mixed together at mux time (`video_export::mux_audio_clips`).
    #[serde(default)]
    pub audio_clips: Vec<AudioClip>,
    /// Playback-rate multiplier — see `wallclock_span`/`advance_playhead`.
    /// Never rescales authored content itself (clip start/end, drive
    /// freq/phase, duration/trim) — only how fast wall-clock or frame-index
    /// sweeps across that fixed authored-seconds ruler.
    #[serde(default = "one")]
    pub delta_t: f64,
    pub duration_s: f64,
    pub trim_start_s: f64,
    pub trim_end_s: f64,
    /// T axis's user-set BASE value — the constant this timeline's T track
    /// adds its own clips' offsets on top of (`anim_eval::build_frame_params`).
    /// Persisted (Phase 7) so a batch render reproduces exactly what the
    /// interactive T slider was set to, not always 0.0.
    #[serde(default)]
    pub t_value: f64,
    /// The manual orbit camera — yaw/pitch/distance BEFORE this frame's
    /// Rotation-track composition (`anim_eval::camera_orientation`) is
    /// applied. Persisted (Phase 7) for the same reason `t_value` is: a
    /// batch render must reproduce the exact view the user was looking at,
    /// not a hardcoded default. `camera_distance`'s default (5.0) is a
    /// generic estimate for the default ±1.6 bounding box — a freshly
    /// created timeline overrides it with a box-aware
    /// `anim_eval::recommended_orbit_radius` computation instead (see
    /// `anim_viewer.rs::App::new`'s `TimelineOrigin::NewlyCreated` arm).
    #[serde(default)]
    pub camera_yaw: f64,
    #[serde(default = "default_camera_pitch")]
    pub camera_pitch: f64,
    #[serde(default = "default_camera_distance")]
    pub camera_distance: f64,
    /// Manual pan offset added to the box pivot before orbiting around it —
    /// middle-mouse-drag in the live preview, MeshLab-style (Carl,
    /// 2026-09-22: "the mouse wheel button allow to move the origin like in
    /// meshlab"). Unlike `camera_yaw`/`pitch`/`distance` this ISN'T
    /// rotated by the Rotation-track `orientation` quaternion — it's a
    /// manual framing adjustment applied in world space, the same way
    /// panning in any 3D tool moves where you're looking, not the object.
    #[serde(default)]
    pub camera_pivot_offset: (f64, f64, f64),
    /// Always-on shortcut for the single most common `SlideIteration` use
    /// case: hides the top `hide_top_percentile` percent of escape-time
    /// values actually visible on screen, without needing an authored
    /// effects-lane clip. 0.0 = off. Stored as a plain PERCENTAGE (e.g.
    /// 15.0, not 0.15) to match how it's shown in the GUI.
    ///
    /// This is rank-based, not an absolute iteration count, and
    /// deliberately so — Carl, 2026-09-28: the gray/white "shell" "is
    /// always the most annoying since it effectively hide the whole
    /// fractal... I think just a checkbox with hide unescaped would fix
    /// that easy." A first cut shipped as exactly that (a literal
    /// `et >= max_iter` checkbox), but Carl found it removed NOTHING on a
    /// real genome: coloring is histogram-equalized against each frame's
    /// OWN escape-time distribution (`colormap::apply_colormap_equalized`),
    /// so "white" means "highest-ranked pixel in THIS frame," not "near
    /// max_iter" — a fixed absolute number can't reliably target it. Carl:
    /// "build the top N% version properly." The actual cutoff is resolved
    /// fresh per render from a real probe pass — see
    /// `anim_eval::percentile_escape_time_cutoff` and
    /// `quat_dag::RaymarchDagParams::hide_above` — never a fixed value
    /// stored here.
    #[serde(default)]
    pub hide_top_percentile: f64,
}

fn default_camera_pitch() -> f64 { 0.25 }
fn default_camera_distance() -> f64 { 5.0 }

impl AnimationTimeline {
    /// A fresh, empty timeline for a genome that has never been opened in
    /// the animation viewer before.
    pub fn new(genome_content_hash: u64) -> Self {
        AnimationTimeline {
            schema_version: schema_v1(),
            genome_content_hash,
            axis_assignment: AxisAssignment::default(),
            bounds: BoundingBox::default(),
            x_track: AxisTimeline::default(),
            y_track: AxisTimeline::default(),
            z_track: AxisTimeline::default(),
            t_track: AxisTimeline::default(),
            effects_lanes: Vec::new(),
            audio: None,
            audio_clips: Vec::new(),
            delta_t: 1.0,
            duration_s: 10.0,
            trim_start_s: 0.0,
            trim_end_s: 10.0,
            t_value: 0.0,
            camera_yaw: 0.0,
            camera_pitch: default_camera_pitch(),
            camera_distance: default_camera_distance(),
            camera_pivot_offset: (0.0, 0.0, 0.0),
            hide_top_percentile: 0.0,
        }
    }

    /// One-time migration from the deprecated single-audio `audio` field to
    /// the multi-clip `audio_clips` lane. Fires ONLY when `audio_clips` is
    /// still empty AND a deprecated `audio` value is present — never
    /// clobbers real edits already made to `audio_clips`, and is a no-op on
    /// a timeline that was never on the old format. Idempotent: safe to
    /// call unconditionally on every load (see `anim_persist::load_timeline`).
    pub fn migrate_audio_track(&mut self) {
        if self.audio_clips.is_empty() {
            if let Some(old) = self.audio.take() {
                self.audio_clips.push(AudioClip {
                    file_path: old.file_path,
                    start_s: 0.0,
                    offset_s: old.offset_s,
                    trim_out_s: default_trim_out(),
                    source_duration_s: 0.0,
                    gain_db: old.gain_db,
                    peaks: old.peaks,
                });
            }
        }
    }

    /// Keeps the manual middle-drag pan target from wandering outside the
    /// fractal's own static bounding box — Carl, 2026-09-23: "I get issues
    /// if I try moving the mousewheel button to change the origin. The
    /// fractal vanish." Confirmed live: against the pure black background
    /// (no reference grid), an unclamped pan tracks the cursor 1:1 in world
    /// space, and dragging across roughly half the frame is already enough
    /// to carry the look-at target fully out of the camera's frustum with
    /// nothing left on screen to navigate back by. Called both from the
    /// live middle-drag handler (so it can no longer happen) and on every
    /// genome load (so a timeline saved before this fix — already parked
    /// outside the box — self-heals the moment it's reopened, the same
    /// on-load-repair pattern as `migrate_audio_track`). Idempotent and
    /// safe to call unconditionally: a already-in-range offset is untouched.
    pub fn clamp_pivot_offset_to_bounds(&mut self) {
        let half_x = (self.bounds.x.max - self.bounds.x.min).abs() * 0.5;
        let half_y = (self.bounds.y.max - self.bounds.y.min).abs() * 0.5;
        let half_z = (self.bounds.z.max - self.bounds.z.min).abs() * 0.5;
        let o = &mut self.camera_pivot_offset;
        o.0 = o.0.clamp(-half_x, half_x);
        o.1 = o.1.clamp(-half_y, half_y);
        o.2 = o.2.clamp(-half_z, half_z);
    }
}

// ── Delta-T / time-mapping (section 3 of the plan) ───────────────────────

/// delta_t=2.0 plays twice as fast (half the wall-clock time for the same
/// authored span); delta_t=0.5 plays half speed. The ONLY place delta_t
/// enters the math — never applied to authored-seconds coordinates
/// themselves (clip start/end, drive freq/phase, duration/trim).
pub fn wallclock_span(authored_span_s: f64, delta_t: f64) -> f64 {
    authored_span_s / delta_t.max(1e-6)
}

/// Live preview: called once per real frame with real elapsed dt.
pub fn advance_playhead(pos_s: f64, real_dt_s: f64, delta_t: f64, duration_s: f64, looping: bool) -> f64 {
    let next = pos_s + real_dt_s * delta_t;
    if looping { next.rem_euclid(duration_s.max(1e-6)) } else { next.min(duration_s) }
}

/// Render frame mapping: frame `i` of `frame_count`, evenly spaced across
/// the trim range. Deliberately takes NO delta_t — delta_t has already
/// been folded into `frame_count` by the render prompt, so this is the one
/// function where a double-application bug could sneak in, and structurally
/// can't.
pub fn frame_pos_s(frame_i: u32, frame_count: u32, trim_start_s: f64, trim_end_s: f64) -> f64 {
    if frame_count <= 1 { return trim_start_s; }
    trim_start_s + (frame_i as f64 / (frame_count - 1) as f64) * (trim_end_s - trim_start_s)
}

// ── Track evaluation (section 2 of the plan) ─────────────────────────────

/// Sum of Rotation clips' contribution to a track's accumulated angle
/// (degrees) at absolute authored time `t`. Closed-form: `degrees_per_second`
/// is constant-only (GUI-enforced), so this is a plain sum over clips with
/// no integration, evaluable at any scrub position.
pub fn accumulated_angle(track: &AxisTimeline, t: f64) -> f64 {
    track.clips.iter().filter_map(|c| match &c.effect {
        Effect::Rotation { degrees_per_second } => {
            let rate = degrees_per_second.constant;
            let overlap = (t.min(c.end_s) - c.start_s).max(0.0);
            let overlap = if t <= c.start_s { 0.0 } else { overlap };
            Some(rate * overlap)
        }
        _ => None,
    }).sum()
}

/// This track's Translation/Scale contribution at absolute authored time
/// `t`. Translation clips SUM (mirrors `Genome::at_time`'s additive-offset
/// rule); Scale clips MULTIPLY (1.0 is the identity, not 0.0, so summing
/// would break the no-op invariant when clips overlap). Clips are
/// hard-edged: outside `[start_s, end_s)` a clip contributes nothing.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct AxisContribution { pub translation: f64, pub scale: f64 }

pub fn evaluate_axis_track(track: &AxisTimeline, t: f64) -> AxisContribution {
    let mut translation = 0.0;
    let mut scale = 1.0;
    for clip in &track.clips {
        if !clip.active_at(t) { continue; }
        match &clip.effect {
            Effect::Translation { offset } => translation += eval_anim_param(offset, t, clip.start_s, clip.end_s),
            Effect::Scale { factor } => scale *= eval_anim_param(factor, t, clip.start_s, clip.end_s),
            Effect::Rotation { .. } => {} // handled by accumulated_angle
            Effect::Difference { .. } | Effect::FixedDifference { .. } | Effect::SlideIteration { .. } => {} // never on an axis track (GUI-enforced)
        }
    }
    AxisContribution { translation, scale }
}

/// The resolved active cutaway plane — either kind of `Effect`
/// `evaluate_difference` can find, with its `AnimParam`s already evaluated
/// at time `t` but NOT yet placed in world space. `CameraFixed` needs the
/// current frame's actual camera (eye/target/up_hint) to resolve into world
/// coordinates — see `anim_eval::build_frame_params`, the only caller.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DifferencePlane {
    WorldFixed { pos: [f64; 3], normal: [f64; 3] },
    CameraFixed { camera_offset: [f64; 3], normal: [f64; 3] },
}

/// The active clip plane at absolute authored time `t`, or `None` if no lane
/// has an active `Difference`/`FixedDifference` clip. When more than one
/// lane has one active at once, the LAST lane in `lanes` (topmost, per
/// `AnimationTimeline::effects_lanes`'s doc) wins — regardless of which of
/// the two kinds it is.
pub fn evaluate_difference(lanes: &[AxisTimeline], t: f64) -> Option<DifferencePlane> {
    for lane in lanes.iter().rev() {
        for clip in &lane.clips {
            if !clip.active_at(t) { continue; }
            let (s, e) = (clip.start_s, clip.end_s);
            match &clip.effect {
                Effect::Difference { pos, normal } => {
                    let p = [eval_anim_param(&pos[0], t, s, e), eval_anim_param(&pos[1], t, s, e), eval_anim_param(&pos[2], t, s, e)];
                    let n = [eval_anim_param(&normal[0], t, s, e), eval_anim_param(&normal[1], t, s, e), eval_anim_param(&normal[2], t, s, e)];
                    return Some(DifferencePlane::WorldFixed { pos: p, normal: n });
                }
                Effect::FixedDifference { camera_offset, normal } => {
                    let o = [eval_anim_param(&camera_offset[0], t, s, e), eval_anim_param(&camera_offset[1], t, s, e), eval_anim_param(&camera_offset[2], t, s, e)];
                    let n = [eval_anim_param(&normal[0], t, s, e), eval_anim_param(&normal[1], t, s, e), eval_anim_param(&normal[2], t, s, e)];
                    return Some(DifferencePlane::CameraFixed { camera_offset: o, normal: n });
                }
                _ => {}
            }
        }
    }
    None
}

/// The active hidden-iteration band `(min_iter, max_iter)` at absolute
/// authored time `t`, or `None` if no lane has an active `SlideIteration`
/// clip. Scanned independently of `evaluate_difference` — a
/// Difference/FixedDifference cutaway and a SlideIteration reveal can
/// both be active at once, on the same or different effects lanes. Same
/// "topmost active lane wins" rule as `evaluate_difference`.
pub fn evaluate_slide_iteration(lanes: &[AxisTimeline], t: f64) -> Option<(f64, f64)> {
    for lane in lanes.iter().rev() {
        for clip in &lane.clips {
            if !clip.active_at(t) { continue; }
            if let Effect::SlideIteration { min_iter, max_iter } = &clip.effect {
                let (s, e) = (clip.start_s, clip.end_s);
                let lo = eval_anim_param(min_iter, t, s, e);
                let hi = eval_anim_param(max_iter, t, s, e);
                return Some((lo, hi));
            }
        }
    }
    None
}

// ── Audio mux resolution (multi-clip audio lane) ─────────────────────────

/// One `AudioClip` resolved into ffmpeg's own seek/length/delay knobs for a
/// render covering `[trim_start_s, trim_end_s)`. Pure and ffmpeg-free, so
/// it's fully unit-testable without shelling out — `video_export::
/// mux_audio_clips` is what actually spends these on a real ffmpeg call.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedAudioInput {
    pub file_path: PathBuf,
    /// ffmpeg `-ss` (before `-i`): seconds into the source file to start reading.
    pub source_seek_s: f64,
    /// ffmpeg `-t`: how many seconds of source audio to read.
    pub play_len_s: f64,
    /// ffmpeg `-itsoffset` (always >= 0 — ffmpeg can't delay by a negative
    /// amount): seconds of silence to prepend in the OUTPUT before this
    /// clip's audio starts.
    pub output_delay_s: f64,
    pub gain_db: f64,
}

/// Maps `clip` onto the render's own `[trim_start_s, trim_end_s)` window, or
/// `None` if the clip doesn't overlap that window at all. A clip that starts
/// before `trim_start_s` is handled by seeking FURTHER INTO the source
/// (`source_seek_s > offset_s`) rather than attempting a negative
/// `output_delay_s`, which ffmpeg's `-itsoffset` cannot express.
pub fn resolve_audio_clip_for_render(clip: &AudioClip, trim_start_s: f64, trim_end_s: f64) -> Option<ResolvedAudioInput> {
    let render_span = trim_end_s - trim_start_s;
    if render_span <= 0.0 {
        return None;
    }
    // Position of the clip's own (untrimmed-by-the-window) start, relative
    // to the RENDERED video's own t=0.
    let video_start = clip.start_s - trim_start_s;
    let trim_out = if clip.source_duration_s > 0.0 { clip.trim_out_s.min(clip.source_duration_s) } else { clip.trim_out_s };
    let natural_len = (trim_out - clip.offset_s).max(0.0);
    if natural_len <= 0.0 || video_start >= render_span || video_start + natural_len <= 0.0 {
        return None; // entirely outside the rendered window, or zero-length
    }
    let (source_seek_s, output_delay_s) = if video_start >= 0.0 {
        (clip.offset_s, video_start)
    } else {
        // The clip started playing before the window opened — skip ahead
        // into the source by exactly how much of it we've already missed.
        (clip.offset_s - video_start, 0.0)
    };
    let consumed = source_seek_s - clip.offset_s;
    let play_len_s = (natural_len - consumed).min(render_span - output_delay_s).max(0.0);
    if play_len_s <= 0.0 {
        return None;
    }
    Some(ResolvedAudioInput { file_path: clip.file_path.clone(), source_seek_s, play_len_s, output_delay_s, gain_db: clip.gain_db })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clip(effect: Effect, start_s: f64, end_s: f64) -> EffectClip {
        EffectClip { effect, start_s, end_s, label: String::new() }
    }

    #[test]
    fn scale_xyz_around_center_preserves_an_off_center_center() {
        let mut b = BoundingBox {
            x: AxisBound { min: 1.0, max: 5.0 }, // center 3.0, half-extent 2.0
            y: AxisBound { min: -2.0, max: 4.0 }, // center 1.0, half-extent 3.0
            z: AxisBound::default(),
            t: AxisBound { min: -1.6, max: 1.6 },
        };
        b.scale_xyz_around_center(2.0);
        assert_eq!(b.x, AxisBound { min: -1.0, max: 7.0 }, "center 3.0 preserved, half-extent doubled to 4.0");
        assert_eq!(b.y, AxisBound { min: -5.0, max: 7.0 }, "center 1.0 preserved, half-extent doubled to 6.0");
    }

    #[test]
    fn scale_xyz_around_center_never_touches_t() {
        let mut b = BoundingBox { t: AxisBound { min: -3.0, max: 9.0 }, ..BoundingBox::default() };
        let t_before = b.t;
        b.scale_xyz_around_center(5.0);
        assert_eq!(b.t, t_before);
    }

    #[test]
    fn set_xyz_symmetric_sets_exact_value_and_never_touches_t() {
        let mut b = BoundingBox { x: AxisBound { min: 1.0, max: 5.0 }, ..BoundingBox::default() };
        let t_before = b.t;
        b.set_xyz_symmetric(4.2);
        assert_eq!(b.x, AxisBound { min: -4.2, max: 4.2 });
        assert_eq!(b.y, AxisBound { min: -4.2, max: 4.2 });
        assert_eq!(b.z, AxisBound { min: -4.2, max: 4.2 });
        assert_eq!(b.t, t_before, "T is untouched by set_xyz_symmetric");
        b.set_xyz_symmetric(-3.0);
        assert_eq!(b.x, AxisBound { min: -3.0, max: 3.0 }, "a negative input is treated as a magnitude");
    }

    #[test]
    fn axis_assignment_default_is_a_bijection() {
        let a = AxisAssignment::default();
        let mut parts = [a.x, a.y, a.z, a.t];
        parts.sort_by_key(|p| p.index());
        assert_eq!(parts, [QuatPart::R, QuatPart::A, QuatPart::B, QuatPart::C]);
    }

    #[test]
    fn assign_swaps_instead_of_duplicating() {
        let mut a = AxisAssignment::default(); // x=A y=B z=R t=C
        assert_eq!(a.axis_of(QuatPart::R), ViewerAxis::Z);
        assert_eq!(a.axis_of(QuatPart::A), ViewerAxis::X);
        // Drag R onto X: X should become R, and whatever was on X (A) must
        // land wherever R used to be (Z) — never disappear or duplicate.
        a.assign(ViewerAxis::X, QuatPart::R);
        assert_eq!(a.x, QuatPart::R);
        assert_eq!(a.z, QuatPart::A);
        let mut parts = [a.x, a.y, a.z, a.t];
        parts.sort_by_key(|p| p.index());
        assert_eq!(parts, [QuatPart::R, QuatPart::A, QuatPart::B, QuatPart::C], "bijection must hold after a swap");
    }

    #[test]
    fn assign_to_same_axis_is_a_no_op() {
        let mut a = AxisAssignment::default();
        let before = a;
        a.assign(ViewerAxis::X, a.x);
        assert_eq!(a, before);
    }

    #[test]
    fn assemble_places_each_axis_into_its_assigned_part() {
        let mut a = AxisAssignment::default(); // x=A y=B z=R t=C
        let q = a.assemble(10.0, 20.0, 30.0, 40.0);
        assert_eq!(q, Quat { r: 30.0, a: 10.0, b: 20.0, c: 40.0 });
        a.assign(ViewerAxis::X, QuatPart::C); // x=C now, C displaced from t, so t gets whatever was on x (A)
        let q2 = a.assemble(1.0, 2.0, 3.0, 4.0);
        assert_eq!(q2.c, 1.0);
        assert_eq!(q2.a, 4.0);
    }

    #[test]
    fn drive_offset_amp_zero_is_always_a_no_op() {
        for shape in [WaveShape::Sine, WaveShape::Triangle, WaveShape::Sawtooth, WaveShape::Orbit] {
            let d = Drive { shape, amp: 0.0, freq: 1.3, phase: 0.2 };
            for t in [0.0, 0.5, 1.0, 3.7, -2.0] {
                assert_eq!(drive_offset(&d, t), 0.0, "shape={shape:?} t={t}");
            }
        }
    }

    #[test]
    fn eval_anim_param_with_no_drive_is_just_the_constant() {
        let p = AnimParam::constant(4.2);
        for t in [0.0, 1.0, 100.0] {
            assert_eq!(eval_anim_param(&p, t, 0.0, 10.0), 4.2);
        }
    }

    #[test]
    fn ramp_rolled_smoothsteps_from_constant_to_target_across_the_clip_span() {
        let mut p = AnimParam::constant(2.0);
        p.ramp = Some(Ramp { target: 10.0, style: RampStyle::Rolled });
        assert_eq!(eval_anim_param(&p, 0.0, 0.0, 10.0), 2.0, "u=0 must be exactly the 'from' value");
        assert_eq!(eval_anim_param(&p, 10.0, 0.0, 10.0), 10.0, "u=1 must be exactly the 'to' value");
        let mid = eval_anim_param(&p, 5.0, 0.0, 10.0);
        assert_eq!(mid, 6.0, "u=0.5 smoothstep(0.5)=0.5, so exactly the midpoint 2.0+0.5*(10.0-2.0)");
        // Monotonic and strictly between endpoints partway through — not a step function.
        let q1 = eval_anim_param(&p, 2.5, 0.0, 10.0);
        let q3 = eval_anim_param(&p, 7.5, 0.0, 10.0);
        assert!(2.0 < q1 && q1 < mid && mid < q3 && q3 < 10.0, "q1={q1} mid={mid} q3={q3}");
    }

    #[test]
    fn ramp_teleport_snaps_at_teleport_at_not_before() {
        let mut p = AnimParam::constant(2.0);
        p.ramp = Some(Ramp { target: 10.0, style: RampStyle::Teleport { teleport_at: 0.5 } });
        assert_eq!(eval_anim_param(&p, 0.0, 0.0, 10.0), 2.0);
        assert_eq!(eval_anim_param(&p, 4.9, 0.0, 10.0), 2.0, "just before the 50% mark: still 'from'");
        assert_eq!(eval_anim_param(&p, 5.0, 0.0, 10.0), 10.0, "at exactly the 50% mark: snapped to 'to'");
        assert_eq!(eval_anim_param(&p, 5.1, 0.0, 10.0), 10.0, "just after: still 'to'");
        assert_eq!(eval_anim_param(&p, 9.9, 0.0, 10.0), 10.0);
    }

    #[test]
    fn ramp_clamps_outside_the_clip_span_never_extrapolates() {
        let mut rolled = AnimParam::constant(0.0);
        rolled.ramp = Some(Ramp { target: 100.0, style: RampStyle::Rolled });
        assert_eq!(eval_anim_param(&rolled, -5.0, 0.0, 10.0), 0.0, "before start_s: clamps to u=0, the 'from' value");
        assert_eq!(eval_anim_param(&rolled, 50.0, 0.0, 10.0), 100.0, "past end_s: clamps to u=1, the 'to' value");

        let mut teleport = AnimParam::constant(0.0);
        teleport.ramp = Some(Ramp { target: 100.0, style: RampStyle::Teleport { teleport_at: 0.5 } });
        assert_eq!(eval_anim_param(&teleport, -5.0, 0.0, 10.0), 0.0);
        assert_eq!(eval_anim_param(&teleport, 50.0, 0.0, 10.0), 100.0);
    }

    #[test]
    fn ramp_none_leaves_the_existing_drive_path_untouched() {
        // Regression guard: when ramp is None, eval_anim_param must behave
        // exactly as it did before this field existed (constant + drive).
        let p = AnimParam { constant: 1.0, drive: Some(Drive { shape: WaveShape::Sine, amp: 2.0, freq: 1.0, phase: 0.0 }), ramp: None };
        for t in [0.0, 0.25, 0.5, 1.0, 3.3] {
            let expected = 1.0 + drive_offset(&Drive { shape: WaveShape::Sine, amp: 2.0, freq: 1.0, phase: 0.0 }, t);
            assert_eq!(eval_anim_param(&p, t, 0.0, 10.0), expected, "clip span must be irrelevant when ramp is None");
            // Confirm the clip span truly doesn't matter for the drive path.
            assert_eq!(eval_anim_param(&p, t, -50.0, 500.0), expected);
        }
    }

    #[test]
    fn wallclock_span_matches_delta_t_definition() {
        assert!((wallclock_span(10.0, 2.0) - 5.0).abs() < 1e-9, "2x speed halves duration");
        assert!((wallclock_span(10.0, 0.5) - 20.0).abs() < 1e-9, "half speed doubles duration");
        assert!((wallclock_span(10.0, 1.0) - 10.0).abs() < 1e-9, "1x speed is unchanged");
    }

    #[test]
    fn advance_playhead_loops_and_clamps() {
        assert!((advance_playhead(9.0, 2.0, 1.0, 10.0, true) - 1.0).abs() < 1e-9, "looping wraps past duration");
        assert!((advance_playhead(9.0, 2.0, 1.0, 10.0, false) - 10.0).abs() < 1e-9, "non-looping clamps at duration");
    }

    #[test]
    fn frame_pos_s_spans_the_trim_range_evenly() {
        assert_eq!(frame_pos_s(0, 5, 2.0, 12.0), 2.0);
        assert_eq!(frame_pos_s(4, 5, 2.0, 12.0), 12.0);
        assert!((frame_pos_s(2, 5, 2.0, 12.0) - 7.0).abs() < 1e-9, "midpoint frame lands halfway");
        assert_eq!(frame_pos_s(0, 1, 2.0, 12.0), 2.0, "a single frame is trim_start, never divides by zero");
    }

    #[test]
    fn accumulated_angle_is_a_closed_form_sum_over_clips() {
        let track = AxisTimeline {
            clips: vec![
                clip(Effect::Rotation { degrees_per_second: AnimParam::constant(10.0) }, 0.0, 5.0),
                clip(Effect::Rotation { degrees_per_second: AnimParam::constant(20.0) }, 2.0, 100.0),
            ],
        };
        assert_eq!(accumulated_angle(&track, 0.0), 0.0, "at the very start, nothing has accumulated yet");
        assert_eq!(accumulated_angle(&track, 2.0), 20.0, "first clip only: 10 deg/s * 2s");
        assert_eq!(accumulated_angle(&track, 4.0), 10.0 * 4.0 + 20.0 * 2.0, "both clips active and overlapping sum");
        assert_eq!(accumulated_angle(&track, 10.0), 10.0 * 5.0 + 20.0 * 8.0, "first clip stops contributing once past its end_s");
    }

    #[test]
    fn translation_sums_scale_multiplies_and_both_are_hard_edged() {
        let track = AxisTimeline {
            clips: vec![
                clip(Effect::Translation { offset: AnimParam::constant(1.0) }, 0.0, 10.0),
                clip(Effect::Translation { offset: AnimParam::constant(2.0) }, 5.0, 15.0),
                clip(Effect::Scale { factor: AnimParam::constant(2.0) }, 0.0, 10.0),
                clip(Effect::Scale { factor: AnimParam::constant(3.0) }, 5.0, 15.0),
            ],
        };
        let at_1 = evaluate_axis_track(&track, 1.0);
        assert_eq!(at_1.translation, 1.0, "only the first translation clip is active");
        assert_eq!(at_1.scale, 2.0, "only the first scale clip is active");

        let at_7 = evaluate_axis_track(&track, 7.0);
        assert_eq!(at_7.translation, 3.0, "both translation clips overlap and sum: 1.0 + 2.0");
        assert_eq!(at_7.scale, 6.0, "both scale clips overlap and multiply: 2.0 * 3.0");

        let at_20 = evaluate_axis_track(&track, 20.0);
        assert_eq!(at_20.translation, 0.0, "past every clip's end_s, translation is the identity 0.0");
        assert_eq!(at_20.scale, 1.0, "past every clip's end_s, scale is the identity 1.0, not 0.0");
    }

    /// Phase 4's own checkpoint scenario, exercised directly: a sine-driven
    /// Translation clip on one track must genuinely oscillate WITHIN its
    /// window (not just return a single fixed offset) and hold flat at
    /// exactly the identity (0.0) outside it.
    #[test]
    fn sine_driven_translation_oscillates_inside_the_clip_and_holds_flat_outside() {
        let track = AxisTimeline {
            clips: vec![clip(
                Effect::Translation { offset: AnimParam { constant: 0.0, drive: Some(Drive { shape: WaveShape::Sine, amp: 0.4, freq: 0.5, phase: 0.0 }), ramp: None } },
                0.0, 10.0,
            )],
        };
        let a = evaluate_axis_track(&track, 1.0).translation;
        let b = evaluate_axis_track(&track, 2.0).translation;
        let c = evaluate_axis_track(&track, 3.0).translation;
        assert_ne!(a, b, "a sine drive must actually vary across the clip's window, not sit at one value");
        assert_ne!(b, c);
        assert!(a.abs() <= 0.4 + 1e-9 && b.abs() <= 0.4 + 1e-9 && c.abs() <= 0.4 + 1e-9, "amplitude 0.4 bounds every sample");
        assert_eq!(evaluate_axis_track(&track, 15.0).translation, 0.0, "past end_s=10.0, the clip contributes nothing at all");
        assert_eq!(evaluate_axis_track(&track, -1.0).translation, 0.0, "before start_s=0.0, the clip contributes nothing at all");
    }

    #[test]
    fn evaluate_difference_topmost_lane_wins() {
        let bottom = AxisTimeline {
            clips: vec![clip(Effect::Difference {
                pos: [AnimParam::constant(0.0), AnimParam::constant(0.0), AnimParam::constant(0.0)],
                normal: [AnimParam::constant(1.0), AnimParam::constant(0.0), AnimParam::constant(0.0)],
            }, 0.0, 100.0)],
        };
        let top = AxisTimeline {
            clips: vec![clip(Effect::Difference {
                pos: [AnimParam::constant(9.0), AnimParam::constant(9.0), AnimParam::constant(9.0)],
                normal: [AnimParam::constant(0.0), AnimParam::constant(1.0), AnimParam::constant(0.0)],
            }, 0.0, 100.0)],
        };
        let lanes = vec![bottom, top];
        let plane = evaluate_difference(&lanes, 1.0).expect("an active clip exists");
        assert_eq!(plane, DifferencePlane::WorldFixed { pos: [9.0, 9.0, 9.0], normal: [0.0, 1.0, 0.0] },
            "the LAST (topmost) lane's clip must win, not the first");
    }

    #[test]
    fn evaluate_difference_none_when_no_lane_active() {
        let lanes: Vec<AxisTimeline> = vec![AxisTimeline::default()];
        assert!(evaluate_difference(&lanes, 1.0).is_none());
    }

    /// Phase 6's own checkpoint scenario: a Sawtooth-driven position
    /// component on a Difference clip must genuinely slide across the
    /// clip's window (not sit at one value), confirming the cut can be
    /// animated to "slide open over time" exactly like any other
    /// AnimParam — Difference doesn't need special-case support for this,
    /// since eval_anim_param already handles any driven component.
    #[test]
    fn sawtooth_driven_difference_position_slides_across_the_window() {
        let lane = AxisTimeline {
            clips: vec![clip(Effect::Difference {
                pos: [
                    AnimParam { constant: 0.0, drive: Some(Drive { shape: WaveShape::Sawtooth, amp: 1.6, freq: 0.1, phase: 0.0 }), ramp: None },
                    AnimParam::constant(0.0),
                    AnimParam::constant(0.0),
                ],
                normal: [AnimParam::constant(1.0), AnimParam::constant(0.0), AnimParam::constant(0.0)],
            }, 0.0, 10.0)],
        };
        let world_fixed_x = |plane: DifferencePlane| match plane {
            DifferencePlane::WorldFixed { pos, .. } => pos[0],
            other => panic!("expected WorldFixed, got {other:?}"),
        };
        let pos_at_0 = world_fixed_x(evaluate_difference(&[lane.clone()], 0.0).unwrap());
        let pos_at_5 = world_fixed_x(evaluate_difference(&[lane.clone()], 5.0).unwrap());
        let pos_at_9 = world_fixed_x(evaluate_difference(&[lane], 9.0).unwrap());
        assert_ne!(pos_at_0, pos_at_5, "the cut plane's X position must move as the playhead advances");
        assert_ne!(pos_at_5, pos_at_9);
        // A sawtooth ramps up then snaps back — the two later samples on
        // the ramp should differ by roughly the expected linear rate,
        // not just "differ by something."
        assert!(pos_at_9 > pos_at_5, "within one un-wrapped ramp segment, position should increase with t");
    }

    #[test]
    fn animation_timeline_serde_round_trip() {
        let mut tl = AnimationTimeline::new(0xDEADBEEFCAFEu64);
        tl.axis_assignment.assign(ViewerAxis::X, QuatPart::C);
        tl.x_track.clips.push(clip(Effect::Translation { offset: AnimParam { constant: 0.0, drive: Some(Drive { shape: WaveShape::Sine, amp: 0.5, freq: 0.1, phase: 0.0 }), ramp: None } }, 0.0, 10.0));
        tl.effects_lanes.push(AxisTimeline { clips: vec![clip(Effect::Difference {
            pos: [AnimParam::constant(0.0); 3], normal: [AnimParam::constant(1.0), AnimParam::constant(0.0), AnimParam::constant(0.0)],
        }, 0.0, 5.0)] });

        let json = serde_json::to_string_pretty(&tl).unwrap();
        let back: AnimationTimeline = serde_json::from_str(&json).unwrap();
        assert_eq!(tl, back);
    }

    #[test]
    fn old_json_without_new_fields_still_loads_via_serde_default() {
        // Simulates a minimal/older persisted file missing every optional
        // field — must not fail to deserialize, per the #[serde(default)]
        // convention every field here follows.
        let minimal = serde_json::json!({
            "genome_content_hash": 42,
            "duration_s": 10.0,
            "trim_start_s": 0.0,
            "trim_end_s": 10.0,
        });
        let tl: AnimationTimeline = serde_json::from_value(minimal).unwrap();
        assert_eq!(tl.genome_content_hash, 42);
        assert_eq!(tl.delta_t, 1.0);
        assert!(tl.effects_lanes.is_empty());
        assert!(tl.audio.is_none());
        assert!(tl.audio_clips.is_empty());
    }

    #[test]
    fn migrate_audio_track_converts_deprecated_field_into_a_single_audio_clip() {
        let mut tl = AnimationTimeline::new(1);
        tl.audio = Some(AudioTrack {
            file_path: PathBuf::from("/tmp/example.wav"),
            offset_s: 1.5,
            gain_db: -3.0,
            peaks: None,
        });
        tl.migrate_audio_track();
        assert!(tl.audio.is_none(), "the deprecated field must be cleared after migration");
        assert_eq!(tl.audio_clips.len(), 1);
        let c = &tl.audio_clips[0];
        assert_eq!(c.file_path, PathBuf::from("/tmp/example.wav"));
        assert_eq!(c.start_s, 0.0, "a migrated clip starts at the timeline's own t=0");
        assert_eq!(c.offset_s, 1.5, "trim-in preserved from the old offset_s");
        assert_eq!(c.gain_db, -3.0);
    }

    #[test]
    fn migrate_audio_track_is_a_no_op_once_audio_clips_is_populated() {
        // Guards against clobbering real edits on every subsequent load —
        // migration must fire at most once per timeline, ever.
        let mut tl = AnimationTimeline::new(1);
        tl.audio_clips.push(AudioClip {
            file_path: PathBuf::from("/tmp/real_edit.wav"),
            start_s: 3.0, offset_s: 0.0, trim_out_s: f64::INFINITY, source_duration_s: 0.0,
            gain_db: 0.0, peaks: None,
        });
        tl.audio = Some(AudioTrack {
            file_path: PathBuf::from("/tmp/should_be_ignored.wav"),
            offset_s: 0.0, gain_db: 0.0, peaks: None,
        });
        tl.migrate_audio_track();
        assert_eq!(tl.audio_clips.len(), 1, "must not add a second clip from the stale deprecated field");
        assert_eq!(tl.audio_clips[0].file_path, PathBuf::from("/tmp/real_edit.wav"));
    }

    #[test]
    fn old_json_with_only_deprecated_audio_field_migrates_to_one_clip() {
        // A hand-built JSON shaped like a pre-migration file: the OLD
        // "audio" key present, no "audio_clips" key at all.
        let old_shaped = serde_json::json!({
            "genome_content_hash": 7,
            "duration_s": 10.0, "trim_start_s": 0.0, "trim_end_s": 10.0,
            "audio": { "file_path": "/tmp/old_style.wav", "offset_s": 0.25, "gain_db": 1.0 },
        });
        let mut tl: AnimationTimeline = serde_json::from_value(old_shaped).unwrap();
        assert!(tl.audio.is_some(), "deserializes with the old field still populated, before migration runs");
        assert!(tl.audio_clips.is_empty());
        tl.migrate_audio_track();
        assert!(tl.audio.is_none());
        assert_eq!(tl.audio_clips.len(), 1);
        assert_eq!(tl.audio_clips[0].file_path, PathBuf::from("/tmp/old_style.wav"));
        assert_eq!(tl.audio_clips[0].offset_s, 0.25);
    }

    fn audio_clip(file: &str, start_s: f64, offset_s: f64, trim_out_s: f64) -> AudioClip {
        AudioClip { file_path: PathBuf::from(file), start_s, offset_s, trim_out_s, source_duration_s: 0.0, gain_db: 0.0, peaks: None }
    }

    #[test]
    fn resolve_audio_clip_fully_inside_the_render_window() {
        // Clip starts 2s into the render, plays 3s of source starting at
        // its own offset_s=1.0, render window is [0,10).
        let c = audio_clip("a.wav", 2.0, 1.0, 4.0);
        let r = resolve_audio_clip_for_render(&c, 0.0, 10.0).expect("overlaps the window");
        assert_eq!(r.source_seek_s, 1.0, "no window-start compensation needed");
        assert_eq!(r.output_delay_s, 2.0, "delayed by exactly how far into the render it starts");
        assert_eq!(r.play_len_s, 3.0, "trim_out_s(4.0) - offset_s(1.0)");
    }

    #[test]
    fn resolve_audio_clip_that_starts_before_trim_start_seeks_further_into_source() {
        // Clip's timeline start (1.0) is BEFORE the render's trim_start (5.0)
        // by 4 seconds — those 4 seconds of source must be skipped, and the
        // output can't start with a negative delay.
        let c = audio_clip("a.wav", 1.0, 0.0, 20.0);
        let r = resolve_audio_clip_for_render(&c, 5.0, 15.0).expect("still overlaps [5,15) with the tail of its length");
        assert_eq!(r.output_delay_s, 0.0, "delay can never be negative");
        assert_eq!(r.source_seek_s, 4.0, "compensates by seeking 4s further into the source");
        assert_eq!(r.play_len_s, 10.0, "natural_len(20) - consumed(4), capped at the 10s render span");
    }

    #[test]
    fn resolve_audio_clip_entirely_before_or_after_the_window_is_none() {
        let before = audio_clip("a.wav", -10.0, 0.0, 2.0); // ends at -8.0, long before [0,10)
        assert!(resolve_audio_clip_for_render(&before, 0.0, 10.0).is_none());
        let after = audio_clip("a.wav", 20.0, 0.0, 2.0); // starts at 20.0, after [0,10)
        assert!(resolve_audio_clip_for_render(&after, 0.0, 10.0).is_none());
    }

    #[test]
    fn resolve_audio_clip_play_len_never_exceeds_the_render_span() {
        let c = audio_clip("a.wav", -5.0, 0.0, 1000.0); // huge source, starts well before the window
        let r = resolve_audio_clip_for_render(&c, 0.0, 10.0).expect("still overlaps");
        assert!(r.play_len_s <= 10.0, "play_len_s={} must not exceed the 10s render span", r.play_len_s);
    }
}
