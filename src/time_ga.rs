//! Evolve a time formula for one fractal: a genetic search over
//! [`crate::time_program::TimeProgram`] stacks, judged at several depths along
//! the zoom the reel will actually travel.
//!
//! # Why a search at all, and why at several depths
//!
//! `time_explore` enumerates a fixed grid — every animatable target × seven
//! shapes × three amplitudes — which is exhaustive over a small vocabulary.
//! A time PROGRAM has no such vocabulary to enumerate: the space is every DAG
//! the register file can hold, with continuous amplitude, frequency and phase,
//! stacked across several channels at once. That is a search problem, and the
//! project already owns a genetic programming implementation for exactly this
//! shape of object (`genome::random_program`/`mutate_program`/
//! `crossover_program`), so this module is mostly a fitness function.
//!
//! The depths are the part that is specific to video. A modulation that visibly
//! transforms the whole set at zoom 1 can be completely invisible at zoom 1e9,
//! where the frame shows a region so small that moving the Julia constant by
//! 0.01 changes nothing you can see — and the reverse happens too, a modulation
//! that is a gentle sway up close tearing the frame apart deep down. A single
//! sample view cannot tell those apart, so every individual is judged at several
//! points along the zoom and scored by its **worst** one.
//!
//! Worst rather than mean, for the reason this codebase has already paid to
//! learn once: an average is dragged up by the good samples and hides exactly
//! the moment the eye is drawn to. `time_explore::ClipStats` says the same thing
//! about frames within a clip; this says it about depths within a shot.
//!
//! # The budget
//!
//! Deep frames are expensive — past zoom ~4096 even a 128px probe is on the
//! CPU/f64 path — so the search is funnelled the way `video_zoom_explore`'s
//! `cheap_funnel` is:
//!
//! 1. The four free gates in `time_program` reject most random programs with no
//!    rendering at all.
//! 2. Survivors get a CHEAP pass: one depth, few frames, small.
//! 3. Only the top handful get the FULL pass: several depths, more frames.
//!
//! The full pass runs every generation rather than once at the end, so
//! behaviour-at-depth feeds back into selection instead of merely filtering the
//! result.

use rand::Rng;

use crate::config::Config;
use crate::genome::Genome;
use crate::time_explore::{score_clip, ClipStats, TimeExploreOpts};
use crate::time_program::TimeProgram;
use crate::video_export::{lerp_view, time_frames, CapturedView, View};

/// Probe geometry for one evaluation tier.
#[derive(Clone, Copy, Debug)]
pub struct Tier {
    pub w: u32,
    pub h: u32,
    pub frames: u32,
    /// How many points along the zoom get sampled.
    pub depths: usize,
    /// Whether to apply the spatial noise gate at this tier.
    ///
    /// Off for the cheap tier, and this is not a shortcut — it is the one gate
    /// of the five that is a function of SAMPLING rather than of the animation.
    /// `fitness::noise_tile_fraction` tiles the frame 4x4 and asks how many
    /// tiles look like dither, and `time_explore::MAX_CLIP_NOISE` (0.15) was
    /// calibrated against 192x144 probes. Measured at 96x72 on a real genome:
    /// six of twelve individuals were rejected as "noise" on a shallow shot
    /// where amplitude cannot be the problem — the small probe aliases fine
    /// detail into exactly the pattern the gate exists to catch. The other four
    /// gates are luminance statistics that survive downsampling, so they stay on.
    pub apply_noise_gate: bool,
}

#[derive(Clone, Debug)]
pub struct TimeGaOpts {
    pub population: usize,
    pub generations: usize,
    /// Kept unchanged (and unre-evaluated) into the next generation.
    pub elites: usize,
    /// Simultaneous channels an individual may carry.
    pub max_channels: usize,
    /// How many of the population get the full multi-depth pass each generation.
    pub finalists: usize,
    pub cheap: Tier,
    pub full: Tier,
    pub fps: u32,
    pub angle_coloring: bool,
    /// Amplitude range a random or mutated channel draws from, sampled
    /// LOG-uniformly.
    ///
    /// Log-uniform because the useful amplitude is not known to within an order
    /// of magnitude and differs per target and per depth: a uniform draw over
    /// (0.01, 0.35) puts 97% of its mass above 0.01, and on a real run the
    /// rejections were dominated by "incoherent" and "flash" — both symptoms of
    /// too much amplitude. In log space every decade gets equal attention.
    pub amp_range: (f32, f32),
    /// The five clip gates. Shared with `time_explore` so both searches agree on
    /// what "an animation" means.
    pub clip: TimeExploreOpts,
    /// Frames in the clip this search is a proxy for — the DELIVERED video, not
    /// the short probes. It sets how deep the formula can still animate
    /// smoothly (`time_program::animatable_zoom_limit`), and using the probe's
    /// own frame count would let the search approve depths the finished video
    /// cannot hold.
    pub clip_frames: u32,
    pub seed: u64,
}

impl Default for TimeGaOpts {
    fn default() -> Self {
        TimeGaOpts {
            population: 24,
            generations: 8,
            elites: 4,
            max_channels: 2,
            finalists: 6,
            // The cheap tier ranks; the full tier decides. The full tier's
            // width is NOT free to choose: 192x144 is what MAX_CLIP_NOISE was
            // calibrated at, and a verdict taken at another resolution is a
            // verdict about the probe.
            cheap: Tier { w: 128, h: 96, frames: 8, depths: 1, apply_noise_gate: false },
            full: Tier { w: 192, h: 144, frames: 12, depths: 3, apply_noise_gate: true },
            fps: 24,
            angle_coloring: false,
            // Upper bound well under 1: `time_explore`'s own amplitude sweep
            // found the useful window is the LARGEST amplitude that still reads
            // as a continuous morph, and that the top of the range is where
            // clips stop being animations and start being cuts. The bottom is
            // low enough to cover a deep zoom, where the visible region spans
            // far less of the parameter plane than at the establishing shot.
            amp_range: (1.0e-4, 0.35),
            clip: TimeExploreOpts::default(),
            clip_frames: crate::video_export::DEFAULT_TIME_FRAMES,
            seed: 0,
        }
    }
}

/// One candidate time formula: a stack of channels and how it scored.
#[derive(Clone, Debug)]
pub struct Individual {
    pub progs: Vec<TimeProgram>,
    /// Worst per-depth compression score, or 0.0 when a gate rejected it.
    pub score: f64,
    /// Which sampled depth was the worst, and what it measured there.
    pub worst_depth: usize,
    pub stats: ClipStats,
    pub rejected: Option<&'static str>,
    /// Which tier last scored it — a cheap score and a full score are not
    /// directly comparable, and confusing them would let a cheaply-scored
    /// individual outrank a properly-scored one.
    pub tier_depths: usize,
}

impl Individual {
    pub fn passed(&self) -> bool {
        self.rejected.is_none() && self.score > 0.0
    }

    /// `loops` only if EVERY channel does — one one-shot channel is enough to
    /// leave the clip somewhere other than where it started.
    pub fn loops(&self) -> bool {
        !self.progs.is_empty() && self.progs.iter().all(|p| p.profile().loops)
    }

    pub fn label(&self) -> String {
        self.progs.iter().map(|p| p.label()).collect::<Vec<_>>().join("  +  ")
    }
}

/// Sample views along the ANIMATABLE part of the shot, from shallow to deep.
///
/// Samples at `(i + 0.5) / n` rather than at the endpoints: the opening frame is
/// the cheapest to render and the least representative of the shot (it is a wide
/// establishing view of the whole set), and the closing frame is a single
/// extreme. Interior samples describe the zoom the audience actually watches.
///
/// The range stops at `time_program::animatable_zoom_limit()`. Past that depth
/// one ULP of an f32 genome scalar is already more than a frame wide, so the
/// modulation cannot be an animation at any amplitude — and since an individual
/// is scored by its WORST depth, including one impossible sample makes every
/// individual fail. Measured before this clamp existed: a 41-doubling shot
/// returned `0 winners · 5 flash · 4 static` at every generation, which is
/// exactly what "too big and too small with nothing in between" looks like.
pub fn depth_views(
    start: &CapturedView, end: &CapturedView, n: usize, clip_frames: u32,
) -> Vec<View> {
    let n = n.max(1);
    let limit = crate::time_program::animatable_zoom_limit(clip_frames);
    // Where along the shot the limit falls, in [0,1]. Zoom interpolates
    // geometrically (`lerp_view`), so this is a ratio of logs.
    let span = (end.zoom / start.zoom).log2();
    let t_max = if end.zoom <= limit || span <= 0.0 {
        1.0
    } else {
        ((limit / start.zoom).log2() / span).clamp(0.0, 1.0)
    };
    (0..n)
        .map(|i| lerp_view(start, end, t_max * (i as f64 + 0.5) / n as f64))
        .collect()
}

/// The fraction of a shot over which the formula can still animate, and the
/// zoom where that stops.
///
/// Reported per reel because it is the single most surprising thing about
/// combining a time axis with a deep zoom: past ~1.7e7 the camera keeps going
/// and the formula necessarily holds still, and a viewer wants to know that was
/// a limit rather than a bug.
pub fn animatable_span(
    start: &CapturedView, end: &CapturedView, clip_frames: u32,
) -> (f64, f64) {
    let limit = crate::time_program::animatable_zoom_limit(clip_frames);
    let reachable = end.zoom.min(limit).max(start.zoom);
    let span = (end.zoom / start.zoom).log2();
    let animated = (reachable / start.zoom).log2();
    (if span > 0.0 { (animated / span).clamp(0.0, 1.0) } else { 0.0 }, reachable)
}

/// Score one individual at `views`, worst-depth-wins, stopping at the first
/// depth that fails.
///
/// The early stop is not only an optimisation: a rejection at depth 0 is a
/// complete answer, and the deeper renders are the expensive ones.
pub fn evaluate(
    g: &Genome, config: &Config, views: &[View], ind: &mut Individual,
    opts: &TimeGaOpts, tier: &Tier,
) {
    ind.tier_depths = views.len();
    ind.score = 0.0;
    ind.worst_depth = 0;
    ind.stats = ClipStats::default();

    // The free gates first — no rendering, and they reject most random DAGs.
    if ind.progs.is_empty() {
        ind.rejected = Some("no channels");
        return;
    }
    if let Some(why) = ind.progs.iter().find_map(|p| p.profile().rejected) {
        ind.rejected = Some(why);
        return;
    }

    let mut clip_opts = opts.clip.clone();
    clip_opts.probe_w = tier.w;
    clip_opts.probe_h = tier.h;
    clip_opts.frames = tier.frames;
    clip_opts.fps = opts.fps;
    if !tier.apply_noise_gate {
        // 1.0 is the exact upper bound of what `noise_tile_fraction` can
        // return, so the gate provably never fires rather than being loosened
        // to some large-looking number.
        clip_opts.max_noise = 1.0;
    }

    let mut worst = f64::MAX;
    for (i, view) in views.iter().enumerate() {
        let mut probe = g.clone();
        probe.time_prog = ind.progs.clone();
        let frames: Vec<Vec<u8>> = time_frames(
            &probe, config, opts.angle_coloring, view, tier.frames, tier.w, tier.h,
        ).collect();
        let (stats, rejected, score) = score_clip(&frames, &clip_opts);
        if rejected.is_some() {
            ind.rejected = rejected;
            ind.worst_depth = i;
            ind.stats = stats;
            ind.score = 0.0;
            return;
        }
        if score < worst {
            worst = score;
            ind.worst_depth = i;
            ind.stats = stats;
        }
    }
    ind.rejected = None;
    ind.score = if worst == f64::MAX { 0.0 } else { worst };
}

/// Draw an amplitude log-uniformly from `range`.
fn rand_amp(rng: &mut impl Rng, range: (f32, f32)) -> f32 {
    let (lo, hi) = (range.0.max(1e-9), range.1.max(range.0.max(1e-9) * 1.000_001));
    let u: f32 = rng.random();
    lo * (hi / lo).powf(u)
}

/// A fresh individual: 1..=`max_channels` random programs on distinct targets.
pub fn random_individual(
    rng: &mut impl Rng, targets: &[crate::formula::ModTarget], opts: &TimeGaOpts,
) -> Individual {
    let mut progs = Vec::new();
    let want = rng.random_range(1..=opts.max_channels.max(1)).min(targets.len().max(1));
    let mut used: Vec<crate::formula::ModTarget> = Vec::new();
    for _ in 0..want {
        let Some(&target) = targets.iter().find(|t| !used.contains(t)).or(targets.first()) else {
            break;
        };
        used.push(target);
        let amp = rand_amp(rng, opts.amp_range);
        if let Some(tp) = crate::time_program::random(rng, target, amp, true, 32) {
            progs.push(tp);
        }
    }
    Individual {
        progs, score: 0.0, worst_depth: 0, stats: ClipStats::default(),
        rejected: None, tier_depths: 0,
    }
}

/// Mutate one channel in place, keeping it inside the free gates.
///
/// Retries a bounded number of times rather than accepting whatever came out:
/// a mutation that introduces a pole is very common (one `DIV` rewire is
/// enough), and letting it through would spend a real render discovering what
/// the free gates already know.
fn mutate_channel(rng: &mut impl Rng, tp: &mut TimeProgram, opts: &TimeGaOpts) {
    let original = tp.clone();
    for _ in 0..8 {
        *tp = original.clone();
        match rng.random_range(0..10) {
            0..=5 => {
                tp.prog = crate::genome::strip_dead(&crate::genome::mutate_program(
                    &tp.prog, rng,
                    crate::time_program::DEFAULT_MAX_NODES,
                    crate::time_program::DEFAULT_MAX_DEPTH,
                ));
            }
            6 | 7 => {
                // Multiplicative, matching the log-uniform draw: an additive
                // step would be a huge move at 1e-4 and a rounding error at 0.3.
                let (lo, hi) = opts.amp_range;
                tp.amp = (tp.amp * rng.random_range(0.2..=5.0)).clamp(lo, hi);
            }
            8 => {
                let f = crate::time_program::LOOP_FREQS;
                tp.freq = f[rng.random_range(0..f.len())];
            }
            _ => tp.phase = rng.random::<f32>(),
        }
        if tp.profile().passed() {
            return;
        }
    }
    *tp = original;
}

/// Recombine two channel stacks: take each parent's channels with even odds,
/// capped at `max_channels`.
///
/// Crossover happens at the CHANNEL level, not inside a program. A time
/// program's channels are independent — each drives its own scalar — so
/// swapping whole channels recombines meaning, where splicing two graphs
/// together (what `crossover_program` does, and the right thing for a fractal
/// formula whose root is the answer) would mostly produce a third unrelated
/// shape.
fn crossover(rng: &mut impl Rng, a: &Individual, b: &Individual, opts: &TimeGaOpts) -> Vec<TimeProgram> {
    let mut out: Vec<TimeProgram> = Vec::new();
    for tp in a.progs.iter().chain(b.progs.iter()) {
        if out.len() >= opts.max_channels {
            break;
        }
        // One channel per target: two channels on the same scalar just sum, so
        // keeping both spends a slot to change an amplitude.
        if out.iter().any(|o| o.target == tp.target) {
            continue;
        }
        if rng.random_bool(0.5) {
            out.push(tp.clone());
        }
    }
    if out.is_empty() {
        out = if rng.random_bool(0.5) { a.progs.clone() } else { b.progs.clone() };
    }
    out
}

/// Rank the population: fully-scored survivors first, then provisional ones,
/// then rejects — each group by score.
///
/// A cheap score and a full score are NOT comparable, and mixing them is a real
/// bug rather than a theoretical one. Measured on a real run: the best score
/// reported went 0.0346 → 0.0306 between generations, and the population was
/// re-evaluated from scratch each time. The cause was a cheaply-scored
/// individual — one depth, eight frames — outranking properly-scored ones on a
/// number that means something different, taking an elite slot, and then being
/// re-scored cheaply again next generation instead of being carried.
///
/// So the cheap pass gets exactly one job, choosing who is worth the deep
/// renders, and the ranking that decides elites and the reported best is over
/// full scores only.
pub fn rank(pop: &mut [Individual], full_depths: usize) {
    pop.sort_by(|a, b| {
        let tier = |i: &Individual| (i.passed(), i.tier_depths >= full_depths);
        tier(b).cmp(&tier(a))
            .then(b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal))
    });
}

/// The individuals worth carrying into the next generation unchanged.
///
/// Only ones that passed AND were judged at full depth — a provisional score is
/// not a licence to skip re-evaluation. Elitism on a population where nothing
/// qualifies freezes `n` arbitrary rejects in place forever; on a real run the
/// same rejected formula was still "best" at generation 3 and the search had
/// stopped generating anything new. `pop` must already be [`rank`]ed.
pub fn survivor_elites(pop: &[Individual], n: usize, full_depths: usize) -> Vec<Individual> {
    pop.iter()
        .filter(|i| i.passed() && i.tier_depths >= full_depths)
        .take(n)
        .cloned()
        .collect()
}

fn tournament<'a>(rng: &mut impl Rng, pop: &'a [Individual]) -> &'a Individual {
    let a = &pop[rng.random_range(0..pop.len())];
    let b = &pop[rng.random_range(0..pop.len())];
    if a.score >= b.score { a } else { b }
}

/// What one generation did, for the caller's progress display.
///
/// Beyond the one-line summary (`best`/`passed`/`evaluated`), this exists to
/// answer "why isn't it converging" without re-running anything: the funnel has
/// two tiers and a population-wide pass count conflates them, so a search that
/// clears the cheap gates every generation and dies at the SAME full-tier gate
/// every time looks identical, from `passed` alone, to one that is making real
/// progress and losing it to bad luck. The breakdowns below are read from
/// individuals that were actually (re-)evaluated THIS generation, not the whole
/// population — carried-over elites already explain themselves via `best_label`.
#[derive(Clone, Debug)]
pub struct GenReport {
    pub generation: usize,
    /// Score of the population's true full-tier winner, or 0.0 if none exists
    /// yet. Distinct from `best_label`, which always names SOMETHING.
    pub best: f64,
    /// Whole-population count of `Individual::passed()`, cheap or full tier —
    /// the same number the original one-line log reported.
    pub passed: usize,
    pub evaluated: usize,
    /// Label of the population's current leader: a true winner if one exists,
    /// otherwise `time_ga::best_effort`'s pick — so this is never empty once
    /// the population holds anything at all.
    pub best_label: String,
    /// Whether `best_label` is a real winner (`Individual::passed()`) or just
    /// the closest-to-passing reject.
    pub best_passed: bool,
    /// Why `best_label`'s individual was rejected, if it was.
    pub best_rejected: Option<&'static str>,
    /// New cheap-tier evaluations this generation, by rejection reason
    /// (omits gate names that rejected nobody).
    pub cheap_rejected: Vec<(&'static str, usize)>,
    /// New full-tier evaluations this generation (i.e. this generation's
    /// finalists) and how many of them passed.
    pub full_evaluated: usize,
    pub full_passed: usize,
    /// Full-tier rejections this generation, by reason.
    pub full_rejected: Vec<(&'static str, usize)>,
    /// Full-tier rejections this generation, by WHICH sampled depth caused
    /// them (index into the depth samples, shallow to deep) — the single most
    /// useful field for "why": a bottleneck concentrated at the deepest index
    /// every generation is a real depth limit, one spread evenly is a noisier,
    /// more tractable amplitude problem.
    pub full_rejected_by_depth: Vec<(usize, usize)>,
    /// Distinct `ModTarget`s present anywhere in the current population — a
    /// search stuck on one or two targets generation after generation is
    /// failing to explore, not failing to find an answer.
    pub unique_targets: usize,
}

/// The best individual actually fit to ship: one that passed AND was judged at
/// full depth.
///
/// A cheap-tier survivor is not a winner. It cleared the gates at ONE sampled
/// depth out of several, and the whole reason for judging at several is that a
/// modulation which looks right at one zoom can be invisible or violent at
/// another. Measured on a real shot: five individuals passed the cheap tier
/// every generation while none survived the full one, and taking the cheap
/// leader would have shipped an animation that was never tested where it fails.
pub fn best_shippable(pop: &[Individual], full_depths: usize) -> Option<&Individual> {
    pop.iter().find(|i| i.passed() && i.tier_depths >= full_depths)
}

/// A rough "closeness to acceptable" ordering for an individual that FAILED
/// the gates, built from the same `ClipStats` the gates already computed —
/// so choosing a fallback costs no extra rendering.
///
/// Not comparable to `Individual::score` (a compression ratio, and zero for
/// every reject): this exists only to rank rejects against each other. The
/// weights are unballasted by calibration on purpose — the only claim made is
/// an ordering, not a threshold, so there is nothing to calibrate against.
fn reject_quality(stats: &ClipStats) -> f64 {
    stats.min_coherence as f64
        + (stats.mean_change as f64 / 255.0).min(1.0)
        - stats.max_noise as f64
        - stats.longest_still_run as f64
        - (stats.max_level_jump as f64 / 255.0)
}

/// The individual to ship when nothing passed every gate at full depth.
///
/// A reel with NO time formula is worse than one with an imperfect one: the
/// review GUI is a human looking at every clip anyway, and "re-roll" exists
/// precisely for a shot the search didn't nail on the first try. So this
/// only returns `None` when `pop` has nothing to offer at all — every
/// individual `run` produces carries at least one channel, so in practice
/// that means an empty population.
///
/// Prefers individuals judged at full depth (their stats describe the actual
/// shot, not a cheap proxy for it) and falls back to the best cheap-tier
/// attempt only if nothing ever reached full depth — a small population or
/// generation budget can leave it that way.
pub fn best_effort(pop: &[Individual], full_depths: usize) -> Option<&Individual> {
    if let Some(win) = best_shippable(pop, full_depths) {
        return Some(win);
    }
    let by_quality = |a: &&Individual, b: &&Individual| {
        reject_quality(&a.stats).partial_cmp(&reject_quality(&b.stats))
            .unwrap_or(std::cmp::Ordering::Equal)
    };
    pop.iter()
        .filter(|i| !i.progs.is_empty() && i.tier_depths >= full_depths)
        .max_by(by_quality)
        .or_else(|| pop.iter().filter(|i| !i.progs.is_empty()).max_by(by_quality))
}

/// Evolve a time formula for `g` along the shot `start → end`.
///
/// Returns the population ranked best-first. Rejected individuals are kept (with
/// score 0 and their reason) so a run that finds nothing can still say what it
/// tried and why each attempt failed — the zoom search once shipped a bare
/// "0 winners" and it cost real debugging time.
pub fn run(
    g: &Genome, config: &Config, start: &CapturedView, end: &CapturedView,
    opts: &TimeGaOpts, on_generation: &dyn Fn(&GenReport),
) -> Vec<Individual> {
    use rand::SeedableRng;
    let mut rng = if opts.seed != 0 {
        rand::rngs::StdRng::seed_from_u64(opts.seed)
    } else {
        rand::rngs::StdRng::from_rng(&mut rand::rng())
    };

    // Defensive: `enumerate_targets` always offers at least Phoenix and
    // Bailout, so this is unreachable today — but `random_individual` would
    // silently produce empty stacks if that ever changed, and an empty stack is
    // rejected rather than diagnosed.
    let targets = crate::time_explore::enumerate_targets(g);
    if targets.is_empty() {
        return Vec::new();
    }
    let cheap_views = depth_views(start, end, opts.cheap.depths, opts.clip_frames);
    let full_views = depth_views(start, end, opts.full.depths, opts.clip_frames);

    let mut pop: Vec<Individual> = (0..opts.population.max(2))
        .map(|_| random_individual(&mut rng, &targets, opts))
        .collect();

    for generation in 0..opts.generations.max(1) {
        let mut evaluated = 0usize;
        let mut cheap_rejected: std::collections::BTreeMap<&'static str, usize> = Default::default();
        for ind in pop.iter_mut() {
            // An elite carried over already holds a full-tier score; re-scoring
            // it cheaply would throw that away and make it look worse than a
            // newcomer that has only ever been cheaply scored.
            if ind.tier_depths >= opts.full.depths {
                continue;
            }
            evaluate(g, config, &cheap_views, ind, opts, &opts.cheap);
            evaluated += 1;
            if let Some(why) = ind.rejected {
                *cheap_rejected.entry(why).or_insert(0) += 1;
            }
        }
        // Choosing finalists is the ONE thing the cheap scores are used for.
        pop.sort_by(|a, b| {
            b.passed().cmp(&a.passed())
                .then(b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal))
        });

        // The funnel: only what already looks good cheaply earns the deep
        // renders, and its full score then replaces the cheap one.
        let mut full_evaluated = 0usize;
        let mut full_passed = 0usize;
        let mut full_rejected: std::collections::BTreeMap<&'static str, usize> = Default::default();
        let mut full_rejected_by_depth: std::collections::BTreeMap<usize, usize> = Default::default();
        for ind in pop.iter_mut().take(opts.finalists).filter(|i| i.tier_depths < opts.full.depths) {
            evaluate(g, config, &full_views, ind, opts, &opts.full);
            evaluated += 1;
            full_evaluated += 1;
            if ind.passed() {
                full_passed += 1;
            } else if let Some(why) = ind.rejected {
                *full_rejected.entry(why).or_insert(0) += 1;
                *full_rejected_by_depth.entry(ind.worst_depth).or_insert(0) += 1;
            }
        }
        rank(&mut pop, opts.full.depths);

        let unique_targets = {
            let mut seen = std::collections::BTreeSet::new();
            for ind in &pop {
                for p in &ind.progs {
                    seen.insert(format!("{:?}", p.target));
                }
            }
            seen.len()
        };

        let leader = pop.iter().find(|i| i.passed() && i.tier_depths >= opts.full.depths);
        // `best_effort` always names something once the population holds
        // anything at all, so `best_label` is never empty even in a
        // generation where nothing has ever passed.
        let effort = best_effort(&pop, opts.full.depths);
        on_generation(&GenReport {
            generation,
            best: leader.map(|i| i.score).unwrap_or(0.0),
            passed: pop.iter().filter(|i| i.passed()).count(),
            evaluated,
            best_label: effort.map(|i| i.label()).unwrap_or_default(),
            best_passed: leader.is_some(),
            best_rejected: effort.and_then(|i| i.rejected),
            cheap_rejected: cheap_rejected.into_iter().collect(),
            full_evaluated,
            full_passed,
            full_rejected: full_rejected.into_iter().collect(),
            full_rejected_by_depth: full_rejected_by_depth.into_iter().collect(),
            unique_targets,
        });

        if generation + 1 == opts.generations.max(1) {
            break;
        }

        let mut next: Vec<Individual> = survivor_elites(&pop, opts.elites, opts.full.depths);
        while next.len() < pop.len() {
            let mut child = {
                let a = tournament(&mut rng, &pop);
                let b = tournament(&mut rng, &pop);
                Individual {
                    progs: crossover(&mut rng, a, b, opts),
                    score: 0.0, worst_depth: 0, stats: ClipStats::default(),
                    rejected: None, tier_depths: 0,
                }
            };
            for tp in child.progs.iter_mut() {
                if rng.random_bool(0.6) {
                    mutate_channel(&mut rng, tp, opts);
                }
            }
            // Structural mutation: gain or lose a whole channel.
            if rng.random_bool(0.15) && child.progs.len() > 1 {
                let i = rng.random_range(0..child.progs.len());
                child.progs.remove(i);
            } else if rng.random_bool(0.15) && child.progs.len() < opts.max_channels {
                let used: Vec<_> = child.progs.iter().map(|p| p.target).collect();
                if let Some(&target) = targets.iter().find(|t| !used.contains(t)) {
                    let amp = rand_amp(&mut rng, opts.amp_range);
                    if let Some(tp) = crate::time_program::random(&mut rng, target, amp, true, 32) {
                        child.progs.push(tp);
                    }
                }
            }
            if child.progs.is_empty() {
                child = random_individual(&mut rng, &targets, opts);
            }
            next.push(child);
        }
        pop = next;
    }

    rank(&mut pop, opts.full.depths);
    pop
}

/// Write `time_ga_winners.jsonl` (every individual, ranked, with its rejection
/// reason if any) plus a clip for the top `keep` survivors.
///
/// Clips are re-rendered rather than kept from the search: holding every
/// individual's frames would cost hundreds of megabytes, and re-rendering the
/// handful that won is seconds.
pub fn write_manifest(
    out_dir: &std::path::Path, pop: &[Individual], g: &Genome,
    start: &CapturedView, end: &CapturedView, opts: &TimeGaOpts, keep: usize,
) -> std::io::Result<Vec<std::path::PathBuf>> {
    use std::io::Write;
    std::fs::create_dir_all(out_dir)?;

    let mut f = std::fs::File::create(out_dir.join("time_ga_winners.jsonl"))?;
    for (rank, ind) in pop.iter().enumerate() {
        let row = serde_json::json!({
            "rank": rank,
            "score": ind.score,
            "rejected": ind.rejected,
            "loops": ind.loops(),
            "worst_depth": ind.worst_depth,
            "depths_scored": ind.tier_depths,
            "label": ind.label(),
            "stats": {
                "max_noise": ind.stats.max_noise,
                "min_coherence": ind.stats.min_coherence,
                "mean_change": ind.stats.mean_change,
                "longest_still_run": ind.stats.longest_still_run,
                "max_level_jump": ind.stats.max_level_jump,
            },
            "time_prog": ind.progs,
        });
        writeln!(f, "{row}")?;
    }

    // A `.nn` per winner as well as the JSON: it is the artifact everything
    // else in the project consumes — the viewer opens it, the queue renders it.
    let mut out = Vec::new();
    for (rank, ind) in pop.iter().filter(|i| i.passed()).take(keep).enumerate() {
        let mut winner = g.clone();
        winner.time_prog = ind.progs.clone();
        let path = out_dir.join(format!("winner_{rank:02}.nn"));
        if crate::io::save_genome(&winner, &path).is_ok() {
            out.push(path);
        }
    }

    let _ = std::fs::write(
        out_dir.join("time_ga_summary.txt"),
        format!(
            "{}\nshot {:.4e}x → {:.4e}x ({:.1} doublings)\npopulation {} × {} generations, \
             cheap {}x{}@{}f×{}d, full {}x{}@{}f×{}d\n",
            summary(pop), start.zoom, end.zoom, (end.zoom / start.zoom).log2().max(0.0),
            opts.population, opts.generations,
            opts.cheap.w, opts.cheap.h, opts.cheap.frames, opts.cheap.depths,
            opts.full.w, opts.full.h, opts.full.frames, opts.full.depths,
        ),
    );
    Ok(out)
}

/// "3 winners · 12 discontinuous · 6 incoherent · 3 noise" — what a run threw
/// out and why.
///
/// Derived from the reasons themselves rather than from a hardcoded list of
/// gate names, so a new gate cannot silently go unreported and the counts always
/// add up to the population.
pub fn summary(pop: &[Individual]) -> String {
    let winners = pop.iter().filter(|i| i.passed()).count();
    let mut by_reason: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    for i in pop.iter().filter(|i| !i.passed()) {
        *by_reason.entry(i.rejected.unwrap_or("scored zero")).or_insert(0) += 1;
    }
    // Which sampled depth did the damage. Without this a run reports five
    // plausible-looking gate names and no clue that they all happened at the
    // deepest sample.
    let mut by_depth: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
    for i in pop.iter().filter(|i| !i.passed()) {
        *by_depth.entry(i.worst_depth).or_insert(0) += 1;
    }
    let counted: usize = winners + by_reason.values().sum::<usize>();
    debug_assert_eq!(counted, pop.len(), "every individual must be accounted for");
    let mut parts = vec![format!("{winners} winners")];
    parts.extend(by_reason.iter().map(|(why, n)| format!("{n} {why}")));
    let mut out = parts.join(" · ");
    if !by_depth.is_empty() {
        let depths: Vec<String> = by_depth.iter().map(|(d, n)| format!("d{d}:{n}")).collect();
        out.push_str(&format!("  (failed at {})", depths.join(" ")));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formula::{op, ModTarget, OpNode};
    use rand::SeedableRng;

    fn opts() -> TimeGaOpts {
        TimeGaOpts { seed: 7, ..Default::default() }
    }


    fn cv(cx: f64, cy: f64, zoom: f64) -> CapturedView {
        CapturedView { cx, cx_lo: 0.0, cy, cy_lo: 0.0, zoom, aspect: 1.0 }
    }

    fn phasor(target: ModTarget, amp: f32) -> TimeProgram {
        TimeProgram::new(target, vec![OpNode { op: op::C, a: 0, b: 0, kre: 0.0, kim: 0.0 }], amp)
    }

    fn ind(progs: Vec<TimeProgram>) -> Individual {
        Individual { progs, score: 0.0, worst_depth: 0, stats: ClipStats::default(),
                     rejected: None, tier_depths: 0 }
    }

    #[test]
    fn depth_views_walk_from_shallow_to_deep_without_hitting_the_endpoints() {
        let (a, b) = (cv(0.0, 0.0, 1.0), cv(-0.5, 0.25, 1.0e9));
        let v = depth_views(&a, &b, 3, 40);
        assert_eq!(v.len(), 3);
        assert!(v[0].zoom > a.zoom, "the first sample must be inside the shot, not at its start");
        assert!(v[2].zoom < b.zoom, "the last sample must be inside the shot, not at its end");
        assert!(v[0].zoom < v[1].zoom && v[1].zoom < v[2].zoom, "samples must descend in order");
    }

    #[test]
    fn depth_views_stop_where_the_formula_stops_being_able_to_move() {
        // The real measured shot: 0.147 → 3.14e11, 41 doublings, of which only
        // the first ~27 are animatable at all.
        let (a, b) = (cv(0.0, 0.0, 0.147), cv(-0.5, 0.25, 3.14e11));
        let v = depth_views(&a, &b, 3, 40);
        let limit = crate::time_program::animatable_zoom_limit(40);
        assert!(v.iter().all(|x| x.zoom <= limit),
                "deepest sample {:.3e} is past the limit {limit:.3e}", v[2].zoom);
        // The samples still span most of the animatable range — they sit at
        // (i+0.5)/n, so the top 1/2n of it is deliberately not sampled.
        assert!(v[2].zoom / v[0].zoom > 100.0,
                "samples {:.3e}..{:.3e} barely spread", v[0].zoom, v[2].zoom);
        assert!(v[2].zoom > 1.0e4, "deepest sample {:.3e} is too shallow to test depth", v[2].zoom);

        let (frac, reachable) = animatable_span(&a, &b, 40);
        // 41 doublings of shot, of which the first ~20 can animate at 40 frames.
        assert!(frac > 0.4 && frac < 0.6, "expected roughly half, got {frac}");
        assert!((reachable - limit).abs() / limit < 1e-9);
        // A longer clip needs finer steps, so less of the same shot animates.
        let (frac_long, _) = animatable_span(&a, &b, 300);
        assert!(frac_long < frac, "300 frames should animate less of the shot than 40");
    }

    #[test]
    fn a_shot_that_never_reaches_the_limit_is_sampled_end_to_end() {
        let (a, b) = (cv(0.0, 0.0, 1.0), cv(0.0, 0.0, 1.0e5));
        let v = depth_views(&a, &b, 3, 40);
        assert!(v[2].zoom > 1.0e4, "a shallow shot must still sample near its end, got {:.3e}", v[2].zoom);
        let (frac, reachable) = animatable_span(&a, &b, 40);
        assert!((frac - 1.0).abs() < 1e-9, "the whole shot animates, got {frac}");
        assert!((reachable - 1.0e5).abs() < 1.0);
    }

    #[test]
    fn a_shot_that_starts_past_the_limit_reports_nothing_animatable() {
        let (a, b) = (cv(0.0, 0.0, 1.0e9), cv(0.0, 0.0, 1.0e11));
        let (frac, _) = animatable_span(&a, &b, 40);
        assert!(frac < 1e-9, "nothing should be animatable, got {frac}");
    }

    #[test]
    fn depth_views_are_geometric_because_zoom_is() {
        // lerp_view interpolates zoom geometrically, so evenly spaced samples
        // are evenly spaced in DOUBLINGS — which is what "a few depths" means.
        let v = depth_views(&cv(0.0, 0.0, 1.0), &cv(0.0, 0.0, 1.0e12), 3, 40);
        let r1 = v[1].zoom / v[0].zoom;
        let r2 = v[2].zoom / v[1].zoom;
        assert!((r1 / r2 - 1.0).abs() < 0.01, "ratios {r1:.3e} and {r2:.3e} should match");
    }

    #[test]
    fn an_individual_loops_only_if_every_channel_does() {
        let looping = phasor(ModTarget::JuliaC, 0.1);
        let mut one_shot = phasor(ModTarget::Phoenix, 0.1);
        one_shot.prog = vec![OpNode { op: op::Z, a: 0, b: 0, kre: 0.0, kim: 0.0 }];
        assert!(ind(vec![looping.clone()]).loops());
        assert!(!ind(vec![looping, one_shot]).loops(), "one one-shot channel breaks the loop");
        assert!(!ind(vec![]).loops(), "nothing to loop");
    }

    #[test]
    fn crossover_never_puts_two_channels_on_one_scalar() {
        // Two channels aimed at the same target just sum, so keeping both spends
        // a slot to change an amplitude.
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
        let a = ind(vec![phasor(ModTarget::JuliaC, 0.1), phasor(ModTarget::Phoenix, 0.2)]);
        let b = ind(vec![phasor(ModTarget::JuliaC, 0.3), phasor(ModTarget::Bailout, 0.4)]);
        for _ in 0..200 {
            let kids = crossover(&mut rng, &a, &b, &opts());
            assert!(!kids.is_empty(), "crossover must always produce something usable");
            let mut seen: Vec<ModTarget> = Vec::new();
            for k in &kids {
                assert!(!seen.contains(&k.target), "duplicate target {:?}", k.target);
                seen.push(k.target);
            }
            assert!(kids.len() <= opts().max_channels);
        }
    }

    #[test]
    fn mutation_keeps_a_channel_inside_the_free_gates() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let mut tp = phasor(ModTarget::JuliaC, 0.1);
        for _ in 0..300 {
            mutate_channel(&mut rng, &mut tp, &opts());
            let p = tp.profile();
            assert!(p.passed(), "mutation produced a program the gates reject: {p:?} {}", tp.expr());
        }
    }

    #[test]
    fn mutation_actually_changes_things() {
        // The counterpart to the test above: a `mutate_channel` that always
        // reverted would trivially satisfy it while making the search inert.
        let mut rng = rand::rngs::StdRng::seed_from_u64(5);
        let start = phasor(ModTarget::JuliaC, 0.1);
        let mut changed = 0;
        for _ in 0..100 {
            let mut tp = start.clone();
            mutate_channel(&mut rng, &mut tp, &opts());
            if tp != start { changed += 1; }
        }
        assert!(changed > 60, "only {changed}/100 mutations did anything");
    }

    #[test]
    fn amplitude_mutation_stays_inside_the_configured_range() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(9);
        let o = opts();
        let mut tp = phasor(ModTarget::JuliaC, o.amp_range.1);
        for _ in 0..500 {
            mutate_channel(&mut rng, &mut tp, &o);
            assert!(tp.amp >= o.amp_range.0 - 1e-6 && tp.amp <= o.amp_range.1 + 1e-6,
                    "amp {} escaped {:?}", tp.amp, o.amp_range);
        }
    }

    #[test]
    fn a_channel_stack_with_a_bad_program_is_rejected_before_anything_renders() {
        // The point of the free gates: this must not need a config, a view or a
        // single rendered pixel to answer.
        let pole = TimeProgram::new(
            ModTarget::JuliaC,
            vec![
                OpNode { op: op::C, a: 0, b: 0, kre: 0.0, kim: 0.0 },
                OpNode { op: op::CONST, a: 0, b: 0, kre: 1.0, kim: 0.0 },
                OpNode { op: op::SUB, a: 0, b: 1, kre: 0.0, kim: 0.0 },
                OpNode { op: op::RECIP, a: 2, b: 0, kre: 0.0, kim: 0.0 },
            ],
            0.1,
        );
        assert_eq!(pole.profile().rejected, Some("discontinuous"));
    }

    #[test]
    fn the_deciding_tier_probes_at_the_resolution_the_noise_gate_was_calibrated_for() {
        // MAX_CLIP_NOISE (0.15) is a property of 192x144 sampling. Probing the
        // deciding tier at anything smaller turns the gate into a verdict about
        // the probe: measured at 96x72, six of twelve individuals were thrown
        // out as "noise" on a shot where amplitude could not be the cause.
        let d = TimeGaOpts::default();
        assert!(d.full.apply_noise_gate, "the deciding tier must apply it");
        assert!(!d.cheap.apply_noise_gate, "the ranking tier must not");
        assert_eq!((d.full.w, d.full.h), (192, 144),
                   "changing this without recalibrating MAX_CLIP_NOISE breaks the gate");
        assert!(d.cheap.w * d.cheap.h < d.full.w * d.full.h, "the cheap tier must be cheaper");
    }

    #[test]
    fn amplitudes_are_drawn_across_decades_not_bunched_at_the_top() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(17);
        let range = (1.0e-4f32, 0.35f32);
        let draws: Vec<f32> = (0..2000).map(|_| rand_amp(&mut rng, range)).collect();
        assert!(draws.iter().all(|&a| a >= range.0 && a <= range.1 * 1.001), "out of range");
        // A uniform draw would put ~97% above 0.01. Log-uniform should put
        // roughly a third of the mass in each decade.
        let below = draws.iter().filter(|&&a| a < 0.01).count();
        assert!(below > 400 && below < 1600,
                "only {below}/2000 below 0.01 — the draw is not log-uniform");
    }

    #[test]
    fn a_provisional_score_never_outranks_a_real_one() {
        // The measured bug: a cheaply-scored individual (one depth, eight
        // frames) outranked properly-scored ones on a number that means
        // something different, took an elite slot, and was then re-scored
        // cheaply again — so the reported best went DOWN between generations
        // and the population was re-evaluated from scratch every time.
        let mk = |score: f64, depths: usize, rejected| {
            let mut i = ind(vec![phasor(ModTarget::JuliaC, 0.1)]);
            i.score = score;
            i.tier_depths = depths;
            i.rejected = rejected;
            i
        };
        let mut pop = vec![
            mk(0.90, 1, None),          // provisional, and flattered by it
            mk(0.30, 3, None),          // properly scored
            mk(0.00, 3, Some("noise")), // rejected
            mk(0.40, 3, None),          // properly scored, better
        ];
        rank(&mut pop, 3);
        assert_eq!(pop[0].score, 0.40, "the best FULL score must lead");
        assert_eq!(pop[1].score, 0.30);
        assert_eq!(pop[2].score, 0.90, "the provisional one ranks below every full score");
        assert!(pop[3].rejected.is_some(), "rejects last");
        // And it must not take an elite slot it has not earned.
        let e = survivor_elites(&pop, 3, 3);
        assert_eq!(e.len(), 2, "only the two judged at full depth");
        assert!(e.iter().all(|i| i.tier_depths >= 3));
    }

    #[test]
    fn elitism_never_carries_a_rejected_individual() {
        // On a real run where nothing passed, the same rejected formula was
        // still "best" at generation 3: four elite slots held four arbitrary
        // rejects, and because a full-tier score is not re-evaluated they were
        // never displaced. The search had stopped.
        let good = |score: f64| {
            let mut i = ind(vec![phasor(ModTarget::JuliaC, 0.1)]);
            i.score = score;
            i.tier_depths = 3;
            i
        };
        let bad = |why| {
            let mut i = ind(vec![phasor(ModTarget::Phoenix, 0.1)]);
            i.rejected = Some(why);
            i
        };
        assert_eq!(survivor_elites(&[bad("noise"), bad("flash")], 4, 3).len(), 0,
                   "a population with no survivors must contribute no elites");
        let mixed = [good(0.5), good(0.4), bad("noise"), bad("flash")];
        let e = survivor_elites(&mixed, 4, 3);
        assert_eq!(e.len(), 2, "only the two that passed");
        assert!(e.iter().all(|i| i.passed()));
        assert_eq!(survivor_elites(&mixed, 1, 3).len(), 1, "the cap still applies");
    }

    #[test]
    fn a_cheap_tier_survivor_is_not_shippable() {
        // Measured: five individuals passed the cheap tier every generation
        // while none survived the full one. Shipping the cheap leader would
        // mean an animation validated at one depth out of three — and the whole
        // point of several depths is that one is not enough.
        let mk = |score: f64, depths: usize| {
            let mut i = ind(vec![phasor(ModTarget::JuliaC, 0.1)]);
            i.score = score;
            i.tier_depths = depths;
            i
        };
        assert!(best_shippable(&[mk(0.9, 1), mk(0.8, 1)], 3).is_none(),
                "cheap survivors must not be shippable");
        let pop = [mk(0.9, 3), mk(0.8, 1)];
        assert_eq!(best_shippable(&pop, 3).map(|i| i.score), Some(0.9));
        assert!(best_shippable(&[], 3).is_none());
    }

    #[test]
    fn best_effort_prefers_a_real_winner_when_one_exists() {
        let mut winner = ind(vec![phasor(ModTarget::JuliaC, 0.1)]);
        winner.score = 0.7;
        winner.tier_depths = 3;
        let mut reject = ind(vec![phasor(ModTarget::Phoenix, 0.1)]);
        reject.rejected = Some("flash");
        reject.tier_depths = 3;
        let pop = [reject, winner];
        assert_eq!(best_effort(&pop, 3).map(|i| i.score), Some(0.7));
    }

    #[test]
    fn best_effort_never_returns_none_when_the_population_is_nonempty() {
        // Nothing passed anywhere — every reel this batch touches must still
        // carry SOME time formula rather than falling back to a plain zoom.
        let mk = |why, depths| {
            let mut i = ind(vec![phasor(ModTarget::JuliaC, 0.1)]);
            i.rejected = Some(why);
            i.tier_depths = depths;
            i
        };
        let pop = [mk("noise", 3), mk("flash", 3), mk("incoherent", 1)];
        assert!(best_effort(&pop, 3).is_some());
        assert!(best_effort(&[], 3).is_none(), "an empty population has nothing to offer");
    }

    #[test]
    fn best_effort_prefers_full_depth_rejects_over_cheap_only_ones() {
        let mut cheap = ind(vec![phasor(ModTarget::JuliaC, 0.1)]);
        cheap.rejected = Some("noise");
        cheap.tier_depths = 1;
        cheap.stats.min_coherence = 0.99; // would win on raw quality alone
        let mut full = ind(vec![phasor(ModTarget::Phoenix, 0.1)]);
        full.rejected = Some("flash");
        full.tier_depths = 3;
        full.stats.min_coherence = 0.2;
        let pop = [cheap, full];
        assert_eq!(best_effort(&pop, 3).map(|i| i.stats.min_coherence), Some(0.2),
                   "a full-depth reject describes the real shot; a cheap one does not");
    }

    #[test]
    fn best_effort_picks_the_least_bad_reject_by_clip_stats() {
        let mut noisy = ind(vec![phasor(ModTarget::JuliaC, 0.1)]);
        noisy.rejected = Some("noise");
        noisy.tier_depths = 3;
        noisy.stats = ClipStats { max_noise: 0.9, min_coherence: 0.5, mean_change: 20.0,
                                   ..ClipStats::default() };
        let mut mild = ind(vec![phasor(ModTarget::Phoenix, 0.1)]);
        mild.rejected = Some("incoherent");
        mild.tier_depths = 3;
        mild.stats = ClipStats { max_noise: 0.05, min_coherence: 0.4, mean_change: 15.0,
                                  ..ClipStats::default() };
        let pop = [noisy, mild];
        assert_eq!(best_effort(&pop, 3).map(|i| i.progs[0].target), Some(ModTarget::Phoenix));
    }

    #[test]
    fn summary_accounts_for_every_individual() {
        let mut pop = vec![ind(vec![phasor(ModTarget::JuliaC, 0.1)]); 5];
        pop[0].score = 0.4;
        pop[1].score = 0.3;
        pop[2].rejected = Some("discontinuous");
        pop[3].rejected = Some("incoherent");
        pop[4].rejected = Some("discontinuous");
        let s = summary(&pop);
        assert!(s.contains("2 winners"), "{s}");
        assert!(s.contains("2 discontinuous"), "{s}");
        assert!(s.contains("1 incoherent"), "{s}");
        assert!(s.contains("failed at d"), "the summary must say WHICH depth: {s}");
    }

    #[test]
    fn every_genome_offers_something_to_animate() {
        // The invariant `random_individual` depends on: with no targets it
        // would build empty channel stacks, which `evaluate` rejects — a search
        // that silently produced nothing rather than saying why. Phoenix and
        // bailout exist on every genome, so this holds even for an empty one.
        for g in [Genome::default(), mandelbrot()] {
            assert!(!crate::time_explore::enumerate_targets(&g).is_empty());
        }
    }

    fn test_config() -> crate::config::Config {
        use crate::config::{
            Config, DedupConfig, MassExtinctionConfig, OptimizationConfig, OutputConfig,
            RenderingConfig,
        };
        #[cfg(feature = "wgpu-backend")]
        crate::render_gpu::init_gpu();
        Config {
            dedup: DedupConfig::default(),
            mass_extinction: MassExtinctionConfig::default(),
            rendering: RenderingConfig {
                default_width: 800, default_height: 800, max_iter: 120, bailout: 4.0,
                colormap: "turbo".into(),
                view_x_min: -2.0, view_x_max: 2.0, view_y_min: -2.0, view_y_max: 2.0,
            },
            optimization: OptimizationConfig::default(),
            output: OutputConfig {
                save_dir: "./fractals".into(), population_dir: "./populations".into(),
                ..Default::default()
            },
        }
    }

    fn mandelbrot() -> Genome {
        let mut g = Genome::default();
        g.program = vec![
            OpNode { op: op::Z, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::C, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::SQR, a: 0, b: 0, kre: 0.0, kim: 0.0 },
            OpNode { op: op::ADD, a: 2, b: 1, kre: 0.0, kim: 0.0 },
        ];
        g.bailout_radius = 4.0;
        g
    }

    #[test]
    fn a_tiny_run_terminates_and_returns_a_ranked_population() {
        // Deliberately minimal — this checks the loop, the funnel and the
        // ranking, not the search's quality. A default-sized run renders
        // thousands of deep frames and has no place in a unit test.
        let tiny = TimeGaOpts {
            population: 4, generations: 2, elites: 1, finalists: 2, seed: 42,
            cheap: Tier { w: 32, h: 24, frames: 4, depths: 1, apply_noise_gate: false },
            full: Tier { w: 32, h: 24, frames: 4, depths: 2, apply_noise_gate: true },
            ..Default::default()
        };
        let gens = std::cell::Cell::new(0usize);
        let pop = run(&mandelbrot(), &test_config(), &cv(-0.5, 0.0, 1.0), &cv(-0.75, 0.1, 64.0),
                      &tiny, &|_| gens.set(gens.get() + 1));
        assert_eq!(pop.len(), 4, "the population size must be stable across generations");
        assert_eq!(gens.get(), 2, "every generation must report");
        for w in pop.windows(2) {
            assert!(w[0].score >= w[1].score, "population must come back ranked best-first");
        }
        // Whatever the outcome, every individual carries a verdict: a score or a
        // reason, never neither.
        for ind in &pop {
            assert!(ind.passed() || ind.rejected.is_some() || ind.score == 0.0);
        }
    }

    #[test]
    fn gen_report_always_names_a_best_label_and_accounts_for_every_evaluation() {
        let tiny = TimeGaOpts {
            population: 4, generations: 2, elites: 1, finalists: 2, seed: 42,
            cheap: Tier { w: 32, h: 24, frames: 4, depths: 1, apply_noise_gate: false },
            full: Tier { w: 32, h: 24, frames: 4, depths: 2, apply_noise_gate: true },
            ..Default::default()
        };
        let reports = std::cell::RefCell::new(Vec::new());
        run(&mandelbrot(), &test_config(), &cv(-0.5, 0.0, 1.0), &cv(-0.75, 0.1, 64.0),
            &tiny, &|r| reports.borrow_mut().push(r.clone()));
        let reports = reports.into_inner();
        assert_eq!(reports.len(), 2);
        for r in &reports {
            // The whole point of `best_effort` feeding this field: there is
            // always something to look at, winner or not.
            assert!(!r.best_label.is_empty(), "gen {}: best_label must never be empty", r.generation);
            assert!(r.unique_targets >= 1);
            // A passing best has no rejection reason and vice versa.
            assert_eq!(r.best_passed, r.best_rejected.is_none());
            // Every full-tier evaluation is either counted as passed or shows
            // up in the rejection breakdown — nothing falls through the crack
            // between them.
            let full_reject_total: usize = r.full_rejected.iter().map(|(_, n)| n).sum();
            assert_eq!(r.full_passed + full_reject_total, r.full_evaluated);
            // Every full-tier rejection is attributed to some sampled depth.
            let by_depth_total: usize = r.full_rejected_by_depth.iter().map(|(_, n)| n).sum();
            assert_eq!(by_depth_total, full_reject_total);
        }
    }

    #[test]
    fn elites_keep_their_full_tier_score_instead_of_being_rescored_cheaply() {
        // The funnel's one subtlety: a cheap score and a full score are not
        // comparable, so an elite that earned a full score must not be knocked
        // back down to a cheap one next generation and lose its place.
        let tiny = TimeGaOpts {
            population: 4, generations: 2, elites: 2, finalists: 2, seed: 3,
            cheap: Tier { w: 32, h: 24, frames: 4, depths: 1, apply_noise_gate: false },
            full: Tier { w: 32, h: 24, frames: 4, depths: 2, apply_noise_gate: true },
            ..Default::default()
        };
        let pop = run(&mandelbrot(), &test_config(), &cv(-0.5, 0.0, 1.0), &cv(-0.75, 0.1, 64.0),
                      &tiny, &|_| {});
        let best = &pop[0];
        if best.passed() {
            assert_eq!(best.tier_depths, tiny.full.depths,
                       "the winner must have been judged at full depth, not cheaply");
        }
    }
}
