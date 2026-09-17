//! Standalone multi-metric, diversity-aware fractal explorer CLI.
//!
//! The actual search engine (sweep/drill/scoring/diversity) lives in
//! `nnfractals::explore` — shared with the viewer's "Explore" button. This
//! binary is the batch/headless layer on top: quick comparisons, the 24h
//! "hidden gems" tiled search, and archive curation.
//!
//!   cargo run --release --all-features --bin nnfractals-explorer -- compare [out_dir]
//!   cargo run --release --all-features --bin nnfractals-explorer -- run <method> [n_seeds] [max_rounds] [out_dir]
//!   cargo run --release --all-features --bin nnfractals-explorer -- gems <method|method,...|mixed> [hours] [n_cols] [n_rows] [out_dir]
//!   cargo run --release --all-features --bin nnfractals-explorer -- curate [archive.jsonl] [top_n] [min_score] [min_aesthetic] [min_dist] [out_dir]
//!     method: entropy | edge | gated-entropy | gated-edge

use std::path::{Path, PathBuf};

use nnfractals::aesthetic::AestheticScorer;
use nnfractals::config::Config;
use nnfractals::explore::{
    best_orientation_correlation, debug_sweep_candidates, drill, explore_config, explore_diverse, fingerprint,
    pick_seeds, save_shot, Logger, Metrics, RoundResult, ScoreMethod, EXPLORE_WIDE_RADIUS,
    MIN_DIVERSITY_DISTANCE, SCALES, WIDE_SCALES,
};
use nnfractals::fitness;
use nnfractals::fractal::dihedral_variants;
use nnfractals::genome::Genome;
use nnfractals::io::{self, save_genome};
use nnfractals::known_formulas;
use nnfractals::novelty::NoveltyScorer;
use nnfractals::render_gpu;
use nnfractals::vae_explore::{self, RecursionOpts, SelectBy, ZoneGate};
use nnfractals::vae_score::VaeScorer;
use nnfractals::video_export::{needs_f64, render_complex_field, render_escape_times, View};
use nnfractals::video_zoom_explore;
use nnfractals::time_explore;
use nnfractals::formula::ModShape;
use rand::seq::{IndexedRandom, SliceRandom};
use rand::{Rng, SeedableRng};
use std::process::Command;

const SHOT_RES: u32 = 960;
const FP_PS: usize = 12; // must match explore::SWEEP_RES's fingerprint pooling size

// ── Genome / config setup ───────────────────────────────────────────────

fn build_genome(name: &str) -> Genome {
    let entry = known_formulas::LIBRARY.iter()
        .find(|f| f.name.eq_ignore_ascii_case(name) || f.name.split(' ').next() == Some(name))
        .unwrap_or_else(|| panic!("unknown formula {name:?} — known: {:?}", known_formulas::LIBRARY.iter().map(|f| f.name).collect::<Vec<_>>()));
    Genome { program: (entry.build)(), bailout_radius: 4.0, view_zoom: 1.0, ..Default::default() }
}

fn load_config() -> Config {
    let cfg = Config::load(Path::new("config.toml")).expect("load config.toml");
    explore_config(&cfg)
}

fn timestamp() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

// ── Hidden gems: long-running, resumable, systematically-tiled search ──────
//
// Everything above ("run"/"compare") does ONE wide sweep to pick a handful
// of seeds, then drills each a few rounds — fast, but it can only ever
// surface what a coarse sweep happens to rank highly. A genuinely "hidden"
// gem — small, easy to miss, maybe only visible after a shallow-then-deep
// two-stage descent — needs systematic coverage of the whole boundary, not
// just the winners of one sweep. This mode: tiles the classic view densely
// (only the UPPER half, cy >= 0 — the classic Mandelbrot has EXACT
// conjugate symmetry, f(cx, -cy) mirrors f(cx, cy), confirmed the hard way
// in the "run" mode's mirror-pair bug — so the lower half is provably
// redundant, not just probably), shuffles deterministically for even
// coverage under a time budget, and for each tile: a fast GPU shallow
// drill, and — only for tiles that already look promising — a slower CPU/DD
// deep drill with no zoom ceiling. A find only becomes a saved "gem" if it
// clears a quality bar AND is dihedral-novel against EVERY gem found so far
// this run (not just this batch). Archive + resume cursor persist to disk
// after every gem and periodically anyway, so a multi-hour run surviving a
// restart doesn't lose progress or re-count already-found gems as novel.

const GEMS_SHUFFLE_SEED: u64 = 20260802;
const GEMS_SHALLOW_ROUNDS: usize = 4;
const GEMS_DEEP_ROUNDS: usize = 6;
/// Cheap early-reject before paying for the slow CPU/DD deep phase — well
/// below a typical good shallow winner (~0.5-0.85 in practice) but well
/// above the ~0.0-0.2 exterior/boring range.
const GEMS_SHALLOW_BAR: f32 = 0.35;
/// Final bar for actually saving a gem. Deliberately not higher than the
/// shallow bar: deep drilling routinely finds LOWER-scoring but genuinely
/// rarer structure than the shallow winner that qualified it (same lesson
/// as the wormhole-search work — depth and raw score are not the same
/// axis), and the point of this mode is to surface exactly those.
const GEMS_QUALITY_BAR: f32 = 0.35;
/// Classic Mandelbrot's main cardioid + bulbs + antenna, upper half only
/// (see module doc comment for why the lower half is redundant).
const GEMS_X_MIN: f64 = -2.2;
const GEMS_X_MAX: f64 = 0.8;
const GEMS_Y_MIN: f64 = 0.0;
const GEMS_Y_MAX: f64 = 1.3;

struct Gem {
    cx: f64,
    cy: f64,
    zoom: f64,
    score: f32,
    metrics: Metrics,
    fingerprint: Vec<Vec<f32>>,
    method_name: &'static str,
    // Only populated by `load_gem_archive` (for `cmd_curate`) — `process_tile`'s
    // in-memory archive never needs these, since the path/aesthetic score
    // aren't known until AFTER a gem is accepted and saved by the caller.
    path: String,
    aesthetic_ensemble: Option<f32>,
}

fn gem_to_json(g: &Gem, path: &str) -> serde_json::Value {
    serde_json::json!({
        "event": "gem", "cx": g.cx, "cy": g.cy, "zoom": g.zoom, "score": g.score,
        "entropy": g.metrics.entropy, "edge_density": g.metrics.edge_density, "intricacy": g.metrics.intricacy,
        "fingerprint": g.fingerprint[0], "path": path, "method": g.method_name, "t": timestamp(),
    })
}

fn load_gem_archive(path: &Path) -> Vec<Gem> {
    let Ok(content) = std::fs::read_to_string(path) else { return Vec::new() };
    content.lines().filter_map(|line| {
        let v: serde_json::Value = serde_json::from_str(line).ok()?;
        let fp_base: Vec<f32> = v["fingerprint"].as_array()?.iter().filter_map(|x| x.as_f64()).map(|x| x as f32).collect();
        if fp_base.len() != FP_PS * FP_PS { return None; }
        let method_name = v["method"].as_str().and_then(ScoreMethod::parse).unwrap_or(ScoreMethod::EdgeDensity).name();
        Some(Gem {
            cx: v["cx"].as_f64()?, cy: v["cy"].as_f64()?, zoom: v["zoom"].as_f64()?,
            score: v["score"].as_f64()? as f32,
            metrics: Metrics {
                entropy: v["entropy"].as_f64()? as f32, edge_density: v["edge_density"].as_f64()? as f32,
                intricacy: v["intricacy"].as_f64()? as f32, degenerate: false,
            },
            fingerprint: dihedral_variants(&fp_base, FP_PS),
            method_name,
            path: v["path"].as_str().unwrap_or("").to_string(),
            aesthetic_ensemble: v["aesthetic_ensemble"].as_f64().map(|x| x as f32),
        })
    }).collect()
}

/// Alternate loader for `cmd_curate`, used when its `archive_path` argument
/// is a DIRECTORY (a `cmd_pool` output folder) rather than a `cmd_gems`-style
/// `gems_archive.jsonl` file. Scans for `STEM.nn`/`STEM.png` pairs directly
/// (via `nnfractals::dedup::find_pairs`-equivalent logic) rather than
/// reading `pool_log.jsonl` — that log reflects only the MOST RECENT
/// `cmd_pool` invocation into a given directory prior to its append-mode
/// fix, and even after that fix a directory built up from several separate
/// runs is safer read from what's actually on disk than from a log that
/// could still be partial (manually pruned entries, an interrupted run,
/// etc). `score` is a sentinel 1.0 (always clears any real `min_score`) —
/// unrecoverable per-image from the .nn file alone, and low-value anyway
/// since `cmd_pool` already gated on it before saving. `aesthetic_ensemble`
/// starts `None`; `cmd_curate`'s caller re-scores it fresh right after this
/// returns. `fingerprint` is left empty — `cmd_curate`'s own selection logic
/// never reads `Gem::fingerprint`, only `load_gem_archive`'s caller
/// (`process_tile`) does.
fn load_pool_dir(dir: &Path) -> Vec<Gem> {
    let Ok(rd) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut gems: Vec<Gem> = rd.filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|x| x == "nn"))
        .filter_map(|e| {
            let nn_path = e.path();
            let png_path = nn_path.with_extension("png");
            if !png_path.exists() { return None; }
            let genome = io::load_genome(&nn_path).ok()?;
            Some(Gem {
                cx: genome.view_cx as f64, cy: genome.view_cy as f64, zoom: genome.view_zoom as f64,
                score: 1.0,
                metrics: Metrics { entropy: 0.0, edge_density: 0.0, intricacy: 0.0, degenerate: false },
                fingerprint: Vec::new(),
                method_name: "pool",
                path: png_path.to_string_lossy().to_string(),
                aesthetic_ensemble: None,
            })
        })
        .collect();
    gems.sort_by(|a, b| a.path.cmp(&b.path));
    gems
}

fn load_cursor(path: &Path) -> usize {
    std::fs::read_to_string(path).ok().and_then(|s| s.trim().parse().ok()).unwrap_or(0)
}

fn save_cursor(path: &Path, n: usize) {
    let _ = std::fs::write(path, n.to_string());
}

/// Dense grid over the classic view's upper half, deterministically
/// shuffled — shuffling matters because a time-boxed run that died partway
/// through a naive row-by-row sweep would have covered only the top strip
/// of the set; a shuffled order gets broad coverage no matter when the
/// budget runs out.
fn generate_tiles(n_cols: usize, n_rows: usize) -> Vec<(f64, f64)> {
    let mut tiles = Vec::with_capacity(n_cols * n_rows);
    for row in 0..n_rows {
        for col in 0..n_cols {
            let tx = (col as f64 + 0.5) / n_cols as f64;
            let ty = (row as f64 + 0.5) / n_rows as f64;
            tiles.push((GEMS_X_MIN + tx * (GEMS_X_MAX - GEMS_X_MIN), GEMS_Y_MIN + ty * (GEMS_Y_MAX - GEMS_Y_MIN)));
        }
    }
    let mut rng = rand::rngs::StdRng::seed_from_u64(GEMS_SHUFFLE_SEED);
    tiles.shuffle(&mut rng);
    tiles
}

/// One tile: fast GPU shallow drill, early-reject if it doesn't clear
/// `GEMS_SHALLOW_BAR`, else a slow CPU/DD deep drill continuing from the
/// shallow winner with no zoom ceiling. Returns `Some(gem)` if the best
/// point across BOTH phases clears `GEMS_QUALITY_BAR` and is dihedral-novel
/// against the archive — the caller just saves whatever comes back.
/// `tile_zoom`: `4.0 / tile_span`, set by the caller from its own grid
/// spacing so tiles are contiguous (not a `process_tile` concern).
#[allow(clippy::too_many_arguments)]
fn process_tile(
    genome: &Genome, config: &Config, cx: f64, cy: f64, tile_zoom: f64, tile_id: usize,
    method: ScoreMethod, log: &mut Logger, archive: &[Gem],
) -> Option<Gem> {
    let start = View::new_square(cx, cy, tile_zoom);
    let shallow = drill(genome, config, start, GEMS_SHALLOW_ROUNDS, method, tile_id, log, true, 0);
    let shallow_best = shallow.iter().max_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal))?;
    if shallow_best.score < GEMS_SHALLOW_BAR {
        log.log(&serde_json::json!({"event": "tile", "tile": tile_id, "cx": cx, "cy": cy, "outcome": "shallow_reject", "score": shallow_best.score}));
        return None;
    }

    let deep = drill(genome, config, shallow_best.view.clone(), GEMS_DEEP_ROUNDS, method, tile_id, log, false, GEMS_SHALLOW_ROUNDS);
    let best = shallow.iter().chain(deep.iter())
        .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal))
        .expect("shallow is non-empty — already checked above");

    if best.score < GEMS_QUALITY_BAR {
        log.log(&serde_json::json!({"event": "tile", "tile": tile_id, "cx": cx, "cy": cy, "outcome": "quality_reject", "score": best.score}));
        return None;
    }

    let fp = fingerprint(genome, config, &best.view);
    let min_dist = archive.iter().map(|g| 1.0 - best_orientation_correlation(&fp, &g.fingerprint)).fold(f32::INFINITY, f32::min);
    if !archive.is_empty() && min_dist < MIN_DIVERSITY_DISTANCE {
        log.log(&serde_json::json!({"event": "tile", "tile": tile_id, "cx": cx, "cy": cy, "outcome": "not_novel", "score": best.score, "min_dist": min_dist}));
        return None;
    }

    log.log(&serde_json::json!({"event": "tile", "tile": tile_id, "cx": cx, "cy": cy, "outcome": "gem", "score": best.score, "min_dist": if archive.is_empty() { None } else { Some(min_dist) }}));
    Some(Gem {
        cx: best.view.cx, cy: best.view.cy, zoom: best.view.zoom, score: best.score, metrics: best.metrics,
        fingerprint: fp, method_name: method.name(), path: String::new(), aesthetic_ensemble: None,
    })
}

fn cmd_compare(out_dir: &Path) {
    std::fs::create_dir_all(out_dir).expect("create out_dir");
    let genome = build_genome("Mandelbrot");
    let config = load_config();
    let start = View::new_square(-0.5, 0.0, 1.0);
    let methods = ScoreMethod::ALL;

    println!("{:<15} {:>8} {:>8} {:>10} {:>10} {:>10}  path", "method", "rounds", "score", "cx", "cy", "zoom");
    for method in methods {
        let mut log = Logger::new(&out_dir.join(format!("compare_{}.jsonl", method.name()))).expect("open log");
        let history = drill(&genome, &config, start.clone(), 6, method, 0, &mut log, true, 0);
        let Some(best) = history.iter().max_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal)) else {
            println!("{:<15} {:>8} — no non-degenerate round found", method.name(), 0);
            continue;
        };
        let path = out_dir.join(format!("compare_{}.png", method.name()));
        save_shot(&genome, &config, &best.view, SHOT_RES, &path);
        println!("{:<15} {:>8} {:>8.4} {:>10.6} {:>10.6} {:>10.3e}  {}",
            method.name(), best.round + 1, best.score, best.view.cx, best.view.cy, best.view.zoom, path.display());
    }
}

fn cmd_run(method: ScoreMethod, n_seeds: usize, max_rounds: usize, out_dir: &Path) {
    std::fs::create_dir_all(out_dir).expect("create out_dir");
    let genome = build_genome("Mandelbrot");
    let config = load_config();
    let start = View::new_square(-0.5, 0.0, 1.0);

    let mut log = Logger::new(&out_dir.join("explorer_log.jsonl")).expect("open log");
    log.log(&serde_json::json!({"event": "run_meta", "method": method.name(), "n_seeds": n_seeds, "max_rounds": max_rounds, "genome": "Mandelbrot"}));

    println!("picking {n_seeds} diverse seeds from a wide sweep...");
    let results: Vec<(usize, RoundResult, Vec<Vec<f32>>)> = explore_diverse(&genome, &config, &start, method, n_seeds, max_rounds, &mut log);
    println!("\nselected {} diverse best-shots:", results.len());

    let mut aesthetic = AestheticScorer::new();
    if aesthetic.is_none() {
        println!("(aesthetic_scorer.py sidecar unavailable — skipping aesthetic cross-check)");
    }

    let mut paths = Vec::new();
    for (rank, (seed_id, best, fp)) in results.iter().enumerate() {
        let min_dist: Option<f32> = results.iter().filter(|(sid, ..)| sid != seed_id)
            .map(|(_, _, ofp)| 1.0 - best_orientation_correlation(fp, ofp)).fold(None, |acc, d| Some(acc.map_or(d, |a: f32| a.min(d))));
        let path = out_dir.join(format!("shot_{:02}_seed{}_r{}.png", rank + 1, seed_id, best.round));
        save_shot(&genome, &config, &best.view, SHOT_RES, &path);

        let aesthetic_ensemble = aesthetic.as_mut().and_then(|a| a.score_blocking(path.clone())).map(|s| s.ensemble());

        println!("  [{:>2}] seed={:<3} round={:<2} score={:.4} min_div_dist={:>6} aesthetic={:>6}  cx={:.6} cy={:.6} zoom={:.3e}\n       {}",
            rank + 1, seed_id, best.round, best.score,
            min_dist.map(|v| format!("{v:.4}")).unwrap_or_else(|| "n/a".into()),
            aesthetic_ensemble.map(|v| format!("{v:.3}")).unwrap_or_else(|| "n/a".into()),
            best.view.cx, best.view.cy, best.view.zoom, path.display());

        log.log(&serde_json::json!({
            "event": "selected", "rank": rank + 1, "seed": seed_id, "round": best.round,
            "cx": best.view.cx, "cy": best.view.cy, "zoom": best.view.zoom, "score": best.score,
            "entropy": best.metrics.entropy, "edge_density": best.metrics.edge_density, "intricacy": best.metrics.intricacy,
            "min_diversity_distance": min_dist, "aesthetic_ensemble": aesthetic_ensemble,
            "path": path.display().to_string(),
        }));
        paths.push(path);
    }

    println!("\nlog: {}", out_dir.join("explorer_log.jsonl").display());
    for p in &paths { println!("shot: {}", p.display()); }
}

/// Wide exploration for a TRAINING/CURATION POOL, not a final showcase:
/// seeds are split across `methods` (same reasoning as
/// `explore_diverse_mixed` — one fixed method narrows what's found to one
/// visual family) and drilled, but every result clearing `min_score` is
/// saved — deliberately WITHOUT `select_diverse`'s final filtering. Volume
/// (and even some redundancy) is what a self-supervised embedding wants to
/// train on; throwing away "too similar" candidates here is `cmd_curate`'s
/// job, done LATER against the model trained on this exact pool. Each
/// result is saved as a matching STEM.nn + STEM.png pair in the same
/// directory (`scripts/dedup.py::find_pairs`' and `train_novelty.py`'s
/// expected layout) — PNG rendered from the SAME f32-precision-snapped
/// view the .nn stores, not the raw internal f64 `best.view` (see
/// viewer.rs's `start_explore` doc comment for why: near a chaotic
/// boundary at real max_iter, a single f32 ULP position difference can
/// make the two silently disagree).
#[allow(clippy::too_many_arguments)]
fn cmd_pool(
    formula: &str, methods: &[ScoreMethod], cx: f64, cy: f64, zoom: f64,
    n_seeds: usize, max_rounds: usize, min_score: f32, max_intricacy: f32, min_aesthetic: f32, min_edge_density: f32, out_dir: &Path,
) {
    std::fs::create_dir_all(out_dir).expect("create out_dir");
    let genome = build_genome(formula);
    let config = load_config();
    let view = View::new_square(cx, cy, zoom);
    // Append-mode numbering: a contact-sheet review of a single-reference-view
    // pool showed heavy compositional redundancy (drilling repeatedly
    // converges on sub-crops of the same handful of attractive local
    // regions) — the fix is running `pool` again from a DIFFERENT (cx, cy,
    // zoom) into the SAME out_dir, so seed numbering has to continue past
    // whatever's already there instead of restarting at 0 and overwriting it.
    // Keyed off the max existing stem index (not a file count) so it's safe
    // even if earlier stems were manually pruned, leaving gaps.
    let mut saved = std::fs::read_dir(out_dir)
        .map(|rd| rd.filter_map(|e| e.ok())
            .filter_map(|e| e.path().file_stem().and_then(|s| s.to_str().and_then(|s| s.strip_prefix("pool_")).map(str::to_string)))
            .filter_map(|n| n.parse::<usize>().ok())
            .max().map_or(0, |m| m + 1))
        .unwrap_or(0);
    // `append`, not `new`/truncate — the whole point of the resume-numbering
    // logic just above is to support running `pool` again from a different
    // (cx, cy, zoom) into the SAME out_dir to broaden territory coverage;
    // truncating the log on each invocation would silently discard every
    // earlier run's "saved" metadata (cmd_curate's pool-dir mode reads this
    // back later to know what's in the directory).
    let mut log = Logger::append(&out_dir.join("pool_log.jsonl")).expect("open log");
    log.verbose = false; // per-candidate detail isn't needed for a pool run and gets huge fast
    log.log(&serde_json::json!({
        "event": "run_meta", "formula": formula,
        "methods": methods.iter().map(|m| m.name()).collect::<Vec<_>>(),
        "cx": cx, "cy": cy, "zoom": zoom, "n_seeds": n_seeds, "max_rounds": max_rounds,
        "min_score": min_score, "max_intricacy": max_intricacy, "min_aesthetic": min_aesthetic,
        "min_edge_density": min_edge_density, "resume_from": saved,
    }));
    // Second, independent noise gate: `field_intricacy` alone (direction-
    // reversal density) does NOT cleanly separate "genuine boundary
    // structure" from "looks like static to a human" for every formula —
    // confirmed empirically on Burning Ship, where several candidates well
    // under `max_intricacy` still rendered as visually obvious noise. The
    // aesthetic ensemble (nima/topiq/ap25) is trained to correlate with
    // human visual judgment, which is a fundamentally different — and here,
    // necessary — signal than any structural heuristic.
    let mut aesthetic = AestheticScorer::new();
    if aesthetic.is_none() {
        println!("(aesthetic_scorer.py sidecar unavailable — pool will rely on the intricacy gate alone)");
    }

    let per_method = (n_seeds / methods.len()).max(1);
    let mut seed_id = 0usize;
    for &method in methods {
        println!("method {}: picking {per_method} seeds...", method.name());
        let seeds = pick_seeds(&genome, &config, &view, method, per_method, &mut log, 1.0, SCALES);
        for seed_view in &seeds {
            let history = drill(&genome, &config, seed_view.clone(), max_rounds, method, seed_id, &mut log, true, 0);
            seed_id += 1;
            let Some(best) = history.iter().max_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal)) else { continue };
            if best.score < min_score { continue; }
            // Explicit, method-INDEPENDENT noise gate — see cmd_pool's doc
            // comment. Two of the four scoring methods (entropy, edge) have
            // no intricacy ceiling of their own at all, and the two that do
            // (gated-entropy, gated-edge) use `WORMHOLE_INTRIC_CEIL_HI`
            // (0.40), tuned against Mandelbrot — confirmed empirically NOT
            // strict enough here: a real Burning Ship "gated-entropy" save
            // still had an obviously-noisy region. Applying a formula-
            // agnostic ceiling here, on top of whatever each method already
            // does, is the fix, not re-tuning per-formula in advance.
            if best.metrics.intricacy > max_intricacy {
                log.log(&serde_json::json!({
                    "event": "rejected_noise", "seed": seed_id, "method": method.name(),
                    "score": best.score, "intricacy": best.metrics.intricacy,
                    "cx": best.view.cx, "cy": best.view.cy, "zoom": best.view.zoom,
                }));
                continue;
            }
            // Third gate: a contact-sheet review of an ungated Burning Ship
            // pool found a distinct failure mode intricacy/aesthetic both
            // missed — smooth escape-time gradients and flat near-solid
            // regions with essentially no boundary structure at all (one had
            // intricacy 0.0 AND aesthetic 4.8, since a smooth gradient reads
            // as "clean" to both signals). `edge_density` cleanly separated
            // these (0.03-0.10) from every genuinely structured save
            // (>=0.20) in that same batch, so gate on it directly rather than
            // trusting intricacy/aesthetic to catch a "too little structure"
            // failure they're not measuring.
            if best.metrics.edge_density < min_edge_density {
                log.log(&serde_json::json!({
                    "event": "rejected_flat", "seed": seed_id, "method": method.name(),
                    "score": best.score, "edge_density": best.metrics.edge_density,
                    "cx": best.view.cx, "cy": best.view.cy, "zoom": best.view.zoom,
                }));
                continue;
            }

            let mut g = genome.clone();
            g.view_cx = best.view.cx as f32;
            g.view_cy = best.view.cy as f32;
            g.view_zoom = best.view.zoom as f32;
            let snapped = View::new_square(g.view_cx as f64, g.view_cy as f64, g.view_zoom as f64);

            let stem = format!("pool_{saved:04}");
            let png_path = out_dir.join(format!("{stem}.png"));
            save_shot(&genome, &config, &snapped, 960, &png_path);

            let aesthetic_ensemble = aesthetic.as_mut().and_then(|a| a.score_blocking(png_path.clone())).map(|s| s.ensemble());
            if aesthetic_ensemble.is_some_and(|v| v < min_aesthetic) {
                let _ = std::fs::remove_file(&png_path);
                log.log(&serde_json::json!({
                    "event": "rejected_aesthetic", "seed": seed_id, "method": method.name(),
                    "score": best.score, "intricacy": best.metrics.intricacy, "aesthetic_ensemble": aesthetic_ensemble,
                    "cx": g.view_cx, "cy": g.view_cy, "zoom": g.view_zoom,
                }));
                continue;
            }

            let nn_path = out_dir.join(format!("{stem}.nn"));
            save_genome(&g, &nn_path).expect("save genome");
            log.log(&serde_json::json!({
                "event": "saved", "stem": stem, "method": method.name(), "score": best.score,
                "entropy": best.metrics.entropy, "edge_density": best.metrics.edge_density, "intricacy": best.metrics.intricacy,
                "aesthetic_ensemble": aesthetic_ensemble, "cx": g.view_cx, "cy": g.view_cy, "zoom": g.view_zoom,
            }));
            println!("  [{saved:>4}] ({}) score={:.4} intricacy={:.4} aesthetic={}  cx={:.6} cy={:.6} zoom={:.3e}",
                method.name(), best.score, best.metrics.intricacy,
                aesthetic_ensemble.map(|v| format!("{v:.3}")).unwrap_or_else(|| "n/a".into()),
                g.view_cx, g.view_cy, g.view_zoom);
            saved += 1;
        }
    }
    println!("\nsaved {saved} paired .nn/.png images to {}", out_dir.display());
}

/// Long-running, resumable, systematically-tiled search — see the module
/// doc comment above `process_tile`. `n_cols`/`n_rows` control tile density
/// directly (not seed count — every tile is tried, cheaply, and only
/// promising ones pay for the deep phase); `hours` is a wall-clock budget,
/// checked between tiles, not a hard preemption (a tile in progress always
/// finishes). `methods`: cycled round-robin by tile index — see the module
/// doc comment addition on why a single fixed method systematically
/// converges on one visual family (a real 1791-gem run under "edge" alone
/// leaned heavily toward radiating-starburst compositions, since that's
/// what edge_density rewards almost everywhere) and mixing scoring
/// functions across tiles is what actually diversifies pattern TYPE, not
/// just position — tiles are already shuffled, so round-robin on the
/// shuffled index also means each method lands on a spatially random
/// subset, not e.g. every method confined to one region.
fn cmd_gems(methods: &[ScoreMethod], hours: f64, n_cols: usize, n_rows: usize, out_dir: &Path) {
    std::fs::create_dir_all(out_dir).expect("create out_dir");
    let genome = build_genome("Mandelbrot");
    let config = load_config();

    let archive_path = out_dir.join("gems_archive.jsonl");
    let state_path = out_dir.join("gems_state.json");
    let mut archive = load_gem_archive(&archive_path);
    let start_at = load_cursor(&state_path);
    println!("resuming: {} gems already found, starting from tile {start_at}", archive.len());

    let tiles = generate_tiles(n_cols, n_rows);
    // View's full span = 4.0/zoom — pick zoom so a tile's own view spans
    // one grid cell (contiguous coverage, not gapped or overlapping).
    let tile_span_x = (GEMS_X_MAX - GEMS_X_MIN) / n_cols as f64;
    let tile_zoom = 4.0 / tile_span_x;
    let method_names: Vec<&str> = methods.iter().map(|m| m.name()).collect();
    println!("{} tiles ({n_cols}x{n_rows}), tile_zoom={tile_zoom:.2}, budget={hours}h, methods={method_names:?}", tiles.len());

    let mut log = Logger::append(&out_dir.join("gems_log.jsonl")).expect("open log");
    log.verbose = false; // per-candidate detail would run into the millions of lines over 24h — see module doc comment
    log.log(&serde_json::json!({"event": "run_meta", "methods": method_names, "n_cols": n_cols, "n_rows": n_rows, "hours": hours, "resumed_from": start_at}));

    let mut aesthetic = AestheticScorer::new();
    if aesthetic.is_none() {
        println!("(aesthetic_scorer.py sidecar unavailable — skipping aesthetic cross-check)");
    }

    let t0 = std::time::Instant::now();
    let deadline = std::time::Duration::from_secs_f64(hours * 3600.0);
    let mut processed = start_at;

    for (i, &(cx, cy)) in tiles.iter().enumerate().skip(start_at) {
        if t0.elapsed() >= deadline {
            println!("[{i}/{}] time budget reached ({:.1}h elapsed)", tiles.len(), t0.elapsed().as_secs_f64() / 3600.0);
            break;
        }
        let method = methods[i % methods.len()];
        if let Some(gem) = process_tile(&genome, &config, cx, cy, tile_zoom, i, method, &mut log, &archive) {
            let idx = archive.len();
            let path = out_dir.join(format!("gem_{idx:04}.png"));
            let view = View { cx: gem.cx, cx_lo: 0.0, cy: gem.cy, cy_lo: 0.0, zoom: gem.zoom, aspect: 1.0 };
            save_shot(&genome, &config, &view, SHOT_RES, &path);
            let aesthetic_ensemble = aesthetic.as_mut().and_then(|a| a.score_blocking(path.clone())).map(|s| s.ensemble());
            let mut rec = gem_to_json(&gem, &path.display().to_string());
            rec["aesthetic_ensemble"] = serde_json::json!(aesthetic_ensemble);
            let _ = {
                use std::io::Write;
                std::fs::OpenOptions::new().create(true).append(true).open(&archive_path)
                    .and_then(|mut f| writeln!(f, "{rec}"))
            };
            println!("[{i}/{}] GEM #{idx} ({}): cx={:.6} cy={:.6} zoom={:.3e} score={:.4} aesthetic={}  {}",
                tiles.len(), gem.method_name, gem.cx, gem.cy, gem.zoom, gem.score,
                aesthetic_ensemble.map(|v| format!("{v:.3}")).unwrap_or_else(|| "n/a".into()), path.display());
            archive.push(gem);
        }
        processed = i + 1;
        if processed % 25 == 0 {
            save_cursor(&state_path, processed);
            let elapsed = t0.elapsed().as_secs_f64();
            println!("[{processed}/{}] {:.1}h elapsed, {} gems, {:.1}s/tile avg",
                tiles.len(), elapsed / 3600.0, archive.len(), elapsed / (processed - start_at).max(1) as f64);
        }
    }
    save_cursor(&state_path, processed);
    if processed >= tiles.len() {
        println!("all {} tiles exhausted before the time budget — increase n_cols/n_rows for a finer re-run", tiles.len());
    }
    println!("done: {} gems in {:.1}h, archive at {}", archive.len(), t0.elapsed().as_secs_f64() / 3600.0, archive_path.display());
}

/// Curate a raw gems archive down to a small set of genuine standouts.
/// `cmd_gems`'s own novelty floor (`MIN_DIVERSITY_DISTANCE = 0.3` against
/// an incrementally-growing archive) is deliberately loose — its job is
/// "don't save literal near-repeats," not "only save the very best," and
/// at scale (1791 gems from a real run) that leaves plenty of merely-good,
/// thematically-similar content mixed in with the genuine standouts. This
/// applies a much stricter filter in two stages: (1) BOTH the structural
/// search score AND the independently-validated aesthetic ensemble
/// (nima/topiq/ap25 — the project's own trained "beauty by human standard"
/// model, not the cheap proxy the search itself optimized) must clear a
/// bar, then (2) the same dihedral-aware greedy diversity selection as
/// `select_diverse`, with a stricter distance floor, on whatever survives.
#[allow(clippy::too_many_arguments)]
fn l2_dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
}

/// Curate a raw gems archive down to a small set of genuine standouts.
/// `cmd_gems`'s own novelty floor (`MIN_DIVERSITY_DISTANCE = 0.3` against
/// an incrementally-growing archive) is deliberately loose — its job is
/// "don't save literal near-repeats," not "only save the very best," and
/// at scale that leaves plenty of merely-good, thematically-similar
/// content mixed in with the genuine standouts. This applies a much
/// stricter filter in two stages: (1) BOTH the structural search score AND
/// the independently-validated aesthetic ensemble (nima/topiq/ap25 — the
/// project's own trained "beauty by human standard" model, not the cheap
/// proxy the search itself optimized) must clear a bar, then (2) greedy
/// farthest-point diversity selection — LATENT distance (the novelty
/// model's own 128-d embedding, `novelty::NoveltyScorer::embed_blocking`),
/// not the hand-crafted pixel-pooling fingerprint `select_diverse` uses.
/// Confirmed necessary the hard way: Carl's own read of an earlier
/// pixel-fingerprint-curated set was "very much look alike themselves" —
/// two images can differ a lot in exact pixel LAYOUT (different specific
/// blob positions, which the pooling fingerprint is sensitive to) while
/// still being the same KIND of pattern to a human (e.g. "yet another
/// spiral vortex"), which is a PERCEPTUAL judgment a coarse pixel-pooling
/// distance was never going to capture — that's exactly what a backbone
/// trained on real images (DINOv2) is for. Final selection is RE-RENDERED
/// at `res` (not copied from the archive's smaller cached PNG) since a
/// curated top-N is worth a much higher resolution than the search itself
/// needed.
#[allow(clippy::too_many_arguments)]
fn cmd_curate(archive_path: &Path, top_n: usize, min_score: f32, min_aesthetic: f32, min_dist: f32, res: u32, formula: &str, out_dir: &Path, model: Option<(&Path, &Path)>) {
    std::fs::create_dir_all(out_dir).expect("create out_dir");
    // `cmd_gems` writes one continuously-growing gems_archive.jsonl file;
    // `cmd_pool` writes a directory of STEM.nn/STEM.png pairs plus its own
    // pool_log.jsonl — dispatch on which one `archive_path` actually is
    // rather than needing two separate subcommands for what's otherwise the
    // same selection algorithm.
    let mut archive = if archive_path.is_dir() { load_pool_dir(archive_path) } else { load_gem_archive(archive_path) };
    println!("loaded {} gems from {}", archive.len(), archive_path.display());
    if archive_path.is_dir() {
        // `pool_log.jsonl` may span several `cmd_pool` invocations into this
        // directory (before the append-mode fix, only the LAST one's
        // "saved" events survived at all) — re-score aesthetic fresh for
        // every candidate rather than trust log data that could be partial
        // or stale. Cheap: these PNGs already exist, no re-render needed.
        if let Some(mut aesthetic) = AestheticScorer::new() {
            println!("re-scoring aesthetic for {} pool candidates...", archive.len());
            for g in archive.iter_mut() {
                g.aesthetic_ensemble = aesthetic.score_blocking(PathBuf::from(&g.path)).map(|s| s.ensemble());
            }
        } else {
            println!("(aesthetic_scorer.py sidecar unavailable — trusting whatever pool_log.jsonl had, which may be incomplete)");
        }
    }

    let mut eligible: Vec<&Gem> = archive.iter()
        .filter(|g| g.score >= min_score && g.aesthetic_ensemble.is_none_or(|a| a >= min_aesthetic))
        .collect();
    println!("{} clear score>={min_score} and aesthetic>={min_aesthetic}", eligible.len());
    eligible.sort_by(|a, b| {
        let ka = a.aesthetic_ensemble.unwrap_or(0.0);
        let kb = b.aesthetic_ensemble.unwrap_or(0.0);
        kb.partial_cmp(&ka).unwrap_or(std::cmp::Ordering::Equal)
    });

    let config = load_config();
    // The sidecar's `save_dir` arg only seeds its archive-vs-candidate
    // novelty SCORE (a single float `cmd_curate` never reads — only
    // `embed_blocking`'s vector half is used below); harmless to point it
    // at `archive_path` unconditionally rather than needing it to agree
    // with `model`'s own training pool by construction.
    let mut novelty = NoveltyScorer::with_model(archive_path, model)
        .unwrap_or_else(|| panic!("novelty_scorer.py sidecar unavailable — latent-distance curation needs it (see novelty_model.npz/novelty_head.pt, or a custom --model-path/--head-path pair)"));
    println!("embedding {} eligible candidates...", eligible.len());
    let mut embedded: Vec<(&Gem, Vec<f32>)> = Vec::with_capacity(eligible.len());
    for (i, g) in eligible.iter().enumerate() {
        match novelty.embed_blocking(PathBuf::from(&g.path)) {
            Some((_, vec)) => embedded.push((g, vec)),
            None => eprintln!("  [{i}] embed FAILED for {} — skipping", g.path),
        }
        if (i + 1) % 50 == 0 { println!("  {}/{}", i + 1, eligible.len()); }
    }
    println!("embedded {}/{}", embedded.len(), eligible.len());

    // Proper greedy farthest-point: start from the best-aesthetic embedded
    // candidate, then EACH round re-scan every remaining candidate and take
    // whichever actually MAXIMIZES its distance to everything already
    // selected — not just the first one that happens to clear the bar in
    // aesthetic order. Matters more here than it did for the old pixel-
    // fingerprint version: latent space is the whole point of this
    // rewrite, so it's worth searching properly, not settling for
    // first-fit.
    let mut remaining: Vec<usize> = (0..embedded.len()).collect();
    let mut selected: Vec<usize> = if remaining.is_empty() { Vec::new() } else { vec![remaining.remove(0)] };
    while selected.len() < top_n && !remaining.is_empty() {
        let best = remaining.iter().enumerate()
            .map(|(pos, &i)| {
                let d = selected.iter().map(|&s| l2_dist(&embedded[i].1, &embedded[s].1)).fold(f32::MAX, f32::min);
                (pos, d)
            })
            .max_by(|&(_, da), &(_, db)| da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal));
        match best {
            Some((pos, d)) if d >= min_dist => selected.push(remaining.remove(pos)),
            _ => break,
        }
    }
    let selected: Vec<(&Gem, &Vec<f32>)> = selected.iter().map(|&i| (embedded[i].0, &embedded[i].1)).collect();
    println!("selected {} diverse standouts (latent min_dist>={min_dist}):\n", selected.len());

    let genome = build_genome(formula);
    for (rank, (g, _)) in selected.iter().enumerate() {
        let dest = out_dir.join(format!("gem_{:02}.png", rank + 1));
        let view = View::new_square(g.cx, g.cy, g.zoom);
        save_shot(&genome, &config, &view, res, &dest);
        let min_d = selected.iter().enumerate()
            .filter(|&(j, _)| j != rank)
            .map(|(_, &(_, sv))| l2_dist(selected[rank].1, sv))
            .fold(f32::MAX, f32::min);
        println!("  [{:>2}] score={:.4} aesthetic={:>6} latent_min_dist={:.4} cx={:.6} cy={:.6} zoom={:.3e}\n       {}",
            rank + 1, g.score, g.aesthetic_ensemble.map(|v| format!("{v:.3}")).unwrap_or_else(|| "n/a".into()),
            min_d, g.cx, g.cy, g.zoom, dest.display());
    }
}

// ── VAE-driven per-formula recursive exploration ────────────────────────
//
// New, PARALLEL pipeline alongside cmd_pool/cmd_curate — NOT a modification
// of them. Reuses pick_seeds/apply_offset/Logger/append-mode-numbering
// directly (see vae_explore.rs); the genuinely new mechanism is training a
// from-scratch VAE per formula on RAW escape-time crops and using
// reconstruction error, not the DINOv2+VICReg NoveltyScorer, to guide a
// recursive high-resolution drill. See the project plan ("VAE-driven
// per-formula recursive exploration") for the full design.

/// Replays a queued chain's EXACT export frame sequence offline and reports
/// per-frame richness, so a dead video is caught in seconds instead of after
/// an hour of full-resolution rendering.
///
/// Deliberately renders through `video_export::chain_frame_views` +
/// `render_save(.., VIDEO_FRAME_ALLOW_DD)` — the same sequence generator and
/// the same precision tier the real exporter uses. Re-deriving either is how
/// both previous "flat video" bugs slipped through: the validator scored an
/// image the exporter would never produce.
///
/// `flood` is the fraction of the frame taken by its single most common
/// colour. That is the direct detector for the observed failure — a frame
/// progressively swallowed by one escape-time band — and unlike an entropy
/// score it cannot be fooled by fine dither in an otherwise dead frame.
/// Where a chain to verify/render came from: a queue item, or a
/// `video_zoom_winners.jsonl` entry straight out of a search (which lets the
/// whole explore -> verify -> render pipeline run headlessly, without going
/// through the viewer's queue UI at all).
struct ChainSpec {
    label: String,
    waypoints: Vec<nnfractals::video_export::CapturedView>,
    steps: u32, fps: u32, width: u32, height: u32,
    invert_coords: bool, invert_range: bool,
    colormap: String, angle_coloring: bool,
}

fn cmd_verify_chain(
    queue_id: Option<&str>, stride: usize, max_iter_override: Option<u32>, dump_dir: Option<&Path>,
    iter_sweep: Option<&str>, sweep_res: u32,
    render_video: Option<&Path>, render_dims: (Option<u32>, Option<u32>), render_steps: Option<u32>,
    winners: Option<&Path>, rank: usize, nn_override: Option<&Path>, fps_override: Option<u32>,
    max_frames: Option<u32>, keyframe_stride: u32, angle_override: bool,
) {
    use nnfractals::video_export::{chain_frame_views, render_save, VIDEO_FRAME_ALLOW_DD};

    let mut config = Config::load(Path::new("config.toml")).expect("load config.toml");

    let (item, genome) = match winners {
        Some(manifest) => {
            let nn = nn_override.unwrap_or_else(|| panic!("--winners also needs --nn <genome.nn>"));
            let genome = io::load_genome(nn).unwrap_or_else(|e| panic!("load {}: {e}", nn.display()));
            let content = std::fs::read_to_string(manifest)
                .unwrap_or_else(|e| panic!("read {}: {e}", manifest.display()));
            let entry = content.lines()
                .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
                .find(|v| v["rank"].as_u64() == Some(rank as u64))
                .unwrap_or_else(|| panic!("no winner with rank {rank} in {}", manifest.display()));
            let waypoints: Vec<nnfractals::video_export::CapturedView> =
                serde_json::from_value(entry["chain"].clone()).expect("parse winner chain");
            let spec = ChainSpec {
                label: format!("{}#rank{rank}", manifest.display()),
                waypoints,
                steps: render_steps.unwrap_or(2400),
                fps: fps_override.unwrap_or(30),
                width: render_dims.0.unwrap_or(1080),
                height: render_dims.1.unwrap_or(1980),
                invert_coords: false, invert_range: false,
                colormap: config.rendering.colormap.clone(),
                // Follow whatever the SEARCH used (recorded per winner by
                // `write_winners_manifest`), so the render reproduces the
                // frames the chain was actually scored on. `--angle-coloring`
                // forces it on for manifests written before that field
                // existed, or to re-render an old chain in angle mode.
                angle_coloring: angle_override
                    || entry["angle_coloring"].as_bool().unwrap_or(false),
            };
            (spec, genome)
        }
        None => {
            let queue = nnfractals::video_export::load_queue();
            let it = match queue_id {
                Some(id) => queue.iter().find(|i| i.id == id)
                    .unwrap_or_else(|| panic!("no queue item with id {id}")),
                // Newest multi-waypoint item — the one just queued is what you
                // almost always want to check before letting it render.
                None => queue.iter().filter(|i| i.waypoints.len() >= 2)
                    .max_by_key(|i| i.created_at)
                    .unwrap_or_else(|| panic!("queue has no multi-waypoint chain items")),
            };
            let nn_path = nn_override.map(Path::to_path_buf)
                .unwrap_or_else(|| nnfractals::video_export::queue_dir().join(&it.nn_filename));
            let genome = io::load_genome(&nn_path)
                .unwrap_or_else(|e| panic!("load {}: {e}", nn_path.display()));
            let spec = ChainSpec {
                label: it.id.clone(), waypoints: it.waypoints.clone(),
                steps: it.steps, fps: it.fps, width: it.width, height: it.height,
                invert_coords: it.invert_coords, invert_range: it.invert_range,
                colormap: it.colormap.clone(), angle_coloring: it.angle_coloring,
            };
            (spec, genome)
        }
    };

    config.rendering.colormap = item.colormap.clone();
    if let Some(mi) = max_iter_override { config.rendering.max_iter = mi; }

    let views = chain_frame_views(
        &item.waypoints, item.steps, item.width, item.height,
        item.invert_coords, item.invert_range,
    );
    println!(
        "verify-chain {} — {} legs, {}x{}, {} frames, max_iter={}{}",
        item.label, item.waypoints.len() - 1, item.width, item.height, views.len(),
        config.rendering.max_iter,
        if max_iter_override.is_some() { " (OVERRIDE)" } else { "" },
    );
    if let Some(d) = dump_dir { let _ = std::fs::create_dir_all(d); }

    // `--iter-sweep`: for each sampled frame, find the smallest max_iter in
    // the list that keeps the frame alive. Renders at `sweep_res` (flood is
    // a whole-frame statistic, so it survives downscaling) purely for speed
    // — this is a measurement of the ITERATION requirement vs zoom depth,
    // not a prediction of the shipped frame's exact byte size.
    if let Some(list) = iter_sweep {
        let iters: Vec<u32> = list.split(',').filter_map(|s| s.trim().parse().ok()).collect();
        // Downscale preserving the ITEM's aspect. A square sweep canvas for
        // a portrait export makes `render_save` letterbox, and the black
        // bars then dominate the flood statistic (a constant 0.44 for
        // 1080x1920 into 384x384) — masking the very collapse being
        // measured. Measured, not theorised: that artifact showed up on the
        // first sweep run.
        let scale = sweep_res as f64 / item.width.max(item.height) as f64;
        let sw = ((item.width as f64 * scale).round() as u32).max(1);
        let sh = ((item.height as f64 * scale).round() as u32).max(1);
        println!("iter-sweep at {sw}x{sh} (item aspect preserved): {iters:?}\n");
        println!("{:>5}  {:>11}  {}", "frame", "zoom", "min_iter_alive (flood per iter)");
        for (i, v) in views.iter().enumerate() {
            if i % stride.max(1) != 0 && i != views.len() - 1 { continue; }
            let mut cells = Vec::new();
            let mut min_alive: Option<u32> = None;
            for &it in &iters {
                let mut c = config.clone();
                c.rendering.max_iter = it;
                let rgb = render_save(&genome, &c, v, sw, sh, item.angle_coloring, VIDEO_FRAME_ALLOW_DD);
                let mut counts: std::collections::HashMap<[u8; 3], u32> = std::collections::HashMap::new();
                for px in rgb.chunks_exact(3) { *counts.entry([px[0], px[1], px[2]]).or_insert(0) += 1; }
                let flood = counts.values().copied().max().unwrap_or(0) as f64 / (sw * sh) as f64;
                if flood <= 0.99 && min_alive.is_none() { min_alive = Some(it); }
                cells.push(format!("{it}:{flood:.2}"));
            }
            println!(
                "{i:>5}  {:>11.3e}  {:>6}  [{}]", v.zoom,
                min_alive.map(|x| x.to_string()).unwrap_or_else(|| "NONE".into()),
                cells.join(" "),
            );
        }
        return;
    }

    if let Some(path) = render_video {
        let (rw, rh) = (render_dims.0.unwrap_or(item.width), render_dims.1.unwrap_or(item.height));
        let steps = render_steps.unwrap_or(item.steps);
        match max_frames {
            Some(m) => println!("\nrendering {rw}x{rh}, {steps} steps, {} fps, CAPPED at {m} frames -> {}", item.fps, path.display()),
            None => println!("\nrendering {rw}x{rh}, {steps} steps, {} fps{} -> {}", item.fps,
                if keyframe_stride > 1 { format!(", keyframe-interpolated every {keyframe_stride}") } else { String::new() },
                path.display()),
        }
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            for m in rx {
                match m {
                    nnfractals::video_export::VideoMsg::Progress { done, total } => {
                        if done % 25 == 0 || done == total {
                            println!("  frame {done}/{total}");
                        }
                    }
                    nnfractals::video_export::VideoMsg::Done(p) => println!("  DONE {}", p.display()),
                    nnfractals::video_export::VideoMsg::Failed(e) => println!("  FAILED {e}"),
                    _ => {}
                }
            }
        });
        nnfractals::video_export::export_video_chain_interpolated(
            &genome, &config, item.angle_coloring, &item.waypoints, steps, item.fps,
            rw, rh, item.invert_coords, item.invert_range, path, &tx, &|| {},
            max_frames, keyframe_stride,
        );
        return;
    }

    println!("{:>5}  {:>11}  {:>9}  {:>7}  {:>6}  {:>7}", "frame", "zoom", "png_bytes", "flood", "colors", "noisy");
    let mut worst_flood = 0.0f64;
    let mut worst_frame = 0usize;
    let mut first_dead: Option<usize> = None;
    let mut worst_detail = f64::NEG_INFINITY;
    let mut worst_detail_frame = 0usize;
    let mut first_noise: Option<usize> = None;
    let mut rows = 0usize;

    for (i, v) in views.iter().enumerate() {
        if i % stride.max(1) != 0 && i != views.len() - 1 { continue; }
        let rgb = render_save(&genome, &config, v, item.width, item.height, item.angle_coloring, VIDEO_FRAME_ALLOW_DD);

        let mut counts: std::collections::HashMap<[u8; 3], u32> = std::collections::HashMap::new();
        for px in rgb.chunks_exact(3) { *counts.entry([px[0], px[1], px[2]]).or_insert(0) += 1; }
        let total_px = (item.width * item.height) as f64;
        let top = counts.values().copied().max().unwrap_or(0) as f64;
        let flood = top / total_px;
        let n_colors = counts.len();

        let png_bytes = encode_png_len(&rgb, item.width, item.height);

        // Luminance field, so the SAME coherence metric the search gates on
        // applies to the shipped RGB frame.
        let lum: Vec<f32> = rgb.chunks_exact(3)
            .map(|p| (p[0] as f32 + p[1] as f32 + p[2] as f32) / 3.0).collect();
        let detail = nnfractals::fitness::noise_tile_fraction(&lum, item.width, item.height) as f64;
        if detail > worst_detail { worst_detail = detail; worst_detail_frame = i; }
        if (detail as f32) > nnfractals::fitness::MAX_NOISE_TILE_FRACTION && first_noise.is_none() {
            first_noise = Some(i);
        }

        if flood > worst_flood { worst_flood = flood; worst_frame = i; }
        // 99% one colour = visually dead. Chosen from measurement, not taste:
        // the healthy frames of Carl's flat video sat at 21-62% flood while
        // the dead tail pinned at 99.95%.
        if flood > 0.99 && first_dead.is_none() { first_dead = Some(i); }

        if let Some(d) = dump_dir {
            let p = d.join(format!("frame_{i:04}.png"));
            let _ = io::save_png(&rgb, item.width, item.height, &p);
            // Raw ESCAPE-TIME field alongside the RGB, for analysing what the
            // speckle actually is: colour is a lossy view of it (the colormap
            // can alias distinct escape times together, and vice versa), so
            // any question about repeating/cycling VALUES has to be asked of
            // the field itself.
            let use_f64 = nnfractals::video_export::needs_f64(v, item.width);
            let eff = nnfractals::video_export::effective_max_iter(v, config.rendering.max_iter);
            let field = nnfractals::video_export::render_escape_times(
                &genome, &config, v, item.width, item.height, eff, use_f64, VIDEO_FRAME_ALLOW_DD,
            );
            let mut bytes = Vec::with_capacity(12 + field.len() * 4);
            bytes.extend_from_slice(&item.width.to_le_bytes());
            bytes.extend_from_slice(&item.height.to_le_bytes());
            bytes.extend_from_slice(&eff.to_le_bytes());
            for f in &field { bytes.extend_from_slice(&f.to_le_bytes()); }
            let _ = std::fs::write(d.join(format!("frame_{i:04}.f32")), &bytes);
        }
        println!("{i:>5}  {:>11.3e}  {png_bytes:>9}  {flood:>7.4}  {n_colors:>6}  {detail:>7.3}", v.zoom);
        rows += 1;
    }

    println!("\nchecked {rows} frames (stride {stride})");
    println!("worst flood {:.4} at frame {worst_frame}", worst_flood);
    println!("worst noisy-tile fraction {:.3} at frame {worst_detail_frame}", worst_detail);
    // Both failure modes disqualify. They are opposites — a dead frame is
    // one flat colour, a noise frame is maximally busy — and a chain only
    // ships if it avoids BOTH along its whole length.
    match (first_dead, first_noise) {
        (Some(f), _) => println!(
            "VERDICT: DEAD — frame {f} of {} ({:.0}% through) is >99% a single colour",
            views.len(), 100.0 * f as f64 / views.len() as f64,
        ),
        (None, Some(f)) => println!(
            "VERDICT: NOISE — frame {f} of {} ({:.0}% through) has {:.0}% of its textured tiles as dither, not structure",
            views.len(), 100.0 * f as f64 / views.len() as f64, worst_detail * 100.0,
        ),
        (None, None) => println!("VERDICT: ALIVE — no frame flat (>99% one colour) and none noise (>25% dither tiles)"),
    }
}

fn encode_png_len(rgb: &[u8], w: u32, h: u32) -> usize {
    let mut buf = std::io::Cursor::new(Vec::new());
    let img = image::RgbImage::from_raw(w, h, rgb.to_vec()).expect("rgb buffer size");
    img.write_to(&mut buf, image::ImageFormat::Png).expect("encode png");
    buf.into_inner().len()
}

fn get_flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map(String::as_str)
}

fn get_flag_or<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    get_flag(args, name).and_then(|s| s.parse().ok()).unwrap_or(default)
}

/// Parses a "R,A,B" flag value (3 comma-separated floats) for
/// `quat-mandelbrot`'s vector-valued flags (`--pivot`, `--axis`, `--origin`,
/// `--direction`, `--basis-u`, `--basis-v`).
fn parse_vec3(s: &str) -> (f64, f64, f64) {
    let parts: Vec<f64> = s
        .split(',')
        .map(|p| {
            p.trim()
                .parse()
                .unwrap_or_else(|_| panic!("bad vec3 component {p:?} in {s:?} (expected \"R,A,B\")"))
        })
        .collect();
    match parts.as_slice() {
        [r, a, b] => (*r, *a, *b),
        _ => panic!("expected \"R,A,B\" (3 comma-separated numbers), got {s:?}"),
    }
}

/// Parses `--time-axis` (r|a|b|c) for the quat-* subcommands — which
/// quaternion component the animation/gravity/voxel time value drives; the
/// other three are what the Slice/mass-field treat as spatial.
fn parse_time_axis(s: &str) -> nnfractals::quat_fractal::TimeAxis {
    nnfractals::quat_fractal::TimeAxis::parse(s)
        .unwrap_or_else(|| panic!("unknown --time-axis {s:?} — expected one of: r, a, b, c"))
}

/// Reads a `scripts/tune_autoencoder.py` result JSON (`{"arch":...,
/// "latent_dim":..., "kl_weight":..., ...}`) — fields are individually
/// optional so a hand-edited or partial config still loads whatever it
/// has.
struct TunedConfig { arch: Option<String>, latent_dim: Option<usize>, kl_weight: Option<f64> }

fn load_tuned_config(path: &Path) -> TunedConfig {
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read --tuned-config {}: {e}", path.display()));
    let v: serde_json::Value = serde_json::from_str(&content)
        .unwrap_or_else(|e| panic!("parse --tuned-config {}: {e}", path.display()));
    TunedConfig {
        arch: v["arch"].as_str().map(str::to_string),
        latent_dim: v["latent_dim"].as_u64().map(|n| n as usize),
        kl_weight: v["kl_weight"].as_f64(),
    }
}

fn parse_select_by(s: &str) -> SelectBy {
    match s {
        "max-error" => SelectBy::MaxError,
        "min-error" => SelectBy::MinError,
        "random" => SelectBy::Random,
        other => panic!("--select-by must be one of max-error|min-error|random, got {other:?}"),
    }
}

/// Mean `recon_mse` across every line of a `score_vae_corpus.py`-produced
/// manifest — the authoritative per-iteration number (computed here in
/// Rust from the manifest file itself, not scraped from the Python
/// subprocess's stdout).
fn mean_recon_error(manifest_path: &Path) -> Option<f32> {
    let content = std::fs::read_to_string(manifest_path).ok()?;
    let vals: Vec<f32> = content.lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .filter_map(|v| v["recon_mse"].as_f64())
        .map(|x| x as f32)
        .collect();
    if vals.is_empty() { return None; }
    Some(vals.iter().sum::<f32>() / vals.len() as f32)
}

/// Reads a `score_vae_corpus.py` manifest, sorts by `recon_mse` per
/// `select_by`, re-renders the top `top_n` at `res` into `out_dir`. Each
/// zone's own saved `.nn` (in `pool_dir`) already fully describes its
/// genome+view, so no separate formula/genome argument is needed — this
/// is what makes `vae-curate` a thin, standalone, separately-invocable
/// tail (mirrors the existing `gems`/`curate` split).
fn cmd_vae_curate(pool_dir: &Path, top_n: usize, out_dir: &Path, res: u32, select_by: SelectBy) {
    std::fs::create_dir_all(out_dir).expect("create out_dir");
    let manifest_path = pool_dir.join("vae_recon_manifest.jsonl");
    let content = std::fs::read_to_string(&manifest_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", manifest_path.display()));
    let mut entries: Vec<(String, f32)> = content.lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .filter_map(|v| Some((v["stem"].as_str()?.to_string(), v["recon_mse"].as_f64()? as f32)))
        .collect();
    match select_by {
        SelectBy::MaxError => entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)),
        SelectBy::MinError | SelectBy::Random => entries.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)),
    }
    entries.truncate(top_n);
    println!("curating top {} of {} zones (select_by={select_by:?}) from {}:\n", entries.len(), content.lines().count(), pool_dir.display());

    let config = load_config();
    for (rank, (stem, recon_mse)) in entries.iter().enumerate() {
        let nn_path = pool_dir.join(format!("{stem}.nn"));
        let Ok(genome) = io::load_genome(&nn_path) else { eprintln!("  [{:>2}] {stem}: missing/unreadable .nn, skipping", rank + 1); continue };
        let view = View::new_square(genome.view_cx as f64, genome.view_cy as f64, genome.view_zoom as f64);
        let dest = out_dir.join(format!("zone_{:02}.png", rank + 1));
        save_shot(&genome, &config, &view, res, &dest);
        println!("  [{:>2}] {stem} recon_mse={recon_mse:.6} cx={:.6} cy={:.6} zoom={:.3e}\n       {}",
            rank + 1, genome.view_cx, genome.view_cy, genome.view_zoom, dest.display());
    }
}

/// Fixed training-canvas resolution the saliency net is designed around —
/// small and deliberately far under the GPU dispatch/precision limits that
/// matter for a REAL exploration canvas (see `vae_explore::CANVAS_RES`'s
/// doc comment): this is a synthetic "what would the canvas around this
/// already-scored zone have looked like" render, so it can stay cheap.
const SALIENCY_CANVAS_RES: u32 = 256;
/// Zoom-out factors a training canvas is rendered at, relative to the
/// labeled zone's own zoom — mirrors `vae_explore::CANVAS_SCAN_SCALES`'
/// implied range (a candidate at scale s came from a canvas roughly 1/s
/// times shallower: 0.5→2x, 0.25→4x, 0.125→8x), so the synthetic canvases
/// this builds look like the ones the live search actually sees.
const SALIENCY_ZOOM_FACTORS: &[f64] = &[2.0, 4.0, 8.0];

/// Builds a saliency-net training set from EXISTING scored `vae-explore`
/// pools — deliberately NOT a new exploration/rendering-heavy data
/// collection pass: every pool already has thousands of zones with known
/// (cx, cy, zoom) and (once `score_vae_corpus.py` has run as part of a
/// normal `vae-explore` iteration) a known VAE reconstruction-error label
/// in `vae_recon_manifest.jsonl`. For each sampled zone: pick a random
/// `SALIENCY_ZOOM_FACTORS` entry and a random off-center offset (so the
/// zone ISN'T always dead-center — training on always-centered labels
/// would let the net shortcut to "predict center is always interesting"
/// instead of learning real position-dependent content), render that
/// wider canvas fresh, and record the zone's normalized (px, py) position
/// within it alongside its known reconstruction-error label. `px`/`py`
/// follow the exact same row/col-to-coordinate convention
/// `render_escape_times` itself uses (col 0 = xmin, row 0 = ymin) so a
/// later consumer can invert the mapping without guessing a sign
/// convention.
/// A saliency-dataset entry before its label is resolved: `nn_path` always
/// carries its OWN genome+view (not a shared per-pool genome — needed
/// because a manual-marks directory can mix zones from different
/// formulas/genomes across sessions, unlike a single vae-explore pool
/// where every zone shares one formula). `precomputed_label` is `Some` for
/// a manifest-scored pool, `None` for a raw `.nn` that needs live-scoring.
struct SaliencyEntry {
    nn_path: PathBuf,
    stem: String,
    precomputed_label: Option<f32>,
}

/// `pool_dirs` is deliberately heterogeneous: a normal vae-explore pool
/// (has `vae_recon_manifest.jsonl` — real, already-measured VAE
/// reconstruction-error labels) OR a plain directory of `.nn` files with
/// no manifest at all (e.g. `explorer_out/saliency_manual_marks/`, written
/// by the viewer's Shift-drag "mark a zone" feature — Carl's request,
/// 2026-08-10: "the algo is ignoring an interesting part... I would like
/// to add a way to give more data for the conv2d to train on"). The
/// latter needs `vae_model_path` to actually score each mark for real
/// (rendering it and asking a trained VAE for its reconstruction error) —
/// deliberately NOT a synthetic/assumed-high label: a human finding a spot
/// visually interesting isn't the same claim as "the VAE finds this hard
/// to reconstruct," and this project's whole point is training on the
/// real signal. Pools without a manifest AND no `vae_model_path` given are
/// skipped with a warning, same as before this existed.
fn cmd_saliency_data(pool_dirs: &[PathBuf], out_dir: &Path, canvas_res: u32, max_per_pool: usize, vae_model_path: Option<&Path>) {
    std::fs::create_dir_all(out_dir).expect("create out_dir");
    let config = load_config();
    let mut log = Logger::new(&out_dir.join("saliency_dataset.jsonl")).expect("open dataset log");
    log.verbose = false;
    // Lazily spawned on first actual need (a pool without a manifest) —
    // most callers only pass already-scored pools, and sidecar startup
    // isn't free.
    let mut vae_scorer: Option<VaeScorer> = None;

    let mut total = 0usize;
    for pool_dir in pool_dirs {
        let manifest_path = pool_dir.join("vae_recon_manifest.jsonl");
        let (mut entries, mode): (Vec<SaliencyEntry>, &str) = if let Ok(text) = std::fs::read_to_string(&manifest_path) {
            let entries = text.lines()
                .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
                .filter_map(|v| Some(SaliencyEntry {
                    nn_path: pool_dir.join(format!("{}.nn", v["stem"].as_str()?)),
                    stem: v["stem"].as_str()?.to_string(),
                    precomputed_label: Some(v["recon_mse"].as_f64()? as f32),
                }))
                .collect();
            (entries, "pre-scored")
        } else {
            let entries = std::fs::read_dir(pool_dir).ok().into_iter().flatten()
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("nn"))
                .map(|p| SaliencyEntry {
                    stem: p.file_stem().and_then(|s| s.to_str()).unwrap_or("mark").to_string(),
                    nn_path: p,
                    precomputed_label: None,
                })
                .collect();
            (entries, "live-scoring")
        };
        if entries.is_empty() {
            eprintln!("skipping {}: no vae_recon_manifest.jsonl and no .nn files found", pool_dir.display());
            continue;
        }
        entries.shuffle(&mut rand::rng());
        entries.truncate(max_per_pool);
        let pool_name = pool_dir.file_name().and_then(|s| s.to_str()).unwrap_or("pool").to_string();
        println!("{}: {} examples, mode={mode}", pool_dir.display(), entries.len());

        for entry in &entries {
            let Ok(zone_g) = io::load_genome(&entry.nn_path) else { continue };

            let label = match entry.precomputed_label {
                Some(l) => l,
                None => {
                    if vae_scorer.is_none() {
                        vae_scorer = vae_model_path.and_then(VaeScorer::new);
                        if vae_scorer.is_none() {
                            eprintln!("skipping {}: no manifest and no usable --vae-model to live-score raw marks", pool_dir.display());
                            break;
                        }
                    }
                    let zone_view = View::new_square(zone_g.view_cx as f64, zone_g.view_cy as f64, zone_g.view_zoom.max(0.1) as f64);
                    let use_f64 = needs_f64(&zone_view, vae_explore::ZONE_RES);
                    let field = render_escape_times(&zone_g, &config, &zone_view, vae_explore::ZONE_RES, vae_explore::ZONE_RES, config.rendering.max_iter, use_f64, true);
                    let tmp_png = out_dir.join("_mark_score_tmp.png");
                    if io::save_raw_field(&field, vae_explore::ZONE_RES, vae_explore::ZONE_RES, config.rendering.max_iter, &tmp_png).is_err() { continue; }
                    let Some(scorer) = vae_scorer.as_mut() else { continue };
                    let Some(mse) = scorer.score_blocking(tmp_png) else { continue };
                    mse
                }
            };

            let zone_cx = zone_g.view_cx as f64;
            let zone_cy = zone_g.view_cy as f64;
            let zone_zoom = zone_g.view_zoom.max(0.1) as f64;

            let factor = *SALIENCY_ZOOM_FACTORS.choose(&mut rand::rng()).unwrap();
            let canvas_zoom = zone_zoom / factor;
            let canvas_half = 2.0 / canvas_zoom;
            // Up to 60% of the canvas half-width off-center — the zone
            // still reliably lands INSIDE the canvas (not clipped out) but
            // rarely dead-center.
            let mut rng = rand::rng();
            let off_x = rng.random_range(-0.6..0.6) * canvas_half;
            let off_y = rng.random_range(-0.6..0.6) * canvas_half;
            let canvas_view = View::new_square(zone_cx - off_x, zone_cy - off_y, canvas_zoom);

            let use_f64 = needs_f64(&canvas_view, canvas_res);
            let field = render_escape_times(&zone_g, &config, &canvas_view, canvas_res, canvas_res, config.rendering.max_iter, use_f64, true);

            let canvas_name = format!("canvas_{total:06}.png");
            io::save_raw_field(&field, canvas_res, canvas_res, config.rendering.max_iter, &out_dir.join(&canvas_name)).expect("save canvas");

            let px = 0.5 + off_x / (2.0 * canvas_half);
            let py = 0.5 + off_y / (2.0 * canvas_half);
            log.log(&serde_json::json!({
                "canvas": canvas_name, "px": px, "py": py, "label": label,
                "pool": pool_name, "zone_stem": entry.stem,
                // True exactly for a live-scored (no manifest) entry —
                // i.e. a manually marked zone, not an existing pool's
                // already-measured one. train_saliency.py oversamples
                // these to a target training-mass fraction, since with
                // typically only a handful of marks against thousands of
                // pool examples, plain unweighted sampling makes them
                // statistically invisible (Carl's observation, 2026-08-10:
                // "the behavior didn't change much").
                "is_manual_mark": entry.precomputed_label.is_none(),
            }));
            total += 1;
            if total.is_multiple_of(200) { println!("  {total} examples so far..."); }
        }
    }
    let _ = std::fs::remove_file(out_dir.join("_mark_score_tmp.png"));
    println!("saliency-data: {total} examples written to {}/saliency_dataset.jsonl", out_dir.display());
}

/// Auto-discovers every clean vae-explore pool (`explorer_out/*_vae` —
/// deliberately an exact `_vae` suffix match, which naturally excludes
/// known backup/superseded dirs like `..._precorpusfix`/`..._buggy_cy10`/
/// `..._wrong_genome`, none of which end in plain `_vae`) plus the manual-
/// marks directory if it exists, regenerates the saliency dataset from
/// scratch across all of them, then retrains — a one-button "incorporate
/// everything I've marked since last time" (Carl's request, 2026-08-10).
/// Regenerating fully each time (not incrementally) is deliberate: simpler
/// and avoids any duplicate-accumulation risk, and a full ~7000-example
/// regen only takes ~100s.
fn cmd_retrain_saliency(out_dir: &Path, canvas_res: u32, max_per_pool: usize, vae_model_path: &Path, epochs: usize) {
    let mut pool_dirs: Vec<PathBuf> = std::fs::read_dir("explorer_out").ok().into_iter().flatten()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .filter(|p| p.file_name().and_then(|s| s.to_str()).is_some_and(|s| s.ends_with("_vae")))
        .collect();
    pool_dirs.sort();
    let marks_dir = PathBuf::from("explorer_out/saliency_manual_marks");
    if marks_dir.is_dir() { pool_dirs.push(marks_dir); }
    if pool_dirs.is_empty() {
        panic!("retrain-saliency: found no explorer_out/*_vae pools and no saliency_manual_marks — nothing to train on");
    }
    println!("retrain-saliency: {} pools found:", pool_dirs.len());
    for p in &pool_dirs { println!("  {}", p.display()); }

    cmd_saliency_data(&pool_dirs, out_dir, canvas_res, max_per_pool, Some(vae_model_path));

    println!("=== retrain-saliency: training ===");
    let python = nnfractals::python_bin(Path::new("."));
    let status = Command::new(python)
        .arg("scripts/train_saliency.py")
        .arg("--data").arg(out_dir)
        .arg("--out").arg(vae_explore::SALIENCY_DEFAULT_MODEL_PATH)
        .arg("--epochs").arg(epochs.to_string())
        .status()
        .expect("run train_saliency.py");
    if !status.success() {
        panic!("train_saliency.py failed ({status})");
    }
    println!("retrain-saliency: done — {} updated", vae_explore::SALIENCY_DEFAULT_MODEL_PATH);
}

/// Exports the raw escape-time tensor AND the real/imaginary/magnitude of
/// the bailout z value for one zone or every zone in a directory —
/// exploratory data for a possible complex-valued autoencoder (Carl's
/// request, 2026-08-07). Deliberately standalone: reuses any already-saved
/// `.nn` file from any prior pool/gems/vae-explore run (same
/// `View::new_square(genome.view_cx, ...)` reconstruction `cmd_vae_curate`
/// already uses), so it doesn't touch or slow down the live `vae-explore`
/// loop for a feature that's still at the "is this worth it" stage.
fn cmd_complex_export(input: &Path, out_dir: &Path, res: u32, limit: usize) {
    std::fs::create_dir_all(out_dir).expect("create out_dir");
    let config = load_config();

    let nn_paths: Vec<PathBuf> = if input.is_dir() {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(input).expect("read input dir")
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("nn"))
            .collect();
        paths.sort();
        paths.truncate(limit);
        paths
    } else {
        vec![input.to_path_buf()]
    };

    println!("complex-export: {} zone(s) -> {}", nn_paths.len(), out_dir.display());
    for nn_path in &nn_paths {
        let Ok(genome) = io::load_genome(nn_path) else {
            eprintln!("  {}: unreadable .nn, skipping", nn_path.display());
            continue;
        };
        let stem = nn_path.file_stem().and_then(|s| s.to_str()).unwrap_or("zone").to_string();
        let view = View::new_square(genome.view_cx as f64, genome.view_cy as f64, genome.view_zoom as f64);
        let use_f64 = needs_f64(&view, res);

        let escape_field = render_escape_times(&genome, &config, &view, res, res, config.rendering.max_iter, use_f64, true);
        let complex_field = render_complex_field(&genome, &view, res, res, config.rendering.max_iter, use_f64);

        io::save_raw_field(&escape_field, res, res, config.rendering.max_iter, &out_dir.join(format!("{stem}_tensor.png")))
            .expect("save tensor");
        io::save_complex_channels(
            &complex_field, genome.bailout_radius, res, res,
            &out_dir.join(format!("{stem}_re.png")),
            &out_dir.join(format!("{stem}_im.png")),
            &out_dir.join(format!("{stem}_mag.png")),
        ).expect("save complex channels");

        println!("  {stem}: tensor + re + im + mag ({} px, {})", res, if use_f64 { "f64" } else { "f32" });
    }
    println!("done: {} zone(s) exported to {}", nn_paths.len(), out_dir.display());
}

/// Consecutive zero-growth OUTER iterations before `cmd_vae_explore`
/// recenters its search anchor — see that function's recentering logic
/// for why this is needed at all. Originally 2 ("give unlucky method
/// rotation a second chance"), lowered to 1 after a real observation
/// (2026-08-09, Carl): each outer iteration here is expensive (6+ seeds ×
/// deep recursion × slow CPU-tier canvas renders can easily be 10+
/// minutes), and — critically — `pick_seeds` is a DETERMINISTIC function
/// of `(genome, view, method)`. On a RESUMED run especially, iteration 0
/// reuses the exact same seeds/method the PRIOR run's iteration 0 already
/// fully explored, so a single zero there is already conclusive, not bad
/// luck — waiting for a second confirming zero just burns another full,
/// slow iteration for no new information.
const RECENTER_AFTER_STALL: usize = 1;

/// Picks a fresh search anchor when `pick_seeds`' fixed wide-radius sweep
/// around ONE unchanging view has been exhausted (`RECENTER_AFTER_STALL`
/// straight zero-growth iterations) — the real bug `cmd_vae_explore`'s
/// original design had: every outer iteration called `pick_seeds` with
/// the SAME `base_view`, so once that neighborhood's distinct candidates
/// were found, the whole run was structurally stuck no matter how many
/// iterations/seeds were thrown at it (confirmed on Mandelbrot: plateaued
/// hard at 354 zones, and on Burning Ship: only 6 seeds ever findable
/// near its one fixed anchor, regardless of `--n-seeds`).
///
/// Picks a RANDOM zone already saved this run and reuses its `(cx,cy)` —
/// any of them is, by construction, a genuinely interesting spot for THIS
/// formula/genome (found by the same coarse-scan/gate this whole pipeline
/// already trusts), unlike guessing a fresh random point cold, which
/// risks landing in boring exterior/interior territory with no way to
/// know in advance. Reuses the RUN'S ORIGINAL zoom, not the picked zone's
/// own (likely much deeper) zoom, so `pick_seeds`' wide-radius sweep has
/// real breadth to search from the new anchor rather than starting
/// already zoomed into one exact point.
fn pick_recenter_anchor(out_dir: &Path, original_zoom: f64) -> Option<View> {
    let candidates: Vec<PathBuf> = std::fs::read_dir(out_dir).ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.file_stem().and_then(|s| s.to_str()).is_some_and(|s| s.starts_with("zone_")))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("nn"))
        .collect();
    let picked = candidates.choose(&mut rand::rng())?;
    let g = io::load_genome(picked).ok()?;
    Some(View::new_square(g.view_cx as f64, g.view_cy as f64, original_zoom))
}

#[allow(clippy::too_many_arguments)]
fn cmd_vae_explore(
    formula: &str, genome_override: Option<Genome>, cx: f64, cy: f64, zoom: f64, out_dir: &Path,
    iterations: usize, n_seeds: usize, recursion_depth: usize, top_k: usize, canvas_res: u32,
    method_arg: &str, select_by: SelectBy, gate: ZoneGate,
    arch: &str, latent_dim: usize, kl_weight: f64, epochs: usize,
    target_recon_mse: Option<f32>, min_improvement: f32, patience: usize,
    saliency_model_path: Option<PathBuf>,
) {
    std::fs::create_dir_all(out_dir).expect("create out_dir");
    // genome_override: an arbitrary GA-discovered genome loaded directly
    // from a .nn file, rather than one of known_formulas::LIBRARY's named
    // formulas — see cmd_complex_export/cmd_vae_curate for the same
    // load-a-saved-genome pattern. Lets vae-explore point at genuinely
    // novel, already-vetted structure instead of only the handful of
    // textbook formulas, several of which (Burning Ship, Tricorn) turned
    // out to have unusable default reference views.
    let genome = genome_override.unwrap_or_else(|| build_genome(formula));
    let config = load_config();
    let mut base_view = View::new_square(cx, cy, zoom);

    // Append-mode resume — identical logic to cmd_pool's, so re-running
    // vae-explore into the same out_dir grows it instead of overwriting.
    let mut next_stem: usize = std::fs::read_dir(out_dir)
        .map(|rd| rd.filter_map(|e| e.ok())
            .filter_map(|e| e.path().file_stem().and_then(|s| s.to_str().and_then(|s| s.strip_prefix("zone_")).map(str::to_string)))
            .filter_map(|n| n.parse::<usize>().ok())
            .max().map_or(0, |m| m + 1))
        .unwrap_or(0);

    // Dedup registry (see `vae_explore::is_near_duplicate`), seeded from
    // every zone ALREADY in out_dir — without this, append-mode resume
    // would only dedup NEW zones against each other, not against the
    // existing corpus, defeating the point on every resumed run.
    let mut seen: Vec<(f64, f64, f64)> = std::fs::read_dir(out_dir)
        .map(|rd| rd.filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("nn"))
            .filter_map(|e| io::load_genome(&e.path()).ok())
            .map(|g| (g.view_cx as f64, g.view_cy as f64, g.view_zoom as f64))
            .collect())
        .unwrap_or_default();

    let mut log = Logger::append(&out_dir.join("vae_explore_log.jsonl")).expect("open log");
    log.verbose = false;
    log.log(&serde_json::json!({
        "event": "run_meta", "formula": formula, "cx": cx, "cy": cy, "zoom": zoom,
        "iterations": iterations, "n_seeds": n_seeds, "recursion_depth": recursion_depth,
        "top_k": top_k, "canvas_res": canvas_res, "resume_from": next_stem,
        "target_recon_mse": target_recon_mse, "min_improvement": min_improvement, "patience": patience,
    }));

    let python = nnfractals::python_bin(Path::new("."));
    let vae_model_path = out_dir.join("vae_model.pt");
    let manifest_path = out_dir.join("vae_recon_manifest.jsonl");
    let opts = RecursionOpts { top_k, canvas_res, select_by };
    // Global, cross-run/cross-formula pointer to whatever VAE last finished
    // training successfully, anywhere — "the VAE is unique to a single
    // fractal formula but the ideal VAE structure is shared" (Carl's own
    // framing) means warm-starting from a DIFFERENT formula's checkpoint is
    // a reasonable default, not just within-run warm-starting between
    // iterations. train_autoencoder.py's --init-from falls back to random
    // init on any architecture mismatch, so pointing at this unconditionally
    // is always safe. One flat file rather than per-formula bookkeeping —
    // Carl asked for "a simple feature".
    let last_successful_vae = Path::new("explorer_out/last_successful_vae.pt");

    let mut vae_scorer: Option<VaeScorer> = None;
    // Unlike `vae_scorer` (retrained and respawned every outer iteration —
    // see `vae_model_path` below), a saliency model is trained OFFLINE
    // beforehand (`explorer saliency-data` + `scripts/train_saliency.py`)
    // and doesn't change during a run, so it's spawned once, up front.
    // Defaults to `vae_explore::SALIENCY_DEFAULT_MODEL_PATH` (see
    // `saliency_model_path`'s call site) — `None` here means either that
    // default file doesn't exist yet, or `--saliency-model` pointed at a
    // missing file (`SaliencyScorer::new` itself checks existence), which
    // falls back to exactly the pre-Phase-22 behavior: the grid alone, no
    // extra candidates. This is deliberately additive, not a replacement —
    // even with a real checkpoint loaded, a bad prediction can't remove or
    // override anything the proven grid-based search already finds.
    let mut saliency_scorer: Option<nnfractals::saliency::SaliencyScorer> = saliency_model_path
        .filter(|p| p.exists())
        .and_then(|p| nnfractals::saliency::SaliencyScorer::new(&p));
    if saliency_scorer.is_some() {
        println!("saliency model loaded — coarse_scan will be augmented with predicted-heatmap candidates each level");
    }
    let mut prev_mean: Option<f32> = None;
    // best_mean/stall_count track a plateau independent of target_recon_mse:
    // Mandelbrot and Burning Ship runs bottomed out at very different means
    // (~0.023 vs ~0.037) and both bounced non-monotonically along the way
    // (e.g. Mandelbrot iter5->6: 0.030->0.048), so "best seen so far, with
    // patience" generalizes across formulas where a single hardcoded
    // absolute floor would not.
    let mut best_mean: Option<f32> = None;
    let mut stall_count: usize = 0;
    let mut iterations_ran: usize = 0;
    // Tracks CONSECUTIVE zero-new-zone outer iterations, independent of
    // `stall_count` (which tracks reconstruction-error plateau, a
    // different signal — see RECENTER_AFTER_STALL's doc comment).
    let mut zero_growth_iters: usize = 0;

    for outer_iter in 0..iterations {
        iterations_ran = outer_iter + 1;
        let method = match method_arg {
            "mixed" => ScoreMethod::ALL[outer_iter % ScoreMethod::ALL.len()],
            s => ScoreMethod::parse(s).unwrap_or_else(|| panic!("method must be one of entropy|edge|gated-entropy|gated-edge|mixed")),
        };
        println!("=== iteration {outer_iter}/{iterations}: select (method={}) ===", method.name());
        let seeds = pick_seeds(&genome, &config, &base_view, method, n_seeds, &mut log, nnfractals::explore::EXPLORE_WIDE_RADIUS, nnfractals::explore::WIDE_SCALES);
        let mut zones_this_iter = 0;
        for (i, seed) in seeds.iter().enumerate() {
            let n = vae_explore::recursive_drill(
                &genome, &config, seed.clone(), recursion_depth, vae_scorer.as_mut(), saliency_scorer.as_mut(),
                method, &opts, &gate, out_dir, &mut next_stem, &mut log, &mut seen,
            );
            zones_this_iter += n;
            println!("  seed {}/{}: {n} zones saved", i + 1, seeds.len());
        }
        println!("iteration {outer_iter}: {zones_this_iter} zones this iteration, {next_stem} total in corpus");

        if zones_this_iter == 0 {
            zero_growth_iters += 1;
            if zero_growth_iters >= RECENTER_AFTER_STALL
                && let Some(new_anchor) = pick_recenter_anchor(out_dir, zoom) {
                println!(
                    "=== recentering: {zero_growth_iters} straight zero-growth iterations — \
                     moving search anchor from ({:.6},{:.6}) to ({:.6},{:.6}) ===",
                    base_view.cx, base_view.cy, new_anchor.cx, new_anchor.cy,
                );
                log.log(&serde_json::json!({
                    "event": "recenter", "iter": outer_iter,
                    "from_cx": base_view.cx, "from_cy": base_view.cy,
                    "to_cx": new_anchor.cx, "to_cy": new_anchor.cy,
                }));
                base_view = new_anchor;
                zero_growth_iters = 0;
            }
        } else {
            zero_growth_iters = 0;
        }

        drop(vae_scorer.take()); // release the stale-checkpoint sidecar before retraining

        println!("=== iteration {outer_iter}: train ===");
        // Per-iteration, inside out_dir — without this, train_autoencoder.py
        // falls back to its own default (`vae_recon.png` in the CWD), which
        // would silently overwrite one shared file at the repo root on
        // every iteration of every formula's run.
        let contact_sheet_path = out_dir.join(format!("vae_recon_iter{outer_iter:02}.png"));
        let mut train_cmd = Command::new(&python);
        train_cmd
            .arg("scripts/train_autoencoder.py")
            .args(["--dirs", out_dir.to_str().expect("out_dir must be valid UTF-8")])
            .args(["--variant", "vae"])
            .args(["--res", "512"])
            .args(["--channels", "1"])
            .args(["--arch", arch])
            .args(["--latent-dim", &latent_dim.to_string()])
            .args(["--kl-weight", &kl_weight.to_string()])
            .args(["--epochs", &epochs.to_string()])
            // train_autoencoder.py's defaults (200 images / 64 held-out) are
            // sized for the big RGB gallery corpus — a per-formula vae-explore
            // corpus is realistically tens to low hundreds of zones,
            // especially in early iterations, so both floors need to be much
            // smaller here. 20 matches train_novelty.py's own established
            // floor for an equally self-supervised loss.
            .args(["--min-images", "20"])
            .args(["--min-val", "8"])
            .args(["--out", vae_model_path.to_str().expect("out_dir must be valid UTF-8")])
            .args(["--contact-sheet", contact_sheet_path.to_str().expect("out_dir must be valid UTF-8")]);
        if last_successful_vae.exists() {
            train_cmd.args(["--init-from", last_successful_vae.to_str().expect("path must be valid UTF-8")]);
        }
        let status = train_cmd.status().expect("spawn train_autoencoder.py");
        if !status.success() { panic!("train_autoencoder.py failed (exit {status})"); }
        let archived = out_dir.join(format!("vae_model_iter{outer_iter:02}.pt"));
        std::fs::copy(&vae_model_path, &archived).expect("archive checkpoint");
        std::fs::copy(&vae_model_path, last_successful_vae).expect("update last-successful-vae pointer");

        println!("=== iteration {outer_iter}: rescore ===");
        let status = Command::new(&python)
            .arg("scripts/score_vae_corpus.py")
            .args(["--dirs", out_dir.to_str().expect("out_dir must be valid UTF-8")])
            .args(["--model-path", vae_model_path.to_str().expect("out_dir must be valid UTF-8")])
            .args(["--out", manifest_path.to_str().expect("out_dir must be valid UTF-8")])
            .status()
            .expect("spawn score_vae_corpus.py");
        if !status.success() { panic!("score_vae_corpus.py failed (exit {status})"); }

        let mean = mean_recon_error(&manifest_path);

        let mut stop_reason: Option<String> = None;
        if let Some(m) = mean {
            let improved = match best_mean {
                None => true,
                Some(best) => m <= best * (1.0 - min_improvement),
            };
            if improved {
                best_mean = Some(m);
                stall_count = 0;
            } else {
                stall_count += 1;
            }
            if let Some(target) = target_recon_mse
                && m <= target {
                stop_reason = Some(format!("target reconstruction MSE {target:.5} reached (mean={m:.5})"));
            }
            if stop_reason.is_none() && stall_count >= patience {
                stop_reason = Some(format!(
                    "no improvement >= {:.1}% over {patience} iterations (best so far = {:.5})",
                    min_improvement * 100.0, best_mean.expect("stall_count > 0 implies best_mean is set")
                ));
            }
        }

        log.log(&serde_json::json!({
            "event": "iteration_summary", "iter": outer_iter, "n_corpus": next_stem,
            "mean_recon_error": mean, "mean_recon_error_prev_iter": prev_mean,
            "best_mean_recon_error": best_mean, "stall_count": stall_count,
        }));
        println!("iteration {outer_iter}: mean recon error = {mean:?} (prev iteration: {prev_mean:?}, best: {best_mean:?}, stall: {stall_count}/{patience})");
        prev_mean = mean;

        if let Some(reason) = stop_reason {
            println!("=== stopping early: {reason} ===");
            break;
        }

        vae_scorer = VaeScorer::new(&vae_model_path);
        if vae_scorer.is_none() {
            eprintln!("warning: vae_scorer_sidecar.py unavailable after training — next iteration's selection will fall back to random");
        }
    }

    println!("\n=== done: {iterations_ran}/{iterations} iterations, {next_stem} zones in corpus (best mean recon error: {best_mean:?}) ===");
    cmd_vae_curate(out_dir, 30, &out_dir.join("curated"), 4000, select_by);
}

// ── Video-zoom exploration ──────────────────────────────────────────────

/// Single-shot (unlike `cmd_vae_explore` — no outer retrain loop, since
/// there's no model to train here): resolves the seed view(s), runs
/// `video_zoom_explore::run`, writes the winners manifest, prints a
/// one-line summary.
#[allow(clippy::too_many_arguments)]
/// Re-run only the time-formula search for one already-planned reel, then
/// re-render its preview.
///
/// The framing and the aim are kept: they are the expensive, deterministic half
/// of planning a shot, and re-deriving them would give the same answer. Only the
/// search is re-rolled, which is what "I like the shot but not the animation"
/// asks for. Runs as a subprocess so the review GUI never renders in its own
/// process.
fn cmd_auto_reel_redo(record_path: &Path, args: &[String]) {
    use nnfractals::auto_reel;
    let Ok(text) = std::fs::read_to_string(record_path) else {
        eprintln!("cannot read {}", record_path.display());
        std::process::exit(2);
    };
    let Ok(mut rec) = serde_json::from_str::<auto_reel::ReelRecord>(&text) else {
        eprintln!("{} is not a reel record", record_path.display());
        std::process::exit(2);
    };
    let Some(batch) = record_path.parent() else {
        eprintln!("no batch directory");
        std::process::exit(2);
    };
    let config = Config::load(Path::new("config.toml")).expect("config.toml");
    let Ok(genome) = io::load_genome(&batch.join(&rec.nn_file)) else {
        eprintln!("cannot load {}", batch.join(&rec.nn_file).display());
        std::process::exit(2);
    };

    println!("re-rolling the time formula for {} ({:.1} doublings)", rec.label, rec.doublings());
    let ga_opts = nnfractals::time_ga::TimeGaOpts {
        clip_frames: rec.preview_frames,
        ..ga_opts_from(args)
    };
    let t = std::time::Instant::now();
    let pop = nnfractals::time_ga::run(&genome, &config, &rec.start, &rec.end, &ga_opts,
                                        &print_gen_report);
    rec.time_summary = nnfractals::time_ga::summary(&pop);
    println!("{}", rec.time_summary);
    match nnfractals::time_ga::best_effort(&pop, ga_opts.full.depths) {
        Some(best) if best.passed() => {
            println!("  {}", best.label());
            rec.time_prog = best.progs.clone();
            rec.time_score = best.score;
            rec.time_loops = best.loops();
        }
        Some(best) => {
            println!("  no formula passed every gate — shipping the closest one for review \
                      ({}): {}", best.rejected.unwrap_or("?"), best.label());
            rec.time_prog = best.progs.clone();
            rec.time_score = best.score;
            rec.time_loops = best.loops();
        }
        None => {
            println!("  no time formula could be evolved at all — the reel stays a plain zoom");
            rec.time_prog.clear();
            rec.time_score = 0.0;
            rec.time_loops = false;
        }
    }
    if let Some(best) = nnfractals::time_ga::best_effort(&pop, ga_opts.full.depths) {
        let views = nnfractals::time_ga::depth_views(&rec.start, &rec.end, ga_opts.full.depths,
                                                       ga_opts.clip_frames);
        let explains = nnfractals::time_ga::explain_individual(
            &genome, &config, best, &views, &ga_opts, &ga_opts.full);
        print_explain(best, rec.start.zoom, &explains);
    }
    rec.secs_evolve = t.elapsed().as_secs_f32();

    // The genome on disk carries the formula, so rewrite it too.
    let mut reel_genome = genome.clone();
    reel_genome.time_prog = rec.time_prog.clone();
    if let Err(e) = save_genome(&reel_genome, &batch.join(&rec.nn_file)) {
        eprintln!("cannot save genome: {e}");
        std::process::exit(3);
    }
    let t = std::time::Instant::now();
    if let Err(e) = render_preview(&reel_genome, &config, &rec, &batch.join(&rec.preview_file)) {
        eprintln!("render failed: {e}");
        std::process::exit(3);
    }
    rec.secs_render = t.elapsed().as_secs_f32();
    // Back to undecided: it is a different clip now.
    rec.status = auto_reel::ReelStatus::Pending;
    if let Err(e) = auto_reel::save_record(batch, &rec) {
        eprintln!("cannot write record: {e}");
        std::process::exit(3);
    }
    println!("re-rolled {} in {:.0}s", rec.label, rec.secs_evolve + rec.secs_render);
}

/// Render one reel's preview clip.
fn render_preview(
    genome: &Genome, config: &Config, rec: &nnfractals::auto_reel::ReelRecord, out_path: &Path,
) -> Result<(), String> {
    use nnfractals::video_export::{self, VideoMsg};
    let (tx, rx) = std::sync::mpsc::channel::<VideoMsg>();
    let waypoints = [rec.start, rec.end];
    if rec.time_prog.is_empty() {
        video_export::export_video_chain(
            genome, config, false, &waypoints, rec.preview_frames, rec.preview_fps,
            rec.preview_w, rec.preview_h, false, false, out_path, &tx, &|| {});
    } else {
        video_export::export_chain_time_video(
            genome, None, config, false, &waypoints, rec.preview_frames, rec.preview_fps,
            rec.preview_w, rec.preview_h, false, false,
            nnfractals::formula::ModShape::Sine, 1.0, 0.0, 0.0, out_path, &tx, &|| {});
    }
    drop(tx);
    match rx.iter().find_map(|m| match m { VideoMsg::Failed(w) => Some(w), _ => None }) {
        Some(why) => Err(why),
        None => Ok(()),
    }
}

/// Renders a `quat-mandelbrot` clip: standalone quaternion-Mandelbrot
/// prototype (see `nnfractals::quat_fractal`/`quat_motion`), deliberately
/// NOT going through `Genome` — `motion.sample(t)` gives a `Slice` + the
/// time-driven C coordinate for each frame, `render_quat_frame` computes its
/// escape-time buffer, and `colormap::apply_colormap` turns that into RGB
/// exactly like every other renderer in this project. Frames feed straight
/// into `encode_rgb_frames`, following `render_preview`'s synchronous
/// (tx,rx) call pattern above.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn cmd_quat_mandelbrot(
    formula: nnfractals::quat_fractal::QuatFormula,
    time_axis: nnfractals::quat_fractal::TimeAxis,
    motion: nnfractals::quat_motion::SliceMotion,
    frames: u32, fps: u32, width: u32, height: u32,
    max_iter: u32, bailout: f64, colormap_name: &str, out_path: &Path,
    overlay_coords: bool,
) {
    use nnfractals::video_export::{encode_rgb_frames, VideoMsg};
    let bailout_sq = bailout * bailout;
    let n = frames.max(2);
    let colormap_name = colormap_name.to_string();
    let (width_us, height_us) = (width as usize, height as usize);

    let frame_iter = (0..n).map(move |i| {
        let t = i as f64 / n as f64;
        let (slice, time_val) = motion.sample(t);
        let et = nnfractals::quat_fractal::render_quat_frame(formula, time_axis, &slice, time_val, width, height, max_iter, bailout_sq);
        let mut rgb = nnfractals::colormap::apply_colormap(&et, max_iter, &colormap_name);
        if overlay_coords {
            // Burned directly into the frame (not a separate file) so the
            // coordinates can never drift out of sync with what's on
            // screen — read straight off the video itself.
            let q = time_axis.assemble(slice.origin, time_val);
            const SCALE: usize = 3;
            const MARGIN: usize = 12;
            let lh = nnfractals::debug_overlay::line_height(SCALE);
            let lines = [
                format!("F{i:04} T={t:.4} AXIS={}", time_axis.name().to_uppercase()),
                format!("R={:>9.4} A={:>9.4}", q.r, q.a),
                format!("B={:>9.4} C={:>9.4}", q.b, q.c),
            ];
            for (li, line) in lines.iter().enumerate() {
                nnfractals::debug_overlay::draw_text(
                    &mut rgb, width_us, height_us, MARGIN, MARGIN + li * lh,
                    line, SCALE, [255, 255, 0], Some([0, 0, 0]),
                );
            }
        }
        rgb
    });
    if let Some(dir) = out_path.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir).expect("create out dir");
        }
    }
    let (tx, rx) = std::sync::mpsc::channel::<VideoMsg>();
    encode_rgb_frames(frame_iter, n, fps, width, height, out_path, &tx, &|| {});
    drop(tx);
    for msg in rx {
        match msg {
            VideoMsg::Started { pid } => eprintln!("ffmpeg started (pid {pid})"),
            VideoMsg::Progress { done, total } => eprint!("\rframe {done}/{total}   "),
            VideoMsg::Done(p) => eprintln!("\ndone: {}", p.display()),
            VideoMsg::Failed(e) => {
                eprintln!("\nFAILED: {e}");
                std::process::exit(1);
            }
        }
    }
}

/// Renders a `quat-voxel-stl`: voxelizes a quaternion fractal at a FIXED C
/// (the time axis every other quat-mandelbrot render animates) into a dense
/// (R,A,B) field, extracts+smooths+decimates a surface mesh, and writes an
/// STL. Prints per-stage timing and triangle counts — this is a heavy,
/// resource-sensitive operation (a dense field at real resolutions is
/// gigabytes; the raw isosurface before decimation can be tens of millions
/// of triangles), so visibility into what actually happened matters more
/// here than for the video renderers above.
fn cmd_quat_voxel_stl(opts: nnfractals::quat_voxel::VoxelStlOpts, out_path: &Path) {
    eprintln!(
        "quat-voxel-stl: {} res={}^3 time_axis={} time_val={:.3} max-iter={} bailout={:.2} -> {}",
        opts.formula.name(), opts.res, opts.time_axis.name(), opts.time_val, opts.max_iter, opts.bailout, out_path.display()
    );
    match nnfractals::quat_voxel::build_voxel_stl(&opts, out_path) {
        Ok(r) => {
            eprintln!(
                "  field {:.1}s | mesh {:.1}s ({} verts, {} tris raw) | smooth {:.1}s | decimate {:.1}s ({} tris final) | write {:.1}s",
                r.field_secs, r.mesh_secs, r.raw_vertices, r.raw_triangles,
                r.smooth_secs, r.decimate_secs, r.final_triangles, r.write_secs
            );
        }
        Err(e) => {
            eprintln!("  FAILED: {e}");
            std::process::exit(1);
        }
    }
}

/// Shared by `quat-raymarch` (one PNG) and `quat-raymarch-video` (many
/// frames, one call per frame): background tinted `bg_color`, hit pixels
/// colored via `colormap::apply_colormap_equalized` (same palette catalog
/// every other render in this project uses, histogram-equalized against
/// THIS frame's own color_t distribution — see that function's doc comment
/// for why a fixed ratio doesn't work here) driven by each surface point's
/// own color-source escape time, scaled by the returned Lambertian shading
/// value for actual 3D depth.
fn raymarch_frame_to_rgb(shading: &[f32], color_t: &[f32], max_iter: u32, colormap_name: &str, bg_color: (f32, f32, f32)) -> Vec<u8> {
    // Background pixels carry a meaningless color_t (see
    // render_raymarch_frame's doc comment) — pin them to max_iter so
    // escape_equalize's own interior filter excludes them from the fit,
    // instead of a big cluster of background zeros compressing every real
    // hit pixel's rank toward the top of the palette.
    let mut color_t_for_fit = color_t.to_vec();
    for (i, &v) in shading.iter().enumerate() {
        if v <= 0.0 {
            color_t_for_fit[i] = max_iter as f32;
        }
    }
    let colored = nnfractals::colormap::apply_colormap_equalized(&color_t_for_fit, max_iter, colormap_name);
    let to_byte = |c: f32| (c.clamp(0.0, 1.0) * 255.0).round() as u8;
    let mut rgb = Vec::with_capacity(shading.len() * 3);
    for (i, &v) in shading.iter().enumerate() {
        if v <= 0.0 {
            rgb.extend_from_slice(&[to_byte(bg_color.0), to_byte(bg_color.1), to_byte(bg_color.2)]);
        } else {
            let base = &colored[i * 3..i * 3 + 3];
            rgb.push((base[0] as f32 * v).clamp(0.0, 255.0) as u8);
            rgb.push((base[1] as f32 * v).clamp(0.0, 255.0) as u8);
            rgb.push((base[2] as f32 * v).clamp(0.0, 255.0) as u8);
        }
    }
    rgb
}

/// Tries the GPU ray-march path (`render_gpu_raymarch`, WGSL compute
/// shader — validated pixel-for-pixel against this exact CPU function in
/// that module's own tests) when `use_gpu` is set and a GPU adapter is
/// available; falls back to the CPU renderer otherwise, including when
/// the binary was built without the `wgpu-backend` feature (off by
/// default only via `--no-default-features`, so this is a rare path in
/// practice, but kept correct rather than a hard compile error).
fn render_raymarch_frame_dispatch(
    params: &nnfractals::quat_raymarch::RaymarchParams,
    cam: &nnfractals::quat_raymarch::RaymarchCamera,
    width: u32, height: u32,
    use_gpu: bool,
) -> (Vec<f32>, Vec<f32>) {
    #[cfg(feature = "wgpu-backend")]
    if use_gpu {
        if let Some(result) = nnfractals::render_gpu_raymarch::render_raymarch_frame_gpu(params, cam, width, height) {
            return result;
        }
        eprintln!("  [gpu] no adapter available, falling back to CPU");
    }
    #[cfg(not(feature = "wgpu-backend"))]
    if use_gpu {
        eprintln!("  [gpu] built without the wgpu-backend feature, falling back to CPU");
    }
    nnfractals::quat_raymarch::render_raymarch_frame(params, cam, width, height)
}

/// Renders and saves one `quat-raymarch` frame as a PNG.
fn cmd_quat_raymarch(
    params: &nnfractals::quat_raymarch::RaymarchParams,
    cam: &nnfractals::quat_raymarch::RaymarchCamera,
    width: u32, height: u32,
    colormap_name: &str,
    bg_color: (f32, f32, f32),
    use_gpu: bool,
    out_path: &Path,
) {
    eprintln!(
        "quat-raymarch: {} time_axis={} time_val={:.3} max_march_steps={} hit_eps={:.2e} step_safety={:.2} domain_radius={:.2} max_iter={} colormap={colormap_name} gpu={use_gpu} {width}x{height} -> {}",
        params.formula.name(), params.time_axis.name(), params.time_val, params.max_march_steps, params.hit_epsilon, params.step_safety, params.domain_radius, params.max_iter, out_path.display()
    );
    let start = std::time::Instant::now();
    let (shading, color_t) = render_raymarch_frame_dispatch(params, cam, width, height, use_gpu);
    let secs = start.elapsed().as_secs_f64();
    let hits = shading.iter().filter(|&&v| v > 0.0).count();
    eprintln!("  {secs:.1}s | {hits}/{} pixels hit ({:.0}%)", shading.len(), 100.0 * hits as f64 / shading.len().max(1) as f64);
    let rgb = raymarch_frame_to_rgb(&shading, &color_t, params.max_iter, colormap_name, bg_color);
    if let Err(e) = io::save_png(&rgb, width, height, out_path) {
        eprintln!("  FAILED: {e}");
        std::process::exit(1);
    }
}

/// Tries the GPU DAG-interpreter path (`render_gpu_raymarch_dag`, a
/// general WGSL register-VM that uploads the genome's program as data —
/// validated pixel-for-pixel against this exact CPU function in that
/// module's own tests) when `use_gpu` is set and a GPU adapter is
/// available; falls back to the CPU renderer otherwise. Mirrors
/// `render_raymarch_frame_dispatch` exactly, for the DAG/genome path.
fn render_raymarch_dag_frame_dispatch(
    params: &nnfractals::quat_dag::RaymarchDagParams,
    cam: &nnfractals::quat_raymarch::RaymarchCamera,
    width: u32, height: u32,
    use_gpu: bool,
) -> (Vec<f32>, Vec<f32>) {
    #[cfg(feature = "wgpu-backend")]
    if use_gpu {
        if let Some(result) = nnfractals::render_gpu_raymarch_dag::render_raymarch_dag_frame_gpu(params, cam, width, height) {
            return result;
        }
        eprintln!("  [gpu] no adapter available, falling back to CPU");
    }
    #[cfg(not(feature = "wgpu-backend"))]
    if use_gpu {
        eprintln!("  [gpu] built without the wgpu-backend feature, falling back to CPU");
    }
    nnfractals::quat_dag::render_raymarch_dag_frame(params, cam, width, height)
}

/// A per-genome GPU rendering strategy, resolved ONCE before a genome's
/// frame(s) render (not re-resolved per frame): prefer a specialized,
/// code-generated pipeline (`render_gpu_raymarch_dag_codegen` —
/// benchmarked far faster than the general interpreter, since the
/// genome's program becomes fixed, unrolled WGSL instead of a runtime
/// register-VM loop), fall back to the general GPU interpreter
/// (`render_gpu_raymarch_dag`) if codegen compilation isn't available,
/// then to CPU.
#[cfg(feature = "wgpu-backend")]
type DagPipelineHandle = nnfractals::render_gpu_raymarch_dag_codegen::CompiledDagPipeline;
#[cfg(not(feature = "wgpu-backend"))]
type DagPipelineHandle = ();

enum DagGpuMode {
    Codegen(DagPipelineHandle),
    Interpreter,
    Cpu,
}

fn resolve_dag_gpu_mode(prog: &[nnfractals::formula::OpNode], warp: &[nnfractals::formula::OpNode], use_gpu: bool) -> DagGpuMode {
    if !use_gpu {
        return DagGpuMode::Cpu;
    }
    #[cfg(feature = "wgpu-backend")]
    {
        if let Some(p) = nnfractals::render_gpu_raymarch_dag_codegen::CompiledDagPipeline::compile(prog, warp) {
            // Deliberately no eprintln on this, the happy path taken once
            // per genome render — at population=120 this drowned out the
            // per-generation summary line under 100x its own volume in the
            // night run's log/terminal (Carl: "some of the messages are
            // useless"). The fallback paths below stay logged since those
            // indicate real degradation, not routine success.
            return DagGpuMode::Codegen(p);
        }
        eprintln!("  [gpu] codegen pipeline unavailable, falling back to the general interpreter");
        if nnfractals::render_gpu_raymarch_dag::gpu_available() {
            return DagGpuMode::Interpreter;
        }
        eprintln!("  [gpu] no adapter available, falling back to CPU");
    }
    #[cfg(not(feature = "wgpu-backend"))]
    {
        eprintln!("  [gpu] built without the wgpu-backend feature, falling back to CPU");
    }
    DagGpuMode::Cpu
}

fn render_dag_frame_with_mode(
    mode: &mut DagGpuMode,
    params: &nnfractals::quat_dag::RaymarchDagParams,
    cam: &nnfractals::quat_raymarch::RaymarchCamera,
    width: u32, height: u32,
) -> (Vec<f32>, Vec<f32>) {
    match mode {
        #[cfg(feature = "wgpu-backend")]
        DagGpuMode::Codegen(p) => p.render(params, cam, width, height),
        #[cfg(not(feature = "wgpu-backend"))]
        DagGpuMode::Codegen(_) => unreachable!("DagGpuMode::Codegen is never constructed without wgpu-backend"),
        DagGpuMode::Interpreter => render_raymarch_dag_frame_dispatch(params, cam, width, height, true),
        DagGpuMode::Cpu => nnfractals::quat_dag::render_raymarch_dag_frame(params, cam, width, height),
    }
}

/// Renders and saves one `quat-raymarch-genome` frame as a PNG — same
/// pipeline as `cmd_quat_raymarch`, but the fractal function is an
/// ARBITRARY loaded GA genome's expression-DAG (`quat_dag`'s quaternion
/// port of `formula.rs`'s 21-opcode register-VM) instead of one of the 10
/// hand-built `QuatFormula` variants.
fn cmd_quat_raymarch_dag(
    params: &nnfractals::quat_dag::RaymarchDagParams,
    cam: &nnfractals::quat_raymarch::RaymarchCamera,
    genome_label: &str,
    width: u32, height: u32,
    colormap_name: &str,
    bg_color: (f32, f32, f32),
    use_gpu: bool,
    out_path: &Path,
) {
    eprintln!(
        "quat-raymarch-genome: {genome_label} time_axis={} time_val={:.3} max_march_steps={} hit_eps={:.2e} step_safety={:.2} domain_radius={:.2} max_iter={} colormap={colormap_name} gpu={use_gpu} {width}x{height} -> {}",
        params.time_axis.name(), params.time_val, params.max_march_steps, params.hit_epsilon, params.step_safety, params.domain_radius, params.max_iter, out_path.display()
    );
    let start = std::time::Instant::now();
    let mut mode = resolve_dag_gpu_mode(params.formula.prog, params.formula.warp, use_gpu);
    let (shading, color_t) = render_dag_frame_with_mode(&mut mode, params, cam, width, height);
    let secs = start.elapsed().as_secs_f64();
    let hits = shading.iter().filter(|&&v| v > 0.0).count();
    eprintln!("  {secs:.1}s | {hits}/{} pixels hit ({:.0}%)", shading.len(), 100.0 * hits as f64 / shading.len().max(1) as f64);
    let rgb = raymarch_frame_to_rgb(&shading, &color_t, params.max_iter, colormap_name, bg_color);
    if let Err(e) = io::save_png(&rgb, width, height, out_path) {
        eprintln!("  FAILED: {e}");
        std::process::exit(1);
    }
}

/// Renders a `quat-raymarch-genome-video`: the video (orbit + optional
/// C-pulse) twin of `cmd_quat_raymarch_video`, but ray-marching an
/// ARBITRARY loaded GA genome's expression-DAG instead of a hand-built
/// `QuatFormula`. Mirrors `cmd_quat_raymarch_video` exactly.
#[allow(clippy::too_many_arguments)]
fn cmd_quat_raymarch_dag_video(
    params: nnfractals::quat_dag::RaymarchDagParams,
    orbit: nnfractals::quat_raymarch::RaymarchOrbitParams,
    pulse: CPulseParams,
    genome_label: String,
    frames: u32, fps: u32, width: u32, height: u32,
    colormap_name: String,
    bg_color: (f32, f32, f32),
    use_gpu: bool,
    out_path: &Path,
) {
    use nnfractals::video_export::{encode_rgb_frames, VideoMsg};
    eprintln!(
        "quat-raymarch-genome-video: {genome_label} frames={frames} fps={fps} {width}x{height} radius={:.2} turns={:.2} c0={:.2} c1={:.2} c_shape={:?} c_freq={:.2} colormap={colormap_name} gpu={use_gpu} -> {}",
        orbit.radius, orbit.turns, pulse.c0, pulse.c1, pulse.shape, pulse.freq, out_path.display()
    );
    // Resolved (and, for the codegen path, compiled) ONCE for the whole
    // clip — the entire point of the codegen path is that this one-time
    // shader-compile cost is amortized over every frame, not repeated.
    let mut mode = resolve_dag_gpu_mode(params.formula.prog, params.formula.warp, use_gpu);
    let n = frames.max(2);
    let frame_iter = (0..n).map(move |i| {
        let t = i as f64 / n as f64;
        let cam = orbit.sample(t);
        let time_val = nnfractals::formula::ramp_or_pulse(pulse.c0, pulse.c1, pulse.shape, pulse.freq, pulse.phase, t);
        let frame_params = nnfractals::quat_dag::RaymarchDagParams { time_val, ..params };
        let (shading, color_t) = render_dag_frame_with_mode(&mut mode, &frame_params, &cam, width, height);
        raymarch_frame_to_rgb(&shading, &color_t, frame_params.max_iter, &colormap_name, bg_color)
    });
    if let Some(dir) = out_path.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir).expect("create out dir");
        }
    }
    let (tx, rx) = std::sync::mpsc::channel::<VideoMsg>();
    encode_rgb_frames(frame_iter, n, fps, width, height, out_path, &tx, &|| {});
    drop(tx);
    for msg in rx {
        match msg {
            VideoMsg::Started { pid } => eprintln!("ffmpeg started (pid {pid})"),
            VideoMsg::Progress { done, total } => eprint!("\rframe {done}/{total}   "),
            VideoMsg::Done(p) => eprintln!("\ndone: {}", p.display()),
            VideoMsg::Failed(e) => {
                eprintln!("\nFAILED: {e}");
                std::process::exit(1);
            }
        }
    }
}

/// How far back the camera needs to sit so the `domain_radius` bounding
/// sphere occupies a sensible, well-margined fraction of frame instead of
/// filling it edge-to-edge — the second bug Carl caught by eye (the HD
/// portrait batch's objects had "no margin, letting you never appreciate
/// the outline"). Root cause: a FIXED vertical FOV (`fov_y`) combined with
/// `half_w = half_h * aspect` narrows the effective horizontal FOV a lot
/// for a portrait frame (aspect = width/height < 1) — at the SAME eye
/// distance that framed the object nicely in a square probe, the same
/// object nearly fills a 1080x1920 portrait frame, because the tighter
/// (horizontal) axis is now the binding constraint, not the vertical one
/// the eye distance was originally tuned against.
///
/// `fill_frac` is the target: the domain sphere's angular half-size, as a
/// fraction of the TIGHTER axis's half-FOV. First calibrated (0.822)
/// against genome `a6598cb3b21c491b` gave a well-margined but, per Carl's
/// direct feedback on the rendered batch, TOO DISTANT a frame ("I do not
/// want to be this far apart from the object"). Recalibrated against
/// genome `9aaae7276b77b52c` at domain_radius=1.6 — picked by comparing
/// several explicit distances by eye (eye=4 clipped the silhouette, eye=5
/// filled the frame nicely with the outline still fully visible) and
/// converting the chosen eye=5/fov=45 pair back to a fill_frac, then
/// cross-checked independently at `fov_deg=37` (radius≈6.12, visually
/// confirmed ~59% frame fill vs. the eye=5/fov=45 baseline's 60%) — the
/// same fraction holding across two different FOVs is what makes this a
/// reusable formula rather than one hardcoded magic number.
///
/// **That recalibration landed on `FILL_FRAC = 0.82`, not `1.42`** —
/// `1.42` sat in this constant for a while (also duplicated into
/// `quat_viewer.rs`, since fixed to match) before being caught: solving
/// the formula below backwards from the same eye=5/fov=45/domain=1.6
/// case this comment cites gives `sin⁻¹(1.6/5.0) / (45°/2) ≈ 0.83`, and
/// the fov=37° cross-check gives the same ≈0.82 independently — whoever
/// wrote `1.42` into the constant made an arithmetic slip converting the
/// chosen eye/fov pair back to a fraction, not a bad calibration choice.
/// Any `FILL_FRAC > 1.0` is mathematically guaranteed to crop the render
/// (it places the target angular half-size OUTSIDE the camera's own
/// half-FOV), which is exactly the "too zoomed in" symptom Carl reported
/// in both the interactive viewer and the gallery thumbnails — every
/// framing call site in this codebase shares this one function. Revisit
/// again if Carl's sense of the right distance shifts further, but
/// derive any new constant the same way (an explicit eye distance judged
/// good by eye, solved backwards through this exact formula) rather than
/// re-deriving by hand off to the side, which is what produced the slip.
fn recommended_orbit_radius(domain_radius: f64, fov_deg: f64, width: u32, height: u32) -> f64 {
    const FILL_FRAC: f64 = 0.82;
    let aspect = width as f64 / height.max(1) as f64;
    let half_fov_y = (fov_deg / 2.0).to_radians();
    let half_fov_x = (half_fov_y.tan() * aspect).atan();
    let tight_half_fov = half_fov_y.min(half_fov_x);
    let target_angular_half_size = FILL_FRAC * tight_half_fov;
    domain_radius / target_angular_half_size.sin()
}

/// Probe configurations used by `score_genome_dag`: 2 camera angles
/// (front-on plus a 3/4 turn) crossed with 2 time-axis (C) values
/// spanning the actual video's pulse range (`--c0/--c1` default to
/// ±0.6) — so a genome that looks solid at C=0 but balloons or
/// degenerates at the pulse extremes doesn't score well on a check that
/// only ever looked at one moment of its clip (mirrors this project's
/// existing "gate on worst moment, not average" principle for video
/// time-formulas). Camera distance uses `recommended_orbit_radius` at the
/// SQUARE probe aspect, so probe framing matches what a properly-framed
/// video would show, not the too-close bug being fixed here.
fn score_probes(domain_radius: f64, probe_size: u32) -> Vec<(nnfractals::quat_raymarch::RaymarchCamera, f64)> {
    let fov_deg: f64 = 45.0;
    let fov_y = fov_deg.to_radians();
    let radius = recommended_orbit_radius(domain_radius, fov_deg, probe_size, probe_size);
    let cams = [
        nnfractals::quat_raymarch::RaymarchCamera { eye: (0.0, 0.0, -radius), target: (0.0, 0.0, 0.0), up_hint: (0.0, 1.0, 0.0), fov_y },
        nnfractals::quat_raymarch::RaymarchCamera {
            eye: (-radius * 0.65, radius * 0.4, -radius * 0.65),
            target: (0.0, 0.0, 0.0),
            up_hint: (0.0, 1.0, 0.0),
            fov_y,
        },
    ];
    let c_values = [0.0, 0.5];
    let mut probes = Vec::with_capacity(cams.len() * c_values.len());
    for cam in cams {
        for &c in &c_values {
            probes.push((
                nnfractals::quat_raymarch::RaymarchCamera { eye: cam.eye, target: cam.target, up_hint: cam.up_hint, fov_y: cam.fov_y },
                c,
            ));
        }
    }
    probes
}

/// Scores one genome's genuinely-3D-ness / visual quality per
/// `quat_dag_fitness`'s breakdown: `anisotropy` (cheap, no rendering) plus
/// low-res probe renders from `score_probes` for coverage/solidity/
/// shading-richness/color-entropy/silhouette-irregularity, averaged over
/// camera angle AND C-pulse position. `probe_size` is the square
/// resolution of each probe render (96 is plenty — this only needs to
/// rank genomes, not look good).
fn score_genome_dag(genome: &Genome, probe_size: u32, use_gpu: bool) -> nnfractals::quat_dag_fitness::QuatFitnessBreakdown {
    let formula = nnfractals::quat_dag::QuatDagFormula {
        prog: &genome.program,
        warp: &genome.warp,
        julia: genome.julia_mode,
        jc: (genome.julia_cre, genome.julia_cim),
        phoenix: (genome.phoenix_re, genome.phoenix_im),
    };
    let bailout_sq = (genome.bailout_radius as f64) * (genome.bailout_radius as f64);
    let anisotropy = nnfractals::quat_dag::anisotropy_score(&formula, 60, bailout_sq);

    let domain_radius = 1.6;
    let base_params = nnfractals::quat_dag::RaymarchDagParams {
        formula,
        time_axis: nnfractals::quat_fractal::TimeAxis::C,
        time_val: 0.0,
        domain_radius,
        max_iter: 60,
        bailout: genome.bailout_radius as f64,
        max_march_steps: 150,
        hit_epsilon: domain_radius * 1e-4,
        step_safety: 0.8,
        light_dir: (0.5, 0.8, 0.3),
        normal_eps: domain_radius * 1e-3,
        color_probe_offset: domain_radius * 1e-2,
        aa: 1,
    };
    let mut mode = resolve_dag_gpu_mode(base_params.formula.prog, base_params.formula.warp, use_gpu);
    let per_view: Vec<(f32, f32, f32, f32, f32)> = score_probes(domain_radius, probe_size)
        .iter()
        .map(|(cam, time_val)| {
            let params = nnfractals::quat_dag::RaymarchDagParams { time_val: *time_val, ..base_params };
            let (shading, color_t) = render_dag_frame_with_mode(&mut mode, &params, cam, probe_size, probe_size);
            nnfractals::quat_dag_fitness::view_metrics(&shading, &color_t, probe_size, probe_size)
        })
        .collect();
    nnfractals::quat_dag_fitness::combine_views(anisotropy, &per_view)
}

/// Every `quat_*` field `Genome` can persist — the original 6-component
/// breakdown (never actually saved to disk before now, despite being
/// computed every evolve run) plus the ~24 new metrics from
/// `quat_dag_fitness`'s extended module. Field names match `Genome`'s
/// `quat_*` fields 1:1 so callers can just destructure-and-assign.
struct QuatFullMetrics {
    breakdown: nnfractals::quat_dag_fitness::QuatFitnessBreakdown,
    extended: nnfractals::quat_dag_fitness::QuatExtendedMetrics,
    cross_view_iou: f32,
    cross_view_coverage_delta: f32,
    c_sensitivity: f32,
    c_coverage_range: f32,
    node_count: f32,
    opcode_diversity: f32,
    max_depth: f32,
    warp_node_count: f32,
    warp_opcode_diversity: f32,
}

/// Full metric sweep for one genome — everything `score_genome_dag`
/// already computes (unchanged, still what selection during evolution
/// uses — this function is NOT on that hot path) plus every extended
/// metric, reusing the exact same 4 probe renders (`score_probes`: 2
/// camera angles × 2 C values) rather than rendering anything extra.
/// Probe index 0/2 share C=0.0 at different cameras (cross-view IoU/
/// delta); probe index 0/1 share camera 0 at different C (C-sensitivity/
/// range) — see `score_probes`'s own doc comment for the exact layout.
/// Deliberately kept OUT of `score_individual`'s per-generation hot path
/// (used only when a genome is actually being saved, or by
/// `quat-dag-rescore`) — the extra box-counting/convex-hull/lacunarity/
/// symmetry passes cost real CPU time that a 200-genome/40-generation
/// evolve run shouldn't pay for on every child that never survives.
fn score_genome_dag_full(genome: &Genome, probe_size: u32, use_gpu: bool) -> QuatFullMetrics {
    let formula = nnfractals::quat_dag::QuatDagFormula {
        prog: &genome.program,
        warp: &genome.warp,
        julia: genome.julia_mode,
        jc: (genome.julia_cre, genome.julia_cim),
        phoenix: (genome.phoenix_re, genome.phoenix_im),
    };
    let bailout_sq = (genome.bailout_radius as f64) * (genome.bailout_radius as f64);
    let anisotropy = nnfractals::quat_dag::anisotropy_score(&formula, 60, bailout_sq);

    let domain_radius = 1.6;
    let base_params = nnfractals::quat_dag::RaymarchDagParams {
        formula,
        time_axis: nnfractals::quat_fractal::TimeAxis::C,
        time_val: 0.0,
        domain_radius,
        max_iter: 60,
        bailout: genome.bailout_radius as f64,
        max_march_steps: 150,
        hit_epsilon: domain_radius * 1e-4,
        step_safety: 0.8,
        light_dir: (0.5, 0.8, 0.3),
        normal_eps: domain_radius * 1e-3,
        color_probe_offset: domain_radius * 1e-2,
        aa: 1,
    };
    let mut mode = resolve_dag_gpu_mode(base_params.formula.prog, base_params.formula.warp, use_gpu);

    let mut basic_per_view = Vec::with_capacity(4);
    let mut extended_per_view = Vec::with_capacity(4);
    let mut shadings: Vec<Vec<f32>> = Vec::with_capacity(4);
    for (cam, time_val) in score_probes(domain_radius, probe_size) {
        let params = nnfractals::quat_dag::RaymarchDagParams { time_val, ..base_params };
        let (shading, color_t) = render_dag_frame_with_mode(&mut mode, &params, &cam, probe_size, probe_size);
        basic_per_view.push(nnfractals::quat_dag_fitness::view_metrics(&shading, &color_t, probe_size, probe_size));
        extended_per_view.push(nnfractals::quat_dag_fitness::view_extended_metrics(&shading, &color_t, probe_size, probe_size, base_params.max_iter as f32));
        shadings.push(shading);
    }
    let breakdown = nnfractals::quat_dag_fitness::combine_views(anisotropy, &basic_per_view);
    let extended = nnfractals::quat_dag_fitness::combine_extended_views(&extended_per_view);

    // score_probes lays out probes as [cam0@c0, cam0@c1, cam1@c0, cam1@c1].
    let cross_view_iou = nnfractals::quat_dag_fitness::hitmask_iou(&shadings[0], &shadings[2]);
    let cross_view_coverage_delta = nnfractals::quat_dag_fitness::coverage_delta(&shadings[0], &shadings[2]);
    let c_iou = nnfractals::quat_dag_fitness::hitmask_iou(&shadings[0], &shadings[1]);
    let c_sensitivity = (1.0 - c_iou).clamp(0.0, 1.0);
    let c_coverage_range = nnfractals::quat_dag_fitness::coverage_delta(&shadings[0], &shadings[1]);

    let (node_count, opcode_diversity, max_depth) = nnfractals::quat_dag_fitness::structural_metrics(&genome.program);
    let (warp_node_count, warp_opcode_diversity, _warp_max_depth) = nnfractals::quat_dag_fitness::structural_metrics(&genome.warp);

    QuatFullMetrics {
        breakdown, extended, cross_view_iou, cross_view_coverage_delta, c_sensitivity, c_coverage_range,
        node_count, opcode_diversity, max_depth, warp_node_count, warp_opcode_diversity,
    }
}

/// Same metric machinery as `score_genome_dag_full`, but for a classic
/// hardcoded `QuatFormula` (Mandelbulb etc.) instead of an evolved DAG
/// genome — Carl's ask: "analyse mandelbulb on all 4D and make it your
/// role model. Compare its metrics to all the top ranking fractals."
/// "On all 4D" is taken literally: `--time-axis` picks which quaternion
/// component (R/A/B/C) the time-driven value fills and the other three
/// become the spatial subspace explored (see `TimeAxis`'s own docs) — a
/// single render only ever sees ONE of the four possible 3D cross-
/// sections through the full 4D object, so this renders and scores all
/// four and returns one result per axis, exactly `score_probes`'s 2
/// camera x 2 time-value layout each time (identical framing to every
/// evolved genome's own scoring, so the numbers are directly
/// comparable). `anisotropy` is intentionally left out of the returned
/// breakdown's meaning here: it's `quat_dag::anisotropy_score`, a purely
/// analytic shortcut over a DAG `program`'s structure that classic
/// formulas don't have — passing a fake value would silently corrupt
/// any `.total()` a caller took, so callers of this function must not
/// call `.total()` on the returned breakdown, only read the other 5
/// fields plus `extended`/cross-view/time-sensitivity.
fn score_quat_formula_full(
    formula: nnfractals::quat_fractal::QuatFormula,
    axis: nnfractals::quat_fractal::TimeAxis,
    probe_size: u32,
    use_gpu: bool,
) -> QuatFullMetrics {
    let domain_radius = 1.6;
    let base_params = nnfractals::quat_raymarch::RaymarchParams {
        formula,
        time_axis: axis,
        time_val: 0.0,
        domain_radius,
        max_iter: 60,
        bailout: 4.0,
        max_march_steps: 150,
        hit_epsilon: domain_radius * 1e-4,
        step_safety: 0.8,
        light_dir: (0.5, 0.8, 0.3),
        normal_eps: domain_radius * 1e-3,
        color_probe_offset: domain_radius * 1e-2,
        aa: 1,
        bulb_power: 8.0,
        mandelbox_scale: -1.5,
    };
    let mut basic_per_view = Vec::with_capacity(4);
    let mut extended_per_view = Vec::with_capacity(4);
    let mut shadings: Vec<Vec<f32>> = Vec::with_capacity(4);
    for (cam, time_val) in score_probes(domain_radius, probe_size) {
        let params = nnfractals::quat_raymarch::RaymarchParams { time_val, ..base_params };
        let (shading, color_t) = render_raymarch_frame_dispatch(&params, &cam, probe_size, probe_size, use_gpu);
        basic_per_view.push(nnfractals::quat_dag_fitness::view_metrics(&shading, &color_t, probe_size, probe_size));
        extended_per_view.push(nnfractals::quat_dag_fitness::view_extended_metrics(&shading, &color_t, probe_size, probe_size, base_params.max_iter as f32));
        shadings.push(shading);
    }
    let breakdown = nnfractals::quat_dag_fitness::combine_views(0.0, &basic_per_view);
    let extended = nnfractals::quat_dag_fitness::combine_extended_views(&extended_per_view);
    let cross_view_iou = nnfractals::quat_dag_fitness::hitmask_iou(&shadings[0], &shadings[2]);
    let cross_view_coverage_delta = nnfractals::quat_dag_fitness::coverage_delta(&shadings[0], &shadings[2]);
    let c_iou = nnfractals::quat_dag_fitness::hitmask_iou(&shadings[0], &shadings[1]);
    let c_sensitivity = (1.0 - c_iou).clamp(0.0, 1.0);
    let c_coverage_range = nnfractals::quat_dag_fitness::coverage_delta(&shadings[0], &shadings[1]);
    QuatFullMetrics {
        breakdown, extended, cross_view_iou, cross_view_coverage_delta, c_sensitivity, c_coverage_range,
        // No DAG program exists for a classic formula — 0.0 here means
        // "not applicable", not "zero complexity"; excluded from the
        // JSON this feeds downstream (see cmd_quat_formula_full_metrics).
        node_count: 0.0, opcode_diversity: 0.0, max_depth: 0.0, warp_node_count: 0.0, warp_opcode_diversity: 0.0,
    }
}

/// `quat-formula-full-metrics` — renders and scores a classic formula
/// (default: bulb, the one genuine Mandelbulb-style angle-multiplication
/// formula in the catalog, see `quat_fractal.rs`'s module docs for why
/// the others are solids of revolution) across all 4 `TimeAxis` choices,
/// writes one JSON object per axis plus an all-axes average to
/// `--out`, and prints a human-readable summary. Feeds
/// `scripts/quat_role_model_report.py`'s comparison against the evolved
/// archive's saved `quat_*` fields.
fn cmd_quat_formula_full_metrics(formula_name: &str, probe_size: u32, use_gpu: bool, out_path: &Path) {
    let formula = nnfractals::quat_fractal::QuatFormula::parse(formula_name).unwrap_or_else(|| {
        let names: Vec<&str> = nnfractals::quat_fractal::QuatFormula::ALL.iter().map(|f| f.name()).collect();
        panic!("unknown --formula {formula_name:?} — expected one of: {}", names.join(", "))
    });
    let axes = [
        nnfractals::quat_fractal::TimeAxis::R,
        nnfractals::quat_fractal::TimeAxis::A,
        nnfractals::quat_fractal::TimeAxis::B,
        nnfractals::quat_fractal::TimeAxis::C,
    ];
    let mut per_axis = serde_json::Map::new();
    let mut sums = nnfractals::quat_dag_fitness::QuatExtendedMetrics::default();
    let mut basic_sums = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32); // coverage, solidity, shading_richness, color_entropy, silhouette_irregularity
    let mut cross_iou_sum = 0.0f32;
    let mut cross_delta_sum = 0.0f32;
    let mut c_sens_sum = 0.0f32;
    let mut c_range_sum = 0.0f32;
    for axis in axes {
        eprintln!("quat-formula-full-metrics: {} time_axis={} ...", formula.name(), axis.name());
        let m = score_quat_formula_full(formula, axis, probe_size, use_gpu);
        let b = &m.breakdown;
        basic_sums.0 += b.coverage; basic_sums.1 += b.solidity; basic_sums.2 += b.shading_richness;
        basic_sums.3 += b.color_entropy; basic_sums.4 += b.silhouette_irregularity;
        sums.box_dim += m.extended.box_dim; sums.lacunarity += m.extended.lacunarity; sums.convexity += m.extended.convexity;
        sums.isoperimetric += m.extended.isoperimetric; sums.bilateral_symmetry += m.extended.bilateral_symmetry;
        sums.centroid_offset += m.extended.centroid_offset; sums.largest_component_frac += m.extended.largest_component_frac;
        sums.shading_gradient += m.extended.shading_gradient; sums.shading_skewness += m.extended.shading_skewness;
        sums.specular_fraction += m.extended.specular_fraction; sums.crevice_fraction += m.extended.crevice_fraction;
        sums.color_gradient += m.extended.color_gradient; sums.color_shading_corr += m.extended.color_shading_corr;
        sums.color_band_autocorr += m.extended.color_band_autocorr; sums.color_range_utilization += m.extended.color_range_utilization;
        cross_iou_sum += m.cross_view_iou; cross_delta_sum += m.cross_view_coverage_delta;
        c_sens_sum += m.c_sensitivity; c_range_sum += m.c_coverage_range;
        eprintln!(
            "  coverage={:.3} solidity={:.3} shading_richness={:.3} color_entropy={:.3} silhouette_irregularity={:.3}",
            b.coverage, b.solidity, b.shading_richness, b.color_entropy, b.silhouette_irregularity
        );
        eprintln!(
            "  box_dim={:.3} lacunarity={:.3} convexity={:.3} isoperimetric={:.3} bilateral_symmetry={:.3} color_shading_corr={:.3}",
            m.extended.box_dim, m.extended.lacunarity, m.extended.convexity, m.extended.isoperimetric, m.extended.bilateral_symmetry, m.extended.color_shading_corr
        );
        per_axis.insert(axis.name().to_string(), serde_json::json!({
            "quat_coverage": b.coverage, "quat_solidity": b.solidity, "quat_shading_richness": b.shading_richness,
            "quat_color_entropy": b.color_entropy, "quat_silhouette_irregularity": b.silhouette_irregularity,
            "quat_box_dim": m.extended.box_dim, "quat_lacunarity": m.extended.lacunarity, "quat_convexity": m.extended.convexity,
            "quat_isoperimetric": m.extended.isoperimetric, "quat_bilateral_symmetry": m.extended.bilateral_symmetry,
            "quat_centroid_offset": m.extended.centroid_offset, "quat_largest_component_frac": m.extended.largest_component_frac,
            "quat_shading_gradient": m.extended.shading_gradient, "quat_shading_skewness": m.extended.shading_skewness,
            "quat_specular_fraction": m.extended.specular_fraction, "quat_crevice_fraction": m.extended.crevice_fraction,
            "quat_color_gradient": m.extended.color_gradient, "quat_color_shading_corr": m.extended.color_shading_corr,
            "quat_color_band_autocorr": m.extended.color_band_autocorr, "quat_color_range_utilization": m.extended.color_range_utilization,
            "quat_cross_view_iou": m.cross_view_iou, "quat_cross_view_coverage_delta": m.cross_view_coverage_delta,
            "quat_c_sensitivity": m.c_sensitivity, "quat_c_coverage_range": m.c_coverage_range,
        }));
    }
    let n = axes.len() as f32;
    let average = serde_json::json!({
        "quat_coverage": basic_sums.0 / n, "quat_solidity": basic_sums.1 / n, "quat_shading_richness": basic_sums.2 / n,
        "quat_color_entropy": basic_sums.3 / n, "quat_silhouette_irregularity": basic_sums.4 / n,
        "quat_box_dim": sums.box_dim / n, "quat_lacunarity": sums.lacunarity / n, "quat_convexity": sums.convexity / n,
        "quat_isoperimetric": sums.isoperimetric / n, "quat_bilateral_symmetry": sums.bilateral_symmetry / n,
        "quat_centroid_offset": sums.centroid_offset / n, "quat_largest_component_frac": sums.largest_component_frac / n,
        "quat_shading_gradient": sums.shading_gradient / n, "quat_shading_skewness": sums.shading_skewness / n,
        "quat_specular_fraction": sums.specular_fraction / n, "quat_crevice_fraction": sums.crevice_fraction / n,
        "quat_color_gradient": sums.color_gradient / n, "quat_color_shading_corr": sums.color_shading_corr / n,
        "quat_color_band_autocorr": sums.color_band_autocorr / n, "quat_color_range_utilization": sums.color_range_utilization / n,
        "quat_cross_view_iou": cross_iou_sum / n, "quat_cross_view_coverage_delta": cross_delta_sum / n,
        "quat_c_sensitivity": c_sens_sum / n, "quat_c_coverage_range": c_range_sum / n,
    });
    let out = serde_json::json!({ "formula": formula.name(), "per_axis": per_axis, "average_all_4_axes": average });
    std::fs::write(out_path, serde_json::to_string_pretty(&out).unwrap()).unwrap_or_else(|e| panic!("failed to write {out_path:?}: {e}"));
    eprintln!("quat-formula-full-metrics: wrote {out_path:?}");
}

/// Non-finite `f32` -> `0.0`. `QuatExtendedMetrics` already sanitizes
/// itself internally, but `m.breakdown` (`QuatFitnessBreakdown`, from
/// the existing `view_metrics`/`combine_views` — battle-tested across a
/// full night of evolution, but never previously SERIALIZED, since
/// scores lived in memory only) and the standalone cross-view/C/
/// structural scalars have no such guard. Confirmed this is a real risk,
/// not theoretical: an unstable evolved formula produced an infinite
/// shading value during testing, and a non-finite `f32` field silently
/// serializes to JSON `null` (serde_json's documented behavior), which
/// then fails to DESERIALIZE back into `f32` — corrupting the genome
/// file's next load. This is the single choke point every `quat_*`
/// field passes through on the way into `Genome`, so it's the one place
/// this needs guarding regardless of which upstream computation is
/// responsible.
fn finite_or_zero(v: f32) -> f32 {
    if v.is_finite() { v } else { 0.0 }
}

/// Writes every field `score_genome_dag_full` computed onto `g`'s
/// `quat_*` fields, in place.
fn apply_quat_full_metrics(g: &mut Genome, m: &QuatFullMetrics) {
    g.fractal_kind = "quat".to_string();
    g.quat_anisotropy = finite_or_zero(m.breakdown.anisotropy);
    g.quat_coverage = finite_or_zero(m.breakdown.coverage);
    g.quat_solidity = finite_or_zero(m.breakdown.solidity);
    g.quat_shading_richness = finite_or_zero(m.breakdown.shading_richness);
    g.quat_color_entropy = finite_or_zero(m.breakdown.color_entropy);
    g.quat_silhouette_irregularity = finite_or_zero(m.breakdown.silhouette_irregularity);

    g.quat_box_dim = m.extended.box_dim;
    g.quat_lacunarity = m.extended.lacunarity;
    g.quat_convexity = m.extended.convexity;
    g.quat_isoperimetric = m.extended.isoperimetric;
    g.quat_bilateral_symmetry = m.extended.bilateral_symmetry;
    g.quat_centroid_offset = m.extended.centroid_offset;
    g.quat_largest_component_frac = m.extended.largest_component_frac;
    g.quat_shading_gradient = m.extended.shading_gradient;
    g.quat_shading_skewness = m.extended.shading_skewness;
    g.quat_specular_fraction = m.extended.specular_fraction;
    g.quat_crevice_fraction = m.extended.crevice_fraction;
    g.quat_color_gradient = m.extended.color_gradient;
    g.quat_color_shading_corr = m.extended.color_shading_corr;
    g.quat_color_band_autocorr = m.extended.color_band_autocorr;
    g.quat_color_range_utilization = m.extended.color_range_utilization;

    g.quat_cross_view_iou = finite_or_zero(m.cross_view_iou);
    g.quat_cross_view_coverage_delta = finite_or_zero(m.cross_view_coverage_delta);
    g.quat_c_sensitivity = finite_or_zero(m.c_sensitivity);
    g.quat_c_coverage_range = finite_or_zero(m.c_coverage_range);

    g.quat_node_count = finite_or_zero(m.node_count);
    g.quat_opcode_diversity = finite_or_zero(m.opcode_diversity);
    g.quat_max_depth = finite_or_zero(m.max_depth);
    g.quat_warp_node_count = finite_or_zero(m.warp_node_count);
    g.quat_warp_opcode_diversity = finite_or_zero(m.warp_opcode_diversity);
}

/// Computes and writes the 5 whole-4D-object organization metrics
/// (`quat_organization.rs`) onto `g` — a completely separate code path
/// from `score_genome_dag_full`/`apply_quat_full_metrics` above: no
/// render, no camera, no `TimeAxis` choice, every one of R/A/B/C sampled
/// as an equal spatial coordinate. Domain radius matches the render-based
/// probes' own convention (1.6) so the two families of metrics stay
/// comparable in scale even though they're computed completely
/// differently; `max_iter`/`bailout_sq` come from the genome's own
/// `bailout_radius`, same convention `score_genome_dag_full` uses.
fn apply_organization_metrics(g: &mut Genome) {
    let formula = nnfractals::quat_dag::QuatDagFormula {
        prog: &g.program,
        warp: &g.warp,
        julia: g.julia_mode,
        jc: (g.julia_cre, g.julia_cim),
        phoenix: (g.phoenix_re, g.phoenix_im),
    };
    let bailout_sq = (g.bailout_radius as f64) * (g.bailout_radius as f64);
    let m = nnfractals::quat_organization::compute_organization_metrics(&formula, 1.6, 60, bailout_sq);
    g.quat_organization_statistical = finite_or_zero(m.statistical_complexity);
    g.quat_organization_ordinal = finite_or_zero(m.ordinal_complexity);
    g.quat_organization_multifractal = finite_or_zero(m.multifractal_width);
    g.quat_organization_compression = finite_or_zero(m.compression_complexity);
    g.quat_organization_chaoticity = finite_or_zero(m.chaoticity);
    g.quat_sphericity = finite_or_zero(m.sphericity);
}

/// Reads one named `quat_*` metric straight off a `QuatFullMetrics` —
/// the same right-hand sides as `apply_quat_full_metrics`, just returned
/// instead of assigned onto a `Genome`. Exists so `quat_pref::
/// QuatPrefModel::score` (a trained model's feature-name -> value
/// lookup) can be driven directly off a freshly-computed
/// `QuatFullMetrics` during evolution, without round-tripping through a
/// saved `Genome`. Unknown names return 0.0 (a model trained against a
/// newer/older metric set than this binary knows about degrades
/// gracefully rather than panicking mid-evolution); see
/// `quat_full_metrics_feature_matches_every_model_field_name` below for
/// the test that keeps this in sync with `apply_quat_full_metrics`.
fn quat_full_metrics_feature(m: &QuatFullMetrics, name: &str) -> f32 {
    match name {
        "quat_anisotropy" => finite_or_zero(m.breakdown.anisotropy),
        "quat_coverage" => finite_or_zero(m.breakdown.coverage),
        "quat_solidity" => finite_or_zero(m.breakdown.solidity),
        "quat_shading_richness" => finite_or_zero(m.breakdown.shading_richness),
        "quat_color_entropy" => finite_or_zero(m.breakdown.color_entropy),
        "quat_silhouette_irregularity" => finite_or_zero(m.breakdown.silhouette_irregularity),
        "quat_box_dim" => m.extended.box_dim,
        "quat_lacunarity" => m.extended.lacunarity,
        "quat_convexity" => m.extended.convexity,
        "quat_isoperimetric" => m.extended.isoperimetric,
        "quat_bilateral_symmetry" => m.extended.bilateral_symmetry,
        "quat_centroid_offset" => m.extended.centroid_offset,
        "quat_largest_component_frac" => m.extended.largest_component_frac,
        "quat_shading_gradient" => m.extended.shading_gradient,
        "quat_shading_skewness" => m.extended.shading_skewness,
        "quat_specular_fraction" => m.extended.specular_fraction,
        "quat_crevice_fraction" => m.extended.crevice_fraction,
        "quat_color_gradient" => m.extended.color_gradient,
        "quat_color_shading_corr" => m.extended.color_shading_corr,
        "quat_color_band_autocorr" => m.extended.color_band_autocorr,
        "quat_color_range_utilization" => m.extended.color_range_utilization,
        "quat_cross_view_iou" => finite_or_zero(m.cross_view_iou),
        "quat_cross_view_coverage_delta" => finite_or_zero(m.cross_view_coverage_delta),
        "quat_c_sensitivity" => finite_or_zero(m.c_sensitivity),
        "quat_c_coverage_range" => finite_or_zero(m.c_coverage_range),
        "quat_node_count" => finite_or_zero(m.node_count),
        "quat_opcode_diversity" => finite_or_zero(m.opcode_diversity),
        "quat_max_depth" => finite_or_zero(m.max_depth),
        "quat_warp_node_count" => finite_or_zero(m.warp_node_count),
        "quat_warp_opcode_diversity" => finite_or_zero(m.warp_opcode_diversity),
        _ => 0.0,
    }
}

#[cfg(test)]
mod quat_pref_feature_tests {
    use super::*;

    /// Independently re-typed expected mapping (not derived from
    /// `quat_full_metrics_feature`'s own source) — a copy-paste of the
    /// implementation would pass trivially and catch nothing; this
    /// catches a mixed-up name/value pairing, which would otherwise
    /// silently misalign a trained model's weights against the wrong
    /// metrics.
    #[test]
    fn quat_full_metrics_feature_matches_every_model_field_name() {
        let m = QuatFullMetrics {
            breakdown: nnfractals::quat_dag_fitness::QuatFitnessBreakdown {
                anisotropy: 1.0, coverage: 2.0, solidity: 3.0,
                shading_richness: 4.0, color_entropy: 5.0, silhouette_irregularity: 6.0,
            },
            extended: nnfractals::quat_dag_fitness::QuatExtendedMetrics {
                box_dim: 7.0, lacunarity: 8.0, convexity: 9.0, isoperimetric: 10.0,
                bilateral_symmetry: 11.0, centroid_offset: 12.0, largest_component_frac: 13.0,
                shading_gradient: 14.0, shading_skewness: 15.0, specular_fraction: 16.0, crevice_fraction: 17.0,
                color_gradient: 18.0, color_shading_corr: 19.0, color_band_autocorr: 20.0, color_range_utilization: 21.0,
            },
            cross_view_iou: 22.0, cross_view_coverage_delta: 23.0, c_sensitivity: 24.0, c_coverage_range: 25.0,
            node_count: 26.0, opcode_diversity: 27.0, max_depth: 28.0, warp_node_count: 29.0, warp_opcode_diversity: 30.0,
        };
        let expected: &[(&str, f32)] = &[
            ("quat_anisotropy", 1.0), ("quat_coverage", 2.0), ("quat_solidity", 3.0), ("quat_shading_richness", 4.0),
            ("quat_color_entropy", 5.0), ("quat_silhouette_irregularity", 6.0), ("quat_box_dim", 7.0), ("quat_lacunarity", 8.0),
            ("quat_convexity", 9.0), ("quat_isoperimetric", 10.0), ("quat_bilateral_symmetry", 11.0), ("quat_centroid_offset", 12.0),
            ("quat_largest_component_frac", 13.0), ("quat_shading_gradient", 14.0), ("quat_shading_skewness", 15.0),
            ("quat_specular_fraction", 16.0), ("quat_crevice_fraction", 17.0), ("quat_color_gradient", 18.0),
            ("quat_color_shading_corr", 19.0), ("quat_color_band_autocorr", 20.0), ("quat_color_range_utilization", 21.0),
            ("quat_cross_view_iou", 22.0), ("quat_cross_view_coverage_delta", 23.0), ("quat_c_sensitivity", 24.0),
            ("quat_c_coverage_range", 25.0), ("quat_node_count", 26.0), ("quat_opcode_diversity", 27.0), ("quat_max_depth", 28.0),
            ("quat_warp_node_count", 29.0), ("quat_warp_opcode_diversity", 30.0),
        ];
        assert_eq!(expected.len(), 30, "test itself should cover all 30 trained-model fields");
        for (name, val) in expected {
            assert_eq!(quat_full_metrics_feature(&m, name), *val, "field {name} read the wrong value — mismatched vs apply_quat_full_metrics");
        }
        assert_eq!(quat_full_metrics_feature(&m, "not_a_real_field"), 0.0, "unknown field name should degrade to 0.0, not panic");
    }
}

/// One evolving individual in `quat-dag-evolve`'s population. Only
/// `program` is mutated/crossed-over across generations — `warp`, the
/// Julia/phoenix dynamics, and `bailout_radius` are inherited unchanged
/// from a parent, keeping the first real run of this pipeline focused on
/// exactly the thing the fitness function measures (main-iteration
/// structure), not a second simultaneous search over warp space too.
#[derive(Clone)]
struct QuatIndividual {
    program: Vec<nnfractals::formula::OpNode>,
    warp: Vec<nnfractals::formula::OpNode>,
    julia_mode: bool,
    jc: (f32, f32),
    phoenix: (f32, f32),
    bailout_radius: f32,
    geometric: f64,
    aesthetic: f64,
    /// Novelty in opcode-histogram space (`Genome::formula_descriptor`) —
    /// mean distance to this individual's K nearest neighbors in the
    /// CURRENT population, recomputed every generation (see
    /// `finalize_fitness`). Added after a first real run converged hard:
    /// 40 genomes over 10 generations collapsed to just 18 unique program
    /// shapes, essentially two "champion" lineages with cosmetic mutation
    /// noise around them — geometric+aesthetic fitness alone had nothing
    /// in it to reward being structurally different, so elitism plus mild
    /// mutation did exactly what it's built to do: converge fast. This is
    /// the same opcode-histogram k-NN descriptor the 2D GA already uses
    /// for its own `formula_diversity` selection pressure, not a new idea.
    diversity: f64,
    total: f64,
    /// The raw per-component geometric breakdown (solidity, coverage,
    /// anisotropy, silhouette_irregularity, shading_richness,
    /// color_entropy — see `quat_dag_fitness::QuatFitnessBreakdown`),
    /// kept around as a PHENOTYPE descriptor for novelty, not just the
    /// aggregate `.geometric` score. Added in cycle 2 after cycle 1's own
    /// contact sheet showed the opcode-histogram novelty
    /// (`Genome::formula_descriptor`) wasn't enough: the top-ranked third
    /// of the population — genuinely different DAG programs, hence
    /// "diverse" by that genotype metric — rendered as near-identical
    /// striped-barrel textures. Different code, same look. Comparing
    /// genomes by how they SCORE on these six visual axes instead of by
    /// opcode counts is a much more direct proxy for "does this actually
    /// look different."
    phenotype: [f32; 6],
    /// Raw per-term values of the active `--fitness-metric` spec, in the
    /// SAME order as the spec itself — e.g. for
    /// `"solidity:0.8,coverage:0.5"` this is `[solidity_value,
    /// coverage_value]`. Only populated when `score_individual_for_metric`
    /// runs (empty otherwise — the default blended-fitness path doesn't
    /// use it). This is what `quat_predator::Predator`s hunt in: a
    /// predator's `weights` are a direction in this exact vector space,
    /// so a predator can only ever specialize on axes the GA is already
    /// selecting on, never some unrelated signal.
    metric_vec: Vec<f64>,
    /// `--map-elites` mode only: this individual's SigLIP taste score
    /// (`quat_taste.rs`), [0,1] — either the quality signal itself
    /// (`--fitness-metric` not given) or just carried along for the saved
    /// genome's `quat_taste` field when a scalar spec supplies quality
    /// instead. 0.0 (not "unscored") outside `--map-elites` mode.
    taste: f64,
    /// `--map-elites` mode only: (sphericity, hit_rate, organization_ordinal)
    /// — the archive's 3 behaviour-descriptor axes, see `quat_map_elites`.
    descriptors: [f64; 3],
    /// `--map-elites` mode only: fraction of `sphericity_full_diagnostic`'s
    /// 300 sampled directions that hit the surface — same value as
    /// `descriptors[1]`, kept as its own named field since it also gates
    /// archive insertion (see `map_elites_evaluate`'s hard constraint).
    hit_rate: f64,
}

impl QuatIndividual {
    fn from_genome(g: &Genome) -> Self {
        QuatIndividual {
            program: g.program.clone(),
            warp: g.warp.clone(),
            julia_mode: g.julia_mode,
            jc: (g.julia_cre, g.julia_cim),
            phoenix: (g.phoenix_re, g.phoenix_im),
            bailout_radius: g.bailout_radius,
            geometric: 0.0,
            aesthetic: 0.0,
            diversity: 0.0,
            total: 0.0,
            phenotype: [0.0; 6],
            metric_vec: Vec::new(),
            taste: 0.0,
            descriptors: [0.0; 3],
            hit_rate: 0.0,
        }
    }

    /// A stable content-addressable id — FNV-1a over the program/warp DAGs
    /// (serialized deterministically via serde_json) plus julia/phoenix/
    /// bailout — used as the `--map-elites` archive's saved-filename stem
    /// and the taste sidecar's embedding-cache key. Unlike
    /// `save_quat_population`'s per-checkpoint `rng2.random()` id (which
    /// re-randomizes every save, so nothing can be tracked across
    /// checkpoints), the SAME genome content always hashes to the SAME id
    /// — required for an elite that survives many generations unchanged
    /// to be recognized as "already embedded" rather than re-scored from
    /// scratch every checkpoint.
    fn content_hash(&self) -> u64 {
        let mut s = String::new();
        s.push_str(&serde_json::to_string(&self.program).unwrap_or_default());
        s.push('|');
        s.push_str(&serde_json::to_string(&self.warp).unwrap_or_default());
        s.push('|');
        s.push_str(&format!("{}|{:?}|{:?}|{:?}", self.julia_mode, self.jc, self.phoenix, self.bailout_radius));
        let mut h: u64 = 0xcbf29ce484222325;
        for b in s.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }

    /// A throwaway `Genome` wrapping this individual's fields — lets us
    /// reuse `score_genome_dag` and the rendering/save pipeline exactly as
    /// they already exist, instead of duplicating them for owned
    /// prog/warp `Vec`s.
    fn as_genome(&self) -> Genome {
        Genome {
            program: self.program.clone(),
            warp: self.warp.clone(),
            julia_mode: self.julia_mode,
            julia_cre: self.jc.0,
            julia_cim: self.jc.1,
            phoenix_re: self.phoenix.0,
            phoenix_im: self.phoenix.1,
            bailout_radius: self.bailout_radius,
            ..Default::default()
        }
    }
}

/// Scores one individual's geometric + aesthetic components only.
/// Geometric is the same cheap pre-filter as before (`score_genome_dag`'s
/// breakdown — just now sourced from `score_genome_dag_full`, which
/// reuses the identical 4 probe renders and adds no extra GPU work, see
/// that function's doc comment). Aesthetic no longer renders a separate
/// still or calls out to Python at all: when `pref_model` is `Some`
/// (Carl's own trained weights, `quat_pref::QuatPrefModel` — see that
/// module's doc comment for why: "discontinue the aesthetic scorer
/// unless it is one I trained myself"), it's a dot product over the same
/// `quat_*` metrics already computed for this individual. When
/// `pref_model` is `None` (no self-trained model yet), aesthetic mirrors
/// geometric exactly rather than defaulting to 0.0 — the selection math
/// downstream (`finalize_fitness`'s `quality = 0.5*geometric +
/// 0.5*aesthetic` and its tuned `QUALITY_FLOOR`) assumes a real 0-1
/// aesthetic signal on the same scale as geometric; silently zeroing it
/// would halve `quality` for everyone and desync the floor from every
/// constant tuned against it, not just "drop the aesthetic axis." Either
/// way, aesthetic still only applies once `geometric_gate` is cleared,
/// matching the old gate's purpose (don't bother scoring obvious junk).
/// Does NOT set `.total` — `.diversity` (and hence `.total`) can only be
/// computed relative to the rest of the current population, so that's
/// `finalize_fitness`'s job, called once after a whole batch of
/// individuals has been scored.
fn score_individual(
    ind: &mut QuatIndividual,
    pref_model: Option<&nnfractals::quat_pref::QuatPrefModel>,
    probe_size: u32,
    use_gpu: bool,
    geometric_gate: f64,
) {
    let genome = ind.as_genome();
    let full = score_genome_dag_full(&genome, probe_size, use_gpu);
    let bd = &full.breakdown;
    ind.geometric = bd.total() as f64;
    ind.phenotype = [bd.anisotropy, bd.coverage, bd.solidity, bd.shading_richness, bd.color_entropy, bd.silhouette_irregularity];

    ind.aesthetic = if ind.geometric < geometric_gate {
        0.0
    } else {
        match pref_model {
            Some(model) => model.score(|name| quat_full_metrics_feature(&full, name)) as f64,
            None => ind.geometric,
        }
    };
}

/// Mean distance (over the population, k-NN, K=min(5,n-1)) from each
/// individual's descriptor to its K nearest neighbors, normalized to
/// [0,1] by `max_dist` (the greatest possible distance for that
/// descriptor space, given its own per-axis value range).
fn knn_novelty(descriptors: &[Vec<f64>], max_dist: f64) -> Vec<f64> {
    let n = descriptors.len();
    let k = 5.min(n.saturating_sub(1)).max(1);
    (0..n)
        .map(|i| {
            let mut dists: Vec<f64> = (0..n)
                .filter(|&j| j != i)
                .map(|j| descriptors[i].iter().zip(&descriptors[j]).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt())
                .collect();
            dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            if dists.is_empty() { 0.0 } else { (dists.iter().take(k).sum::<f64>() / k as f64 / max_dist).clamp(0.0, 1.0) }
        })
        .collect()
}

/// Computes each individual's novelty and folds geometric+aesthetic+
/// diversity into `.total` — as a QUALITY FLOOR plus an ADDITIVE novelty
/// bonus, not a 3-way blend. The first diversity-aware run (a straight
/// `0.35*geo + 0.35*aes + 0.30*div` blend) DID fix the convergence
/// problem — unique shapes held at 83-95% all 25 generations — but the
/// population's own contact sheet showed why a blend is the wrong shape
/// for this: with novelty carrying 30% of the score unconditionally, a
/// broken/noisy genome (which is JUST as structurally "novel" as a good
/// one) could out-rank a genuinely decent one on total fitness alone.
/// Gating the bonus behind `quality_floor` means novelty can only ever
/// help a genome that already clears a baseline of looking like
/// *something*, never rescue one that doesn't — `total` is always
/// `>= quality`, so quality remains the primary axis and diversity is
/// strictly a tiebreaker/bonus among genomes that already passed muster.
///
/// Cycle 2 changed novelty from `Genome::formula_descriptor()` (opcode
/// usage histogram — genotype) to `ind.phenotype` (`QuatFitnessBreakdown`
/// components — visual-axis summary stats). That was diagnosed from
/// cycle 1's OWN contact sheet, which showed top survivors converging
/// hard to one striped-barrel look despite healthy opcode-histogram
/// diversity (165-192/200 unique shapes all 40 generations) — different
/// DAG programs rendering to near-identical output. Reasonable fix in
/// principle. In practice cycle 2's contact sheet came back WORSE: over
/// HALF the population (top ~5.5 of 14 rows, vs cycle 1's top ~2) was
/// the exact same look, and unique-shape count (still tracked, no longer
/// selected on) fell much further (200→105-130 vs cycle 1's 165-192
/// floor). Root cause, visible in the data: `ind.phenotype`'s six
/// components are the SAME axes `.geometric` is a weighted sum of, and
/// `quality_floor` already gates the whole population on `quality =
/// 0.5*geo+0.5*aes` — so among genomes that clear the floor, phenotype
/// values are already clustered by construction (that's what "clearing
/// the floor" means), leaving phenotype-space novelty almost no room to
/// discriminate. It's not a bad measurement, it's just entangled with
/// quality, which the opcode histogram never was — that's exactly why
/// genotype novelty (orthogonal to quality) had more resolving power
/// among the "good" subpopulation, even though it's blind to appearance.
///
/// Cycle 3 fix: use BOTH. `ind.diversity` is now the mean of the
/// genotype k-NN novelty and the phenotype k-NN novelty, each computed
/// and normalized independently (same method as before). A genome only
/// gets full novelty credit for being different on BOTH axes — different
/// DAG structure AND a different visual-summary-stat profile — which is
/// a stronger, more specific bar than either alone, and directly
/// addresses both failure modes observed so far: genotype-only missed
/// visual convergence (cycle 1), phenotype-only collapsed because
/// quality-gating already collapses its resolving power (cycle 2).
/// Every `--fitness-metric` name `cmd_quat_dag_evolve` accepts, and
/// what each one costs to compute. The 5 structural names need NO
/// rendering at all (pure DAG-program math, same cost class as
/// `anisotropy_score`) — selecting on one of these is dramatically
/// cheaper than the normal geometric+aesthetic blend, since it also
/// skips the aesthetic-scorer Python round-trip entirely. Everything
/// else needs `score_genome_dag_full`'s extended sweep, which is
/// normally kept off this exact hot path (see that function's own doc
/// comment) — `--fitness-metric` is the deliberate exception: Carl's
/// own ask is to test whether the GA can actually climb one of these
/// metrics, which requires scoring every child on it, not just the
/// genomes that end up saved.
const STRUCTURAL_METRIC_NAMES: [&str; 5] = ["node_count", "opcode_diversity", "max_depth", "warp_node_count", "warp_opcode_diversity"];
/// `quat_organization.rs`'s 5 whole-4D-object metrics — like the
/// structural names, need no rendered probe (no `QuatFullMetrics`), but
/// UNLIKE structural metrics they aren't free: each needs its own
/// Halton-sample pass over the genome's program (see
/// `compute_organization_metrics`), computed once per individual and
/// shared across every organization term in a combo spec, exactly the
/// same one-computation-many-lookups pattern `full` already uses for the
/// render-based metrics.
const ORGANIZATION_METRIC_NAMES: [&str; 7] = [
    "organization_statistical",
    "organization_ordinal",
    "organization_multifractal",
    "organization_compression",
    "organization_compression_capped",
    "organization_chaoticity",
    "sphericity",
];

/// Compression ratio past which `organization_compression_capped` starts
/// heavily penalizing — Carl's own read after watching all 5 raw
/// organization metrics run: "the top metrics always get chaotic...
/// however, the middle individuals tend to be better," so pick a
/// specific target ceiling and punish crossing it hard, rather than
/// maximizing the raw (monotonic, noise-seeking) ratio the way
/// `organization_compression` alone does. This is the simple, practical
/// version of the "peaks between order and chaos" shape the research
/// memo argued for — a hard cap instead of the full entropy×disequilibrium
/// construction `organization_statistical`/`organization_ordinal` use.
const COMPRESSION_CAP: f64 = 0.4;
/// How much fitness each unit of overshoot past `COMPRESSION_CAP` costs —
/// large enough that no amount of extra structure below the cap can ever
/// be worth crossing it (max achievable value AT the cap is
/// `COMPRESSION_CAP` itself, so a penalty slope well above 1.0 guarantees
/// that).
const COMPRESSION_OVERSHOOT_PENALTY: f64 = 5.0;

/// Parses `--fitness-metric`'s value into a weighted list. Accepts a
/// bare name (`convexity`, weight defaults to 1.0), or a comma-separated
/// list of `name` / `name:weight` for a COMBINED objective (e.g.
/// `"silhouette_irregularity:1.0,shading_gradient:0.7,color_shading_corr:0.5"`)
/// — the actual fitness value is the weighted SUM of each named metric's
/// raw value (every metric is already [0,1]-scaled, so the sum stays a
/// sane, comparable range for a small handful of terms; it doesn't need
/// to be renormalized back to [0,1] itself, since only ranking/sorting
/// ever reads it).
fn parse_fitness_metric_spec(s: &str) -> Vec<(String, f64)> {
    s.split(',')
        .map(|term| {
            let term = term.trim();
            match term.split_once(':') {
                Some((name, w)) => (name.trim().to_string(), w.trim().parse::<f64>().unwrap_or_else(|_| panic!("bad weight in --fitness-metric term {term:?} — expected name:weight"))),
                None => (term.to_string(), 1.0),
            }
        })
        .collect()
}

/// Looks up ONE named metric's raw value from an already-computed
/// `QuatFullMetrics` and/or `OrganizationMetrics` (or, for the 5
/// structural names — free, no render, no sampling — computes it
/// directly without needing either).
fn lookup_metric(genome: &Genome, name: &str, full: Option<&QuatFullMetrics>, organization: Option<&nnfractals::quat_organization::OrganizationMetrics>) -> f64 {
    if STRUCTURAL_METRIC_NAMES.contains(&name) {
        let (node_count, opcode_diversity, max_depth) = nnfractals::quat_dag_fitness::structural_metrics(&genome.program);
        let (warp_node_count, warp_opcode_diversity, _) = nnfractals::quat_dag_fitness::structural_metrics(&genome.warp);
        return (match name {
            "node_count" => node_count,
            "opcode_diversity" => opcode_diversity,
            "max_depth" => max_depth,
            "warp_node_count" => warp_node_count,
            "warp_opcode_diversity" => warp_opcode_diversity,
            _ => unreachable!(),
        }) as f64;
    }
    if ORGANIZATION_METRIC_NAMES.contains(&name) {
        let organization = organization.unwrap_or_else(|| panic!("metric {name:?} needs organization metrics, which weren't computed — internal bug in score_individual_for_metric"));
        if name == "organization_compression_capped" {
            let c = organization.compression_complexity as f64;
            return if c <= COMPRESSION_CAP {
                c
            } else {
                COMPRESSION_CAP - (c - COMPRESSION_CAP) * COMPRESSION_OVERSHOOT_PENALTY
            };
        }
        return (match name {
            "organization_statistical" => organization.statistical_complexity,
            "organization_ordinal" => organization.ordinal_complexity,
            "organization_multifractal" => organization.multifractal_width,
            "organization_compression" => organization.compression_complexity,
            "organization_chaoticity" => organization.chaoticity,
            "sphericity" => organization.sphericity,
            _ => unreachable!(),
        }) as f64;
    }
    let full = full.unwrap_or_else(|| panic!("metric {name:?} needs the full metric sweep, which wasn't computed — internal bug in combo_metric_value"));
    (match name {
        "anisotropy" => full.breakdown.anisotropy,
        "coverage" => full.breakdown.coverage,
        "solidity" => full.breakdown.solidity,
        "shading_richness" => full.breakdown.shading_richness,
        "color_entropy" => full.breakdown.color_entropy,
        "silhouette_irregularity" => full.breakdown.silhouette_irregularity,
        "box_dim" => full.extended.box_dim,
        "lacunarity" => full.extended.lacunarity,
        "convexity" => full.extended.convexity,
        "isoperimetric" => full.extended.isoperimetric,
        "bilateral_symmetry" => full.extended.bilateral_symmetry,
        "centroid_offset" => full.extended.centroid_offset,
        "largest_component_frac" => full.extended.largest_component_frac,
        "shading_gradient" => full.extended.shading_gradient,
        "shading_skewness" => full.extended.shading_skewness,
        "specular_fraction" => full.extended.specular_fraction,
        "crevice_fraction" => full.extended.crevice_fraction,
        "color_gradient" => full.extended.color_gradient,
        "color_shading_corr" => full.extended.color_shading_corr,
        "color_band_autocorr" => full.extended.color_band_autocorr,
        "color_range_utilization" => full.extended.color_range_utilization,
        "cross_view_iou" => full.cross_view_iou,
        "cross_view_coverage_delta" => full.cross_view_coverage_delta,
        "c_sensitivity" => full.c_sensitivity,
        "c_coverage_range" => full.c_coverage_range,
        other => panic!("unknown --fitness-metric term {other:?} — see STRUCTURAL_METRIC_NAMES/lookup_metric for the full list"),
    }) as f64
}

/// `score_individual`'s single/combo-metric counterpart: sets
/// `.geometric` to the WEIGHTED SUM of every named metric in `spec`,
/// `.aesthetic` to 0.0 (unused in this mode), and `.phenotype` to the
/// extended breakdown's 6-axis summary when it was computed anyway
/// (skipped only if EVERY term in `spec` is structural, in which case
/// there's nothing to compute it from — genotype novelty still works
/// fine, diversity just loses one of its two normal inputs). The
/// expensive full sweep runs AT MOST ONCE per individual regardless of
/// how many non-structural terms are in the combo — computed once,
/// looked up per term.
/// Lazily-spawned taste sidecar, shared across every `taste_score_for_genome`
/// call in a run — spawning the SigLIP backbone process per-genome would be
/// far too slow. Stays `None` for the whole run after a failed first spawn
/// (missing `quat_taste_scorer.py`/`taste_model_quat.npz`), logged once
/// rather than silently scoring 0.0 forever without saying why.
static TASTE_SCORER: std::sync::OnceLock<std::sync::Mutex<Option<nnfractals::quat_taste::QuatTasteScorer>>> = std::sync::OnceLock::new();

/// Renders the exact still every saved genome's browser thumbnail uses
/// (300x400, C=0, "lava" colormap — the corpus's `train_taste_quat.py`
/// trained against this exact framing) and scores it through the taste
/// sidecar. Returns 0.0 (not a fallback score — a real "no signal")
/// whenever the sidecar/model aren't available, so a `taste` term in a
/// `--fitness-metric` combo degrades to "contributes nothing" rather than
/// crashing the run.
///
/// The sidecar's embedding cache is keyed by an FNV-1a hash of the
/// rendered pixel bytes (not a genome id) — this run predates
/// `QuatIndividual::content_hash` (that stable id lands in Phase 2b for
/// the MAP-Elites archive), but "same rendered image -> same cache key"
/// is exactly the property the cache needs, and is trivially correct
/// without any genome-schema coupling.
fn taste_score_for_genome(genome: &Genome, use_gpu: bool) -> f64 {
    let mutex = TASTE_SCORER.get_or_init(|| {
        let scorer = nnfractals::quat_taste::QuatTasteScorer::new();
        if scorer.is_none() {
            eprintln!("  [taste] quat_taste_scorer.py/taste_model_quat.npz not found — the 'taste' metric term will score 0.0 for every genome this run.");
        }
        std::sync::Mutex::new(scorer)
    });
    let mut guard = mutex.lock().unwrap();
    let Some(scorer) = guard.as_mut() else { return 0.0 };

    let formula = nnfractals::quat_dag::QuatDagFormula { prog: &genome.program, warp: &genome.warp, julia: genome.julia_mode, jc: (genome.julia_cre, genome.julia_cim), phoenix: (genome.phoenix_re, genome.phoenix_im) };
    let domain_radius = 1.6;
    let fov_deg = 45.0;
    let bg_color = (0.03, 0.02, 0.06);
    let params = nnfractals::quat_dag::RaymarchDagParams {
        formula, time_axis: nnfractals::quat_fractal::TimeAxis::C, time_val: 0.0, domain_radius,
        max_iter: 50, bailout: genome.bailout_radius as f64, max_march_steps: 150,
        hit_epsilon: domain_radius * 1e-4, step_safety: 0.8, light_dir: (0.5, 0.8, 0.3),
        normal_eps: domain_radius * 1e-3, color_probe_offset: domain_radius * 1e-2, aa: 1,
    };
    let (w, h) = (300u32, 400u32);
    let radius = recommended_orbit_radius(domain_radius, fov_deg, w, h);
    let cam = nnfractals::quat_raymarch::RaymarchCamera { eye: (0.0, 0.0, -radius), target: (0.0, 0.0, 0.0), up_hint: (0.0, 1.0, 0.0), fov_y: fov_deg.to_radians() };
    let mut mode = resolve_dag_gpu_mode(params.formula.prog, params.formula.warp, use_gpu);
    let (shading, color_t) = render_dag_frame_with_mode(&mut mode, &params, &cam, w, h);
    let rgb = raymarch_frame_to_rgb(&shading, &color_t, params.max_iter, "lava", bg_color);

    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in &rgb {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    let key = format!("{hash:016x}");
    let scratch_dir = std::path::Path::new("explorer_out/taste_scratch");
    std::fs::create_dir_all(scratch_dir).ok();
    let path = scratch_dir.join(format!("{key}.png"));
    if io::save_png(&rgb, w, h, &path).is_err() { return 0.0; }
    let score = scorer.score_blocking(&key, std::slice::from_ref(&path)).unwrap_or(0.0);
    std::fs::remove_file(&path).ok();
    score as f64
}

/// `--map-elites` mode's per-individual evaluation: scores quality (taste
/// by default, or the active `--fitness-metric` spec if one is given —
/// same "quality = taste by default, or the spec's scalar if given" rule
/// the plan calls for), computes the 3 archive behaviour-descriptor axes,
/// and enforces the hard pre-insertion constraints (see the plan's Phase
/// 2b: reject, don't penalize). Returns `None` for anything that should
/// never occupy an archive cell — the caller must not insert on `None`.
fn map_elites_evaluate(
    ind: &mut QuatIndividual,
    fitness_metric: &Option<Vec<(String, f64)>>,
    probe_size: u32,
    use_gpu: bool,
) -> Option<([u8; 3], f64)> {
    let genome = ind.as_genome();
    let quality = match fitness_metric {
        Some(metric) => {
            score_individual_for_metric(ind, metric, probe_size, use_gpu);
            ind.geometric
        }
        None => {
            let t = taste_score_for_genome(&genome, use_gpu);
            ind.taste = t;
            t
        }
    };
    let formula = nnfractals::quat_dag::QuatDagFormula { prog: &ind.program, warp: &ind.warp, julia: ind.julia_mode, jc: ind.jc, phoenix: ind.phoenix };
    let bailout_sq = (ind.bailout_radius as f64) * (ind.bailout_radius as f64);
    let (hit_count, n_dir, _mean_r, _roundness, sphericity) = nnfractals::quat_organization::sphericity_full_diagnostic(&formula, 1.6, 60, bailout_sq);
    let hit_rate = if n_dir > 0 { hit_count as f64 / n_dir as f64 } else { 0.0 };
    let organization = nnfractals::quat_organization::compute_organization_metrics(&formula, 1.6, 60, bailout_sq);
    ind.hit_rate = hit_rate;
    ind.descriptors = [sphericity as f64, hit_rate, organization.ordinal_complexity as f64];

    // Hard constraints (reject, don't penalize — sphericity itself is an
    // archive AXIS, never a rejection reason). hit_rate < 0.05 doubles as
    // both "empty" and "nothing visible" — sphericity_full_diagnostic's
    // 300-direction scan already IS a visibility probe, so no separate
    // multi-view render is needed just to check this.
    if hit_rate < 0.05 || !quality.is_finite() || ind.descriptors.iter().any(|d| !d.is_finite()) {
        return None;
    }
    let cell = nnfractals::quat_map_elites::descriptor_cell(ind.descriptors[0], ind.descriptors[1], ind.descriptors[2]);
    Some((cell, quality))
}

fn score_individual_for_metric(ind: &mut QuatIndividual, spec: &[(String, f64)], probe_size: u32, use_gpu: bool) {
    let genome = ind.as_genome();
    let needs_full = spec.iter().any(|(name, _)| name != "taste" && !STRUCTURAL_METRIC_NAMES.contains(&name.as_str()) && !ORGANIZATION_METRIC_NAMES.contains(&name.as_str()));
    let needs_organization = spec.iter().any(|(name, _)| ORGANIZATION_METRIC_NAMES.contains(&name.as_str()));
    let full = if needs_full { Some(score_genome_dag_full(&genome, probe_size, use_gpu)) } else { None };
    let organization = if needs_organization {
        let formula = nnfractals::quat_dag::QuatDagFormula { prog: &genome.program, warp: &genome.warp, julia: genome.julia_mode, jc: (genome.julia_cre, genome.julia_cim), phoenix: (genome.phoenix_re, genome.phoenix_im) };
        let bailout_sq = (genome.bailout_radius as f64) * (genome.bailout_radius as f64);
        Some(nnfractals::quat_organization::compute_organization_metrics(&formula, 1.6, 60, bailout_sq))
    } else {
        None
    };
    if let Some(f) = &full {
        ind.phenotype = [f.breakdown.anisotropy, f.breakdown.coverage, f.breakdown.solidity, f.breakdown.shading_richness, f.breakdown.color_entropy, f.breakdown.silhouette_irregularity];
    } else {
        ind.phenotype = [0.0; 6];
    }
    let raw_vals: Vec<f64> = spec.iter().map(|(name, _)| {
        if name == "taste" { taste_score_for_genome(&genome, use_gpu) } else { lookup_metric(&genome, name, full.as_ref(), organization.as_ref()) }
    }).collect();
    ind.geometric = spec.iter().zip(&raw_vals).map(|((_, w), v)| w * v).sum();
    ind.aesthetic = 0.0;
    ind.metric_vec = raw_vals;
}

fn finalize_fitness(population: &mut [QuatIndividual], quality_floor: f64, diversity_bonus_weight: f64, single_metric_mode: bool) {
    let genotype_desc: Vec<Vec<f64>> = population.iter().map(|ind| ind.as_genome().formula_descriptor().iter().map(|v| *v as f64).collect()).collect();
    let phenotype_desc: Vec<Vec<f64>> = population.iter().map(|ind| ind.phenotype.iter().map(|v| *v as f64).collect()).collect();
    // formula_descriptor() is L2-normalized (unit vectors), so the max
    // possible distance between two is 2.0 (opposite directions). Each of
    // phenotype's 6 components is independently clamped to [0,1], so its
    // max possible per-axis gap is 1.0 and max Euclidean distance sqrt(6).
    let genotype_novelty = knn_novelty(&genotype_desc, 2.0);
    let phenotype_novelty = knn_novelty(&phenotype_desc, 6.0f64.sqrt());
    // Cycle 4: reverted the 50/50 blend back to pure genotype. Cycle 3's
    // equal-weight blend under-performed cycle 1's pure-genotype baseline
    // on the contact sheet (top ~4.5/14 rows converged to one look vs
    // cycle 1's ~2/14) despite splitting the numeric metrics down the
    // middle. Read: phenotype novelty's typical magnitude (0.04-0.09,
    // cycle 2) is much smaller than genotype's (0.17-0.47, cycle 1) among
    // quality-floor survivors, so averaging them 50/50 didn't add a
    // second independent signal — it mostly just diluted genotype's own,
    // already-working, larger-magnitude signal by about half, weakening
    // the anti-convergence pressure `diversity_bonus_weight` was
    // calibrated against. Going back to the best-known baseline
    // (genotype-only) before testing the next hypothesis, rather than
    // building further on a config that's now shown to underperform.
    // Weights are named constants (not inlined) so a future cycle can
    // retune this in one place instead of re-deriving it.
    const GENOTYPE_NOVELTY_WEIGHT: f64 = 1.0;
    const PHENOTYPE_NOVELTY_WEIGHT: f64 = 0.0;
    for ((ind, gnov), pnov) in population.iter_mut().zip(genotype_novelty).zip(phenotype_novelty) {
        ind.diversity = GENOTYPE_NOVELTY_WEIGHT * gnov + PHENOTYPE_NOVELTY_WEIGHT * pnov;
        if single_metric_mode {
            // Carl's ask for the single-metric experiment runs (Part 7):
            // a clean, uncontaminated read on whether the GA can climb
            // THIS ONE metric — no quality floor, no aesthetic blend, no
            // diversity bonus muddying the signal. `.diversity` is still
            // computed and logged above (real observability value: does
            // selecting hard on one metric collapse genotype diversity
            // as a side effect?), it just doesn't feed `.total`.
            ind.total = ind.geometric;
        } else {
            let quality = 0.5 * ind.geometric + 0.5 * ind.aesthetic;
            let bonus = if quality >= quality_floor { diversity_bonus_weight * ind.diversity } else { 0.0 };
            ind.total = quality + bonus;
        }
    }
}

/// Glue between `QuatIndividual` and `quat_predator`'s genome-agnostic
/// vector math — extracts each individual's `metric_vec`, applies one
/// generation of predator pressure (discounting `.total` in place) and
/// evolves the predator population, then writes the discounted totals
/// back. Called AFTER `finalize_fitness` (so predators see the real
/// per-individual totals to discount) but BEFORE the population is
/// re-sorted by `.total`, so the discount actually affects selection this
/// generation, not just next generation's report. No-op if `predators` is
/// empty (i.e. `--predator-prey` wasn't passed) or the population is
/// empty. Returns the best predator's correlation-with-commonness this
/// generation for `report_generation`-style logging.
fn apply_predator_pressure(population: &mut [QuatIndividual], predators: &mut Vec<nnfractals::quat_predator::Predator>, rng: &mut impl rand::Rng) -> Option<f64> {
    if predators.is_empty() || population.is_empty() {
        return None;
    }
    let metric_vecs: Vec<Vec<f64>> = population.iter().map(|ind| ind.metric_vec.clone()).collect();
    let mut totals: Vec<f64> = population.iter().map(|ind| ind.total).collect();
    let best_predator_fitness = nnfractals::quat_predator::apply_predator_pressure(&metric_vecs, &mut totals, predators, nnfractals::quat_predator::DEFAULT_PENALTY_WEIGHT, rng);
    for (ind, t) in population.iter_mut().zip(totals) {
        ind.total = t;
    }
    Some(best_predator_fitness)
}

/// Number of distinct `(opcode-sequence, program-length)` shapes in the
/// population — a blunt but immediately legible diversity readout,
/// printed alongside the finer-grained mean novelty each generation so a
/// convergence problem like the one that prompted `diversity` shows up
/// directly in the log, not just as a vague impression from thumbnails.
fn count_unique_shapes(population: &[QuatIndividual]) -> usize {
    let mut shapes: std::collections::HashSet<Vec<u8>> = std::collections::HashSet::new();
    for ind in population {
        shapes.insert(ind.program.iter().map(|n| n.op).collect());
    }
    shapes.len()
}

/// Runs a real evolution phase over the quaternion DAG genome system,
/// selecting on a 3-way blend of the cheap geometric pre-filter
/// (`quat_dag_fitness`), the learned aesthetic ensemble, and structural
/// novelty (`finalize_fitness`) — the last added after a first run with
/// only the first two collapsed 40 genomes to 18 unique program shapes
/// dominated by two lineages ("very much alike", Carl's own words after
/// looking at the results). Elitist: `survivors` carry over each
/// generation (their geometric/aesthetic stay cached, never re-rendered;
/// novelty is recomputed every generation since it depends on the rest of
/// the population). The remaining slots are filled by, per offspring:
/// crossover (30%), fresh immigrants loaded straight from `--pool-dir`
/// with no mutation at all (15% — the other lever against convergence:
/// keeps injecting genuinely outside genetic material every generation,
/// not just recombining what's already survived), or mutation (55%, now
/// TWO `mutate_program` passes composed per offspring instead of one —
/// the single-pass version was producing only cosmetic variants of the
/// same 2 champions). Seeded from a random sample of `--pool-dir` (the
/// existing, already-evolved 2D archive) rather than from scratch, since
/// re-interpreting already-interesting 2D structure under the new 3D
/// fitness is a more informative experiment than `random_program`'s blank
/// slate. Saves the final population (genomes + static/animated
/// thumbnails) to `--out-dir`, deliberately SEPARATE from `--pool-dir` —
/// never writes into the existing archive — and writes one contact sheet
/// (`_contact_sheet.png`, the static stills tiled into a grid) so the
/// whole population can be eyeballed for diversity at a glance instead of
/// only through the browser's small thumbnails.
/// Which crossover operator `cmd_quat_dag_evolve` uses. `Legacy` is
/// `genome.rs::crossover_program` (whole-program grafting, shared with
/// the 2D GA, unchanged). `Subtree` is the new
/// `quat_genome_ops::crossover_program_subtree` (real subtree exchange,
/// quaternion-evolution-only) — see that module's docs for why it's a
/// separate function rather than a fix to the shared one. Defaults to
/// `Legacy` until the single-metric A/B runs (explorer.rs's
/// `--fitness-metric`) actually confirm `Subtree` does better, not
/// before — a brand-new, not-yet-validated operator shouldn't silently
/// become the default.
#[derive(Clone, Copy, PartialEq)]
enum CrossoverMode {
    Legacy,
    Subtree,
}

#[allow(clippy::too_many_arguments)]
fn cmd_quat_dag_evolve(
    pool_dir: &str,
    out_dir: &str,
    population: usize,
    generations: usize,
    survivors: usize,
    probe_size: u32,
    use_gpu: bool,
    seed: u64,
    crossover_mode: CrossoverMode,
    mutation_strength: nnfractals::quat_genome_ops::MutationStrength,
    fitness_metric: Option<Vec<(String, f64)>>,
    predator_prey: bool,
    stagnation_gens: usize,
    pref_model_path: &str,
) {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    // Predator-prey coevolution (Carl's idea, see quat_predator's module
    // doc for the full motivation — the plateau this is meant to break)
    // needs a metric-vector space to hunt in, so it only makes sense
    // alongside --fitness-metric; there's no equivalent vector for the
    // default blended geometric+aesthetic+diversity path.
    if predator_prey && fitness_metric.is_none() {
        panic!("--predator-prey requires --fitness-metric — predators hunt in the metric-vector space that spec defines");
    }
    const PREDATOR_COUNT: usize = 24;
    let mut predator_rng = rand::rngs::StdRng::seed_from_u64(seed ^ 0xF00D_CAFE_u64);
    let mut predators: Vec<nnfractals::quat_predator::Predator> = match &fitness_metric {
        Some(spec) if predator_prey => nnfractals::quat_predator::spawn_predators(PREDATOR_COUNT, spec.len(), &mut predator_rng),
        _ => Vec::new(),
    };
    // Quality (0.5*geometric + 0.5*aesthetic) must clear this before
    // novelty counts for anything at all — see finalize_fitness's doc
    // comment for why a straight 3-way blend let broken/noisy genomes
    // outrank decent ones on "novelty" alone in the first tuning pass.
    const QUALITY_FLOOR: f64 = 0.45;
    const DIVERSITY_BONUS_WEIGHT: f64 = 0.25;
    // Cycle 4 change: 0.15 -> 0.35. Three cycles of contact-sheet evidence
    // now agree the novelty DESCRIPTOR (genotype vs phenotype vs blend)
    // isn't the main lever — every variant still let one lucky lineage
    // dominate a large chunk of the top-ranked survivors. Root cause,
    // found by actually reading the breeding loop below: crossover and
    // mutation parents are drawn ONLY from `next_gen[0..survivor_count]`
    // — this generation's elite carryovers — never from the wider
    // population. So if those `survivors` slots get captured by
    // near-clones of one genome, ~(1-IMMIGRANT_FRAC) of every subsequent
    // generation is bred from that same narrow gene pool regardless of
    // how well novelty scores the resulting children — there's little
    // real diversity left to select FROM. Immigrants (fresh, unrelated
    // draws from `pool_dir`, no mutation) are the ONLY channel that
    // doesn't depend on the current survivor pool's composition, so
    // raising their share is the most direct fix for "elitism is too
    // sticky," more than either strengthening novelty selection (already
    // tried three ways) or shrinking `survivors` (which would shrink the
    // parent pool further and likely make capture easier, not harder).
    // Cycle 5 change: 0.35 -> 0.50. Cycle 4 (0.35) clearly beat cycle 1's
    // 0.15 baseline on the contact sheet — its largest converged cluster
    // shrank to roughly 15-20/200 cells vs cycle 1's ~25-30 — while
    // quality held (best total 0.783-0.808, comparable to cycle 1's
    // 0.768-0.822). Pushing the same lever further to see whether more
    // immigrant material keeps helping or the improvement plateaus.
    // Cycle 6 change: 0.50 -> 0.60. Cycle 5 (0.50) was the best result
    // yet — the FIRST contact sheet with no visible repeated-clone
    // cluster anywhere, and best total (0.782-0.818) at least as good as
    // cycle 4. A quick diagnostic run also confirmed zero-geometric-score
    // children (the "worst total=0.000" seen every generation, every
    // cycle) are rare (~1-2 out of ~50 non-survivor children/gen) and
    // spread evenly across immigrant/crossover/mutation, NOT concentrated
    // in immigrants specifically — so pushing immigrant share further
    // isn't spending slots on unusually-likely duds. Testing whether the
    // improvement keeps scaling or has found its ceiling. CROSSOVER_FRAC
    // held at 0.30 so mutation still gets a real ~0.10 share.
    // Cycle 7: settled at 0.55, the midpoint of the range cycles 5 (0.50)
    // and 6 (0.60) both validated cleanly (no repeated-clone cluster on
    // either contact sheet, no quality cost). This run is a confirmation
    // sample at the settled config, not a further push — see the TL;DR
    // at the top of fractals_dag_quat/OVERNIGHT_LOG.md.
    const IMMIGRANT_FRAC: f64 = 0.55;
    const CROSSOVER_FRAC: f64 = 0.30;

    // Aesthetic component: Carl's own trained preference model
    // (quat_pref::QuatPrefModel — a linear fit over the ~30 quat_*
    // metrics from his browser ⚖ Rate comparisons) if one exists at
    // `pref_model_path`, else geometric-only. Deliberately NOT a
    // fallback to the generic NIMA/TOPIQ/AP25 ensemble — Carl: "discontinue
    // the aesthetic scorer unless it is one I trained myself." Loaded once
    // up front (cheap: a few hundred bytes of JSON) rather than per
    // individual.
    let pref_model = nnfractals::quat_pref::QuatPrefModel::load(std::path::Path::new(pref_model_path));

    // Carl: "I also want to know what is the fitness method used at the
    // moment" — printed once, up front, since it's fixed for the whole
    // run (unlike the per-generation state below). Kept separate from
    // the top-pool readout so it survives scrolling: it's the one line
    // you need if you only ever look at the top of the terminal.
    eprintln!(
        "fitness method: {}",
        match &fitness_metric {
            Some(spec) => format!(
                "single/combo metric — {}",
                spec.iter().map(|(n, w)| format!("{n}:{w:.2}")).collect::<Vec<_>>().join(", ")
            ),
            None => match &pref_model {
                Some(m) => format!(
                    "default blend — 50% geometric (quat_dag_fitness) + 50% aesthetic (Carl's own trained model, {} at {pref_model_path}, {} features), +diversity bonus (weight={DIVERSITY_BONUS_WEIGHT:.2}) once quality>={QUALITY_FLOOR:.2}",
                    "quat_pref::QuatPrefModel", m.feature_count()
                ),
                None => format!(
                    "geometric-only (quat_dag_fitness) — no self-trained model at {pref_model_path}; generic aesthetic scorer deliberately NOT used, +diversity bonus (weight={DIVERSITY_BONUS_WEIGHT:.2}) once quality>={QUALITY_FLOOR:.2}"
                ),
            },
        }
    );
    eprintln!(
        "breeding mix: crossover={:.0}% ({}) | immigrant={:.0}% | mutation={:.0}% | survivors={survivors} | predator-prey={} | stagnation-gens={stagnation_gens}",
        CROSSOVER_FRAC * 100.0,
        match crossover_mode { CrossoverMode::Legacy => "legacy", CrossoverMode::Subtree => "subtree" },
        IMMIGRANT_FRAC * 100.0,
        (1.0 - IMMIGRANT_FRAC - CROSSOVER_FRAC) * 100.0,
        if predator_prey { "on" } else { "off" },
    );

    let mut pool_files: Vec<PathBuf> = std::fs::read_dir(pool_dir)
        .unwrap_or_else(|e| panic!("failed to read --pool-dir {pool_dir:?}: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("nn"))
        .collect();
    pool_files.shuffle(&mut rng);

    eprintln!("quat-dag-evolve: seeding population={population} from {pool_dir} ({} files available)", pool_files.len());
    let mut population_vec: Vec<QuatIndividual> = Vec::with_capacity(population);
    for path in &pool_files {
        if population_vec.len() >= population {
            break;
        }
        if let Ok(g) = io::load_genome(path) {
            if !g.program.is_empty() {
                population_vec.push(QuatIndividual::from_genome(&g));
            }
        }
    }
    eprintln!("  seeded {} individuals", population_vec.len());

    // Created up front (not just at the first checkpoint/final save) so
    // log_gen_stats can append gen-0's stats immediately — it used to
    // silently fail every write until the first checkpoint created this
    // directory, an empty evolve_stats.jsonl the whole time the dashboard
    // most needed early data.
    std::fs::create_dir_all(out_dir).ok();

    // Gen 0: score everyone (nothing inherited yet).
    let start = std::time::Instant::now();
    for ind in population_vec.iter_mut() {
        match &fitness_metric {
            Some(metric) => score_individual_for_metric(ind, metric, probe_size, use_gpu),
            None => score_individual(ind, pref_model.as_ref(), probe_size, use_gpu, 0.3),
        }
    }
    finalize_fitness(&mut population_vec, QUALITY_FLOOR, DIVERSITY_BONUS_WEIGHT, fitness_metric.is_some());
    if let Some(best_pred) = apply_predator_pressure(&mut population_vec, &mut predators, &mut predator_rng) {
        eprintln!("  [predator] best fitness (corr. w/ commonness) = {best_pred:.3}");
    }
    population_vec.sort_by(|a, b| b.total.partial_cmp(&a.total).unwrap_or(std::cmp::Ordering::Equal));
    report_generation(0, &population_vec, start.elapsed().as_secs_f64());
    print_top_pool(&population_vec, 5);
    let gen0_best_pred = predators.iter().map(|p| p.fitness).fold(f64::NEG_INFINITY, f64::max);
    log_gen_stats(out_dir, 0, &population_vec, start.elapsed().as_secs_f64(), (!predators.is_empty()).then_some(gen0_best_pred.max(0.0)), false);

    // Stagnation-reset tracking (Carl's ask, after the combo run that
    // plateaued at best total=3.200 from generation 139 onward — 126+
    // generations, zero improvement, 31% of the final population one
    // captured lineage). Mirrors the 2D GA's `restart_population`
    // (`optimizer.rs`) exactly in spirit: a deterministic reset once the
    // record hasn't moved in `stagnation_gens` generations, distinct from
    // `mass_extinction`'s rare random wipe (which this GA has no
    // equivalent of — not needed, since predator-prey pressure already
    // provides continuous anti-capture force; stagnation reset is the
    // backstop for when it isn't enough). Same epsilon (0.005) as the 2D
    // GA's stagnation check — matching or nearly matching the record
    // resets the clock, only a genuine drop below it counts as declining.
    const STAGNATION_EPSILON: f64 = 0.005;
    let mut best_total_ever = population_vec[0].total;
    let mut stagnant_gens: usize = 0;

    // Periodic checkpoint saving — a run started with a large
    // --generations count and no fixed end time ("I will tell you when
    // to stop") previously lost EVERYTHING if killed mid-run, since
    // saving only ever happened once, at the very end. Every
    // CHECKPOINT_INTERVAL generations, save the current population to
    // --out-dir exactly like the final save does (full metric sweep,
    // thumbnails, contact sheet), deleting the previous checkpoint's
    // files first so out_dir always reflects the latest snapshot rather
    // than accumulating stale intermediate populations forever. Worst
    // case on a kill: lose up to CHECKPOINT_INTERVAL generations of
    // progress, never the whole run.
    const CHECKPOINT_INTERVAL: usize = 15;
    let mut rng2 = rand::rngs::StdRng::seed_from_u64(seed ^ 0xDEAD_BEEF_u64);
    let mut checkpoint_stems: Vec<String> = Vec::new();

    for gen_idx in 1..=generations {
        let gen_start = std::time::Instant::now();
        let survivor_count = survivors.min(population_vec.len());
        let mut next_gen: Vec<QuatIndividual> = population_vec.drain(..survivor_count).collect();

        // Diagnostic tally (cycles 1-5 all showed "worst total=0.000" in
        // EVERY generation, never investigated) — counts how many
        // children from each breeding channel come back with a totally
        // failed geometric score (bd.total()==0.0 exactly, meaning the
        // probe render found essentially nothing), to see whether the
        // zero-scorers are concentrated in one channel (e.g. raw
        // immigrants that just don't happen to be viable quaternion
        // formulas) rather than being generic noise.
        let mut channel_tally: std::collections::HashMap<&'static str, (u32, u32)> = std::collections::HashMap::new();
        while next_gen.len() < population {
            let roll: f64 = rng.random();
            let source: &'static str;
            let mut child = if roll < IMMIGRANT_FRAC && !pool_files.is_empty() {
                source = "immigrant";
                let path = &pool_files[rng.random_range(0..pool_files.len())];
                match io::load_genome(path) {
                    Ok(g) if !g.program.is_empty() => QuatIndividual::from_genome(&g),
                    _ => continue, // bad/legacy file — just retry the loop
                }
            } else if roll < IMMIGRANT_FRAC + CROSSOVER_FRAC && survivor_count >= 2 {
                source = "crossover";
                let ia = rng.random_range(0..survivor_count);
                let ib = rng.random_range(0..survivor_count);
                let parent_a = &next_gen[ia];
                let parent_b = &next_gen[ib];
                let program = match crossover_mode {
                    CrossoverMode::Legacy => nnfractals::genome::crossover_program(&parent_a.program, &parent_b.program, &mut rng, 24),
                    CrossoverMode::Subtree => nnfractals::quat_genome_ops::crossover_program_subtree(&parent_a.program, &parent_b.program, &mut rng, 24),
                };
                QuatIndividual {
                    program,
                    warp: parent_a.warp.clone(),
                    julia_mode: parent_a.julia_mode,
                    jc: parent_a.jc,
                    phoenix: parent_a.phoenix,
                    bailout_radius: parent_a.bailout_radius,
                    geometric: 0.0, aesthetic: 0.0, diversity: 0.0, total: 0.0, phenotype: [0.0; 6], metric_vec: Vec::new(),
                    taste: 0.0, descriptors: [0.0; 3], hit_rate: 0.0,
                }
            } else {
                source = "mutation";
                let ia = rng.random_range(0..survivor_count);
                let parent = &next_gen[ia];
                // Two composed mutation passes: one pass on a 15-24 node
                // program only edits a small fraction of it (1-2 nodes),
                // which is exactly why gen-0's two lucky winners barely
                // drifted over 10 generations of single-pass mutation.
                // Always goes through the new quat_genome_ops tunable
                // mutation (not gated by crossover_mode — its default
                // `MutationStrength` reproduces the old hardcoded
                // behavior exactly, so this is a strict superset, not a
                // silent change, unless `--mutation-*` flags override it).
                let p1 = nnfractals::quat_genome_ops::mutate_program_tuned(&parent.program, &mut rng, 24, mutation_strength.max_depth, &mutation_strength);
                let program = nnfractals::quat_genome_ops::mutate_program_tuned(&p1, &mut rng, 24, mutation_strength.max_depth, &mutation_strength);
                QuatIndividual {
                    program,
                    warp: parent.warp.clone(),
                    julia_mode: parent.julia_mode,
                    jc: parent.jc,
                    phoenix: parent.phoenix,
                    bailout_radius: parent.bailout_radius,
                    geometric: 0.0, aesthetic: 0.0, diversity: 0.0, total: 0.0, phenotype: [0.0; 6], metric_vec: Vec::new(),
                    taste: 0.0, descriptors: [0.0; 3], hit_rate: 0.0,
                }
            };
            match &fitness_metric {
                Some(metric) => score_individual_for_metric(&mut child, metric, probe_size, use_gpu),
                None => score_individual(&mut child, pref_model.as_ref(), probe_size, use_gpu, 0.3),
            }
            let entry = channel_tally.entry(source).or_insert((0, 0));
            entry.0 += 1;
            if child.geometric <= 0.0 {
                entry.1 += 1;
            }
            next_gen.push(child);
        }
        {
            let mut channels: Vec<_> = channel_tally.into_iter().collect();
            channels.sort_by_key(|(name, _)| *name);
            let parts: Vec<String> = channels.iter().map(|(name, (n, z))| format!("{name}={z}/{n}")).collect();
            eprintln!("  zero-geometric-score by channel: {}", parts.join(" "));
        }
        finalize_fitness(&mut next_gen, QUALITY_FLOOR, DIVERSITY_BONUS_WEIGHT, fitness_metric.is_some());
        let best_pred = apply_predator_pressure(&mut next_gen, &mut predators, &mut predator_rng);
        next_gen.sort_by(|a, b| b.total.partial_cmp(&a.total).unwrap_or(std::cmp::Ordering::Equal));
        population_vec = next_gen;
        report_generation(gen_idx, &population_vec, gen_start.elapsed().as_secs_f64());
        print_top_pool(&population_vec, 5);
        if let Some(best_pred) = best_pred {
            eprintln!("  [predator] best fitness (corr. w/ commonness) = {best_pred:.3}");
        }

        // Stagnation check (see the tracking vars' doc comment above for
        // why this exists and how it mirrors the 2D GA's convention).
        let current_best = population_vec[0].total;
        if current_best > best_total_ever + STAGNATION_EPSILON {
            best_total_ever = current_best;
            stagnant_gens = 0;
        } else if current_best >= best_total_ever - STAGNATION_EPSILON {
            stagnant_gens = 0;
        } else {
            stagnant_gens += 1;
        }
        let mut stagnation_event = false;
        if stagnant_gens >= stagnation_gens {
            stagnation_event = true;
            eprintln!("  [stagnation] no improvement in {stagnant_gens} generations (best={best_total_ever:.3}) — resetting population, keeping only the champion");
            let champion = population_vec[0].clone();
            let mut fresh: Vec<QuatIndividual> = Vec::with_capacity(population);
            fresh.push(champion);
            // Refill entirely from fresh pool immigrants — deliberately
            // NOT bred from the (just-proven-stagnant) survivor pool,
            // same reasoning as the 2D GA's restart_population(): the
            // point is genuinely new genetic material, not a re-shuffle
            // of what already converged.
            let mut attempts = 0;
            while fresh.len() < population && attempts < population * 20 {
                attempts += 1;
                if pool_files.is_empty() {
                    break;
                }
                let path = &pool_files[rng.random_range(0..pool_files.len())];
                let Ok(g) = io::load_genome(path) else { continue };
                if g.program.is_empty() {
                    continue;
                }
                let mut ind = QuatIndividual::from_genome(&g);
                match &fitness_metric {
                    Some(metric) => score_individual_for_metric(&mut ind, metric, probe_size, use_gpu),
                    None => score_individual(&mut ind, pref_model.as_ref(), probe_size, use_gpu, 0.3),
                }
                fresh.push(ind);
            }
            finalize_fitness(&mut fresh, QUALITY_FLOOR, DIVERSITY_BONUS_WEIGHT, fitness_metric.is_some());
            // Predators reset too — their learned correlations were
            // against the population that's now being wiped, so they'd
            // otherwise be hunting a cluster that no longer exists.
            if predator_prey {
                if let Some(spec) = &fitness_metric {
                    predators = nnfractals::quat_predator::spawn_predators(PREDATOR_COUNT, spec.len(), &mut predator_rng);
                }
            }
            apply_predator_pressure(&mut fresh, &mut predators, &mut predator_rng);
            fresh.sort_by(|a, b| b.total.partial_cmp(&a.total).unwrap_or(std::cmp::Ordering::Equal));
            population_vec = fresh;
            stagnant_gens = 0;
        }
        log_gen_stats(out_dir, gen_idx, &population_vec, gen_start.elapsed().as_secs_f64(), best_pred, stagnation_event);

        if gen_idx % CHECKPOINT_INTERVAL == 0 && gen_idx != generations {
            let ckpt_start = std::time::Instant::now();
            checkpoint_stems = save_quat_population(&population_vec, out_dir, &fitness_metric, gen_idx, probe_size, use_gpu, &mut rng2, &checkpoint_stems);
            write_contact_sheet(std::path::Path::new(out_dir), &checkpoint_stems);
            eprintln!("  [checkpoint] gen {gen_idx}/{generations}: saved {} genomes to {out_dir} in {:.0}s", checkpoint_stems.len(), ckpt_start.elapsed().as_secs_f64());
        }
    }

    let final_stems = save_quat_population(&population_vec, out_dir, &fitness_metric, generations, probe_size, use_gpu, &mut rng2, &checkpoint_stems);
    eprintln!("quat-dag-evolve: saved {} genomes + thumbnails to {out_dir}", population_vec.len());
    write_contact_sheet(std::path::Path::new(out_dir), &final_stems);
}

/// `quat-dag-evolve --map-elites`'s main loop — see `quat_map_elites`'s
/// module doc and the approved plan's Phase 2 for why this exists (every
/// scalar-fitness run of this GA converges onto one lineage; an archive of
/// niches structurally can't). Population = the archive itself: gen 0
/// seeds and inserts pool immigrants directly; every later generation
/// breeds `population` children whose parents come from
/// `archive.sample_parent` (never from a truncated "survivors" list —
/// there is no such list here), scores each, and inserts it into its own
/// niche. Deliberately does NOT share `cmd_quat_dag_evolve`'s breeding
/// block verbatim: parent selection is structurally different (archive
/// cells, not `next_gen[0..survivor_count]`), so unifying them would cost
/// more clarity than it would save duplication.
#[allow(clippy::too_many_arguments)]
fn cmd_quat_dag_evolve_map_elites(
    pool_dir: &str,
    out_dir: &str,
    population: usize,
    generations: usize,
    probe_size: u32,
    use_gpu: bool,
    seed: u64,
    crossover_mode: CrossoverMode,
    mutation_strength: nnfractals::quat_genome_ops::MutationStrength,
    fitness_metric: Option<Vec<(String, f64)>>,
    stagnation_gens: usize,
) {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    const IMMIGRANT_FRAC: f64 = 0.55;
    const CROSSOVER_FRAC: f64 = 0.30;
    const RECENT_WINDOW: usize = 5;

    eprintln!(
        "fitness method: MAP-Elites — quality={}",
        match &fitness_metric {
            Some(spec) => format!("single/combo metric — {}", spec.iter().map(|(n, w)| format!("{n}:{w:.2}")).collect::<Vec<_>>().join(", ")),
            None => "taste (SigLIP preference model, src/quat_taste.rs)".to_string(),
        }
    );
    eprintln!(
        "breeding mix: crossover={:.0}% ({}) | immigrant={:.0}% | mutation={:.0}% | archive axes=sphericity,hit_rate,organization_ordinal ({} bins each = {} cells) | stagnation-gens={stagnation_gens}",
        CROSSOVER_FRAC * 100.0,
        match crossover_mode { CrossoverMode::Legacy => "legacy", CrossoverMode::Subtree => "subtree" },
        IMMIGRANT_FRAC * 100.0,
        (1.0 - IMMIGRANT_FRAC - CROSSOVER_FRAC) * 100.0,
        nnfractals::quat_map_elites::N_BINS,
        nnfractals::quat_map_elites::N_BINS.pow(3),
    );

    let mut pool_files: Vec<PathBuf> = std::fs::read_dir(pool_dir)
        .unwrap_or_else(|e| panic!("failed to read --pool-dir {pool_dir:?}: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("nn"))
        .collect();
    pool_files.shuffle(&mut rng);
    eprintln!("quat-dag-evolve --map-elites: seeding from {pool_dir} ({} files available)", pool_files.len());

    std::fs::create_dir_all(out_dir).ok();

    let mut archive: nnfractals::quat_map_elites::Archive<QuatIndividual> = nnfractals::quat_map_elites::Archive::new();

    // Gen 0: seed directly from the pool and insert whatever clears the
    // hard constraints — no breeding yet, nothing to inherit from.
    let start = std::time::Instant::now();
    let mut seeded = 0usize;
    for path in &pool_files {
        if seeded >= population { break; }
        let Ok(g) = io::load_genome(path) else { continue };
        if g.program.is_empty() { continue; }
        let mut ind = QuatIndividual::from_genome(&g);
        if let Some((cell, quality)) = map_elites_evaluate(&mut ind, &fitness_metric, probe_size, use_gpu) {
            archive.insert(cell, ind, quality, 0);
        }
        seeded += 1;
    }
    eprintln!(
        "gen 0: {:.0}s | seeded {seeded} candidates | archive: {}/{} cells ({:.0}% coverage) | mean quality={:.3} max quality={:.3}",
        start.elapsed().as_secs_f64(), archive.len(), nnfractals::quat_map_elites::N_BINS.pow(3), archive.coverage() * 100.0, archive.mean_quality(), archive.max_quality()
    );
    log_map_elites_stats(out_dir, 0, &archive, start.elapsed().as_secs_f64(), false);

    const STAGNATION_EPSILON: f64 = 0.005;
    let mut best_coverage_ever = archive.coverage();
    let mut best_mean_quality_ever = archive.mean_quality();
    let mut stagnant_gens: usize = 0;

    const CHECKPOINT_INTERVAL: usize = 15;
    let mut checkpoint_stems: Vec<String> = Vec::new();

    for gen_idx in 1..=generations {
        let gen_start = std::time::Instant::now();
        let mut inserted = 0usize;
        let mut new_cells = 0usize;
        let mut attempts = 0usize;
        while inserted < population && attempts < population * 10 {
            attempts += 1;
            let roll: f64 = rng.random();
            let mut child = if roll < IMMIGRANT_FRAC && !pool_files.is_empty() {
                let path = &pool_files[rng.random_range(0..pool_files.len())];
                match io::load_genome(path) {
                    Ok(g) if !g.program.is_empty() => QuatIndividual::from_genome(&g),
                    _ => continue,
                }
            } else if roll < IMMIGRANT_FRAC + CROSSOVER_FRAC && archive.len() >= 2 {
                let Some(parent_a) = archive.sample_parent(&mut rng, gen_idx, RECENT_WINDOW) else { continue };
                let parent_a = parent_a.clone();
                let Some(parent_b) = archive.sample_parent(&mut rng, gen_idx, RECENT_WINDOW) else { continue };
                let program = match crossover_mode {
                    CrossoverMode::Legacy => nnfractals::genome::crossover_program(&parent_a.program, &parent_b.program, &mut rng, 24),
                    CrossoverMode::Subtree => nnfractals::quat_genome_ops::crossover_program_subtree(&parent_a.program, &parent_b.program, &mut rng, 24),
                };
                QuatIndividual {
                    program,
                    warp: parent_a.warp.clone(),
                    julia_mode: parent_a.julia_mode,
                    jc: parent_a.jc,
                    phoenix: parent_a.phoenix,
                    bailout_radius: parent_a.bailout_radius,
                    geometric: 0.0, aesthetic: 0.0, diversity: 0.0, total: 0.0, phenotype: [0.0; 6], metric_vec: Vec::new(),
                    taste: 0.0, descriptors: [0.0; 3], hit_rate: 0.0,
                }
            } else {
                let Some(parent) = archive.sample_parent(&mut rng, gen_idx, RECENT_WINDOW) else { continue };
                let p1 = nnfractals::quat_genome_ops::mutate_program_tuned(&parent.program, &mut rng, 24, mutation_strength.max_depth, &mutation_strength);
                let program = nnfractals::quat_genome_ops::mutate_program_tuned(&p1, &mut rng, 24, mutation_strength.max_depth, &mutation_strength);
                QuatIndividual {
                    program,
                    warp: parent.warp.clone(),
                    julia_mode: parent.julia_mode,
                    jc: parent.jc,
                    phoenix: parent.phoenix,
                    bailout_radius: parent.bailout_radius,
                    geometric: 0.0, aesthetic: 0.0, diversity: 0.0, total: 0.0, phenotype: [0.0; 6], metric_vec: Vec::new(),
                    taste: 0.0, descriptors: [0.0; 3], hit_rate: 0.0,
                }
            };
            if let Some((cell, quality)) = map_elites_evaluate(&mut child, &fitness_metric, probe_size, use_gpu) {
                match archive.insert(cell, child, quality, gen_idx) {
                    nnfractals::quat_map_elites::InsertOutcome::NewCell => { new_cells += 1; }
                    nnfractals::quat_map_elites::InsertOutcome::Improved => {}
                    nnfractals::quat_map_elites::InsertOutcome::Rejected => {}
                }
            }
            inserted += 1;
        }
        eprintln!(
            "gen {gen_idx}: {:.0}s | {inserted} candidates ({new_cells} new cells) | archive: {}/{} cells ({:.0}% coverage) | mean quality={:.3} max quality={:.3}",
            gen_start.elapsed().as_secs_f64(), archive.len(), nnfractals::quat_map_elites::N_BINS.pow(3), archive.coverage() * 100.0, archive.mean_quality(), archive.max_quality()
        );

        // Stagnation (Phase 2b: measured on archive coverage + mean
        // quality, NEVER wipes the archive — unlike the scalar path's
        // champion-only reset, throwing away 124/125 niches to chase one
        // scalar record would defeat the entire point of an archive).
        let coverage = archive.coverage();
        let mean_quality = archive.mean_quality();
        let improved = coverage > best_coverage_ever + STAGNATION_EPSILON || mean_quality > best_mean_quality_ever + STAGNATION_EPSILON;
        if improved {
            best_coverage_ever = best_coverage_ever.max(coverage);
            best_mean_quality_ever = best_mean_quality_ever.max(mean_quality);
            stagnant_gens = 0;
        } else {
            stagnant_gens += 1;
        }
        let mut stagnation_event = false;
        if stagnant_gens >= stagnation_gens {
            stagnation_event = true;
            stagnant_gens = 0;
            eprintln!("  [stagnation] no new cells/quality gain in {stagnation_gens} generations — next {RECENT_WINDOW} gens replace ALL parents with fresh pool immigrants (archive itself is untouched)");
            // Implemented as a temporary override of the breeding roll for
            // the next few generations rather than a separate code path —
            // simplest correct way to inject fresh genetic material
            // without touching archive state.
            for _ in 0..RECENT_WINDOW {
                let extra_gen = gen_idx;
                for _ in 0..(population / 4).max(4) {
                    if pool_files.is_empty() { break; }
                    let path = &pool_files[rng.random_range(0..pool_files.len())];
                    let Ok(g) = io::load_genome(path) else { continue };
                    if g.program.is_empty() { continue; }
                    let mut ind = QuatIndividual::from_genome(&g);
                    if let Some((cell, quality)) = map_elites_evaluate(&mut ind, &fitness_metric, probe_size, use_gpu) {
                        archive.insert(cell, ind, quality, extra_gen);
                    }
                }
            }
        }
        log_map_elites_stats(out_dir, gen_idx, &archive, gen_start.elapsed().as_secs_f64(), stagnation_event);

        if gen_idx % CHECKPOINT_INTERVAL == 0 && gen_idx != generations {
            let ckpt_start = std::time::Instant::now();
            let (stems, stems_by_cell) = save_map_elites_archive(&archive, out_dir, gen_idx, probe_size, use_gpu, &checkpoint_stems);
            checkpoint_stems = stems;
            write_map_elites_grid(std::path::Path::new(out_dir), &archive, &stems_by_cell);
            eprintln!("  [checkpoint] gen {gen_idx}/{generations}: saved {} elites to {out_dir} in {:.0}s", checkpoint_stems.len(), ckpt_start.elapsed().as_secs_f64());
        }
    }

    let (final_stems, stems_by_cell) = save_map_elites_archive(&archive, out_dir, generations, probe_size, use_gpu, &checkpoint_stems);
    eprintln!("quat-dag-evolve --map-elites: saved {} elites to {out_dir} ({:.0}% archive coverage)", final_stems.len(), archive.coverage() * 100.0);
    write_map_elites_grid(std::path::Path::new(out_dir), &archive, &stems_by_cell);
}

/// MAP-Elites twin of `save_quat_population`: saves every archive elite
/// under a stable `me_<cellname>_<content_hash>` filename (NOT a random
/// id — see `QuatIndividual::content_hash`'s doc comment) so an elite that
/// survives unchanged across checkpoints keeps the same filename instead
/// of being deleted and re-saved under a new random name every time.
/// Deletes `prev_stems` first, same "out_dir always reflects only the
/// latest snapshot" convention as `save_quat_population`. Returns the new
/// stems plus a cell -> stem map (`write_map_elites_grid` needs to find
/// each cell's thumbnail).
#[allow(clippy::too_many_arguments)]
fn save_map_elites_archive(
    archive: &nnfractals::quat_map_elites::Archive<QuatIndividual>,
    out_dir: &str,
    generations: usize,
    probe_size: u32,
    use_gpu: bool,
    prev_stems: &[String],
) -> (Vec<String>, std::collections::HashMap<[u8; 3], String>) {
    std::fs::create_dir_all(out_dir).expect("create out_dir");
    for stem in prev_stems {
        let base = std::path::Path::new(out_dir).join(stem);
        let _ = std::fs::remove_file(base.with_extension("nn"));
        let _ = std::fs::remove_file(base.with_extension("png"));
        for i in 0..THUMB_ANIM_FRAMES {
            let _ = std::fs::remove_file(std::path::Path::new(out_dir).join(format!("{stem}_thumb_{i:02}.png")));
        }
    }
    let mut saved_stems = Vec::new();
    let mut stems_by_cell = std::collections::HashMap::new();
    for (cell, elite) in archive.elites() {
        let ind = &elite.individual;
        let mut g = ind.as_genome();
        g.fitness = elite.quality as f32;
        g.quat_taste = ind.taste as f32;
        let cname = nnfractals::quat_map_elites::cell_name(*cell);
        g.quat_me_cell = cname.clone();
        g.formula_readable = format!(
            "quat-dag-evolve --map-elites gen={generations} cell={cname} quality={:.3} gen_added={} sph={:.3} sol={:.3} ord={:.3}",
            elite.quality, elite.gen_added, ind.descriptors[0], ind.descriptors[1], ind.descriptors[2]
        );
        let full = score_genome_dag_full(&g, probe_size, use_gpu);
        apply_quat_full_metrics(&mut g, &full);
        apply_organization_metrics(&mut g);
        let stem = format!("me_{cname}_{:016x}", ind.content_hash());
        let path = std::path::Path::new(out_dir).join(format!("{stem}.nn"));
        io::save_genome(&g, &path).unwrap_or_else(|e| panic!("failed to save {path:?}: {e}"));
        render_genome_thumbnails(&g, std::path::Path::new(out_dir), &stem, use_gpu);
        saved_stems.push(stem.clone());
        stems_by_cell.insert(*cell, stem);
    }
    (saved_stems, stems_by_cell)
}

/// Contact sheet for `--map-elites`: rows = sphericity bin, cols =
/// hit_rate/solidity bin — the third axis (organization_ordinal) folds
/// into "show whichever ordinal bin currently holds the best quality at
/// that (sphericity, solidity) pair," rather than a 3rd nested strip
/// dimension, to keep one picture readable. Empty (sphericity, solidity)
/// pairs (no occupied ordinal bin at all) render as a dark placeholder,
/// same convention as `write_contact_sheet`'s general "worth looking at
/// once" goal.
fn write_map_elites_grid(out_dir: &std::path::Path, archive: &nnfractals::quat_map_elites::Archive<QuatIndividual>, stems_by_cell: &std::collections::HashMap<[u8; 3], String>) {
    let cell_w = 150u32;
    let cell_h = 200u32;
    let n = nnfractals::quat_map_elites::N_BINS as u32;
    let mut sheet = image::RgbImage::from_pixel(n * cell_w, n * cell_h, image::Rgb([20, 20, 24]));
    for sph in 0..n {
        for sol in 0..n {
            let mut best: Option<(&[u8; 3], f64)> = None;
            for (cell, elite) in archive.elites() {
                if cell[0] as u32 == sph && cell[1] as u32 == sol && best.map(|(_, q)| elite.quality > q).unwrap_or(true) {
                    best = Some((cell, elite.quality));
                }
            }
            let Some((cell, _)) = best else { continue };
            let Some(stem) = stems_by_cell.get(cell) else { continue };
            let path = out_dir.join(format!("{stem}.png"));
            let Ok(img) = image::open(&path) else { continue };
            let thumb = img.resize(cell_w, cell_h, image::imageops::FilterType::Triangle).to_rgb8();
            image::imageops::overlay(&mut sheet, &thumb, (sph * cell_w) as i64, (sol * cell_h) as i64);
        }
    }
    let path = out_dir.join("_me_grid.png");
    sheet.save(&path).ok();
    eprintln!("  wrote MAP-Elites grid ({n}x{n}, rows=sphericity cols=solidity): {}", path.display());
}

/// MAP-Elites twin of `log_gen_stats` — appends to the same
/// `evolve_stats.jsonl` (a run only ever runs in one mode, so there's no
/// schema collision within one file) with the archive-shaped fields the
/// plan's Phase 2b calls for: `cells_filled`, `coverage`, `mean_quality`,
/// `max_quality`, plus `taste_model_mtime` so the dashboard can show
/// whether a retrain (Phase 3) has landed since the run started.
fn log_map_elites_stats(out_dir: &str, gen_idx: usize, archive: &nnfractals::quat_map_elites::Archive<QuatIndividual>, secs: f64, stagnation_event: bool) {
    let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let taste_mtime = std::fs::metadata("taste_model_quat.npz").ok().and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_secs());
    let taste_mtime_field = taste_mtime.map(|v| v.to_string()).unwrap_or_else(|| "null".to_string());
    let total_cells = nnfractals::quat_map_elites::N_BINS.pow(3);
    let line = format!(
        "{{\"gen\":{gen_idx},\"ts\":{ts},\"secs\":{secs:.1},\"mode\":\"map_elites\",\"cells_filled\":{},\"total_cells\":{total_cells},\"coverage\":{:.4},\"mean_quality\":{:.4},\"max_quality\":{:.4},\"stagnation_event\":{stagnation_event},\"taste_model_mtime\":{taste_mtime_field}}}",
        archive.len(), archive.coverage(), archive.mean_quality(), archive.max_quality().max(0.0)
    );
    let path = std::path::Path::new(out_dir).join("evolve_stats.jsonl");
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
        use std::io::Write as _;
        let _ = writeln!(f, "{line}");
    }
}

/// `taste-pairs`: picks up to `n` genomes from a `--map-elites` archive
/// dir worth Carl rating next, copies them (`.nn` + `.png` + thumbs — all
/// already rendered, since `save_map_elites_archive` wrote them) into
/// `out_dir`, and writes `pairs.txt` listing the intended pairings. The
/// browser needs NO routing change to pick these up: `is_quat_genome_path`
/// prefix-matches on `fractals_dag_quat` (see `src/lib.rs`), and
/// `out_dir`'s default (`fractals_dag_quat_to_rate`) already satisfies
/// that — confirmed during planning, not assumed.
///
/// Selection (simplified from the plan's full 4-criterion design — see
/// the doc comment below on what's deliberately NOT implemented yet):
///   (a) uncertainty — genomes whose `quat_taste` sits closest to the
///       archive's median (the model is least confident there, so a
///       human comparison teaches it the most).
///   (b) diversity — capped at 2 picks per archive CELL (`quat_me_cell`),
///       so one crowded niche can't dominate a rating batch.
///   (c) recency — automatic: a `--map-elites` out-dir only ever holds
///       the LATEST checkpoint's elites (`save_map_elites_archive`
///       deletes the previous checkpoint's files first), so every
///       candidate here already IS the most recent snapshot.
///   (d) always includes the current top-3 `quat_taste` elites, so Carl
///       can veto the model's own favourites.
///
/// NOT implemented (the plan's fuller version, deferred — this is a
/// first working cut, not the final word): using the taste sidecar's
/// `EMBED` command for an embedding-distance diversity check on top of
/// the cell cap, and folding in `Starred/`/`favorite:true` genomes as
/// extra positives (that's training-time, in `train_taste_quat.py
/// --starred`, already supported there — see Phase 1's `--starred` flag
/// — just not wired into THIS selection step).
fn cmd_taste_pairs(archive_dir: &str, out_dir: &str, n: usize) {
    #[derive(Clone)]
    struct Candidate {
        stem: String,
        taste: f32,
        cell: String,
    }
    let mut candidates: Vec<Candidate> = std::fs::read_dir(archive_dir)
        .unwrap_or_else(|e| panic!("failed to read --archive {archive_dir:?}: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("nn"))
        .filter_map(|p| {
            let g = io::load_genome(&p).ok()?;
            let stem = p.file_stem()?.to_str()?.to_string();
            Some(Candidate { stem, taste: g.quat_taste, cell: g.quat_me_cell })
        })
        .collect();
    if candidates.is_empty() {
        eprintln!("taste-pairs: no genomes found in --archive {archive_dir:?} — nothing to do.");
        return;
    }

    let mut tastes: Vec<f32> = candidates.iter().map(|c| c.taste).collect();
    tastes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = tastes[tastes.len() / 2];

    candidates.sort_by(|a, b| a.taste.partial_cmp(&b.taste).unwrap_or(std::cmp::Ordering::Equal));
    let mut top3: Vec<Candidate> = candidates.iter().rev().take(3).cloned().collect();

    let mut by_uncertainty = candidates.clone();
    by_uncertainty.sort_by(|a, b| (a.taste - median).abs().partial_cmp(&(b.taste - median).abs()).unwrap_or(std::cmp::Ordering::Equal));

    let mut selected: Vec<Candidate> = Vec::new();
    let mut seen_stems: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut per_cell: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

    // Top-3 taste elites always get a slot first (Carl's veto power).
    for c in top3.drain(..) {
        if seen_stems.insert(c.stem.clone()) {
            *per_cell.entry(c.cell.clone()).or_insert(0) += 1;
            selected.push(c);
        }
    }
    // Then fill the rest by uncertainty, capped at 2/cell.
    for c in by_uncertainty {
        if selected.len() >= n { break; }
        if seen_stems.contains(&c.stem) { continue; }
        let count = per_cell.entry(c.cell.clone()).or_insert(0);
        if *count >= 2 { continue; }
        *count += 1;
        seen_stems.insert(c.stem.clone());
        selected.push(c);
    }

    std::fs::create_dir_all(out_dir).expect("create --out dir");
    for c in &selected {
        for ext in ["nn", "png"] {
            let src = std::path::Path::new(archive_dir).join(format!("{}.{ext}", c.stem));
            let dst = std::path::Path::new(out_dir).join(format!("{}.{ext}", c.stem));
            let _ = std::fs::copy(&src, &dst);
        }
        for i in 0..THUMB_ANIM_FRAMES {
            let name = format!("{}_thumb_{i:02}.png", c.stem);
            let src = std::path::Path::new(archive_dir).join(&name);
            let dst = std::path::Path::new(out_dir).join(&name);
            let _ = std::fs::copy(&src, &dst);
        }
    }

    // Pairs.txt: shuffle, pair consecutive picks (odd one out is dropped
    // from pairs.txt but stays in out_dir — the browser's own ⚖ Rate mode
    // draws random pairs from whatever's in the folder anyway, this file
    // is just a suggested pairing for a human/script working outside the
    // browser).
    let mut rng = rand::rng();
    let mut shuffled = selected.clone();
    shuffled.shuffle(&mut rng);
    let mut pairs_txt = String::new();
    for pair in shuffled.chunks(2) {
        if let [a, b] = pair {
            pairs_txt.push_str(&format!("{}.nn vs {}.nn\n", a.stem, b.stem));
        }
    }
    std::fs::write(std::path::Path::new(out_dir).join("pairs.txt"), pairs_txt).ok();

    eprintln!("taste-pairs: selected {}/{} candidates from {archive_dir} -> {out_dir} (median taste={median:.3})", selected.len(), candidates.len());
}

/// Deletes `prev_stems`' files (`.nn`, `.png`, the 8 animated-thumbnail
/// frames — everything `render_genome_thumbnails` produces for a stem)
/// then saves `population` fresh under newly-drawn ids, returning the
/// new stems. Shared by both the final save and the periodic mid-run
/// checkpoint below — deleting the previous save first keeps `out_dir`
/// always reflecting exactly the LATEST snapshot rather than
/// accumulating every checkpoint's files forever.
#[allow(clippy::too_many_arguments)]
fn save_quat_population(
    population_vec: &[QuatIndividual],
    out_dir: &str,
    fitness_metric: &Option<Vec<(String, f64)>>,
    generations: usize,
    probe_size: u32,
    use_gpu: bool,
    rng2: &mut rand::rngs::StdRng,
    prev_stems: &[String],
) -> Vec<String> {
    std::fs::create_dir_all(out_dir).expect("create out_dir");
    for stem in prev_stems {
        let base = std::path::Path::new(out_dir).join(stem);
        let _ = std::fs::remove_file(base.with_extension("nn"));
        let _ = std::fs::remove_file(base.with_extension("png"));
        for i in 0..THUMB_ANIM_FRAMES {
            let _ = std::fs::remove_file(std::path::Path::new(out_dir).join(format!("{stem}_thumb_{i:02}.png")));
        }
    }
    let mut saved_stems = Vec::with_capacity(population_vec.len());
    for ind in population_vec {
        let id: u64 = rng2.random();
        let mut g = ind.as_genome();
        g.id = id;
        g.fitness = ind.total as f32;
        g.formula_readable = match fitness_metric {
            Some(spec) => {
                let spec_str: String = spec.iter().map(|(n, w)| format!("{n}:{w:.2}")).collect::<Vec<_>>().join(",");
                format!("quat-dag-evolve gen={generations} fitness-metric=[{spec_str}] value={:.3}", ind.total)
            }
            None => format!(
                "quat-dag-evolve gen={generations} geometric={:.3} aesthetic={:.3} diversity={:.3} total={:.3}",
                ind.geometric, ind.aesthetic, ind.diversity, ind.total
            ),
        };
        // Full metric sweep — deliberately only here, once per SAVED
        // genome, not in score_individual's per-generation hot path (see
        // score_genome_dag_full's doc comment).
        let full = score_genome_dag_full(&g, probe_size, use_gpu);
        apply_quat_full_metrics(&mut g, &full);
        apply_organization_metrics(&mut g);
        // A single/combo-metric run (--fitness-metric) gets the metric
        // name(s) baked into every saved filename — Carl's own ask, so a
        // batch of test runs saved into the same --out-dir stays
        // identifiable by eye in the gallery/file listing, not just in
        // formula_readable's text. Multiple metrics join with "+"; exact
        // weights live in formula_readable, not the filename (keeps it
        // from becoming unreadably long).
        let stem = match fitness_metric {
            Some(spec) => {
                let names: String = spec.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>().join("+");
                format!("{names}_{id:016x}")
            }
            None => format!("{id:016x}"),
        };
        let path = std::path::Path::new(out_dir).join(format!("{stem}.nn"));
        io::save_genome(&g, &path).unwrap_or_else(|e| panic!("failed to save {path:?}: {e}"));
        render_genome_thumbnails(&g, std::path::Path::new(out_dir), &stem, use_gpu);
        saved_stems.push(stem);
    }
    saved_stems
}

/// Generates `count` genuinely RANDOM (never-evolved, no selection
/// pressure of any kind) quaternion DAG genomes — fresh
/// `random_program` draws plus `randomize_dynamics` (julia/phoenix/
/// warp/bailout, the same probabilities a brand-new 2D genome gets) —
/// scores each with the full metric sweep, and saves genomes +
/// thumbnails + a contact sheet into `out_dir`.
///
/// Exists specifically to test the new metrics against an UNBIASED
/// sample of the formula space, per Carl's own framing: "Use only
/// random fractals, no evolution... It is to test metrics, not GA."
/// Seeding `quat-dag-evolve` from `--pool-dir fractals_dag` (the normal
/// path) draws from genomes the 2D GA already selected for 2D beauty —
/// not a neutral baseline for asking "does this metric track anything
/// real," and evolution's own selection pressure is exactly the
/// confound this needs to be free of.
fn cmd_quat_dag_random(out_dir: &str, count: usize, use_gpu: bool, seed: u64) {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    std::fs::create_dir_all(out_dir).expect("create out_dir");
    let mut saved_stems = Vec::with_capacity(count);
    let start = std::time::Instant::now();
    for i in 0..count {
        let exotic = rng.random_bool(0.3);
        let program = nnfractals::genome::random_program(&mut rng, 24, 6, exotic);
        let mut g = Genome { program, ..Default::default() };
        g.randomize_dynamics(&mut rng);
        let id: u64 = rng.random();
        g.id = id;
        g.formula_readable = format!("quat-dag-random seed={seed} #{i} exotic={exotic}");
        let full = score_genome_dag_full(&g, 128, use_gpu);
        apply_quat_full_metrics(&mut g, &full);
        apply_organization_metrics(&mut g);
        g.fitness = full.breakdown.total();
        let path = std::path::Path::new(out_dir).join(format!("{id:016x}.nn"));
        io::save_genome(&g, &path).unwrap_or_else(|e| panic!("failed to save {path:?}: {e}"));
        render_genome_thumbnails(&g, std::path::Path::new(out_dir), &format!("{id:016x}"), use_gpu);
        saved_stems.push(format!("{id:016x}"));
        if (i + 1) % 20 == 0 || i + 1 == count {
            eprint!("\r  {}/{count} ({:.0}s)   ", i + 1, start.elapsed().as_secs_f64());
        }
    }
    eprintln!("\r  quat-dag-random: saved {count} random genomes + thumbnails to {out_dir} in {:.0}s", start.elapsed().as_secs_f64());
    write_contact_sheet(std::path::Path::new(out_dir), &saved_stems);
}

/// Tiles every saved genome's static thumbnail (`{stem}.png`, already
/// rendered by `render_genome_thumbnails`) into one grid image — a single
/// picture worth actually looking at to judge population diversity,
/// rather than scrolling 40-100 tiny animated cells in the browser one at
/// a time.
fn write_contact_sheet(out_dir: &std::path::Path, stems: &[String]) {
    let cell_w = 150u32;
    let cell_h = 200u32;
    let cols = (stems.len() as f64).sqrt().ceil().max(1.0) as u32;
    let rows = (stems.len() as u32).div_ceil(cols.max(1));
    let mut sheet = image::RgbImage::from_pixel(cols * cell_w, rows * cell_h, image::Rgb([20, 20, 24]));
    for (i, stem) in stems.iter().enumerate() {
        let path = out_dir.join(format!("{stem}.png"));
        let Ok(img) = image::open(&path) else { continue };
        let thumb = img.resize(cell_w, cell_h, image::imageops::FilterType::Triangle).to_rgb8();
        let (x0, y0) = (((i as u32) % cols) * cell_w, ((i as u32) / cols) * cell_h);
        let (ox, oy) = ((cell_w.saturating_sub(thumb.width())) / 2, (cell_h.saturating_sub(thumb.height())) / 2);
        image::imageops::overlay(&mut sheet, &thumb, (x0 + ox) as i64, (y0 + oy) as i64);
    }
    let path = out_dir.join("_contact_sheet.png");
    sheet.save(&path).ok();
    eprintln!("  wrote contact sheet: {}", path.display());
}

/// How many animated-thumbnail frames to render per genome and at what
/// resolution — must match `browser.rs`'s `ANIM_THUMB_FRAMES`/naming
/// convention (`{stem}_thumb_{i:02}.png`) exactly, since that's the only
/// contract between this writer and that reader.
const THUMB_ANIM_FRAMES: usize = 8;
const THUMB_ANIM_SIZE: u32 = 96;

/// Renders both the static `{stem}.png` (the browser's existing
/// fallback/rating-view thumbnail — a still at C=0) and the
/// `{stem}_thumb_00..07.png` animated sequence (one orbit turn around the
/// object, C held fixed so the sequence shows the SHAPE rotating rather
/// than conflating that with a color pulse) for one genome, into
/// `out_dir`. Reuses the exact rendering/coloring/framing path every
/// other quat-raymarch command already uses — no new render logic.
fn render_genome_thumbnails(genome: &Genome, out_dir: &std::path::Path, stem: &str, use_gpu: bool) {
    let formula = nnfractals::quat_dag::QuatDagFormula { prog: &genome.program, warp: &genome.warp, julia: genome.julia_mode, jc: (genome.julia_cre, genome.julia_cim), phoenix: (genome.phoenix_re, genome.phoenix_im) };
    let domain_radius = 1.6;
    let fov_deg = 45.0;
    let bg_color = (0.03, 0.02, 0.06);
    let base_params = nnfractals::quat_dag::RaymarchDagParams {
        formula,
        time_axis: nnfractals::quat_fractal::TimeAxis::C,
        time_val: 0.0,
        domain_radius,
        max_iter: 50,
        bailout: genome.bailout_radius as f64,
        max_march_steps: 150,
        hit_epsilon: domain_radius * 1e-4,
        step_safety: 0.8,
        light_dir: (0.5, 0.8, 0.3),
        normal_eps: domain_radius * 1e-3,
        color_probe_offset: domain_radius * 1e-2,
        aa: 1,
    };
    let mut mode = resolve_dag_gpu_mode(base_params.formula.prog, base_params.formula.warp, use_gpu);

    // Static thumbnail: same still-framing convention as
    // render_still_for_aesthetic, at a size worth looking at in the
    // rating view.
    {
        let (w, h) = (300u32, 400u32);
        let radius = recommended_orbit_radius(domain_radius, fov_deg, w, h);
        let cam = nnfractals::quat_raymarch::RaymarchCamera { eye: (0.0, 0.0, -radius), target: (0.0, 0.0, 0.0), up_hint: (0.0, 1.0, 0.0), fov_y: fov_deg.to_radians() };
        let (shading, color_t) = render_dag_frame_with_mode(&mut mode, &base_params, &cam, w, h);
        let rgb = raymarch_frame_to_rgb(&shading, &color_t, base_params.max_iter, "lava", bg_color);
        let path = out_dir.join(format!("{stem}.png"));
        io::save_png(&rgb, w, h, &path).ok();
    }

    // Animated sequence: one orbit turn, small and fast.
    let radius = recommended_orbit_radius(domain_radius, fov_deg, THUMB_ANIM_SIZE, THUMB_ANIM_SIZE);
    let orbit = nnfractals::quat_raymarch::RaymarchOrbitParams {
        target: (0.0, 0.0, 0.0),
        axis: (0.35, 1.0, 0.15),
        radius,
        turns: 1.0,
        phase0: 0.0,
        fov_y: fov_deg.to_radians(),
    };
    for i in 0..THUMB_ANIM_FRAMES {
        let t = i as f64 / THUMB_ANIM_FRAMES as f64;
        let cam = orbit.sample(t);
        let (shading, color_t) = render_dag_frame_with_mode(&mut mode, &base_params, &cam, THUMB_ANIM_SIZE, THUMB_ANIM_SIZE);
        let rgb = raymarch_frame_to_rgb(&shading, &color_t, base_params.max_iter, "lava", bg_color);
        let path = out_dir.join(format!("{stem}_thumb_{i:02}.png"));
        io::save_png(&rgb, THUMB_ANIM_SIZE, THUMB_ANIM_SIZE, &path).ok();
    }
}

/// View-set size for the taste model's "4D" input — see `render_genome_views`.
const VIEW_ANGLES: usize = 3;
const VIEW_C_VALUES: usize = 3;
const VIEW_SIZE: u32 = 224;

/// Renders a fixed 9-image view set (`{stem}_view_00..08.png`, 224×224 —
/// SigLIP's native input size) for one genome: 3 orbit angles (0°, 120°,
/// 240° around the same orbit axis `render_genome_thumbnails` uses) crossed
/// with 3 C values (0, +c_half, -c_half). This is the taste model's "4D"
/// input — a single still at C=0 only shows one cross-section of a
/// quaternion object; sweeping C is how the model sees the shape actually
/// changing through the 4th (julia-parameter) axis, the same way the
/// existing animated thumbnail sweeps orbit angle to show 3D shape.
///
/// `c_half` follows the same bailout-scaled convention `quat_viewer.rs`'s
/// `CRangeScan::start` uses as its initial probe step
/// (`(bailout_radius * 0.25).max(0.02)`) — not a full adaptive boundary
/// scan (that's tuned for finding the true degenerate-crossing point
/// interactively), just a fixed, cheap fraction of bailout that stays
/// in-range for the vast majority of genomes.
///
/// Deliberately NOT reusing `render_genome_thumbnails`' 96×96
/// `_thumb_00..07.png` sequence: those are one orbit turn at FIXED C
/// (shape only, no C-sweep) and too small for SigLIP's 224×224 native
/// input — resizing up would add no information, only blur.
fn render_genome_views(genome: &Genome, out_dir: &std::path::Path, stem: &str, use_gpu: bool) -> Vec<std::path::PathBuf> {
    let formula = nnfractals::quat_dag::QuatDagFormula { prog: &genome.program, warp: &genome.warp, julia: genome.julia_mode, jc: (genome.julia_cre, genome.julia_cim), phoenix: (genome.phoenix_re, genome.phoenix_im) };
    let domain_radius = 1.6;
    let fov_deg = 45.0;
    let bg_color = (0.03, 0.02, 0.06);
    let base_params = nnfractals::quat_dag::RaymarchDagParams {
        formula,
        time_axis: nnfractals::quat_fractal::TimeAxis::C,
        time_val: 0.0,
        domain_radius,
        max_iter: 50,
        bailout: genome.bailout_radius as f64,
        max_march_steps: 150,
        hit_epsilon: domain_radius * 1e-4,
        step_safety: 0.8,
        light_dir: (0.5, 0.8, 0.3),
        normal_eps: domain_radius * 1e-3,
        color_probe_offset: domain_radius * 1e-2,
        aa: 1,
    };
    let mut mode = resolve_dag_gpu_mode(base_params.formula.prog, base_params.formula.warp, use_gpu);
    let radius = recommended_orbit_radius(domain_radius, fov_deg, VIEW_SIZE, VIEW_SIZE);
    let orbit = nnfractals::quat_raymarch::RaymarchOrbitParams {
        target: (0.0, 0.0, 0.0),
        axis: (0.35, 1.0, 0.15),
        radius,
        turns: 1.0,
        phase0: 0.0,
        fov_y: fov_deg.to_radians(),
    };
    let c_half = (genome.bailout_radius as f64 * 0.25).max(0.02);
    let c_values = [0.0, c_half, -c_half];
    let mut paths = Vec::with_capacity(VIEW_ANGLES * VIEW_C_VALUES);
    let mut idx = 0usize;
    for a in 0..VIEW_ANGLES {
        let t = a as f64 / VIEW_ANGLES as f64;
        let cam = orbit.sample(t);
        for &c in &c_values {
            let frame_params = nnfractals::quat_dag::RaymarchDagParams { time_val: c, ..base_params };
            let (shading, color_t) = render_dag_frame_with_mode(&mut mode, &frame_params, &cam, VIEW_SIZE, VIEW_SIZE);
            let rgb = raymarch_frame_to_rgb(&shading, &color_t, frame_params.max_iter, "lava", bg_color);
            let path = out_dir.join(format!("{stem}_view_{idx:02}.png"));
            io::save_png(&rgb, VIEW_SIZE, VIEW_SIZE, &path).ok();
            paths.push(path);
            idx += 1;
        }
    }
    paths
}

/// Renders one role-model still (a classic `QuatFormula`, not an evolved
/// DAG genome) in the EXACT style `render_genome_thumbnails`' static
/// branch uses (same 300x400 size, camera, domain_radius, colormap, bg
/// color) so its embedding sits in the same distribution as every rated
/// corpus genome's `.png` — anything else would make "role model vs.
/// chaotic genome" bootstrap pairs compare apples to oranges. Also writes
/// a stub `.nn` (`fractal_kind = "quat_builtin"`, empty `program` — never
/// re-rendered or re-evolved, only read as JSON by the Python trainer for
/// its `png_for` path and field writes) so it fits the same `--dirs
/// train_corpus_quat` corpus-scanning convention every other tool uses.
fn render_role_model_still(
    formula: nnfractals::quat_fractal::QuatFormula,
    bulb_power: f64,
    mandelbox_scale: f64,
    out_dir: &std::path::Path,
    stem: &str,
    label: &str,
    use_gpu: bool,
) {
    let domain_radius = 1.6;
    let fov_deg = 45.0;
    let bg_color = (0.03, 0.02, 0.06);
    let params = nnfractals::quat_raymarch::RaymarchParams {
        formula,
        time_axis: nnfractals::quat_fractal::TimeAxis::C,
        time_val: 0.0,
        domain_radius,
        max_iter: 60,
        bailout: 4.0,
        max_march_steps: 150,
        hit_epsilon: domain_radius * 1e-4,
        step_safety: 0.8,
        light_dir: (0.5, 0.8, 0.3),
        normal_eps: domain_radius * 1e-3,
        color_probe_offset: domain_radius * 1e-2,
        aa: 1,
        bulb_power,
        mandelbox_scale,
    };
    let (w, h) = (300u32, 400u32);
    let radius = recommended_orbit_radius(domain_radius, fov_deg, w, h);
    let cam = nnfractals::quat_raymarch::RaymarchCamera { eye: (0.0, 0.0, -radius), target: (0.0, 0.0, 0.0), up_hint: (0.0, 1.0, 0.0), fov_y: fov_deg.to_radians() };
    let (shading, color_t) = render_raymarch_frame_dispatch(&params, &cam, w, h, use_gpu);
    let rgb = raymarch_frame_to_rgb(&shading, &color_t, params.max_iter, "lava", bg_color);
    std::fs::create_dir_all(out_dir).expect("create out_dir");
    io::save_png(&rgb, w, h, &out_dir.join(format!("{stem}.png"))).ok();

    let mut g = Genome::default();
    g.fractal_kind = "quat_builtin".to_string();
    g.formula_readable = label.to_string();
    let path = out_dir.join(format!("{stem}.nn"));
    io::save_genome(&g, &path).unwrap_or_else(|e| panic!("failed to save {path:?}: {e}"));
}

fn report_generation(gen_idx: usize, population: &[QuatIndividual], secs: f64) {
    let n = population.len().max(1) as f64;
    let mean_total: f64 = population.iter().map(|i| i.total).sum::<f64>() / n;
    let mean_div: f64 = population.iter().map(|i| i.diversity).sum::<f64>() / n;
    let best = &population[0];
    let worst = population.last().unwrap();
    let shapes = count_unique_shapes(population);
    eprintln!(
        "gen {gen_idx}: {secs:.0}s | best total={:.3} (geo={:.3} aes={:.3} div={:.3}) | mean total={mean_total:.3} div={mean_div:.3} | worst total={:.3} | {shapes}/{} unique shapes",
        best.total, best.geometric, best.aesthetic, best.diversity, worst.total, population.len()
    );
}

/// Compact top-of-pool readout, printed every generation right after
/// `report_generation`'s aggregate line — Carl's ask ("a list of the
/// individuals at the top of the pool, with formula and fitness... I
/// need to be able to evaluate in the blink of an eye what is going
/// on"). `population` is already kept sorted by `.total` descending
/// (every call site sorts right before calling this), so `.take(n)` is
/// exactly the current elite. Reuses `Genome::formula_expr()` (the same
/// renderer the 2D GA and `.nn` files' `formula_readable` convention are
/// built on) via the same throwaway-`Genome` bridge `as_genome()` uses
/// for scoring/saving — cheap (string formatting over a capped-size
/// AST), safe to do unconditionally every generation. Truncated to one
/// line per individual so a whole top-N block stays glanceable instead
/// of wrapping the terminal.
fn print_top_pool(population: &[QuatIndividual], n: usize) {
    const MAX_EXPR_LEN: usize = 120;
    eprintln!("  top {}:", n.min(population.len()));
    for (i, ind) in population.iter().take(n).enumerate() {
        let mut expr = ind.as_genome().formula_expr();
        if expr.chars().count() > MAX_EXPR_LEN {
            expr = expr.chars().take(MAX_EXPR_LEN).collect::<String>();
            expr.push('…');
        }
        eprintln!(
            "    #{} total={:.3} (geo={:.3} aes={:.3} div={:.3})  {expr}",
            i + 1, ind.total, ind.geometric, ind.aesthetic, ind.diversity
        );
    }
}

/// Appends one JSON line per generation to `<out_dir>/evolve_stats.jsonl`
/// — a structured, easy-to-parse companion to `report_generation`'s
/// human-readable stderr line, built so an external process (the live
/// dashboard Carl asked for — "I would like to have a visual of what is
/// happening... whenever I ask you to run a pool") can tail the run's
/// progress without scraping stderr. Append-only, one line per
/// generation, never rewritten — safe to read from while the run is
/// still going.
fn log_gen_stats(out_dir: &str, gen_idx: usize, population: &[QuatIndividual], secs: f64, best_predator_fitness: Option<f64>, stagnation_event: bool) {
    let n = population.len().max(1) as f64;
    let mean_total: f64 = population.iter().map(|i| i.total).sum::<f64>() / n;
    let mean_diversity: f64 = population.iter().map(|i| i.diversity).sum::<f64>() / n;
    let best = &population[0];
    let worst = population.last().unwrap();
    let shapes = count_unique_shapes(population);
    let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let predator_field = best_predator_fitness.map(|v| format!("{v:.4}")).unwrap_or_else(|| "null".to_string());
    let line = format!(
        "{{\"gen\":{gen_idx},\"ts\":{ts},\"secs\":{secs:.1},\"best_total\":{:.4},\"best_geometric\":{:.4},\"mean_total\":{mean_total:.4},\"mean_diversity\":{mean_diversity:.4},\"worst_total\":{:.4},\"unique_shapes\":{shapes},\"population\":{},\"predator_best_fitness\":{predator_field},\"stagnation_event\":{stagnation_event}}}",
        best.total, best.geometric, worst.total, population.len()
    );
    let path = std::path::Path::new(out_dir).join("evolve_stats.jsonl");
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
        use std::io::Write as _;
        let _ = writeln!(f, "{line}");
    }
}

/// Parameters for pulsing the time-axis value (C by default) while the
/// camera orbits — reuses `formula::ramp_or_pulse`, the exact same
/// ModShape vocabulary `quat_gravity`'s time-value ramp already uses.
/// `c_shape: Ramp` (the default) keeps `time_val` fixed across the clip
/// IF `c0 == c1`, matching this feature's original non-animated behavior;
/// any other shape (`Sine` etc.) oscillates within `[c0,c1]`.
struct CPulseParams {
    c0: f64,
    c1: f64,
    shape: nnfractals::formula::ModShape,
    freq: f64,
    phase: f64,
}

/// Renders a `quat-raymarch-video`: an orbiting camera around a ray-marched
/// fractal, with the (otherwise-fixed) time-axis value optionally pulsing
/// over the clip via `pulse` — see `CPulseParams`. Reuses `video_export::
/// encode_rgb_frames` exactly like `cmd_quat_mandelbrot`.
fn cmd_quat_raymarch_video(
    params: nnfractals::quat_raymarch::RaymarchParams,
    orbit: nnfractals::quat_raymarch::RaymarchOrbitParams,
    pulse: CPulseParams,
    frames: u32, fps: u32, width: u32, height: u32,
    colormap_name: String,
    bg_color: (f32, f32, f32),
    use_gpu: bool,
    out_path: &Path,
) {
    use nnfractals::video_export::{encode_rgb_frames, VideoMsg};
    eprintln!(
        "quat-raymarch-video: {} frames={frames} fps={fps} {width}x{height} radius={:.2} turns={:.2} c0={:.2} c1={:.2} c_shape={:?} c_freq={:.2} colormap={colormap_name} gpu={use_gpu} -> {}",
        params.formula.name(), orbit.radius, orbit.turns, pulse.c0, pulse.c1, pulse.shape, pulse.freq, out_path.display()
    );
    let n = frames.max(2);
    let frame_iter = (0..n).map(move |i| {
        let t = i as f64 / n as f64;
        let cam = orbit.sample(t);
        let time_val = nnfractals::formula::ramp_or_pulse(pulse.c0, pulse.c1, pulse.shape, pulse.freq, pulse.phase, t);
        let frame_params = nnfractals::quat_raymarch::RaymarchParams { time_val, ..params };
        let (shading, color_t) = render_raymarch_frame_dispatch(&frame_params, &cam, width, height, use_gpu);
        raymarch_frame_to_rgb(&shading, &color_t, frame_params.max_iter, &colormap_name, bg_color)
    });
    if let Some(dir) = out_path.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir).expect("create out dir");
        }
    }
    let (tx, rx) = std::sync::mpsc::channel::<VideoMsg>();
    encode_rgb_frames(frame_iter, n, fps, width, height, out_path, &tx, &|| {});
    drop(tx);
    for msg in rx {
        match msg {
            VideoMsg::Started { pid } => eprintln!("ffmpeg started (pid {pid})"),
            VideoMsg::Progress { done, total } => eprint!("\rframe {done}/{total}   "),
            VideoMsg::Done(p) => eprintln!("\ndone: {}", p.display()),
            VideoMsg::Failed(e) => {
                eprintln!("\nFAILED: {e}");
                std::process::exit(1);
            }
        }
    }
}

/// Prints a `quat-gravity-report`: simulates+probes a trajectory (no video
/// export at all) and reports whether it stays visually alive across its
/// full length, so `gravity`'s physics parameters can be retuned for a
/// long clip in seconds instead of by rendering it and finding out.
fn cmd_quat_gravity_report(
    sim_params: &nnfractals::quat_gravity::ProjectileParams, frames: u32, probe_res: u32, buckets: usize, delta_window: usize,
) {
    let report = nnfractals::quat_gravity::probe_trajectory(sim_params, frames, probe_res, delta_window);
    println!("quat-gravity-report: {} frames, {}x{} probe, delta-window={}", frames, probe_res, probe_res, delta_window);
    println!();
    println!("{}", report.ascii_timeline(buckets));
    println!("(. alive  _ blank  # flat/interior  = static — each char is ~{} frames)", frames as usize / buckets.max(1));
    println!();
    println!(
        "blank={} ({:.0}%)  flat={} ({:.0}%)  static={} ({:.0}%)  longest dead run={} frames ({:.0}%)",
        report.blank_frames, 100.0 * report.blank_frames as f32 / frames as f32,
        report.flat_frames, 100.0 * report.flat_frames as f32 / frames as f32,
        report.static_frames, 100.0 * report.static_frames as f32 / frames as f32,
        report.longest_dead_run, 100.0 * report.longest_dead_run as f32 / frames as f32,
    );
    let notable: Vec<_> = report.dead_runs.iter().filter(|r| r.len as f32 > frames as f32 * 0.02).collect();
    if !notable.is_empty() {
        println!();
        println!("notable runs (>2% of clip), kind/start_frame/len:");
        for r in notable {
            println!("  {:>6}  frame {:>5}  len {:>5}  ({:.0}%)", r.kind, r.start_frame, r.len, 100.0 * r.len as f32 / frames as f32);
        }
    }
}

/// One generation of `time_ga::run`, printed with enough detail to diagnose a
/// search that isn't converging without re-running it.
///
/// The one-line summary answers "is it working"; everything after answers
/// "why not". `full_rejected_by_depth` is the single most useful line here —
/// a rejection concentrated at the same (deepest) sample index generation
/// after generation is a real depth limit the search cannot search around; one
/// spread across shallow and deep alike means the population itself is the
/// problem (not exploring, or amplitudes landing in a bad range) and more
/// generations or population are likely to help.
fn print_gen_report(r: &nnfractals::time_ga::GenReport) {
    let flag = if r.best_passed { "✓" } else { "·" };
    println!("  gen {:>2}  best {:.4}  {}/{} full-tier  {} passing  ({} evaluated, {} targets)",
             r.generation, r.best, r.full_passed, r.full_evaluated, r.passed, r.evaluated,
             r.unique_targets);
    if !r.cheap_rejected.is_empty() {
        let parts: Vec<String> = r.cheap_rejected.iter().map(|(why, n)| format!("{n} {why}")).collect();
        println!("        cheap rejects: {}", parts.join(", "));
    }
    if r.full_evaluated > 0 && !r.full_rejected.is_empty() {
        let by_reason: Vec<String> = r.full_rejected.iter().map(|(why, n)| format!("{n} {why}")).collect();
        let by_depth: Vec<String> = r.full_rejected_by_depth.iter()
            .map(|(d, n)| format!("d{d}:{n}")).collect();
        println!("        full  rejects: {}  (at depth {})",
                 by_reason.join(", "), by_depth.join(" "));
    }
    println!("        {flag} {}", r.best_label);
}

/// The full decomposition of one individual's fitness, depth by depth, with
/// every gate's margin and nothing short-circuited — what backs a single
/// number ("score 0.42" or "rejected: noise") when that number alone isn't
/// enough to say why the search landed where it did.
fn print_explain(
    ind: &nnfractals::time_ga::Individual, start_zoom: f64,
    explains: &[nnfractals::time_ga::DepthExplain],
) {
    println!("\n── decomposition: {} ──", ind.label());
    for (i, tp) in ind.progs.iter().enumerate() {
        let p = tp.profile();
        println!("  [{i}] {}  amp={:.6}  freq={:.2}  phase={:.3}  {}",
                 tp.target.label(), tp.amp, tp.freq, tp.phase, tp.expr());
        println!("       free-gate profile: finite={} travel_rel={:.3} max_step={:.3} loops={}",
                 p.finite, p.travel_rel, p.max_step, p.loops);
    }
    for d in explains {
        let doublings = (d.zoom / start_zoom).max(1e-300).log2().max(0.0);
        println!("\n  d{}  zoom={:.3e}  ({doublings:.1} doublings)  cx={:.4e}  cy={:.4e}",
                 d.depth, d.zoom, d.cx, d.cy);
        for g in &d.gates {
            let mark = if g.passed { "✓" } else { "✗" };
            let cmp = if g.direction == "at most" { "≤" } else { "≥" };
            println!("        {:<11} {:>10.4}  {cmp} {:<8.4}  {mark}", g.name, g.value, g.threshold);
        }
        println!("        (reference, not gated) mean_coherence={:.3}  min_change={:.3}",
                 d.stats.mean_coherence, d.stats.min_change);
        match d.rejected {
            Some(why) => println!("        → REJECTED: {why}   score(would-be)={:.4}", d.score),
            None => println!("        → PASS   score={:.4}", d.score),
        }
    }
    println!();
}

/// The GA settings both the batch and a re-roll read from the same flags.
fn ga_opts_from(args: &[String]) -> nnfractals::time_ga::TimeGaOpts {
    let mut o = nnfractals::time_ga::TimeGaOpts {
        population: get_flag_or(args, "--pop", 24),
        generations: get_flag_or(args, "--gens", 8),
        finalists: get_flag_or(args, "--finalists", 6),
        max_channels: get_flag_or(args, "--channels", 2),
        seed: get_flag_or(args, "--seed", 0u64),
        ..Default::default()
    };
    o.full.depths = get_flag_or(args, "--depths", o.full.depths);
    o.full.frames = get_flag_or(args, "--frames", o.full.frames);
    o.cheap.frames = get_flag_or(args, "--cheap-frames", o.cheap.frames);
    // The five clip gates, same flag names `time-explore` takes — which is what
    // the viewer's ⏱ "Rejection criteria" controls already send. Without these
    // the GA silently ignored every one of them, so turning a gate off in the
    // GUI changed the sweep and not the search.
    o.clip.min_coherence = get_flag_or(args, "--min-coherence", o.clip.min_coherence);
    o.clip.min_change = get_flag_or(args, "--min-change", o.clip.min_change);
    o.clip.max_noise = get_flag_or(args, "--max-noise", o.clip.max_noise);
    o.clip.max_still_run = get_flag_or(args, "--max-still-run", o.clip.max_still_run);
    o.clip.max_level_jump = get_flag_or(args, "--max-level-jump", o.clip.max_level_jump);
    o
}

/// Stage 1 of the automated pipeline, end to end.
///
/// Deliberately incremental: each reel's record and preview are written the
/// moment they exist, so a batch that runs for hours and is then killed keeps
/// everything it finished. Every stage is timed and every refusal is printed
/// with its reason, because the expected way to use this is to start it, walk
/// away, and read the log afterwards.
fn cmd_auto_reel(pool: &Path, args: &[String]) {
    use nnfractals::video_export::VideoMsg;
    use nnfractals::{auto_reel, time_ga, video_export};

    let config = Config::load(Path::new("config.toml")).expect("config.toml");
    let count: usize = get_flag_or(args, "--count", 10);
    let final_width: u32 = get_flag_or(args, "--final-width", 1920);
    let pw: u32 = get_flag_or(args, "--preview-w", 512);
    let ph: u32 = get_flag_or(args, "--preview-h", 512);
    let pfps: u32 = get_flag_or(args, "--preview-fps", 5);
    let seconds: f32 = get_flag_or(args, "--seconds", 12.0);
    let frames = ((seconds * pfps as f32).round() as u32).max(2);
    let skip_ga = args.iter().any(|a| a == "--no-time");
    let fit_time = args.iter().any(|a| a == "--fit-time");
    // Off by default: a batch runs unattended over many reels for hours, and
    // the full depth-by-depth breakdown is verbose by design (see
    // `print_explain`). `auto-reel --redo` and `time-ga` — the single-shot
    // debugging entry points — print it unconditionally instead.
    let explain = args.iter().any(|a| a == "--explain");

    let batch = get_flag(args, "--out").map(PathBuf::from)
        .unwrap_or_else(|| auto_reel::reels_dir().join(format!("{}", timestamp())));
    std::fs::create_dir_all(&batch).expect("create batch dir");

    // Don't re-plan fractals earlier batches already used: a ranked pool hands
    // back the same top entries every time.
    let seen = auto_reel::already_reeled(&auto_reel::list_batches());
    println!("batch {}  ({} fractals already reeled)", batch.display(), seen.len());

    let rows: Vec<_> = rank_pool(pool, count + seen.len())
        .into_iter()
        .filter(|(p, _, _)| !seen.contains(&p.to_string_lossy().into_owned()))
        .take(count)
        .collect();
    if rows.is_empty() {
        eprintln!("nothing left to reel in {}", pool.display());
        std::process::exit(2);
    }

    let dest_opts = auto_reel::DestinationOpts { final_width, ..Default::default() };
    // The GA is a proxy for the clip that actually ships, so it must know how
    // many frames that clip has — that is what sets the depth past which no
    // representable offset is small enough to animate smoothly.
    let ga_opts = time_ga::TimeGaOpts { clip_frames: frames, ..ga_opts_from(args) };

    println!("planning {} reels from {}: preview {pw}x{ph} @{pfps}fps, {frames} frames",
             rows.len(), pool.display());
    let batch_t0 = std::time::Instant::now();
    let (mut made, mut skipped) = (0usize, 0usize);

    for (path, genome, rank) in &rows {
        let label = path.file_stem().and_then(|s| s.to_str()).unwrap_or("?").to_string();
        println!("\n── {label}  (rank {rank:.3}) ──");

        let t = std::time::Instant::now();
        let fit = match auto_reel::auto_frame(genome, &config) {
            Ok(f) => f,
            Err(why) => { println!("[skip]  {label}  {why}"); skipped += 1; continue; }
        };
        let secs_frame = t.elapsed().as_secs_f32();
        println!("{}  ({secs_frame:.1}s)", auto_reel::frame_report(&label, &fit));

        let t = std::time::Instant::now();
        let dest = match auto_reel::find_destination(genome, &config, &fit.view, &dest_opts) {
            Ok(d) => d,
            Err(why) => { println!("[noaim] {label}  {why}"); skipped += 1; continue; }
        };
        let secs_aim = t.elapsed().as_secs_f32();
        println!("{}  ({secs_aim:.1}s)", auto_reel::destination_report(&label, fit.view.zoom, &dest));

        // "Animate throughout" trades depth for a live third axis across the
        // whole shot, which is the right call when the time axis IS the point.
        // Off by default: most of the value of a reel is the descent.
        //
        // MUST happen before the search: the GA optimises against `dest.end`,
        // and pulling the end back afterwards would hand the renderer a shot the
        // formula was never judged on.
        let mut dest = dest;
        if fit_time {
            let (_, reachable) = time_ga::animatable_span(&fit.view, &dest.end, frames);
            if reachable < dest.end.zoom {
                let span = (dest.end.zoom / fit.view.zoom).log2();
                let t = if span > 0.0 { (reachable / fit.view.zoom).log2() / span } else { 1.0 };
                let before = dest.doublings_travelled(fit.view.zoom);
                dest.end = video_export::CapturedView::from_view(
                    &video_export::lerp_view(&fit.view, &dest.end, t.clamp(0.0, 1.0)));
                println!("  ⏱ --fit-time: end pulled back to {:.2e}x so the formula animates for \
                          the whole shot — {:.1} doublings instead of {:.1}",
                         dest.end.zoom, dest.doublings_travelled(fit.view.zoom), before);
            }
        }

        // Evolve the time formula over the shot that was just chosen.
        let t = std::time::Instant::now();
        let (time_prog, time_score, time_loops, time_summary) = if skip_ga {
            (Vec::new(), 0.0, false, "skipped".to_string())
        } else {
            let pop = time_ga::run(genome, &config, &fit.view, &dest.end, &ga_opts, &print_gen_report);
            let sum = time_ga::summary(&pop);
            // `best_effort` only returns `None` for a genuinely empty
            // population — every reel gets SOME time formula, even an
            // imperfect one, because Stage 2 is a human reviewing every clip
            // anyway and re-roll exists for exactly this case.
            let result = match time_ga::best_effort(&pop, ga_opts.full.depths) {
                Some(best) if best.passed() => {
                    println!("  {}", best.label());
                    (best.progs.clone(), best.score, best.loops(), sum)
                }
                Some(best) => {
                    println!("  no formula passed every gate — shipping the closest one for \
                              review ({}): {}", best.rejected.unwrap_or("?"), best.label());
                    (best.progs.clone(), best.score, best.loops(), sum)
                }
                None => {
                    println!("  no time formula could be evolved at all — rendering as a plain zoom");
                    (Vec::new(), 0.0, false, sum)
                }
            };
            if explain {
                if let Some(best) = time_ga::best_effort(&pop, ga_opts.full.depths) {
                    let views = time_ga::depth_views(&fit.view, &dest.end, ga_opts.full.depths,
                                                      ga_opts.clip_frames);
                    let explains = time_ga::explain_individual(
                        genome, &config, best, &views, &ga_opts, &ga_opts.full);
                    print_explain(best, fit.view.zoom, &explains);
                }
            }
            result
        };
        let secs_evolve = t.elapsed().as_secs_f32();

        let id = format!("{:x}", std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0));
        let nn_file = format!("{id}.nn");
        let mut reel_genome = genome.clone();
        reel_genome.time_prog = time_prog.clone();
        if let Err(e) = save_genome(&reel_genome, &batch.join(&nn_file)) {
            println!("[fail]  {label}  cannot save genome: {e}");
            skipped += 1;
            continue;
        }

        let preview_file = format!("{id}.mp4");
        let out_path = batch.join(&preview_file);
        let t = std::time::Instant::now();
        let (tx, rx) = std::sync::mpsc::channel::<VideoMsg>();
        if time_prog.is_empty() {
            video_export::export_video_chain(
                &reel_genome, &config, false, &[fit.view, dest.end], frames, pfps, pw, ph,
                false, false, &out_path, &tx, &|| {});
        } else {
            video_export::export_chain_time_video(
                &reel_genome, None, &config, false, &[fit.view, dest.end], frames, pfps, pw, ph,
                false, false, nnfractals::formula::ModShape::Sine, 1.0, 0.0, 0.0,
                &out_path, &tx, &|| {});
        }
        drop(tx);
        let failure = rx.iter().find_map(|m| match m {
            VideoMsg::Failed(why) => Some(why),
            _ => None,
        });
        let secs_render = t.elapsed().as_secs_f32();
        if let Some(why) = failure {
            println!("[fail]  {label}  render failed: {why}");
            skipped += 1;
            continue;
        }

        let (animated_fraction, animatable_to) =
            time_ga::animatable_span(&fit.view, &dest.end, frames);
        if animated_fraction < 0.999 {
            println!("  ⏱ animatable over the first {:.0}% of the shot (to {animatable_to:.2e}x) — \
                      past that a {frames}-frame clip has no representable step small enough \
                      to move smoothly",
                     animated_fraction * 100.0);
        }
        let rec = auto_reel::ReelRecord {
            id,
            source: path.to_string_lossy().into_owned(),
            label: label.clone(),
            nn_file,
            preview_file,
            start: fit.view,
            end: dest.end,
            time_prog,
            frame: fit,
            destination: dest,
            time_score,
            time_loops,
            time_summary,
            animated_fraction,
            animatable_to,
            preview_w: pw, preview_h: ph, preview_fps: pfps, preview_frames: frames,
            secs_frame, secs_aim, secs_evolve, secs_render,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
            status: auto_reel::ReelStatus::Pending,
        };
        if let Err(e) = auto_reel::save_record(&batch, &rec) {
            println!("[fail]  {label}  cannot write record: {e}");
            skipped += 1;
            continue;
        }
        made += 1;
        println!("[reel]  {label}  {:.1} doublings, {:.0}s total → {}",
                 rec.doublings(), rec.total_secs(), out_path.display());
    }

    println!("\n{made} reels, {skipped} skipped, {:.1} min total",
             batch_t0.elapsed().as_secs_f32() / 60.0);
    println!("review them with `nnfractals-reels` (batch {})", batch.display());
}

/// Evolve a time formula for one genome, over the shot the pipeline would give
/// it unless start/end are named explicitly.
fn cmd_time_ga(genome: &Genome, label: &str, out_dir: &Path, args: &[String]) {
    use nnfractals::{auto_reel, time_ga};
    let config = Config::load(Path::new("config.toml")).expect("config.toml");
    let final_width: u32 = get_flag_or(args, "--final-width", 1920);

    // Explicit endpoints win; otherwise frame and aim exactly as a reel would.
    let start = match (get_flag(args, "--start-zoom"), get_flag(args, "--cx"), get_flag(args, "--cy")) {
        (Some(z), Some(cx), Some(cy)) => nnfractals::video_export::CapturedView {
            cx: cx.parse().unwrap_or(0.0), cx_lo: 0.0,
            cy: cy.parse().unwrap_or(0.0), cy_lo: 0.0,
            zoom: z.parse().unwrap_or(1.0), aspect: 1.0,
        },
        _ => match auto_reel::auto_frame(genome, &config) {
            Ok(f) => { println!("{}", auto_reel::frame_report(label, &f)); f.view }
            Err(why) => { eprintln!("cannot frame {label}: {why}"); std::process::exit(3); }
        },
    };
    let end = match get_flag(args, "--end-zoom").and_then(|z| z.parse::<f64>().ok()) {
        Some(z) => nnfractals::video_export::CapturedView { zoom: z, ..start },
        None => {
            let opts = auto_reel::DestinationOpts { final_width, ..Default::default() };
            match auto_reel::find_destination(genome, &config, &start, &opts) {
                Ok(d) => { println!("{}", auto_reel::destination_report(label, start.zoom, &d)); d.end }
                Err(why) => { eprintln!("cannot aim {label}: {why}"); std::process::exit(3); }
            }
        }
    };

    let mut opts = time_ga::TimeGaOpts {
        population: get_flag_or(args, "--pop", 24),
        generations: get_flag_or(args, "--gens", 8),
        elites: get_flag_or(args, "--elites", 4),
        max_channels: get_flag_or(args, "--channels", 2),
        finalists: get_flag_or(args, "--finalists", 6),
        fps: get_flag_or(args, "--fps", 24),
        angle_coloring: args.iter().any(|a| a == "--angle-coloring"),
        seed: get_flag_or(args, "--seed", 0u64),
        ..Default::default()
    };
    opts.full.depths = get_flag_or(args, "--depths", opts.full.depths);
    opts.full.frames = get_flag_or(args, "--frames", opts.full.frames);
    opts.cheap.frames = get_flag_or(args, "--cheap-frames", opts.cheap.frames);

    println!("evolving a time formula for {label}: pop {} x {} gens, {} channels, \\
              {} depths over {:.1} doublings",
             opts.population, opts.generations, opts.max_channels, opts.full.depths,
             (end.zoom / start.zoom).log2().max(0.0));

    let t0 = std::time::Instant::now();
    let pop = time_ga::run(genome, &config, &start, &end, &opts, &print_gen_report);
    if pop.is_empty() {
        eprintln!("{label} has nothing animatable — every candidate scalar is absent or unread");
        std::process::exit(3);
    }
    println!("{}  in {:.1}s", time_ga::summary(&pop), t0.elapsed().as_secs_f32());

    if let Some(best) = time_ga::best_effort(&pop, opts.full.depths) {
        let views = time_ga::depth_views(&start, &end, opts.full.depths, opts.clip_frames);
        let explains = time_ga::explain_individual(genome, &config, best, &views, &opts, &opts.full);
        print_explain(best, start.zoom, &explains);
    }

    let keep: usize = get_flag_or(args, "--top-k", 6);
    match time_ga::write_manifest(out_dir, &pop, genome, &start, &end, &opts, keep) {
        Ok(files) => println!("wrote {} winners to {}", files.len(), out_dir.display()),
        Err(e) => eprintln!("cannot write manifest: {e}"),
    }
}

/// Rank a pool by the scores the browser already sorts on.
///
/// Deliberately the same signals the taste model and the GA already produce
/// rather than a new one: `aesthetic_ensemble` is what the save gate selects on,
/// `pref_score` is Carl's own trained preference, and `novelty_score` keeps a
/// batch from being ten variations of one family. A genome missing a score
/// contributes 0 for it rather than being dropped — the archive was scored in
/// waves and older entries genuinely lack some fields.
fn rank_pool(pool: &Path, count: usize) -> Vec<(PathBuf, Genome, f32)> {
    let mut rows: Vec<(PathBuf, Genome, f32)> = std::fs::read_dir(pool)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("nn"))
        .filter_map(|p| io::load_genome(&p).ok().map(|g| (p, g)))
        .map(|(p, g)| {
            let score = g.aesthetic_ensemble / 10.0 + 0.5 * g.pref_score + 0.25 * g.novelty_score;
            (p, g, score)
        })
        .collect();
    rows.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    rows.truncate(count);
    rows
}

/// Plan (but do not render) automated reels, writing the opening and closing
/// frame of each so the framing and aiming rules can be judged by eye.
fn cmd_auto_reel_plan(
    pool: &Path, count: usize, out_dir: &Path, final_width: u32, shot_res: u32, aim: bool,
    fo: nnfractals::auto_reel::FrameOpts,
) {
    use nnfractals::auto_reel;
    let config = Config::load(Path::new("config.toml")).expect("config.toml");
    std::fs::create_dir_all(out_dir).expect("create out dir");

    let rows = rank_pool(pool, count);
    if rows.is_empty() {
        eprintln!("no .nn files in {}", pool.display());
        std::process::exit(2);
    }
    println!("planning {} reels from {} (final width {final_width}px)", rows.len(), pool.display());

    let mut report = String::new();
    let (mut framed, mut aimed) = (0usize, 0usize);
    let dest_opts = auto_reel::DestinationOpts { final_width, ..Default::default() };

    for (path, g, rank) in &rows {
        let label = path.file_stem().and_then(|s| s.to_str()).unwrap_or("?").to_string();
        let t0 = std::time::Instant::now();
        let fit = match auto_reel::auto_frame_with(g, &config, &fo) {
            Ok(f) => f,
            Err(why) => {
                let line = format!("[skip]  {label}  rank {rank:.3}  {why}");
                println!("{line}");
                report.push_str(&line);
                report.push('\n');
                continue;
            }
        };
        framed += 1;
        let line = format!("{}  rank {rank:.3}  ({:.1}s)",
                           auto_reel::frame_report(&label, &fit), t0.elapsed().as_secs_f32());
        println!("{line}");
        report.push_str(&line);
        report.push('\n');
        save_shot(g, &config, &fit.view.to_view(), shot_res, &out_dir.join(format!("{label}_a_start.png")));

        if !aim { continue; }
        let t1 = std::time::Instant::now();
        match auto_reel::find_destination(g, &config, &fit.view, &dest_opts) {
            Ok(d) => {
                aimed += 1;
                let line = format!("{}  ({:.1}s)",
                                   auto_reel::destination_report(&label, fit.view.zoom, &d), t1.elapsed().as_secs_f32());
                println!("{line}");
                report.push_str(&line);
                report.push('\n');
                save_shot(g, &config, &d.end.to_view(), shot_res,
                          &out_dir.join(format!("{label}_b_end.png")));
                // The midpoint is where a straight line most often dies, so it
                // is the frame worth looking at beyond the two endpoints.
                let mid = nnfractals::video_export::lerp_view(&fit.view, &d.end, 0.5);
                save_shot(g, &config, &mid, shot_res, &out_dir.join(format!("{label}_m_mid.png")));
            }
            Err(why) => {
                let line = format!("[noaim] {label}  {why}");
                println!("{line}");
                report.push_str(&line);
                report.push('\n');
            }
        }
    }

    let summary = format!("\n{framed}/{} framed, {aimed}/{framed} aimed\n", rows.len());
    println!("{summary}");
    report.push_str(&summary);
    let _ = std::fs::write(out_dir.join("plan.txt"), &report);
    println!("wrote {}", out_dir.join("plan.txt").display());
}

/// Headless queue processor — what `explorer queue-run` runs, and so what a
/// nightly cron job runs. No GUI, no egui dependency: progress is printed,
/// not routed to a repaint.
///
/// `--until HH:MM` makes this a long-lived loop rather than a single pass:
/// process whatever's Pending, and when the queue runs dry, wait and check
/// again (a fresh Approve from `nnfractals-reels` mid-run picks up without a
/// second cron trigger) — until the deadline, at which point it stops
/// STARTING new items (an item already rendering always finishes, the same
/// rule the interactive window's hold window uses) and exits. Without
/// `--until`, it drains once through whatever is Pending right now and exits
/// — useful for testing, or for a cron line that fires every few minutes
/// instead of once nightly.
fn cmd_queue_run(args: &[String]) {
    use nnfractals::queue_runner::{
        now_minute_of_day, parse_hhmm, process_queue_item, ProcessorLock, QueueProgress,
    };
    use nnfractals::video_export::{load_queue, queue_dir, save_queue, QueueStatus};

    let until_str = get_flag(args, "--until");
    let until = until_str.and_then(|s| parse_hhmm(s));
    // Always leave at least this many cores free even when the system is
    // otherwise idle — never hand out literally every core to a background
    // batch job, whatever else may want to start using the machine.
    let min_free_cores: usize = get_flag_or(args, "--min-free-cores", 1);
    let check_secs: u64 = get_flag_or(args, "--check-interval-secs", 20);

    // A one-shot deadline computed now, not re-parsed as wall-clock HH:MM on
    // every loop iteration: that would need its own midnight-wrap logic, and
    // this project already has one wrap-aware primitive (`HoldWindow`) that
    // the interactive window uses for the SAME setting — reusing it here
    // instead of writing a second, subtly different version of the same rule.
    let deadline = until.map(|until_min| {
        let now_min = now_minute_of_day() as i64;
        let mut delta = until_min as i64 - now_min;
        if delta <= 0 { delta += 24 * 60; }
        std::time::Instant::now() + std::time::Duration::from_secs(delta as u64 * 60)
    });

    match (until_str, until) {
        (Some(s), Some(_)) => println!("queue-run: starting (until {s})"),
        (Some(s), None) => {
            eprintln!("queue-run: --until {s} is not HH:MM — ignoring, running a single pass");
        }
        (None, _) => println!("queue-run: starting (single pass)"),
    }

    // Load-monitor thread: runs for the whole session, not per item, so it
    // keeps adjusting while a single long render is in flight rather than
    // only ever sampling between items.
    let monitor_stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let monitor_handle = {
        let stop = monitor_stop.clone();
        std::thread::spawn(move || {
            let mut current: Option<usize> = None;
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                let want = nnfractals::queue_runner::recommend_thread_count(
                    min_free_cores, std::time::Duration::from_millis(800));
                if current != Some(want) {
                    match current {
                        Some(prev) if want < prev =>
                            println!("queue-run: more other-process load detected — dropping to {want} core(s)"),
                        Some(prev) if want > prev =>
                            println!("queue-run: less other-process load detected — using {want} core(s)"),
                        _ => println!("queue-run: using {want} core(s) for rendering"),
                    }
                    nnfractals::video_export::RENDER_CONTROL.set_threads(want);
                    current = Some(want);
                }
                std::thread::sleep(std::time::Duration::from_secs(check_secs));
            }
        })
    };

    let mut processed = 0usize;
    loop {
        if let Some(d) = deadline {
            if std::time::Instant::now() >= d {
                println!("queue-run: reached the deadline — stopping (finished {processed} item(s))");
                break;
            }
        }
        let Some(_lock) = ProcessorLock::acquire() else {
            // The interactive window (or another queue-run) is already
            // processing something — wait for it rather than racing it.
            std::thread::sleep(std::time::Duration::from_secs(5));
            continue;
        };
        let mut items = load_queue();
        let next_idx = items.iter().enumerate()
            .filter(|(_, it)| it.status == QueueStatus::Pending)
            .min_by_key(|(_, it)| it.created_at)
            .map(|(i, _)| i);
        let Some(idx) = next_idx else {
            if until.is_none() {
                println!("queue-run: nothing Pending — done (finished {processed} item(s))");
                break;
            }
            // Keep polling until the deadline: a reel approved mid-run
            // should still get picked up without a second cron trigger.
            drop(_lock);
            std::thread::sleep(std::time::Duration::from_secs(30));
            continue;
        };

        items[idx].status = QueueStatus::Processing;
        save_queue(&items);
        let item = items[idx].clone();
        println!("queue-run: {} ({})", item.genome_label, item.id);

        let result = process_queue_item(&item, &|p| match p {
            QueueProgress::Pid(pid) => println!("  pid {pid}"),
            QueueProgress::Frame(done, total) => {
                if total == 0 || done % 20 == 0 || done == total {
                    println!("  frame {done}/{total}");
                }
            }
            QueueProgress::Rife(s) => println!("  {s}"),
        });

        let mut items = load_queue();
        if let Some(it) = items.iter_mut().find(|it| it.id == item.id) {
            match &result {
                Ok(out) => { it.status = QueueStatus::Done; it.output_path = Some(out.clone()); it.error = None; }
                Err(e) => { it.status = QueueStatus::Failed; it.error = Some(e.clone()); }
            }
        }
        save_queue(&items);
        let _ = std::fs::remove_file(queue_dir().join(&item.nn_filename));
        match &result {
            Ok(out) => println!("  done -> {out}"),
            Err(e) => println!("  FAILED: {e}"),
        }
        processed += 1;
    }

    monitor_stop.store(true, std::sync::atomic::Ordering::Relaxed);
    let _ = monitor_handle.join();
}

fn cmd_time_explore(
    formula: &str, genome_override: Option<Genome>, cx: f64, cy: f64, zoom: f64, out_dir: &Path,
    opts: time_explore::TimeExploreOpts, keep_clips: bool,
) {
    std::fs::create_dir_all(out_dir).unwrap_or_else(|e| panic!("create {}: {e}", out_dir.display()));
    let genome = genome_override.unwrap_or_else(|| build_genome(formula));
    let config = load_config();
    let view = View::new_square(cx, cy, zoom);

    let targets = time_explore::enumerate_targets(&genome);
    let n_cands = time_explore::candidate_mods(&genome, &opts).len();
    println!(
        "time-explore: {} animatable target(s) on {:016x} -> {n_cands} candidates \
         ({}x{} x {} frames each)",
        targets.len(), genome.id, opts.probe_w, opts.probe_h, opts.frames
    );
    for t in &targets {
        println!("  target: {}", t.label());
    }

    let started = std::time::Instant::now();
    let cands = time_explore::run(&genome, &config, &view, &opts, &|i, total, tmod, cand| {
        let verdict = match cand.rejected {
            Some(why) => format!("rejected: {why}"),
            None => format!("score {:.4}", cand.score),
        };
        println!(
            "  [{i}/{total}] {} {} amp {:.3} -> {verdict}  (worst-coh {:.2}, change {:.1}, noise {:.2}, jump {:.1})",
            tmod.target.label(), tmod.shape.label(), tmod.amp,
            cand.stats.min_coherence, cand.stats.mean_change, cand.stats.max_noise, cand.stats.max_level_jump
        );
    });

    time_explore::write_manifest(out_dir, &cands, &genome, &config, &view, &opts, keep_clips)
        .unwrap_or_else(|e| panic!("write {}/time_winners.jsonl: {e}", out_dir.display()));

    println!(
        "time-explore: {} in {:.1}s -> {}",
        time_explore::summary(&cands), started.elapsed().as_secs_f32(), out_dir.display()
    );
}

fn cmd_video_zoom_explore(
    formula: &str, genome_override: Option<Genome>, cx: f64, cy: f64, zoom: f64, out_dir: &Path,
    depth: usize, finalists: usize, lookahead_plies: usize, method_arg: &str,
    final_width: u32, final_height: u32, canvas_res: u32, top_winners: usize, n_seeds: usize, angle_coloring: bool,
    min_score: f32, min_file_size_ratio: f32, min_file_size_step_ratio: f32, min_step_zoom: f64, min_frame_richness: f32, gate: ZoneGate, lookahead_probe: video_zoom_explore::ProbeSize, final_probe: video_zoom_explore::ProbeSize,
    dd_margin_ulps: f64,
) {
    std::fs::create_dir_all(out_dir).unwrap_or_else(|e| panic!("create {}: {e}", out_dir.display()));
    let genome = genome_override.unwrap_or_else(|| build_genome(formula));
    let config = load_config();
    let base_view = View::new_square(cx, cy, zoom);

    let opts = video_zoom_explore::VideoZoomOpts {
        max_depth: depth, finalists_per_level: finalists, lookahead_plies,
        final_export_width: final_width, final_export_height: final_height, canvas_res, top_winners, min_score, min_file_size_ratio, min_file_size_step_ratio, min_step_zoom, min_frame_richness, gate,
        lookahead_probe, final_probe, dd_margin_ulps,
    };

    // Shares vae_explore's log filename/shape deliberately — both write the
    // same "level_scanning" event, so the viewer's existing scan overlay
    // (polls this file, not a stdout channel) works against a video-zoom
    // run with no viewer-side changes needed for that part.
    let mut log = Logger::append(&out_dir.join("vae_explore_log.jsonl")).unwrap_or_else(|e| panic!("open log: {e}"));
    log.verbose = false;

    let seeds = if n_seeds <= 1 {
        vec![base_view]
    } else {
        pick_seeds(&genome, &config, &base_view, ScoreMethod::GatedEntropy, n_seeds, &mut log, EXPLORE_WIDE_RADIUS, WIDE_SCALES)
    };

    let winners = video_zoom_explore::run(&genome, &config, angle_coloring, &seeds, method_arg, &opts, out_dir, &mut log);
    video_zoom_explore::write_winners_manifest(out_dir, &winners, &genome, &config, angle_coloring)
        .unwrap_or_else(|e| panic!("write {}/video_zoom_winners.jsonl: {e}", out_dir.display()));

    match winners.first() {
        Some(w) => println!(
            "video-zoom-explore: {} winners in {} — best: {:.4} ratio, {} legs, ended={:?}",
            winners.len(), out_dir.display(), w.final_probe_ratio.unwrap_or(0.0), w.chain.len() - 1, w.ended_reason
        ),
        None => println!(
            "video-zoom-explore: 0 winners in {} — the start view may already be past the DD boundary at --final-width {final_width}, or genuinely degenerate everywhere nearby",
            out_dir.display()
        ),
    }
}

// ── Navigation-imitation data prep ──────────────────────────────────────

/// `(u, v, log_zoom_ratio)` — where `after` landed relative to `before`'s
/// own frame, DD-precise. Same parameterization `sweep_positions`/
/// `apply_offset` use internally, so a trained model's output plugs
/// straight back in with no new geometry code. Mirrors
/// `scripts/mine_nav_history.py`'s `label_for_step` (plain-float, since
/// mined data only has 4-decimal-rounded filename coordinates) — this is
/// the DD-precise version, for live `nav_log.jsonl` entries which do carry
/// full precision.
fn nav_label(before: &View, after: &View) -> (f32, f32, f32) {
    let d_cx = after.cx_dd() - before.cx_dd();
    let d_cy = after.cy_dd() - before.cy_dd();
    let half_x = 2.0 / before.zoom * before.aspect;
    let half_y = 2.0 / before.zoom;
    let u = (d_cx.hi / half_x) as f32;
    let v = (d_cy.hi / half_y) as f32;
    let log_zoom_ratio = (after.zoom / before.zoom).ln() as f32;
    (u, v, log_zoom_ratio)
}

fn view_from_json(v: &serde_json::Value) -> Option<View> {
    Some(View {
        cx: v["cx"].as_f64()?, cx_lo: v["cx_lo"].as_f64().unwrap_or(0.0),
        cy: v["cy"].as_f64()?, cy_lo: v["cy_lo"].as_f64().unwrap_or(0.0),
        zoom: v["zoom"].as_f64()?, aspect: v["aspect"].as_f64().unwrap_or(1.0),
    })
}

/// Best-effort `{genome_id}.nn` lookup across every directory a genome
/// might live in — mirrors `scripts/mine_nav_history.py`'s `resolve_nn`
/// search list, kept in sync deliberately (same underlying data).
const NAV_GENOME_SEARCH_DIRS: &[&str] = &[
    "fractals_1", "fractals_2", "fractals_3", "fractals_4", "fractals", "fractals_dag",
    "oldfractals", "Starred", "train_corpus",
];

fn resolve_genome(genome_id: &str) -> Option<Genome> {
    for dir in NAV_GENOME_SEARCH_DIRS {
        let p = Path::new(dir).join(format!("{genome_id}.nn"));
        if p.exists()
            && let Ok(g) = io::load_genome(&p) { return Some(g); }
    }
    None
}

/// The project's canonical "is this visually interesting" metric (same
/// one the GA itself optimizes against — `fitness::png_compression_entropy`)
/// applied to the TARGET (`after`) view of a nav-training example, not the
/// `before` view the model is fed. Added 2026-08-04: Carl reported
/// Auto-Select often landing on low-entropy (boring/flat) zones — this
/// scores whether the TRAINING DATA itself is teaching that, by measuring
/// how visually rich Carl's own past zoom TARGETS actually were. Same
/// resolution/max_iter/colormap as `save_shot`'s renders, so scores are
/// comparable across every record regardless of source (live vs mined,
/// which otherwise have very different native resolutions — 224 renders
/// vs original 4000x4000 saves).
fn target_entropy(genome: &Genome, config: &Config, view: &View, res: u32) -> f32 {
    let use_f64 = needs_f64(view, res);
    let field = render_escape_times(genome, config, view, res, res, config.rendering.max_iter, use_f64, true);
    fitness::png_compression_entropy(&field, res, res, config.rendering.max_iter, &config.rendering.colormap)
}

/// Renders every qualifying `nav_log.jsonl` event's `before` view to a
/// cached PNG (skipping ones already rendered — the log only grows, so a
/// repeat run should only do new work) and writes `nav_manifest.jsonl`:
/// one `{"path", "u", "v", "log_zoom_ratio", "genome_id", "action"}` line
/// per usable event, the same shape `nav_log_mined.jsonl` already is (that
/// file needs no rendering — its `before`/`after` already point at real,
/// existing PNGs from when they were originally saved — so
/// `scripts/train_navigate.py` reads both manifests directly with no
/// special-casing between "live" and "mined" sources).
///
/// Only `drag_zoom`/`zoom_in_btn`/`zoom_in_key` qualify — the well-formed
/// "zoomed into a sub-region of what I was looking at" actions (see
/// [[project-nav-imitation-model]]); pan/zoom-out/undo/reset are logged
/// but aren't valid (before -> after) training targets for this label
/// shape and are skipped here.
fn cmd_prep_nav_data(nav_log_path: &Path, out_dir: &Path, manifest_path: &Path) {
    std::fs::create_dir_all(out_dir).expect("create out_dir");
    let config = load_config();
    let content = std::fs::read_to_string(nav_log_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", nav_log_path.display()));

    const QUALIFYING: &[&str] = &["drag_zoom", "zoom_in_btn", "zoom_in_key"];
    let mut genome_cache: std::collections::HashMap<String, Option<Genome>> = std::collections::HashMap::new();
    let (mut n_rendered, mut n_cached, mut n_missing_genome, mut n_skipped_action) = (0usize, 0usize, 0usize, 0usize);
    let mut manifest = std::fs::File::create(manifest_path).expect("create manifest");

    for (i, line) in content.lines().enumerate() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else { continue };
        if v["event"].as_str() != Some("nav") { continue; }
        let action = v["action"].as_str().unwrap_or("").to_string();
        if !QUALIFYING.contains(&action.as_str()) { n_skipped_action += 1; continue; }
        let genome_id = v["genome_id"].as_str().unwrap_or("").to_string();
        let (Some(before), Some(after)) = (view_from_json(&v["before"]), view_from_json(&v["after"])) else { continue };

        let genome = genome_cache.entry(genome_id.clone())
            .or_insert_with(|| resolve_genome(&genome_id));
        let Some(genome) = genome else { n_missing_genome += 1; continue; };

        let stem = format!("{genome_id}_{i:06}");
        let png_path = out_dir.join(format!("{stem}.png"));
        if png_path.exists() {
            n_cached += 1;
        } else {
            save_shot(genome, &config, &before, 224, &png_path);
            n_rendered += 1;
        }

        let (u, vv, log_zoom_ratio) = nav_label(&before, &after);
        let entropy = target_entropy(genome, &config, &after, 224);
        let rec = serde_json::json!({
            "path": png_path.to_string_lossy(), "u": u, "v": vv, "log_zoom_ratio": log_zoom_ratio,
            "genome_id": genome_id, "action": action, "source": "live", "target_entropy": entropy,
        });
        use std::io::Write;
        writeln!(manifest, "{rec}").expect("write manifest");
    }
    println!(
        "rendered={n_rendered} cached={n_cached} missing_genome={n_missing_genome} skipped_action={n_skipped_action} -> {}",
        manifest_path.display()
    );
}

/// Adds `target_entropy` to every record in `nav_log_mined.jsonl` in
/// place — same metric, same resolution as `cmd_prep_nav_data`'s live
/// path, so the two sources land on one comparable scale (mined records
/// already carry full `before`/`after` view + `nn_path`, per
/// `mine_nav_history.py`'s schema, so no rendering-cache bookkeeping is
/// needed here — just read, score, rewrite). Overwrites the file: this is
/// a derived artifact `mine_nav_history.py` regenerates from scratch
/// anyway, not hand-edited data.
///
/// A record whose genome can't be resolved (its `.nn` moved/deleted since
/// mining — confirmed to happen, 14/94 on the real archive) is written
/// back UNCHANGED, never dropped: `target_entropy` is a NEW, optional
/// enrichment, but `before.path`/`label` alone are everything the actual
/// training scripts need (genome-agnostic, they just load the image) — an
/// earlier version of this function `continue`d past unresolvable-genome
/// records instead of re-emitting them, which silently deleted 14 good,
/// already-usable training examples from the file. Records missing
/// `target_entropy` are treated as "unscored, don't filter" downstream.
fn cmd_score_mined_targets(mined_path: &Path) {
    let config = load_config();
    let content = std::fs::read_to_string(mined_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", mined_path.display()));

    let mut genome_cache: std::collections::HashMap<String, Option<Genome>> = std::collections::HashMap::new();
    let (mut n_scored, mut n_missing_genome, mut n_bad_record) = (0usize, 0usize, 0usize);
    let mut out_lines: Vec<String> = Vec::new();

    for line in content.lines() {
        let Ok(mut v) = serde_json::from_str::<serde_json::Value>(line) else {
            n_bad_record += 1;
            if !line.trim().is_empty() { out_lines.push(line.to_string()); }
            continue;
        };
        let genome_id = v["genome_id"].as_str().unwrap_or("").to_string();

        let scored = (|| {
            let after = view_from_json(&v["after"])?;
            let genome = genome_cache.entry(genome_id.clone()).or_insert_with(|| {
                v["nn_path"].as_str()
                    .and_then(|p| io::load_genome(Path::new(p)).ok())
                    .or_else(|| resolve_genome(&genome_id))
            }).as_ref()?;
            Some(target_entropy(genome, &config, &after, 224))
        })();

        match scored {
            Some(entropy) => { v["target_entropy"] = serde_json::json!(entropy); n_scored += 1; }
            None => n_missing_genome += 1,
        }
        out_lines.push(v.to_string());
    }

    std::fs::write(mined_path, out_lines.join("\n") + "\n").expect("rewrite mined manifest");
    println!("scored={n_scored} missing_genome={n_missing_genome} bad_record={n_bad_record} (all still written) -> {}", mined_path.display());
}

fn main() {
    render_gpu::init_gpu();
    if !render_gpu::gpu_available() {
        eprintln!("warning: no GPU adapter found — falling back to per-candidate CPU rendering (render_batch_dag's own fallback), much slower than intended.");
    }
    let args: Vec<String> = std::env::args().skip(1).collect();
    // Positional-only view: every subcommand below mixes positional args
    // with `--flag value` pairs, and plain `args.get(N)` doesn't know to
    // stop at the first flag — omit a trailing positional and jump
    // straight to a flag, and the flag's OWN VALUE token silently slides
    // into that positional slot (real bug hit in production: `vae-explore
    // "Celtic Mandelbrot" --iterations 10` parsed cy=10.0 from the "10",
    // not the intended default 0.0 — corrupted an entire 10-iteration
    // run). `pos` truncates at the first `--`-prefixed token so a missing
    // positional falls through to its default instead; `get_flag`/
    // `get_flag_or` still search the full, untruncated `args`.
    let flag_boundary = args.iter().position(|a| a.starts_with("--")).unwrap_or(args.len());
    let pos = &args[..flag_boundary];
    match args.first().map(String::as_str) {
        Some("compare") => {
            let out_dir = pos.get(1).map(PathBuf::from).unwrap_or_else(|| PathBuf::from(format!("explorer_out/compare_{}", timestamp())));
            cmd_compare(&out_dir);
        }
        Some("run") => {
            let method = pos.get(1).and_then(|s| ScoreMethod::parse(s)).unwrap_or_else(|| panic!("method must be one of entropy|edge|gated-entropy|gated-edge"));
            let n_seeds: usize = pos.get(2).and_then(|s| s.parse().ok()).unwrap_or(6);
            let max_rounds: usize = pos.get(3).and_then(|s| s.parse().ok()).unwrap_or(6);
            let out_dir = pos.get(4).map(PathBuf::from).unwrap_or_else(|| PathBuf::from(format!("explorer_out/mandelbrot_{}", timestamp())));
            cmd_run(method, n_seeds, max_rounds, &out_dir);
        }
        Some("pool") => {
            let formula = pos.get(1).cloned().unwrap_or_else(|| "Mandelbrot".to_string());
            let methods: Vec<ScoreMethod> = match pos.get(2).map(String::as_str) {
                Some("mixed") => ScoreMethod::ALL.to_vec(),
                Some(s) => s.split(',').map(|m| ScoreMethod::parse(m).unwrap_or_else(|| panic!("method must be one of entropy|edge|gated-entropy|gated-edge|mixed, or a comma-separated list"))).collect(),
                None => ScoreMethod::ALL.to_vec(),
            };
            let cx: f64 = pos.get(3).and_then(|s| s.parse().ok()).unwrap_or(-0.5);
            let cy: f64 = pos.get(4).and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let zoom: f64 = pos.get(5).and_then(|s| s.parse().ok()).unwrap_or(1.0);
            let n_seeds: usize = pos.get(6).and_then(|s| s.parse().ok()).unwrap_or(100);
            let max_rounds: usize = pos.get(7).and_then(|s| s.parse().ok()).unwrap_or(6);
            let min_score: f32 = pos.get(8).and_then(|s| s.parse().ok()).unwrap_or(0.3);
            let max_intricacy: f32 = pos.get(9).and_then(|s| s.parse().ok()).unwrap_or(0.30);
            let min_aesthetic: f32 = pos.get(10).and_then(|s| s.parse().ok()).unwrap_or(3.5);
            let min_edge_density: f32 = pos.get(11).and_then(|s| s.parse().ok()).unwrap_or(0.15);
            let out_dir = pos.get(12).map(PathBuf::from).unwrap_or_else(|| PathBuf::from(format!("explorer_out/{}_pool", formula.to_lowercase().replace(' ', "_"))));
            cmd_pool(&formula, &methods, cx, cy, zoom, n_seeds, max_rounds, min_score, max_intricacy, min_aesthetic, min_edge_density, &out_dir);
        }
        Some("gems") => {
            // "mixed" cycles all 4 methods round-robin by tile (see cmd_gems'
            // doc comment on why one fixed method converges on one visual
            // family); otherwise a single name, or a comma-separated list.
            let methods: Vec<ScoreMethod> = match pos.get(1).map(String::as_str) {
                Some("mixed") => ScoreMethod::ALL.to_vec(),
                Some(s) => s.split(',').map(|m| ScoreMethod::parse(m).unwrap_or_else(|| panic!("method must be one of entropy|edge|gated-entropy|gated-edge|mixed, or a comma-separated list"))).collect(),
                None => panic!("method must be one of entropy|edge|gated-entropy|gated-edge|mixed, or a comma-separated list"),
            };
            let hours: f64 = pos.get(2).and_then(|s| s.parse().ok()).unwrap_or(24.0);
            let n_cols: usize = pos.get(3).and_then(|s| s.parse().ok()).unwrap_or(150);
            let n_rows: usize = pos.get(4).and_then(|s| s.parse().ok()).unwrap_or(65);
            let out_dir = pos.get(5).map(PathBuf::from).unwrap_or_else(|| PathBuf::from("explorer_out/mandelbrot_gems"));
            cmd_gems(&methods, hours, n_cols, n_rows, &out_dir);
        }
        Some("curate") => {
            let archive_path = pos.get(1).map(PathBuf::from).unwrap_or_else(|| PathBuf::from("explorer_out/mandelbrot_gems/gems_archive.jsonl"));
            let top_n: usize = pos.get(2).and_then(|s| s.parse().ok()).unwrap_or(30);
            let min_score: f32 = pos.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.35);
            let min_aesthetic: f32 = pos.get(4).and_then(|s| s.parse().ok()).unwrap_or(4.8);
            // L2 distance between two L2-normalized 128-d latent embeddings
            // (range [0,2]) — NOT the same scale as the old pixel-pooling
            // fingerprint distance ([0,~1], typically 0.3-ish). 0.9 is a
            // starting point, not yet calibrated against a real
            // distribution the way the old 0.3 was — check actual
            // pairwise distances on a real run before trusting this default.
            let min_dist: f32 = pos.get(5).and_then(|s| s.parse().ok()).unwrap_or(0.9);
            let res: u32 = pos.get(6).and_then(|s| s.parse().ok()).unwrap_or(4000);
            let formula = pos.get(7).cloned().unwrap_or_else(|| "Mandelbrot".to_string());
            let out_dir = pos.get(8).map(PathBuf::from).unwrap_or_else(|| PathBuf::from("explorer_out/mandelbrot_gems/curated"));
            // Both optional, both-or-neither: a formula-specific model
            // trained by scripts/train_novelty.py (e.g. against a cmd_pool
            // output dir) instead of the production novelty_model.npz/
            // novelty_head.pt every other caller (live GA scoring, the
            // viewer's Explore feature) relies on.
            let model_path = pos.get(9).map(PathBuf::from);
            let head_path = pos.get(10).map(PathBuf::from);
            let model = model_path.as_deref().zip(head_path.as_deref());
            cmd_curate(&archive_path, top_n, min_score, min_aesthetic, min_dist, res, &formula, &out_dir, model);
        }
        Some("prep-nav-data") => {
            let nav_log_path = pos.get(1).map(PathBuf::from).unwrap_or_else(|| PathBuf::from("nav_log.jsonl"));
            let out_dir = pos.get(2).map(PathBuf::from).unwrap_or_else(|| PathBuf::from("nav_train_cache"));
            let manifest_path = pos.get(3).map(PathBuf::from).unwrap_or_else(|| PathBuf::from("nav_manifest.jsonl"));
            cmd_prep_nav_data(&nav_log_path, &out_dir, &manifest_path);
        }
        Some("score-mined-targets") => {
            let mined_path = pos.get(1).map(PathBuf::from).unwrap_or_else(|| PathBuf::from("nav_log_mined.jsonl"));
            cmd_score_mined_targets(&mined_path);
        }
        Some("vae-explore") => {
            let formula = pos.get(1).cloned().unwrap_or_else(|| "Mandelbrot".to_string());
            // formula doubles as a genome-file path: if it names an
            // existing .nn file, load that genome directly instead of
            // looking it up in known_formulas::LIBRARY — lets vae-explore
            // target an arbitrary GA-discovered genome, not just the
            // textbook formulas. Its OWN saved view_cx/view_cy/view_zoom
            // becomes the default reference point (already a curated,
            // presumably-good view — the genome was rendered/rated from
            // it), sidestepping the "which coordinate is even good for
            // this genome" problem entirely rather than guessing.
            let formula_path = Path::new(&formula);
            let genome_override: Option<Genome> = (formula_path.extension().and_then(|e| e.to_str()) == Some("nn"))
                .then(|| io::load_genome(formula_path).ok())
                .flatten();
            let (default_cx, default_cy, default_zoom) = match &genome_override {
                Some(g) => (g.view_cx as f64, g.view_cy as f64, g.view_zoom as f64),
                None => (-0.5, 0.0, 1.0),
            };
            let cx: f64 = pos.get(2).and_then(|s| s.parse().ok()).unwrap_or(default_cx);
            let cy: f64 = pos.get(3).and_then(|s| s.parse().ok()).unwrap_or(default_cy);
            let zoom: f64 = pos.get(4).and_then(|s| s.parse().ok()).unwrap_or(default_zoom);
            let default_out_name = match &genome_override {
                Some(_) => formula_path.file_stem().and_then(|s| s.to_str()).unwrap_or("genome").to_string(),
                None => formula.to_lowercase().replace(' ', "_"),
            };
            let out_dir = pos.get(5).map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from(format!("explorer_out/{default_out_name}_vae")));
            let iterations: usize = get_flag_or(&args, "--iterations", 5);
            let n_seeds: usize = get_flag_or(&args, "--n-seeds", 6);
            let recursion_depth: usize = get_flag_or(&args, "--recursion-depth", 4);
            let top_k: usize = get_flag_or(&args, "--top-k", 6);
            let canvas_res: u32 = get_flag_or(&args, "--canvas-res", 1024);
            let method_arg: String = get_flag(&args, "--method").unwrap_or("mixed").to_string();
            let select_by = parse_select_by(get_flag(&args, "--select-by").unwrap_or("max-error"));
            let max_intricacy: f32 = get_flag_or(&args, "--max-intricacy", 0.30);
            // NOT cmd_pool's 0.15 — that's calibrated against edge_density
            // computed on a FRESH render at the candidate's own zoom
            // (SWEEP_RES=64). coarse_scan's metrics come from a strided
            // pixel-crop of the shallower canvas instead (see coarse_scan's
            // doc comment for why that's the right tradeoff for this
            // stage), which measurably reads lower: a real Mandelbrot
            // canvas's best coarse candidates clustered at edge_density
            // 0.08-0.12, never reaching 0.15 — 0.15 rejected every single
            // candidate. 0.05 leaves real headroom below that observed
            // floor while still rejecting genuinely flat crops.
            let min_edge_density: f32 = get_flag_or(&args, "--min-edge-density", 0.05);
            let mut arch: String = get_flag(&args, "--arch").unwrap_or("conv").to_string();
            let mut latent_dim: usize = get_flag_or(&args, "--latent-dim", 256);
            let mut kl_weight: f64 = get_flag_or(&args, "--kl-weight", 1e-3);
            // Optional: load arch/latent_dim/kl_weight from a
            // scripts/tune_autoencoder.py study result instead of the
            // flags/defaults above — "the ideal VAE structure is shared"
            // (Carl's own framing): one study's winning config, reused
            // across formulas, not searched per-run. Explicit --arch/
            // --latent-dim/--kl-weight still win if BOTH are given
            // (checked in this order, tuned-config first, so a caller can
            // start from a tuned baseline and override just one field).
            if let Some(path) = get_flag(&args, "--tuned-config") {
                let tuned = load_tuned_config(Path::new(path));
                arch = tuned.arch.unwrap_or(arch);
                latent_dim = tuned.latent_dim.unwrap_or(latent_dim);
                kl_weight = tuned.kl_weight.unwrap_or(kl_weight);
                println!("loaded tuned config from {path}: arch={arch} latent_dim={latent_dim} kl_weight={kl_weight}");
            }
            let epochs: usize = get_flag_or(&args, "--epochs", 15);
            // No default target: None means "rely on the patience-based
            // plateau stop only" (see cmd_vae_explore) rather than an
            // absolute floor that may not generalize across formulas.
            let target_recon_mse: Option<f32> = get_flag(&args, "--target-recon-mse").and_then(|s| s.parse().ok());
            let min_improvement: f32 = get_flag_or(&args, "--min-improvement", 0.02);
            let patience: usize = get_flag_or(&args, "--patience", 4);
            // A saliency-net checkpoint (scripts/train_saliency.py) that
            // augments coarse_scan's grid with predicted-heatmap candidates
            // each level (see recursion_level's doc comment). Defaults to
            // SALIENCY_DEFAULT_MODEL_PATH (Carl's request, 2026-08-10: "use
            // the saliency model by default") — cmd_vae_explore still
            // checks the file actually exists before enabling anything, so
            // a fresh checkout with no trained model behaves identically to
            // before this default existed. `--saliency-model none` (or any
            // nonexistent path) opts back out.
            let saliency_model_path: Option<PathBuf> = Some(PathBuf::from(
                get_flag(&args, "--saliency-model").unwrap_or(vae_explore::SALIENCY_DEFAULT_MODEL_PATH)
            ));
            cmd_vae_explore(
                &formula, genome_override, cx, cy, zoom, &out_dir, iterations, n_seeds, recursion_depth, top_k, canvas_res,
                &method_arg, select_by, ZoneGate { max_intricacy, min_edge_density },
                &arch, latent_dim, kl_weight, epochs,
                target_recon_mse, min_improvement, patience,
                saliency_model_path,
            );
        }
        Some("video-zoom-explore") => {
            // Same genome-path-vs-formula-name override / default-view
            // convention as "vae-explore" above.
            let formula = pos.get(1).cloned().unwrap_or_else(|| "Mandelbrot".to_string());
            let formula_path = Path::new(&formula);
            let genome_override: Option<Genome> = (formula_path.extension().and_then(|e| e.to_str()) == Some("nn"))
                .then(|| io::load_genome(formula_path).ok())
                .flatten();
            let (default_cx, default_cy, default_zoom) = match &genome_override {
                Some(g) => (g.view_cx as f64, g.view_cy as f64, g.view_zoom as f64),
                None => (-0.5, 0.0, 1.0),
            };
            let cx: f64 = pos.get(2).and_then(|s| s.parse().ok()).unwrap_or(default_cx);
            let cy: f64 = pos.get(3).and_then(|s| s.parse().ok()).unwrap_or(default_cy);
            let zoom: f64 = pos.get(4).and_then(|s| s.parse().ok()).unwrap_or(default_zoom);
            let default_out_name = match &genome_override {
                Some(_) => formula_path.file_stem().and_then(|s| s.to_str()).unwrap_or("genome").to_string(),
                None => formula.to_lowercase().replace(' ', "_"),
            };
            let out_dir = pos.get(5).map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from(format!("explorer_out/{default_out_name}_video_zoom")));

            // 30, not 5: measured across 12 real runs, every deep chain ended
            // `DepthReached` at the cap while still 179x short of the f64
            // precision wall. See `VideoZoomOpts::max_depth`.
            let depth: usize = get_flag_or(&args, "--depth", 30);
            let finalists: usize = get_flag_or(&args, "--finalists", 3);
            let lookahead_plies: usize = get_flag_or(&args, "--lookahead-plies", 2);
            let method_arg: String = get_flag(&args, "--method").unwrap_or("mixed").to_string();
            let final_width: u32 = get_flag_or(&args, "--final-width", 1280);
            let final_height: u32 = get_flag_or(&args, "--final-height", 720);
            let canvas_res: u32 = get_flag_or(&args, "--canvas-res", 1024);
            let top_winners: usize = get_flag_or(&args, "--top-winners", 10);
            let n_seeds: usize = get_flag_or(&args, "--n-seeds", 1);
            let angle_coloring: bool = args.iter().any(|a| a == "--angle-coloring");
            // Absolute floor on a candidate's raw score — see
            // `VideoZoomOpts::min_score`'s doc comment for why this exists
            // (without it, a uniformly-bad neighborhood never registers as
            // a dead end, so the search just keeps drilling deeper into it
            // instead of backtracking). 0.15 is a provisional starting
            // point, not a precise calibration — tune down if real, valid
            // zones are getting rejected, up if a run is still ending up in
            // near-flat territory.
            let min_score: f32 = get_flag_or(&args, "--min-score", 0.15);
            // Fraction of the SEED view's own file-size entropy that a
            // candidate must reach to be descended into — see
            // `VideoZoomOpts::min_file_size_ratio`. Lower it if runs
            // dead-end too early; raise it toward 1.0 to demand the zoom
            // stay as rich as it started.
            let min_file_size_ratio: f32 = get_flag_or(&args, "--min-file-size-ratio", 0.45);
            let min_file_size_step_ratio: f32 = get_flag_or(&args, "--min-file-size-step-ratio", 0.80);
            let min_step_zoom: f64 = get_flag_or(&args, "--min-step-zoom", 2.0);
            let min_frame_richness: f32 = get_flag_or(&args, "--min-frame-richness", 0.30);
            // How close to the f64 floor a chain may zoom, in ULPs of pixel
            // step. 1.0 = run until f64 visibly pixelates (the default, and
            // what Carl asked for); 4.0 = the viewer's conservative margin,
            // which stops while output is still perfectly smooth but gives
            // up 4x the zoom for no benefit here, since video export never
            // escalates to DD anyway.
            let dd_margin_ulps: f64 = get_flag_or(&args, "--dd-margin-ulps",
                nnfractals::video_export::DD_MARGIN_ULPS_PIXELATE);
            // Independent structural floor — same flags/defaults as
            // `vae-explore`'s own gate (`ZoneGate`), needed because a
            // method-specific floor alone can't catch every degenerate
            // case: see `VideoZoomOpts::gate`'s doc comment for the real,
            // measured failure mode (entropy plateaus near 0.2 for a
            // collapsed-to-2-histogram-bins crop regardless of whether any
            // real structure survives; edge_density/intricacy don't share
            // that blind spot).
            let max_intricacy: f32 = get_flag_or(&args, "--max-intricacy", 0.30);
            let min_edge_density: f32 = get_flag_or(&args, "--min-edge-density", 0.05);
            let gate = ZoneGate { max_intricacy, min_edge_density };
            let lookahead_probe = video_zoom_explore::ProbeSize {
                w: get_flag_or(&args, "--lookahead-probe-w", 128),
                h: get_flag_or(&args, "--lookahead-probe-h", 96),
                steps: get_flag_or(&args, "--lookahead-probe-steps", 12),
                fps: get_flag_or(&args, "--lookahead-probe-fps", 24),
            };
            let final_probe = video_zoom_explore::ProbeSize {
                w: get_flag_or(&args, "--final-probe-w", 320),
                h: get_flag_or(&args, "--final-probe-h", 240),
                steps: get_flag_or(&args, "--final-probe-steps", 48),
                fps: get_flag_or(&args, "--final-probe-fps", 24),
            };
            cmd_video_zoom_explore(
                &formula, genome_override, cx, cy, zoom, &out_dir,
                depth, finalists, lookahead_plies, &method_arg, final_width, final_height, canvas_res,
                top_winners, n_seeds, angle_coloring, min_score, min_file_size_ratio,
                min_file_size_step_ratio, min_step_zoom, min_frame_richness, gate, lookahead_probe, final_probe,
                dd_margin_ulps,
            );
        }
        Some("time-explore") => {
            // Same genome-path-vs-formula-name convention as the other explore
            // subcommands. A .nn path is the normal case here: the sweep needs a
            // real genome's dynamics (julia mode, phoenix, warp, CONST nodes) to
            // have anything to animate.
            let formula = pos.get(1).cloned().unwrap_or_else(|| "Mandelbrot".to_string());
            let formula_path = Path::new(&formula);
            let genome_override: Option<Genome> = (formula_path.extension().and_then(|e| e.to_str()) == Some("nn"))
                .then(|| io::load_genome(formula_path).ok())
                .flatten();
            let (default_cx, default_cy, default_zoom) = match &genome_override {
                Some(g) => (g.view_cx as f64, g.view_cy as f64, g.view_zoom as f64),
                None => (-0.5, 0.0, 1.0),
            };
            let cx: f64 = pos.get(2).and_then(|s| s.parse().ok()).unwrap_or(default_cx);
            let cy: f64 = pos.get(3).and_then(|s| s.parse().ok()).unwrap_or(default_cy);
            let zoom: f64 = pos.get(4).and_then(|s| s.parse().ok()).unwrap_or(default_zoom);
            let default_out_name = match &genome_override {
                Some(_) => formula_path.file_stem().and_then(|s| s.to_str()).unwrap_or("genome").to_string(),
                None => formula.to_lowercase().replace(' ', "_"),
            };
            // `--out` as well as the 5th positional: with cx/cy/zoom omitted a
            // trailing path lands in the cx slot, fails to parse as a number and
            // is silently discarded — the same positional/flag trap this CLI has
            // been bitten by before. The flag makes the intent unambiguous.
            // A path that ENDS in .nn but did not load is a missing or corrupt
            // file, not a formula name. Falling through to build_genome reports
            // it as `unknown formula "fractals_1/....nn"` alongside a list of
            // built-in formulas, which sends you looking in entirely the wrong
            // place — and genomes really do vanish from a pool while you work,
            // since dedup prunes it every two hours.
            if genome_override.is_none()
                && formula_path.extension().and_then(|e| e.to_str()) == Some("nn")
            {
                eprintln!("cannot load genome {}: no such file, or it failed to parse",
                          formula_path.display());
                std::process::exit(2);
            }
            let out_dir = get_flag(&args, "--out").map(PathBuf::from)
                .or_else(|| pos.get(5).map(PathBuf::from))
                .unwrap_or_else(|| PathBuf::from(format!("explorer_out/{default_out_name}_time")));
            // A positional that should be a number but looks like a path is
            // almost certainly a misplaced out_dir; say so rather than silently
            // using the default.
            for (slot, name) in [(2usize, "cx"), (3, "cy"), (4, "zoom")] {
                if let Some(v) = pos.get(slot) {
                    if v.parse::<f64>().is_err() {
                        eprintln!(
                            "warning: positional #{slot} ({name}) is \"{v}\", which is not a number \
                             — it was ignored. Did you mean --out {v}?"
                        );
                    }
                }
            }

            let amps: Vec<f32> = get_flag(&args, "--amps")
                .map(|s| s.split(',').filter_map(|v| v.trim().parse().ok()).collect())
                .filter(|v: &Vec<f32>| !v.is_empty())
                .unwrap_or_else(|| vec![0.02, 0.08, 0.25]);
            let shapes: Vec<ModShape> = get_flag(&args, "--shapes")
                .map(|s| s.split(',').filter_map(ModShape::parse).collect())
                .filter(|v: &Vec<ModShape>| !v.is_empty())
                .unwrap_or_else(|| vec![ModShape::Sine, ModShape::Triangle, ModShape::Orbit]);

            let opts = time_explore::TimeExploreOpts {
                probe_w: get_flag_or(&args, "--probe-w", 192),
                probe_h: get_flag_or(&args, "--probe-h", 144),
                frames: get_flag_or(&args, "--frames", 48),
                fps: get_flag_or(&args, "--fps", 24),
                amps,
                shapes,
                top_k: get_flag_or(&args, "--top-k", 8),
                angle_coloring: args.iter().any(|a| a == "--angle-coloring"),
                // Raise if winners still read as cuts rather than morphs; lower
                // if a genome that visibly animates well keeps getting rejected
                // as "incoherent".
                min_coherence: get_flag_or(&args, "--min-coherence",
                    time_explore::MIN_TEMPORAL_COHERENCE),
                min_change: get_flag_or(&args, "--min-change",
                    time_explore::MIN_TEMPORAL_CHANGE),
                max_noise: get_flag_or(&args, "--max-noise", time_explore::MAX_CLIP_NOISE),
                max_still_run: get_flag_or(&args, "--max-still-run", time_explore::MAX_STILL_RUN),
                max_level_jump: get_flag_or(&args, "--max-level-jump", time_explore::MAX_LEVEL_JUMP),
            };
            let keep_clips = args.iter().any(|a| a == "--keep-clips");
            cmd_time_explore(&formula, genome_override, cx, cy, zoom, &out_dir, opts, keep_clips);
        }
        Some("time-ga") => {
            // Evolve a time FORMULA (crate::time_program) for one genome along
            // the shot it will actually travel. Start/end default to the
            // pipeline's own choices, so the search optimises the same zoom the
            // reel will use rather than an arbitrary view.
            let formula = pos.get(1).cloned().unwrap_or_else(|| "Mandelbrot".to_string());
            let formula_path = Path::new(&formula);
            if formula_path.extension().and_then(|e| e.to_str()) != Some("nn") {
                eprintln!("time-ga needs a .nn genome path — a time formula has nothing to \
                           animate without one");
                std::process::exit(2);
            }
            let Ok(genome) = io::load_genome(formula_path) else {
                eprintln!("cannot load genome {}: no such file, or it failed to parse",
                          formula_path.display());
                std::process::exit(2);
            };
            let label = formula_path.file_stem().and_then(|s| s.to_str()).unwrap_or("genome").to_string();
            let out_dir = get_flag(&args, "--out").map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from(format!("explorer_out/{label}_time_ga")));
            cmd_time_ga(&genome, &label, &out_dir, &args);
        }
        Some("auto-reel") => {
            // The whole of stage 1: pick, frame, aim, evolve a time formula,
            // render a low-resolution preview. Unattended and incremental — a
            // batch killed partway keeps every reel it finished.
            if let Some(rec) = get_flag(&args, "--redo") {
                cmd_auto_reel_redo(Path::new(rec), &args);
                return;
            }
            let pool = get_flag(&args, "--pool").map(PathBuf::from)
                .or_else(|| pos.get(1).map(PathBuf::from))
                .unwrap_or_else(|| PathBuf::from("fractals_1"));
            cmd_auto_reel(&pool, &args);
        }
        Some("auto-reel-plan") => {
            // Stages 1-3 of the automated pipeline (pick, frame, aim) WITHOUT
            // rendering a reel: writes the opening and closing frame of each
            // shot as PNGs plus a report line, so the framing and zoom rules can
            // be checked by looking at real output rather than by argument.
            // This is how `auto_frame`'s thresholds get calibrated — the same
            // contact-sheet method that caught all three time-gate
            // miscalibrations.
            let pool = get_flag(&args, "--pool").map(PathBuf::from)
                .or_else(|| pos.get(1).map(PathBuf::from))
                .unwrap_or_else(|| PathBuf::from("fractals_1"));
            let count: usize = get_flag_or(&args, "--count", 10);
            let out_dir = get_flag(&args, "--out").map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from(format!("explorer_out/auto_reel_{}", timestamp())));
            let final_width: u32 = get_flag_or(&args, "--final-width", 1920);
            let shot_res: u32 = get_flag_or(&args, "--shot-res", 384);
            let aim = !args.iter().any(|a| a == "--frame-only");
            let fo = nnfractals::auto_reel::FrameOpts {
                body_escape_fraction: get_flag_or(&args, "--body-frac",
                    nnfractals::auto_reel::BODY_ESCAPE_FRACTION),
                fill: get_flag_or(&args, "--fill", nnfractals::auto_reel::FRAME_FILL),
                res: get_flag_or(&args, "--frame-res", nnfractals::auto_reel::FRAME_RES),
            };
            cmd_auto_reel_plan(&pool, count, &out_dir, final_width, shot_res, aim, fo);
        }
        Some("queue-run") => {
            // Headless video-queue processor: same render path the
            // interactive `nnfractals-queue` window uses
            // (`queue_runner::process_queue_item`), with no GUI at all — what
            // a cron job fires at night. `--until HH:MM` keeps it processing
            // Pending items in a loop until that wall-clock time, so one cron
            // trigger covers the whole night rather than needing one per
            // item; omit it to drain whatever is Pending right now and exit.
            cmd_queue_run(&args);
        }
        Some("blend-explore") => {
            // Search the pool for a fractal whose FORMULA morphs well into this
            // one's. Same gates and the same compression score as
            // `time-explore`; the axis is a second genome instead of a scalar.
            let formula = pos.get(1).cloned().unwrap_or_default();
            let formula_path = Path::new(&formula);
            let Some(genome) = io::load_genome(formula_path).ok() else {
                eprintln!("cannot load genome {}: blend-explore needs a .nn file", formula_path.display());
                std::process::exit(2);
            };
            let cx: f64 = pos.get(2).and_then(|s| s.parse().ok()).unwrap_or(genome.view_cx as f64);
            let cy: f64 = pos.get(3).and_then(|s| s.parse().ok()).unwrap_or(genome.view_cy as f64);
            let zoom: f64 = pos.get(4).and_then(|s| s.parse().ok()).unwrap_or(genome.view_zoom as f64);
            let stem = formula_path.file_stem().and_then(|s| s.to_str()).unwrap_or("genome");
            let out_dir = get_flag(&args, "--out").map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from(format!("explorer_out/{stem}_blend")));
            let pool = get_flag(&args, "--pool")
                .map(PathBuf::from)
                .unwrap_or_else(|| formula_path.parent().unwrap_or(Path::new(".")).to_path_buf());
            let samples: usize = get_flag_or(&args, "--samples", 40);
            let seed: u64 = get_flag_or(&args, "--seed", 0);

            let opts = time_explore::TimeExploreOpts {
                probe_w: get_flag_or(&args, "--probe-w", 160),
                probe_h: get_flag_or(&args, "--probe-h", 120),
                frames: get_flag_or(&args, "--frames", 24),
                fps: get_flag_or(&args, "--fps", 24),
                top_k: get_flag_or(&args, "--top-k", 8),
                angle_coloring: args.iter().any(|a| a == "--angle-coloring"),
                min_coherence: get_flag_or(&args, "--min-coherence", time_explore::MIN_TEMPORAL_COHERENCE),
                min_change: get_flag_or(&args, "--min-change", time_explore::MIN_TEMPORAL_CHANGE),
                max_noise: get_flag_or(&args, "--max-noise", time_explore::MAX_CLIP_NOISE),
                max_still_run: get_flag_or(&args, "--max-still-run", time_explore::MAX_STILL_RUN),
                max_level_jump: get_flag_or(&args, "--max-level-jump", time_explore::MAX_LEVEL_JUMP),
                ..Default::default()
            };
            let keep_clips = args.iter().any(|a| a == "--keep-clips");
            let seed = if seed == 0 {
                std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs()).unwrap_or(1)
            } else { seed };

            std::fs::create_dir_all(&out_dir).unwrap_or_else(|e| panic!("create {}: {e}", out_dir.display()));
            let config = load_config();
            let view = View::new_square(cx, cy, zoom);
            println!("blend-explore: {stem} vs up to {samples} draws from {} ({}x{} x {} frames)",
                     pool.display(), opts.probe_w, opts.probe_h, opts.frames);

            let started = std::time::Instant::now();
            let cands = time_explore::blend_pool_search(
                &genome, &config, &view, &pool, samples, seed, &opts,
                &|i, total, c| {
                    let verdict = match c.rejected {
                        Some("incompatible") => format!("incompatible: {}", c.note),
                        Some(why) => format!("rejected: {why}"),
                        None => format!("score {:.4}", c.score),
                    };
                    println!("  [{i}/{total}] {} {} amp {:.2} -> {verdict}", c.partner_id, c.shape.label(), c.amp);
                });
            time_explore::write_blend_manifest(&out_dir, &cands, &genome, &config, &view, &opts, keep_clips)
                .unwrap_or_else(|e| panic!("write {}/blend_winners.jsonl: {e}", out_dir.display()));
            println!("blend-explore: {} in {:.1}s -> {}",
                     time_explore::blend_summary(&cands), started.elapsed().as_secs_f32(), out_dir.display());
        }
        Some("shot") => {
            // Ad-hoc visual inspection utility: render one genome+view
            // straight to a PNG, no pool/manifest/out_dir bookkeeping.
            // Added 2026-08-11 for diagnosing the coarse-scan zoom-depth
            // regression visually rather than purely from logs.
            let genome_path = pos.get(1).map(PathBuf::from)
                .unwrap_or_else(|| panic!("shot needs a genome .nn path"));
            let genome = io::load_genome(&genome_path).unwrap_or_else(|e| panic!("load {}: {e}", genome_path.display()));
            let cx: f64 = pos.get(2).and_then(|s| s.parse().ok()).unwrap_or(genome.view_cx as f64);
            let cy: f64 = pos.get(3).and_then(|s| s.parse().ok()).unwrap_or(genome.view_cy as f64);
            let zoom: f64 = pos.get(4).and_then(|s| s.parse().ok()).unwrap_or(genome.view_zoom.max(0.1) as f64);
            let res: u32 = pos.get(5).and_then(|s| s.parse().ok()).unwrap_or(1024);
            let out_path = pos.get(6).map(PathBuf::from).unwrap_or_else(|| PathBuf::from("shot.png"));
            let config = load_config();
            let view = View::new_square(cx, cy, zoom);
            // --angle-coloring renders through the SAME `render_save` path the
            // video exporter uses, so what this writes is what a video frame
            // would look like — the point of having it here is comparing the
            // two colourings on one view without launching the GUI.
            if args.iter().any(|a| a == "--angle-coloring") {
                let rgb = nnfractals::video_export::render_save(
                    &genome, &config, &view, res, res, true, false);
                nnfractals::io::save_png(&rgb, res, res, &out_path).expect("save screenshot");
            } else {
                save_shot(&genome, &config, &view, res, &out_path);
            }
            println!("saved {} at cx={cx} cy={cy} zoom={zoom}", out_path.display());
        }
        Some("debug-sweep") => {
            // Diagnostic-only: runs the SAME wide sweep pick_seeds uses
            // (WIDE_SCALES, EXPLORE_WIDE_RADIUS) from a given base view and
            // prints the full ranked candidate list plus wherever the
            // named target position landed in it — added 2026-08-11 to
            // investigate why a specific circular structure never got
            // picked as a seed, with real numbers instead of guessing.
            let genome_path = pos.get(1).map(PathBuf::from)
                .unwrap_or_else(|| panic!("debug-sweep needs a genome .nn path"));
            let genome = io::load_genome(&genome_path).unwrap_or_else(|e| panic!("load {}: {e}", genome_path.display()));
            let base_cx: f64 = pos.get(2).and_then(|s| s.parse().ok()).unwrap_or(genome.view_cx as f64);
            let base_cy: f64 = pos.get(3).and_then(|s| s.parse().ok()).unwrap_or(genome.view_cy as f64);
            let base_zoom: f64 = pos.get(4).and_then(|s| s.parse().ok()).unwrap_or(genome.view_zoom.max(0.1) as f64);
            let target_cx: Option<f64> = pos.get(5).and_then(|s| s.parse().ok());
            let target_cy: Option<f64> = pos.get(6).and_then(|s| s.parse().ok());
            let top_n: usize = get_flag_or(&args, "--top-n", 15);
            let config = load_config();
            let view = View::new_square(base_cx, base_cy, base_zoom);
            for method in ScoreMethod::ALL {
                let ranked = debug_sweep_candidates(
                    &genome, &config, &view, nnfractals::explore::EXPLORE_WIDE_RADIUS, method, nnfractals::explore::WIDE_SCALES,
                );
                println!("\n=== method={} — {} candidates ===", method.name(), ranked.len());
                for (i, (cx, cy, zoom, m, score)) in ranked.iter().take(top_n).enumerate() {
                    println!("  #{:>2} score={:.4} cx={:.6} cy={:.6} zoom={:.4e}  entropy={:.3} edge={:.3} intric={:.3} degenerate={}",
                        i + 1, score, cx, cy, zoom, m.entropy, m.edge_density, m.intricacy, m.degenerate);
                }
                if let (Some(tx), Some(ty)) = (target_cx, target_cy) {
                    let mut best: Option<(usize, f64, &(f64, f64, f64, Metrics, f32))> = None;
                    for (i, c) in ranked.iter().enumerate() {
                        let d = ((c.0 - tx).powi(2) + (c.1 - ty).powi(2)).sqrt();
                        if best.as_ref().is_none_or(|(_, bd, _)| d < *bd) { best = Some((i, d, c)); }
                    }
                    if let Some((rank, dist, (cx, cy, zoom, m, score))) = best {
                        println!("  closest CENTER to target ({tx:.6},{ty:.6}): rank #{}/{} dist={dist:.4} score={score:.4} cx={cx:.6} cy={cy:.6} zoom={zoom:.4e} entropy={:.3} edge={:.3} intric={:.3} degenerate={}",
                            rank + 1, ranked.len(), m.entropy, m.edge_density, m.intricacy, m.degenerate);
                    }
                    // Distinct question: does the target fall WITHIN any
                    // candidate's own crop extent at all, regardless of
                    // how far that crop's reported CENTER is? A huge/wide
                    // candidate can legitimately contain the target while
                    // being centered far from it.
                    let containing: Vec<&(f64, f64, f64, Metrics, f32)> = ranked.iter()
                        .filter(|c| {
                            let half = 2.0 / c.2;
                            (c.0 - tx).abs() < half && (c.1 - ty).abs() < half
                        })
                        .collect();
                    println!("  {} / {} candidates' OWN crop actually contains the target", containing.len(), ranked.len());
                    for (cx, cy, zoom, m, score) in containing.iter().take(5) {
                        println!("    contains target: score={score:.4} cx={cx:.6} cy={cy:.6} zoom={zoom:.4e} entropy={:.3} edge={:.3} intric={:.3} degenerate={}",
                            m.entropy, m.edge_density, m.intricacy, m.degenerate);
                    }
                }
            }
        }
        Some("vae-curate") => {
            let pool_dir = pos.get(1).map(PathBuf::from).unwrap_or_else(|| PathBuf::from("explorer_out/mandelbrot_vae"));
            let top_n: usize = pos.get(2).and_then(|s| s.parse().ok()).unwrap_or(30);
            let out_dir = pos.get(3).map(PathBuf::from).unwrap_or_else(|| pool_dir.join("curated"));
            let res: u32 = pos.get(4).and_then(|s| s.parse().ok()).unwrap_or(4000);
            let select_by = parse_select_by(get_flag(&args, "--select-by").unwrap_or("max-error"));
            cmd_vae_curate(&pool_dir, top_n, &out_dir, res, select_by);
        }
        Some("saliency-data") => {
            let out_dir = pos.get(1).map(PathBuf::from).unwrap_or_else(|| PathBuf::from("explorer_out/saliency_dataset"));
            let pool_dirs: Vec<PathBuf> = pos.iter().skip(2).map(PathBuf::from).collect();
            if pool_dirs.is_empty() {
                panic!("saliency-data needs at least one pool_dir (a vae-explore output directory with a vae_recon_manifest.jsonl, or a plain directory of .nn files scored live via --vae-model)");
            }
            let canvas_res: u32 = get_flag_or(&args, "--canvas-res", SALIENCY_CANVAS_RES);
            let max_per_pool: usize = get_flag_or(&args, "--max-per-pool", 3000);
            let vae_model_path: Option<PathBuf> = get_flag(&args, "--vae-model").map(PathBuf::from);
            cmd_saliency_data(&pool_dirs, &out_dir, canvas_res, max_per_pool, vae_model_path.as_deref());
        }
        Some("retrain-saliency") => {
            let out_dir = pos.get(1).map(PathBuf::from).unwrap_or_else(|| PathBuf::from("explorer_out/saliency_dataset"));
            let canvas_res: u32 = get_flag_or(&args, "--canvas-res", SALIENCY_CANVAS_RES);
            let max_per_pool: usize = get_flag_or(&args, "--max-per-pool", 1500);
            let vae_model_path: PathBuf = get_flag(&args, "--vae-model")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("explorer_out/last_successful_vae.pt"));
            let epochs: usize = get_flag_or(&args, "--epochs", 40);
            cmd_retrain_saliency(&out_dir, canvas_res, max_per_pool, &vae_model_path, epochs);
        }
        Some("complex-export") => {
            let input = pos.get(1).map(PathBuf::from)
                .unwrap_or_else(|| panic!("complex-export needs an input .nn file or directory"));
            let out_dir = pos.get(2).map(PathBuf::from).unwrap_or_else(|| PathBuf::from("explorer_out/complex_export"));
            let res: u32 = get_flag_or(&args, "--res", 512);
            let limit: usize = get_flag_or(&args, "--limit", 20);
            cmd_complex_export(&input, &out_dir, res, limit);
        }
        Some("verify-chain") => {
            let queue_id = get_flag(&args, "--queue-id").map(str::to_string);
            let stride: usize = get_flag_or(&args, "--stride", 1);
            let max_iter_override: Option<u32> = get_flag(&args, "--max-iter").and_then(|s| s.parse().ok());
            let dump_dir = get_flag(&args, "--dump-frames").map(PathBuf::from);
            let iter_sweep = get_flag(&args, "--iter-sweep").map(str::to_string);
            let sweep_res: u32 = get_flag_or(&args, "--sweep-res", 384);
            let render_video = get_flag(&args, "--render-video").map(PathBuf::from);
            let render_dims = (get_flag(&args, "--render-width").and_then(|s| s.parse().ok()),
                               get_flag(&args, "--render-height").and_then(|s| s.parse().ok()));
            let render_steps = get_flag(&args, "--render-steps").and_then(|s| s.parse().ok());
            let max_frames = get_flag(&args, "--max-frames").and_then(|s| s.parse().ok());
            // Default 16 (interpolate), matching the viewer pref and zoom_batch.sh —
            // see `default_video_keyframe_stride`. Pass 1 for an exact render.
            let keyframe_stride: u32 = get_flag_or(&args, "--keyframe-stride", 16);
            let winners = get_flag(&args, "--winners").map(PathBuf::from);
            let rank: usize = get_flag_or(&args, "--rank", 0);
            let nn_override = get_flag(&args, "--nn").map(PathBuf::from);
            let fps_override = get_flag(&args, "--render-fps").and_then(|s| s.parse().ok());
            cmd_verify_chain(
                queue_id.as_deref(), stride, max_iter_override, dump_dir.as_deref(),
                iter_sweep.as_deref(), sweep_res, render_video.as_deref(), render_dims, render_steps,
                winners.as_deref(), rank, nn_override.as_deref(), fps_override, max_frames,
                keyframe_stride, args.iter().any(|a| a == "--angle-coloring"),
            );
        }
        Some("quat-mandelbrot") => {
            let motion_kind = pos.get(1).map(String::as_str)
                .unwrap_or_else(|| panic!("quat-mandelbrot needs a motion: orbit|panzoom|gravity"));
            let out_path = pos.get(2).map(PathBuf::from).unwrap_or_else(||
                PathBuf::from(format!("explorer_out/quat_mandelbrot/{motion_kind}_{}.mp4", timestamp())));
            let frames: u32 = get_flag_or(&args, "--frames", 144);
            let fps: u32 = get_flag_or(&args, "--fps", 24);
            let width: u32 = get_flag_or(&args, "--width", SHOT_RES);
            let height: u32 = get_flag_or(&args, "--height", SHOT_RES);
            let max_iter: u32 = get_flag_or(&args, "--max-iter", 192);
            let bailout: f64 = get_flag_or(&args, "--bailout", 4.0);
            let colormap_name = get_flag(&args, "--colormap").unwrap_or("turbo").to_string();
            let formula_name = get_flag(&args, "--formula").unwrap_or("mandelbrot");
            let formula = nnfractals::quat_fractal::QuatFormula::parse(formula_name).unwrap_or_else(|| {
                let names: Vec<&str> = nnfractals::quat_fractal::QuatFormula::ALL.iter().map(|f| f.name()).collect();
                panic!("unknown --formula {formula_name:?} — expected one of: {}", names.join(", "))
            });
            let time_axis = parse_time_axis(get_flag(&args, "--time-axis").unwrap_or("c"));

            let motion = match motion_kind {
                "orbit" => nnfractals::quat_motion::SliceMotion::Orbit(nnfractals::quat_motion::OrbitParams {
                    pivot: parse_vec3(get_flag(&args, "--pivot").unwrap_or("0,0,0")),
                    axis: parse_vec3(get_flag(&args, "--axis").unwrap_or("0,1,0")),
                    radius: get_flag_or(&args, "--radius", 1.0),
                    turns: get_flag_or(&args, "--turns", 1.0),
                    phase0: get_flag_or(&args, "--phase0", 0.0),
                    zoom: get_flag_or(&args, "--zoom", 1.0),
                    c0: get_flag_or(&args, "--c0", 0.0),
                    c1: get_flag_or(&args, "--c1", 0.0),
                }),
                "panzoom" => nnfractals::quat_motion::SliceMotion::PanZoom(nnfractals::quat_motion::PanZoomParams {
                    origin0: parse_vec3(get_flag(&args, "--origin").unwrap_or("0,0,0")),
                    direction: parse_vec3(get_flag(&args, "--direction").unwrap_or("1,0,0")),
                    distance: get_flag_or(&args, "--distance", 1.0),
                    basis_u: parse_vec3(get_flag(&args, "--basis-u").unwrap_or("1,0,0")),
                    basis_v: parse_vec3(get_flag(&args, "--basis-v").unwrap_or("0,1,0")),
                    zoom0: get_flag_or(&args, "--zoom0", 1.0),
                    zoom1: get_flag_or(&args, "--zoom1", 4.0),
                    c0: get_flag_or(&args, "--c0", 0.0),
                    c1: get_flag_or(&args, "--c1", 0.0),
                }),
                "gravity" => {
                    let c0: f64 = get_flag_or(&args, "--c0", get_flag_or(&args, "--c", 0.0));
                    let c_shape_name = get_flag(&args, "--c-shape").unwrap_or("ramp");
                    let time_shape = ModShape::parse(c_shape_name).unwrap_or_else(|| {
                        let names: Vec<&str> = ModShape::ALL.iter().map(|s| s.label()).collect();
                        panic!("unknown --c-shape {c_shape_name:?} — expected one of: {}", names.join(", "))
                    });
                    // Presence of --perspective enables a genuine pinhole
                    // camera instead of the default orthographic slice — see
                    // PerspectiveCamera's doc comment for why --perspective-
                    // tilt-u/-v (a LATERAL eye offset, not just the pullback
                    // distance) are what actually turn a centered circle
                    // into a visible ellipse for the spherically-symmetric
                    // formulas under --time-axis r.
                    let perspective = get_flag(&args, "--perspective").and_then(|s| s.parse::<f64>().ok()).map(|distance| {
                        nnfractals::quat_fractal::PerspectiveCamera {
                            tilt_u: get_flag_or(&args, "--perspective-tilt-u", 0.5),
                            tilt_v: get_flag_or(&args, "--perspective-tilt-v", 0.0),
                            distance,
                        }
                    });
                    let sim_params = nnfractals::quat_gravity::ProjectileParams {
                        formula,
                        time_axis,
                        time_val0: c0,
                        time_val1: get_flag_or(&args, "--c1", c0),
                        time_shape,
                        time_freq: get_flag_or(&args, "--c-freq", 1.0),
                        time_phase: get_flag_or(&args, "--c-phase", 0.0),
                        domain_extent: get_flag_or(&args, "--extent", 1.6),
                        max_iter: get_flag_or(&args, "--mass-max-iter", 48),
                        bailout,
                        mass_samples: get_flag_or(&args, "--mass-samples", 48),
                        mass_power: get_flag_or(&args, "--mass-power", 3.0),
                        // NOT a basis vector on purpose — see quat_gravity.rs's
                        // ProjectileParams::axis doc comment. A basis-aligned
                        // axis (the old "0,1,0" default) confines the orbit to
                        // a plane where that one raw R/A/B/C coordinate is
                        // exactly frozen for the entire clip.
                        axis: parse_vec3(get_flag(&args, "--axis").unwrap_or("0.65,0.42,0.83")),
                        start_radius: get_flag_or(&args, "--start-radius", 1.8),
                        mu: get_flag_or(&args, "--mu", 25.0),
                        damping_per_sec: get_flag_or(&args, "--damping", 0.12),
                        sim_dt: get_flag_or(&args, "--sim-dt", 0.05),
                        softening: get_flag_or(&args, "--softening", 0.05),
                        zoom: get_flag_or(&args, "--zoom", 1.5),
                        perspective,
                    };
                    let trajectory = nnfractals::quat_gravity::simulate_projectile(&sim_params, frames);
                    nnfractals::quat_motion::SliceMotion::Trajectory(trajectory)
                }
                other => panic!("unknown motion {other:?} — expected orbit|panzoom|gravity"),
            };
            let overlay_coords = args.iter().any(|a| a == "--overlay-coords");
            cmd_quat_mandelbrot(formula, time_axis, motion, frames, fps, width, height, max_iter, bailout, &colormap_name, &out_path, overlay_coords);
        }
        Some("quat-voxel-stl") => {
            let out_path = pos.get(1).map(PathBuf::from).unwrap_or_else(||
                PathBuf::from(format!("explorer_out/quat_mandelbrot/voxel_{}.stl", timestamp())));
            let formula_name = get_flag(&args, "--formula").unwrap_or("mandelbrot");
            let formula = nnfractals::quat_fractal::QuatFormula::parse(formula_name).unwrap_or_else(|| {
                let names: Vec<&str> = nnfractals::quat_fractal::QuatFormula::ALL.iter().map(|f| f.name()).collect();
                panic!("unknown --formula {formula_name:?} — expected one of: {}", names.join(", "))
            });
            let time_axis = parse_time_axis(get_flag(&args, "--time-axis").unwrap_or("c"));
            let opts = nnfractals::quat_voxel::VoxelStlOpts {
                formula,
                time_axis,
                time_val: get_flag_or(&args, "--c", 0.0),
                res: get_flag_or(&args, "--res", 256),
                domain_extent: get_flag_or(&args, "--extent", 1.6),
                max_iter: get_flag_or(&args, "--max-iter", 48),
                bailout: get_flag_or(&args, "--bailout", 4.0),
                smooth_iters: get_flag_or(&args, "--smooth-iters", 4),
                smooth_alpha: get_flag_or(&args, "--smooth-alpha", 0.5),
                target_tris: get_flag_or(&args, "--target-tris", 400_000),
                target_error: get_flag_or(&args, "--target-error", 0.02),
            };
            cmd_quat_voxel_stl(opts, &out_path);
        }
        Some("quat-raymarch") => {
            let out_path = pos.get(1).map(PathBuf::from).unwrap_or_else(||
                PathBuf::from(format!("explorer_out/quat_mandelbrot/raymarch_{}.png", timestamp())));
            let formula_name = get_flag(&args, "--formula").unwrap_or("bulb");
            let formula = nnfractals::quat_fractal::QuatFormula::parse(formula_name).unwrap_or_else(|| {
                let names: Vec<&str> = nnfractals::quat_fractal::QuatFormula::ALL.iter().map(|f| f.name()).collect();
                panic!("unknown --formula {formula_name:?} — expected one of: {}", names.join(", "))
            });
            let time_axis = parse_time_axis(get_flag(&args, "--time-axis").unwrap_or("c"));
            let width: u32 = get_flag_or(&args, "--width", 960);
            let height: u32 = get_flag_or(&args, "--height", 960);
            let params = nnfractals::quat_raymarch::RaymarchParams {
                formula,
                time_axis,
                time_val: get_flag_or(&args, "--c", 0.0),
                domain_radius: get_flag_or(&args, "--domain-radius", 1.6),
                max_iter: get_flag_or(&args, "--max-iter", 60),
                bailout: get_flag_or(&args, "--bailout", 4.0),
                max_march_steps: get_flag_or(&args, "--max-march-steps", 200),
                hit_epsilon: get_flag_or(&args, "--hit-eps", 1e-4 * get_flag_or(&args, "--domain-radius", 1.6)),
                step_safety: get_flag_or(&args, "--step-safety", 0.8),
                light_dir: parse_vec3(get_flag(&args, "--light").unwrap_or("0.5,0.8,0.3")),
                normal_eps: get_flag_or(&args, "--normal-eps", 1e-3 * get_flag_or(&args, "--domain-radius", 1.6)),
                color_probe_offset: get_flag_or(&args, "--color-probe-offset", 1e-2 * get_flag_or(&args, "--domain-radius", 1.6)),
                aa: get_flag_or(&args, "--aa", 1),
                bulb_power: get_flag_or(&args, "--power", 8.0),
                mandelbox_scale: get_flag_or(&args, "--mandelbox-scale", -1.5),
            };
            let cam = nnfractals::quat_raymarch::RaymarchCamera {
                eye: parse_vec3(get_flag(&args, "--eye").unwrap_or("0,0,-4")),
                target: parse_vec3(get_flag(&args, "--target").unwrap_or("0,0,0")),
                up_hint: parse_vec3(get_flag(&args, "--up").unwrap_or("0,1,0")),
                fov_y: get_flag_or::<f64>(&args, "--fov-deg", 50.0).to_radians(),
            };
            let colormap_name = get_flag(&args, "--colormap").unwrap_or("turbo").to_string();
            let bg_color = {
                let v = parse_vec3(get_flag(&args, "--bg-color").unwrap_or("0.03,0.02,0.06"));
                (v.0 as f32, v.1 as f32, v.2 as f32)
            };
            let use_gpu = args.iter().any(|a| a == "--gpu");
            cmd_quat_raymarch(&params, &cam, width, height, &colormap_name, bg_color, use_gpu, &out_path);
        }
        Some("quat-raymarch-genome") => {
            let out_path = pos.get(1).map(PathBuf::from).unwrap_or_else(||
                PathBuf::from(format!("explorer_out/quat_mandelbrot/raymarch_genome_{}.png", timestamp())));
            let genome_path = get_flag(&args, "--genome").unwrap_or_else(|| panic!("quat-raymarch-genome needs --genome path.nn"));
            let genome = io::load_genome(std::path::Path::new(genome_path)).unwrap_or_else(|e| panic!("failed to load genome {genome_path:?}: {e}"));
            if genome.program.is_empty() {
                panic!("genome {genome_path:?} has an empty DAG program (legacy 58-basis genome?) — quat-raymarch-genome only supports DAG-based genomes");
            }
            let time_axis = parse_time_axis(get_flag(&args, "--time-axis").unwrap_or("c"));
            let width: u32 = get_flag_or(&args, "--width", 960);
            let height: u32 = get_flag_or(&args, "--height", 960);
            let dag_formula = nnfractals::quat_dag::QuatDagFormula {
                prog: &genome.program,
                warp: &genome.warp,
                julia: genome.julia_mode,
                jc: (genome.julia_cre, genome.julia_cim),
                phoenix: (genome.phoenix_re, genome.phoenix_im),
            };
            let params = nnfractals::quat_dag::RaymarchDagParams {
                formula: dag_formula,
                time_axis,
                time_val: get_flag_or(&args, "--c", 0.0),
                domain_radius: get_flag_or(&args, "--domain-radius", 1.6),
                max_iter: get_flag_or(&args, "--max-iter", 60),
                bailout: get_flag_or(&args, "--bailout", genome.bailout_radius as f64),
                max_march_steps: get_flag_or(&args, "--max-march-steps", 200),
                hit_epsilon: get_flag_or(&args, "--hit-eps", 1e-4 * get_flag_or(&args, "--domain-radius", 1.6)),
                step_safety: get_flag_or(&args, "--step-safety", 0.8),
                light_dir: parse_vec3(get_flag(&args, "--light").unwrap_or("0.5,0.8,0.3")),
                normal_eps: get_flag_or(&args, "--normal-eps", 1e-3 * get_flag_or(&args, "--domain-radius", 1.6)),
                color_probe_offset: get_flag_or(&args, "--color-probe-offset", 1e-2 * get_flag_or(&args, "--domain-radius", 1.6)),
                aa: get_flag_or(&args, "--aa", 1),
            };
            let cam = nnfractals::quat_raymarch::RaymarchCamera {
                eye: parse_vec3(get_flag(&args, "--eye").unwrap_or("0,0,-4")),
                target: parse_vec3(get_flag(&args, "--target").unwrap_or("0,0,0")),
                up_hint: parse_vec3(get_flag(&args, "--up").unwrap_or("0,1,0")),
                fov_y: get_flag_or::<f64>(&args, "--fov-deg", 50.0).to_radians(),
            };
            let colormap_name = get_flag(&args, "--colormap").unwrap_or("turbo").to_string();
            let bg_color = {
                let v = parse_vec3(get_flag(&args, "--bg-color").unwrap_or("0.03,0.02,0.06"));
                (v.0 as f32, v.1 as f32, v.2 as f32)
            };
            let genome_label = std::path::Path::new(genome_path).file_stem().and_then(|s| s.to_str()).unwrap_or(genome_path).to_string();
            let use_gpu = args.iter().any(|a| a == "--gpu");
            cmd_quat_raymarch_dag(&params, &cam, &genome_label, width, height, &colormap_name, bg_color, use_gpu, &out_path);
        }
        Some("quat-raymarch-genome-video") => {
            let out_path = pos.get(1).map(PathBuf::from).unwrap_or_else(||
                PathBuf::from(format!("explorer_out/quat_mandelbrot/raymarch_genome_orbit_{}.mp4", timestamp())));
            let genome_path = get_flag(&args, "--genome").unwrap_or_else(|| panic!("quat-raymarch-genome-video needs --genome path.nn"));
            let genome = io::load_genome(std::path::Path::new(genome_path)).unwrap_or_else(|e| panic!("failed to load genome {genome_path:?}: {e}"));
            if genome.program.is_empty() {
                panic!("genome {genome_path:?} has an empty DAG program (legacy 58-basis genome?) — quat-raymarch-genome-video only supports DAG-based genomes");
            }
            let time_axis = parse_time_axis(get_flag(&args, "--time-axis").unwrap_or("c"));
            let width: u32 = get_flag_or(&args, "--width", 640);
            let height: u32 = get_flag_or(&args, "--height", 640);
            let frames: u32 = get_flag_or(&args, "--frames", 90);
            let fps: u32 = get_flag_or(&args, "--fps", 30);
            let c_fixed = get_flag(&args, "--c").and_then(|s| s.parse::<f64>().ok());
            let c0: f64 = get_flag_or(&args, "--c0", c_fixed.unwrap_or(-0.6));
            let c1: f64 = get_flag_or(&args, "--c1", c_fixed.unwrap_or(0.6));
            let c_shape_name = get_flag(&args, "--c-shape").unwrap_or("sine");
            let c_shape = nnfractals::formula::ModShape::parse(c_shape_name).unwrap_or_else(|| {
                let names: Vec<&str> = nnfractals::formula::ModShape::ALL.iter().map(|s| s.label()).collect();
                panic!("unknown --c-shape {c_shape_name:?} — expected one of: {}", names.join(", "))
            });
            let pulse = CPulseParams {
                c0, c1, shape: c_shape,
                freq: get_flag_or(&args, "--c-freq", 1.5),
                phase: get_flag_or(&args, "--c-phase", 0.0),
            };
            let dag_formula = nnfractals::quat_dag::QuatDagFormula {
                prog: &genome.program,
                warp: &genome.warp,
                julia: genome.julia_mode,
                jc: (genome.julia_cre, genome.julia_cim),
                phoenix: (genome.phoenix_re, genome.phoenix_im),
            };
            let params = nnfractals::quat_dag::RaymarchDagParams {
                formula: dag_formula,
                time_axis,
                time_val: c0,
                domain_radius: get_flag_or(&args, "--domain-radius", 1.6),
                max_iter: get_flag_or(&args, "--max-iter", 60),
                bailout: get_flag_or(&args, "--bailout", genome.bailout_radius as f64),
                max_march_steps: get_flag_or(&args, "--max-march-steps", 200),
                hit_epsilon: get_flag_or(&args, "--hit-eps", 1e-4 * get_flag_or(&args, "--domain-radius", 1.6)),
                step_safety: get_flag_or(&args, "--step-safety", 0.8),
                light_dir: parse_vec3(get_flag(&args, "--light").unwrap_or("0.5,0.8,0.3")),
                normal_eps: get_flag_or(&args, "--normal-eps", 1e-3 * get_flag_or(&args, "--domain-radius", 1.6)),
                color_probe_offset: get_flag_or(&args, "--color-probe-offset", 1e-2 * get_flag_or(&args, "--domain-radius", 1.6)),
                aa: get_flag_or(&args, "--aa", 1),
            };
            let fov_deg_for_orbit: f64 = get_flag_or(&args, "--fov-deg", 45.0);
            let domain_radius_for_orbit: f64 = get_flag_or(&args, "--domain-radius", 1.6);
            // Default --radius is now ASPECT-AWARE (see
            // recommended_orbit_radius's docs) — a fixed default like the
            // old "4.0" framed a square probe fine but left the object
            // filling almost the entire frame with no margin once
            // rendered at a portrait aspect like 1080x1920 (a fixed
            // vertical FOV narrows the effective horizontal FOV a lot for
            // aspect<1), which is exactly the "camera too close, never
            // letting appreciate the outline" bug Carl caught by eye.
            let default_radius = recommended_orbit_radius(domain_radius_for_orbit, fov_deg_for_orbit, width, height);
            let orbit = nnfractals::quat_raymarch::RaymarchOrbitParams {
                target: parse_vec3(get_flag(&args, "--target").unwrap_or("0,0,0")),
                axis: parse_vec3(get_flag(&args, "--axis").unwrap_or("0,1,0")),
                radius: get_flag_or(&args, "--radius", default_radius),
                turns: get_flag_or(&args, "--turns", 1.0),
                phase0: get_flag_or(&args, "--phase0", 0.0),
                fov_y: fov_deg_for_orbit.to_radians(),
            };
            let colormap_name = get_flag(&args, "--colormap").unwrap_or("lava").to_string();
            let bg_color = {
                let v = parse_vec3(get_flag(&args, "--bg-color").unwrap_or("0.03,0.02,0.06"));
                (v.0 as f32, v.1 as f32, v.2 as f32)
            };
            let genome_label = std::path::Path::new(genome_path).file_stem().and_then(|s| s.to_str()).unwrap_or(genome_path).to_string();
            let use_gpu = args.iter().any(|a| a == "--gpu");
            eprintln!("  [framing] --radius defaulted to {default_radius:.2} for {width}x{height} at fov-deg={fov_deg_for_orbit} (aspect-aware; pass --radius to override)");
            cmd_quat_raymarch_dag_video(params, orbit, pulse, genome_label, frames, fps, width, height, colormap_name, bg_color, use_gpu, &out_path);
        }
        Some("quat-formula-full-metrics") => {
            let formula_name = get_flag(&args, "--formula").unwrap_or("bulb");
            let probe_size: u32 = get_flag_or(&args, "--probe-size", 128);
            let use_gpu = args.iter().any(|a| a == "--gpu");
            let out_path = PathBuf::from(get_flag(&args, "--out").unwrap_or("explorer_out/quat_formula_full_metrics.json"));
            if let Some(dir) = out_path.parent() {
                std::fs::create_dir_all(dir).ok();
            }
            cmd_quat_formula_full_metrics(formula_name, probe_size, use_gpu, &out_path);
        }
        Some("quat-dag-score") => {
            let use_gpu = args.iter().any(|a| a == "--gpu");
            let probe_size: u32 = get_flag_or(&args, "--probe-size", 96);
            if let Some(genome_path) = get_flag(&args, "--genome") {
                let genome = io::load_genome(std::path::Path::new(genome_path)).unwrap_or_else(|e| panic!("failed to load genome {genome_path:?}: {e}"));
                if genome.program.is_empty() {
                    panic!("genome {genome_path:?} has an empty DAG program (legacy 58-basis genome?) — quat-dag-score only supports DAG-based genomes");
                }
                let bd = score_genome_dag(&genome, probe_size, use_gpu);
                println!(
                    "{genome_path}  total={:.3}  silhouette_irregularity={:.3} anisotropy={:.3} coverage={:.3} solidity={:.3} shading_richness={:.3} color_entropy={:.3}",
                    bd.total(), bd.silhouette_irregularity, bd.anisotropy, bd.coverage, bd.solidity, bd.shading_richness, bd.color_entropy
                );
            } else if let Some(pool_dir) = get_flag(&args, "--pool-dir") {
                let sample_n: usize = get_flag_or(&args, "--sample", 500);
                let top_k: usize = get_flag_or(&args, "--top", 20);
                let seed: u64 = get_flag_or(&args, "--seed", 42);
                let out_list = get_flag(&args, "--out-list").map(PathBuf::from);

                let mut all_files: Vec<PathBuf> = std::fs::read_dir(pool_dir)
                    .unwrap_or_else(|e| panic!("failed to read --pool-dir {pool_dir:?}: {e}"))
                    .filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("nn"))
                    .collect();
                eprintln!("quat-dag-score: {} genomes found in {pool_dir}", all_files.len());
                let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
                all_files.shuffle(&mut rng);
                all_files.truncate(sample_n);
                eprintln!("quat-dag-score: scoring a random sample of {} (seed={seed}, probe_size={probe_size}, gpu={use_gpu})", all_files.len());

                let start = std::time::Instant::now();
                let mut scored: Vec<(PathBuf, nnfractals::quat_dag_fitness::QuatFitnessBreakdown)> = Vec::new();
                let mut skipped = 0usize;
                for (i, path) in all_files.iter().enumerate() {
                    let genome = match io::load_genome(path) {
                        Ok(g) => g,
                        Err(_) => { skipped += 1; continue; }
                    };
                    if genome.program.is_empty() {
                        skipped += 1;
                        continue;
                    }
                    let bd = score_genome_dag(&genome, probe_size, use_gpu);
                    scored.push((path.clone(), bd));
                    if (i + 1) % 50 == 0 {
                        let secs = start.elapsed().as_secs_f64();
                        eprint!("\r  scored {}/{} ({secs:.0}s, skipped {skipped})   ", i + 1, all_files.len());
                    }
                }
                eprintln!("\r  scored {}/{} in {:.0}s ({skipped} skipped — empty/legacy programs)", scored.len(), all_files.len(), start.elapsed().as_secs_f64());
                scored.sort_by(|a, b| b.1.total().partial_cmp(&a.1.total()).unwrap_or(std::cmp::Ordering::Equal));

                if let Some(out_path) = &out_list {
                    if let Some(dir) = out_path.parent() {
                        if !dir.as_os_str().is_empty() {
                            std::fs::create_dir_all(dir).expect("create out-list dir");
                        }
                    }
                    let mut csv = String::from("path,total,silhouette_irregularity,anisotropy,coverage,solidity,shading_richness,color_entropy\n");
                    for (path, bd) in &scored {
                        csv.push_str(&format!(
                            "{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}\n",
                            path.display(), bd.total(), bd.silhouette_irregularity, bd.anisotropy, bd.coverage, bd.solidity, bd.shading_richness, bd.color_entropy
                        ));
                    }
                    std::fs::write(out_path, csv).unwrap_or_else(|e| panic!("failed to write --out-list {out_path:?}: {e}"));
                    eprintln!("  wrote ranked list to {}", out_path.display());
                }

                println!("top {}:", top_k.min(scored.len()));
                for (path, bd) in scored.iter().take(top_k) {
                    println!(
                        "  {}  total={:.3}  silhouette_irregularity={:.3} anisotropy={:.3} coverage={:.3} solidity={:.3} shading_richness={:.3} color_entropy={:.3}",
                        path.display(), bd.total(), bd.silhouette_irregularity, bd.anisotropy, bd.coverage, bd.solidity, bd.shading_richness, bd.color_entropy
                    );
                }
            } else {
                panic!("quat-dag-score needs either --genome path.nn (score one) or --pool-dir DIR (score+rank a random sample)");
            }
        }
        Some("quat-dag-rescore") => {
            // Backfills every quat_* metric field (the original 6 +
            // ~24 new ones from quat_dag_fitness's extended module) onto
            // genomes that already exist on disk — genotype (program/
            // warp/id/...) is untouched, only the quat_* fields and
            // `fitness` are rewritten. Exists so an already-evolved
            // archive (e.g. an overnight run from before these metrics
            // existed) is immediately sortable by them in the gallery,
            // without re-running evolution.
            let dir = get_flag(&args, "--dir").unwrap_or_else(|| panic!("quat-dag-rescore needs --dir DIR"));
            let use_gpu = args.iter().any(|a| a == "--gpu");
            let probe_size: u32 = get_flag_or(&args, "--probe-size", 128);
            let files: Vec<PathBuf> = std::fs::read_dir(dir)
                .unwrap_or_else(|e| panic!("failed to read --dir {dir:?}: {e}"))
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("nn"))
                .collect();
            eprintln!("quat-dag-rescore: {} genomes in {dir}", files.len());
            let start = std::time::Instant::now();
            let mut done = 0usize;
            let mut skipped = 0usize;
            for path in &files {
                let mut genome = match io::load_genome(path) {
                    Ok(g) if !g.program.is_empty() => g,
                    _ => { skipped += 1; continue; }
                };
                let full = score_genome_dag_full(&genome, probe_size, use_gpu);
                apply_quat_full_metrics(&mut genome, &full);
                apply_organization_metrics(&mut genome);
                // Deliberately NOT touching `fitness` — it already holds
                // whatever meaning the genome was originally saved with
                // (e.g. quat-dag-evolve's blended geometric+aesthetic+
                // diversity total). Rescoring only adds the new quat_*
                // fields; it doesn't redefine what the existing
                // `fitness` column means for genomes that already have
                // one.
                io::save_genome(&genome, path).unwrap_or_else(|e| panic!("failed to re-save {path:?}: {e}"));
                done += 1;
                if done % 20 == 0 {
                    eprint!("\r  rescored {done}/{} ({:.0}s, {skipped} skipped)   ", files.len(), start.elapsed().as_secs_f64());
                }
            }
            eprintln!("\r  rescored {done}/{} in {:.0}s ({skipped} skipped — empty/legacy programs)", files.len(), start.elapsed().as_secs_f64());
        }
        Some("quat-dag-evolve") => {
            let pool_dir = get_flag(&args, "--pool-dir").unwrap_or("fractals_dag");
            let out_dir = get_flag(&args, "--out-dir").unwrap_or("fractals_dag_quat");
            let population: usize = get_flag_or(&args, "--population", 120);
            let generations: usize = get_flag_or(&args, "--generations", 25);
            let survivors: usize = get_flag_or(&args, "--survivors", 15);
            let probe_size: u32 = get_flag_or(&args, "--probe-size", 128);
            let seed: u64 = get_flag_or(&args, "--seed", 1);
            let use_gpu = args.iter().any(|a| a == "--gpu");
            let crossover_mode = match get_flag(&args, "--crossover-mode").unwrap_or("legacy") {
                "legacy" => CrossoverMode::Legacy,
                "subtree" => CrossoverMode::Subtree,
                other => panic!("unknown --crossover-mode {other:?} — expected legacy or subtree"),
            };
            let default_strength = nnfractals::quat_genome_ops::MutationStrength::default();
            let mutation_strength = nnfractals::quat_genome_ops::MutationStrength {
                min_edits: get_flag_or(&args, "--mutation-min-edits", default_strength.min_edits),
                max_edits: get_flag_or(&args, "--mutation-max-edits", default_strength.max_edits),
                const_perturb_scale: get_flag_or(&args, "--mutation-const-scale", default_strength.const_perturb_scale),
                max_depth: get_flag_or(&args, "--mutation-max-depth", default_strength.max_depth),
            };
            let fitness_metric = get_flag(&args, "--fitness-metric").map(parse_fitness_metric_spec);
            let predator_prey = args.iter().any(|a| a == "--predator-prey");
            let map_elites = args.iter().any(|a| a == "--map-elites");
            let stagnation_gens: usize = get_flag_or(&args, "--stagnation-gens", 25);
            let pref_model_path = get_flag(&args, "--pref-model").unwrap_or("pref_model_quat.json");
            if map_elites && predator_prey {
                panic!("--map-elites and --predator-prey are mutually exclusive — MAP-Elites' niches ARE the diversity mechanism, predator-prey pressure has nothing to act on");
            }
            if map_elites {
                cmd_quat_dag_evolve_map_elites(pool_dir, out_dir, population, generations, probe_size, use_gpu, seed, crossover_mode, mutation_strength, fitness_metric, stagnation_gens);
            } else {
                cmd_quat_dag_evolve(pool_dir, out_dir, population, generations, survivors, probe_size, use_gpu, seed, crossover_mode, mutation_strength, fitness_metric, predator_prey, stagnation_gens, pref_model_path);
            }
        }
        Some("quat-dag-random") => {
            let out_dir = get_flag(&args, "--out-dir").unwrap_or("fractals_dag_quat");
            let count: usize = get_flag_or(&args, "--count", 200);
            let seed: u64 = get_flag_or(&args, "--seed", 1);
            let use_gpu = args.iter().any(|a| a == "--gpu");
            cmd_quat_dag_random(out_dir, count, use_gpu, seed);
        }
        Some("quat-dag-thumbs") => {
            let dir = get_flag(&args, "--dir").unwrap_or_else(|| panic!("quat-dag-thumbs needs --dir DIR"));
            let use_gpu = args.iter().any(|a| a == "--gpu");
            let views = args.iter().any(|a| a == "--views");
            let files: Vec<PathBuf> = std::fs::read_dir(dir)
                .unwrap_or_else(|e| panic!("failed to read --dir {dir:?}: {e}"))
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("nn"))
                .collect();
            eprintln!("quat-dag-thumbs: {} genomes in {dir} (views={views})", files.len());
            let start = std::time::Instant::now();
            let mut done = 0usize;
            let mut stems = Vec::with_capacity(files.len());
            for path in &files {
                let genome = match io::load_genome(path) {
                    Ok(g) if !g.program.is_empty() => g,
                    _ => continue,
                };
                let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("thumb").to_string();
                render_genome_thumbnails(&genome, std::path::Path::new(dir), &stem, use_gpu);
                if views {
                    render_genome_views(&genome, std::path::Path::new(dir), &stem, use_gpu);
                }
                stems.push(stem);
                done += 1;
                if done % 10 == 0 {
                    eprint!("\r  {done}/{} ({:.0}s)   ", files.len(), start.elapsed().as_secs_f64());
                }
            }
            eprintln!("\r  done: {done}/{} thumbnails in {:.0}s", files.len(), start.elapsed().as_secs_f64());
            // Sorted so a combined sheet (e.g. several --fitness-metric
            // runs saved into the same --out-dir, filenames prefixed by
            // metric name) groups by metric instead of by random id/mtime.
            stems.sort();
            write_contact_sheet(std::path::Path::new(dir), &stems);
        }
        Some("quat-role-model-views") => {
            // Bulb power sweep + Mandelbox scale sweep + every other
            // QuatFormula::ALL variant at defaults — the honest scope of
            // usable role models found during planning: no .fract parser
            // exists anywhere in this repo, so the 788 imported
            // Mandelbulber files are NOT usable here, only these built-in
            // hand-coded formulas (see project-taste-driven-quat-ga
            // memory).
            let out_dir = get_flag(&args, "--out").unwrap_or("explorer_out/role_models_views");
            let use_gpu = args.iter().any(|a| a == "--gpu");
            use nnfractals::quat_fractal::QuatFormula;
            let mut stems = Vec::new();
            let bulb_powers = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 12.0];
            for p in bulb_powers {
                let stem = format!("rm_bulb_p{p:.0}");
                let label = format!("role-model bulb power={p:.0}");
                render_role_model_still(QuatFormula::Bulb, p, -1.5, std::path::Path::new(out_dir), &stem, &label, use_gpu);
                stems.push(stem);
            }
            let mandelbox_scales: [f64; 7] = [-2.0, -1.8, -1.5, -1.2, 2.0, 2.5, 3.0];
            for s in mandelbox_scales {
                let stem = format!("rm_mandelbox_s{}", (s * 10.0).round() as i32);
                let label = format!("role-model mandelbox scale={s:.1}");
                render_role_model_still(QuatFormula::Mandelbox, 8.0, s, std::path::Path::new(out_dir), &stem, &label, use_gpu);
                stems.push(stem);
            }
            for f in QuatFormula::ALL {
                if f == QuatFormula::Bulb || f == QuatFormula::Mandelbox {
                    continue;
                }
                let stem = format!("rm_{}", f.name());
                let label = format!("role-model {}", f.name());
                render_role_model_still(f, 8.0, -1.5, std::path::Path::new(out_dir), &stem, &label, use_gpu);
                stems.push(stem);
            }
            eprintln!("quat-role-model-views: rendered {} role models to {out_dir}", stems.len());
            stems.sort();
            write_contact_sheet(std::path::Path::new(out_dir), &stems);
        }
        Some("taste-pairs") => {
            let archive_dir = get_flag(&args, "--archive").unwrap_or_else(|| panic!("taste-pairs needs --archive DIR (a --map-elites out-dir)"));
            let out_dir = get_flag(&args, "--out").unwrap_or("fractals_dag_quat_to_rate");
            let n: usize = get_flag_or(&args, "--n", 60);
            cmd_taste_pairs(archive_dir, out_dir, n);
        }
        Some("quat-raymarch-video") => {
            let out_path = pos.get(1).map(PathBuf::from).unwrap_or_else(||
                PathBuf::from(format!("explorer_out/quat_mandelbrot/raymarch_orbit_{}.mp4", timestamp())));
            let formula_name = get_flag(&args, "--formula").unwrap_or("bulb");
            let formula = nnfractals::quat_fractal::QuatFormula::parse(formula_name).unwrap_or_else(|| {
                let names: Vec<&str> = nnfractals::quat_fractal::QuatFormula::ALL.iter().map(|f| f.name()).collect();
                panic!("unknown --formula {formula_name:?} — expected one of: {}", names.join(", "))
            });
            let time_axis = parse_time_axis(get_flag(&args, "--time-axis").unwrap_or("c"));
            let width: u32 = get_flag_or(&args, "--width", 640);
            let height: u32 = get_flag_or(&args, "--height", 640);
            let frames: u32 = get_flag_or(&args, "--frames", 90);
            let fps: u32 = get_flag_or(&args, "--fps", 30);
            // The time-axis value PULSES across the clip by default (a
            // sine breathing within [c0,c1], "once or twice per rotation"
            // via --c-freq — Carl's request) rather than staying fixed —
            // pass --c alone (no --c0/--c1) for the old fixed-value
            // behavior instead.
            let c_fixed = get_flag(&args, "--c").and_then(|s| s.parse::<f64>().ok());
            let c0: f64 = get_flag_or(&args, "--c0", c_fixed.unwrap_or(-0.6));
            let c1: f64 = get_flag_or(&args, "--c1", c_fixed.unwrap_or(0.6));
            let c_shape_name = get_flag(&args, "--c-shape").unwrap_or("sine");
            let c_shape = nnfractals::formula::ModShape::parse(c_shape_name).unwrap_or_else(|| {
                let names: Vec<&str> = nnfractals::formula::ModShape::ALL.iter().map(|s| s.label()).collect();
                panic!("unknown --c-shape {c_shape_name:?} — expected one of: {}", names.join(", "))
            });
            let pulse = CPulseParams {
                c0, c1, shape: c_shape,
                freq: get_flag_or(&args, "--c-freq", 1.5),
                phase: get_flag_or(&args, "--c-phase", 0.0),
            };
            let params = nnfractals::quat_raymarch::RaymarchParams {
                formula,
                time_axis,
                time_val: c0,
                domain_radius: get_flag_or(&args, "--domain-radius", 1.6),
                max_iter: get_flag_or(&args, "--max-iter", 60),
                bailout: get_flag_or(&args, "--bailout", 4.0),
                max_march_steps: get_flag_or(&args, "--max-march-steps", 200),
                hit_epsilon: get_flag_or(&args, "--hit-eps", 1e-4 * get_flag_or(&args, "--domain-radius", 1.6)),
                step_safety: get_flag_or(&args, "--step-safety", 0.8),
                light_dir: parse_vec3(get_flag(&args, "--light").unwrap_or("0.5,0.8,0.3")),
                normal_eps: get_flag_or(&args, "--normal-eps", 1e-3 * get_flag_or(&args, "--domain-radius", 1.6)),
                color_probe_offset: get_flag_or(&args, "--color-probe-offset", 1e-2 * get_flag_or(&args, "--domain-radius", 1.6)),
                aa: get_flag_or(&args, "--aa", 1),
                bulb_power: get_flag_or(&args, "--power", 8.0),
                mandelbox_scale: get_flag_or(&args, "--mandelbox-scale", -1.5),
            };
            let orbit = nnfractals::quat_raymarch::RaymarchOrbitParams {
                target: parse_vec3(get_flag(&args, "--target").unwrap_or("0,0,0")),
                axis: parse_vec3(get_flag(&args, "--axis").unwrap_or("0,1,0")),
                radius: get_flag_or(&args, "--radius", 4.0),
                turns: get_flag_or(&args, "--turns", 1.0),
                phase0: get_flag_or(&args, "--phase0", 0.0),
                fov_y: get_flag_or::<f64>(&args, "--fov-deg", 45.0).to_radians(),
            };
            let colormap_name = get_flag(&args, "--colormap").unwrap_or("lava").to_string();
            let bg_color = {
                let v = parse_vec3(get_flag(&args, "--bg-color").unwrap_or("0.03,0.02,0.06"));
                (v.0 as f32, v.1 as f32, v.2 as f32)
            };
            let use_gpu = args.iter().any(|a| a == "--gpu");
            cmd_quat_raymarch_video(params, orbit, pulse, frames, fps, width, height, colormap_name, bg_color, use_gpu, &out_path);
        }
        Some("quat-gravity-report") => {
            let frames: u32 = get_flag_or(&args, "--frames", 144);
            let formula_name = get_flag(&args, "--formula").unwrap_or("mandelbrot");
            let formula = nnfractals::quat_fractal::QuatFormula::parse(formula_name).unwrap_or_else(|| {
                let names: Vec<&str> = nnfractals::quat_fractal::QuatFormula::ALL.iter().map(|f| f.name()).collect();
                panic!("unknown --formula {formula_name:?} — expected one of: {}", names.join(", "))
            });
            let time_axis = parse_time_axis(get_flag(&args, "--time-axis").unwrap_or("c"));
            let c0: f64 = get_flag_or(&args, "--c0", get_flag_or(&args, "--c", 0.0));
            let c_shape_name = get_flag(&args, "--c-shape").unwrap_or("ramp");
            let time_shape = ModShape::parse(c_shape_name).unwrap_or_else(|| {
                let names: Vec<&str> = ModShape::ALL.iter().map(|s| s.label()).collect();
                panic!("unknown --c-shape {c_shape_name:?} — expected one of: {}", names.join(", "))
            });
            let perspective = get_flag(&args, "--perspective").and_then(|s| s.parse::<f64>().ok()).map(|distance| {
                nnfractals::quat_fractal::PerspectiveCamera {
                    tilt_u: get_flag_or(&args, "--perspective-tilt-u", 0.5),
                    tilt_v: get_flag_or(&args, "--perspective-tilt-v", 0.0),
                    distance,
                }
            });
            let sim_params = nnfractals::quat_gravity::ProjectileParams {
                formula,
                time_axis,
                time_val0: c0,
                time_val1: get_flag_or(&args, "--c1", c0),
                time_shape,
                time_freq: get_flag_or(&args, "--c-freq", 1.0),
                time_phase: get_flag_or(&args, "--c-phase", 0.0),
                domain_extent: get_flag_or(&args, "--extent", 1.6),
                max_iter: get_flag_or(&args, "--mass-max-iter", 48),
                bailout: get_flag_or(&args, "--bailout", 4.0),
                mass_samples: get_flag_or(&args, "--mass-samples", 48),
                mass_power: get_flag_or(&args, "--mass-power", 3.0),
                axis: parse_vec3(get_flag(&args, "--axis").unwrap_or("0.65,0.42,0.83")),
                start_radius: get_flag_or(&args, "--start-radius", 1.8),
                mu: get_flag_or(&args, "--mu", 25.0),
                damping_per_sec: get_flag_or(&args, "--damping", 0.12),
                sim_dt: get_flag_or(&args, "--sim-dt", 0.05),
                softening: get_flag_or(&args, "--softening", 0.05),
                zoom: get_flag_or(&args, "--zoom", 1.5),
                perspective,
            };
            let probe_res: u32 = get_flag_or(&args, "--probe-res", 32);
            let buckets: usize = get_flag_or(&args, "--buckets", 100);
            let delta_window: usize = get_flag_or(&args, "--delta-window", 1);
            cmd_quat_gravity_report(&sim_params, frames, probe_res, buckets, delta_window);
        }
        _ => {
            eprintln!("usage:");
            eprintln!("  nnfractals-explorer compare [out_dir]");
            eprintln!("  nnfractals-explorer run <entropy|edge|gated-entropy|gated-edge> [n_seeds] [max_rounds] [out_dir]");
            eprintln!("  nnfractals-explorer pool [formula] [method|mixed] [cx] [cy] [zoom] [n_seeds] [max_rounds] [min_score] [max_intricacy] [min_aesthetic] [min_edge_density] [out_dir]");
            eprintln!("  nnfractals-explorer prep-nav-data [nav_log.jsonl] [out_dir=nav_train_cache] [manifest=nav_manifest.jsonl]");
            eprintln!("  nnfractals-explorer score-mined-targets [nav_log_mined.jsonl]");
            eprintln!("  nnfractals-explorer vae-explore [formula] [cx] [cy] [zoom] [out_dir] [--iterations N] [--n-seeds N] [--recursion-depth N] [--top-k N] [--canvas-res N] [--method name|mixed] [--select-by max-error|min-error|random] [--max-intricacy F] [--min-edge-density F] [--arch conv|resnet|inception] [--latent-dim N] [--kl-weight F] [--tuned-config path.json] [--epochs N] [--target-recon-mse F] [--min-improvement F] [--patience N] [--saliency-model path.pt (default: explorer_out/saliency_model.pt if it exists)]");
            eprintln!("  nnfractals-explorer vae-curate [pool_dir] [top_n] [out_dir] [res] [--select-by max-error|min-error|random]");
            eprintln!("  nnfractals-explorer video-zoom-explore [formula|genome.nn] [cx] [cy] [zoom] [out_dir] [--depth N] [--finalists N] [--lookahead-plies N] [--method name|mixed] [--final-width N] [--final-height N] [--canvas-res N] [--top-winners N] [--n-seeds N] [--min-score F (0.15)] [--min-file-size-ratio F (0.45)] [--min-file-size-step-ratio F (0.80)] [--min-step-zoom F (2.0)] [--max-intricacy F (0.30)] [--min-edge-density F (0.05)] [--angle-coloring] [--lookahead-probe-w/h/steps/fps N] [--final-probe-w/h/steps/fps N] [--dd-margin-ulps F (1.0 = zoom until f64 pixelates; 4.0 = stop while still smooth)]");
            eprintln!("  nnfractals-explorer time-explore <formula|genome.nn> [cx] [cy] [zoom] [out_dir | --out DIR] [--frames N (48)] [--fps N (24)] [--probe-w N (192)] [--probe-h N (144)] [--amps 0.02,0.08,0.25] [--shapes sine,cosine,triangle,sawtooth,pulse,ramp,orbit] [--top-k N (8)] [--min-coherence F (0.55)] [--min-change F (1.0)] [--max-noise F (0.15)] [--max-still-run F (0.15)] [--max-level-jump F (12)] [--angle-coloring] [--keep-clips]");
            eprintln!("      searches the TIME axis: animates one scalar inside the formula (julia c / phoenix / bailout / a program or warp constant / an inserted scale node) and ranks by how well the clip resists video compression.");
            eprintln!("      three gates, all reported per candidate: 'noise' (spatially dithered frames), 'static' (amplitude too small to see — raise --amps), 'incoherent' (amplitude so large consecutive frames are unrelated: cuts, not a morph — lower --amps).");
            eprintln!("  nnfractals-explorer blend-explore <genome.nn> [cx] [cy] [zoom] [--out DIR] [--pool DIR] [--samples N (40)] [--seed N] [--frames N (24)] [--probe-w/h N] [--top-k N (8)] [--angle-coloring] [--keep-clips]");
            eprintln!("      morphs this fractal's FORMULA into other genomes from the pool (z' = f_A + s*(f_B - f_A)) and ranks the partners that morph best, using the same gates as time-explore.");
            eprintln!("  nnfractals-explorer complex-export <zone.nn|zone_dir> [out_dir] [--res 512] [--limit 20]");
            eprintln!("  nnfractals-explorer verify-chain [--queue-id ID (default: newest chain item)] [--stride N] [--max-iter N] [--dump-frames DIR]");
            eprintln!("      replays a queued chain's EXACT export frames offline: per-frame png size + 'flood' (fraction of the frame that is one colour).");
            eprintln!("      [--iter-sweep 192,384,768 --sweep-res N] finds the min iteration depth each zoom level needs;");
            eprintln!("      [--render-video OUT.mp4 --render-width N --render-height N --render-steps N --render-fps N] renders the chain from the CLI;\n      [--max-frames N] stops early; [--keyframe-stride N (default 16)] renders only every Nth frame and warps the rest (~8x faster); 1 = exact;\n      [--angle-coloring] forces exit-angle colouring (otherwise follows what the winners manifest recorded for the search).");
            eprintln!("  nnfractals-explorer shot <genome.nn> [cx] [cy] [zoom] [res=1024] [out.png=shot.png] [--angle-coloring]");
            eprintln!("  nnfractals-explorer saliency-data [out_dir] <pool_dir> [pool_dir...] [--canvas-res 256] [--max-per-pool 3000] [--vae-model path.pt (needed for pool_dirs with no vae_recon_manifest.jsonl, e.g. manual marks)]");
            eprintln!("  nnfractals-explorer retrain-saliency [out_dir] [--canvas-res 256] [--max-per-pool 1500] [--vae-model explorer_out/last_successful_vae.pt] [--epochs 40]");
            eprintln!("  nnfractals-explorer gems <method|method,method,...|mixed> [hours] [n_cols] [n_rows] [out_dir]");
            eprintln!("  nnfractals-explorer curate [archive.jsonl|pool_dir] [top_n] [min_score] [min_aesthetic] [min_dist] [res] [formula] [out_dir] [model_path] [head_path]");
            eprintln!("  nnfractals-explorer quat-mandelbrot <orbit|panzoom|gravity> [out.mp4] [--frames N (144)] [--fps N (24)] [--width N (960)] [--height N (960)] [--max-iter N (192)] [--bailout F (4.0)] [--colormap name (turbo)] [--formula name (mandelbrot)] [--time-axis r|a|b|c (c)]");
            eprintln!("      --time-axis: which quaternion component the time-driven value (--c/--c0/--c1) fills — the other three become the SPATIAL subspace the slice/mass-field explore. Default 'c' matches every render made before this flag existed. NOTE: under 'r', the pure-power formulas (mandelbrot/tricorn/cubic/quartic) become perfectly spherically symmetric in the spatial part (see quat_fractal.rs TimeAxis docs) — bulb and the abs-based formulas (burning-ship*/celtic/perpendicular-*) are unaffected and stay structured.");
            eprintln!("      --formula: mandelbrot|tricorn|burning-ship|burning-ship-cubic|perpendicular-burning-ship|celtic|perpendicular-mandelbrot|cubic|quartic|bulb — quaternion generalizations of the known_formulas.rs catalog, plus bulb (Mandelbulb-style angle multiplication — the only one that isn't secretly a solid of revolution, see quat_fractal.rs module docs).");
            eprintln!("      gravity: the slice is a projectile in a circular orbit around the fractal's own escape-time-weighted mass centroid, decaying inward over the clip and always facing the centroid (see quat_gravity.rs). The time-axis value ramps --c0 -> --c1 across the clip (independent of the orbit; the mass centroid is computed once, from --c0 only) — pass just --c (or nothing) for the old fixed-value behavior.");
            eprintln!("               [--c F (0.0) | --c0 F (0.0)] [--c1 F (=c0)] [--c-shape ramp|sine|cosine|triangle|sawtooth|pulse|orbit (ramp — a plain c0->c1 sweep, unchanged; any other shape instead oscillates within [c0,c1], center=midpoint amp=half-width)] [--c-freq F (1.0, cycles/clip)] [--c-phase F (0.0, turns)] [--extent F (1.6)] [--mass-max-iter N (48)] [--mass-samples N (48)] [--mass-power F (3.0)] [--axis R,A,B (0.65,0.42,0.83 — deliberately no component is 0 or 1, so it is not parallel or close to parallel to any single coordinate axis, else one raw coordinate freezes for the whole clip)] [--start-radius F (1.8)] [--mu F (25.0)] [--damping F (0.12)] [--sim-dt F (0.05)] [--softening F (0.05)] [--zoom F (1.5)] [--perspective F (unset = orthographic; a pullback distance enables a pinhole camera)] [--perspective-tilt-u F (0.5)] [--perspective-tilt-v F (0.0) (the LATERAL eye offset that actually foreshortens a centered circle into an ellipse — pullback distance alone stays rotationally symmetric, see PerspectiveCamera's doc comment)]");
            eprintln!("      [--overlay-coords] burns FRAME/T/AXIS and R/A/B/C directly onto every rendered frame (top-left corner) — debugging aid to see exactly where the slice sat in full 4D quaternion space without a separate file to keep in sync.");
            eprintln!("  nnfractals-explorer quat-voxel-stl [out.stl] [--formula name (mandelbrot)] [--c F (0.0)] [--time-axis r|a|b|c (c)] [--res N (256)] [--extent F (1.6)] [--max-iter N (48)] [--bailout F (4.0)] [--smooth-iters N (4)] [--smooth-alpha F (0.5)] [--target-tris N (400000)] [--target-error F (0.02)]");
            eprintln!("  nnfractals-explorer quat-raymarch [out.png] [--formula name (bulb — use a STRUCTURED formula: bulb|burning-ship|burning-ship-cubic|perpendicular-burning-ship|perpendicular-mandelbrot; the others are a pure function of one radial parameter under --time-axis r and won't show anything a flat plot wouldn't)] [--c F (0.0)] [--time-axis r|a|b|c (c)] [--width N (960)] [--height N (960)] [--eye R,A,B (0,0,-4)] [--target R,A,B (0,0,0)] [--up R,A,B (0,1,0)] [--fov-deg F (50.0)] [--domain-radius F (1.6)] [--max-iter N (60)] [--bailout F (4.0)] [--max-march-steps N (200)] [--hit-eps F (=1e-4*domain-radius)] [--step-safety F (0.8; <1.0 trades speed for robustness against quat_escape_de not being a certified distance bound)] [--light R,A,B (0.5,0.8,0.3)] [--normal-eps F (=1e-3*domain-radius)] [--colormap name (turbo — same palette catalog as every other render in this project: viridis|inferno|plasma|magma|cool|warm|cubehelix|earth|bone|neon|lava|aurora|galaxy|sunset|arctic|ember|grayscale)] [--bg-color R,G,B (0.03,0.02,0.06)] [--aa N (1; NxN supersampling)] [--gpu (dispatch the WGSL compute shader instead of CPU rayon; falls back to CPU if no GPU adapter is available)]");
            eprintln!("  nnfractals-explorer quat-raymarch-genome [out.png] --genome path.nn (a DAG-based genome — legacy 58-basis genomes aren't supported) [--c F (0.0)] [--time-axis r|a|b|c (c)] [--width N (960)] [--height N (960)] [--eye R,A,B (0,0,-4)] [--target R,A,B (0,0,0)] [--up R,A,B (0,1,0)] [--fov-deg F (50.0)] [--domain-radius F (1.6)] [--max-iter N (60)] [--bailout F (=genome's own bailout_radius)] [--max-march-steps N (200)] [--hit-eps F (=1e-4*domain-radius)] [--step-safety F (0.8 — analytic distance estimate, same default as quat-raymarch; see quat_dag.rs for the derivation)] [--light R,A,B (0.5,0.8,0.3)] [--normal-eps F (=1e-3*domain-radius)] [--colormap name (turbo)] [--bg-color R,G,B (0.03,0.02,0.06)] [--aa N (1)] [--gpu (dispatch the general WGSL DAG interpreter instead of CPU rayon — verified pixel-for-pixel against the CPU path in render_gpu_raymarch_dag's own tests; falls back to CPU if no GPU adapter is available)]");
            eprintln!("  nnfractals-explorer quat-raymarch-genome-video [out.mp4] --genome path.nn (a DAG-based genome) [--time-axis r|a|b|c (c)] [--c F (fixed, no pulse) | --c0/--c1 F (-0.6/0.6, sine pulse by default)] [--c-shape ramp|sine|cosine|triangle|sawtooth|pulse|orbit (sine)] [--c-freq F (1.5)] [--c-phase F (0.0)] [--frames N (90)] [--fps N (30)] [--width N (640)] [--height N (640)] [--target R,A,B (0,0,0)] [--axis R,A,B (0,1,0)] [--radius F (4.0)] [--turns F (1.0)] [--phase0 F (0.0)] [--fov-deg F (45.0)] [--domain-radius F (1.6)] [--max-iter N (60)] [--bailout F (=genome's own bailout_radius)] [--max-march-steps N (200)] [--hit-eps F (=1e-4*domain-radius)] [--step-safety F (0.8)] [--light R,A,B (0.5,0.8,0.3)] [--normal-eps F (=1e-3*domain-radius)] [--color-probe-offset F (=1e-2*domain-radius)] [--colormap name (lava)] [--bg-color R,G,B (0.03,0.02,0.06)] [--aa N (1)] [--gpu]");
            eprintln!("  nnfractals-explorer quat-formula-full-metrics [--formula name (bulb)] [--probe-size N (128)] [--gpu] [--out path.json (explorer_out/quat_formula_full_metrics.json)] — scores a classic hardcoded formula (not an evolved genome) across ALL 4 --time-axis choices (r/a/b/c), same probe camera/metric machinery as an evolved genome's own scoring, so the numbers are directly comparable to a .nn file's saved quat_* fields; writes one JSON object per axis plus an all-4-axes average");
            eprintln!("  nnfractals-explorer quat-dag-score --genome path.nn (score ONE genome: anisotropy/coverage/solidity/shading_richness/color_entropy + weighted total, see quat_dag_fitness.rs) | --pool-dir DIR (score+rank a random SAMPLE of genomes in DIR) [--sample N (500)] [--seed N (42)] [--top N (20)] [--out-list path.csv (write the full ranked list)] [--probe-size N (96, square low-res probe renders)] [--gpu]");
            eprintln!("  nnfractals-explorer quat-dag-evolve [--pool-dir DIR (fractals_dag, seeds the initial population)] [--out-dir DIR (fractals_dag_quat — deliberately separate, never writes into --pool-dir)] [--population N (40)] [--generations N (10)] [--survivors N (12), elitist: carried over each generation unchanged] [--probe-size N (128)] [--seed N (1)] [--gpu] [--crossover-mode legacy|subtree (legacy)] [--mutation-min-edits N] [--mutation-max-edits N] [--mutation-const-scale F] [--mutation-max-depth N] [--fitness-metric SPEC] [--pref-model PATH (pref_model_quat.json)] — evolves the DAG genome's `program` (mutation/crossover, same as the 2D GA unless --crossover-mode subtree) selecting on a 50/50 blend of the geometric pre-filter (quat_dag_fitness) and Carl's own trained preference model (quat_pref::QuatPrefModel — scripts/train_pref_quat.py, a linear fit over the ~30 quat_* metrics from the browser's ⚖ Rate pairwise comparisons; falls back to geometric-only, NOT the generic 2D NIMA/TOPIQ/AP25 ensemble, if --pref-model isn't found), UNLESS --fitness-metric SPEC is set, which selects purely on SPEC instead — a bare metric name (any quat_dag_fitness/quat_dag_fitness's extended-metrics field, e.g. convexity/box_dim/opcode_diversity — structural names need no rendering at all, much faster), or a comma-separated weighted combo (e.g. \"silhouette_irregularity:1.0,shading_gradient:0.7,color_shading_corr:0.5\", fitness = weighted sum); every saved filename is prefixed with the metric name(s) used (see quat-dag-thumbs for static + animated thumbnails) and every genome gets all ~30 quat_* metric fields backfilled (see quat-dag-rescore to backfill an existing archive)");
            eprintln!("  nnfractals-explorer quat-dag-thumbs --dir DIR [--gpu] — (re)generates the static `{{stem}}.png` and animated `{{stem}}_thumb_00..07.png` thumbnail sequence (nnfractals-browser reads both) for every genome already in DIR, without touching the genomes themselves");
            eprintln!("  nnfractals-explorer quat-raymarch-video [out.mp4] [--formula name (bulb)] [--time-axis r|a|b|c (c)] [--c F (fixed value, no pulse — omit this and use --c0/--c1 instead for the default pulsing behavior)] [--c0 F (-0.6)] [--c1 F (0.6)] [--c-shape ramp|sine|cosine|triangle|sawtooth|pulse|orbit (sine — the time-axis value breathes within [c0,c1] as the camera orbits, NOT fixed like the single-frame quat-raymarch; pass --c-shape ramp for a one-way sweep instead, or --c alone for the old fixed-value behavior)] [--c-freq F (1.5 — cycles per clip; with the default --turns 1.0, this IS cycles per camera revolution)] [--c-phase F (0.0)] [--frames N (90)] [--fps N (30)] [--width N (640)] [--height N (640)] [--target R,A,B (0,0,0)] [--axis R,A,B (0,1,0)] [--radius F (4.0)] [--turns F (1.0)] [--phase0 F (0.0)] [--fov-deg F (45.0)] [--domain-radius F (1.6)] [--max-iter N (60)] [--bailout F (4.0)] [--max-march-steps N (200)] [--hit-eps F (=1e-4*domain-radius)] [--step-safety F (0.8)] [--light R,A,B (0.5,0.8,0.3)] [--normal-eps F (=1e-3*domain-radius)] [--color-probe-offset F (=1e-2*domain-radius)] [--colormap name (lava)] [--bg-color R,G,B (0.03,0.02,0.06)] [--aa N (1)] [--gpu (dispatch the WGSL compute shader instead of CPU rayon — verified pixel-for-pixel against the CPU path in render_gpu_raymarch's own tests; falls back to CPU if no GPU adapter is available)]");
            eprintln!("      voxelizes the quaternion fractal at a FIXED C (the time axis quat-mandelbrot animates) into an res^3 (R,A,B) field, extracts a surface with Naive Surface Nets, smooths it, decimates to ~target-tris, writes an STL.");
            eprintln!("  nnfractals-explorer quat-gravity-report [--formula name (mandelbrot)] [--frames N (144)] [--probe-res N (32)] [--buckets N (100)] [--delta-window N (1, use ~your fps for a long/high-frame-rate clip)] [same gravity flags as quat-mandelbrot: --c/--c0/--c1/--time-axis/--axis/--start-radius/--mu/--damping/--sim-dt/--mass-samples/--mass-power/--zoom/...]");
            eprintln!("      simulates a `gravity` trajectory and probes it at tiny resolution (no video export) to check pacing BEFORE spending time on a real render: prints an ASCII timeline ('.' alive, '_' blank, '#' flat/interior, '=' static) plus blank/flat/static frame counts and the longest dead run.");
            eprintln!("      standalone quaternion-Mandelbrot prototype (q<-q^2+Q, Mandelbrot convention): the screen is a 2D slice through the (R,A,B) subspace, C is driven by time.");
            eprintln!("      orbit:   [--pivot R,A,B (0,0,0)] [--axis R,A,B (0,1,0)] [--radius F (1.0)] [--turns F (1.0)] [--phase0 F (0.0)] [--zoom F (1.0)] [--c0 F (0.0)] [--c1 F (0.0)]");
            eprintln!("      panzoom: [--origin R,A,B (0,0,0)] [--direction R,A,B (1,0,0)] [--distance F (1.0)] [--basis-u R,A,B (1,0,0)] [--basis-v R,A,B (0,1,0)] [--zoom0 F (1.0)] [--zoom1 F (4.0)] [--c0 F (0.0)] [--c1 F (0.0)]");
            std::process::exit(1);
        }
    }
}
