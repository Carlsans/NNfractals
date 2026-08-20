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
        // Report the measured reason rather than guessing. The old message
        // blamed the DD boundary or a degenerate neighbourhood; on a real
        // failing run both were wrong and the actual cause was one knob.
        None => match video_zoom_explore::dominant_rejection() {
            Some(why) => println!("video-zoom-explore: 0 winners in {} — {why}", out_dir.display()),
            None => println!(
                "video-zoom-explore: 0 winners in {} — every candidate was scored but none cleared the file-size floor (--min-file-size-ratio / --min-file-size-step-ratio), or the start view is past the DD boundary at --final-width {final_width}",
                out_dir.display()
            ),
        },
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
            std::process::exit(1);
        }
    }
}
