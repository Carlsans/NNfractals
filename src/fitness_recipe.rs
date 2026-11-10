//! Fitness recipes for the quaternion GA: a named `--fitness-metric` spec (weighted terms)
//! plus the scoring settings that belong with it (projection, taste-only knobs, view-range
//! scan, a suggested seed pool), saved as `recipes/<name>.json`.
//!
//! [`CATALOG`] describes every term `explorer quat-dag-evolve --fitness-metric` accepts:
//! what it measures, which shared computation ([`Pass`]) produces it, how much rendering
//! that costs, and how the term relates to Carl's own ratings. The launcher's recipe builder
//! is built on it, and [`Recipe::advice`] turns the lessons of past runs into warnings.
//! [`needs_render`] / [`uses_taste`] let both the launcher and the explorer leave out work a
//! recipe cannot use (the taste-model sidecar, the view-range scan, ...).

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const RECIPES_DIR: &str = "recipes";

// ── Catalog ─────────────────────────────────────────────────────────────────────

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Group { Picture, Motion, Sheet, Shape, Field, Planes, Structure, Taste }

impl Group {
    pub const ALL: [Group; 8] = [Group::Picture, Group::Motion, Group::Sheet, Group::Shape, Group::Field, Group::Planes, Group::Structure, Group::Taste];

    pub fn label(self) -> &'static str {
        match self {
            Group::Picture => "Picture — the rendered still",
            Group::Motion => "Motion — how the picture changes with T",
            Group::Sheet => "Sheet views — the orbiting camera",
            Group::Shape => "Shape probe — silhouette, shading and colour",
            Group::Field => "4D field — the whole object, no rendering",
            Group::Planes => "Axis planes — six flat slices, no rendering",
            Group::Structure => "Formula structure — free",
            Group::Taste => "Taste model — your trained SigLIP model",
        }
    }

    pub fn short(self) -> &'static str {
        match self {
            Group::Picture => "picture",
            Group::Motion => "motion",
            Group::Sheet => "sheet",
            Group::Shape => "shape",
            Group::Field => "4D field",
            Group::Planes => "planes",
            Group::Structure => "structure",
            Group::Taste => "taste",
        }
    }

    pub fn blurb(self) -> &'static str {
        match self {
            Group::Picture => "Computed on the same 300×400 'lava' still the browser thumbnail shows. These judge the picture as a whole: is there something to see, is it noise?",
            Group::Motion => "Compare stills at different T. A recipe without any motion term evolves shapes that do nothing when T moves.",
            Group::Sheet => "Six small views from the contact-sheet camera (3 orbit angles × 2 T). They catch shapes that only look good from the front.",
            Group::Shape => "Four probe renders (2 view angles × 2 C values) at the probe size. Classic shape statistics: coverage, symmetry, roughness, colour bands. Every value is in 0…1.",
            Group::Field => "Sample the quaternion function in 4D (R, A, B, C as equal coordinates): about how organised or chaotic the whole object is. CPU only.",
            Group::Planes => "Escape times on the six raw R-A-B-C axis planes (120×120, ±1.6). CPU only, ignores the projection and the view range.",
            Group::Structure => "Counted on the formula itself. Free, but they say nothing about how it looks.",
            Group::Taste => "Your SigLIP preference model, trained on your ⚖ ratings, looking at the still at several T moments (half mean, half worst).",
        }
    }
}

/// A computation shared by several terms (computed once per genome however many of its
/// terms the recipe uses).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Pass { Stills, FgStill, Resolution, MotionSteps, TCont, Sheet, Probe, Taste, Field, Planes, Free }

impl Pass {
    pub fn label(self) -> &'static str {
        match self {
            Pass::Stills => "4 stills 300×400 (T = 0, ¼, ½, ¾)",
            Pass::FgStill => "1 still 300×400",
            Pass::Resolution => "1 still 150×200 + 1 still 600×800",
            Pass::MotionSteps => "8 stills 300×400 at tiny T steps",
            Pass::TCont => "9 stills 150×200 (3 starts × tiny and big T step)",
            Pass::Sheet => "6 sheet views 112×112",
            Pass::Probe => "4 probe views (probe size²) + anisotropy sampling",
            Pass::Taste => "stills 300×400 (score moments) + SigLIP model",
            Pass::Field => "4D field sampling (CPU)",
            Pass::Planes => "6 axis planes 120×120 (CPU)",
            Pass::Free => "nothing (reads the formula)",
        }
    }

    /// Does this pass render with the 3D raymarcher (GPU when `--gpu`)?
    pub fn renders(self) -> bool {
        !matches!(self, Pass::Field | Pass::Planes | Pass::Free)
    }

    /// Rough rendered megapixels per genome (0 for the CPU passes).
    pub fn mpx(self, probe_size: u32, phases: u32) -> f64 {
        let m = |w: f64, h: f64, n: f64| w * h * n / 1e6;
        match self {
            Pass::Stills => m(300.0, 400.0, 4.0),
            Pass::FgStill => m(300.0, 400.0, 1.0),
            Pass::Resolution => m(150.0, 200.0, 1.0) + m(600.0, 800.0, 1.0),
            Pass::MotionSteps => m(300.0, 400.0, 8.0),
            Pass::TCont => m(150.0, 200.0, 9.0),
            Pass::Sheet => m(112.0, 112.0, 6.0),
            Pass::Probe => m(probe_size as f64, probe_size as f64, 4.0),
            Pass::Taste => m(300.0, 400.0, phases.max(1) as f64),
            Pass::Field | Pass::Planes | Pass::Free => 0.0,
        }
    }
}

/// One term `--fitness-metric` accepts.
pub struct TermInfo {
    pub name: &'static str,
    pub group: Group,
    pub pass: Pass,
    /// One line: what the number measures (all terms are about 0…1).
    pub what: &'static str,
    /// Longer notes: evidence from past runs, pitfalls.
    pub notes: &'static str,
    /// Share of Carl's rated pairs won by the genome with the HIGHER value, measured on
    /// the slice corpus (`train_corpus_quat`, 662 rated genomes) and the S3 corpus
    /// (`train_corpus_quat_s3`, 499 pairs), snapshot 2026-10-17/18. 0.5 = no relation.
    pub wins: (Option<f32>, Option<f32>),
    /// Weight given when the term is added (its sign follows the ratings).
    pub suggest: f64,
    /// Uses the cheap low-res gate (blank / noise / T-inert genomes get −5).
    pub gated: bool,
    /// Non-empty: known to be exploited, or diagnostic only.
    pub caution: &'static str,
}

const fn t(name: &'static str, group: Group, pass: Pass, what: &'static str, notes: &'static str, wins: (Option<f32>, Option<f32>), suggest: f64, caution: &'static str) -> TermInfo {
    TermInfo { name, group, pass, what, notes, wins, suggest, gated: false, caution }
}

const fn gated(mut x: TermInfo) -> TermInfo { x.gated = true; x }

use Group as G;
use Pass as P;

/// Every term, in the order the builder lists them.
pub const CATALOG: &[TermInfo] = &[
    // ── picture ──
    gated(t("image_entropy_band", G::Picture, P::Stills,
        "Is there a picture? Compression band of the still: blank scores 0, noise scores 0, mid complexity scores 1.",
        "Half deflate ratio (peak 0.11), half shuffled-pixel ratio (peak 0.5, the noise test), over 3 T moments. Every recipe that worked had it at about half of the total weight: it is the guard that stops metric sums from evolving blank balls, noise and empty images. Alone it evolves T-inert ring balls (entropy_only: 2.25/5, 0 of 4 moved).",
        (None, Some(0.61)), 3.5, "")),
    t("image_fg_shuffle", G::Picture, P::FgStill,
        "Noise test on the foreground only: near 1 for dotted noise, lower for structure.",
        "Separates dotted noise from structure (0.85–0.92 vs 0.65–0.81 in the 2026-10-17 tournament).",
        (None, Some(0.39)), -0.5, "Diagnostic only: several of your favourite speckled S3 genomes score 0.9–0.98 too. Never use it as a hard noise cut."),
    t("res_diff", G::Picture, P::Resolution,
        "Resolution stability: colour difference between a small render and a 4× larger one shrunk to the same size.",
        "A real surface agrees with itself at both sizes; sub-pixel fractal dust does not. Lower is more stable. Costly: one render is 600×800.",
        (None, Some(0.38)), -1.0, ""),
    t("res_iou", G::Picture, P::Resolution,
        "Resolution stability: overlap of the silhouette at the two resolutions (1 = identical).",
        "Same pair of renders as res_diff.",
        (None, Some(0.45)), 0.5, ""),
    // ── motion ──
    gated(t("t_cont_band", G::Motion, P::TCont,
        "Smooth motion: static scores 0, smooth motion scores up to 1 (peak at t_cont 0.25), jumpy or noisy motion falls back to 0.",
        "The band over t_cont. Moving and smooth won 62% of your slice pairs and 68% of your S3 pairs; all tournament noise was jumpy (t_cont ≥ 0.68). Replacing image_motion by it removed the dotted noise (0 of 8 winners) at ~30% more time.",
        (Some(0.62), Some(0.68)), 1.8, "")),
    gated(t("image_motion", G::Motion, P::Stills,
        "Does T change the picture at all? Largest frame-to-frame change over 4 T moments, full marks at 3%.",
        "Shares its 4 stills with image_entropy_band (no extra cost). Without any motion term every recipe collapsed to T-inert ring and lathe balls (2026-10-17 tournament).",
        (None, Some(0.47)), 1.8, "Also rewards flicker: about 1 in 4 winners was dotted noise. Prefer t_cont_band.")),
    t("t_cont", G::Motion, P::TCont,
        "Raw T-continuity ratio: picture change for a tiny T step ÷ change for a quarter-turn step (−1 when static).",
        "Small = smooth, near 1 = jumpy or chaotic. Raw, so a + weight rewards chaos; use t_cont_band instead unless you know why.",
        (Some(0.48), Some(0.33)), -0.5, ""),
    t("t_big", G::Motion, P::TCont,
        "How much a quarter-turn of T changes the picture.",
        "Shares its stills with t_cont_band.",
        (Some(0.49), Some(0.45)), 0.3, ""),
    t("motion_ratio", G::Motion, P::MotionSteps,
        "Organised motion: compressed size of the frame differences ÷ compressed size of the frames (8 tiny T steps).",
        "About 0 = static, about 1 = every step reshuffles the picture (chaotic). Costly: 8 full stills.",
        (None, Some(0.48)), -0.3, ""),
    t("motion_step", G::Motion, P::MotionSteps,
        "Mean picture change per tiny T step (8 stills).",
        "Same renders as motion_ratio.",
        (None, Some(0.45)), 0.3, ""),
    // ── sheet ──
    t("front_cover_band", G::Sheet, P::Sheet,
        "Lit from every side: full marks once at least 4% of each sheet view is lit.",
        "Stops 'empty crescents': solid surfaces whose visible side is dark from most angles (crescents 0.9–2.4% lit, good genomes 4.6–8.1%). Added in the second Optuna campaign after crescents won.",
        (None, None), 1.8, ""),
    t("front_cover", G::Sheet, P::Sheet,
        "Raw: smallest lit fraction over the 6 sheet views.",
        "Same views as front_cover_band; the band is the tested form.",
        (None, None), 1.0, ""),
    t("grey_band", G::Sheet, P::Sheet,
        "Not chrome: full marks up to 30% grey among the lit pixels, zero at 70%.",
        "Once dark fronts were penalised the GA drifted to chrome/silver balls (grey 0.75–0.92). The less grey genome wins 57% of your slice pairs and 64% of your S3 pairs.",
        (Some(0.57), Some(0.64)), 1.8, ""),
    t("grey_share", G::Sheet, P::Sheet,
        "Raw: mean share of grey (saturation < 0.25) among the lit pixels of the sheet views.",
        "Same views as grey_band.",
        (Some(0.43), Some(0.36)), -1.0, ""),
    // ── shape probe ──
    t("silhouette_irregularity", G::Shape, P::Probe, "How much the outline's radius varies with angle: 0 = smooth ball, high = protrusions, notches, concavities.", "Your strongest single slice signal.", (Some(0.67), Some(0.56)), 0.3, ""),
    t("sphericity", G::Field, P::Field, "How round the C = 0 slice is (1 = a perfect sphere).", "Computed with the 4D field metrics (CPU). Your most consistent dislike: the rounder genome loses 63% of slice pairs and 67% of S3 pairs.", (Some(0.37), Some(0.33)), -0.4, ""),
    t("bilateral_symmetry", G::Shape, P::Probe, "Mirror symmetry of the silhouette about its own centre (1 = symmetric both ways).", "You prefer less symmetric shapes, strongly in S3.", (Some(0.42), Some(0.36)), -0.37, ""),
    t("box_dim", G::Shape, P::Probe, "Box-counting dimension of the outline: rough at every scale.", "", (Some(0.60), Some(0.59)), 0.31, ""),
    t("lacunarity", G::Shape, P::Probe, "Gappiness of the texture: low = dense and uniform, high = porous.", "", (Some(0.61), Some(0.54)), 0.25, ""),
    t("isoperimetric", G::Shape, P::Probe, "Outline length relative to area: 0 = round like a disc, high = elongated or convoluted.", "", (Some(0.61), Some(0.59)), 0.25, ""),
    t("convexity", G::Shape, P::Probe, "Area ÷ convex-hull area: 1 = convex blob, low = deep bites, crescents.", "", (Some(0.42), Some(0.46)), -0.2, ""),
    t("coverage", G::Shape, P::Probe, "Share of the probe view covered by the surface (scored up to a 50% fill).", "", (Some(0.45), Some(0.46)), -0.26, ""),
    t("solidity", G::Shape, P::Probe, "Share of surface pixels surrounded by surface: solid volume (1) vs scattered filaments.", "You prefer less solid, more open shapes.", (Some(0.44), Some(0.41)), -0.34, ""),
    t("largest_component_frac", G::Shape, P::Probe, "Share of the surface in its largest connected piece (1 = one coherent mass).", "", (Some(0.46), Some(0.47)), -0.1, ""),
    t("centroid_offset", G::Shape, P::Probe, "How far the mass sits from the centre of its bounding box (lopsidedness).", "Opposite signs: lopsided wins in slice, centred wins in S3.", (Some(0.60), Some(0.44)), 0.2, ""),
    t("anisotropy", G::Shape, P::Probe, "How different the formula looks from different directions (0 = identical from every angle).", "Necessary, not sufficient, for a non-spherical silhouette.", (Some(0.52), Some(0.41)), 0.1, ""),
    t("cross_view_iou", G::Shape, P::Probe, "Overlap of the silhouettes seen from two view angles (1 = the same from both).", "", (Some(0.44), Some(0.46)), -0.27, ""),
    t("cross_view_coverage_delta", G::Shape, P::Probe, "Difference in coverage between two view angles.", "", (Some(0.60), Some(0.56)), 0.29, ""),
    t("c_sensitivity", G::Shape, P::Probe, "How much the silhouette changes between two C values (1 − overlap).", "A cheap motion hint, but on a single C step.", (Some(0.56), Some(0.61)), 0.2, ""),
    t("c_coverage_range", G::Shape, P::Probe, "Coverage change between two C values.", "", (Some(0.61), Some(0.61)), 0.2, ""),
    t("shading_richness", G::Shape, P::Probe, "Lighting variation on the solid interior: curvature detail.", "", (Some(0.56), Some(0.48)), 0.1, ""),
    t("shading_gradient", G::Shape, P::Probe, "Local shading gradient: fine surface roughness.", "Opposite signs in slice and S3.", (Some(0.56), Some(0.43)), 0.1, ""),
    t("shading_skewness", G::Shape, P::Probe, "Light/dark balance: low = bright with dark crevices, high = dark with highlights (0.5 = balanced).", "", (Some(0.58), Some(0.50)), 0.1, ""),
    t("specular_fraction", G::Shape, P::Probe, "Share of interior pixels near full light.", "", (Some(0.55), Some(0.44)), 0.1, ""),
    t("crevice_fraction", G::Shape, P::Probe, "Share of interior pixels near the ambient floor (dark crevices).", "", (Some(0.49), Some(0.45)), -0.1, ""),
    t("color_entropy", G::Shape, P::Probe, "How evenly the escape-time colour bands are used over the surface.", "Opposite signs: more colours win in slice, fewer in S3.", (Some(0.62), Some(0.44)), 0.25, ""),
    t("color_gradient", G::Shape, P::Probe, "Spatial colour change: busy colour vs smooth colour fields.", "You prefer smoother colour in S3.", (Some(0.55), Some(0.37)), -0.36, ""),
    t("color_shading_corr", G::Shape, P::Probe, "Do the colour bands follow the surface curvature? (0.5 = unrelated).", "", (Some(0.57), Some(0.60)), 0.25, ""),
    t("color_band_autocorr", G::Shape, P::Probe, "Regular fine colour banding.", "", (Some(0.62), Some(0.54)), 0.24, ""),
    t("color_range_utilization", G::Shape, P::Probe, "Span of escape-time colours used, relative to the iteration limit.", "", (Some(0.57), Some(0.54)), 0.1, ""),
    // ── 4D field ──
    t("organization_continuity", G::Field, P::Field, "Smoothness of the 4D escape field (1 − discontinuity).", "Opposite signs: smooth wins 67% of S3 pairs but loses 60% of slice pairs.", (Some(0.40), Some(0.67)), 0.34, ""),
    t("organization_chaoticity", G::Field, P::Field, "Jumpiness of the 4D escape field along the axes.", "You prefer less chaotic fields in S3 (35%).", (Some(0.54), Some(0.35)), -0.36, ""),
    t("organization_entropy", G::Field, P::Field, "Entropy of the 4D escape-time histogram.", "Opposite signs in slice and S3.", (Some(0.59), Some(0.41)), -0.31, ""),
    t("organization_richness", G::Field, P::Field, "Entropy × continuity: 'high entropy, but continuous'.", "", (Some(0.60), Some(0.57)), 0.3, ""),
    t("organization_statistical", G::Field, P::Field, "Statistical complexity (entropy × disequilibrium): peaks between order and chaos.", "", (Some(0.54), Some(0.51)), 0.2, ""),
    t("organization_ordinal", G::Field, P::Field, "Ordinal-pattern complexity of the field.", "", (Some(0.55), Some(0.55)), 0.2, ""),
    t("organization_compression", G::Field, P::Field, "How badly the 4D field compresses.", "", (Some(0.58), Some(0.43)), 0.1, "Monotonic: maximising it seeks noise. Use organization_compression_capped to aim for the middle."),
    t("organization_compression_capped", G::Field, P::Field, "organization_compression with a ceiling at 0.4; going past it costs five times the overshoot.", "From your remark that the middle individuals look better than the most complex ones.", (None, None), 0.3, ""),
    t("organization_multifractal", G::Field, P::Field, "Width of the multifractal spectrum of the field.", "Showed no variation at all in the rated corpora.", (None, None), 0.1, "No measurable signal so far."),
    // ── planes ──
    t("planes_smooth", G::Planes, P::Planes, "Smooth escape fields on the axis planes: short boundaries and little fine detail.", "Matches your preference for smooth fields (wins 66% of S3 pairs).", (None, Some(0.66)), 1.8, "Slice tournament 2026-10-18: no gain, and exploited by a smooth empty ball. Pair it with picture terms."),
    t("planes_open", G::Planes, P::Planes, "A small bounded set on the axis planes (1 − bounded fraction / 0.25).", "", (None, Some(0.59)), 1.0, "Exploited by an empty ball when used without picture terms."),
    t("planes_asym", G::Planes, P::Planes, "How different the RA, RB and RC planes are: 0 for the ring/lathe-ball symmetry.", "Asymmetry only mattered in slice.", (None, Some(0.51)), 0.8, ""),
    t("planes_raw_boundary", G::Planes, P::Planes, "Raw: boundary pixels of the bounded set ÷ all pixels.", "", (None, Some(0.35)), -0.5, ""),
    t("planes_raw_bounded", G::Planes, P::Planes, "Raw: bounded fraction of the plane pixels.", "", (None, Some(0.41)), -0.5, ""),
    t("planes_raw_detail", G::Planes, P::Planes, "Raw: mean |Laplacian| of the normalised log escape time (fine detail).", "", (None, Some(0.34)), -0.5, ""),
    t("planes_raw_asym", G::Planes, P::Planes, "Raw: mean difference between the RA, RB and RC planes.", "", (None, None), 0.5, ""),
    // ── structure ──
    t("node_count", G::Structure, P::Free, "Formula size (nodes ÷ 24).", "", (Some(0.56), Some(0.53)), 0.1, ""),
    t("max_depth", G::Structure, P::Free, "Longest live chain of operations (÷ 24).", "", (Some(0.57), Some(0.49)), 0.1, ""),
    t("opcode_diversity", G::Structure, P::Free, "Distinct operations ÷ nodes.", "", (Some(0.48), Some(0.46)), 0.1, ""),
    t("warp_node_count", G::Structure, P::Free, "Size of the warp program (bends the input before iterating).", "", (Some(0.49), Some(0.53)), 0.1, ""),
    t("warp_opcode_diversity", G::Structure, P::Free, "Distinct operations in the warp program.", "", (Some(0.49), Some(0.58)), 0.1, ""),
    // ── taste ──
    t("taste", G::Taste, P::Taste, "Your SigLIP taste model's score of the still at several T moments (half mean, half worst).",
        "Drives the taste rating loop. With look diversity (embed diversity 1.0–1.5) a run gave 60 clearly different looks. Starts the Python sidecar (~1.5 GB RAM plus GPU memory), and scoring runs one genome at a time. S3 projections use taste_model_quat_s3 when it exists.",
        (None, Some(0.72)), 1.0, "Generous to fine grain. Its held-out accuracy fell from 89% to 75% as your ratings grew; the S3 number is in-sample."),
];

pub fn info(name: &str) -> Option<&'static TermInfo> { CATALOG.iter().find(|t| t.name == name) }

/// True when `name` needs the 3D renderer (unknown names count as rendering, to be safe).
pub fn term_renders(name: &str) -> bool { info(name).is_none_or(|t| t.pass.renders()) }

/// Does any term of a parsed spec render a picture? With none, the view range never
/// affects the score, so the view-range scan is wasted work.
pub fn needs_render(spec: &[(String, f64)]) -> bool { spec.iter().any(|(n, w)| *w != 0.0 && term_renders(n)) }

/// Does the spec use the taste model (the SigLIP sidecar)?
pub fn uses_taste(spec: &[(String, f64)]) -> bool { spec.iter().any(|(n, w)| *w != 0.0 && n == "taste") }

/// `name:weight,...` → terms, same rules as `explorer --fitness-metric` (a bare name has
/// weight 1). Unknown names are an error here (the explorer would panic on them).
pub fn parse_spec(s: &str) -> Result<Vec<Term>, String> {
    let mut out = Vec::new();
    let mut unknown = Vec::new();
    for part in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        let (name, w) = match part.split_once(':') {
            Some((n, w)) => (n.trim(), w.trim().parse::<f64>().map_err(|_| format!("bad weight in {part:?} (expected name:number)"))?),
            None => (part, 1.0),
        };
        if info(name).is_none() { unknown.push(name.to_string()); }
        out.push(Term { name: name.to_string(), weight: w });
    }
    if !unknown.is_empty() { return Err(format!("unknown term(s): {}", unknown.join(", "))); }
    Ok(out)
}

/// Same terms and weights (to 4 decimals), in any order.
pub fn same_spec(a: &str, b: &str) -> bool {
    let norm = |s: &str| -> Option<Vec<(String, i64)>> {
        let mut v: Vec<(String, i64)> = parse_spec(s).ok()?.into_iter().filter(|t| t.weight != 0.0).map(|t| (t.name, (t.weight * 1e4).round() as i64)).collect();
        v.sort();
        Some(v)
    };
    match (norm(a), norm(b)) { (Some(x), Some(y)) => x == y, _ => a.trim() == b.trim() }
}

/// Same term names with every weight within `tol` (for specs read back from a
/// run log, which rounds weights to two decimals).
pub fn spec_close(a: &str, b: &str, tol: f64) -> bool {
    let (Ok(mut x), Ok(mut y)) = (parse_spec(a), parse_spec(b)) else { return false };
    x.retain(|t| t.weight != 0.0);
    y.retain(|t| t.weight != 0.0);
    x.sort_by(|p, q| p.name.cmp(&q.name));
    y.sort_by(|p, q| p.name.cmp(&q.name));
    x.len() == y.len() && x.iter().zip(&y).all(|(p, q)| p.name == q.name && (p.weight - q.weight).abs() <= tol)
}

/// The spec a run used, from the explorer's `fitness method: single/combo metric
/// — a:1.00, b:-0.37` log line (weights rounded to 2 decimals).
pub fn spec_from_log(log: &str) -> Option<String> {
    let line = log.lines().find_map(|l| l.split_once("fitness method: single/combo metric — ").map(|x| x.1))?;
    let spec = line.split(", ").map(str::trim).collect::<Vec<_>>().join(",");
    parse_spec(&spec).ok().filter(|t| !t.is_empty()).map(|_| spec)
}

// ── Recipe ──────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Term { pub name: String, pub weight: f64 }

fn d_collapse() -> String { "slice".into() }
fn d_phases() -> u32 { 3 }

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Recipe {
    pub name: String,
    /// What the recipe is for, in a sentence or two.
    #[serde(default)]
    pub description: String,
    /// Where it comes from and how it did.
    #[serde(default)]
    pub evidence: String,
    pub terms: Vec<Term>,
    /// 4D→3D projection the weights were made for ("slice", "s3").
    #[serde(default = "d_collapse")]
    pub collapse: String,
    /// T moments the taste term scores (taste only).
    #[serde(default = "d_phases")]
    pub eval_phases: u32,
    /// Flash penalty between T moments (taste only).
    #[serde(default)]
    pub smooth_weight: f32,
    /// Keep survivors looking different, by taste-model embedding (taste only).
    #[serde(default)]
    pub embed_diversity: f32,
    /// Scan each new genome for its best view range (uses the taste model to choose).
    #[serde(default)]
    pub auto_range: bool,
    /// Suggested seed pool; empty = keep the current one.
    #[serde(default)]
    pub pool_dir: String,
    /// Shipped with the program (read-only; save a copy to change it).
    #[serde(skip)]
    pub builtin: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level { Error, Warn, Info }

#[derive(Clone, Debug)]
pub struct Advice { pub level: Level, pub text: String }

/// One part of the program a run may start, and whether this recipe starts it.
#[derive(Clone, Debug)]
pub struct Module { pub name: &'static str, pub on: bool, pub why: String }

/// What a run looks like beyond the recipe itself.
#[derive(Clone, Copy, Debug)]
pub struct RunContext { pub gpu: bool, pub map_elites: bool, pub probe_size: u32, pub workers: usize }

impl Recipe {
    pub fn empty(name: &str) -> Recipe {
        Recipe { name: name.into(), description: String::new(), evidence: String::new(), terms: Vec::new(), collapse: d_collapse(), eval_phases: 3, smooth_weight: 0.0, embed_diversity: 0.0, auto_range: false, pool_dir: String::new(), builtin: false }
    }

    /// The `--fitness-metric` value (zero-weight terms left out).
    pub fn spec(&self) -> String {
        self.terms.iter().filter(|t| t.weight != 0.0).map(|t| {
            let w = format!("{:.4}", t.weight);
            let w = w.trim_end_matches('0').trim_end_matches('.');
            format!("{}:{}", t.name, if w.is_empty() || w == "-" { "0" } else { w })
        }).collect::<Vec<_>>().join(",")
    }

    pub fn pairs(&self) -> Vec<(String, f64)> { self.terms.iter().map(|t| (t.name.clone(), t.weight)).collect() }
    pub fn has(&self, name: &str) -> bool { self.terms.iter().any(|t| t.name == name && t.weight != 0.0) }
    pub fn uses_taste(&self) -> bool { uses_taste(&self.pairs()) }
    pub fn needs_render(&self) -> bool { needs_render(&self.pairs()) }
    pub fn gated(&self) -> bool { self.terms.iter().any(|t| t.weight != 0.0 && info(&t.name).is_some_and(|i| i.gated)) }

    /// No term with a weight: a run then scores without a recipe.
    pub fn is_empty(&self) -> bool { self.terms.iter().all(|t| t.weight == 0.0) }

    /// The view-range scan only runs when it can change a score (a run without
    /// a recipe renders, so it scans too).
    pub fn effective_auto_range(&self) -> bool { self.auto_range && (self.needs_render() || self.is_empty()) }

    /// Without a recipe, MAP-Elites and every projection but slice score with
    /// the taste model alone (the launcher adds `--fitness-metric taste`).
    fn implicit_taste(&self, ctx: RunContext) -> bool { self.is_empty() && (ctx.map_elites || self.collapse != "slice") }

    /// Lowest and highest total a genome can get (every term is about 0…1; a gated
    /// recipe gives −5 to genomes failing the gate).
    pub fn score_range(&self) -> (f64, f64) {
        let lo: f64 = self.terms.iter().map(|t| t.weight.min(0.0)).sum();
        let hi: f64 = self.terms.iter().map(|t| t.weight.max(0.0)).sum();
        (if self.gated() { lo.min(-5.0) } else { lo }, hi)
    }

    /// Each term's share of the total |weight|.
    pub fn shares(&self) -> Vec<f64> {
        let total: f64 = self.terms.iter().map(|t| t.weight.abs()).sum();
        self.terms.iter().map(|t| if total > 0.0 { t.weight.abs() / total } else { 0.0 }).collect()
    }

    /// Distinct computations per genome and their rough rendered megapixels.
    pub fn passes(&self, probe_size: u32) -> Vec<(Pass, f64, Vec<&str>)> {
        let mut v: Vec<(Pass, f64, Vec<&str>)> = Vec::new();
        for term in self.terms.iter().filter(|t| t.weight != 0.0) {
            let Some(i) = info(&term.name) else { continue };
            match v.iter_mut().find(|(p, _, _)| *p == i.pass) {
                Some(e) => e.2.push(i.name),
                None => v.push((i.pass, i.pass.mpx(probe_size, self.eval_phases), vec![i.name])),
            }
        }
        // image_motion shares the four stills of image_entropy_band (one cache).
        v.sort_by_key(|(p, _, _)| *p);
        v
    }

    pub fn render_mpx(&self, probe_size: u32) -> f64 { self.passes(probe_size).iter().map(|x| x.1).sum() }

    /// Which parts of the program a run with this recipe starts (matching what
    /// the launcher's `build_quat_evolve_args` actually launches).
    pub fn modules(&self, ctx: RunContext) -> Vec<Module> {
        let empty = self.is_empty();
        let implicit = self.implicit_taste(ctx);
        let blend = empty && !implicit;
        let taste_term = self.uses_taste() || implicit;
        let renders = self.needs_render() || empty;
        let scan = self.effective_auto_range();
        let names = |pass: &[Pass]| -> String {
            self.terms.iter().filter(|t| t.weight != 0.0 && info(&t.name).is_some_and(|i| pass.contains(&i.pass))).map(|t| t.name.as_str()).collect::<Vec<_>>().join(", ")
        };
        let mut m = Vec::new();
        let taste_why = if self.uses_taste() { "the taste term".to_string() }
            else if implicit { "no recipe: MAP-Elites, or a projection other than slice, scores with the taste model alone".to_string() }
            else if scan { "auto view range: it picks each new genome's framing with the taste model".to_string() }
            else if self.auto_range && !renders { "not started: auto view range is skipped because no term renders".to_string() }
            else { "not started: no taste term and auto view range off".to_string() };
        m.push(Module { name: "Taste model (Python SigLIP sidecar, ~1.5 GB RAM + GPU memory)", on: taste_term || scan, why: taste_why });
        m.push(Module {
            name: "3D renderer for scoring",
            on: renders,
            why: if self.needs_render() { format!("{} ({:.2} Mpx per genome{})", if ctx.gpu { "GPU raymarcher" } else { "CPU raymarcher — tick GPU, it is much faster" }, self.render_mpx(ctx.probe_size), if self.gated() { ", less for genomes failing the gate" } else { "" }) }
                 else if empty { "the run without a recipe renders".into() }
                 else { "not used for scoring: no term renders (thumbnails of the saved genomes still render)".into() },
        });
        m.push(Module { name: "Low-res image gate (blank / noise / T-inert → −5)", on: self.gated(), why: if self.gated() { format!("because of {}", self.terms.iter().filter(|t| t.weight != 0.0 && info(&t.name).is_some_and(|i| i.gated)).map(|t| t.name.as_str()).collect::<Vec<_>>().join(", ")) } else { "no gated term (image_entropy_band, image_motion, t_cont_band)".into() } });
        let probe = names(&[Pass::Probe]);
        m.push(Module { name: "Shape probe sweep", on: !probe.is_empty() || blend, why: if blend { "the default blend's geometric score".into() } else if probe.is_empty() { "no shape term".into() } else { format!("{probe} (probe size {})", ctx.probe_size) } });
        let field = names(&[Pass::Field]);
        m.push(Module { name: "4D field sampler (CPU)", on: !field.is_empty(), why: if field.is_empty() { "no 4D field term".into() } else { field } });
        let planes = names(&[Pass::Planes]);
        m.push(Module { name: "Axis-plane slices (CPU)", on: !planes.is_empty(), why: if planes.is_empty() { "no plane term".into() } else { planes } });
        m.push(Module { name: "Old preference model (pref_model_quat.json)", on: blend, why: if blend { "the default blend (no recipe, slice, not MAP-Elites) uses it".into() } else { "not loaded: only the default blend uses it".into() } });
        m.push(Module { name: "2D fractal renderer", on: false, why: "never used by the quaternion GA, not started".into() });
        m.push(Module {
            name: "Parallel scoring",
            on: !ctx.map_elites && !empty && !taste_term && ctx.workers > 1,
            why: if ctx.map_elites { "off: MAP-Elites scores one genome at a time".into() }
                 else if blend { "off: the default blend scores one genome at a time".into() }
                 else if taste_term { "off: the taste sidecar scores one genome at a time".into() }
                 else if ctx.workers > 1 { format!("{} workers", ctx.workers) } else { "1 worker".into() },
        });
        m
    }

    /// Warnings and hints, from the lessons of past runs.
    pub fn advice(&self, ctx: RunContext) -> Vec<Advice> {
        let mut a = Vec::new();
        let mut add = |level, text: String| a.push(Advice { level, text });
        let live: Vec<&Term> = self.terms.iter().filter(|t| t.weight != 0.0).collect();
        if live.is_empty() {
            let without = if self.implicit_taste(ctx) { "the taste model alone (MAP-Elites, or a projection other than slice)" } else { "the old default blend (geometry + the old preference model)" };
            add(Level::Error, format!("No term yet. Add terms from the list on the right, or start from a built-in recipe. Without terms a run scores with {without}, not a recipe."));
            return a;
        }
        for t in &live {
            if info(&t.name).is_none() { add(Level::Error, format!("{} is not a term the GA knows; the run would stop at once.", t.name)); }
        }
        let mut seen = std::collections::HashSet::new();
        for t in &live {
            if !seen.insert(t.name.as_str()) { add(Level::Warn, format!("{} appears twice: both weights count.", t.name)); }
        }
        let total: f64 = live.iter().map(|t| t.weight.abs()).sum();
        let group_of = |t: &&Term| info(&t.name).map(|i| i.group);
        let has_group = |g: Group| live.iter().any(|t| group_of(t) == Some(g));
        let taste = self.has("taste");
        let guard = self.has("image_entropy_band") || taste;
        let renders = self.needs_render();
        if !renders {
            add(Level::Warn, "No term looks at a rendered picture. Fast, but 4D-field terms alone are maxed by a blank or solid ball and plane terms by a smooth empty ball (slice tournament). Add image_entropy_band as a guard.".into());
        } else if !guard {
            add(Level::Warn, "No guard term. Pure metric sums were exploited by blank balls, noise, sparse dots and blank images (S3 runs, 2026-10-16). Every recipe that worked had image_entropy_band (or taste) carrying a large share of the weight.".into());
        } else if self.has("image_entropy_band") && !taste {
            let w = live.iter().filter(|t| t.name == "image_entropy_band").map(|t| t.weight).sum::<f64>();
            if w <= 0.0 { add(Level::Warn, "image_entropy_band has a negative weight: that rewards blank images and noise.".into()); }
            else if w / total < 0.2 { add(Level::Info, format!("image_entropy_band is only {:.0}% of the weight. In the recipes that worked it was 29–50%.", 100.0 * w / total)); }
        }
        let motion = live.iter().any(|t| t.weight > 0.0 && matches!(t.name.as_str(), "t_cont_band" | "image_motion" | "t_big" | "motion_step" | "c_sensitivity" | "c_coverage_range"));
        if renders && !motion && !taste {
            add(Level::Warn, "Nothing rewards motion along T. Every recipe without a motion term evolved shapes that do nothing when T moves (ring and lathe balls; 2026-10-17 tournament). Add t_cont_band.".into());
        }
        if self.has("image_motion") && !self.has("t_cont_band") {
            add(Level::Info, "image_motion also rewards flicker: about 1 in 4 winners was dotted noise. t_cont_band rewards smooth motion and removed it (0 of 8).".into());
        }
        if (has_group(Group::Picture) || has_group(Group::Shape)) && !self.has("front_cover_band") && !self.has("grey_band") && !taste {
            add(Level::Info, "The slice campaigns drifted to dark 'empty crescents', then to chrome balls, until front_cover_band and grey_band were added.".into());
        }
        if self.has("organization_compression") && live.iter().any(|t| t.name == "organization_compression" && t.weight > 0.0) {
            add(Level::Warn, "organization_compression with + weight seeks noise (it never stops rising). Use organization_compression_capped.".into());
        }
        for t in &live {
            if let Some(i) = info(&t.name) {
                if !i.caution.is_empty() && !matches!(t.name.as_str(), "image_motion" | "organization_compression") {
                    add(Level::Info, format!("{}: {}", t.name, i.caution));
                }
            }
        }
        // Sign against Carl's ratings for this projection.
        let s3 = self.collapse == "s3";
        for t in &live {
            let Some(i) = info(&t.name) else { continue };
            let win = if s3 { i.wins.1 } else { i.wins.0 };
            if let Some(w) = win {
                let corpus = if s3 { "S3" } else { "slice" };
                if t.weight > 0.0 && w < 0.42 {
                    add(Level::Info, format!("{} has a + weight, but in your {corpus} ratings the LOWER value wins ({:.0}% of pairs go to the higher one).", t.name, w * 100.0));
                } else if t.weight < 0.0 && w > 0.58 {
                    add(Level::Info, format!("{} has a − weight, but in your {corpus} ratings the HIGHER value wins {:.0}% of pairs.", t.name, w * 100.0));
                }
            }
        }
        if !taste && self.embed_diversity > 0.0 {
            add(Level::Info, "Look diversity uses the taste model's embeddings; without a taste term it does nothing, so it is not passed to the run.".into());
        }
        if !taste && self.smooth_weight > 0.0 {
            add(Level::Info, "The flash penalty only applies to the taste term; it is not passed to the run.".into());
        }
        if self.auto_range && !renders {
            add(Level::Info, "Auto view range only changes how far out the pictures are taken, and no term renders: it is skipped.".into());
        } else if self.auto_range && !taste {
            add(Level::Info, "Auto view range starts the taste model just to frame each new genome. The recipes that worked ran without it.".into());
        }
        if taste && ctx.workers > 1 {
            add(Level::Info, "The taste term scores one genome at a time; more workers do not speed this recipe up.".into());
        }
        if renders && !ctx.gpu {
            add(Level::Warn, format!("GPU is off: about {:.1} Mpx per genome will render on the CPU.", self.render_mpx(ctx.probe_size)));
        }
        a.sort_by_key(|x| x.level);
        a
    }

    // ── files ──

    pub fn path(dir: &Path, name: &str) -> PathBuf { dir.join(format!("{}.json", crate::tuning::sanitize(name))) }

    pub fn save(&self, root: &Path) -> std::io::Result<PathBuf> {
        let dir = root.join(RECIPES_DIR);
        std::fs::create_dir_all(&dir)?;
        let p = Self::path(&dir, &self.name);
        std::fs::write(&p, serde_json::to_string_pretty(self).map_err(std::io::Error::other)?)?;
        Ok(p)
    }

    pub fn load_file(path: &Path) -> Option<Recipe> {
        serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()
    }

    /// Saved recipes (file names without `.json`), sorted.
    pub fn list(root: &Path) -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(root.join(RECIPES_DIR)).into_iter().flatten().flatten()
            .filter_map(|e| { let p = e.path(); (p.extension()? == "json").then(|| p.file_stem()?.to_str().map(String::from))? })
            .collect();
        v.sort();
        v
    }

    /// A built-in recipe by name, else a saved one.
    pub fn find(root: &Path, name: &str) -> Option<Recipe> {
        builtins().into_iter().find(|r| r.name == name)
            .or_else(|| Self::load_file(&Self::path(&root.join(RECIPES_DIR), name)))
    }
}

// ── Built-in recipes (ones that worked) ─────────────────────────────────────────

/// The 11 metrics of H1_s3_rho25 (|ρ| ≥ 0.25 against Carl's S3 wins) with the weights
/// `taste_analysis/combo_specs.json` holds, rounded as the campaign scripts round them.
const H1_METRICS: [(&str, f64); 11] = [
    ("sphericity", -0.3906), ("organization_continuity", 0.3388), ("organization_chaoticity", -0.3557),
    ("bilateral_symmetry", -0.3686), ("color_gradient", -0.3582), ("box_dim", 0.3093), ("solidity", -0.3366),
    ("organization_entropy", -0.3127), ("cross_view_coverage_delta", 0.2948), ("cross_view_iou", -0.266), ("coverage", -0.2561),
];
const H1_ENTROPY: f64 = 3.5875;
const H1_HALF: f64 = 1.7938;

fn terms(list: &[(&str, f64)]) -> Vec<Term> { list.iter().map(|(n, w)| Term { name: n.to_string(), weight: *w }).collect() }

fn h1_with(extra: &[(&str, f64)]) -> Vec<Term> {
    let mut v = terms(&[("image_entropy_band", H1_ENTROPY)]);
    v.extend(terms(&H1_METRICS));
    v.extend(terms(extra));
    v
}

pub fn builtins() -> Vec<Recipe> {
    let base = |name: &str, description: &str, evidence: &str, terms: Vec<Term>, collapse: &str| Recipe {
        name: name.into(), description: description.into(), evidence: evidence.into(), terms, collapse: collapse.into(),
        builtin: true, ..Recipe::empty(name)
    };
    let mut v = vec![
        Recipe {
            pool_dir: "fractals_dag_mixed_kbc_seed".into(),
            ..base("★ Slice campaign: H1 smooth + lit + not grey",
                "Half 'is there a picture' (image_entropy_band), half the 11 metrics that tracked your S3 picks, plus smooth motion (t_cont_band), lit from every side (front_cover_band) and not chrome (grey_band). Slice projection.",
                "The Optuna campaigns of 2026-10-18/19 (scripts/optuna_campaigns.py, campaign 3 on; pop 32 × 10 gens, stop when flat 6, pool fractals_dag_mixed_kbc_seed). You: \"those results are the very best in a while\". Your GUI campaigns Carl_Test (355 runs) and wkend1 used the same terms.",
                h1_with(&[("t_cont_band", H1_HALF), ("front_cover_band", H1_HALF), ("grey_band", H1_HALF)]), "slice")
        },
        base("★ S3 H1: picture + motion",
            "Half image_entropy_band, half the 11 metrics that tracked your S3 picks, plus image_motion so T must change the picture. S3 projection.",
            "Won the S3 recipe tournament of 2026-10-17 (best of 7: 3.1/5 by eye, 3 of 4 winners move with T). The endless S3 loop ran on it; you: \"they are all really good\". Weak spot: about 1 in 4 winners is dotted noise.",
            h1_with(&[("image_motion", H1_HALF)]), "s3"),
        Recipe {
            pool_dir: "fractals_dag_bulb_seed".into(),
            ..base("★ S3 H1 smooth: bulb seeds",
                "The S3 H1 recipe with t_cont_band (smooth motion) instead of image_motion, seeded from the bulb pool to break the ring-ball symmetry.",
                "Round 3 of the S3 tournament (2026-10-18): 0 of 8 winners dotted noise, all move smoothly, ~30% slower than H1. Seeded from fractals_dag_bulb_seed: 8 of 8 kept the bulb, 5 of 8 broke the one-axis ring symmetry (lobes, marble, blob cloud), and it gave the best piece so far.",
                h1_with(&[("t_cont_band", H1_HALF)]), "s3")
        },
        Recipe {
            embed_diversity: 1.0,
            ..base("★ Taste model: distinct looks",
                "Your own SigLIP taste model, with survivors kept visually different by its embeddings (look diversity 1.0).",
                "Taste rating loop, runs r14 and r15 (2026-10-15; pop 200, survivors 40–60, immigrants 0.15, crossover 0.45, save top 60, stop when flat 25): 60 clearly different looks per run, elite mean 0.98. Only as good as the model: its held-out accuracy fell from 89% to 75% as your ratings grew.",
                terms(&[("taste", 1.0)]), "slice")
        },
    ];
    for r in &mut v { r.builtin = true; }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> RunContext { RunContext { gpu: true, map_elites: false, probe_size: 128, workers: 8 } }

    #[test]
    fn catalog_names_are_unique_and_every_builtin_parses() {
        let mut names: Vec<&str> = CATALOG.iter().map(|t| t.name).collect();
        names.sort();
        let n = names.len();
        names.dedup();
        assert_eq!(n, names.len(), "duplicate catalog entry");
        for r in builtins() {
            let parsed = parse_spec(&r.spec()).unwrap_or_else(|e| panic!("{}: {e}", r.name));
            assert_eq!(parsed.len(), r.terms.len());
            assert!(r.advice(ctx()).iter().all(|a| a.level != Level::Error), "{} has errors", r.name);
        }
    }

    #[test]
    fn the_campaign_recipe_matches_the_campaign_script() {
        // What Carl_Test's runs logged (rounded to 2 decimals by the explorer's banner).
        let r = &builtins()[0];
        let want = [("image_entropy_band", 3.59), ("t_cont_band", 1.79), ("front_cover_band", 1.79), ("grey_band", 1.79), ("sphericity", -0.39), ("coverage", -0.26)];
        for (n, w) in want {
            let got = r.terms.iter().find(|t| t.name == n).unwrap().weight;
            assert!((got - w).abs() < 0.006, "{n}: {got} vs {w}");
        }
        assert_eq!(r.terms.len(), 15);
        assert!(!r.uses_taste() && !r.effective_auto_range());
    }

    #[test]
    fn spec_round_trips_and_compares_in_any_order() {
        let r = &builtins()[1];
        let back = parse_spec(&r.spec()).unwrap();
        for (a, b) in r.terms.iter().zip(&back) { assert!((a.weight - b.weight).abs() < 1e-4); }
        assert!(same_spec("taste:1,grey_band:0.5", "grey_band:0.50, taste"));
        assert!(!same_spec("taste:1", "taste:2"));
        assert!(parse_spec("taste:x").is_err());
        assert!(parse_spec("nonsense:1").unwrap_err().contains("nonsense"));
        let mut e = Recipe::empty("e");
        e.terms.push(Term { name: "grey_band".into(), weight: -0.5 });
        e.terms.push(Term { name: "taste".into(), weight: 0.0 });
        assert_eq!(e.spec(), "grey_band:-0.5", "zero weights are left out");
    }

    #[test]
    fn a_logged_spec_is_matched_back_to_its_recipe() {
        let log = "seed: 1\nfitness method: single/combo metric — image_entropy_band:3.59, sphericity:-0.39, organization_continuity:0.34, organization_chaoticity:-0.36, bilateral_symmetry:-0.37, color_gradient:-0.36, box_dim:0.31, solidity:-0.34, organization_entropy:-0.31, cross_view_coverage_delta:0.29, cross_view_iou:-0.27, coverage:-0.26, t_cont_band:1.79, front_cover_band:1.79, grey_band:1.79\nbreeding mix: x";
        let logged = spec_from_log(log).unwrap();
        let campaign = &builtins()[0];
        assert!(spec_close(&campaign.spec(), &logged, 0.006), "{logged}");
        assert!(!spec_close(&builtins()[2].spec(), &logged, 0.006));
        assert!(!same_spec(&campaign.spec(), &logged), "rounded weights are not the same spec");
        assert!(spec_from_log("fitness method: default blend — ...").is_none());
    }

    #[test]
    fn modules_follow_the_terms() {
        let on = |r: &Recipe, name: &str| r.modules(ctx()).into_iter().find(|m| m.name.starts_with(name)).unwrap().on;
        let mut r = Recipe::empty("planes");
        r.terms = terms(&[("planes_smooth", 1.0), ("sphericity", -0.5)]);
        r.auto_range = true;
        assert!(!r.needs_render() && !r.effective_auto_range());
        assert!(!on(&r, "Taste model"), "no rendering term: the range scan (and so the taste model) is skipped");
        assert!(!on(&r, "3D renderer") && on(&r, "Axis-plane") && on(&r, "4D field"));
        r.terms.push(Term { name: "image_entropy_band".into(), weight: 1.0 });
        assert!(on(&r, "Taste model"), "auto range with a rendered term uses the taste model");
        r.auto_range = false;
        assert!(!on(&r, "Taste model") && on(&r, "Low-res image gate") && on(&r, "3D renderer"));
        assert!(!on(&r, "Old preference model"));
        let taste = &builtins()[3];
        assert!(on(taste, "Taste model") && !on(taste, "Parallel scoring"));
        // No recipe: what the launcher really runs.
        let mut none = Recipe::empty("none");
        assert!(on(&none, "Old preference model") && on(&none, "3D renderer") && !on(&none, "Taste model") && !on(&none, "Parallel scoring"), "default blend");
        none.collapse = "s3".into();
        assert!(on(&none, "Taste model") && !on(&none, "Old preference model"), "S3 without a recipe scores with taste");
        none.collapse = "slice".into();
        none.auto_range = true;
        assert!(on(&none, "Taste model"), "the blend renders, so auto range scans with taste");
        let me = RunContext { map_elites: true, ..ctx() };
        assert!(Recipe::empty("me").modules(me).iter().any(|m| m.name.starts_with("Taste model") && m.on));
        assert!(!builtins()[0].modules(me).iter().any(|m| m.name == "Parallel scoring" && m.on), "MAP-Elites scores serially");
    }

    #[test]
    fn advice_catches_the_known_failures() {
        let has = |r: &Recipe, s: &str| r.advice(ctx()).iter().any(|a| a.text.contains(s));
        let mut r = Recipe::empty("x");
        assert!(r.advice(ctx())[0].level == Level::Error);
        r.terms = terms(&[("silhouette_irregularity", 1.0), ("box_dim", 0.5)]);
        assert!(has(&r, "No guard term") && has(&r, "motion"));
        r.terms.push(Term { name: "image_entropy_band".into(), weight: 3.0 });
        r.terms.push(Term { name: "image_motion".into(), weight: 1.0 });
        assert!(!has(&r, "No guard term") && has(&r, "flicker"));
        r.terms.push(Term { name: "sphericity".into(), weight: 0.5 });
        assert!(has(&r, "sphericity has a + weight"));
        r.embed_diversity = 1.0;
        assert!(has(&r, "Look diversity"));
        let (lo, hi) = r.score_range();
        assert_eq!(hi, 6.0);
        assert_eq!(lo, -5.0, "gated: failing genomes get -5");
    }

    #[test]
    fn saved_recipes_load_back_and_builtins_win_by_name() {
        let dir = std::env::temp_dir().join(format!("recipes_test_{}", std::process::id()));
        let mut r = builtins()[2].clone();
        r.name = "my copy".into();
        r.builtin = false;
        r.save(&dir).unwrap();
        assert_eq!(Recipe::list(&dir), vec!["my_copy".to_string()]);
        let back = Recipe::find(&dir, "my copy").unwrap();
        assert_eq!(back, r);
        assert!(Recipe::find(&dir, &builtins()[0].name).unwrap().builtin);
        let _ = std::fs::remove_dir_all(dir);
    }
}
