use rand::{Rng, SeedableRng, rngs::StdRng};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::time::Instant;

use crate::config::Config;
use crate::genome::Genome;
use crate::fractal::{evaluate_fitness_full, render_cpu, render_cpu_iter};
#[cfg(feature = "wgpu-backend")]
use crate::fractal::evaluate_fitness_batch;
use rayon::prelude::*;
use crate::colormap::apply_colormap;
use crate::fitness::{novelty_score, is_degenerate, behavior_descriptor, beauty_score_full};
use crate::io::{save_genome, save_png};
use crate::display;
use crate::aesthetic::AestheticScorer;

pub struct Optimizer {
    config: Config,
    population: Vec<Genome>,
    rng: StdRng,
    generation: u64,
    saved_count: u64,
    start: Instant,
    best_ever: Option<Genome>,
    stagnant_gens: u64,
    behavior_archive: VecDeque<Vec<f32>>,
    formula_archive:  VecDeque<Vec<f32>>,
    save_descriptors: Vec<Vec<f32>>,
    aesthetic: Option<AestheticScorer>,
    last_sub_scores: Option<[f32; 5]>,  // [boundary, edge, entropy, self_sim, cool_zone]
    max_png_entropy: f32,               // max PNG compression ratio seen across all evaluations
    max_clip_score: f32,
    max_laion_score: f32,
    recursion_model: Option<crate::recursion_model::RecursionModel>,
    clip_model:      Option<crate::recursion_model::RecursionModel>,
}

impl Optimizer {
    pub fn new(config: Config) -> Self {
        let mut rng = StdRng::from_os_rng();

        std::fs::create_dir_all(&config.output.save_dir).unwrap_or(());
        std::fs::create_dir_all(&config.output.population_dir).unwrap_or(());
        display::init();

        // Warm-start: seed up to N_SEEDS best genomes from the archive
        const N_SEEDS: usize = 12;
        const N_RANDOM_INJECT: usize = 8;
        let seeds = Self::load_archive_seeds(&config, N_SEEDS);

        let pop_size = config.optimization.population_size;
        let mut population: Vec<Genome> = seeds.clone();

        // Always inject N_RANDOM_INJECT fully random genomes for exploration diversity.
        // These are added before mutations so they're always guaranteed regardless of seed count.
        let n_random = N_RANDOM_INJECT.min(pop_size.saturating_sub(population.len()));
        population.extend((0..n_random).map(|_| Genome::random(&config, &mut rng)));
        display::print_status(&format!(
            "Warm-start: {} archive seeds + {} fresh random individuals",
            seeds.len(), n_random
        ));

        // Fill remainder with mutations of seeds
        let n_fill = pop_size.saturating_sub(population.len());
        if seeds.is_empty() {
            population.extend((0..n_fill).map(|_| Genome::random(&config, &mut rng)));
        } else {
            for i in 0..n_fill {
                let src = &seeds[i % seeds.len()];
                population.push(src.mutate(&config, &mut rng));
            }
        }

        // Prime the formula archive with basis-weight vectors of the starting population,
        // mirroring the behavior_archive priming logic below.
        let formula_archive: VecDeque<Vec<f32>> = population.iter()
            .map(|g| g.formula_descriptor())
            .collect();

        // Prime the novelty archive with the initial population's OWN behavioral descriptors.
        // Without this the archive is empty on gen 1, so novelty_score() is near-maximal for
        // every genome (≈1.0) — a flat ~+novelty_weight bonus that inflates generation-1
        // fitness. By gen 2 the archive is flooded with gen-1's behaviors and novelty collapses,
        // producing the characteristic gen-1 spike. Seeding with the starting pool makes gen-1
        // novelty measure "difference from the current pool", exactly as every later gen does.
        let ew  = config.optimization.eval_width;
        let eh  = config.optimization.eval_height;
        let emi = config.optimization.eval_max_iter;
        let behavior_archive: VecDeque<Vec<f32>> = population.iter()
            .map(|g| {
                let et = render_cpu_iter(g, &config, ew, eh, emi);
                behavior_descriptor(&et, emi)
            })
            .collect();

        let aesthetic = AestheticScorer::new();
        if aesthetic.is_some() {
            display::print_status("Aesthetic scorer: spawning Python sidecar...");
        }

        let recursion_model = crate::recursion_model::RecursionModel::load(
            std::path::Path::new("recursion_model.json"));
        match &recursion_model {
            Some(m) => display::print_status(&format!(
                "Recursion predictor: loaded (n={}, cv_r={:.3}, weight={:.2})",
                m.n_samples, m.cv_pearson, config.optimization.recursion_pred_weight)),
            None => display::print_status(
                "Recursion predictor: no model file — recursion criterion inert this run"),
        }

        let clip_model = crate::recursion_model::RecursionModel::load(
            std::path::Path::new("clip_model.json"));
        match &clip_model {
            Some(m) => display::print_status(&format!(
                "CLIP predictor:      loaded (n={}, cv_r={:.3}, weight={:.2})",
                m.n_samples, m.cv_pearson, config.optimization.clip_pred_weight)),
            None => display::print_status(
                "CLIP predictor:      no model file — CLIP criterion inert this run"),
        }

        Self {
            config,
            population,
            rng,
            generation: 0,
            saved_count: 0,
            start: Instant::now(),
            save_descriptors: Vec::new(),
            best_ever: None,
            stagnant_gens: 0,
            behavior_archive,
            formula_archive,
            aesthetic,
            last_sub_scores: None,
            max_png_entropy: 0.0,
            max_clip_score: 0.0,
            max_laion_score: 0.0,
            recursion_model,
            clip_model,
        }
    }

    fn load_archive_seeds(config: &Config, n: usize) -> Vec<Genome> {
        use crate::io::load_genome;
        let mut candidates: Vec<(f32, Genome)> = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&config.output.save_dir) {
            for e in entries.flatten() {
                let path = e.path();
                if path.extension().and_then(|x| x.to_str()) != Some("nn") { continue; }
                if let Ok(g) = load_genome(&path) {
                    // Seed primarily by aesthetic quality (CLIP+LAION), with smaller
                    // bonuses for recursion/self-replication. Doubling CLIP weight
                    // ensures epoch restarts inherit the most aesthetically successful
                    // genomes, not just the most recursive ones.
                    let clip = if g.clip_score > 0.0 { g.clip_score } else { g.beauty };
                    let laion_norm = g.laion_score / 10.0;
                    let score = 2.0 * clip
                        + 0.15 * laion_norm
                        + 0.20 * g.self_replication
                        + 0.20 * g.fractal_recursion;
                    candidates.push((score, g));
                }
            }
        }
        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(n);
        candidates.into_iter().map(|(_, g)| g).collect()
    }

    pub fn run_forever(&mut self) {
        loop { self.step(); }
    }

    fn step(&mut self) {
        self.generation += 1;
        let nw          = self.config.optimization.novelty_weight;
        let nk          = self.config.optimization.novelty_k;
        let archive_max = self.config.optimization.archive_size;
        let n_pop       = self.population.len();

        // ── Evaluate all genomes ─────────────────────────────────────────
        let archive_snap: Vec<Vec<f32>> = self.behavior_archive.iter().cloned().collect();

        #[cfg(feature = "wgpu-backend")]
        let fitnesses: Vec<(f32, f32, Vec<f32>)> = if crate::render_gpu::gpu_available() {
            display::print_status(&format!("Gen {}  Evaluating {} genomes (GPU)...", self.generation, n_pop));
            evaluate_fitness_batch(&self.population, &self.config)
        } else {
            display::print_status(&format!("Gen {}  Evaluating {} genomes (CPU×{})...", self.generation, n_pop, rayon::current_num_threads()));
            self.population.par_iter().map(|genome| evaluate_fitness_full(genome, &self.config)).collect()
        };

        #[cfg(not(feature = "wgpu-backend"))]
        let fitnesses: Vec<(f32, f32, Vec<f32>)> = {
            display::print_status(&format!("Gen {}  Evaluating {} genomes...", self.generation, n_pop));
            self.population.par_iter().map(|genome| evaluate_fitness_full(genome, &self.config)).collect()
        };

        let rpw = self.config.optimization.recursion_pred_weight;
        let fdw = self.config.optimization.formula_diversity_weight;
        let cpw = self.config.optimization.clip_pred_weight;
        let formula_snap: Vec<Vec<f32>> = self.formula_archive.iter().cloned().collect();
        for (i, fitness_result) in fitnesses.into_iter().enumerate() {
            let (raw_png, structured_ent, descriptor) = fitness_result;
            // raw_png → save-gate thresholding (beauty_entropy), thresholds unchanged.
            // structured_ent → geometric mean(fine, coarse) PNG entropy: selection fitness.
            //   Noise scores high at fine scale but near-zero at coarse (averages to uniform).
            //   Structured fractals stay complex at all scales → both terms stay high.
            if raw_png > self.max_png_entropy { self.max_png_entropy = raw_png; }
            let novelty = novelty_score(&descriptor, &archive_snap, nk);
            let feats = self.population[i].recursion_features();
            let pred_rec = self.recursion_model.as_ref()
                .map(|m| m.predict(&feats))
                .unwrap_or(0.0);
            let pred_clip_val = self.clip_model.as_ref()
                .map(|m| m.predict(&feats))
                .unwrap_or(0.0);
            let formula_feats = self.population[i].formula_descriptor();
            let formula_div = novelty_score(&formula_feats, &formula_snap, nk);
            // Anti-bloat: small penalty on DAG program size so the GA prefers
            // compact expressions over ones that pad to noise (multiscale entropy
            // is the other backstop). Legacy genomes have no program → no penalty.
            const COMPLEXITY_PENALTY: f32 = 0.012;
            let cxpen = COMPLEXITY_PENALTY * self.population[i].program.len() as f32;
            self.population[i].beauty_entropy    = raw_png;       // save gate uses raw
            self.population[i].pred_recursion    = pred_rec;
            self.population[i].pred_clip         = pred_clip_val;
            self.population[i].formula_diversity = formula_div;
            self.population[i].fitness =
                structured_ent + nw * novelty + rpw * pred_rec + fdw * formula_div + cpw * pred_clip_val - cxpen;
            if self.behavior_archive.len() >= archive_max { self.behavior_archive.pop_front(); }
            self.behavior_archive.push_back(descriptor);
            if self.formula_archive.len() >= archive_max { self.formula_archive.pop_front(); }
            self.formula_archive.push_back(formula_feats);
        }

        // ── Sort best → worst ─────────────────────────────────────────────
        self.population.sort_by(|a, b|
            b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal));

        // ── Track best-ever (raw beauty, no novelty inflation) ───────────
        let (current_beauty, _, _) = evaluate_fitness_full(&self.population[0], &self.config);
        let best_ever_beauty    = self.best_ever.as_ref().map(|g| g.fitness).unwrap_or(0.0);
        if current_beauty > best_ever_beauty + 0.005 {
            let mut clone = self.population[0].clone();
            clone.fitness = current_beauty;
            self.best_ever = Some(clone);
            self.stagnant_gens = 0;
        } else {
            self.stagnant_gens += 1;
        }

        // Elite count drives the save-scan budget below. Gradient refinement was removed:
        // the formula is now evolved directly by the GA (no transformer to backprop through).
        let elite = self.config.optimization.elitism_count.min(self.population.len());

        // ── Poll for aesthetic score, then request a new probe every 5 gens ──
        if let Some(scorer) = &mut self.aesthetic {
            scorer.poll(self.generation);
        }
        if self.generation % 3 == 0 {
            let probe_path = PathBuf::from("/tmp/nnfractals_probe.png");
            let et  = render_cpu(&self.population[0], &self.config, 256, 256);
            let (_, bd) = beauty_score_full(&et, 256, self.config.rendering.max_iter);
            self.last_sub_scores = Some([bd.boundary, bd.edge, bd.entropy, bd.self_sim, bd.cool_zone]);
            if self.aesthetic.is_some() {
                display::print_status(&format!("Gen {}  Rendering aesthetic probe...", self.generation));
                let rgb = apply_colormap(&et, self.config.rendering.max_iter, &self.config.rendering.colormap);
                save_png(&rgb, 256, 256, &probe_path).unwrap_or(());
                if let Some(scorer) = &mut self.aesthetic {
                    scorer.request(probe_path, self.generation);
                }
            }
        }

        // ── Display ───────────────────────────────────────────────────────
        let aes_line = self.aesthetic.as_ref().map(|s| s.status_line());
        display::refresh(
            self.generation,
            &self.population,
            self.saved_count,
            self.start.elapsed().as_secs(),
            self.stagnant_gens,
            self.best_ever.as_ref().map(|g| g.fitness).unwrap_or(0.0),
            aes_line.as_deref(),
            self.last_sub_scores.as_ref(),
            self.max_clip_score,
            self.max_laion_score,
        );
        display::print_status(&format!("Gen {} complete", self.generation));

        // ── Save gate: attempt saves on the highest raw-PNG-entropy genomes,
        // not the novelty-inflated fitness leaders (which starve the gate). ──
        // Only consider genomes already in the savable PNG band [min, max]: genomes above
        // max are salt-and-pepper noise (auto-rejected) and below min are too uniform, so
        // including them would waste the per-generation CLIP-scoring budget. Filtering here
        // focuses every real attempt on genomes that can actually clear the gate.
        let min_png = self.config.output.min_entropy_prefilter;
        let max_png = self.config.output.max_entropy_prefilter;
        let n_candidates = elite;
        let mut by_png: Vec<usize> = (0..self.population.len())
            .filter(|&i| {
                let e = self.population[i].beauty_entropy;
                e >= min_png && e <= max_png
            })
            .collect();
        by_png.sort_by(|&a, &b| {
            self.population[b].beauty_entropy
                .partial_cmp(&self.population[a].beauty_entropy)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        // Scan down the png ranking; "exists" attempts (already-saved warm-start seeds)
        // are cheap and don't consume the real-attempt budget, so genuinely new genomes
        // always get their CLIP/LAION shot.
        let mut tally: std::collections::HashMap<&'static str, u32> = std::collections::HashMap::new();
        let mut real_attempts = 0;
        for &i in by_png.iter() {
            let reason = self.try_save(i);
            *tally.entry(reason).or_insert(0) += 1;
            if reason != "exists" {
                real_attempts += 1;
                if real_attempts >= n_candidates { break; }
            }
        }
        let summary: String = tally.iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect::<Vec<_>>()
            .join(" ");
        display::print_status(&format!("Gen {}  Save scan: {}", self.generation, summary));

        // ── Stagnation restart ────────────────────────────────────────────
        if self.stagnant_gens >= self.config.optimization.restart_after_gens {
            self.restart_population();
        } else {
            display::print_status(&format!("Gen {}  Evolving population...", self.generation));
            self.evolve();
        }
    }

    fn force_save(&mut self, genome: &Genome) {
        let w = self.config.rendering.default_width;
        let h = self.config.rendering.default_height;
        let escape_times = render_cpu(genome, &self.config, w, h);
        let rgb = apply_colormap(&escape_times, self.config.rendering.max_iter,
                                 &self.config.rendering.colormap);
        let name     = format!("best_{:016x}", genome.id);
        let png_path = self.config.output.save_dir.join(format!("{name}.png"));
        let nn_path  = self.config.output.save_dir.join(format!("{name}.nn"));
        let (beauty, bd) = beauty_score_full(&escape_times, w as usize, self.config.rendering.max_iter);
        save_png(&rgb, w, h, &png_path).unwrap_or(());

        // Score with CLIP+LAION if the aesthetic scorer is ready (non-blocking)
        let aesthetic_scores = self.aesthetic.as_mut()
            .and_then(|a| if a.is_ready() { a.score_blocking(png_path.clone()) } else { None });
        let final_beauty = aesthetic_scores.as_ref()
            .map(|s| s.laion / 10.0).unwrap_or(beauty);

        let mut g = genome.clone();
        g.beauty            = final_beauty;
        g.beauty_boundary   = bd.boundary;
        g.beauty_edge       = bd.edge;
        g.beauty_entropy    = bd.entropy;
        g.beauty_self_sim   = bd.self_sim;
        g.beauty_cool_zone  = bd.cool_zone;
        g.clip_score        = aesthetic_scores.as_ref().map(|s| s.clip).unwrap_or(0.0);
        g.laion_score       = aesthetic_scores.as_ref().map(|s| s.laion).unwrap_or(0.0);
        g.self_replication  = crate::fractal::self_replication_score(&g, &self.config);
        g.fractal_recursion = crate::fractal::fractal_recursion_score(&g, &self.config);
        if g.clip_score  > self.max_clip_score  { self.max_clip_score  = g.clip_score; }
        if g.laion_score > self.max_laion_score { self.max_laion_score = g.laion_score; }
        g.formula_readable = g.formula_expr();   // human-readable comment in the .nn
        save_genome(&g, &nn_path).unwrap_or(());
        display::print_save(&g, &png_path.display().to_string(), final_beauty);
        self.saved_count += 1;
        let eval_et = render_cpu_iter(genome, &self.config,
            self.config.optimization.eval_width,
            self.config.optimization.eval_height,
            self.config.optimization.eval_max_iter);
        let desc = behavior_descriptor(&eval_et, self.config.optimization.eval_max_iter);
        self.save_descriptors.push(desc);
    }

    fn restart_population(&mut self) {
        if let Some(best) = self.best_ever.clone() {
            let already_saved = self.config.output.save_dir
                .join(format!("{:016x}.nn", best.id))
                .exists();
            if !already_saved { self.force_save(&best); }
        }
        display::print_restart(self.generation,
                               self.best_ever.as_ref().map(|g| g.fitness).unwrap_or(0.0));

        // Reload a FEW archive seeds on restart, but bias heavily toward fresh random
        // Seeds now ranked by CLIP score; 6 seeds keeps the top-CLIP genomes in play
        // each epoch without over-constraining exploration (random + exotic fill the rest).
        const N_SEEDS: usize = 6;
        const N_RANDOM_INJECT: usize = 18;
        let seeds = Self::load_archive_seeds(&self.config, N_SEEDS);
        let mut new_pop: Vec<Genome> = seeds.clone();
        // Always include current best_ever if not already in seeds
        if let Some(best) = &self.best_ever {
            if !new_pop.iter().any(|g| g.id == best.id) {
                new_pop.push(best.clone());
            }
        }
        let pop_size = self.config.optimization.population_size;

        // Always inject N_RANDOM_INJECT fresh random genomes on every stagnation restart.
        // 8 of these are "exotic": forced terms from the 6 bases proven to score best on CLIP
        // (z², sin, tan, z/(z²+1), (z²-1)/(z²+1), 1/(z²+c)) — see FormulaTerm::random_exotic.
        const N_EXOTIC: usize = 8;
        let n_random = N_RANDOM_INJECT.min(pop_size.saturating_sub(new_pop.len()));
        for i in 0..n_random {
            let g = if i < N_EXOTIC {
                Genome::random_exotic(&self.config, &mut self.rng)
            } else {
                Genome::random(&self.config, &mut self.rng)
            };
            new_pop.push(g);
        }
        display::print_status(&format!(
            "Restart: {} seeds + {} random ({} exotic) injected",
            seeds.len(), n_random, N_EXOTIC.min(n_random)
        ));

        // Fill remainder with mutations of seeds
        if !seeds.is_empty() {
            let n_fill = pop_size.saturating_sub(new_pop.len());
            for i in 0..n_fill {
                let src = &seeds[i % seeds.len()];
                new_pop.push(src.mutate(&self.config, &mut self.rng));
            }
        } else {
            while new_pop.len() < pop_size {
                new_pop.push(Genome::random(&self.config, &mut self.rng));
            }
        }
        self.population    = new_pop;
        self.stagnant_gens = 0;
        // Retain the 300 most-recent save descriptors across restarts so the diversity gate
        // keeps blocking visual regions already saved in previous epochs. Clearing it caused
        // the same view to be re-saved 10-18× across successive restarts.
        let keep = 300usize;
        if self.save_descriptors.len() > keep {
            self.save_descriptors.drain(..self.save_descriptors.len() - keep);
        }
    }

    /// Returns a static reason string for the per-generation save-scan tally:
    /// "saved" | "exists" | "degenerate" | "low_png" | "low_diversity" | "low_clip_laion".
    fn try_save(&mut self, idx: usize) -> &'static str {
        let genome = &self.population[idx];
        let w = self.config.rendering.default_width;
        let h = self.config.rendering.default_height;

        let nn_path = self.config.output.save_dir.join(format!("{:016x}.nn", genome.id));
        if nn_path.exists() { return "exists"; }

        // ── Stage 1: PNG compression entropy prefilter ───────────────────
        let ew = self.config.optimization.eval_width;
        let eh = self.config.optimization.eval_height;
        let emi = self.config.optimization.eval_max_iter;
        let eval_et = render_cpu_iter(genome, &self.config, ew, eh, emi);
        if is_degenerate(&eval_et) { return "degenerate"; }
        let png_ent = crate::fitness::png_compression_entropy(
            &eval_et, ew, eh, emi, &self.config.rendering.colormap,
        );
        // Admit only if near the maximum visual complexity encountered.
        // Once max is established (>0.30), require ≥ 85% of max to save.
        // Flat threshold: filters boring uniform fractals; CLIP/LAION handles quality.
        // (Near-max criterion was shelved: a single noisy genome can set max very high,
        // blocking all structured-but-compressible beautiful fractals.)
        if png_ent < self.config.output.min_entropy_prefilter { return "low_png"; }
        if png_ent > self.config.output.max_entropy_prefilter { return "noise"; }

        // ── Diversity gate ────────────────────────────────────────────────
        let desc = behavior_descriptor(&eval_et, self.config.optimization.eval_max_iter);
        let min_dist = self.save_descriptors.iter()
            .map(|d| desc.iter().zip(d.iter()).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt())
            .fold(f32::INFINITY, f32::min);
        if min_dist < self.config.output.min_save_distance { return "low_diversity"; }

        // ── Stage 2: render full image ────────────────────────────────────
        // Find the locally best view (3×3 grid, ±15%/zoom pan) using multiscale
        // entropy as a fast composition proxy. CLIP scores the best-view render;
        // the genome's own view params are not modified (archive keeps original).
        let render_genome = crate::fractal::best_entropy_view(genome, &self.config);
        let escape_times = render_cpu(&render_genome, &self.config, w, h);
        let rgb      = apply_colormap(&escape_times, self.config.rendering.max_iter,
                                      &self.config.rendering.colormap);
        let (beauty, bd) = beauty_score_full(&escape_times, w as usize, self.config.rendering.max_iter);

        // ── Stage 2: CLIP + LAION aesthetic scores (pre-trained, image-only) ──
        // The Python sidecar scores by file path, so we must hand it a file. Write to a
        // SINGLE reusable temp file (overwritten per candidate) rather than a uniquely-named
        // PNG in the output dir — most candidates fail the gate, and writing-then-deleting a
        // full-size PNG for each one is pure disk churn. The permanent PNG is only written
        // below, after the candidate passes.
        let aesthetic_scores = match self.aesthetic.as_mut() {
            Some(aes) => {
                let score_tmp = std::path::PathBuf::from("/tmp/nnfractals_score.png");
                save_png(&rgb, w, h, &score_tmp).unwrap_or(());
                aes.score_blocking(score_tmp)
            }
            None => None,
        };

        // Save gate: pass if EITHER clip OR laion exceeds its threshold (OR logic).
        // Privileges fractals that are excellent on at least one aesthetic dimension.
        let passes = match &aesthetic_scores {
            Some(s) => s.clip  >= self.config.output.min_clip_score
                    || s.laion >= self.config.output.min_laion_score,
            None    => beauty  >= self.config.output.min_beauty,
        };
        if !passes {
            return "low_clip_laion";  // nothing written to the output dir
        }

        // Passed — now write the permanent PNG to the output dir.
        let name     = format!("{:016x}", genome.id);
        let png_path = self.config.output.save_dir.join(format!("{name}.png"));
        save_png(&rgb, w, h, &png_path).unwrap_or(());

        // Use LAION (wider range) as the stored beauty when available
        let final_score = aesthetic_scores.as_ref()
            .map(|s| s.laion / 10.0)
            .unwrap_or(beauty);

        let mut g = self.population[idx].clone();
        g.beauty           = final_score;
        g.beauty_boundary  = bd.boundary;
        g.beauty_edge      = bd.edge;
        g.beauty_entropy   = png_ent;
        g.beauty_self_sim  = bd.self_sim;
        g.beauty_cool_zone = bd.cool_zone;
        g.clip_score       = aesthetic_scores.as_ref().map(|s| s.clip).unwrap_or(0.0);
        g.laion_score      = aesthetic_scores.as_ref().map(|s| s.laion).unwrap_or(0.0);
        // Measure zoom self-replication and fractal recursion only for genomes that
        // actually pass the gate (a handful per generation) — they travel with the .nn.
        g.self_replication  = crate::fractal::self_replication_score(&g, &self.config);
        g.fractal_recursion = crate::fractal::fractal_recursion_score(&g, &self.config);
        if g.clip_score  > self.max_clip_score  { self.max_clip_score  = g.clip_score; }
        if g.laion_score > self.max_laion_score { self.max_laion_score = g.laion_score; }
        g.fitness = final_score;
        g.formula_readable = g.formula_expr();   // human-readable comment in the .nn
        save_genome(&g, &nn_path).unwrap_or(());
        self.save_descriptors.push(desc);
        display::print_save(&g, &png_path.display().to_string(), final_score);
        self.saved_count += 1;
        "saved"
    }

    fn evolve(&mut self) {
        let n           = self.population.len();
        let elite_count = self.config.optimization.elitism_count.min(n);

        // Formula-diverse elite selection: one representative per unique formula type
        let mut seen: Vec<String> = Vec::new();
        let mut diverse: Vec<Genome> = Vec::new();
        for g in &self.population {
            if diverse.len() >= elite_count { break; }
            let label = g.formula_ops_label();
            if !seen.contains(&label) { seen.push(label); diverse.push(g.clone()); }
        }
        for g in &self.population {
            if diverse.len() >= elite_count { break; }
            if !diverse.iter().any(|e| e.id == g.id) { diverse.push(g.clone()); }
        }

        let mut new_pop = diverse;

        // Continuous exploration: inject a few fresh random genomes every generation, not
        // only in the periodic restart bursts. Sustained random injection is what broke the
        // CLIP plateau (best rose past 0.50), so keep a steady trickle of new-region search.
        const N_RANDOM_PER_GEN: usize = 3;
        let room = n.saturating_sub(new_pop.len());
        for _ in 0..N_RANDOM_PER_GEN.min(room.saturating_sub(1)) {
            new_pop.push(Genome::random(&self.config, &mut self.rng));
        }

        while new_pop.len() < n {
            let a_idx = self.rng.random_range(0..elite_count);
            let a = &self.population[a_idx];
            let child = if self.rng.random_bool(0.5) {
                let diff: Vec<usize> = (0..elite_count)
                    .filter(|&i| self.population[i].formula_ops_label() != a.formula_ops_label())
                    .collect();
                let b_idx = if !diff.is_empty() {
                    diff[self.rng.random_range(0..diff.len())]
                } else {
                    self.rng.random_range(0..elite_count)
                };
                Genome::crossover(a, &self.population[b_idx], &self.config, &mut self.rng)
                    .mutate(&self.config, &mut self.rng)
            } else {
                a.mutate(&self.config, &mut self.rng)
            };
            new_pop.push(child);
        }
        self.population = new_pop;
    }
}
