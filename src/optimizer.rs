use burn::tensor::{Tensor, TensorData, backend::AutodiffBackend};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::time::Instant;

use crate::config::Config;
use crate::genome::{Genome, LayerData};
use crate::transformer::{TransformerTensors, CLayer, transformer_forward_latent_tensor};
use crate::fractal::{pixel_coords, evaluate_fitness_full, render_cpu, formula_step_tensor};
use crate::colormap::apply_colormap;
use crate::fitness::{novelty_score, is_degenerate, behavior_descriptor, beauty_score};
use crate::io::{save_genome, save_png};
use crate::display;
use crate::aesthetic::AestheticScorer;

pub struct Optimizer<B: AutodiffBackend> {
    config: Config,
    population: Vec<Genome>,
    device: B::Device,
    rng: StdRng,
    generation: u64,
    saved_count: u64,
    start: Instant,
    best_ever: Option<Genome>,
    stagnant_gens: u64,
    behavior_archive: VecDeque<Vec<f32>>,
    save_descriptors: Vec<Vec<f32>>,
    aesthetic: Option<AestheticScorer>,
}

impl<B: AutodiffBackend> Optimizer<B>
where
    B::Device: Clone,
{
    pub fn new(config: Config, device: B::Device) -> Self {
        let mut rng = StdRng::from_os_rng();
        let population: Vec<Genome> = (0..config.optimization.population_size)
            .map(|_| Genome::random(&config, &mut rng))
            .collect();

        std::fs::create_dir_all(&config.output.save_dir).unwrap_or(());
        std::fs::create_dir_all(&config.output.population_dir).unwrap_or(());
        display::init();

        let aesthetic = AestheticScorer::new();
        if aesthetic.is_some() {
            display::print_status("Aesthetic scorer: spawning Python sidecar...");
        }

        Self {
            config,
            population,
            device,
            rng,
            generation: 0,
            saved_count: 0,
            start: Instant::now(),
            save_descriptors: Vec::new(),
            best_ever: None,
            stagnant_gens: 0,
            behavior_archive: VecDeque::new(),
            aesthetic,
        }
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

        // ── Evaluate all genomes ──────────────────────────────────────────
        let archive_snap: Vec<Vec<f32>> = self.behavior_archive.iter().cloned().collect();
        for i in 0..n_pop {
            display::print_status(&format!(
                "Gen {}  Evaluating genome {:2}/{}...",
                self.generation, i + 1, n_pop
            ));
            let (beauty, descriptor) = evaluate_fitness_full(&self.population[i], &self.config);
            let novelty = novelty_score(&descriptor, &archive_snap, nk);
            self.population[i].fitness = beauty + nw * novelty;
            if self.behavior_archive.len() >= archive_max { self.behavior_archive.pop_front(); }
            self.behavior_archive.push_back(descriptor);
        }

        // ── Sort best → worst ─────────────────────────────────────────────
        self.population.sort_by(|a, b|
            b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal));

        // ── Track best-ever (raw beauty, no novelty inflation) ───────────
        let (current_beauty, _) = evaluate_fitness_full(&self.population[0], &self.config);
        let best_ever_beauty    = self.best_ever.as_ref().map(|g| g.fitness).unwrap_or(0.0);
        if current_beauty > best_ever_beauty + 0.005 {
            let mut clone = self.population[0].clone();
            clone.fitness = current_beauty;
            self.best_ever = Some(clone);
            self.stagnant_gens = 0;
        } else {
            self.stagnant_gens += 1;
        }

        // ── Backprop on elite genomes ─────────────────────────────────────
        let elite = self.config.optimization.elitism_count.min(self.population.len());
        for i in 0..elite {
            if self.population[i].fitness > 0.1 {
                display::print_status(&format!(
                    "Gen {}  Backprop elite {}/{}...",
                    self.generation, i + 1, elite
                ));
                self.backprop_step(i);
            }
        }

        // ── Re-evaluate elites after backprop ─────────────────────────────
        let archive_snap2: Vec<Vec<f32>> = self.behavior_archive.iter().cloned().collect();
        for i in 0..elite {
            display::print_status(&format!(
                "Gen {}  Re-evaluating elite {}/{}...",
                self.generation, i + 1, elite
            ));
            let (beauty, descriptor) = evaluate_fitness_full(&self.population[i], &self.config);
            let novelty = novelty_score(&descriptor, &archive_snap2, nk);
            self.population[i].fitness = beauty + nw * novelty;
        }
        self.population.sort_by(|a, b|
            b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal));

        // ── Poll for aesthetic score, then request a new probe every 5 gens ──
        if let Some(scorer) = &mut self.aesthetic {
            scorer.poll(self.generation);
        }
        if self.generation % 5 == 0 && self.aesthetic.is_some() {
            display::print_status(&format!("Gen {}  Rendering aesthetic probe...", self.generation));
            let probe_path = PathBuf::from("/tmp/nnfractals_probe.png");
            let et  = render_cpu(&self.population[0], &self.config, 256, 256);
            let rgb = apply_colormap(&et, self.config.rendering.max_iter, &self.config.rendering.colormap);
            save_png(&rgb, 256, 256, &probe_path).unwrap_or(());
            if let Some(scorer) = &mut self.aesthetic {
                scorer.request(probe_path, self.generation);
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
        );
        display::print_status(&format!("Gen {} complete", self.generation));

        // ── Save gate ─────────────────────────────────────────────────────
        for i in 0..elite {
            self.try_save(i);
        }

        // ── Stagnation restart ────────────────────────────────────────────
        if self.stagnant_gens >= self.config.optimization.restart_after_gens {
            self.restart_population();
        } else {
            display::print_status(&format!("Gen {}  Evolving population...", self.generation));
            self.evolve();
        }
    }

    fn backprop_step(&mut self, idx: usize) {
        let lr        = self.config.optimization.learning_rate;
        let max_iter  = self.config.optimization.eval_max_iter;
        let ew        = self.config.optimization.eval_width;
        let eh        = self.config.optimization.eval_height;
        let n         = (ew * eh) as usize;
        let clamp_val = self.config.optimization.eval_clamp;
        let device    = self.device.clone();
        let d_model   = self.config.network.d_model;

        let c_inner = pixel_coords::<B::InnerBackend>(ew, eh, &self.population[idx], &device);

        let mk2 = |data: Vec<f32>, r: usize, c: usize| -> Tensor<B, 2> {
            Tensor::from_inner(
                Tensor::<B::InnerBackend, 2>::from_data(TensorData::new(data, [r, c]), &device)
            ).require_grad()
        };
        let mk1 = |data: Vec<f32>, len: usize| -> Tensor<B, 1> {
            Tensor::from_inner(
                Tensor::<B::InnerBackend, 1>::from_data(TensorData::new(data, [len]), &device)
            ).require_grad()
        };
        let make_clayer = |ld: &LayerData| -> CLayer<B> {
            let wim = if ld.weights_im.is_empty() { vec![0.0f32; ld.weights.len()] }
                      else { ld.weights_im.clone() };
            let bim = if ld.biases_im.is_empty()  { vec![0.0f32; ld.biases.len()]  }
                      else { ld.biases_im.clone() };
            (mk2(ld.weights.clone(), ld.out_size, ld.in_size),
             mk2(wim,               ld.out_size, ld.in_size),
             mk1(ld.biases.clone(), ld.out_size),
             mk1(bim,               ld.out_size))
        };

        let tw = &self.population[idx].transformer;
        let tt = TransformerTensors {
            embed_z: make_clayer(&tw.embed_z),
            embed_c: make_clayer(&tw.embed_c),
            w_q:     make_clayer(&tw.w_q),
            w_k:     make_clayer(&tw.w_k),
            w_v:     make_clayer(&tw.w_v),
            w_o:     make_clayer(&tw.w_o),
            ff1:     make_clayer(&tw.ff1),
            ff2:     make_clayer(&tw.ff2),
            output:  make_clayer(&tw.output),
        };
        let ff_acts = self.population[idx].transformer.ff_acts.clone();

        for step in 0..self.config.optimization.backprop_steps {
            display::print_status(&format!(
                "Gen {}  Backprop elite {} step {}/{}...",
                self.generation, idx + 1, step + 1, self.config.optimization.backprop_steps
            ));

            // Build differentiable latent tensors (fresh each step to pick up prior updates)
            let lat_re_data: Vec<f32> = self.population[idx].latent.iter().map(|x| x.0).collect();
            let lat_im_data: Vec<f32> = self.population[idx].latent.iter().map(|x| x.1).collect();
            let latent_re = mk2(lat_re_data, 1, d_model);
            let latent_im = mk2(lat_im_data, 1, d_model);

            // Run transformer ONCE per genome: latent → formula weights
            let (fw_re, fw_im) = transformer_forward_latent_tensor(
                &tt, &ff_acts, &latent_re, &latent_im, d_model,
            );

            let c: Tensor<B, 2> = Tensor::from_inner(c_inner.clone());
            let mut z: Tensor<B, 2> = Tensor::zeros([n, 2], &device);

            for _ in 0..max_iter {
                z = formula_step_tensor(&fw_re, &fw_im, &z, &c).clamp(-clamp_val, clamp_val);
            }

            let z_x   = z.clone().narrow(1, 0, 1).flatten::<1>(0, 1);
            let z_y   = z.narrow(1, 1, 1).flatten::<1>(0, 1);
            let loss  = (z_x.var(0) + z_y.var(0)).neg();
            let grads = loss.backward();

            const CLIP: f32 = 1.0;
            macro_rules! apply_grad {
                ($tensor:expr, $vec:expr) => {
                    if let Some(g) = $tensor.grad(&grads) {
                        let v = ($tensor.clone().inner() - g.clamp(-CLIP, CLIP) * lr)
                            .into_data().to_vec::<f32>().unwrap_or_default();
                        if v.len() == $vec.len() { $vec = v; }
                    }
                };
            }
            macro_rules! update_layer {
                ($tt_layer:expr, $gl:expr) => {{
                    let (w_re, w_im, b_re, b_im) = &$tt_layer;
                    let ld = &mut $gl;
                    apply_grad!(w_re, ld.weights);
                    apply_grad!(w_im, ld.weights_im);
                    apply_grad!(b_re, ld.biases);
                    apply_grad!(b_im, ld.biases_im);
                }};
            }

            let tw = &mut self.population[idx].transformer;
            update_layer!(tt.embed_z, tw.embed_z);
            update_layer!(tt.embed_c, tw.embed_c);
            update_layer!(tt.w_q,     tw.w_q);
            update_layer!(tt.w_k,     tw.w_k);
            update_layer!(tt.w_v,     tw.w_v);
            update_layer!(tt.w_o,     tw.w_o);
            update_layer!(tt.ff1,     tw.ff1);
            update_layer!(tt.ff2,     tw.ff2);
            update_layer!(tt.output,  tw.output);

            // Apply gradient to latent
            if let Some(g) = latent_re.grad(&grads) {
                let v = (latent_re.clone().inner() - g.clamp(-CLIP, CLIP) * lr)
                    .into_data().to_vec::<f32>().unwrap_or_default();
                let lat = &mut self.population[idx].latent;
                for (i, val) in v.iter().enumerate() {
                    if i < lat.len() { lat[i].0 = *val; }
                }
            }
            if let Some(g) = latent_im.grad(&grads) {
                let v = (latent_im.clone().inner() - g.clamp(-CLIP, CLIP) * lr)
                    .into_data().to_vec::<f32>().unwrap_or_default();
                let lat = &mut self.population[idx].latent;
                for (i, val) in v.iter().enumerate() {
                    if i < lat.len() { lat[i].1 = *val; }
                }
            }
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
        save_png(&rgb, w, h, &png_path).unwrap_or(());
        save_genome(genome, &nn_path).unwrap_or(());
        let beauty = beauty_score(&escape_times, w as usize, self.config.rendering.max_iter);
        display::print_save(genome, &png_path.display().to_string(), beauty);
        self.saved_count += 1;
        let desc = behavior_descriptor(&escape_times, self.config.rendering.max_iter);
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

        let mut new_pop: Vec<Genome> = Vec::new();
        if let Some(best) = &self.best_ever {
            new_pop.push(best.clone());
            for _ in 0..3 { new_pop.push(best.mutate(&self.config, &mut self.rng)); }
        }
        while new_pop.len() < self.config.optimization.population_size {
            new_pop.push(Genome::random(&self.config, &mut self.rng));
        }
        self.population  = new_pop;
        self.stagnant_gens = 0;
    }

    fn try_save(&mut self, idx: usize) {
        let genome = &self.population[idx];
        let w = self.config.rendering.default_width;
        let h = self.config.rendering.default_height;

        let nn_path = self.config.output.save_dir.join(format!("{:016x}.nn", genome.id));
        if nn_path.exists() { return; }

        let escape_times = render_cpu(genome, &self.config, w, h);
        if is_degenerate(&escape_times) { return; }

        let beauty = beauty_score(&escape_times, w as usize, self.config.rendering.max_iter);
        if beauty < self.config.output.min_beauty { return; }

        // Reject near-duplicates via behavioral descriptor L2 distance
        let desc = behavior_descriptor(&escape_times, self.config.rendering.max_iter);
        let min_dist = self.save_descriptors.iter()
            .map(|d| desc.iter().zip(d.iter()).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt())
            .fold(f32::INFINITY, f32::min);
        if min_dist < self.config.output.min_save_distance { return; }

        let rgb = apply_colormap(&escape_times, self.config.rendering.max_iter,
                                 &self.config.rendering.colormap);
        let name     = format!("{:016x}", genome.id);
        let png_path = self.config.output.save_dir.join(format!("{name}.png"));
        save_png(&rgb, w, h, &png_path).unwrap_or(());

        let genome = &self.population[idx];
        save_genome(genome, &nn_path).unwrap_or(());
        self.save_descriptors.push(desc);

        let mut g = genome.clone();
        g.fitness = beauty;
        display::print_save(&g, &png_path.display().to_string(), beauty);
        self.saved_count += 1;
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
                Genome::crossover(a, &self.population[b_idx], &mut self.rng)
                    .mutate(&self.config, &mut self.rng)
            } else {
                a.mutate(&self.config, &mut self.rng)
            };
            new_pop.push(child);
        }
        self.population = new_pop;
    }
}
