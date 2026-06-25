use crossterm::{
    cursor::{Hide, MoveTo, Show},
    terminal::{Clear, ClearType},
    ExecutableCommand,
};
use std::io::{stdout, Write};
use crate::genome::Genome;

pub fn init() {
    let mut out = stdout();
    let _ = out.execute(Hide);
    let _ = out.execute(Clear(ClearType::All));
    let _ = out.execute(MoveTo(0, 0));
    println!("┌──────────────────────────────────────────────────────────────────────┐");
    println!("│                       NNfractals  Evolution                          │");
    println!("└──────────────────────────────────────────────────────────────────────┘");
}

pub fn teardown() {
    let _ = stdout().execute(Show);
}

/// Overwrite row 3 (the blank line below the header banner) with a live activity message.
/// Called frequently during a generation so the user can see the program is running.
pub fn print_status(msg: &str) {
    let mut out = stdout();
    let _ = out.execute(MoveTo(0, 3));
    print!(" {:<70}", msg);
    let _ = out.flush();
}

pub fn refresh(
    generation: u64,
    population: &[Genome],
    saved_count: u64,
    elapsed_secs: u64,
    stagnant_gens: u64,
    best_ever_fitness: f32,
    aesthetic_line: Option<&str>,
    sub_scores: Option<&[f32; 5]>,
) {
    let mut out = stdout();
    let _ = out.execute(MoveTo(0, 4));

    let fitnesses: Vec<f32> = population.iter().map(|g| g.fitness).collect();
    let best  = fitnesses.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let worst = fitnesses.iter().cloned().fold(f32::INFINITY,     f32::min);
    let avg   = fitnesses.iter().sum::<f32>() / fitnesses.len().max(1) as f32;

    let h = elapsed_secs / 3600;
    let m = (elapsed_secs % 3600) / 60;
    let s = elapsed_secs % 60;

    println!(
        " Gen {:>8}   Saved {:>5}   Elapsed {:02}:{:02}:{:02}          ",
        generation, saved_count, h, m, s
    );
    println!(
        " Fitness ▸  best {:>7.4}   avg {:>7.4}   worst {:>7.4}   ",
        best, avg, worst
    );
    let unique_formulas = {
        let mut seen = std::collections::HashSet::new();
        population.iter().filter(|g| seen.insert(g.formula_ops_label())).count()
    };
    println!(
        " Best-ever {:>7.4}   Stagnant {:>4} gen{}     ",
        best_ever_fitness,
        stagnant_gens,
        if stagnant_gens > 15 { " ← nearing restart" } else { "                  " }
    );
    println!(
        " Formula diversity: {}/{} unique                              ",
        unique_formulas, population.len()
    );
    println!(
        " Aesthetic ▸ {:<57}",
        aesthetic_line.unwrap_or("")
    );
    if let Some(s) = sub_scores {
        fn bar(v: f32) -> String {
            let filled = (v * 10.0).round() as usize;
            format!("{:░<10}", "█".repeat(filled.min(10)))
        }
        println!(
            " Beauty sub  bnd:{:.2}{} edg:{:.2}{} ent:{:.2}{} sim:{:.2}{} coo:{:.2}{}",
            s[0], bar(s[0]), s[1], bar(s[1]), s[2], bar(s[2]), s[3], bar(s[3]), s[4], bar(s[4])
        );
    } else {
        println!("{:72}", "");
    }
    println!();
    println!(
        " {:>4}  {:>18}  {:>8}  {:<22}  {}",
        "Rank", "ID", "Fitness", "Formula", "Tag  "
    );
    println!(" {}", "─".repeat(70));

    let mut ranked: Vec<(usize, f32)> = fitnesses
        .iter()
        .enumerate()
        .map(|(i, &f)| (i, f))
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    for (rank, (idx, fitness)) in ranked.iter().take(15).enumerate() {
        let tag = match rank {
            0     => "★ best",
            1..=3 => "elite ",
            _     => "      ",
        };
        let ops = population[*idx].formula_ops_label();
        let formula_col = if ops.chars().count() > 22 {
            let s: String = ops.chars().take(21).collect();
            format!("{}…", s)
        } else {
            ops
        };
        println!(
            " {:>4}  {:>18x}  {:>8.4}  {:<22}  {}",
            rank + 1,
            population[*idx].id,
            fitness,
            formula_col,
            tag
        );
    }
    // Blank lines to clear any leftover from a longer previous render
    for _ in ranked.len()..16 {
        println!("{:72}", "");
    }
    let _ = out.flush();
}

pub fn print_restart(generation: u64, best_fitness: f32) {
    let mut out = stdout();
    let _ = out.execute(MoveTo(0, 28));
    println!(
        " ↺ RESTART  gen={}  best-ever saved  fitness={:.4}     ",
        generation, best_fitness
    );
    let _ = out.flush();
}

pub fn print_save(genome: &Genome, png_path: &str, beauty: f32) {
    let mut out = stdout();
    let _ = out.execute(MoveTo(0, 27));
    println!(
        " ✓ SAVED  id={:x}  beauty={:.3}  {}     ",
        genome.id, beauty, png_path
    );
    let _ = out.flush();
}
