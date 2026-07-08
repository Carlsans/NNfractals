use std::fs::OpenOptions;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;
use clap::Parser;
use nnfractals::config::Config;
use nnfractals::fractal::render_cpu;
use nnfractals::colormap::apply_colormap;
use nnfractals::io::{load_genome, save_png};
use nnfractals::display;
#[cfg(feature = "wgpu-backend")]
use nnfractals::render_gpu;
use nnfractals::optimizer;

#[derive(Parser)]
#[command(name = "nnfractals", about = "Neural-network fractal evolution")]
struct Args {
    /// Render a saved .nn file instead of running the evolution loop
    #[arg(long)]
    render: Option<PathBuf>,

    /// Output width for --render (0 = use config default)
    #[arg(long, default_value = "0")]
    width: u32,

    /// Output height for --render (0 = use config default)
    #[arg(long, default_value = "0")]
    height: u32,

    /// Path to config file
    #[arg(long, default_value = "config.toml")]
    config: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let config = Config::load(&args.config)?;

    if let Some(nn_path) = args.render {
        // ── Single render mode ───────────────────────────────────────────
        let genome = load_genome(&nn_path)?;
        let w = if args.width > 0 { args.width } else { config.rendering.default_width };
        let h = if args.height > 0 { args.height } else { config.rendering.default_height };

        eprintln!("Rendering {}×{} from {:?}…", w, h, nn_path);
        let escape_times = render_cpu(&genome, &config, w, h);
        let rgb = apply_colormap(&escape_times, config.rendering.max_iter, &config.rendering.colormap);

        let out_path = nn_path.with_extension("png");
        save_png(&rgb, w, h, &out_path)?;
        eprintln!("Saved → {:?}", out_path);
    } else {
        // ── Evolution loop ───────────────────────────────────────────────
        spawn_dedup_cleaner(&config);
        run_evolution(config);
    }

    Ok(())
}

/// Background thread: every `dedup.interval_hours`, run the near-duplicate cleaner
/// over the save dir so the pool stays diverse while evolution runs. Output is
/// redirected to `dedup.log` to avoid corrupting the live TUI. 0 hours disables it.
fn spawn_dedup_cleaner(config: &Config) {
    let hours = config.dedup.interval_hours;
    if hours <= 0.0 {
        return;
    }
    let interval   = Duration::from_secs_f64((hours * 3600.0) as f64);
    let threshold  = config.dedup.similarity_threshold;
    let save_dir   = config.output.save_dir.clone();
    let exe        = std::env::current_exe().ok();
    // Per-instance log so concurrent instances (separate save_dirs) don't
    // interleave their dedup output into one file.
    let log_name = format!(
        "dedup_{}.log",
        save_dir.file_name().and_then(|s| s.to_str()).unwrap_or("pool")
    );

    thread::spawn(move || {
        loop {
            thread::sleep(interval);

            let log = OpenOptions::new()
                .create(true).append(true)
                .open(&log_name)
                .ok();
            let (out, err) = match &log {
                Some(f) => (
                    Stdio::from(f.try_clone().expect("clone log handle")),
                    Stdio::from(f.try_clone().expect("clone log handle")),
                ),
                None => (Stdio::null(), Stdio::null()),
            };

            let mut cmd = Command::new(nnfractals::python_bin(std::path::Path::new(".")));
            cmd.arg("scripts/dedup.py")
                .arg("--run")
                .arg("--threshold").arg(format!("{threshold}"))
                .arg("--dir").arg(&save_dir)
                .stdout(out)
                .stderr(err);
            if let Some(exe) = &exe {
                cmd.arg("--binary").arg(exe);
            }
            // Best-effort: a failed launch (e.g. no python3) is logged and the
            // loop simply tries again next interval.
            let _ = cmd.status();
        }
    });
}

#[cfg(feature = "wgpu-backend")]
fn run_evolution(config: Config) {
    // Init fractal GPU renderer (wgpu compute; no burn/autodiff needed anymore).
    render_gpu::init_gpu();
    let mut opt = optimizer::Optimizer::new(config);
    ctrlc_cleanup();
    opt.run_forever();
}

#[cfg(all(not(feature = "wgpu-backend"), feature = "ndarray-backend"))]
fn run_evolution(config: Config) {
    let mut opt = optimizer::Optimizer::new(config);
    ctrlc_cleanup();
    opt.run_forever();
}

#[cfg(not(any(feature = "wgpu-backend", feature = "ndarray-backend")))]
fn run_evolution(_config: Config) {
    eprintln!("No backend feature enabled. Build with --features wgpu-backend or ndarray-backend.");
    std::process::exit(1);
}

fn ctrlc_cleanup() {
    ctrlc::set_handler(move || {
        display::teardown();
        std::process::exit(0);
    })
    .unwrap_or(());
}
