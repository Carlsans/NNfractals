mod config;
mod cppn;
mod formula;
mod genome;
mod transformer;
mod fractal;
mod colormap;
mod fitness;
mod io;
mod display;
mod optimizer;
mod aesthetic;

use std::path::PathBuf;
use clap::Parser;
use config::Config;
use fractal::render_cpu;
use colormap::apply_colormap;
use io::{load_genome, save_png};

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
        run_evolution(config);
    }

    Ok(())
}

#[cfg(feature = "wgpu-backend")]
fn run_evolution(config: Config) {
    use burn::backend::{Autodiff, wgpu::{Wgpu, WgpuDevice}};
    type B = Autodiff<Wgpu>;
    let device = WgpuDevice::default();
    let mut opt = optimizer::Optimizer::<B>::new(config, device);
    ctrlc_cleanup();
    opt.run_forever();
}

#[cfg(all(not(feature = "wgpu-backend"), feature = "ndarray-backend"))]
fn run_evolution(config: Config) {
    use burn::backend::{Autodiff, ndarray::{NdArray, NdArrayDevice}};
    type B = Autodiff<NdArray>;
    let device = NdArrayDevice;
    let mut opt = optimizer::Optimizer::<B>::new(config, device);
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
