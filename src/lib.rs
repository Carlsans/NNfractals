pub mod config;
pub mod dd;
pub mod formula;
pub mod genome;
pub mod fractal;
pub mod recursion_model;
pub mod colormap;
pub mod fitness;
pub mod io;
pub mod display;
pub mod optimizer;
pub mod aesthetic;
pub mod formula_usage;
#[cfg(any(feature = "viewer", feature = "browser", feature = "launcher"))]
pub mod gui_font;
#[cfg(feature = "wgpu-backend")]
pub mod render_gpu;
