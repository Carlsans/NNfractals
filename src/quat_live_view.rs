//! Live rotating 3D view of a quaternion DAG genome, GUI-agnostic (returns
//! RGB bytes; the caller uploads them to whatever texture type it uses).
//!
//! Exists so the rating UI can show exactly what the taste model is trained
//! on: the stored `_view_*` set is a grid of orbit angles × C values, and the
//! live view walks the SAME orbit (`ORBIT_AXIS`, same radius rule) and the
//! SAME C range (`view_c_half`), just continuously.

use crate::formula::OpNode;
use crate::quat_dag::{QuatDagFormula, RaymarchDagParams};
use crate::quat_fractal::TimeAxis;
use crate::quat_raymarch::RaymarchOrbitParams;
use crate::render_gpu_raymarch_dag_codegen::CompiledDagPipeline;

/// Orbit axis shared with explorer's `render_genome_thumbnails` / `render_genome_views`.
pub const ORBIT_AXIS: (f64, f64, f64) = (0.35, 1.0, 0.15);
pub const DOMAIN_RADIUS: f64 = 1.6;
pub const FOV_DEG: f64 = 45.0;
pub const BG_COLOR: (f32, f32, f32) = (0.03, 0.02, 0.06);
/// Stored view grid: orbit angles × C values (angle-major file index).
pub const VIEW_ANGLES: usize = 6;
pub const VIEW_C_VALUES: usize = 4;

/// Half-range of the C sweep: same bailout-scaled convention as
/// `quat_viewer::CRangeScan::start`'s initial probe step.
pub fn view_c_half(bailout_radius: f64) -> f64 {
    (bailout_radius * 0.25).max(0.02)
}

/// The `VIEW_C_VALUES` C values of the stored grid, evenly spaced over
/// `[-c_half, +c_half]`.
pub fn view_c_values(bailout_radius: f64) -> [f64; VIEW_C_VALUES] {
    let h = view_c_half(bailout_radius);
    let mut out = [0.0; VIEW_C_VALUES];
    for (i, o) in out.iter_mut().enumerate() {
        *o = -h + 2.0 * h * i as f64 / (VIEW_C_VALUES - 1) as f64;
    }
    out
}

/// Same framing rule as explorer's `recommended_orbit_radius` (FILL_FRAC=0.82).
pub fn orbit_radius(width: u32, height: u32) -> f64 {
    const FILL_FRAC: f64 = 0.82;
    let aspect = width as f64 / height.max(1) as f64;
    let half_fov_y = (FOV_DEG / 2.0).to_radians();
    let half_fov_x = (half_fov_y.tan() * aspect).atan();
    let tight = half_fov_y.min(half_fov_x);
    DOMAIN_RADIUS / (FILL_FRAC * tight).sin()
}

pub struct LiveQuatView {
    pipeline: CompiledDagPipeline,
    prog: Vec<OpNode>,
    warp: Vec<OpNode>,
    julia: bool,
    jc: (f32, f32),
    phoenix: (f32, f32),
    bailout: f64,
    colormap: String,
    max_iter: u32,
}

impl LiveQuatView {
    /// `None` if the genome has no DAG program or no GPU is available.
    pub fn new(g: &crate::genome::Genome) -> Option<Self> {
        if g.program.is_empty() {
            return None;
        }
        let pipeline = CompiledDagPipeline::compile(&g.program, &g.warp)?;
        Some(Self {
            pipeline,
            prog: g.program.clone(),
            warp: g.warp.clone(),
            julia: g.julia_mode,
            jc: (g.julia_cre, g.julia_cim),
            phoenix: (g.phoenix_re, g.phoenix_im),
            bailout: g.bailout_radius as f64,
            colormap: "lava".to_string(),
            max_iter: 50,
        })
    }

    pub fn c_half(&self) -> f64 {
        view_c_half(self.bailout)
    }

    /// Renders one square frame. `turn` is the orbit position in turns
    /// (0..1 = one full orbit, identical to `RaymarchOrbitParams::sample`),
    /// `c` the time-axis value. Returns `size*size*3` RGB bytes.
    pub fn render_rgb(&mut self, turn: f64, c: f64, size: u32) -> Vec<u8> {
        let formula = QuatDagFormula { prog: &self.prog, warp: &self.warp, julia: self.julia, jc: self.jc, phoenix: self.phoenix };
        let params = RaymarchDagParams {
            formula,
            time_axis: TimeAxis::C,
            time_val: c,
            domain_radius: DOMAIN_RADIUS,
            box_bounds: None, axis_assignment: None, clip_plane: None, slide_iter: None, hide_above: None,
            max_iter: self.max_iter,
            bailout: self.bailout,
            max_march_steps: 150,
            hit_epsilon: DOMAIN_RADIUS * 1e-4,
            step_safety: 0.8,
            light_dir: (0.5, 0.8, 0.3),
            normal_eps: DOMAIN_RADIUS * 1e-3,
            color_probe_offset: DOMAIN_RADIUS * 1e-2,
            aa: 1,
        };
        let orbit = RaymarchOrbitParams {
            target: (0.0, 0.0, 0.0),
            axis: ORBIT_AXIS,
            radius: orbit_radius(size, size),
            turns: 1.0,
            phase0: 0.0,
            fov_y: FOV_DEG.to_radians(),
        };
        let cam = orbit.sample(turn.rem_euclid(1.0));
        let (shading, color_t) = self.pipeline.render(&params, &cam, size, size);
        let rgb_bytes = crate::colormap::apply_colormap_equalized(&color_t, self.max_iter, &self.colormap);
        let mut rgb = vec![0u8; shading.len() * 3];
        for i in 0..shading.len() {
            let v = shading[i];
            for k in 0..3 {
                rgb[i * 3 + k] = if v <= 0.0 {
                    ([BG_COLOR.0, BG_COLOR.1, BG_COLOR.2][k] * 255.0) as u8
                } else {
                    (rgb_bytes[i * 3 + k] as f32 * v).clamp(0.0, 255.0) as u8
                };
            }
        }
        rgb
    }
}
