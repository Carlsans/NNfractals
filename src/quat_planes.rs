//! 2D slices of a quaternion DAG fractal: the 4D point `(R, A, B, C)` is
//! varied along two axes and the other two are held fixed, giving one
//! escape-time image per axis pair — six planes in all. CPU-only and
//! GUI-agnostic (returns RGB bytes), so the anim viewer can show small
//! thumbnails of all six and render one at any resolution on demand.
//!
//! The escape function is `quat_dag::quat_dag_escape_de`, the same one the
//! ray-marcher uses per point, so a plane is exactly a flat cut through the
//! volume that the 3D view shows (no lighting, just escape time).

use rayon::prelude::*;

use crate::quat_dag::{quat_dag_escape_de, QuatDagFormula};
use crate::quaternion::Quat;

pub const AXIS_NAMES: [&str; 4] = ["R", "A", "B", "C"];

/// The six unordered axis pairs, in display order (first axis is
/// horizontal, second vertical).
pub const PLANES: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

pub fn plane_name(axes: (usize, usize)) -> String {
    format!("{}{}", AXIS_NAMES[axes.0], AXIS_NAMES[axes.1])
}

#[derive(Clone, Debug)]
pub struct PlaneSpec {
    pub axes: (usize, usize),
    /// Values of all four components; the two plane axes are overwritten
    /// per pixel, the other two stay fixed.
    pub fixed: [f64; 4],
    /// Centre of the view along each component (only the two plane axes
    /// are used); lets the viewer zoom/pan a plane.
    pub center: [f64; 4],
    /// Half-width of the visible square, in fractal units.
    pub extent: f64,
    pub max_iter: u32,
    pub bailout: f64,
    pub colormap: String,
}

/// Renders one plane as `size*size*3` RGB bytes (row 0 = top, +v up).
pub fn render_plane(f: &QuatDagFormula, spec: &PlaneSpec, size: u32) -> Vec<u8> {
    let n = size as usize;
    let bailout_sq = spec.bailout * spec.bailout;
    let (i, j) = spec.axes;
    let times: Vec<f32> = (0..n * n)
        .into_par_iter()
        .map(|idx| {
            let (x, y) = (idx % n, idx / n);
            let u = spec.center[i] - spec.extent + 2.0 * spec.extent * (x as f64 + 0.5) / n as f64;
            let v = spec.center[j] + spec.extent - 2.0 * spec.extent * (y as f64 + 0.5) / n as f64;
            let mut p = spec.fixed;
            p[i] = u;
            p[j] = v;
            quat_dag_escape_de(f, Quat::new(p[0], p[1], p[2], p[3]), spec.max_iter, bailout_sq).0
        })
        .collect();
    crate::colormap::apply_colormap_equalized(&times, spec.max_iter, &spec.colormap)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formula_text::parse_program;

    #[test]
    fn plane_varies_only_along_its_two_axes() {
        // z^2 + c, Mandelbrot mode: in the RA plane (B=C=0) the picture is
        // the classic Mandelbrot set, symmetric about the R axis (v -> -v).
        let prog = parse_program("z^2 + c").unwrap();
        let f = QuatDagFormula { prog: &prog, warp: &[], julia: false, jc: (0.0, 0.0), phoenix: (0.0, 0.0) };
        let spec = PlaneSpec { axes: (0, 1), fixed: [0.0; 4], center: [0.0; 4], extent: 2.0, max_iter: 40, bailout: 4.0, colormap: "turbo".into() };
        let n = 64usize;
        let rgb = render_plane(&f, &spec, n as u32);
        assert_eq!(rgb.len(), n * n * 3);
        for y in 0..n / 2 {
            let (top, bot) = (y * n * 3, (n - 1 - y) * n * 3);
            assert_eq!(rgb[top..top + n * 3], rgb[bot..bot + n * 3], "row {y} not mirrored");
        }
        // And it is not a flat image.
        assert!(rgb.chunks(3).any(|p| p != &rgb[0..3]));
    }

    #[test]
    fn six_distinct_planes() {
        let mut s: Vec<_> = PLANES.to_vec();
        s.sort();
        s.dedup();
        assert_eq!(s.len(), 6);
        assert_eq!(plane_name((1, 3)), "AC");
    }
}

#[cfg(test)]
mod visual {
    use super::*;

    /// `PLANES_GENOME=<file.nn> PLANES_OUT=<dir> cargo test --release --lib quat_planes::visual -- --ignored`
    /// writes the six planes as PNGs for eyeballing.
    #[test]
    #[ignore]
    fn dump_planes() {
        let (Ok(g), Ok(out)) = (std::env::var("PLANES_GENOME"), std::env::var("PLANES_OUT")) else { return };
        let g = crate::io::load_genome(std::path::Path::new(&g)).unwrap();
        let f = QuatDagFormula { prog: &g.program, warp: &g.warp, julia: g.julia_mode, jc: (g.julia_cre, g.julia_cim), phoenix: (g.phoenix_re, g.phoenix_im) };
        for (k, &axes) in PLANES.iter().enumerate() {
            let spec = PlaneSpec { axes, fixed: [0.0; 4], center: [0.0; 4], extent: 1.6, max_iter: 60, bailout: g.bailout_radius as f64, colormap: "lava".into() };
            let rgb = render_plane(&f, &spec, 256);
            image::save_buffer(format!("{out}/{k}_{}.png", plane_name(axes)), &rgb, 256, 256, image::ColorType::Rgb8).unwrap();
        }
    }
}
