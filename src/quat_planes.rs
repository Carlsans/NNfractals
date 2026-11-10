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

use crate::quat_dag::{quat_dag_escape_de_mode, QuatDagFormula};
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
    /// Zero C at every iteration, as the `proj3` projection does, so the tiles
    /// show the escape field the 3D view (and its zone picks) uses.
    pub zero_c: bool,
}

/// Renders one plane as `size*size*3` RGB bytes (row 0 = top, +v up).
pub fn render_plane(f: &QuatDagFormula, spec: &PlaneSpec, size: u32) -> Vec<u8> {
    let times = plane_escape_times(f, spec, size);
    crate::colormap::apply_colormap_equalized(&times, spec.max_iter, &spec.colormap)
}

/// The escape time of every pixel of a plane (`size*size`, row 0 = top);
/// `render_plane` colours exactly these.
pub fn plane_escape_times(f: &QuatDagFormula, spec: &PlaneSpec, size: u32) -> Vec<f32> {
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
            quat_dag_escape_de_mode(f, Quat::new(p[0], p[1], p[2], p[3]), spec.max_iter, bailout_sq, spec.zero_c).0
        })
        .collect();
    times
}

/// The kept-pixel mask of a `size*size` plane with the speckle filter's 2D
/// counterpart applied: a kept pixel stays kept only when at least a third of
/// the `(2r+1)²` window around it is kept too (`r` = 0: unchanged). A
/// preview of what the 3D speckle filter removes, not the same test.
pub fn despeckle_mask(kept: &[bool], size: usize, r: usize) -> Vec<bool> {
    if r == 0 {
        return kept.to_vec();
    }
    // Summed-area table for O(1) window counts.
    let w = size + 1;
    let mut sat = vec![0u32; w * w];
    for y in 0..size {
        let mut row = 0u32;
        for x in 0..size {
            row += kept[y * size + x] as u32;
            sat[(y + 1) * w + x + 1] = sat[y * w + x + 1] + row;
        }
    }
    (0..size * size).map(|i| {
        if !kept[i] { return false; }
        let (x, y) = (i % size, i / size);
        let (x0, y0, x1, y1) = (x.saturating_sub(r), y.saturating_sub(r), (x + r + 1).min(size), (y + r + 1).min(size));
        let n = sat[y1 * w + x1] + sat[y0 * w + x0] - sat[y0 * w + x1] - sat[y1 * w + x0];
        3 * n as usize >= (x1 - x0) * (y1 - y0)
    }).collect()
}

/// `rgb` with the zones `keep` accepts standing out (zone picking): the
/// other pixels become a dark grey of their own brightness, and a kept pixel
/// too dark to see against that is lifted toward cyan.
pub fn dim_unkept(rgb: &[u8], times: &[f32], keep: impl Fn(f64) -> bool) -> Vec<u8> {
    let kept: Vec<bool> = times.iter().map(|&t| keep(t as f64)).collect();
    dim_unkept_mask(rgb, &kept)
}

/// `dim_unkept` with the kept pixels given directly.
pub fn dim_unkept_mask(rgb: &[u8], kept: &[bool]) -> Vec<u8> {
    // Nothing left out: the plain picture (the cyan lift would only recolour it).
    if kept.iter().all(|&k| k) {
        return rgb.to_vec();
    }
    let mut out = rgb.to_vec();
    for (i, &k) in kept.iter().enumerate() {
        let px = &mut out[i * 3..i * 3 + 3];
        let lum = (px[0] as u32 * 3 + px[1] as u32 * 6 + px[2] as u32) / 10;
        if !k {
            let g = (8 + lum / 6) as u8;
            px.copy_from_slice(&[g, g, g]);
        } else if lum < 110 {
            for (c, h) in px.iter_mut().zip([90u32, 210, 255]) {
                *c = ((*c as u32 + h) / 2) as u8;
            }
        }
    }
    out
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
        let spec = PlaneSpec { axes: (0, 1), fixed: [0.0; 4], center: [0.0; 4], extent: 2.0, max_iter: 40, bailout: 4.0, colormap: "turbo".into(), zero_c: false };
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
    fn dimming_keeps_picked_pixels_bright_and_greys_the_rest() {
        let rgb = [250, 200, 50, 30, 0, 0, 250, 250, 250];
        let times = [5.0f32, 6.0, 9.0];
        let out = dim_unkept(&rgb, &times, |t| t < 7.0);
        assert_eq!(&out[0..3], &rgb[0..3], "a bright kept pixel is unchanged");
        assert!(out[4] > 90 && out[5] > 90, "a dark kept pixel is lifted: {:?}", &out[3..6]);
        assert!(out[6] == out[7] && out[7] == out[8] && out[6] < 60, "an unkept pixel is dark grey: {:?}", &out[6..9]);
        assert_eq!(dim_unkept(&rgb, &times, |_| true), rgb.to_vec(), "keeping everything leaves the picture alone");
    }

    #[test]
    fn proj3_tiles_use_the_projected_field_the_3d_view_uses() {
        // z^2 + c on the RC plane: with C zeroed every step (proj3) the escape
        // times must match the 3D view's evaluation, which differs from the raw one.
        let prog = parse_program("z^2 + c").unwrap();
        let f = QuatDagFormula { prog: &prog, warp: &[], julia: false, jc: (0.0, 0.0), phoenix: (0.0, 0.0) };
        let mut spec = PlaneSpec { axes: (0, 3), fixed: [0.0; 4], center: [0.0; 4], extent: 1.5, max_iter: 30, bailout: 4.0, colormap: "turbo".into(), zero_c: false };
        let raw = plane_escape_times(&f, &spec, 32);
        spec.zero_c = true;
        let proj = plane_escape_times(&f, &spec, 32);
        assert_ne!(raw, proj, "zeroing C changes the field");
        // Pixel (x, y) of the RC plane is the 4D point (u, 0, 0, v).
        let (x, y) = (5usize, 9usize);
        let u = -1.5 + 3.0 * (x as f64 + 0.5) / 32.0;
        let v = 1.5 - 3.0 * (y as f64 + 0.5) / 32.0;
        let want = crate::quat_dag::quat_dag_escape_de_mode(&f, Quat::new(u, 0.0, 0.0, v), 30, 16.0, true).0;
        assert_eq!(proj[y * 32 + x], want);
    }

    #[test]
    fn despeckle_drops_isolated_pixels_and_keeps_solid_areas() {
        let n = 12;
        let mut kept = vec![false; n * n];
        for y in 2..8 { for x in 2..8 { kept[y * n + x] = true; } } // a solid 6x6 block
        kept[10 * n + 10] = true; // a lone speck
        let out = despeckle_mask(&kept, n, 1);
        assert!(!out[10 * n + 10], "the speck goes");
        assert!(out[4 * n + 4] && out[2 * n + 2], "the block stays, corners included");
        assert_eq!(despeckle_mask(&kept, n, 0), kept);
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
            let spec = PlaneSpec { axes, fixed: [0.0; 4], center: [0.0; 4], extent: 1.6, max_iter: 60, bailout: g.bailout_radius as f64, colormap: "lava".into(), zero_c: false };
            let rgb = render_plane(&f, &spec, 256);
            image::save_buffer(format!("{out}/{k}_{}.png", plane_name(axes)), &rgb, 256, 256, image::ColorType::Rgb8).unwrap();
        }
    }
}
