//! Flat 2D slices through the 3D view: the escape-time field on a plane in
//! view space (X/Y/Z as the anim viewer draws them, after the axis assignment,
//! the 4D→3D projection and T), the same field the 3D render marches through
//! (`quat_dag::escape_time_at`). A plane starts as XY, YZ or XZ through the
//! box centre and can be tilted, spun, moved along its normal and zoomed.
//! CPU only (rayon), so a small preview is cheap and a large one is a file.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::quat_dag::{escape_time_at, RaymarchDagParams};
use crate::quat_motion::{add, cross, dot, normalize, scale, Vec3};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum BasePlane {
    #[default]
    XY,
    YZ,
    XZ,
}

impl BasePlane {
    pub const ALL: [BasePlane; 3] = [BasePlane::XY, BasePlane::YZ, BasePlane::XZ];
    pub fn label(self) -> &'static str {
        match self { BasePlane::XY => "XY", BasePlane::YZ => "YZ", BasePlane::XZ => "XZ" }
    }
    /// (right, up, normal) of the untilted plane.
    fn frame(self) -> (Vec3, Vec3, Vec3) {
        match self {
            BasePlane::XY => ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            BasePlane::YZ => ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
            BasePlane::XZ => ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
        }
    }
}

/// Where the slice is: a base plane, tilted (degrees) about its right and up
/// axes, spun about its normal, moved along the normal (`offset`, in box
/// half-sizes) and zoomed (`zoom` 1 = the square just covers the box).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SliceSpec {
    pub base: BasePlane,
    pub tilt_right: f64,
    pub tilt_up: f64,
    pub spin: f64,
    pub offset: f64,
    pub zoom: f64,
}

impl Default for SliceSpec {
    fn default() -> Self { SliceSpec { base: BasePlane::XY, tilt_right: 0.0, tilt_up: 0.0, spin: 0.0, offset: 0.0, zoom: 1.0 } }
}

/// The plane in view space: its centre, and the half-extent vectors to the right
/// and top edges of the square (`u`, `v`); the unit normal is `normal`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SliceFrame {
    pub center: Vec3,
    pub u: Vec3,
    pub v: Vec3,
    pub normal: Vec3,
}

impl SliceFrame {
    /// The view-space point of pixel `(x, y)` of an `n`×`n` image (row 0 = top).
    pub fn point(&self, x: usize, y: usize, n: usize) -> Vec3 {
        let a = 2.0 * (x as f64 + 0.5) / n as f64 - 1.0;
        let b = 1.0 - 2.0 * (y as f64 + 0.5) / n as f64;
        add(self.center, add(scale(self.u, a), scale(self.v, b)))
    }

    /// The square's corners (top-left, top-right, bottom-right, bottom-left).
    pub fn corners(&self) -> [Vec3; 4] {
        let (c, u, v) = (self.center, self.u, self.v);
        let m = |s: f64, t: f64| add(c, add(scale(u, s), scale(v, t)));
        [m(-1.0, 1.0), m(1.0, 1.0), m(1.0, -1.0), m(-1.0, -1.0)]
    }
}

/// `v` turned by `deg` degrees about the unit axis `k` (Rodrigues).
fn rotate(v: Vec3, k: Vec3, deg: f64) -> Vec3 {
    let (s, c) = deg.to_radians().sin_cos();
    add(add(scale(v, c), scale(cross(k, v), s)), scale(k, dot(k, v) * (1.0 - c)))
}

impl SliceSpec {
    /// The plane for a render box `(min, max)`.
    pub fn frame(&self, bmin: Vec3, bmax: Vec3) -> SliceFrame {
        let center = scale(add(bmin, bmax), 0.5);
        let half = 0.5 * (bmax.0 - bmin.0).abs().max((bmax.1 - bmin.1).abs()).max((bmax.2 - bmin.2).abs()).max(1e-9);
        let (u0, v0, n0) = self.base.frame();
        let (v1, n1) = (rotate(v0, u0, self.tilt_right), rotate(n0, u0, self.tilt_right));
        let (u2, n2) = (rotate(u0, v1, self.tilt_up), rotate(n1, v1, self.tilt_up));
        let (u3, v3) = (rotate(u2, n2, self.spin), rotate(v1, n2, self.spin));
        let h = half / self.zoom.max(1e-6);
        SliceFrame {
            center: add(center, scale(n2, self.offset * half)),
            u: scale(normalize(u3), h),
            v: scale(normalize(v3), h),
            normal: normalize(n2),
        }
    }
}

/// The escape time of every pixel of an `n`×`n` slice (row 0 = top).
pub fn slice_escape_times(p: &RaymarchDagParams, frame: &SliceFrame, n: u32) -> Vec<f32> {
    let n = n as usize;
    (0..n * n).into_par_iter().map(|i| escape_time_at(p, frame.point(i % n, i / n, n))).collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SliceColoring {
    /// The viewer's colour map, histogram-equalised like the Planes tiles.
    #[default]
    Colours,
    /// The same, in grey.
    Grey,
    /// Black where the point never escapes (the set), white elsewhere.
    BlackWhite,
}

impl SliceColoring {
    pub const ALL: [SliceColoring; 3] = [SliceColoring::Colours, SliceColoring::Grey, SliceColoring::BlackWhite];
    pub fn label(self) -> &'static str {
        match self { SliceColoring::Colours => "colours", SliceColoring::Grey => "grey", SliceColoring::BlackWhite => "black & white" }
    }
}

/// RGB bytes for slice escape times.
pub fn colorize(times: &[f32], max_iter: u32, coloring: SliceColoring, colormap: &str) -> Vec<u8> {
    match coloring {
        SliceColoring::Colours => crate::colormap::apply_colormap_equalized(times, max_iter, colormap),
        SliceColoring::Grey => crate::colormap::apply_colormap_equalized(times, max_iter, "grayscale"),
        SliceColoring::BlackWhite => times.iter().flat_map(|&t| { let v = if t >= max_iter as f32 { 0u8 } else { 255u8 }; [v, v, v] }).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(prog: &[crate::formula::OpNode]) -> RaymarchDagParams<'_> {
        RaymarchDagParams {
            formula: crate::quat_dag::QuatDagFormula { prog, warp: &[], julia: false, jc: (0.0, 0.0), phoenix: (0.0, 0.0) },
            time_axis: crate::quat_fractal::TimeAxis::C, time_val: 0.0, domain_radius: 2.0,
            box_bounds: Some(((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0))), axis_assignment: None, clip_plane: None,
            slide_iter: None, hide_above: None, zone: None, axis_remap: None, collapse: None, max_iter: 40, bailout: 4.0, max_march_steps: 200,
            hit_epsilon: 1e-4, pixel_lod: 0.0, solid_interior: true, step_safety: 0.8, light_dir: (0.5, 0.8, 0.3), normal_eps: 1e-3, color_probe_offset: 1e-2, aa: 1,
        }
    }

    #[test]
    fn the_untilted_xy_slice_is_the_mandelbrot_set_and_tilts_keep_it_a_square() {
        // z^2 + c with the slice axes as (R, A, B): the XY plane at the centre is
        // the classic Mandelbrot set (-2 is in it, 1 is not).
        let prog = crate::formula_text::parse_program("z^2 + c").unwrap();
        let p = params(&prog);
        let spec = SliceSpec::default();
        let f = spec.frame((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0));
        let n = 64;
        let t = slice_escape_times(&p, &f, n);
        let at = |x: f64, y: f64| { let px = ((x + 2.0) / 4.0 * n as f64) as usize; let py = ((2.0 - y) / 4.0 * n as f64) as usize; t[py * n as usize + px] };
        assert!(at(-0.2, 0.0) >= 40.0 && at(1.2, 0.0) < 40.0, "inside at -0.2, outside at 1.2");
        let bw = colorize(&t, 40, SliceColoring::BlackWhite, "lava");
        assert!(bw.chunks(3).all(|c| c == [0, 0, 0] || c == [255, 255, 255]));
        // Tilts and spins keep the frame a square of the same size, normal ⟂ plane.
        let g = SliceSpec { tilt_right: 30.0, tilt_up: -50.0, spin: 70.0, offset: 0.3, zoom: 2.0, ..spec }.frame((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0));
        let len = |a: Vec3| dot(a, a).sqrt();
        assert!((len(g.u) - 1.0).abs() < 1e-9 && (len(g.v) - 1.0).abs() < 1e-9, "zoom 2 halves the half-size 2");
        assert!(dot(g.u, g.v).abs() < 1e-9 && dot(g.u, g.normal).abs() < 1e-9 && dot(g.v, g.normal).abs() < 1e-9);
        assert!((dot(g.center, g.normal) - 0.6).abs() < 1e-9, "offset 0.3 of the half-size 2 along the normal");
    }
}
