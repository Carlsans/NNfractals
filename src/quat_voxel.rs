//! Voxelizes a quaternion fractal (fixed C, the "time" axis held still) into
//! a dense (R,A,B) scalar field, extracts an isosurface with Naive Surface
//! Nets, smooths it, decimates it, and writes an STL — a standalone,
//! genuinely 3D artifact of the same quaternion-Mandelbrot renderer used
//! elsewhere in this prototype (`quat_fractal`), reusing its exact escape
//! function for consistency with every rendered clip.
//!
//! Pipeline: scalar field (rayon-parallel) → `fast_surface_nets::surface_nets`
//! (isosurface extraction) → Laplacian smoothing → `meshopt_rs::simplify`
//! (quadric-error decimation) → `stl_io::write_stl`. Every stage after the
//! scalar field operates on a plain (positions, indices) mesh, so smoothing
//! and decimation are simple, swappable passes rather than baked into
//! extraction.

use std::io;
use std::path::Path;
use std::time::Instant;

use fast_surface_nets::ndshape::RuntimeShape;
use fast_surface_nets::{surface_nets, SurfaceNetsBuffer};
use rayon::prelude::*;

use crate::quat_fractal::{quat_escape, QuatFormula, TimeAxis};

pub struct VoxelStlOpts {
    pub formula: QuatFormula,
    /// Which quaternion component is held fixed for this whole volume —
    /// this is the "time" axis everywhere else in the project animates.
    pub time_axis: TimeAxis,
    pub time_val: f64,
    pub res: u32,
    pub domain_extent: f64,
    pub max_iter: u32,
    pub bailout: f64,
    pub smooth_iters: u32,
    pub smooth_alpha: f32,
    pub target_tris: usize,
    pub target_error: f32,
}

#[derive(Debug, Default)]
pub struct VoxelStlReport {
    pub raw_vertices: usize,
    pub raw_triangles: usize,
    pub final_triangles: usize,
    pub field_secs: f64,
    pub mesh_secs: f64,
    pub smooth_secs: f64,
    pub decimate_secs: f64,
    pub write_secs: f64,
}

/// Dense scalar field over the 3 SPATIAL axes (see `time_axis`), `res^3`
/// samples over `[-domain_extent, domain_extent]^3`, the 4th (time)
/// component held fixed at `time_val`. Value is `threshold - escape_time`
/// (negative = never escapes = "inside"), using the exact same smooth
/// escape-time formula (`quat_escape`) as every video render in this
/// project, so the surface it traces is the same boundary you'd see
/// animate in a clip — just solidified.
fn build_scalar_field(opts: &VoxelStlOpts) -> Vec<f32> {
    let res = opts.res as usize;
    let n = res * res * res;
    let half = opts.domain_extent;
    let denom = (res as f64 - 1.0).max(1.0);
    let bailout_sq = opts.bailout * opts.bailout;
    // Just under max_iter: points that never escape return exactly
    // max_iter (flat, no gradient); points near the true boundary approach
    // it smoothly from below, so this threshold sits right in that
    // gradient band, giving surface_nets a well-defined zero crossing.
    let threshold = opts.max_iter as f64 - 0.5;
    (0..n)
        .into_par_iter()
        .map(|idx| {
            let x = idx % res;
            let y = (idx / res) % res;
            let z = idx / (res * res);
            let sx = -half + 2.0 * half * (x as f64) / denom;
            let sy = -half + 2.0 * half * (y as f64) / denom;
            let sz = -half + 2.0 * half * (z as f64) / denom;
            let et = quat_escape(
                opts.formula,
                opts.time_axis.assemble((sx, sy, sz), opts.time_val),
                opts.max_iter,
                bailout_sq,
            );
            (threshold - et as f64) as f32
        })
        .collect()
}

/// Runs Naive Surface Nets over the whole field in one call (no chunking —
/// the field comfortably fits in memory at the resolutions this is meant
/// for) and remaps the resulting grid-space positions into the same
/// spatial world coordinates the scalar field was sampled in.
fn extract_mesh(sdf: &[f32], opts: &VoxelStlOpts) -> (Vec<[f32; 3]>, Vec<u32>) {
    let res = opts.res;
    let shape = RuntimeShape::<u32, 3>::new([res, res, res]);
    let mut buffer = SurfaceNetsBuffer::default();
    surface_nets(sdf, &shape, [0, 0, 0], [res - 1, res - 1, res - 1], &mut buffer);

    let scale = (2.0 * opts.domain_extent / (opts.res as f64 - 1.0).max(1.0)) as f32;
    let offset = -opts.domain_extent as f32;
    let mut positions = buffer.positions;
    positions.par_iter_mut().for_each(|p| {
        p[0] = offset + p[0] * scale;
        p[1] = offset + p[1] * scale;
        p[2] = offset + p[2] * scale;
    });
    (positions, buffer.indices)
}

/// CSR adjacency (undirected, edges duplicated per shared triangle rather
/// than deduped — a heavily-shared edge just gets proportionally more
/// weight in the average, which is a fine, standard simplification for
/// plain umbrella smoothing).
fn build_adjacency(indices: &[u32], vertex_count: usize) -> (Vec<u32>, Vec<u32>) {
    let mut degree = vec![0u32; vertex_count];
    let mut edges: Vec<(u32, u32)> = Vec::with_capacity(indices.len() * 2);
    for tri in indices.chunks_exact(3) {
        for &(x, y) in &[(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
            edges.push((x, y));
            edges.push((y, x));
        }
    }
    for &(x, _) in &edges {
        degree[x as usize] += 1;
    }
    let mut offsets = vec![0u32; vertex_count + 1];
    for i in 0..vertex_count {
        offsets[i + 1] = offsets[i] + degree[i];
    }
    let mut neighbors = vec![0u32; edges.len()];
    let mut cursor = offsets.clone();
    for &(x, y) in &edges {
        neighbors[cursor[x as usize] as usize] = y;
        cursor[x as usize] += 1;
    }
    (offsets, neighbors)
}

/// A few passes of umbrella (Laplacian) smoothing: each vertex moves partway
/// toward the average of its neighbors. Connectivity (`indices`) is
/// unchanged — only vertex positions move — so this is safe to run before
/// or after decimation; here it runs before, on the full-resolution mesh,
/// so decimation afterward is simplifying an already-smooth surface rather
/// than fighting voxel-grid stairstepping.
fn laplacian_smooth(positions: &mut [[f32; 3]], indices: &[u32], iterations: u32, alpha: f32) {
    if positions.is_empty() || iterations == 0 {
        return;
    }
    let (offsets, neighbors) = build_adjacency(indices, positions.len());
    for _ in 0..iterations {
        let old = positions.to_vec();
        positions.par_iter_mut().enumerate().for_each(|(i, p)| {
            let start = offsets[i] as usize;
            let end = offsets[i + 1] as usize;
            if end > start {
                let mut sum = [0f32; 3];
                for &nb in &neighbors[start..end] {
                    let np = old[nb as usize];
                    sum[0] += np[0];
                    sum[1] += np[1];
                    sum[2] += np[2];
                }
                let cnt = (end - start) as f32;
                p[0] += (sum[0] / cnt - p[0]) * alpha;
                p[1] += (sum[1] / cnt - p[1]) * alpha;
                p[2] += (sum[2] / cnt - p[2]) * alpha;
            }
        });
    }
}

/// Quadric-error-metric edge collapse (pure-Rust port of meshoptimizer) down
/// toward `target_tris` triangles. Best-effort: may stop earlier if
/// `target_error` (relative to mesh extents) would otherwise be exceeded.
/// References the ORIGINAL vertex buffer — nothing here needs a compact
/// vertex list, since STL is unindexed triangle soup anyway.
fn decimate(positions: &[[f32; 3]], indices: &[u32], target_tris: usize, target_error: f32) -> Vec<u32> {
    let target_index_count = (target_tris * 3).min(indices.len());
    let mut destination = vec![0u32; indices.len()];
    let new_len = meshopt_rs::simplify::simplify(
        &mut destination,
        indices,
        positions,
        target_index_count,
        target_error,
    );
    destination.truncate(new_len);
    destination
}

fn mesh_to_stl_triangles(positions: &[[f32; 3]], indices: &[u32]) -> Vec<stl_io::Triangle> {
    indices
        .par_chunks_exact(3)
        .map(|tri| {
            let v0 = positions[tri[0] as usize];
            let v1 = positions[tri[1] as usize];
            let v2 = positions[tri[2] as usize];
            let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
            let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
            let n = [
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            ];
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            let normal = if len > 1e-12 {
                [n[0] / len, n[1] / len, n[2] / len]
            } else {
                [0.0, 0.0, 1.0]
            };
            stl_io::Triangle {
                normal: stl_io::Normal::new(normal),
                vertices: [
                    stl_io::Vertex::new(v0),
                    stl_io::Vertex::new(v1),
                    stl_io::Vertex::new(v2),
                ],
            }
        })
        .collect()
}

/// Runs the full pipeline and writes `out_path`. Returns timing/size
/// diagnostics — this is a heavy, exploratory operation (dense 3D field,
/// potentially tens of millions of raw triangles before decimation), so the
/// caller printing these numbers matters for judging whether the settings
/// used were reasonable.
pub fn build_voxel_stl(opts: &VoxelStlOpts, out_path: &Path) -> io::Result<VoxelStlReport> {
    let mut report = VoxelStlReport::default();

    let t0 = Instant::now();
    let sdf = build_scalar_field(opts);
    report.field_secs = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    let (mut positions, mut indices) = extract_mesh(&sdf, opts);
    drop(sdf);
    report.mesh_secs = t1.elapsed().as_secs_f64();
    report.raw_vertices = positions.len();
    report.raw_triangles = indices.len() / 3;

    let t2 = Instant::now();
    laplacian_smooth(&mut positions, &indices, opts.smooth_iters, opts.smooth_alpha);
    report.smooth_secs = t2.elapsed().as_secs_f64();

    let t3 = Instant::now();
    if report.raw_triangles > opts.target_tris {
        indices = decimate(&positions, &indices, opts.target_tris, opts.target_error);
    }
    report.decimate_secs = t3.elapsed().as_secs_f64();
    report.final_triangles = indices.len() / 3;

    let t4 = Instant::now();
    let triangles = mesh_to_stl_triangles(&positions, &indices);
    if let Some(dir) = out_path.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir)?;
        }
    }
    let mut file = io::BufWriter::new(std::fs::File::create(out_path)?);
    stl_io::write_stl(&mut file, triangles.iter())?;
    report.write_secs = t4.elapsed().as_secs_f64();

    Ok(report)
}
