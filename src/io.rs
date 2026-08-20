use std::path::Path;
use anyhow::Result;
use crate::genome::Genome;

pub fn save_genome(genome: &Genome, path: &Path) -> Result<()> {
    let json = serde_json::to_string_pretty(genome)?;
    std::fs::write(path, json)?;
    Ok(())
}

pub fn load_genome(path: &Path) -> Result<Genome> {
    let json = std::fs::read_to_string(path)?;
    Ok(serde_json::from_str(&json)?)
}

pub fn save_png(pixels: &[u8], width: u32, height: u32, path: &Path) -> Result<()> {
    image::save_buffer(path, pixels, width, height, image::ColorType::Rgb8)?;
    Ok(())
}

/// Saves a raw escape-time field (as returned by `render_escape_times`,
/// BEFORE `colormap::apply_colormap`) as an 8-bit grayscale PNG — the VAE
/// training input for `vae_explore`. Same `t / max_iter` normalization
/// idiom `apply_colormap` already uses, just stopping before the colormap
/// LUT step: this is the pure escape-time structure, not a human-facing
/// visualization of it.
pub fn save_raw_field(field: &[f32], width: u32, height: u32, max_iter: u32, path: &Path) -> Result<()> {
    let max = max_iter as f64;
    let bytes: Vec<u8> = field.iter()
        .map(|&t| ((t as f64 / max).clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect();
    image::save_buffer(path, &bytes, width, height, image::ColorType::L8)?;
    Ok(())
}

/// Saves the real part, imaginary part, and magnitude of the bailout z
/// value (as returned by `video_export::render_complex_field`) as three
/// separate 8-bit grayscale PNGs — exploratory data for a possible
/// complex-valued autoencoder (Carl's request, 2026-08-07; see
/// `dag_escape_pixel_z` for why this needs its own render pass rather
/// than reusing the escape-time field).
///
/// Normalization: re/zy at bailout can be negative and can overshoot
/// `bailout_radius` by an arbitrary amount depending on how fast a given
/// formula's dynamics diverge — there's no fixed max the way `t/max_iter`
/// has for escape time. `4 * bailout_radius` is a generous, formula-
/// agnostic clamp (covers the overshoot observed in this project's DAG
/// formulas in practice); values beyond it saturate rather than wrapping,
/// same tradeoff `save_raw_field`'s own `.clamp(0.0, 1.0)` makes. Re/im
/// map their symmetric `[-clamp, clamp]` range to `[0, 255]`; magnitude
/// (always ≥ `bailout_radius` at a real bailout) maps `[0, clamp]` to
/// `[0, 255]` directly. Interior/non-finite pixels (`zx=zy=0.0` from
/// `dag_escape_pixel_z`) land at the middle of the re/im range and at 0
/// for magnitude — visually distinct from a genuine near-zero bailout,
/// which is what matters for a first feasibility look, not a claim that
/// 0.0 is otherwise a meaningful value here.
pub fn save_complex_channels(
    field: &[(f32, f32)], bailout_radius: f32, width: u32, height: u32,
    re_path: &Path, im_path: &Path, mag_path: &Path,
) -> Result<()> {
    let clamp = 4.0 * bailout_radius;
    let norm_signed = |v: f32| (((v / clamp).clamp(-1.0, 1.0) + 1.0) * 0.5 * 255.0).round() as u8;
    let norm_mag = |v: f32| ((v / clamp).clamp(0.0, 1.0) * 255.0).round() as u8;

    let re_bytes: Vec<u8> = field.iter().map(|&(zx, _)| norm_signed(zx)).collect();
    let im_bytes: Vec<u8> = field.iter().map(|&(_, zy)| norm_signed(zy)).collect();
    let mag_bytes: Vec<u8> = field.iter().map(|&(zx, zy)| norm_mag((zx * zx + zy * zy).sqrt())).collect();

    image::save_buffer(re_path, &re_bytes, width, height, image::ColorType::L8)?;
    image::save_buffer(im_path, &im_bytes, width, height, image::ColorType::L8)?;
    image::save_buffer(mag_path, &mag_bytes, width, height, image::ColorType::L8)?;
    Ok(())
}
