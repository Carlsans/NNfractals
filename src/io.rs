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
