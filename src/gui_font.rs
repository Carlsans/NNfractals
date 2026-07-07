//! Shared egui font setup for all NNFractals GUIs.
//!
//! egui's default proportional/monospace fonts don't cover the arrow and star
//! glyphs (← ▶ ■ ↻ ★ ▲ ▼ …) we use in toolbars, so on some systems they render
//! as tofu boxes. We bundle DejaVu Sans and append it as a *fallback* on both
//! font families: default text is unchanged, but any glyph the built-in fonts
//! lack now resolves to DejaVu.

use eframe::egui;

/// Bundled DejaVu Sans (broad Unicode coverage: arrows, geometric shapes, stars).
const DEJAVU_SANS: &[u8] = include_bytes!("../assets/DejaVuSans.ttf");

/// Install the bundled font as a fallback on the given egui context.
/// Call once from each binary's `eframe::run_native` creation closure.
pub fn install(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        "dejavu_sans".to_owned(),
        std::sync::Arc::new(egui::FontData::from_static(DEJAVU_SANS)),
    );
    // Append (not prepend) so the built-in fonts stay primary and DejaVu only
    // fills in glyphs they don't have.
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        fonts
            .families
            .entry(family)
            .or_default()
            .push("dejavu_sans".to_owned());
    }
    ctx.set_fonts(fonts);
}
