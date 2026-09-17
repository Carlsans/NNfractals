//! A minimal, dependency-free 5x7 bitmap font for burning debug text
//! directly onto rendered RGB24 frames — e.g. `explorer quat-mandelbrot
//! --overlay-coords`, which needs the exact quaternion coordinates visible
//! IN the video rather than in a separate file that has to be kept in sync
//! with it by frame number. Covers exactly the characters that kind of
//! label needs (digits, a handful of uppercase letters, `.`, `-`, `=`,
//! space) — nothing more, since pulling in a real font-rendering crate for
//! a debug overlay would be a lot of weight for very little.

const GLYPH_W: usize = 5;
const GLYPH_H: usize = 7;

/// Row-major, MSB-first (bit 4 = leftmost column) 5-bit rows, top to bottom.
/// Unrecognized characters (including space) render as blank — a safe,
/// silent fallback for a debug aid, not a hard error.
fn glyph(c: char) -> [u8; GLYPH_H] {
    match c {
        '0' => [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
        '1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        '2' => [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111],
        '3' => [0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110],
        '4' => [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        '5' => [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
        '6' => [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
        '7' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
        '8' => [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        '9' => [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100],
        'A' => [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'B' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
        'C' => [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110],
        'F' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
        'R' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
        'T' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        'S' => [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
        'I' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b11111],
        'X' => [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001],
        '.' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100],
        '-' => [0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000],
        '=' => [0b00000, 0b00000, 0b11111, 0b00000, 0b11111, 0b00000, 0b00000],
        _ => [0; GLYPH_H],
    }
}

fn fill_rect(buf: &mut [u8], width: usize, height: usize, x0: usize, y0: usize, w: usize, h: usize, color: [u8; 3]) {
    for y in y0..(y0 + h).min(height) {
        for x in x0..(x0 + w).min(width) {
            let idx = (y * width + x) * 3;
            if idx + 2 < buf.len() {
                buf[idx] = color[0];
                buf[idx + 1] = color[1];
                buf[idx + 2] = color[2];
            }
        }
    }
}

/// Draws `text` into an RGB24 `buf` (row-major, 3 bytes/pixel, `width *
/// height * 3` long) at top-left `(x0,y0)`, `scale`x pixel size per glyph
/// cell, in `color`, with an optional solid `bg` rectangle behind the whole
/// line for legibility over arbitrary fractal colors underneath. Silently
/// clips at the buffer edges — never panics regardless of position/length.
pub fn draw_text(
    buf: &mut [u8],
    width: usize,
    height: usize,
    x0: usize,
    y0: usize,
    text: &str,
    scale: usize,
    color: [u8; 3],
    bg: Option<[u8; 3]>,
) {
    let scale = scale.max(1);
    let cell_w = (GLYPH_W + 1) * scale;
    let cell_h = GLYPH_H * scale;
    if let Some(bg) = bg {
        let text_w = text.chars().count() * cell_w;
        fill_rect(buf, width, height, x0, y0, text_w, cell_h + scale, bg);
    }
    for (i, c) in text.chars().enumerate() {
        let g = glyph(c);
        let gx0 = x0 + i * cell_w;
        for (row, bits) in g.iter().enumerate() {
            for col in 0..GLYPH_W {
                if (bits >> (GLYPH_W - 1 - col)) & 1 == 1 {
                    let px0 = gx0 + col * scale;
                    let py0 = y0 + row * scale;
                    fill_rect(buf, width, height, px0, py0, scale, scale, color);
                }
            }
        }
    }
}

/// Height in pixels of one line of text at the given scale, including the
/// small gap `draw_text`'s background box leaves below the glyphs — useful
/// for stacking multiple lines with `draw_text` calls at increasing `y0`.
pub fn line_height(scale: usize) -> usize {
    GLYPH_H * scale.max(1) + scale.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draw_text_does_not_panic_near_or_past_buffer_edges() {
        let (w, h) = (20usize, 20usize);
        let mut buf = vec![0u8; w * h * 3];
        // Well within bounds.
        draw_text(&mut buf, w, h, 0, 0, "F0042", 1, [255, 255, 0], Some([0, 0, 0]));
        // Starting past the edge, and a long string that would overflow —
        // both must clip silently, not panic.
        draw_text(&mut buf, w, h, 18, 18, "R=-1.2346", 2, [255, 0, 0], None);
        draw_text(&mut buf, w, h, 1000, 1000, "OFFSCREEN", 3, [0, 255, 0], Some([1, 1, 1]));
    }

    #[test]
    fn drawn_glyph_pixels_match_the_requested_color() {
        let (w, h) = (40usize, 20usize);
        let mut buf = vec![9u8; w * h * 3]; // distinct sentinel, not black/white
        draw_text(&mut buf, w, h, 0, 0, "1", 2, [200, 100, 50], None);
        // '1' has a lit pixel at row 0, col 2 (the top of the vertical
        // stroke: 0b00100) — at scale 2 that's pixel (4,0).
        let idx = (0 * w + 4) * 3;
        assert_eq!(&buf[idx..idx + 3], &[200, 100, 50]);
    }

    #[test]
    fn unrecognized_characters_render_blank_not_panic() {
        let (w, h) = (30usize, 10usize);
        let mut buf = vec![7u8; w * h * 3];
        draw_text(&mut buf, w, h, 0, 0, "xyz!@#", 1, [255, 255, 255], None);
        // Nothing drawn (all glyphs blank) => buffer unchanged.
        assert!(buf.iter().all(|&v| v == 7));
    }

    #[test]
    fn background_rect_covers_the_full_line_width() {
        let (w, h) = (40usize, 20usize);
        let mut buf = vec![0u8; w * h * 3];
        draw_text(&mut buf, w, h, 0, 0, "AB", 2, [255, 255, 255], Some([9, 9, 9]));
        // Top-left corner pixel should be the background color even where
        // no glyph pixel is lit.
        assert_eq!(&buf[0..3], &[9, 9, 9]);
    }
}
