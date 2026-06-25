use colorous::Gradient;

pub fn apply_colormap(escape_times: &[f32], max_iter: u32, colormap_name: &str) -> Vec<u8> {
    let max = max_iter as f64;
    let mut pixels = Vec::with_capacity(escape_times.len() * 3);
    for &t in escape_times {
        let norm = (t as f64 / max).clamp(0.0, 1.0);
        let (r, g, b) = color_at(colormap_name, norm);
        pixels.push(r);
        pixels.push(g);
        pixels.push(b);
    }
    pixels
}

fn color_at(name: &str, t: f64) -> (u8, u8, u8) {
    match name {
        "earth"     => earth_color(t),
        "bone"      => bone_color(t),
        "neon"      => neon_color(t),
        _ => { let c = pick_gradient(name).eval_continuous(t); (c.r, c.g, c.b) }
    }
}

fn pick_gradient(name: &str) -> Gradient {
    match name {
        "viridis"   => colorous::VIRIDIS,
        "inferno"   => colorous::INFERNO,
        "plasma"    => colorous::PLASMA,
        "magma"     => colorous::MAGMA,
        "turbo"     => colorous::TURBO,
        "cool"      => colorous::COOL,
        "warm"      => colorous::WARM,
        "cubehelix" => colorous::CUBEHELIX,
        _           => colorous::VIRIDIS,
    }
}

// gist_earth approximation: deep ocean → seafloor → lowlands → savanna → highlands → glacier → snow
fn earth_color(t: f64) -> (u8, u8, u8) {
    const STOPS: &[(f64, [u8; 3])] = &[
        (0.000, [0,   0,   60]),
        (0.120, [0,   25, 115]),
        (0.200, [15,  70,  55]),
        (0.330, [75, 115,  45]),
        (0.470, [155,140,  65]),
        (0.600, [125, 85,  42]),
        (0.720, [95,  65,  40]),
        (0.840, [140,140, 155]),
        (1.000, [248,248, 252]),
    ];
    lerp_stops(STOPS, t)
}

// Blue-grey bone: dark → warm grey → off-white
fn bone_color(t: f64) -> (u8, u8, u8) {
    const STOPS: &[(f64, [u8; 3])] = &[
        (0.000, [0,   0,   0]),
        (0.375, [54,  54,  75]),
        (0.750, [140, 140, 160]),
        (1.000, [240, 240, 248]),
    ];
    lerp_stops(STOPS, t)
}

// Vivid neon: black → deep violet → electric blue → cyan → hot green → yellow → white
fn neon_color(t: f64) -> (u8, u8, u8) {
    const STOPS: &[(f64, [u8; 3])] = &[
        (0.000, [0,   0,   0]),
        (0.200, [80,  0, 180]),
        (0.400, [0,   80, 255]),
        (0.600, [0,  220, 220]),
        (0.800, [80, 255,  50]),
        (0.900, [255,220,   0]),
        (1.000, [255,255, 255]),
    ];
    lerp_stops(STOPS, t)
}

fn lerp_stops(stops: &[(f64, [u8; 3])], t: f64) -> (u8, u8, u8) {
    let t = t.clamp(0.0, 1.0);
    for i in 1..stops.len() {
        if t <= stops[i].0 {
            let range = stops[i].0 - stops[i - 1].0;
            let u = if range > 1e-10 { (t - stops[i - 1].0) / range } else { 0.0 };
            let a = stops[i - 1].1;
            let b = stops[i].1;
            return (lerp_u8(a[0], b[0], u), lerp_u8(a[1], b[1], u), lerp_u8(a[2], b[2], u));
        }
    }
    let last = stops.last().unwrap().1;
    (last[0], last[1], last[2])
}

fn lerp_u8(a: u8, b: u8, t: f64) -> u8 {
    (a as f64 * (1.0 - t) + b as f64 * t).round() as u8
}
