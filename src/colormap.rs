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
        "earth"   => earth_color(t),
        "bone"    => bone_color(t),
        "neon"    => neon_color(t),
        "lava"    => lava_color(t),
        "aurora"  => aurora_color(t),
        "galaxy"  => galaxy_color(t),
        "sunset"  => sunset_color(t),
        "arctic"  => arctic_color(t),
        "ember"   => ember_color(t),
        "grayscale" => grayscale_color(t),
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

// Molten lava: black → deep crimson → fiery orange → bright yellow → white-hot
fn lava_color(t: f64) -> (u8, u8, u8) {
    const STOPS: &[(f64, [u8; 3])] = &[
        (0.000, [0,   0,   0]),
        (0.200, [100,  0,   0]),
        (0.420, [200,  30,  0]),
        (0.620, [240, 100,  0]),
        (0.800, [255, 200,  0]),
        (0.920, [255, 240, 120]),
        (1.000, [255, 255, 240]),
    ];
    lerp_stops(STOPS, t)
}

// Northern lights: black → deep teal → electric green → violet → white
fn aurora_color(t: f64) -> (u8, u8, u8) {
    const STOPS: &[(f64, [u8; 3])] = &[
        (0.000, [0,   0,   15]),
        (0.200, [0,   80,  80]),
        (0.380, [0,  200,  80]),
        (0.560, [20, 255, 120]),
        (0.720, [120, 60, 220]),
        (0.880, [200, 80, 255]),
        (1.000, [240, 240, 255]),
    ];
    lerp_stops(STOPS, t)
}

// Deep space: midnight navy → royal purple → rose → gold → cream
fn galaxy_color(t: f64) -> (u8, u8, u8) {
    const STOPS: &[(f64, [u8; 3])] = &[
        (0.000, [5,   5,  30]),
        (0.200, [30,  10,  90]),
        (0.380, [100, 20, 160]),
        (0.560, [200, 60, 140]),
        (0.720, [240, 140,  60]),
        (0.880, [250, 220, 120]),
        (1.000, [255, 250, 230]),
    ];
    lerp_stops(STOPS, t)
}

// Dusk: deep indigo → magenta → coral → saffron → pale gold
fn sunset_color(t: f64) -> (u8, u8, u8) {
    const STOPS: &[(f64, [u8; 3])] = &[
        (0.000, [10,   5,  50]),
        (0.220, [100,  10, 150]),
        (0.420, [220,  40, 120]),
        (0.600, [240, 100,  40]),
        (0.780, [250, 190,  30]),
        (1.000, [255, 245, 180]),
    ];
    lerp_stops(STOPS, t)
}

// Arctic ice: deep navy → polar blue → ice cyan → glacial white
fn arctic_color(t: f64) -> (u8, u8, u8) {
    const STOPS: &[(f64, [u8; 3])] = &[
        (0.000, [0,   10,  40]),
        (0.200, [0,   50, 130]),
        (0.420, [0,  140, 200]),
        (0.620, [60, 210, 235]),
        (0.800, [160, 235, 245]),
        (1.000, [240, 250, 255]),
    ];
    lerp_stops(STOPS, t)
}

// Glowing embers: charcoal → deep burgundy → brick red → burnt orange → amber
fn ember_color(t: f64) -> (u8, u8, u8) {
    const STOPS: &[(f64, [u8; 3])] = &[
        (0.000, [10,   5,   5]),
        (0.180, [60,   8,   8]),
        (0.360, [150,  20,  10]),
        (0.560, [210,  70,  10]),
        (0.750, [235, 150,  20]),
        (0.900, [245, 210,  80]),
        (1.000, [255, 245, 180]),
    ];
    lerp_stops(STOPS, t)
}

// Smooth grayscale gradient: black (t=0) → white (t=1).
fn grayscale_color(t: f64) -> (u8, u8, u8) {
    let v = (t.clamp(0.0, 1.0) * 255.0).round() as u8;
    (v, v, v)
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
