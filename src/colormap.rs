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

/// Cosmetic alternative to `apply_colormap`: hue driven by the bailout
/// exit-angle (arg z at escape — see fractal::dag_escape_pixel), value/
/// lightness driven by the same normalized escape fraction the standard
/// palettes use, fixed high saturation. Never called from the GA's save
/// path (try_save/force_save always use apply_colormap/turbo) — this is
/// purely for the viewer's interactive "∠" toggle. `angles` must be the
/// same length as `escape_times` (interior/non-finite-escape pixels have
/// angle 0.0, which just renders at whatever hue that maps to below — no
/// special-casing needed since those pixels are already at the
/// value-gradient's dark/light extreme, so they stay visually distinct
/// regardless of exact hue).
///
/// Hue is NOT a fixed `angle -> hue` formula — it's HISTOGRAM-EQUALIZED
/// against whatever distribution of angles this specific render actually
/// produced, via `angle_equalize`. Confirmed on a real genome that a fixed
/// mapping looks "monotonic" (Carl's word) far more often than not: many
/// fractals' escaping orbits exit overwhelmingly in one preferred
/// direction (one real case measured 71% of escaped pixels sharing a
/// single 30° bucket out of 12), so a fixed formula spends almost the
/// whole hue wheel on directions nothing ever escapes toward. Two things
/// that look plausible but were checked against the real image and don't
/// work: (1) just ROTATING the fixed mapping to start at a gap — moves a
/// narrow cluster to a different slice of the wheel, doesn't widen it;
/// (2) a MIN-MAX stretch of the observed range — only looks at the two
/// extremes, so if there's a sparse tail on either side (there usually
/// is), the "range" is already ~the whole circle and stretching changes
/// nothing, even though 70% of pixels are still packed into one dense
/// cluster within that range. Equalization (rank in the sorted
/// distribution, not raw value) is what actually fixes this: the 70%
/// cluster occupies 70% of pixels, so it gets stretched across 70% of the
/// OUTPUT hue range regardless of how narrow its RAW angular width was —
/// the fine variation within what used to look like a single color
/// becomes visible. No cut-point/wraparound handling needed: hue is
/// circular (0°≈360°), so rank is computed directly against the natural
/// `atan2` range with no special-casing for where a cluster happens to
/// sit relative to ±π.
pub fn apply_angle_colormap(escape_times: &[f32], angles: &[f32], max_iter: u32) -> Vec<u8> {
    let max = max_iter as f64;
    let equalize = angle_equalize(escape_times, angles, max_iter);
    let mut pixels = Vec::with_capacity(escape_times.len() * 3);
    for (i, &t) in escape_times.iter().enumerate() {
        let norm = (t as f64 / max).clamp(0.0, 1.0);
        let angle = angles.get(i).copied().unwrap_or(0.0);
        let hue = equalize(angle) as f64 * 360.0;
        let (r, g, b) = hsv_to_rgb(hue, 0.75, norm);
        pixels.push(r);
        pixels.push(g);
        pixels.push(b);
    }
    pixels
}

/// Builds the `angle (radians) -> [0,1)` mapping `apply_angle_colormap`
/// scales to hue: the EXACT empirical CDF (sorted escaped-pixel angles +
/// binary search per query), i.e. true histogram equalization, not a
/// binned approximation. A binned version (720 buckets, 0.5° each) was
/// tried first and checked against a real render before trusting it —
/// wrong: one real genome had 67.5% of ALL escaped pixels inside a SINGLE
/// 0.5°-wide bin, far tighter than the bin resolution could resolve, so
/// linear interpolation within that one bin had nothing to work with and
/// collapsed right back to one hue. Exact rank has no resolution floor —
/// however tightly clustered the real distribution is, sorted-order rank
/// still differentiates every distinct value, so a dominant cluster
/// spreads across its correct share of the output range and reveals
/// whatever real fine structure it contains. Cost is one O(n log n) sort
/// per render (paid once, not per pixel) plus an O(log n) binary search
/// per query — this is a cosmetic, opt-in toggle, not the default render
/// path, so this is an acceptable trade for actually being correct.
/// Degenerate input (fewer than 2 escaped pixels) falls back to a fixed
/// linear mapping; there's no real distribution to equalize against.
fn angle_equalize(escape_times: &[f32], angles: &[f32], max_iter: u32) -> impl Fn(f32) -> f32 {
    const TAU: f32 = std::f32::consts::TAU;
    let mut sorted: Vec<f32> = escape_times.iter().zip(angles.iter())
        .filter(|&(&t, _)| (t as u32) < max_iter)
        .map(|(_, &a)| a.rem_euclid(TAU))
        .collect();
    if sorted.len() < 2 {
        return Box::new(|a: f32| (a / TAU).rem_euclid(1.0)) as Box<dyn Fn(f32) -> f32>;
    }
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n_f = sorted.len() as f32;
    Box::new(move |a: f32| {
        let wrapped = a.rem_euclid(TAU);
        // Rank = count of reference angles < this one, i.e. where it would
        // insert to keep `sorted` sorted — exactly the empirical CDF.
        let rank = sorted.partition_point(|&x| x < wrapped);
        (rank as f32 / n_f).clamp(0.0, 1.0)
    })
}

fn hsv_to_rgb(h: f64, s: f64, v: f64) -> (u8, u8, u8) {
    let c = v * s;
    let hp = h / 60.0;
    let x = c * (1.0 - (hp.rem_euclid(2.0) - 1.0).abs());
    let (r1, g1, b1) = match hp as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    let to_u8 = |v: f64| ((v + m).clamp(0.0, 1.0) * 255.0).round() as u8;
    (to_u8(r1), to_u8(g1), to_u8(b1))
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
