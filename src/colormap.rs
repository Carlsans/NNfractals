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

/// Dimmest an escaped pixel may render. Rank alone would put the darkest
/// escaped pixel at value 0 — indistinguishable from the interior, and it
/// throws away the hue that is the whole point of this mode.
const ANGLE_VALUE_FLOOR: f64 = 0.35;
/// The set interior: dark, unsaturated, and deliberately BELOW the floor so
/// the body reads as a silhouette against the escaped filaments.
const ANGLE_INTERIOR_VALUE: f64 = 0.10;
const ANGLE_SATURATION: f64 = 0.75;

/// Cosmetic alternative to `apply_colormap`: hue driven by the bailout
/// exit-angle (arg z at escape — see fractal::dag_escape_pixel), value by
/// the escape time, fixed high saturation. BOTH channels are
/// histogram-equalized against this render's own distribution — see the
/// notes on hue below, and on `escape_equalize` for why brightness needs
/// exactly the same treatment. Never called from the GA's save
/// path (try_save/force_save always use apply_colormap/turbo) — this is
/// purely for the viewer's interactive "∠" toggle, and for video export /
/// video-zoom search when angle colouring is requested. `angles` must be
/// the same length as `escape_times`. Interior/non-finite-escape pixels
/// carry angle 0.0, which is not a real exit direction, so they are
/// special-cased to a dark unsaturated grey rather than being given a hue
/// that would read as meaningful.
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
    let equalize = angle_equalize(escape_times, angles, max_iter);
    // Brightness gets the SAME treatment as hue, for the same reason. The
    // raw `escape_time / max_iter` this used to use is not a brightness
    // scale, it's a ratio against a cap chosen for CORRECTNESS: `max_iter`
    // has to exceed what the deepest pixel needs, and at depth
    // `effective_max_iter` scales it with zoom, so typical escape times sit
    // far below it. Measured on a real view (7cd46280, zoom 38): mean value
    // 0.028 and EVERY pixel under 10% luma — a black screen with hue
    // information in it that no display can show. Rank-in-distribution has
    // no such dependence on the cap: whatever the escape times actually
    // are, they spread across the full output range.
    let brighten = escape_equalize(escape_times, max_iter);
    let mut pixels = Vec::with_capacity(escape_times.len() * 3);
    for (i, &t) in escape_times.iter().enumerate() {
        let angle = angles.get(i).copied().unwrap_or(0.0);
        let (sat, val) = if (t as u32) >= max_iter {
            // Interior: no meaningful exit angle, so hue would be noise.
            (0.0, ANGLE_INTERIOR_VALUE)
        } else {
            (ANGLE_SATURATION, ANGLE_VALUE_FLOOR + (1.0 - ANGLE_VALUE_FLOOR) * brighten(t) as f64)
        };
        let hue = equalize(angle) as f64 * 360.0;
        let (r, g, b) = hsv_to_rgb(hue, sat, val);
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
    let samples: Vec<f32> = escape_times.iter().zip(angles.iter())
        .filter(|&(&t, _)| (t as u32) < max_iter)
        .map(|(_, &a)| a.rem_euclid(TAU))
        .collect();
    let Some(cdf) = empirical_cdf(samples) else {
        return Box::new(|a: f32| (a / TAU).rem_euclid(1.0)) as Box<dyn Fn(f32) -> f32>;
    };
    Box::new(move |a: f32| cdf(a.rem_euclid(TAU)))
}

/// The escape-time counterpart of `angle_equalize`, driving VALUE rather
/// than hue. Built from escaped pixels only: the interior is a single
/// spike at `max_iter` that would otherwise dominate the distribution and
/// squash every real gradient into the remaining range.
fn escape_equalize(escape_times: &[f32], max_iter: u32) -> impl Fn(f32) -> f32 {
    let samples: Vec<f32> = escape_times.iter().copied()
        .filter(|&t| (t as u32) < max_iter)
        .collect();
    let max = max_iter.max(1) as f32;
    let Some(cdf) = empirical_cdf(samples) else {
        // Nothing escaped (or almost nothing): no distribution to equalize
        // against, so fall back to the plain normalized ratio.
        return Box::new(move |t: f32| (t / max).clamp(0.0, 1.0)) as Box<dyn Fn(f32) -> f32>;
    };
    Box::new(cdf)
}

/// Exact empirical CDF over `samples`: sort once, then rank each query by
/// binary search. `None` when there is nothing to equalize against (fewer
/// than 2 samples), leaving the fallback to the caller.
///
/// Exact rank, not a binned histogram — a binned version (720 buckets) was
/// tried first and checked against a real render: one genome had 67.5% of
/// escaped pixels inside a SINGLE 0.5°-wide bin, far tighter than the bin
/// resolution, so interpolation within that bin had nothing to work with
/// and collapsed back to one hue. Rank has no resolution floor: however
/// tightly clustered the real distribution is, sorted order still separates
/// every distinct value. Cost is one O(n log n) sort per render plus an
/// O(log n) query per pixel.
fn empirical_cdf(mut samples: Vec<f32>) -> Option<impl Fn(f32) -> f32> {
    if samples.len() < 2 { return None; }
    samples.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n_f = samples.len() as f32;
    Some(move |v: f32| {
        // Rank = count of samples strictly below `v`, i.e. where it would
        // insert to keep the slice sorted — exactly the empirical CDF.
        (samples.partition_point(|&x| x < v) as f32 / n_f).clamp(0.0, 1.0)
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

#[cfg(test)]
mod angle_colormap_tests {
    use super::*;

    /// Escape times spread over a small range, with a `max_iter` far above
    /// them — the ordinary situation, since `max_iter` must exceed what the
    /// DEEPEST pixel needs and `effective_max_iter` raises it further with
    /// zoom. Angles are varied so hue is not the thing under test.
    fn sample_field(n: usize, max_iter: u32) -> (Vec<f32>, Vec<f32>) {
        let ets = (0..n).map(|i| 20.0 + (i % 40) as f32).collect();
        let angs = (0..n).map(|i| (i as f32 * 0.017).rem_euclid(std::f32::consts::TAU)).collect();
        (ets, angs)
    }

    fn mean_value(px: &[u8]) -> f64 {
        // HSV value is the max channel.
        let n = px.len() / 3;
        (0..n).map(|i| *px[i * 3..i * 3 + 3].iter().max().unwrap() as f64 / 255.0).sum::<f64>()
            / n as f64
    }

    #[test]
    fn brightness_does_not_collapse_when_max_iter_dwarfs_escape_times() {
        // The real bug: value was `escape_time / max_iter`, so raising the
        // iteration cap (which deep zoom does automatically) faded the whole
        // image to black. Measured on a real view: mean value 0.028, every
        // pixel under 10% luma.
        let (ets, angs) = sample_field(4096, 8192);
        let px = apply_angle_colormap(&ets, &angs, 8192);
        assert!(mean_value(&px) > 0.5, "mean value {} — too dark", mean_value(&px));
    }

    #[test]
    fn brightness_is_independent_of_the_iteration_cap() {
        // Rank-in-distribution has no dependence on the cap, so the SAME
        // escape times must render at the same brightness whether the cap is
        // 4x or 100x above them. This is the property that actually fixes it.
        let (ets, angs) = sample_field(4096, 100_000);
        let low = mean_value(&apply_angle_colormap(&ets, &angs, 256));
        let high = mean_value(&apply_angle_colormap(&ets, &angs, 100_000));
        assert!((low - high).abs() < 0.02, "cap changed brightness: {low} vs {high}");
    }

    #[test]
    fn interior_pixels_render_dark_and_unsaturated() {
        // t >= max_iter is the set body: no meaningful exit angle, so hue
        // would be noise. It must read as a silhouette, below the floor that
        // escaped pixels get.
        let mut ets: Vec<f32> = (0..64).map(|i| 10.0 + i as f32).collect();
        let angs = vec![0.3f32; 65];
        ets.push(256.0); // interior
        let px = apply_angle_colormap(&ets, &angs, 256);
        let last = &px[px.len() - 3..];
        assert_eq!(last[0], last[1], "interior should be unsaturated (grey)");
        assert_eq!(last[1], last[2], "interior should be unsaturated (grey)");
        assert!((last[0] as f64 / 255.0) < ANGLE_VALUE_FLOOR,
                "interior {} should be darker than any escaped pixel", last[0]);
    }

    #[test]
    fn a_dense_cluster_is_spread_across_the_output_range() {
        // The whole point of equalization over a fixed mapping: 90% of the
        // samples sit in a 0.001-wide cluster, and must still end up spread
        // over ~90% of the output range, not collapsed onto one value.
        let mut vals: Vec<f32> = (0..900).map(|i| 1.0 + i as f32 * 1e-6).collect();
        vals.extend((0..100).map(|i| 2.0 + i as f32 * 0.01));
        let cdf = empirical_cdf(vals.clone()).expect("enough samples");
        let lo = cdf(1.0);
        let hi = cdf(1.0 + 899.0 * 1e-6);
        assert!(hi - lo > 0.85, "cluster spread only {} of the range", hi - lo);
    }

    #[test]
    fn empirical_cdf_needs_something_to_equalize_against() {
        assert!(empirical_cdf(vec![]).is_none());
        assert!(empirical_cdf(vec![1.0]).is_none());
        assert!(empirical_cdf(vec![1.0, 2.0]).is_some());
    }

    #[test]
    fn an_all_interior_field_still_renders() {
        // Nothing escaped: there is no distribution to equalize against, and
        // the fallback must not panic or divide by zero.
        let ets = vec![256.0f32; 16];
        let angs = vec![0.0f32; 16];
        let px = apply_angle_colormap(&ets, &angs, 256);
        assert_eq!(px.len(), 48);
    }
}
