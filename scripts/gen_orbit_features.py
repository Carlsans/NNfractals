#!/usr/bin/env python3
"""Generates the `orbit_feature` module (slot layout, trap list) in src/quat_dag.rs: the
per-sample measurements behind `explorer quat-anatomy`'s layers. Edit the lists below and
rerun from the repo root; it rewrites the block between the two marker comments."""
import math, re
planes = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
pn = ["ra", "rb", "rc", "ab", "ac", "bc"]
cn = "rabc"
traps = [("origin", "Trap::Origin")]
for k in range(4): traps.append((f"plane_{cn[k]}", f"Trap::Plane({k})"))
traps.append(("cross", "Trap::Cross"))
radii = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0,
         1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5, 1.6, 1.75, 1.9, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 4.5, 5.0]
for R in radii:
    traps.append((f"sphere_{R:g}", f"Trap::Sphere({R:.2f})"))
for k in range(4): traps.append((f"axis_{cn[k]}", f"Trap::Axis({k})"))
for (i, j) in planes:
    for sgn, sn in [(1.0, "="), (-1.0, "=-")]: traps.append((f"diag_{cn[i]}{sn}{cn[j]}", f"Trap::Diag({i}, {j}, {sgn:.1f})"))
for p, (i, j) in enumerate(planes):
    for R in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]: traps.append((f"circle_{pn[p]}_{R:g}", f"Trap::Circle({i}, {j}, {R:.2f})"))
# Off-centre circles in the R-A plane: centres on a 0.5 grid, three radii.
for cx in [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]:
    for cy in [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]:
        if cx == 0 and cy == 0: continue
        for R in [0.25, 0.5, 1.0]:
            traps.append((f"ring_{cx:+g}{cy:+g}a_{R:g}", f"Trap::CircleAt(0, 1, {cx:.2f}, {cy:.2f}, {R:.2f})"))
# Lines in the R-A plane: 18 directions (10 degrees apart) at five offsets; three directions in the others.
for d10 in range(18):
    t = math.radians(10 * d10)
    for off in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        traps.append((f"line_ra_{10 * d10}deg_{off:+g}", f"Trap::Line(0, 1, {math.cos(t):.6f}, {math.sin(t):.6f}, {off:.2f})"))
for p, (i, j) in enumerate(planes[1:], start=1):
    for deg in [0, 30, 60, 90, 120, 150]:
        t = math.radians(deg)
        traps.append((f"line_{pn[p]}_{deg}deg", f"Trap::Line({i}, {j}, {math.cos(t):.6f}, {math.sin(t):.6f}, 0.00)"))
def ptname(q):
    s = [f"{'+' if v > 0 else '-'}{abs(v):g}{'' if k == 0 else cn[k]}" for k, v in enumerate(q) if v != 0]
    return "pt_" + "".join(s)
pts = []
g = [round(-2.5 + 0.25 * k, 2) for k in range(21)]  # R-A plane, 21 x 21 grid
for x in g:
    for y in g:
        if x or y: pts.append((x + 0.0, y + 0.0, 0.0, 0.0))
for p, (i, j) in enumerate(planes[1:], start=1):  # rings of radius 1 in the other planes
    for ang in range(8):
        t = ang * math.pi / 4
        q = [0.0] * 4; q[i] = round(math.cos(t), 2) + 0.0; q[j] = round(math.sin(t), 2) + 0.0
        pts.append(tuple(q))
for sg in range(16):  # the corners (+-0.5)^4
    pts.append(tuple(0.5 if (sg >> k) & 1 else -0.5 for k in range(4)))
seen = set()
for q in pts:
    if q in seen: continue
    seen.add(q)
    traps.append((ptname(q), "Trap::Point([{}])".format(", ".join(f"{v:.2f}" for v in q))))
NT = len(traps)
names = [t[0] for t in traps]
assert len(set(names)) == NT, [n for n in names if names.count(n) > 1]
dom = [names.index("origin"), names.index("cross")]
dom += [k for k, n in enumerate(names) if n.startswith(("sphere_", "axis_", "diag_", "circle_", "line_ra_"))]
dom += [k for k, n in enumerate(names) if n.startswith("ring_") and n.endswith("_0.5")]
dom += [k for k, n in enumerate(names) if n.startswith("pt_") and n.count("a") <= 1 and "b" not in n and "c" not in n
        and all(round(abs(float(v)) * 2) == abs(float(v)) * 2 for v in re.findall(r"[0-9.]+", n))]
# Angle at the closest approach (classic trap-angle colouring): (trap, plane i, j, centre).
tang = []
for k, (n, e) in enumerate(traps):
    if n == "origin" or n.startswith("sphere_") or n.startswith("circle_ra_"):
        tang.append((k, 0, 1, 0.0, 0.0))
    elif n.startswith("ring_"):
        cx, cy = [float(v) for v in re.findall(r"CircleAt\(0, 1, ([-0-9.]+), ([-0-9.]+)", e)[0]]
        tang.append((k, 0, 1, cx, cy))
    elif n.startswith("pt_"):
        q = [float(v) for v in re.findall(r"\[(.*)\]", e)[0].split(",")]
        nz = [c for c in range(4) if q[c] != 0.0]
        i, j = (0, 1) if all(c in (0, 1) for c in nz) else (nz[0], nz[1]) if len(nz) == 2 else (0, 1)
        tang.append((k, i, j, q[i], q[j]))
stripes = [(0, s, 0) for s in range(1, 49)] + [(0, s, 1) for s in range(1, 25)] + [(p, s, 0) for p in range(1, 6) for s in range(1, 13)]
avg_names = ["curvature", "triangle", "mean_log", "mean_abs", "log_var", "angle_spread", "near_unit", "mean_b", "mean_c", "var_r", "var_a", "var_b", "var_c"]
early_steps = list(range(1, 25))
early_comp = [(c, st) for st in range(1, 17) for c in (0, 1)]
early_other = [(p, st) for p in range(1, 6) for st in range(1, 13)]
passage = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 16.0]
slot, n = {}, 0
def take(name, k):
    global n
    slot[name] = n; n += k
for name, k in [("ESCAPED", 1), ("SMOOTH", 1), ("EXIT", 4), ("TRAPS", NT), ("TRAP_STEP", len(dom)), ("STRIPES", len(stripes)),
                ("AVERAGES", len(avg_names)), ("CENTROID", 2), ("WINDING", 6), ("EARLY_ANGLE", len(early_steps)), ("EARLY_LOG", len(early_steps)), ("EARLY_COMP", len(early_comp)), ("EARLY_OTHER", len(early_other)),
                ("EXTENT", 1), ("PASSAGE", len(passage)), ("EARLY_GROWTH", 2), ("LN_DZ", 1), ("LYAPUNOV", 1), ("DE", 1), ("PERIOD", 1), ("TRAP_ANGLE", len(tang))]:
    take(name, k)
NF = n
doc = {"ESCAPED": "1 if the orbit escaped, else 0.", "SMOOTH": "Smooth escape time, unclamped (`max_iter` inside).",
       "EXIT": "z at escape (inside: after the last step), components r, a, b, c.", "TRAPS": "Orbit traps (TRAP_DEFS order): smallest distance over the orbit.",
       "TRAP_STEP": "The step at which the TRAP_STEP_OF traps were closest (region outlines, \"domains\").",
       "STRIPES": "Stripe averages ½ + ½ sin(s·θ) (phase 1: cos) over the orbit, θ the angle of z in a plane: STRIPE_DEFS (plane, s, phase).",
       "AVERAGES": "Orbit averages, AVERAGE_NAMES order.", "CENTROID": "Mean r and mean a of the orbit.",
       "WINDING": "Total turning of the angle of z in each plane (winding), PLANES order.",
       "EARLY_ANGLE": "R-A angle of orbit points EARLY_STEPS (NaN if the orbit escaped before).", "EARLY_LOG": "ln|z| of the EARLY_STEPS points.",
       "EARLY_COMP": "Components of early orbit points: EARLY_COMP_DEFS (component, step), clamped to ±1e6.",
       "EARLY_OTHER": "Angle of early orbit points in other planes: EARLY_OTHER_DEFS (plane, step).",
       "EXTENT": "Largest |z| before escape (inside: over the whole orbit).", "PASSAGE": "First step with |z| above each PASSAGE_R radius (0 = never).",
       "EARLY_GROWTH": "ln|dz| per step after 5 and after 10 steps.", "LN_DZ": "ln|dz| at the end.", "LYAPUNOV": "ln|dz| per step.",
       "DE": "Distance estimate.", "PERIOD": "Inside: period of the final cycle (0 = none found).",
       "TRAP_ANGLE": "Angle of z about a centre in a plane at the step a trap was closest: TRAP_ANGLE_DEFS (trap, i, j, centre i, centre j)."}
m = [f"// BEGIN generated by scripts/gen_orbit_features.py\n/// Length of an orbit feature vector ([`quat_dag_orbit_features`]).\npub const ORBIT_FEATURES: usize = {NF};",
     """/// Slots of the feature vector (generated with the trap list). Planes are the six
/// coordinate planes of z, as pairs of components (r = 0, a = 1, b = 2, c = 3).
pub mod orbit_feature {
    pub const PLANES: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    pub const PLANE_NAMES: [&str; 6] = ["ra", "rb", "rc", "ab", "ac", "bc"];
    pub const COMP_NAMES: [&str; 4] = ["r", "a", "b", "c"];
    /// An object the orbit is measured against: the smallest distance over steps 1.. from z to it.
    #[derive(Clone, Copy, Debug)]
    pub enum Trap {
        Origin,
        /// The hyperplane where component k is 0.
        Plane(usize),
        /// The nearest of the four hyperplanes (smallest |component|).
        Cross,
        /// The sphere |z| = radius.
        Sphere(f64),
        /// The coordinate axis of component k.
        Axis(usize),
        /// The hyperplane z_i = sign · z_j.
        Diag(usize, usize, f64),
        /// The circle of the given radius in the (i, j) plane.
        Circle(usize, usize, f64),
        /// The circle of the given radius about (centre i, centre j) in the (i, j) plane.
        CircleAt(usize, usize, f64, f64, f64),
        /// The line {(x, y): -sin·x + cos·y = offset} in the (i, j) plane (direction (cos, sin)).
        Line(usize, usize, f64, f64, f64),
        Point([f64; 4]),
    }""",
     f"    pub const N_TRAPS: usize = {NT};",
     "    pub const TRAP_DEFS: [Trap; N_TRAPS] = [\n" + "".join(f"        {t[1]},\n" for t in traps) + "    ];",
     "    pub const TRAP_NAMES: [&str; N_TRAPS] = [\n" + "".join(f'        "{t[0]}",\n' for t in traps) + "    ];"]
for k, v in slot.items(): m.append(f"    /// {doc[k]}\n    pub const {k}: usize = {v};")
m.append("    pub const TRAP_STEP_OF: [usize; {}] = [{}];".format(len(dom), ", ".join(map(str, dom))))
m.append("    pub const STRIPE_DEFS: [(usize, u32, u8); {}] = [{}];".format(len(stripes), ", ".join(f"({p}, {s}, {ph})" for p, s, ph in stripes)))
m.append("    pub const AVERAGE_NAMES: [&str; {}] = [{}];".format(len(avg_names), ", ".join(f'"{a}"' for a in avg_names)))
m.append("    pub const EARLY_STEPS: [u32; {}] = [{}];".format(len(early_steps), ", ".join(map(str, early_steps))))
m.append("    pub const EARLY_COMP_DEFS: [(usize, u32); {}] = [{}];".format(len(early_comp), ", ".join(f"({c}, {st})" for c, st in early_comp)))
m.append("    pub const TRAP_ANGLE_DEFS: [(usize, usize, usize, f64, f64); {}] = [{}];".format(len(tang), ", ".join(f"({k}, {i}, {j}, {a:.2f}, {b:.2f})" for k, i, j, a, b in tang)))
ta = [-1] * NT
for a, t in enumerate(tang): ta[t[0]] = a
m.append("    /// Index into TRAP_ANGLE_DEFS for each trap, -1 for none.\n    pub const TRAP_ANGLE_OF: [i32; N_TRAPS] = [{}];".format(", ".join(map(str, ta))))
m.append("    pub const EARLY_OTHER_DEFS: [(usize, u32); {}] = [{}];".format(len(early_other), ", ".join(f"({p}, {s})" for p, s in early_other)))
m.append("    pub const PASSAGE_R: [f64; {}] = [{}];".format(len(passage), ", ".join(f"{r:.2f}" for r in passage)))
m.append("}\n// END generated by scripts/gen_orbit_features.py\n")
block = "\n".join(m)
p = "src/quat_dag.rs"; s = open(p).read()
if "// BEGIN generated by scripts/gen_orbit_features.py" in s:
    a = s.index("// BEGIN generated by scripts/gen_orbit_features.py")
    b = s.index("// END generated by scripts/gen_orbit_features.py\n") + len("// END generated by scripts/gen_orbit_features.py\n")
else:  # first run: the block starts at the ORBIT_FEATURES doc and ends before the function
    a = s.index("/// Length of an orbit feature vector ([`quat_dag_orbit_features`]).")
    b = s.index("\n/// Everything an orbit says about its starting point, as a flat vector") + 1
s = s[:a] + block + s[b:]
open(p, "w").write(s)
print(f"{NT} traps, {len(dom)} domains, {len(tang)} trap angles, {len(stripes)} stripes, {NF} slots")
