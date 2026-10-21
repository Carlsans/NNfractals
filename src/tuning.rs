//! GA hyperparameter "tunings" and Optuna campaigns.
//!
//! A [`Tuning`] is one value per quat-GA operator hyperparameter plus the range the
//! optimizer may search. It is applied to `explorer quat-dag-evolve` through
//! [`Tuning::to_cli_args`]. A [`Campaign`] is the on-disk record of an endless sequence of
//! finite GA runs whose tunings come from an Optuna study (`scripts/optuna_tuner.py`).

use rand::Rng;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const TUNINGS_DIR: &str = "tunings";
pub const CAMPAIGNS_DIR: &str = "campaigns";

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Kind {
    Int,
    Float,
    /// `value` is an index into `Param::choices`.
    Choice,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Param {
    pub name: String,
    pub value: f64,
    pub min: f64,
    pub max: f64,
    #[serde(default)]
    pub log: bool,
    pub kind: Kind,
    /// Optimizer must keep this parameter at `value`.
    #[serde(default)]
    pub locked: bool,
    #[serde(default)]
    pub choices: Vec<String>,
    /// Hard limits the search range can be widened to (the full track of the GUI bar).
    #[serde(default)]
    pub lo: f64,
    #[serde(default)]
    pub hi: f64,
}

impl Param {
    fn new(name: &str, kind: Kind, value: f64, min: f64, max: f64, log: bool) -> Param {
        Param { name: name.into(), value, min, max, log, kind, locked: false, choices: Vec::new(), lo: min, hi: max }
    }

    fn limits(mut self, lo: f64, hi: f64) -> Param { self.lo = lo; self.hi = hi; self }

    /// Snap the search range into the hard limits and `value` into the range (integers
    /// for Int/Choice).
    pub fn clamp(&mut self) {
        if self.hi <= self.lo { self.lo = self.lo.min(self.min); self.hi = self.hi.max(self.max); }
        if self.max < self.min { std::mem::swap(&mut self.min, &mut self.max); }
        self.min = self.min.clamp(self.lo, self.hi);
        self.max = self.max.clamp(self.lo, self.hi);
        self.value = self.value.clamp(self.min, self.max);
        if self.kind != Kind::Float { self.value = self.value.round(); }
    }

    /// Position of `v` along the hard limits (the GUI track), log-scaled when `log`.
    pub fn track_of(&self, v: f64) -> f64 {
        let (lo, hi) = (self.lo, self.hi);
        if hi <= lo { return 0.5; }
        let u = if self.log && lo > 0.0 { (v.max(lo).ln() - lo.ln()) / (hi.ln() - lo.ln()) } else { (v - lo) / (hi - lo) };
        u.clamp(0.0, 1.0)
    }

    pub fn from_track(&self, u: f64) -> f64 {
        let u = u.clamp(0.0, 1.0);
        if self.log && self.lo > 0.0 { (self.lo.ln() + u * (self.hi.ln() - self.lo.ln())).exp() } else { self.lo + u * (self.hi - self.lo) }
    }

    /// Position of `value` in the range as 0..1 (log-scaled when `log`).
    pub fn to_unit(&self) -> f64 { self.unit_of(self.value) }

    pub fn unit_of(&self, v: f64) -> f64 {
        if self.max <= self.min { return 0.5; }
        let u = if self.log && self.min > 0.0 {
            (v.max(self.min).ln() - self.min.ln()) / (self.max.ln() - self.min.ln())
        } else {
            (v - self.min) / (self.max - self.min)
        };
        u.clamp(0.0, 1.0)
    }

    pub fn from_unit(&self, u: f64) -> f64 {
        let u = u.clamp(0.0, 1.0);
        if self.log && self.min > 0.0 {
            (self.min.ln() + u * (self.max.ln() - self.min.ln())).exp()
        } else {
            self.min + u * (self.max - self.min)
        }
    }

    /// Display text: the choice label, an integer, or a float with sensible digits.
    pub fn text(&self) -> String {
        match self.kind {
            Kind::Choice => self.choices.get(self.value as usize).cloned().unwrap_or_default(),
            Kind::Int => format!("{}", self.value as i64),
            Kind::Float => format!("{:.3}", self.value),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Optimizer { Tpe, CmaEs, Random, Qmc, Gp }

impl Optimizer {
    pub const ALL: [Optimizer; 5] = [Optimizer::Tpe, Optimizer::CmaEs, Optimizer::Random, Optimizer::Qmc, Optimizer::Gp];
    pub fn label(self) -> &'static str {
        match self { Optimizer::Tpe => "TPE", Optimizer::CmaEs => "CMA-ES", Optimizer::Random => "Random", Optimizer::Qmc => "QMC (Sobol)", Optimizer::Gp => "GP" }
    }
    /// The name `scripts/optuna_tuner.py` expects.
    pub fn key(self) -> &'static str {
        match self { Optimizer::Tpe => "tpe", Optimizer::CmaEs => "cmaes", Optimizer::Random => "random", Optimizer::Qmc => "qmc", Optimizer::Gp => "gp" }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Tuning {
    pub name: String,
    pub optimizer: Optimizer,
    pub params: Vec<Param>,
}

/// Prefix of the per-operation weight parameters (`op_sqr`, `op_mul`, `op_bulb`, ...).
pub const OP_PREFIX: &str = "op_";

/// Prefix of the generator rows (`gen_bulb_pow_min`, ...); sent as `--gen-params`.
pub const GEN_PREFIX: &str = "gen_";

/// (name, default, search min, search max, log, hard min, hard max)
pub const GEN_ROWS: &[(&str, f64, f64, f64, bool, f64, f64)] = &[
    ("gen_bulb_pow_min", 2.0, 1.0, 6.0, false, 0.5, 16.0),
    ("gen_bulb_pow_max", 9.0, 4.0, 14.0, false, 0.5, 16.0),
    ("gen_bulb_twist", 0.0, 0.0, 1.0, false, 0.0, 1.0),
    ("gen_kbc_scale", 0.5, 0.05, 1.5, true, 0.01, 4.0),
    ("gen_const_range", 1.0, 0.2, 3.0, true, 0.05, 8.0),
    ("gen_const_start", 0.5, 0.0, 1.0, false, 0.0, 1.0),
    ("gen_const_node", 0.05, 0.0, 0.3, false, 0.0, 1.0),
    ("gen_max_nodes", 24.0, 10.0, 24.0, false, 4.0, 24.0),
    ("gen_julia_prob", 0.30, 0.0, 1.0, false, 0.0, 1.0),
    ("gen_julia_scale", 0.9, 0.1, 1.5, false, 0.0, 3.0),
    ("gen_phoenix_prob", 0.25, 0.0, 1.0, false, 0.0, 1.0),
    ("gen_phoenix_scale", 0.5, 0.1, 1.5, false, 0.0, 3.0),
    ("gen_warp_prob", 0.30, 0.0, 1.0, false, 0.0, 1.0),
    ("gen_bailout_min", 2.0, 1.5, 6.0, false, 1.0, 50.0),
    ("gen_bailout_max", 10.0, 4.0, 20.0, false, 1.0, 50.0),
    ("gen_cut_pref_attempts", 3.0, 0.0, 6.0, false, 0.0, 6.0),
    ("gen_cut_leaf_retries", 3.0, 0.0, 6.0, false, 0.0, 8.0),
    ("gen_prune_min", 3.0, 3.0, 8.0, false, 2.0, 12.0),
    ("gen_comb_add", 1.0, 0.0, 1.0, false, 0.0, 3.0),
    ("gen_comb_sub", 1.0, 0.0, 1.0, false, 0.0, 3.0),
    ("gen_comb_mul", 1.0, 0.0, 1.0, false, 0.0, 3.0),
    ("gen_comb_div", 1.0, 0.0, 1.0, false, 0.0, 3.0),
];

/// Every op the GA can place, in display order: unary, binary, then the two that need a
/// companion constant (bulb, kbc). Names are `formula::op::name`.
pub fn op_param_ops() -> Vec<u8> {
    use crate::formula::op;
    let mut v: Vec<u8> = crate::genome::UNARY_OPS.to_vec();
    v.extend_from_slice(&crate::genome::BINARY_OPS);
    v.push(op::BULB);
    v.push(op::CONSTBC);
    v.push(crate::quat_genome_ops::KBC_MUL);
    v
}

/// Row name of an op in the weight table: `formula::op::name`, plus the splice-only `kbcmul`.
pub fn op_row_name(o: u8) -> &'static str {
    if o == crate::quat_genome_ops::KBC_MUL { "kbcmul" } else { crate::formula::op::name(o) }
}

/// Names of the five edit-kind weights, in `--edit-weights` order.
pub const EDIT_WEIGHT_NAMES: [&str; 5] = ["w_perturb", "w_swap", "w_rewire", "w_grow", "w_prune"];

impl Tuning {
    /// The default tuning: the 15 parameters `optuna_campaigns.py` searched, with ranges
    /// identical to that script and the middle-of-the-road values the GA ships with.
    pub fn defaults(name: &str) -> Tuning {
        let mut mode = Param::new("crossover_mode", Kind::Choice, 1.0, 0.0, 1.0, false);
        mode.choices = vec!["legacy".into(), "subtree".into()];
        let mut params = vec![
            Param::new("survivors", Kind::Int, 8.0, 4.0, 16.0, false).limits(2.0, 40.0),
            Param::new("immigrant_frac", Kind::Float, 0.10, 0.0, 0.3, false).limits(0.0, 1.0),
            // Share of the immigrants that are fresh random programs following the op weights
            // below (the rest are loaded from the pool folder).
            Param::new("random_immigrants", Kind::Float, 0.5, 0.0, 1.0, false).limits(0.0, 1.0),
            // Longest random immigrant, in nodes (its length is drawn from 3 up to this).
            Param::new("random_size", Kind::Int, 16.0, 1.0, 24.0, false).limits(1.0, 24.0),
            Param::new("crossover_frac", Kind::Float, 0.35, 0.1, 0.7, false).limits(0.0, 1.0),
            mode,
            Param::new("min_edits", Kind::Int, 1.0, 1.0, 2.0, false).limits(1.0, 4.0),
            Param::new("extra_edits", Kind::Int, 1.0, 0.0, 3.0, false).limits(0.0, 8.0),
            Param::new("const_scale", Kind::Float, 0.2, 0.02, 1.0, true).limits(0.005, 5.0),
            Param::new("max_depth", Kind::Int, 8.0, 4.0, 10.0, false).limits(3.0, 16.0),
        ];
        for n in EDIT_WEIGHT_NAMES {
            params.push(Param::new(n, Kind::Float, 0.5, 0.05, 1.0, false).limits(0.0, 2.0));
        }
        // Generator numbers the GA used to hard-code (bulb power, kbc/constant sizes, node
        // cap, immigrant dynamics); defaults reproduce the old behaviour.
        for (n, v, lo, hi, log, l0, l1) in GEN_ROWS {
            let kind = if matches!(*n, "gen_max_nodes" | "gen_cut_pref_attempts" | "gen_cut_leaf_retries" | "gen_prune_min") { Kind::Int } else { Kind::Float };
            params.push(Param::new(n, kind, *v, *lo, *hi, *log).limits(*l0, *l1));
        }
        // Per-op weights: relative chance that an op is chosen when a mutation swaps in or
        // grows an op. Defaults reproduce the GA's old split (unary ops together = binary ops
        // together), with bulb and kbc as ordinary entries.
        for o in op_param_ops() {
            use crate::formula::op;
            let v = match o { op::BULB => 0.15, op::CONSTBC | crate::quat_genome_ops::KBC_MUL => 0.10, _ if op::arity(o) == 2 => 0.25, _ => 0.07 };
            params.push(Param::new(&format!("{OP_PREFIX}{}", op_row_name(o)), Kind::Float, v, 0.0, 1.0, false).limits(0.0, 3.0));
        }
        Tuning { name: name.into(), optimizer: Optimizer::Tpe, params }
    }

    pub fn get(&self, name: &str) -> Option<&Param> { self.params.iter().find(|p| p.name == name) }
    pub fn get_mut(&mut self, name: &str) -> Option<&mut Param> { self.params.iter_mut().find(|p| p.name == name) }
    fn val(&self, name: &str) -> f64 { self.get(name).map(|p| p.value).unwrap_or(0.0) }

    pub fn clamp(&mut self) { for p in &mut self.params { p.clamp(); } }

    /// Draw every unlocked parameter uniformly (log-uniformly for `log`) inside its range.
    pub fn randomize<R: Rng>(&mut self, rng: &mut R) {
        for p in &mut self.params {
            if p.locked { continue; }
            p.value = p.from_unit(rng.random::<f64>());
            p.clamp();
        }
    }

    /// Replace values from an Optuna suggestion (`{name: value}`); unknown names are ignored.
    pub fn apply_values(&mut self, values: &serde_json::Map<String, serde_json::Value>) {
        for p in &mut self.params {
            let Some(v) = values.get(&p.name) else { continue };
            if let Some(f) = v.as_f64() { p.value = f; }
            else if let Some(s) = v.as_str() { if let Some(i) = p.choices.iter().position(|c| c == s) { p.value = i as f64; } }
            p.clamp();
        }
    }

    /// `{name: value}` for the Python bridge (choices as their label).
    pub fn values_json(&self) -> serde_json::Value {
        let mut m = serde_json::Map::new();
        for p in &self.params {
            let v = match p.kind {
                Kind::Choice => serde_json::Value::String(p.text()),
                Kind::Int => serde_json::json!(p.value as i64),
                Kind::Float => serde_json::json!(p.value),
            };
            m.insert(p.name.clone(), v);
        }
        serde_json::Value::Object(m)
    }

    /// The `explorer quat-dag-evolve` flags this tuning controls.
    pub fn to_cli_args(&self) -> Vec<String> {
        let min_edits = self.val("min_edits").round() as i64;
        let max_edits = min_edits + self.val("extra_edits").round() as i64;
        let weights: Vec<String> = EDIT_WEIGHT_NAMES.iter().map(|n| format!("{:.4}", self.val(n))).collect();
        let mode = self.get("crossover_mode").map(|p| p.text()).unwrap_or_else(|| "subtree".into());
        vec![
            "--survivors".into(), format!("{}", self.val("survivors").round() as i64),
            "--immigrant-frac".into(), format!("{:.4}", self.val("immigrant_frac")),
            "--crossover-frac".into(), format!("{:.4}", self.val("crossover_frac")),
            "--crossover-mode".into(), mode,
            "--mutation-min-edits".into(), min_edits.to_string(),
            "--mutation-max-edits".into(), max_edits.to_string(),
            "--mutation-const-scale".into(), format!("{:.4}", self.val("const_scale")),
            "--mutation-max-depth".into(), format!("{}", self.val("max_depth").round() as i64),
            "--edit-weights".into(), weights.join(","),
            "--op-weights".into(), self.op_weights_arg(),
            "--random-immigrant-frac".into(), format!("{:.4}", self.val("random_immigrants")),
            "--gen-params".into(), self.gen_params_arg(),
            "--random-immigrant-nodes".into(), format!("{}", self.val("random_size").round() as i64),
        ]
    }

    /// `name:weight,...` for `--op-weights`.
    pub fn op_weights_arg(&self) -> String {
        self.op_params().map(|p| format!("{}:{:.4}", &p.name[OP_PREFIX.len()..], p.value)).collect::<Vec<_>>().join(",")
    }

    /// The per-op weight parameters.
    pub fn op_params(&self) -> impl Iterator<Item = &Param> {
        self.params.iter().filter(|p| p.name.starts_with(OP_PREFIX))
    }

    /// Each op's share (0..1) of the total weight: its chance of being the op picked.
    pub fn op_shares(&self) -> Vec<(String, f64)> {
        let total: f64 = self.op_params().map(|p| p.value.max(0.0)).sum();
        self.op_params().map(|p| (p.name[OP_PREFIX.len()..].to_string(), if total > 0.0 { p.value.max(0.0) / total } else { 0.0 })).collect()
    }

    /// Bring an older saved tuning up to date: add parameters it lacks (new defaults), drop
    /// the retired `bulb_prob`/`kbc_prob` (now `op_bulb`/`op_kbc`), keep default ordering.
    /// `name:value,...` for `--gen-params` (the `gen_` prefix removed).
    pub fn gen_params_arg(&self) -> String {
        self.params.iter().filter_map(|p| Some(format!("{}:{}", p.name.strip_prefix(GEN_PREFIX)?, p.value))).collect::<Vec<_>>().join(",")
    }

    pub fn migrate(&mut self) {
        let d = Tuning::defaults("");
        self.params.retain(|p| d.get(&p.name).is_some());
        let mut out = Vec::with_capacity(d.params.len());
        for q in &d.params {
            out.push(self.params.iter().find(|p| p.name == q.name).cloned().unwrap_or_else(|| q.clone()));
        }
        self.params = out;
    }

    pub fn path(dir: &Path, name: &str) -> PathBuf { dir.join(format!("{}.json", sanitize(name))) }

    pub fn save(&self, dir: &Path) -> std::io::Result<()> {
        std::fs::create_dir_all(dir)?;
        let text = serde_json::to_string_pretty(self).map_err(std::io::Error::other)?;
        std::fs::write(Self::path(dir, &self.name), text)
    }

    pub fn load(path: &Path) -> Option<Tuning> {
        let mut t: Tuning = serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
        t.migrate();
        t.clamp();
        Some(t)
    }

    /// Names of the saved tunings in `dir`, sorted.
    pub fn list(dir: &Path) -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(dir).into_iter().flatten().flatten()
            .filter_map(|e| {
                let p = e.path();
                (p.extension().and_then(|x| x.to_str()) == Some("json")).then(|| p.file_stem()?.to_str().map(String::from))?
            })
            .collect();
        v.sort();
        v
    }
}

/// A file-system-safe name (letters, digits, `-`, `_`; everything else becomes `_`).
pub fn sanitize(name: &str) -> String {
    let s: String = name.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect();
    if s.is_empty() { "tuning".into() } else { s }
}

/// One finished GA run of a campaign.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunRecord {
    pub trial: usize,
    pub value: Option<f64>,
    pub params: serde_json::Value,
    pub seed: u64,
    pub secs: f64,
    /// "generations", "stop-flat", "max-seconds", "watchdog" or "failed".
    pub stop_reason: String,
    pub stem: Option<String>,
}

/// On-disk campaign: the editable run settings travel with the GUI's `QuatEvoPrefs`;
/// here only the tuning, optimizer and history are stored.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Campaign {
    pub name: String,
    pub tuning: Tuning,
    pub runs: Vec<RunRecord>,
}

impl Campaign {
    pub fn new(name: &str, tuning: Tuning) -> Campaign { Campaign { name: name.into(), tuning, runs: Vec::new() } }
    pub fn dir(root: &Path, name: &str) -> PathBuf { root.join(CAMPAIGNS_DIR).join(sanitize(name)) }
    pub fn file(root: &Path, name: &str) -> PathBuf { Self::dir(root, name).join("campaign.json") }

    pub fn save(&self, root: &Path) -> std::io::Result<()> {
        std::fs::create_dir_all(Self::dir(root, &self.name))?;
        let text = serde_json::to_string_pretty(self).map_err(std::io::Error::other)?;
        std::fs::write(Self::file(root, &self.name), text)
    }

    pub fn load(root: &Path, name: &str) -> Option<Campaign> {
        let mut c: Campaign = serde_json::from_str(&std::fs::read_to_string(Self::file(root, name)).ok()?).ok()?;
        c.tuning.migrate();
        c.tuning.clamp();
        Some(c)
    }

    pub fn list(root: &Path) -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(root.join(CAMPAIGNS_DIR)).into_iter().flatten().flatten()
            .filter(|e| e.path().join("campaign.json").exists())
            .filter_map(|e| e.file_name().to_str().map(String::from))
            .collect();
        v.sort();
        v
    }

    /// Best finished run so far.
    pub fn best(&self) -> Option<&RunRecord> {
        self.runs.iter().filter(|r| r.value.is_some())
            .max_by(|a, b| a.value.partial_cmp(&b.value).unwrap_or(std::cmp::Ordering::Equal))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn random_stays_in_range_and_respects_locks() {
        let mut t = Tuning::defaults("t");
        t.get_mut("op_bulb").unwrap().locked = true;
        let before = t.val("op_bulb");
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        for _ in 0..200 {
            t.randomize(&mut rng);
            for p in &t.params {
                assert!(p.value >= p.min && p.value <= p.max, "{} out of range", p.name);
                if p.kind != Kind::Float { assert_eq!(p.value, p.value.round()); }
            }
            assert_eq!(t.val("op_bulb"), before);
        }
    }

    #[test]
    fn json_round_trip_and_unit_mapping() {
        let t = Tuning::defaults("rt");
        let back: Tuning = serde_json::from_str(&serde_json::to_string(&t).unwrap()).unwrap();
        assert_eq!(t, back);
        let p = t.get("const_scale").unwrap();
        assert!((p.from_unit(p.unit_of(0.2)) - 0.2).abs() < 1e-9);
        assert!((p.from_unit(0.0) - 0.02).abs() < 1e-9 && (p.from_unit(1.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn track_mapping_and_range_clamping() {
        let mut t = Tuning::defaults("k");
        let p = t.get_mut("op_bulb").unwrap();
        assert!((p.track_of(1.5) - 0.5).abs() < 1e-9);
        p.max = 5.0; p.min = -1.0; p.value = 0.7;
        p.clamp();
        assert_eq!((p.min, p.max, p.value), (0.0, 3.0, 0.7), "range snaps into the hard limits");
        let c = t.get("const_scale").unwrap();
        assert!((c.from_track(c.track_of(0.2)) - 0.2).abs() < 1e-9);
    }

    #[test]
    fn cli_args_match_the_campaign_script() {
        let mut t = Tuning::defaults("c");
        t.get_mut("min_edits").unwrap().value = 2.0;
        t.get_mut("extra_edits").unwrap().value = 3.0;
        let a = t.to_cli_args();
        let at = |flag: &str| a[a.iter().position(|x| x == flag).unwrap() + 1].clone();
        assert_eq!(at("--mutation-min-edits"), "2");
        assert_eq!(at("--mutation-max-edits"), "5");
        assert_eq!(at("--crossover-mode"), "subtree");
        assert_eq!(at("--edit-weights").split(',').count(), 5);
        let ow = at("--op-weights");
        assert_eq!(ow.split(',').count(), 21, "14 unary + 4 binary + bulb + kbc + kbcmul");
        assert!(ow.contains("kbcmul:0.1000"));
        assert_eq!(at("--random-immigrant-frac"), "0.5000");
        assert!(ow.contains("bulb:0.1500") && ow.contains("kbc:0.1000") && ow.contains("sin:0.0700"));
    }

    #[test]
    fn shares_sum_to_one_and_old_tunings_migrate() {
        let t = Tuning::defaults("s");
        let total: f64 = t.op_shares().iter().map(|x| x.1).sum();
        assert!((total - 1.0).abs() < 1e-9);
        let mut old = Tuning::defaults("old");
        old.params.retain(|p| !p.name.starts_with("op_"));
        old.params.push(Param::new("bulb_prob", Kind::Float, 0.2, 0.0, 0.3, false));
        old.get_mut("survivors").unwrap().value = 11.0;
        old.migrate();
        assert!(old.get("bulb_prob").is_none() && old.get("op_bulb").is_some());
        assert_eq!(old.val("survivors"), 11.0, "existing values are kept");
        assert_eq!(old.params.len(), Tuning::defaults("").params.len());
    }

    #[test]
    fn optuna_values_apply_by_name_and_label() {
        let mut t = Tuning::defaults("o");
        let v: serde_json::Value = serde_json::json!({"survivors": 12, "crossover_mode": "legacy", "op_bulb": 9.0, "nope": 1});
        t.apply_values(v.as_object().unwrap());
        assert_eq!(t.val("survivors"), 12.0);
        assert_eq!(t.get("crossover_mode").unwrap().text(), "legacy");
        assert_eq!(t.val("op_bulb"), 1.0, "clamped to the range");
    }

    #[test]
    fn gen_rows_round_trip_into_the_explorer_flag() {
        let t = Tuning::defaults("x");
        let arg = t.gen_params_arg();
        let g = crate::quat_genome_ops::parse_gen_params(&arg).expect("every gen_ row is a known gen param");
        assert_eq!(g, crate::quat_genome_ops::GenParams::default(), "defaults must reproduce the old hard-coded numbers");
        assert!(t.to_cli_args().iter().any(|a| a == "--gen-params"));
    }
}
