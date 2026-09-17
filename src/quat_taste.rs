use std::io::{BufRead, BufReader, Write, BufWriter};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;

/// Mirrors `taste_model_quat.json` — the JSON twin `scripts/train_taste_quat.py`
/// writes alongside `taste_model_quat.npz` (the .npz is what Python actually
/// loads; this twin exists so Rust can sanity-check the model's config —
/// which backbone, which pooling, how many views it expects — at sidecar
/// startup without needing a numpy reader).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct TasteModelConfig {
    pub backbone: String,
    pub pooling: String,
    pub n_views: u32,
    pub dim: u32,
    pub lo: f32,
    pub hi: f32,
}

impl TasteModelConfig {
    pub fn load(path: &Path) -> Option<Self> {
        let s = std::fs::read_to_string(path).ok()?;
        serde_json::from_str(&s).ok()
    }
}

/// Blocking taste scorer backed by the `quat_taste_scorer.py` sidecar
/// (SigLIP backbone + `taste_model_quat.npz` linear head only — NOT the
/// 6-model 2D aesthetic ensemble, so it fits alongside a GA GPU render
/// process in under 1GB VRAM). Mirrors `AestheticScorer`/`NoveltyScorer`'s
/// process/IPC shape (`src/aesthetic.rs`, `src/novelty.rs`), but each
/// request line carries a command + a stable cache key (the genome's
/// content hash), not just a bare path — see `quat_taste_scorer.py`'s
/// module doc for the full protocol.
pub struct QuatTasteScorer {
    tx: mpsc::Sender<String>,
    rx: mpsc::Receiver<String>,
    ready_rx: Option<mpsc::Receiver<bool>>,
    is_ready: bool,
}

impl QuatTasteScorer {
    /// Spawn the production sidecar. Returns `None` if Python, the
    /// script, or `taste_model_quat.npz` are missing — the caller
    /// degrades to taste scoring being inert, same contract as
    /// `AestheticScorer::new()` / `NoveltyScorer::new()`. Never silently
    /// substitutes a generic scorer: a `None` here must propagate as "no
    /// taste score", not a fallback to some other metric.
    pub fn new() -> Option<Self> {
        if !Path::new("quat_taste_scorer.py").exists() || !Path::new("taste_model_quat.npz").exists() {
            return None;
        }
        let python = crate::python_bin(Path::new("."));
        let err_log = std::fs::File::create("quat_taste_sidecar.log")
            .map(Stdio::from)
            .unwrap_or_else(|_| Stdio::null());
        let mut cmd = Command::new(python);
        cmd.arg("quat_taste_scorer.py");
        Self::spawn(cmd, err_log)
    }

    fn spawn(mut cmd: Command, err_log: Stdio) -> Option<Self> {
        let mut child = cmd
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(err_log)
            .spawn()
            .ok()?;

        let child_stdin = child.stdin.take()?;
        let child_stdout = child.stdout.take()?;
        // Drop the process handle; the child exits when its stdin closes
        // (when QuatTasteScorer drops) — same lifetime contract as
        // AestheticScorer/NoveltyScorer.
        drop(child);

        let (req_tx, req_rx) = mpsc::channel::<String>();
        let (resp_tx, resp_rx) = mpsc::channel::<String>();
        let (ready_tx, ready_rx) = mpsc::channel::<bool>();

        thread::spawn(move || {
            let mut writer = BufWriter::new(child_stdin);
            let mut reader = BufReader::new(child_stdout);
            let mut line = String::new();

            // Phase 1: wait for READY, ignoring stray noise (same
            // rationale as aesthetic.rs/novelty.rs).
            loop {
                line.clear();
                match reader.read_line(&mut line) {
                    Ok(0) => { ready_tx.send(false).ok(); return; }
                    Ok(_) => {
                        if line.trim() == "READY" { ready_tx.send(true).ok(); break; }
                    }
                    Err(_) => { ready_tx.send(false).ok(); return; }
                }
            }
            drop(ready_tx);

            // Phase 2: one request line in, one response line out.
            for req in req_rx {
                if writeln!(writer, "{req}").is_err() { break; }
                if writer.flush().is_err() { break; }
                line.clear();
                if reader.read_line(&mut line).is_err() { break; }
                resp_tx.send(line.trim().to_string()).ok();
            }
        });

        Some(Self { tx: req_tx, rx: resp_rx, ready_rx: Some(ready_rx), is_ready: false })
    }

    #[cfg(test)]
    fn new_for_test(mut cmd: Command) -> Option<Self> {
        cmd.stderr(Stdio::null());
        Self::spawn(cmd, Stdio::null())
    }

    /// Waits (up to 240s — SigLIP's load time, same generous budget as
    /// `AestheticScorer::score_blocking`) for the sidecar's READY
    /// handshake. Returns false if the sidecar failed to start or load.
    fn ensure_ready(&mut self) -> bool {
        if self.is_ready { return true; }
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(240);
        while !self.is_ready && std::time::Instant::now() < deadline {
            let result = self.ready_rx.as_ref()
                .map(|rx| rx.recv_timeout(std::time::Duration::from_millis(200)));
            match result {
                Some(Ok(true)) => { self.is_ready = true; self.ready_rx = None; }
                Some(Ok(false)) => { self.ready_rx = None; return false; }
                Some(Err(_)) => {}
                None => return false,
            }
        }
        self.is_ready
    }

    fn request_line(&mut self, line: String) -> Option<String> {
        if !self.ensure_ready() { return None; }
        while self.rx.try_recv().is_ok() {}
        if self.tx.send(line).is_err() { return None; }
        self.rx.recv_timeout(std::time::Duration::from_secs(15)).ok()
    }

    fn fmt_request(cmd: &str, key: &str, paths: &[PathBuf]) -> String {
        let mut s = format!("{cmd}\t{key}");
        for p in paths {
            s.push('\t');
            s.push_str(&p.display().to_string());
        }
        s
    }

    /// Blocking single-genome taste score in [0,1] (normalised against the
    /// trained model's lo/hi), or `None` on any failure (sidecar not
    /// ready, model missing, render paths unreadable, ...).
    pub fn score_blocking(&mut self, key: &str, paths: &[PathBuf]) -> Option<f32> {
        let resp = self.request_line(Self::fmt_request("SCORE", key, paths))?;
        if resp.starts_with("ERROR") { return None; }
        resp.parse::<f32>().ok()
    }

    /// Batch API: one request line per genome, replies read in order — a
    /// whole generation's scoring is one round-trip through this call.
    /// `None` at an index means that genome failed to score (never
    /// silently substituted with a default fitness).
    pub fn score_many(&mut self, items: &[(String, Vec<PathBuf>)]) -> Vec<Option<f32>> {
        items.iter().map(|(key, paths)| self.score_blocking(key, paths)).collect()
    }

    /// Blocking (taste, L2-normalized backbone embedding) pair — exposed
    /// for the MAP-Elites role-model-distance axis and the archive's own
    /// embedding cache (Phase 2), same shape as
    /// `NoveltyScorer::embed_blocking`.
    pub fn embed_blocking(&mut self, key: &str, paths: &[PathBuf]) -> Option<(f32, Vec<f32>)> {
        let resp = self.request_line(Self::fmt_request("EMBED", key, paths))?;
        if resp.starts_with("ERROR") { return None; }
        let (score_str, vec_str) = resp.split_once('|')?;
        let score = score_str.parse::<f32>().ok()?;
        let vec: Vec<f32> = vec_str.split(',').filter_map(|x| x.parse::<f32>().ok()).collect();
        Some((score, vec))
    }

    /// Blocking cosine distance to the nearest role-model centroid, or
    /// `None` if `role_model_centroids.npz` hasn't been exported yet
    /// (Phase 2c, optional) or the sidecar isn't ready.
    pub fn distance_blocking(&mut self, key: &str, paths: &[PathBuf]) -> Option<f32> {
        let resp = self.request_line(Self::fmt_request("DIST", key, paths))?;
        if resp.starts_with("ERROR") { return None; }
        resp.parse::<f32>().ok()
    }

    pub fn is_ready(&self) -> bool { self.is_ready }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn taste_model_config_parses_the_json_twin() {
        // Exact shape scripts/train_taste_quat.py writes to
        // taste_model_quat.json.
        let json = r#"{
            "backbone": "siglip",
            "pooling": "single",
            "n_views": 9,
            "dim": 768,
            "lo": -3.1442697048187256,
            "hi": 2.3075780868530273
        }"#;
        let tmp = std::env::temp_dir().join(format!("taste_model_quat_test_{}.json", std::process::id()));
        std::fs::write(&tmp, json).unwrap();
        let cfg = TasteModelConfig::load(&tmp).expect("config should parse");
        std::fs::remove_file(&tmp).ok();
        assert_eq!(cfg.backbone, "siglip");
        assert_eq!(cfg.pooling, "single");
        assert_eq!(cfg.n_views, 9);
        assert_eq!(cfg.dim, 768);
        assert!((cfg.lo - (-3.1442697)).abs() < 1e-3);
        assert!((cfg.hi - 2.3075781).abs() < 1e-3);
    }

    #[test]
    fn taste_model_config_missing_file_returns_none() {
        let missing = std::env::temp_dir().join("definitely_not_a_real_taste_model_quat.json");
        assert!(TasteModelConfig::load(&missing).is_none());
    }

    /// A `python3 -c` stub implementing just enough of
    /// `quat_taste_scorer.py`'s protocol (READY handshake, then canned
    /// SCORE/EMBED/DIST/unknown-command replies) to exercise
    /// `QuatTasteScorer`'s request/response plumbing end-to-end without
    /// needing SigLIP or a trained model loaded.
    fn fake_sidecar_command() -> Command {
        let script = r#"
import sys
print("READY", flush=True)
for line in sys.stdin:
    line = line.strip()
    parts = line.split("\t")
    cmd = parts[0] if parts else ""
    if cmd == "SCORE":
        print("0.75000", flush=True)
    elif cmd == "EMBED":
        print("0.75000|0.100000,0.200000,0.300000", flush=True)
    elif cmd == "DIST":
        print("0.42000", flush=True)
    else:
        print("ERROR: unknown command", flush=True)
"#;
        let mut cmd = Command::new(crate::python_bin(Path::new(".")));
        cmd.arg("-c").arg(script);
        cmd
    }

    #[test]
    fn fake_sidecar_round_trip() {
        let mut scorer = QuatTasteScorer::new_for_test(fake_sidecar_command())
            .expect("fake sidecar should spawn");

        let paths = vec![PathBuf::from("does_not_need_to_exist.png")];
        let score = scorer.score_blocking("deadbeefcafef00d", &paths).expect("SCORE should reply");
        assert!((score - 0.75).abs() < 1e-4);

        let (score2, vec) = scorer.embed_blocking("deadbeefcafef00d", &paths).expect("EMBED should reply");
        assert!((score2 - 0.75).abs() < 1e-4);
        assert_eq!(vec.len(), 3);
        assert!((vec[0] - 0.1).abs() < 1e-4);

        let dist = scorer.distance_blocking("deadbeefcafef00d", &paths).expect("DIST should reply");
        assert!((dist - 0.42).abs() < 1e-4);

        let batch = scorer.score_many(&[
            ("k1".to_string(), paths.clone()),
            ("k2".to_string(), paths.clone()),
        ]);
        assert_eq!(batch.len(), 2);
        assert!(batch[0].is_some() && batch[1].is_some());
    }

    #[test]
    fn missing_sidecar_script_returns_none() {
        // new() checks quat_taste_scorer.py/taste_model_quat.npz in cwd
        // before spawning anything; only assertable when neither happens
        // to be present in the test binary's working directory.
        if !Path::new("quat_taste_scorer.py").exists() {
            assert!(QuatTasteScorer::new().is_none());
        }
    }
}
