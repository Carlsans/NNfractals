use std::io::{BufRead, BufReader, Write, BufWriter};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;

/// Blocking image-based novelty scorer backed by the `novelty_scorer.py` Python
/// sidecar (frozen DINOv2 + a small VICReg-trained projection head — see
/// scripts/train_novelty.py). Mirrors `AestheticScorer`'s process/IPC shape
/// (`src/aesthetic.rs`). Novelty is a save-time signal (like nima/topiq/ap25),
/// not an async per-generation probe — nothing in the live `step()` loop
/// renders+scores individual genomes through a full transformer, so there's
/// no "poll every generation" use case here the way there is for the
/// display-only CLIP/LAION probe.
pub struct NoveltyScorer {
    tx: mpsc::Sender<PathBuf>,
    rx: mpsc::Receiver<(f32, Vec<f32>)>,
    ready_rx: Option<mpsc::Receiver<bool>>,
    is_ready: bool,
}

impl NoveltyScorer {
    /// Spawn the Python sidecar, pointed at `save_dir` so it can seed its
    /// novelty archive from `<save_dir>/.novelty_cache.npz` (built by
    /// `scripts/train_novelty.py`'s backfill) — novelty is measured against
    /// fractals already created, not just what this run has seen. Returns
    /// `None` if Python, the script, or the trained model files are missing;
    /// the caller degrades to novelty being inert, same contract as
    /// `AestheticScorer::new()`.
    pub fn new(save_dir: &Path) -> Option<Self> {
        Self::with_model(save_dir, None)
    }

    /// Same as `new`, but pointed at a custom `(model_path, head_path)` pair
    /// instead of the production `novelty_model.npz`/`novelty_head.pt` —
    /// e.g. a formula-specific model trained by `scripts/train_novelty.py`
    /// against a single exploration pool, kept separate so it doesn't
    /// disturb the production model other callers (live GA novelty scoring,
    /// the viewer's Explore feature) rely on.
    pub fn with_model(save_dir: &Path, model: Option<(&Path, &Path)>) -> Option<Self> {
        if !Path::new("novelty_scorer.py").exists() {
            return None;
        }

        let python = crate::python_bin(Path::new("."));

        let err_log = std::fs::File::create("novelty_sidecar.log")
            .map(Stdio::from)
            .unwrap_or_else(|_| Stdio::null());
        let mut cmd = Command::new(python);
        cmd.arg("novelty_scorer.py").arg(save_dir);
        if let Some((model_path, head_path)) = model {
            cmd.arg("--model-path").arg(model_path).arg("--head-path").arg(head_path);
        }
        let mut child = cmd
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(err_log)
            .spawn()
            .ok()?;

        let child_stdin  = child.stdin.take()?;
        let child_stdout = child.stdout.take()?;
        // Drop the process handle; Python exits when its stdin closes (when NoveltyScorer drops).
        drop(child);

        let (path_tx, path_rx)   = mpsc::channel::<PathBuf>();
        let (score_tx, score_rx) = mpsc::channel::<(f32, Vec<f32>)>();
        let (ready_tx, ready_rx) = mpsc::channel::<bool>();

        thread::spawn(move || {
            let mut writer = BufWriter::new(child_stdin);
            let mut reader = BufReader::new(child_stdout);
            let mut line   = String::new();

            // Phase 1: wait for READY, ignoring stray noise (same rationale as aesthetic.rs).
            loop {
                line.clear();
                match reader.read_line(&mut line) {
                    Ok(0) => { ready_tx.send(false).ok(); return; } // EOF before READY
                    Ok(_) => {
                        if line.trim() == "READY" { ready_tx.send(true).ok(); break; }
                    }
                    Err(_) => { ready_tx.send(false).ok(); return; }
                }
            }
            drop(ready_tx);

            // Phase 2: scoring loop — one path in, one "<score>|<v0>,<v1>,..." out.
            for path in path_rx {
                if writeln!(writer, "{}", path.display()).is_err() { break; }
                if writer.flush().is_err() { break; }
                line.clear();
                if reader.read_line(&mut line).is_err() { break; }
                let s = line.trim();
                if s.starts_with("ERROR") { continue; }
                let Some((score_str, vec_str)) = s.split_once('|') else { continue };
                let Ok(score) = score_str.parse::<f32>() else { continue };
                let vec: Vec<f32> = vec_str.split(',').filter_map(|x| x.parse::<f32>().ok()).collect();
                score_tx.send((score, vec)).ok();
            }
        });

        Some(Self {
            tx: path_tx,
            rx: score_rx,
            ready_rx: Some(ready_rx),
            is_ready: false,
        })
    }

    /// Blocks until the sidecar is ready (waiting up to 240s on the first
    /// call, for DINOv2 to finish loading — same generous budget as
    /// `AestheticScorer::score_blocking`'s comparably-sized multi-model
    /// load), sends `path`, and returns whatever comes back. Shared by
    /// `score_blocking`/`embed_blocking` so both pay the same one-time
    /// readiness wait and reuse the identical request/response plumbing —
    /// they differ only in which half of the response they keep.
    fn request_blocking(&mut self, path: PathBuf) -> Option<(f32, Vec<f32>)> {
        if !self.is_ready {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(240);
            while !self.is_ready && std::time::Instant::now() < deadline {
                let result = self.ready_rx.as_ref()
                    .map(|rx| rx.recv_timeout(std::time::Duration::from_millis(200)));
                match result {
                    Some(Ok(true))  => { self.is_ready = true; self.ready_rx = None; }
                    Some(Ok(false)) => { self.ready_rx = None; return None; }
                    Some(Err(_))    => {}
                    None            => return None,
                }
            }
            if !self.is_ready { return None; }
        }
        while self.rx.try_recv().is_ok() {}
        if self.tx.send(path).is_err() { return None; }
        self.rx.recv_timeout(std::time::Duration::from_secs(15)).ok()
    }

    /// Send an image path, block until its novelty score (distance to the
    /// archive) comes back.
    pub fn score_blocking(&mut self, path: PathBuf) -> Option<f32> {
        self.request_blocking(path).map(|(score, _)| score)
    }

    /// Send an image path, block until its (novelty score, 128-d
    /// L2-normalized latent embedding) comes back. The embedding is the
    /// same vector the archive novelty score is computed from — exposed so
    /// a caller can compare CANDIDATES TO EACH OTHER in latent space (e.g.
    /// greedy farthest-point diversity selection), not just each one's
    /// distance to the archive. Two L2-normalized vectors' Euclidean
    /// distance is a monotonic function of cosine similarity
    /// (`sqrt(2 - 2·cos_sim)`), so plain L2 distance between embeddings is
    /// already the right comparison, no extra normalization needed.
    pub fn embed_blocking(&mut self, path: PathBuf) -> Option<(f32, Vec<f32>)> {
        self.request_blocking(path)
    }
}
