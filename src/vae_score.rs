use std::io::{BufRead, BufReader, Write, BufWriter};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;

/// Blocking image-based VAE reconstruction-error scorer backed by the
/// `vae_scorer_sidecar.py` Python sidecar (a from-scratch per-formula VAE
/// trained on raw escape-time fields — see `scripts/autoencoder.py`/
/// `train_autoencoder.py`, and `vae_explore.rs` for the caller). Mirrors
/// `NoveltyScorer`'s (`src/novelty.rs`) exact process/IPC shape — same
/// READY-handshake, one-path-per-line protocol — simplified to a single
/// f32 score, no embedding vector (reconstruction error alone is the whole
/// signal here, unlike novelty's archive-distance-plus-latent-vector).
///
/// Unlike `NoveltyScorer`, there is no shared "production" checkpoint to
/// fall back to — the VAE is per-formula by design, so `new` always
/// requires an explicit `model_path`. The checkpoint changes every outer
/// `vae-explore` iteration (retrained on the grown corpus each time); the
/// caller must drop and respawn a `VaeScorer` after each retrain — sidecars
/// load weights once at startup and don't hot-reload.
pub struct VaeScorer {
    tx: mpsc::Sender<PathBuf>,
    rx: mpsc::Receiver<f32>,
    ready_rx: Option<mpsc::Receiver<bool>>,
    is_ready: bool,
}

impl VaeScorer {
    /// Spawn the Python sidecar pointed at `model_path` (a combined
    /// trunk+... checkpoint from `train_autoencoder.py`). Returns `None` if
    /// Python, the script, or the model file are missing — the caller
    /// degrades to scoring being inert (`vae_explore`'s random-selection
    /// fallback), same contract as `NoveltyScorer::new()`.
    pub fn new(model_path: &Path) -> Option<Self> {
        if !Path::new("vae_scorer_sidecar.py").exists() {
            return None;
        }

        let python = crate::python_bin(Path::new("."));

        let err_log = std::fs::File::create("vae_scorer_sidecar.log")
            .map(Stdio::from)
            .unwrap_or_else(|_| Stdio::null());
        let mut child = Command::new(python)
            .arg("vae_scorer_sidecar.py")
            .arg("--model-path").arg(model_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(err_log)
            .spawn()
            .ok()?;

        let child_stdin  = child.stdin.take()?;
        let child_stdout = child.stdout.take()?;
        drop(child);

        let (path_tx, path_rx)   = mpsc::channel::<PathBuf>();
        let (score_tx, score_rx) = mpsc::channel::<f32>();
        let (ready_tx, ready_rx) = mpsc::channel::<bool>();

        thread::spawn(move || {
            let mut writer = BufWriter::new(child_stdin);
            let mut reader = BufReader::new(child_stdout);
            let mut line   = String::new();

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

            for path in path_rx {
                if writeln!(writer, "{}", path.display()).is_err() { break; }
                if writer.flush().is_err() { break; }
                line.clear();
                if reader.read_line(&mut line).is_err() { break; }
                let s = line.trim();
                if s.starts_with("ERROR") { continue; }
                let Ok(score) = s.parse::<f32>() else { continue };
                score_tx.send(score).ok();
            }
        });

        Some(Self { tx: path_tx, rx: score_rx, ready_rx: Some(ready_rx), is_ready: false })
    }

    /// Blocks until the sidecar is ready (up to 240s on the first call, for
    /// the model to finish loading — same budget as `NoveltyScorer`), sends
    /// `path`, returns the reconstruction-error score.
    pub fn score_blocking(&mut self, path: PathBuf) -> Option<f32> {
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
}
