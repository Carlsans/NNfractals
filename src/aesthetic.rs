use std::io::{BufRead, BufReader, Write, BufWriter};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;

pub struct AestheticDisplay {
    pub current: f32,
    pub best: f32,
    pub trend: f32,
    pub n_samples: usize,
}

/// Non-blocking aesthetic scorer backed by a Python sidecar process.
/// `new()` returns immediately; the Python CLIP model loads in the background.
pub struct AestheticScorer {
    tx: mpsc::Sender<PathBuf>,
    rx: mpsc::Receiver<f32>,
    ready_rx: Option<mpsc::Receiver<bool>>,
    is_ready: bool,
    history: Vec<(u64, f32)>,
    pending_gen: Option<u64>,
}

impl AestheticScorer {
    /// Spawn the Python sidecar. Returns None if Python or aesthetic_scorer.py is missing.
    pub fn new() -> Option<Self> {
        if !std::path::Path::new("aesthetic_scorer.py").exists() {
            return None;
        }

        let python = ["python3", "python"]
            .iter()
            .find(|&&cmd| {
                Command::new(cmd)
                    .arg("--version")
                    .output()
                    .map(|o| o.status.success())
                    .unwrap_or(false)
            })
            .copied()?;

        let mut child = Command::new(python)
            .arg("aesthetic_scorer.py")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .ok()?;

        let child_stdin  = child.stdin.take()?;
        let child_stdout = child.stdout.take()?;
        // Drop the process handle; Python exits when its stdin closes (when AestheticScorer drops).
        drop(child);

        let (path_tx, path_rx)   = mpsc::channel::<PathBuf>();
        let (score_tx, score_rx) = mpsc::channel::<f32>();
        let (ready_tx, ready_rx) = mpsc::channel::<bool>();

        thread::spawn(move || {
            let mut writer = BufWriter::new(child_stdin);
            let mut reader = BufReader::new(child_stdout);
            let mut line   = String::new();

            // Phase 1: wait for READY (CLIP model loading takes 30–60s on first run)
            match reader.read_line(&mut line) {
                Ok(_) if line.trim() == "READY" => { ready_tx.send(true).ok(); }
                _                               => { ready_tx.send(false).ok(); return; }
            }
            drop(ready_tx);

            // Phase 2: scoring loop — one path in, one float out
            for path in path_rx {
                if writeln!(writer, "{}", path.display()).is_err() { break; }
                if writer.flush().is_err() { break; }
                line.clear();
                if reader.read_line(&mut line).is_err() { break; }
                let s = line.trim();
                if !s.starts_with("ERROR") {
                    if let Ok(score) = s.parse::<f32>() {
                        score_tx.send(score).ok();
                    }
                }
            }
        });

        Some(Self {
            tx: path_tx,
            rx: score_rx,
            ready_rx: Some(ready_rx),
            is_ready: false,
            history: Vec::new(),
            pending_gen: None,
        })
    }

    /// Call every generation to advance internal state. Returns Some(score) if a new score arrived.
    pub fn poll(&mut self, generation: u64) -> Option<f32> {
        if !self.is_ready {
            if let Some(rr) = &self.ready_rx {
                match rr.try_recv() {
                    Ok(true)  => { self.is_ready = true; self.ready_rx = None; }
                    Ok(false) => { self.ready_rx = None; } // failed to load
                    Err(_)    => {}                         // still loading
                }
            }
        }
        match self.rx.try_recv() {
            Ok(score) => {
                self.pending_gen = None;
                self.history.push((generation, score));
                if self.history.len() > 50 { self.history.remove(0); }
                Some(score)
            }
            Err(_) => None,
        }
    }

    /// Synchronous score: send image path, block until CLIP returns a score.
    /// Returns None if scorer is not ready or scoring fails.
    /// Clears any in-flight async request first.
    pub fn score_blocking(&mut self, path: PathBuf) -> Option<f32> {
        if !self.is_ready { return None; }
        // Drain any stale pending result
        while self.rx.try_recv().is_ok() {}
        self.pending_gen = None;

        if self.tx.send(path).is_err() { return None; }
        // Block until the scorer responds (typically 50–100ms on GPU)
        match self.rx.recv_timeout(std::time::Duration::from_secs(10)) {
            Ok(score) => Some(score),
            Err(_)    => None,
        }
    }

    /// Submit an image for scoring. No-op if scorer not ready or a request is already in flight.
    pub fn request(&mut self, path: PathBuf, generation: u64) {
        if self.is_ready && self.pending_gen.is_none() {
            if self.tx.send(path).is_ok() {
                self.pending_gen = Some(generation);
            }
        }
    }

    pub fn is_loading(&self) -> bool {
        !self.is_ready && self.ready_rx.is_some()
    }

    /// One-line summary for the display header.
    pub fn status_line(&self) -> String {
        if self.is_loading() {
            return "loading CLIP model...".to_string();
        }
        match self.display_info() {
            None => "ready, waiting for first sample...".to_string(),
            Some(d) => {
                let arrow = if d.trend > 0.02 { "↑" } else if d.trend < -0.02 { "↓" } else { "→" };
                format!(
                    "{:.2}  best {:.2}  trend {}{:+.2}  ({} samples)",
                    d.current, d.best, arrow, d.trend, d.n_samples
                )
            }
        }
    }

    fn display_info(&self) -> Option<AestheticDisplay> {
        if self.history.is_empty() { return None; }

        let scores: Vec<f32> = self.history.iter().map(|(_, s)| *s).collect();
        let current = *scores.last().unwrap();
        let best    = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let window = &scores[scores.len().saturating_sub(10)..];
        let trend  = if window.len() >= 2 {
            let n      = window.len() as f32;
            let x_mean = (n - 1.0) / 2.0;
            let y_mean = window.iter().sum::<f32>() / n;
            let num: f32 = window.iter().enumerate()
                .map(|(i, &y)| (i as f32 - x_mean) * (y - y_mean))
                .sum();
            let den: f32 = (0..window.len())
                .map(|i| (i as f32 - x_mean).powi(2))
                .sum::<f32>();
            if den > 1e-8 { num / den } else { 0.0 }
        } else { 0.0 };

        Some(AestheticDisplay { current, best, trend, n_samples: scores.len() })
    }
}
