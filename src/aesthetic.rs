use std::io::{BufRead, BufReader, Write, BufWriter};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;

pub struct AestheticDisplay {
    pub current_clip:  f32,
    pub current_laion: f32,
    pub best_clip:     f32,
    pub best_laion:    f32,
    pub trend:         f32,
    pub n_samples:     usize,
}

#[derive(Clone, Debug, Default)]
pub struct AestheticScores {
    pub clip:      f32,
    pub laion:     f32,
    pub nima:      f32,
    pub topiq_iaa: f32,
    pub ap25:      f32,
    pub musiq:     f32,
    pub pref:      f32,
}

impl AestheticScores {
    /// Fractal-tuned "beauty" = mean of the three aesthetic predictors that
    /// discriminate fractals well (NIMA, TOPIQ-IAA, AP v2.5). Falls back to any
    /// subset that is present (non-zero); 0.0 if none available.
    pub fn ensemble(&self) -> f32 {
        let parts = [self.nima, self.topiq_iaa, self.ap25];
        let present: Vec<f32> = parts.iter().copied().filter(|&v| v > 0.0).collect();
        if present.is_empty() {
            0.0
        } else {
            present.iter().sum::<f32>() / present.len() as f32
        }
    }
}

/// Non-blocking aesthetic scorer backed by a Python sidecar process.
/// `new()` returns immediately; the Python CLIP model loads in the background.
pub struct AestheticScorer {
    tx: mpsc::Sender<PathBuf>,
    rx: mpsc::Receiver<AestheticScores>,
    ready_rx: Option<mpsc::Receiver<bool>>,
    is_ready: bool,
    history: Vec<(u64, AestheticScores)>,
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

        // Route the sidecar's stderr (model-load logs / tracebacks) to a file so
        // failures are diagnosable — a silent /dev/null makes "no score" impossible
        // to debug.
        let err_log = std::fs::File::create("aesthetic_sidecar.log")
            .map(Stdio::from)
            .unwrap_or_else(|_| Stdio::null());
        let mut child = Command::new(python)
            .arg("aesthetic_scorer.py")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(err_log)
            .spawn()
            .ok()?;

        let child_stdin  = child.stdin.take()?;
        let child_stdout = child.stdout.take()?;
        // Drop the process handle; Python exits when its stdin closes (when AestheticScorer drops).
        drop(child);

        let (path_tx, path_rx)   = mpsc::channel::<PathBuf>();
        let (score_tx, score_rx) = mpsc::channel::<AestheticScores>();
        let (ready_tx, ready_rx) = mpsc::channel::<bool>();

        thread::spawn(move || {
            let mut writer = BufWriter::new(child_stdin);
            let mut reader = BufReader::new(child_stdout);
            let mut line   = String::new();

            // Phase 1: wait for READY. Skip any stray stdout lines a library may emit
            // during model loading (e.g. pyiqa's "Loading pretrained model ..."), so a
            // noisy line can't be mistaken for a failed handshake.
            loop {
                line.clear();
                match reader.read_line(&mut line) {
                    Ok(0) => { ready_tx.send(false).ok(); return; } // EOF before READY
                    Ok(_) => {
                        if line.trim() == "READY" { ready_tx.send(true).ok(); break; }
                        // else: ignore noise and keep reading
                    }
                    Err(_) => { ready_tx.send(false).ok(); return; }
                }
            }
            drop(ready_tx);

            // Phase 2: scoring loop — one path in, "clip laion" out
            for path in path_rx {
                if writeln!(writer, "{}", path.display()).is_err() { break; }
                if writer.flush().is_err() { break; }
                line.clear();
                if reader.read_line(&mut line).is_err() { break; }
                let s = line.trim();
                if !s.starts_with("ERROR") {
                    // Protocol: "clip laion [nima topiq_iaa ap25 musiq]" — extra fields
                    // are optional so older sidecars still parse.
                    let f: Vec<f32> = s.split_whitespace()
                        .filter_map(|t| t.parse::<f32>().ok())
                        .collect();
                    if !f.is_empty() {
                        let get = |i: usize| f.get(i).copied().unwrap_or(0.0);
                        score_tx.send(AestheticScores {
                            clip:      get(0),
                            laion:     get(1),
                            nima:      get(2),
                            topiq_iaa: get(3),
                            ap25:      get(4),
                            musiq:     get(5),
                            pref:      get(6),
                        }).ok();
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
    pub fn poll(&mut self, generation: u64) -> Option<AestheticScores> {
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
            Ok(scores) => {
                self.pending_gen = None;
                self.history.push((generation, scores));
                if self.history.len() > 50 { self.history.remove(0); }
                Some(self.history.last().unwrap().1.clone())
            }
            Err(_) => None,
        }
    }

    /// Synchronous dual score: send image path, block until both models respond.
    /// Waits up to 60s for the scorer to become ready (handles slow CLIP model load at startup).
    pub fn score_blocking(&mut self, path: PathBuf) -> Option<AestheticScores> {
        if !self.is_ready {
            // The multi-model sidecar (CLIP+LAION+nima+topiq+ap25+musiq+SigLIP) takes
            // ~90-120s to load; allow generous time before giving up.
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(240);
            while !self.is_ready && std::time::Instant::now() < deadline {
                // Use map to avoid holding a borrow across the self-mutation below
                let result = self.ready_rx.as_ref()
                    .map(|rx| rx.recv_timeout(std::time::Duration::from_millis(200)));
                match result {
                    Some(Ok(true))  => { self.is_ready = true; self.ready_rx = None; }
                    Some(Ok(false)) => { self.ready_rx = None; return None; }
                    Some(Err(_))    => {} // still loading
                    None            => return None, // no ready_rx at all
                }
            }
            if !self.is_ready { return None; }
        }
        while self.rx.try_recv().is_ok() {}
        self.pending_gen = None;
        if self.tx.send(path).is_err() { return None; }
        match self.rx.recv_timeout(std::time::Duration::from_secs(15)) {
            Ok(scores) => Some(scores),
            Err(_)     => None,
        }
    }

    pub fn is_ready(&self) -> bool { self.is_ready }

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
            return "loading CLIP+LAION models...".to_string();
        }
        match self.display_info() {
            None => "ready, waiting for first sample...".to_string(),
            Some(d) => {
                let arrow = if d.trend > 0.02 { "↑" } else if d.trend < -0.02 { "↓" } else { "→" };
                format!(
                    "clip {:.3} (best {:.3})  laion {:.2} (best {:.2})  {}{:+.3}  ({} samples)",
                    d.current_clip, d.best_clip, d.current_laion, d.best_laion,
                    arrow, d.trend, d.n_samples
                )
            }
        }
    }

    fn display_info(&self) -> Option<AestheticDisplay> {
        if self.history.is_empty() { return None; }

        let clip_scores:  Vec<f32> = self.history.iter().map(|(_, s)| s.clip).collect();
        let laion_scores: Vec<f32> = self.history.iter().map(|(_, s)| s.laion).collect();

        let current_clip  = *clip_scores.last().unwrap();
        let current_laion = *laion_scores.last().unwrap();
        let best_clip     = clip_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let best_laion    = laion_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Trend on LAION score (wider range, more informative)
        let window = &laion_scores[laion_scores.len().saturating_sub(10)..];
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

        Some(AestheticDisplay { current_clip, current_laion, best_clip, best_laion, trend, n_samples: clip_scores.len() })
    }
}
