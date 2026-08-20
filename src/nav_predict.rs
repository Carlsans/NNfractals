use std::io::{BufRead, BufReader, Write, BufWriter};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;

/// Blocking image-based navigation predictor backed by the
/// `nav_predict_sidecar.py` Python sidecar (frozen backbone + a small
/// trained regression head — see scripts/train_navigate.py). Mirrors
/// `AestheticScorer`/`NoveltyScorer`'s exact process/IPC shape (same
/// READY-handshake, one-path-per-line protocol) — see those for the
/// established pattern this follows.
///
/// Predicts `(u, v, log_zoom_ratio)` from an image of the CURRENT view:
/// where within that view's own frame Carl would zoom next (u, v roughly
/// in [-1,1]) and by what log-ratio the zoom deepens — the exact
/// `(dx, dy, zoom)` parameterization `explore::apply_offset`/
/// `sweep_positions` already use internally. `to_offset` converts the
/// prediction into that concrete form for a given reference view.
pub struct NavPredictor {
    tx: mpsc::Sender<PathBuf>,
    rx: mpsc::Receiver<(f32, f32, f32)>,
    ready_rx: Option<mpsc::Receiver<bool>>,
    is_ready: bool,
}

impl NavPredictor {
    /// Spawn the Python sidecar. Returns `None` if Python, the script, or
    /// the trained model files are missing — the caller degrades to
    /// prediction being inert, same contract as `AestheticScorer::new()`/
    /// `NoveltyScorer::new()`. `model`: `Some((model_path, head_path))` for
    /// a non-default location, `None` for the default `nav_model.npz`/
    /// `nav_head.pt`.
    pub fn new(model: Option<(&Path, &Path)>) -> Option<Self> {
        if !Path::new("nav_predict_sidecar.py").exists() {
            return None;
        }

        let python = crate::python_bin(Path::new("."));

        let err_log = std::fs::File::create("nav_predict_sidecar.log")
            .map(Stdio::from)
            .unwrap_or_else(|_| Stdio::null());
        let mut cmd = Command::new(python);
        cmd.arg("nav_predict_sidecar.py");
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
        drop(child);

        let (path_tx, path_rx) = mpsc::channel::<PathBuf>();
        let (pred_tx, pred_rx) = mpsc::channel::<(f32, f32, f32)>();
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
                let parts: Vec<&str> = s.split(',').collect();
                if parts.len() != 3 { continue; }
                let (Ok(u), Ok(v), Ok(lz)) = (parts[0].parse::<f32>(), parts[1].parse::<f32>(), parts[2].parse::<f32>()) else { continue };
                pred_tx.send((u, v, lz)).ok();
            }
        });

        Some(Self { tx: path_tx, rx: pred_rx, ready_rx: Some(ready_rx), is_ready: false })
    }

    /// Blocks until the sidecar is ready (up to 240s on the first call, for
    /// the backbone to finish loading — same budget as
    /// `AestheticScorer`/`NoveltyScorer`), sends `path`, returns the
    /// predicted `(u, v, log_zoom_ratio)`.
    pub fn predict_blocking(&mut self, path: PathBuf) -> Option<(f32, f32, f32)> {
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

/// Converts a `(u, v, log_zoom_ratio)` prediction into the concrete
/// `(dx, dy, zoom)` offset `explore::apply_offset` expects, given the
/// reference `View` the prediction was made FROM (i.e. the one whose image
/// was fed to `predict_blocking`). Mirrors `nav_label`'s inverse
/// (`src/bin/explorer.rs`'s `prep-nav-data`, which computes `(u,v,log_zoom_
/// ratio)` from a real `(before, after)` pair) — this is that computation
/// run backwards, same geometry both directions.
pub fn to_offset(view: &crate::video_export::View, u: f32, v: f32, log_zoom_ratio: f32) -> (f64, f64, f64) {
    let half_x = 2.0 / view.zoom * view.aspect;
    let half_y = 2.0 / view.zoom;
    let dx = u as f64 * half_x;
    let dy = v as f64 * half_y;
    let zoom = view.zoom * (log_zoom_ratio as f64).exp();
    (dx, dy, zoom)
}
