use std::io::{BufRead, BufReader, Write, BufWriter};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;

/// A predicted heatmap: `data` is row-major (row 0 = ymin, col 0 = xmin —
/// same convention `render_escape_times` itself uses, see
/// `cmd_saliency_data`'s doc comment in `src/bin/explorer.rs`), `height`/
/// `width` its shape. Values are raw (log-space, unscaled) scores from
/// `SaliencyNet` — only their RELATIVE ordering matters for picking peaks,
/// not their absolute scale.
pub struct Heatmap {
    pub data: Vec<f32>,
    pub height: usize,
    pub width: usize,
}

impl Heatmap {
    pub fn at(&self, row: usize, col: usize) -> f32 {
        self.data[row * self.width + col]
    }
}

/// Blocking canvas -> saliency-heatmap client backed by `saliency_sidecar.py`
/// (a fully-convolutional net — see `scripts/saliency_model.py`) — mirrors
/// `VaeScorer`'s (`src/vae_score.rs`) exact process/IPC shape, differing
/// only in the response: a whole spatial heatmap (JSON-encoded), not one
/// f32, since this model predicts WHERE in a canvas is interesting rather
/// than scoring one already-cropped image.
pub struct SaliencyScorer {
    tx: mpsc::Sender<PathBuf>,
    rx: mpsc::Receiver<Heatmap>,
    ready_rx: Option<mpsc::Receiver<bool>>,
    is_ready: bool,
}

impl SaliencyScorer {
    /// Spawn the Python sidecar pointed at `model_path`. Returns `None` if
    /// Python, the script, or the model file are missing — same graceful-
    /// degrade contract as `VaeScorer::new`/`NoveltyScorer::new`: the
    /// caller falls back to the existing grid-sweep `coarse_scan` when this
    /// is `None`, so a saliency model is optional, not required.
    pub fn new(model_path: &Path) -> Option<Self> {
        if !Path::new("saliency_sidecar.py").exists() {
            return None;
        }

        let python = crate::python_bin(Path::new("."));

        let err_log = std::fs::File::create("saliency_sidecar.log")
            .map(Stdio::from)
            .unwrap_or_else(|_| Stdio::null());
        let mut child = Command::new(python)
            .arg("saliency_sidecar.py")
            .arg("--model-path").arg(model_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(err_log)
            .spawn()
            .ok()?;

        let child_stdin  = child.stdin.take()?;
        let child_stdout = child.stdout.take()?;
        drop(child);

        let (path_tx, path_rx)       = mpsc::channel::<PathBuf>();
        let (heatmap_tx, heatmap_rx) = mpsc::channel::<Heatmap>();
        let (ready_tx, ready_rx)     = mpsc::channel::<bool>();

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
                let Ok(v) = serde_json::from_str::<serde_json::Value>(line.trim()) else { continue };
                if v.get("error").is_some() { continue; }
                let (Some(h), Some(w), Some(data)) = (
                    v.get("h").and_then(|x| x.as_u64()),
                    v.get("w").and_then(|x| x.as_u64()),
                    v.get("data").and_then(|x| x.as_array()),
                ) else { continue };
                let data: Vec<f32> = data.iter().filter_map(|x| x.as_f64()).map(|x| x as f32).collect();
                if data.len() != (h * w) as usize { continue; }
                heatmap_tx.send(Heatmap { data, height: h as usize, width: w as usize }).ok();
            }
        });

        Some(Self { tx: path_tx, rx: heatmap_rx, ready_rx: Some(ready_rx), is_ready: false })
    }

    /// Blocks until the sidecar is ready (up to 240s on the first call, for
    /// the model to finish loading — same budget as `VaeScorer`), sends
    /// `path`, returns the predicted heatmap.
    pub fn score_blocking(&mut self, path: PathBuf) -> Option<Heatmap> {
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
