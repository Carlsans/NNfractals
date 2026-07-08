# Installing NNfractals

*[Ce guide est aussi disponible en français : INSTALL.fr.md](INSTALL.fr.md)*

NNfractals is a Rust genetic-algorithm fractal evolver with three optional
GUIs (viewer, browser, launcher) and a Python sidecar for aesthetic scoring.
This guide gets a fresh Linux desktop machine from clone to running build.

**A GPU is entirely optional.** Everything here works on CPU-only machines —
see [GPU vs CPU](#6-gpu-vs-cpu--no-gpu-required) below.

## 1. Prerequisites

- A desktop Linux distribution — Debian/Ubuntu, Fedora, or Arch (and their
  derivatives: Mint, Pop!_OS, RHEL/CentOS, Manjaro, EndeavourOS, …).
- `git`.
- Internet access (to fetch Rust crates, PyPI packages, and — on first run —
  a few pretrained models from Hugging Face).

## 2. Quick install

```sh
git clone https://github.com/Carlsans/NNfractals.git
cd NNfractals
./scripts/install-deps.sh
```

This one script:
1. Installs the system libraries the GUIs need (X11/Wayland/GL/Vulkan) via
   your distro's package manager (apt, dnf, or pacman — auto-detected).
2. Installs/updates Rust via [rustup](https://rustup.rs) if your distro's
   Rust is too old (this project needs rustc ≥ 1.85, for Rust edition 2024).
3. Creates an isolated Python virtualenv at `.venv/` and installs the right
   `torch` wheel — CPU-only or CUDA, based on whether it detects an NVIDIA
   GPU — plus every other Python dependency.
4. Builds everything: `cargo build --release --features "viewer browser launcher"`.

It's safe to re-run: already-installed system packages, an existing `.venv/`,
and an up-to-date Rust toolchain are all detected and skipped.

## 3. Manual install (or unsupported distros)

If you're not on apt/dnf/pacman, or you'd rather do it by hand — this is
exactly what the script above automates.

### System packages

| Distro family | Command |
|---|---|
| Debian/Ubuntu (apt) | `sudo apt-get install -y build-essential pkg-config curl libx11-dev libxkbcommon-dev libxkbcommon-x11-0 libwayland-dev libxrandr-dev libxi-dev libxcursor-dev libgl1-mesa-dev libegl1-mesa-dev mesa-vulkan-drivers python3 python3-venv python3-pip` |
| Fedora/RHEL (dnf) | `sudo dnf install -y gcc gcc-c++ make pkgconf-pkg-config curl libX11-devel libxkbcommon-devel libxkbcommon-x11 wayland-devel libXrandr-devel libXi-devel libXcursor-devel mesa-libGL-devel mesa-libEGL-devel mesa-vulkan-drivers python3 python3-pip` |
| Arch/Manjaro (pacman) | `sudo pacman -S --needed base-devel pkgconf curl libx11 libxkbcommon libxkbcommon-x11 wayland libxrandr libxi libxcursor mesa vulkan-icd-loader python python-pip` |

These cover the `eframe`/`winit` GUI stack (X11 *and* Wayland, plus
GL/Vulkan for rendering — including Mesa's software fallback used when
there's no GPU) and Python 3 + venv.

### Rust

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
rustc --version   # confirm >= 1.85
```

### Python virtualenv

```sh
python3 -m venv .venv
.venv/bin/pip install --upgrade pip

# Pick ONE of these, matching your hardware:
.venv/bin/pip install torch                                                    # NVIDIA GPU present
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU only

.venv/bin/pip install -r requirements.txt
```

### Build

```sh
cargo build --release --features "viewer browser launcher"
```

## 4. Running

| Command | What it does |
|---|---|
| `./run.sh` | Starts background evolution instance(s) (the core GA loop). `./run.sh --build` rebuilds first; `./run.sh N` starts N instances. |
| `./target/release/nnfractals-launcher` | GUI front door: start/stop/status evolution, train the taste model, dedup a folder, open the browser/viewer. |
| `./target/release/nnfractals-browser` | Gallery: sort/filter fractals, pairwise-rate them, curate a Starred folder. |
| `./target/release/nnfractals-viewer <file.nn>` | Deep-zoom render/save a single fractal. |

The `--features` you build with control which of these binaries exist —
`viewer`/`browser`/`launcher` are independent (`cargo build --release
--features browser` alone, for instance, only builds `nnfractals` +
`nnfractals-browser`). The evolution engine (`nnfractals`) itself needs no
feature flags.

## 5. First run notes

The aesthetic-scoring sidecar (used by evolution's fitness function, by
`nnfractals-launcher`'s "Train taste model"/"Rescore", and by dedup)
downloads several pretrained models from Hugging Face **the first time
they're used**: SigLIP, DINOv2, `pyiqa`'s NIMA/TOPIQ/MUSIQ weights, and
Aesthetic Predictor v2.5. Expect:
- Internet access needed once per model (cached afterwards under
  `~/.cache/huggingface`).
- A few GB of disk space.
- The first score after startup will be slow while models load; subsequent
  ones are fast.

## 6. GPU vs CPU — no GPU required

A GPU is optional everywhere in this project:

- **Evolution & rendering** (`nnfractals`, `nnfractals-viewer`) are always
  built with GPU rendering support (`wgpu-backend`), but at startup they look
  for a usable GPU and, if none is found, print `[gpu] No wgpu adapter —
  using CPU renderer.` and transparently switch to a parallel CPU (Rayon)
  renderer instead. Same binary, same command — just slower without a GPU.
- **Aesthetic scoring** (the Python sidecar, training, dedup) checks
  `torch.cuda.is_available()` before every model call and falls back to CPU
  automatically. `install-deps.sh` installs the CPU-only `torch` wheel when
  no NVIDIA GPU is detected, so you don't pay for a multi-gigabyte CUDA
  download you can't use.
- **The Browser and Launcher GUIs** don't touch the GPU compute path at
  all — they render via OpenGL, which Mesa provides in software
  (`llvmpipe`) when there's no GPU.

## 7. Desktop integration (optional)

```sh
./target/release/nnfractals-launcher --install-desktop   # adds NNFractals to your app menu
bash dist/install-viewer.sh                               # + double-click .nn files to open the viewer
```

## 8. Troubleshooting

- **`error: package requires... edition2024` / build fails on an old
  rustc** — your distro's Rust is too old. Run
  `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh` and
  `source "$HOME/.cargo/env"`, then rebuild. This project needs rustc ≥ 1.85.
- **GUI window fails to open / X11 or Wayland errors** — double check the
  system package list in §3 was fully installed for your distro; on Wayland
  sessions specifically, `libwayland-dev`/`wayland-devel`/`wayland` (per
  distro) must be present.
- **`python3-venv` / "ensurepip is not available"** — on Debian/Ubuntu,
  `venv` ships as the separate `python3-venv` package; install it and re-run
  `python3 -m venv .venv`.
- **Aesthetic scoring silently produces no scores** — check
  `aesthetic_sidecar.log` (evolution) or `train_pref.log` (launcher jobs) in
  the project root; these capture the Python sidecar's stderr/tracebacks.
- **You notice an `ndarray-backend` Cargo feature in `Cargo.toml`** — ignore
  it, it's currently a dead stub with no effect. The CPU fallback described
  in §6 already works via the default `wgpu-backend` build; you never need
  to pass `--features ndarray-backend`.
