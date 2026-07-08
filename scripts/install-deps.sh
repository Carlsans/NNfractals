#!/usr/bin/env bash
# NNfractals — one-command dependency installer + build.
#
# Detects your package manager (apt/dnf/pacman), installs the system
# libraries needed to build the GUIs (eframe/winit: X11+Wayland+GL/Vulkan),
# makes sure Rust is new enough for this project's edition 2024 (rustc
# >= 1.85, installing/updating via rustup if not), creates a project-local
# Python virtualenv (.venv/) with the right torch wheel (CPU-only or CUDA,
# based on whether an NVIDIA GPU is detected) plus every other Python
# dependency, then builds all four binaries in release mode.
#
# No GPU is required anywhere in this project — see INSTALL.md's "GPU vs
# CPU" section. Safe to re-run (idempotent): already-installed packages,
# an existing .venv/, and an up-to-date Rust toolchain are all skipped.
#
# Usage:  ./scripts/install-deps.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

MIN_RUSTC_MAJOR=1
MIN_RUSTC_MINOR=85

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  SUDO="sudo"
fi

echo "==> NNfractals dependency installer"
echo "    project root: $PROJECT_DIR"
echo

# ── 1. System packages ──────────────────────────────────────────────────────

# eframe/winit on Linux need X11 + Wayland + GL/Vulkan dev headers and
# runtime libs (most winit backends dlopen at runtime, but building/running
# reliably wants these present); mesa's Vulkan/GL drivers also provide the
# software (llvmpipe/lavapipe) fallback used when no GPU is present.
APT_PKGS="build-essential pkg-config curl libx11-dev libxkbcommon-dev libxkbcommon-x11-0 libwayland-dev libxrandr-dev libxi-dev libxcursor-dev libgl1-mesa-dev libegl1-mesa-dev mesa-vulkan-drivers python3 python3-venv python3-pip"
DNF_PKGS="gcc gcc-c++ make pkgconf-pkg-config curl libX11-devel libxkbcommon-devel libxkbcommon-x11 wayland-devel libXrandr-devel libXi-devel libXcursor-devel mesa-libGL-devel mesa-libEGL-devel mesa-vulkan-drivers python3 python3-pip"
PACMAN_PKGS="base-devel pkgconf curl libx11 libxkbcommon libxkbcommon-x11 wayland libxrandr libxi libxcursor mesa vulkan-icd-loader python python-pip"

detect_pm() {
  if command -v apt-get >/dev/null 2>&1; then echo apt
  elif command -v dnf >/dev/null 2>&1; then echo dnf
  elif command -v pacman >/dev/null 2>&1; then echo pacman
  else echo unknown
  fi
}

PM="$(detect_pm)"
echo "==> Detected package manager: $PM"

case "$PM" in
  apt)
    echo "==> Installing system packages via apt..."
    $SUDO apt-get update
    # shellcheck disable=SC2086
    $SUDO apt-get install -y $APT_PKGS
    ;;
  dnf)
    echo "==> Installing system packages via dnf..."
    # shellcheck disable=SC2086
    $SUDO dnf install -y $DNF_PKGS
    ;;
  pacman)
    echo "==> Installing system packages via pacman..."
    # shellcheck disable=SC2086
    $SUDO pacman -Sy --needed --noconfirm $PACMAN_PKGS
    ;;
  *)
    cat <<EOF
Could not detect apt/dnf/pacman on this system.

Please install the equivalent of the following manually, then re-run this
script — it will skip straight to the Rust/Python/build steps once these
are present:

  Debian/Ubuntu (apt):    $APT_PKGS
  Fedora/RHEL   (dnf):    $DNF_PKGS
  Arch/Manjaro  (pacman): $PACMAN_PKGS
EOF
    exit 1
    ;;
esac
echo

# ── 2. Rust toolchain (edition 2024 needs rustc >= 1.85) ────────────────────

rustc_new_enough() {
  command -v rustc >/dev/null 2>&1 || return 1
  local ver major minor
  ver="$(rustc --version | awk '{print $2}')"
  major="${ver%%.*}"
  minor="${ver#*.}"; minor="${minor%%.*}"
  [ "$major" -gt "$MIN_RUSTC_MAJOR" ] && return 0
  [ "$major" -eq "$MIN_RUSTC_MAJOR" ] && [ "$minor" -ge "$MIN_RUSTC_MINOR" ]
}

if rustc_new_enough; then
  echo "==> Rust toolchain OK: $(rustc --version)"
else
  echo "==> Installing/updating Rust via rustup (distro-packaged Rust is"
  echo "    frequently too old for this project's edition 2024 / rustc >= ${MIN_RUSTC_MAJOR}.${MIN_RUSTC_MINOR})..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
  # shellcheck disable=SC1091
  source "$HOME/.cargo/env"
  if ! rustc_new_enough; then
    echo "ERROR: rustc is still older than ${MIN_RUSTC_MAJOR}.${MIN_RUSTC_MINOR} after installing rustup." >&2
    exit 1
  fi
  echo "==> Rust toolchain OK: $(rustc --version)"
fi
echo

# ── 3. Python virtualenv ─────────────────────────────────────────────────────

if [ ! -d .venv ]; then
  echo "==> Creating .venv/"
  python3 -m venv .venv
else
  echo "==> Reusing existing .venv/"
fi
PIP=".venv/bin/pip"
"$PIP" install --upgrade pip --quiet

# ── 4. torch: CPU-only or CUDA wheel, based on detected GPU ─────────────────
# No GPU is required anywhere in this project (see INSTALL.md); this just
# avoids pulling torch's multi-GB CUDA runtime bundle on a machine that has
# no NVIDIA GPU to use it with.

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  echo "==> NVIDIA GPU detected — installing the standard (CUDA-capable) torch wheel"
  "$PIP" install torch
else
  echo "==> No NVIDIA GPU detected — installing the CPU-only torch wheel (smaller download)"
  "$PIP" install torch --index-url https://download.pytorch.org/whl/cpu
fi

echo "==> Installing the rest of the Python dependencies"
"$PIP" install -r requirements.txt
echo

# ── 5. Build ──────────────────────────────────────────────────────────────────

echo "==> Building NNfractals (release, all GUIs)..."
cargo build --release --features "viewer browser launcher"
echo

# ── 6. Summary ──────────────────────────────────────────────────────────────

cat <<EOF
==> Done!

Binaries:
  target/release/nnfractals            (evolution engine, CLI)
  target/release/nnfractals-viewer     (deep-zoom render/save GUI)
  target/release/nnfractals-browser    (gallery + rating GUI)
  target/release/nnfractals-launcher   (front door: start/stop/train/dedup)

Run:
  ./run.sh                              # start evolution (background instances)
  ./target/release/nnfractals-launcher  # GUI front door

Optional desktop integration:
  ./target/release/nnfractals-launcher --install-desktop
  bash dist/install-viewer.sh           # .nn file association (double-click to open)

See INSTALL.md (English) or INSTALL.fr.md (Français) for details,
troubleshooting, and the full CPU-vs-GPU story.
EOF
