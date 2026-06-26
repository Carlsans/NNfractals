#!/usr/bin/env bash
# Install nnfractals-viewer and register the .nn file association on Arch Linux.
# Run from the project root: bash dist/install-viewer.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "==> Building nnfractals-viewer (release)..."
cd "$PROJECT_DIR"
cargo build --release --features wgpu-backend,viewer --bin nnfractals-viewer

BIN="$PROJECT_DIR/target/release/nnfractals-viewer"

# ── Choose install prefix ──────────────────────────────────────────────────────
if [[ $EUID -eq 0 ]]; then
    PREFIX="/usr/local"
    MIME_DIR="/usr/share/mime/packages"
    DESKTOP_DIR="/usr/share/applications"
    UPDATE_AS_ROOT=true
else
    PREFIX="$HOME/.local"
    MIME_DIR="$HOME/.local/share/mime/packages"
    DESKTOP_DIR="$HOME/.local/share/applications"
    UPDATE_AS_ROOT=false
fi

mkdir -p "$PREFIX/bin" "$MIME_DIR" "$DESKTOP_DIR"

echo "==> Installing binary → $PREFIX/bin/nnfractals-viewer"
install -m 755 "$BIN" "$PREFIX/bin/nnfractals-viewer"

echo "==> Installing MIME type → $MIME_DIR/nnfractals.xml"
install -m 644 "$SCRIPT_DIR/nnfractals-mime.xml" "$MIME_DIR/nnfractals.xml"

echo "==> Installing .desktop file → $DESKTOP_DIR/nnfractals-viewer.desktop"
install -m 644 "$SCRIPT_DIR/nnfractals-viewer.desktop" "$DESKTOP_DIR/nnfractals-viewer.desktop"

echo "==> Updating MIME and desktop databases..."
update-mime-database "$MIME_DIR/.." 2>/dev/null || true
update-desktop-database "$DESKTOP_DIR" 2>/dev/null || true

# xdg-mime sets the default handler for the MIME type.
xdg-mime default nnfractals-viewer.desktop application/x-nnfractals 2>/dev/null || true

echo ""
echo "Done. You can now double-click any .nn file to open it in NNFractals Viewer."
echo "If your file manager doesn't pick it up immediately, log out and back in,"
echo "or run:  xdg-mime default nnfractals-viewer.desktop application/x-nnfractals"
