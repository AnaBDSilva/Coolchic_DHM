#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/Orange-OpenSource/Cool-Chic.git"
COOL_CHIC_DIR="$PWD/Cool-Chic"

# ── 1. Clone or update ────────────────────────────────────────────────────────
if [ ! -d "$COOL_CHIC_DIR" ]; then
  echo "[1/3] Cloning Cool-Chic..."
  git clone "$REPO_URL" "$COOL_CHIC_DIR"
else
  echo "[1/3] Updating Cool-Chic..."
  git -C "$COOL_CHIC_DIR" pull --ff-only
fi

# ── 2. Create local venv + install pip deps ───────────────────────────────────
echo "[2/3] Setting up virtualenv and pip deps..."

VENV_DIR="$PWD/.venv-coolchic"

if [ ! -d "$VENV_DIR" ]; then
  python -m venv "$VENV_DIR" --system-site-packages
fi

source "$VENV_DIR/bin/activate"

pip install --quiet \
  fvcore \
  einops \
  configargparse \
  constriction==0.4.2

# ── 3. Done ───────────────────────────────────────────────────────────────────
echo "[3/3] Done."
echo ""
echo "Use:  cc-encode / cc-decode"