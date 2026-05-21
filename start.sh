#!/usr/bin/env bash
# One-command bootstrap + run for the medical records summarizer.
#
# Does:
#   1. Verifies Python 3.11+ and Node 20+ are available
#   2. Creates .venv/ and installs backend deps (idempotent)
#   3. Installs frontend deps (idempotent)
#   4. Generates synthetic patient PDFs if missing
#   5. Loads ANTHROPIC_API_KEY from .env (or prompts for it)
#   6. Starts backend on :8000 and frontend on :3000 in parallel
#   7. Opens http://localhost:3000 in your default browser
#   8. Cleans up both processes on Ctrl+C

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------- tool checks ----------
need() { command -v "$1" >/dev/null 2>&1 || { echo "ERROR: missing $1. Please install it." >&2; exit 1; }; }
need python3
need node
need npm

PYV=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
PY_OK=$(python3 -c "import sys; print(int(sys.version_info >= (3, 11)))")
if [ "$PY_OK" != "1" ]; then
  echo "ERROR: Python 3.11+ required, found $PYV." >&2
  exit 1
fi

NV=$(node -v | sed 's/v//' | cut -d. -f1)
if [ "$NV" -lt 20 ]; then
  echo "ERROR: Node 20+ required, found v$NV." >&2
  exit 1
fi

# ---------- API key ----------
if [ -z "${ANTHROPIC_API_KEY:-}" ] && [ -f .env ]; then
  # shellcheck disable=SC1091
  set -a; source .env; set +a
fi
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  printf "Enter your ANTHROPIC_API_KEY (sk-ant-...): "
  read -rs ANTHROPIC_API_KEY
  echo
  export ANTHROPIC_API_KEY
  if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    umask 077
    printf "ANTHROPIC_API_KEY=%s\n" "$ANTHROPIC_API_KEY" > .env
    echo "[setup] saved key to .env (gitignored; delete anytime)"
  fi
fi
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "ERROR: ANTHROPIC_API_KEY is required." >&2
  exit 1
fi

# ---------- backend deps ----------
if [ ! -d .venv ]; then
  echo "[setup] creating Python venv at .venv/"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
echo "[setup] installing/refreshing backend deps"
pip install -q -U pip
pip install -q -r backend/requirements.txt

# ---------- sample PDFs ----------
if [ ! -f samples/outside_records/discharge_summary.pdf ]; then
  echo "[setup] generating synthetic patient PDFs"
  python3 samples/generate_synthetic_patient.py
fi

# ---------- frontend deps ----------
if [ ! -d frontend/node_modules ]; then
  echo "[setup] installing frontend deps (this can take a minute the first time)"
  (cd frontend && npm install --no-audit --no-fund)
fi

# ---------- port checks ----------
port_busy() { lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }
if port_busy 8000; then
  echo "ERROR: something is already listening on port 8000. Stop it and re-run." >&2
  exit 1
fi
if port_busy 3000; then
  echo "ERROR: something is already listening on port 3000. Stop it and re-run." >&2
  exit 1
fi

# ---------- run ----------
PIDS=()
cleanup() {
  echo
  echo "[shutdown] stopping servers..."
  for pid in "${PIDS[@]:-}"; do
    if [ -n "${pid:-}" ]; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

echo "[run] backend  -> http://localhost:8000  (docs at /docs)"
uvicorn app.main:app --reload --app-dir backend --port 8000 &
PIDS+=("$!")

echo "[run] frontend -> http://localhost:3000"
(cd frontend && npm run dev) &
PIDS+=("$!")

# Give the frontend a moment to start, then open the browser.
(
  sleep 4
  if command -v open >/dev/null 2>&1; then
    open http://localhost:3000 >/dev/null 2>&1 || true
  fi
) &

cat <<EOF

============================================================
 App is starting.
   Frontend:  http://localhost:3000
   Backend:   http://localhost:8000
   API docs:  http://localhost:8000/docs

 Sample PDFs to upload: samples/outside_records/*.pdf

 Ctrl+C to stop both servers.
============================================================

EOF

wait
