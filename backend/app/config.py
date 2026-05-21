import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
GUIDELINES_PATH = ROOT / "backend" / "guidelines" / "acc_aha.json"

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
EXTRACTION_MODEL = os.environ.get("EXTRACTION_MODEL", "claude-opus-4-7")
SUMMARY_MODEL = os.environ.get("SUMMARY_MODEL", "claude-opus-4-7")

DATA_DIR.mkdir(parents=True, exist_ok=True)
