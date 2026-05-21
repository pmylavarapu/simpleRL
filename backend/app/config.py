from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    llm_backend: str  # "claude" | "ollama" | "none"
    claude_model: str
    ollama_url: str
    ollama_model: str
    guidelines_dir: Path
    guideline_score_threshold: float  # below this, no plan
    demo_synthetic_banner: bool


def load_config() -> Config:
    return Config(
        llm_backend=os.getenv("LLM_BACKEND", "claude").lower(),
        claude_model=os.getenv("CLAUDE_MODEL", "claude-opus-4-7"),
        ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct-q4_K_M"),
        guidelines_dir=Path(
            os.getenv("GUIDELINES_DIR", str(REPO_ROOT / "guidelines_corpus"))
        ),
        guideline_score_threshold=float(os.getenv("GUIDELINE_SCORE_THRESHOLD", "0.15")),
        demo_synthetic_banner=os.getenv("DEMO_SYNTHETIC", "1") not in ("0", "false", "False"),
    )


config = load_config()
