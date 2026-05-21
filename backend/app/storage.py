from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

from .models import Document, Span


class SessionStore:
    """In-memory, single-process session store. Sessions live until process exit."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = root or Path(tempfile.gettempdir()) / "medrec_sessions"
        self.root.mkdir(parents=True, exist_ok=True)
        self._docs: Dict[str, List[Document]] = {}
        self._spans: Dict[str, List[Span]] = {}
        self._lock = Lock()

    def new_session(self) -> str:
        sid = uuid.uuid4().hex[:12]
        (self.root / sid).mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._docs[sid] = []
            self._spans[sid] = []
        return sid

    def pdf_path(self, sid: str, did: str) -> Path:
        return self.root / sid / f"{did}.pdf"

    def add(self, sid: str, doc: Document, spans: List[Span]) -> None:
        with self._lock:
            self._docs.setdefault(sid, []).append(doc)
            self._spans.setdefault(sid, []).extend(spans)

    def documents(self, sid: str) -> List[Document]:
        return list(self._docs.get(sid, []))

    def spans(self, sid: str) -> List[Span]:
        return list(self._spans.get(sid, []))


store = SessionStore()
