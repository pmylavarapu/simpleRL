"""Guideline corpus storage and TF-IDF retrieval.

The corpus is a folder of PDFs. `build_index()` extracts text-layer chunks per page
and persists a JSON index alongside the PDFs. `Retriever.query()` returns top-k
chunks above a cosine threshold; below threshold, the caller treats the problem as
"no guideline-backed recommendation".

TF-IDF (sklearn) was chosen over embeddings to avoid an embeddings model or API
dependency. Recall is fine for this corpus shape — short, jargon-heavy guideline
prose — and the whole pipeline runs in-process.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


INDEX_FILENAME = "index.json"
CHUNK_TARGET_CHARS = 1200
CHUNK_MIN_CHARS = 200


@dataclass
class GuidelineChunk:
    chunk_id: str
    document: str
    page: int
    text: str


def _split_into_chunks(page_text: str, page_num: int, doc_name: str) -> List[GuidelineChunk]:
    """Roughly paragraph-aware chunking around CHUNK_TARGET_CHARS."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", page_text) if p.strip()]
    chunks: List[GuidelineChunk] = []
    buf: list[str] = []
    buf_len = 0
    for p in paragraphs:
        if buf_len + len(p) > CHUNK_TARGET_CHARS and buf_len >= CHUNK_MIN_CHARS:
            chunks.append(
                GuidelineChunk(
                    chunk_id=uuid.uuid4().hex[:12],
                    document=doc_name,
                    page=page_num,
                    text="\n\n".join(buf),
                )
            )
            buf, buf_len = [], 0
        buf.append(p)
        buf_len += len(p) + 2
    if buf_len >= CHUNK_MIN_CHARS:
        chunks.append(
            GuidelineChunk(
                chunk_id=uuid.uuid4().hex[:12],
                document=doc_name,
                page=page_num,
                text="\n\n".join(buf),
            )
        )
    return chunks


def build_index(corpus_dir: Path) -> List[GuidelineChunk]:
    """Walk every PDF in `corpus_dir`, extract per-page text, chunk, and persist."""
    import pdfplumber

    if not corpus_dir.exists():
        raise FileNotFoundError(f"guidelines directory missing: {corpus_dir}")

    chunks: List[GuidelineChunk] = []
    for pdf_path in sorted(corpus_dir.glob("*.pdf")):
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                for i, page in enumerate(pdf.pages):
                    txt = page.extract_text() or ""
                    if len(txt.strip()) < CHUNK_MIN_CHARS:
                        continue
                    chunks.extend(_split_into_chunks(txt, i + 1, pdf_path.name))
        except Exception as e:  # noqa: BLE001
            print(f"[guidelines] skipping {pdf_path.name}: {e}")

    index_path = corpus_dir / INDEX_FILENAME
    index_path.write_text(
        json.dumps([c.__dict__ for c in chunks], indent=2),
        encoding="utf-8",
    )
    return chunks


def load_index(corpus_dir: Path) -> Optional[List[GuidelineChunk]]:
    index_path = corpus_dir / INDEX_FILENAME
    if not index_path.exists():
        return None
    data = json.loads(index_path.read_text(encoding="utf-8"))
    return [GuidelineChunk(**c) for c in data]


class Retriever:
    """In-process TF-IDF over the loaded chunks. Built lazily on first query."""

    def __init__(self, chunks: List[GuidelineChunk]) -> None:
        if not chunks:
            raise ValueError("retriever needs at least one chunk")
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.chunks = chunks
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.9,
            stop_words="english",
        )
        self._matrix = self._vectorizer.fit_transform(c.text for c in chunks)

    def query(self, q: str, k: int = 3) -> list[tuple[GuidelineChunk, float]]:
        if not q.strip():
            return []
        from sklearn.metrics.pairwise import cosine_similarity

        qv = self._vectorizer.transform([q])
        sims = cosine_similarity(qv, self._matrix)[0]
        order = sims.argsort()[::-1][:k]
        return [(self.chunks[int(i)], float(sims[int(i)])) for i in order if sims[int(i)] > 0]
