from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import config
from .routers.ingest import router as ingest_router
from .routers.summary import router as summary_router


app = FastAPI(title="Medical Records Summarizer")

# The frontend dev server runs on :5173 and proxies /api to :8000, but we also
# accept direct cross-origin requests during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest_router)
app.include_router(summary_router)


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "llm_backend": config.llm_backend,
        "demo_synthetic_banner": config.demo_synthetic_banner,
    }
