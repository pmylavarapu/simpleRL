"""CLI: build the guideline TF-IDF index.

Usage:
  python -m app.guidelines build [--corpus PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..config import config
from .store import build_index, load_index


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="(Re)build the guideline index")
    build.add_argument("--corpus", default=str(config.guidelines_dir))
    info = sub.add_parser("info", help="Show counts from the current index")
    info.add_argument("--corpus", default=str(config.guidelines_dir))
    args = p.parse_args()

    corpus = Path(args.corpus)
    if args.cmd == "build":
        chunks = build_index(corpus)
        print(f"Indexed {len(chunks)} chunks from {corpus}")
        return 0
    if args.cmd == "info":
        chunks = load_index(corpus)
        if chunks is None:
            print("No index found.", file=sys.stderr)
            return 1
        docs: dict[str, int] = {}
        for c in chunks:
            docs[c.document] = docs.get(c.document, 0) + 1
        print(f"{len(chunks)} chunks across {len(docs)} documents:")
        for name, n in sorted(docs.items()):
            print(f"  {name}: {n}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
