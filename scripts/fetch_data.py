"""Download the corpus PDFs listed in data/manifest.json and verify their sha256."""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    manifest = json.loads((DATA / "manifest.json").read_text())
    bad = 0
    for d in manifest:
        out = DATA / d["file"]
        if out.exists() and sha256(out) == d["sha256"]:
            print(f"ok      {d['file']}")
            continue
        req = urllib.request.Request(d["url"], headers={"User-Agent": "Mozilla/5.0 cited-rag"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            out.write_bytes(resp.read())
        status = "fetched" if sha256(out) == d["sha256"] else "MISMATCH"
        bad += status == "MISMATCH"
        print(f"{status:7s} {d['file']}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
