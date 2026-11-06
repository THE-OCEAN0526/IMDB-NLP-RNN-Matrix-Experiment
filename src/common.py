"""Variant-specific config, source checksums and preprocessing; no imports from v1.0."""
from __future__ import annotations
import hashlib
import html
import re
from pathlib import Path
import yaml
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(2 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def config() -> tuple[dict, str]:
    path = ROOT / "config/experiment.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")), file_hash(path)

def clean_review(text: str) -> str:
    """Exact approved v1.0 phase0-text-v1 cleaning, including punctuation preservation."""
    text = html.unescape(str(text))
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()

def review_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def ids_hash(ids) -> str:
    return review_hash("\n".join(ids) + "\n")

def load_splits():
    return {n:pd.read_parquet(ROOT / f"data/processed/{n}.parquet") for n in ("train","validation","test")}
