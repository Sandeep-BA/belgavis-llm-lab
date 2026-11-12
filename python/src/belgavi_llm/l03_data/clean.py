"""Cleaning: normalise text, drop rows that cannot be trained on. Author: Sandeep Belgavi."""

from __future__ import annotations

import re
import unicodedata

from belgavi_llm.l03_data.jsonl import validate_row

_WS = re.compile(r"\s+")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def normalize_text(s: str) -> str:
    """NFKC, collapse whitespace, mask e-mail addresses. Case is kept (the model may need it)."""
    s = unicodedata.normalize("NFKC", s)
    s = _EMAIL.sub("<email>", s)
    return _WS.sub(" ", s).strip()


def _clean_message(m):
    if isinstance(m, dict) and isinstance(m.get("content"), str):
        return {**m, "content": normalize_text(m["content"])}
    return m


def clean(rows: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """Return (kept rows, drop counts by reason). Does not mutate the input."""
    kept: list[dict] = []
    dropped = {"invalid": 0}
    for r in rows:
        r2 = r
        if isinstance(r.get("messages"), list):
            r2 = {**r, "messages": [_clean_message(m) for m in r["messages"]]}
        if validate_row(r2):
            dropped["invalid"] += 1
            continue
        kept.append(r2)
    return kept, dropped
