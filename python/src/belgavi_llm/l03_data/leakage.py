"""Leakage report: what do two splits share? Author: Sandeep Belgavi."""

from __future__ import annotations

from belgavi_llm.l03_data.dedup import jaccard, norm_key, shingles
from belgavi_llm.l03_data.jsonl import user_text


def overlap(train: list[dict], other: list[dict], near_threshold: float = 0.5) -> dict:
    """Share of ``other`` rows that leak from ``train``: same source, same text, or a near copy."""
    t_src = {r["source"] for r in train}
    t_txt = {norm_key(user_text(r)) for r in train}
    t_sh = [shingles(user_text(r)) for r in train]
    shared_src = {r["source"] for r in other} & t_src
    exact = near = 0
    for r in other:
        if norm_key(user_text(r)) in t_txt:
            exact += 1
            continue
        s = shingles(user_text(r))
        if any(jaccard(s, t) >= near_threshold for t in t_sh):
            near += 1
    n = max(1, len(other))
    return {
        "rows": len(other),
        "shared_sources": len(shared_src),
        "rows_from_shared_sources": sum(r["source"] in t_src for r in other),
        "exact_text_overlap": exact,
        "near_text_overlap": near,
        "leaky_fraction": round((sum(r["source"] in t_src for r in other)) / n, 3),
    }
