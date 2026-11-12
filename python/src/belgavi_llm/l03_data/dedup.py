"""Exact and near-duplicate removal (MinHash + LSH, standard library). Author: Sandeep Belgavi."""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict

from belgavi_llm.l03_data.jsonl import user_text

_P = (1 << 61) - 1  # Mersenne prime for the universal hash family


def norm_key(text: str) -> str:
    """Case- and whitespace-insensitive key: what 'the same text' means for exact dedup."""
    return " ".join(text.lower().split())


def shingles(text: str, n: int = 3) -> set[str]:
    w = norm_key(text).split()
    if len(w) <= n:
        return {" ".join(w)}
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def _h64(s: str) -> int:
    # hashlib, not hash(): Python's str hash is randomised per process, which would make runs differ
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")


class MinHasher:
    def __init__(self, num_perm: int = 64, seed: int = 1):
        rng = random.Random(seed)
        self.params = [(rng.randrange(1, _P), rng.randrange(0, _P)) for _ in range(num_perm)]

    def signature(self, sh: set[str]) -> tuple[int, ...]:
        hs = [_h64(s) for s in sh]
        return tuple(min((a * h + b) % _P for h in hs) for a, b in self.params)


def exact_dedup(rows: list[dict]) -> tuple[list[dict], int]:
    seen: set[str] = set()
    out = []
    for r in rows:
        k = norm_key(user_text(r))
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out, len(rows) - len(out)


def near_dedup(rows: list[dict], threshold: float = 0.7, num_perm: int = 64,
               bands: int = 16) -> tuple[list[dict], int]:
    """Keep the first of each near-duplicate cluster. LSH finds candidates cheaply; each candidate
    is confirmed with the exact Jaccard of its shingle sets, so the threshold means what it says."""
    if num_perm % bands:
        raise ValueError("num_perm must be a multiple of bands")
    rpb = num_perm // bands
    mh = MinHasher(num_perm)
    sh = [shingles(user_text(r)) for r in rows]
    sigs = [mh.signature(s) for s in sh]
    buckets: dict[tuple, list[int]] = defaultdict(list)
    drop: set[int] = set()
    for i, sig in enumerate(sigs):
        cands: set[int] = set()
        for b in range(bands):
            cands.update(buckets[(b, sig[b * rpb:(b + 1) * rpb])])
        if any(j not in drop and jaccard(sh[i], sh[j]) >= threshold for j in sorted(cands)):
            drop.add(i)
            continue
        for b in range(bands):
            buckets[(b, sig[b * rpb:(b + 1) * rpb])].append(i)
    return [r for i, r in enumerate(rows) if i not in drop], len(drop)
