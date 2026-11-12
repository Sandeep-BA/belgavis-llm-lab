"""Train/validation/test splits: by row (leaky) and by source (clean). Author: Sandeep Belgavi."""

from __future__ import annotations

import random


def split_by_row(rows: list[dict], seed: int = 1, val: float = 0.1, test: float = 0.2):
    """The tempting default. Rows from one source end up on both sides of the line."""
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    n_test, n_val = int(len(rows) * test), int(len(rows) * val)
    pick = lambda ids: [rows[i] for i in sorted(ids)]  # noqa: E731
    return pick(idx[n_test + n_val:]), pick(idx[n_test:n_test + n_val]), pick(idx[:n_test])


def split_by_group(rows: list[dict], seed: int = 1, val: float = 0.1, test: float = 0.2):
    """Assign whole sources to one split, so no source appears in two."""
    groups = sorted({r["source"] for r in rows})
    random.Random(seed).shuffle(groups)
    n_test, n_val = round(len(groups) * test), round(len(groups) * val)
    test_g = set(groups[:n_test])
    val_g = set(groups[n_test:n_test + n_val])
    train, valid, tst = [], [], []
    for r in rows:
        (tst if r["source"] in test_g else valid if r["source"] in val_g else train).append(r)
    return train, valid, tst
