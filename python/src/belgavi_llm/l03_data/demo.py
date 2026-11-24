"""End-to-end CPU demo: generate, clean, dedup, split, measure leakage. Author: Sandeep Belgavi.

    python -m belgavi_llm.l03_data.demo [--out data/cache/l03] [--rows 2000] [--seeds 10]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from belgavi_llm.l03_data import baseline, clean, dedup, jsonl, leakage, split, synth


def run(rows_n: int = 2000, seeds: int = 10, out: str | None = None) -> dict:
    t0 = time.perf_counter()
    raw = synth.generate(rows_n)
    problems = jsonl.validate(raw)
    cleaned, dropped = clean.clean(raw)
    uniq, n_exact = dedup.exact_dedup(cleaned)
    uniq, n_near = dedup.near_dedup(uniq)
    report = {
        "raw_rows": len(raw), "invalid_rows": len(problems), "dropped_by_clean": dropped["invalid"],
        "exact_duplicates_removed": n_exact, "near_duplicates_removed": n_near,
        "clean_rows": len(uniq),
        "sources": len({r["source"] for r in uniq}),
    }
    leaky_acc, clean_acc = [], []
    for s in range(1, seeds + 1):
        for fn, acc in ((split.split_by_row, leaky_acc), (split.split_by_group, clean_acc)):
            tr, _, te = fn(uniq, seed=s)
            acc.append(baseline.NaiveBayes().fit(tr).accuracy(te))
    tr, va, te = split.split_by_row(uniq, seed=1)
    report["row_split_overlap_test"] = leakage.overlap(tr, te)
    tr, va, te = split.split_by_group(uniq, seed=1)
    report["group_split_overlap_test"] = leakage.overlap(tr, te)
    report["accuracy_row_split"] = {"mean": round(statistics.mean(leaky_acc), 4),
                                    "stdev": round(statistics.pstdev(leaky_acc), 4)}
    report["accuracy_group_split"] = {"mean": round(statistics.mean(clean_acc), 4),
                                      "stdev": round(statistics.pstdev(clean_acc), 4)}
    report["seeds"] = seeds
    report["seconds"] = round(time.perf_counter() - t0, 1)
    if out:
        d = Path(out)
        jsonl.write_jsonl(d / "raw.jsonl", raw)
        for name, part in zip(("train", "validation", "test"), (tr, va, te), strict=True):
            jsonl.write_jsonl(d / f"{name}.jsonl", part)
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/cache/l03")
    p.add_argument("--rows", type=int, default=2000)
    p.add_argument("--seeds", type=int, default=10)
    a = p.parse_args()
    print(json.dumps(run(a.rows, a.seeds, a.out), indent=2))


if __name__ == "__main__":
    main()
