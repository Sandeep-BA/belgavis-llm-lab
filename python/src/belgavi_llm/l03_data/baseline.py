"""A tiny multinomial naive Bayes case classifier, standard library only. Author: Sandeep Belgavi.

It is deliberately simple: the point is the *split*, not the model. Any model scores higher on a
leaky split than on a clean one.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from belgavi_llm.l03_data.jsonl import user_text


def tokens(text: str) -> list[str]:
    return [t.strip(":,.").lower() for t in text.split() if t.strip(":,.")]


class NaiveBayes:
    def fit(self, rows: list[dict]) -> NaiveBayes:
        self.prior = Counter(r["label"] for r in rows)
        self.counts: dict[str, Counter] = defaultdict(Counter)
        for r in rows:
            self.counts[r["label"]].update(tokens(user_text(r)))
        self.vocab = {w for c in self.counts.values() for w in c}
        self.total = {k: sum(c.values()) for k, c in self.counts.items()}
        self.n = sum(self.prior.values())
        return self

    def predict(self, row: dict) -> str:
        toks = tokens(user_text(row))
        best, best_lp = "", -math.inf
        for label in sorted(self.prior):
            lp = math.log(self.prior[label] / self.n)
            for w in toks:
                num = self.counts[label][w] + 1
                lp += math.log(num / (self.total[label] + len(self.vocab) + 1))
            if lp > best_lp:
                best, best_lp = label, lp
        return best

    def accuracy(self, rows: list[dict]) -> float:
        return sum(self.predict(r) == r["label"] for r in rows) / max(1, len(rows))
