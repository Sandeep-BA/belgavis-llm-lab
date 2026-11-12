"""Synthetic help-desk cases (CASE-xxxx) for Belgavi's Book Store, a fictional online bookstore.
No real data. Author: Sandeep Belgavi.

Customers open several cases each and mostly about one recurring problem, with their own customer
handle, device and error code in the text. That is how real help-desk data behaves, and it is
exactly what makes a split by row leak. A fixed seed gives the same file every run.
"""

from __future__ import annotations

import random

CATEGORIES = ("login", "billing", "shipping", "bug", "feature", "refund")

_VOCAB = {
    "login": "password reset locked account sign verification code email library session expired",
    "billing": "receipt charged card payment gift-card membership pre-order tax overcharged "
    "renewal",
    "shipping": "order paperback delivery tracking courier address delayed hardcover arrived "
    "parcel",
    "bug": "crash error e-book download freezes loading bookmarks sync highlights blank",
    "feature": "request wish option dark-mode audiobook wishlist would font reading-list setting",
    "refund": "refund return money back cancel charged damaged copy wrong edition",
}
_VOCAB = {k: v.split() for k, v in _VOCAB.items()}
_FILLER = (
    "please help the my is not on for we our still after since today again can you with"
).split()
# Geetha, Jenya and Tanishka are the regulars; Dinky and Hansini write in less often.
_CUSTOMERS = (
    "Geetha Jenya Tanishka Geetha Jenya Tanishka Geetha Jenya Tanishka Dinky Hansini"
).split()
_DEVICES = "e-reader phone tablet laptop desktop browser watch".split()
_SYSTEM = (
    "You are the help-desk assistant for Belgavi's Book Store. Answer briefly and politely."
)
_REPLY = {
    "login": "Please reset your password from the sign-in page; if the code does not arrive, "
    "contact us again.",
    "billing": "We have flagged the charge for review; the correction shows on your next receipt.",
    "shipping": "We have asked the courier for a fresh tracking update on your books and will "
    "email the result.",
    "bug": "Thanks for the report. We have logged it with your steps and will update this case.",
    "feature": "Thanks for the idea. We have added it to the reading-app wish list.",
    "refund": "We have started the refund for your order; it can take a few days to show on your "
    "statement.",
}


def _case_text(rng: random.Random, cat: str, customer: str, device: str, code: str) -> str:
    words = rng.sample(_VOCAB[cat], 2)
    for _ in range(4):  # words borrowed from any category: real cases never separate cleanly
        words.append(rng.choice(_VOCAB[rng.choice(CATEGORIES)]))
    words += rng.sample(_FILLER, 5)
    rng.shuffle(words)
    return f"{customer} {device} {code}: " + " ".join(words)


def make_row(sup: int, source: str, cat: str, text: str) -> dict:
    return {
        "id": f"CASE-{sup}",
        "source": source,
        "label": cat,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": text},
            {"role": "assistant", "content": _REPLY[cat]},
        ],
    }


def generate(n: int = 2000, seed: int = 7, dirty: bool = True) -> list[dict]:
    """About ``n`` rows. With ``dirty`` set, ~8% of the rows are exact duplicates, near duplicates
    or malformed, so the cleaning and dedup steps have something to find."""
    rng = random.Random(seed)
    rows: list[dict] = []
    sup, cust = 1000, 0
    while len(rows) < n:
        cust += 1
        source = f"cust-{cust:03d}"
        main = rng.choice(CATEGORIES)
        customer = f"{rng.choice(_CUSTOMERS)}-{cust}"
        device = rng.choice(_DEVICES)
        code = f"E{rng.randint(100, 999)}"
        for _ in range(rng.randint(5, 12)):
            if len(rows) >= n:
                break
            cat = main if rng.random() < 0.7 else rng.choice(CATEGORIES)
            sup += 1
            text = _case_text(rng, cat, customer, device, code)
            rows.append(make_row(sup, source, cat, text))
    if dirty:
        rows = _make_dirty(rows, rng, sup)
    return rows


def _make_dirty(rows: list[dict], rng: random.Random, sup: int) -> list[dict]:
    out = list(rows)
    k = max(1, len(rows) // 50)  # 2% of each kind
    for r in rng.sample(rows, k):  # exact duplicate (different id)
        sup += 1
        d = {**r, "id": f"CASE-{sup}", "messages": [dict(m) for m in r["messages"]]}
        out.insert(rng.randrange(len(out)), d)
    for r in rng.sample(rows, k):  # near duplicate: whitespace/case noise plus one extra word
        sup += 1
        d = {**r, "id": f"CASE-{sup}", "messages": [dict(m) for m in r["messages"]]}
        d["messages"][1]["content"] = "  " + d["messages"][1]["content"].upper() + " thanks "
        out.insert(rng.randrange(len(out)), d)
    for r in rng.sample(rows, k):  # empty user message
        sup += 1
        d = {**r, "id": f"CASE-{sup}", "messages": [dict(m) for m in r["messages"]]}
        d["messages"][1]["content"] = "   "
        out.insert(rng.randrange(len(out)), d)
    for r in rng.sample(rows, k):  # unknown role
        sup += 1
        d = {**r, "id": f"CASE-{sup}", "messages": [dict(m) for m in r["messages"]]}
        d["messages"][1]["role"] = "customer"
        out.insert(rng.randrange(len(out)), d)
    return out
