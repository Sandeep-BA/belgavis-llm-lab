"""Chat-format instruction JSONL: read, write, validate. Author: Sandeep Belgavi.

One JSON object per line:
``{"id": "CASE-1001", "source": "cust-007", "label": "login", "messages": [...]}`` where each
message is ``{"role": "system" | "user" | "assistant", "content": "..."}``.
``source`` is the group a row came from (here: the customer). Splits must keep a source whole.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

ROLES = ("system", "user", "assistant")
MAX_CHARS = 2000


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{n}: invalid JSON ({e.msg})") from e
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def validate_row(row: dict) -> list[str]:
    """Return the problems with one row (empty list = valid)."""
    errs: list[str] = []
    for key in ("id", "source", "label"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            errs.append(f"missing {key}")
    msgs = row.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return errs + ["messages must be a non-empty list"]
    for i, m in enumerate(msgs):
        if not isinstance(m, dict) or m.get("role") not in ROLES:
            errs.append(f"message {i}: role must be one of {ROLES}")
        elif not isinstance(m.get("content"), str) or not m["content"].strip():
            errs.append(f"message {i}: empty content")
        elif len(m["content"]) > MAX_CHARS:
            errs.append(f"message {i}: longer than {MAX_CHARS} characters")
    roles = [m.get("role") for m in msgs if isinstance(m, dict)]
    if "user" not in roles:
        errs.append("no user message")
    if roles and roles[-1] != "assistant":
        errs.append("last message must be from the assistant")
    return errs


def validate(rows: Iterable[dict]) -> dict[str, list[str]]:
    """Map row id (or ``#index``) to its problems; valid rows are omitted. Flags duplicate ids."""
    problems: dict[str, list[str]] = {}
    seen: set[str] = set()
    for i, r in enumerate(rows):
        key = r.get("id") if isinstance(r.get("id"), str) and r.get("id") else f"#{i}"
        errs = validate_row(r)
        if key in seen:
            errs.append("duplicate id")
        seen.add(key)
        if errs:
            problems.setdefault(key, []).extend(errs)
    return problems


def user_text(row: dict) -> str:
    """The text a classifier or retriever sees: all user turns joined."""
    return " ".join(m["content"] for m in row["messages"] if m.get("role") == "user")
