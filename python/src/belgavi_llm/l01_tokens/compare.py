"""Compare tokenizers and languages on the same sentences. Author: Sandeep Belgavi.

    python -m belgavi_llm.l01_tokens.fetch      # once
    python -m belgavi_llm.l01_tokens.compare
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer

from .counting import count_tokens, load, load_named, utf8_bytes
from .fetch import DEFAULT_DIR, SPECS
from .samples import LANGUAGES, paragraph


def measure(tokenizers: dict[str, Tokenizer]) -> dict[str, dict[str, int]]:
    """tokens[tokenizer_name][language] for the lesson's sample paragraph."""
    return {
        name: {lang: count_tokens(tk, paragraph(lang)) for lang in LANGUAGES}
        for name, tk in tokenizers.items()
    }


def markdown_table(table: dict[str, dict[str, int]]) -> str:
    """Tokens per tokenizer and language, plus each language's cost relative to English."""
    lines = ["| tokenizer | " + " | ".join(LANGUAGES) + " |", "|---|" + "---:|" * len(LANGUAGES)]
    for name, row in table.items():
        cells = []
        for lang in LANGUAGES:
            ratio = row[lang] / row["English"]
            cells.append(f"{row[lang]} ({ratio:.1f}x)")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    sizes = " | ".join(str(utf8_bytes(paragraph(lang))) for lang in LANGUAGES)
    lines.append(f"| *UTF-8 bytes of the text* | {sizes} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", type=Path, default=DEFAULT_DIR)
    ap.add_argument(
        "--extra", action="append", default=[], metavar="NAME=PATH",
        help="add any tokenizer.json to the table, e.g. qwen2.5=/path/to/tokenizer.json",
    )
    args = ap.parse_args()
    tks = {name: load_named(name, args.cache) for name in SPECS}
    for item in args.extra:
        name, _, path = item.partition("=")
        tks[name] = load(path)
    print(markdown_table(measure(tks)))


if __name__ == "__main__":
    main()
