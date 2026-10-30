"""Count and inspect tokens with a Hugging Face ``tokenizer.json``. Author: Sandeep Belgavi."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer

from .fetch import DEFAULT_DIR, bytes_to_unicode

_U2B = {u: b for b, u in bytes_to_unicode().items()}


@dataclass(frozen=True)
class Piece:
    id: int
    data: bytes

    def show(self) -> str:
        """Readable text if the piece is whole UTF-8, otherwise the raw bytes in hex."""
        try:
            return self.data.decode("utf-8")
        except UnicodeDecodeError:
            return "<" + " ".join(f"0x{b:02X}" for b in self.data) + ">"


def load(path: str | Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def load_named(name: str, cache_dir: Path = DEFAULT_DIR) -> Tokenizer:
    """Load ``<cache_dir>/<name>.json`` written by ``fetch``; say how to get it if missing."""
    path = Path(cache_dir) / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m belgavi_llm.l01_tokens.fetch"
        )
    return load(path)


def encode(tk: Tokenizer, text: str) -> list[int]:
    return tk.encode(text, add_special_tokens=False).ids


def count_tokens(tk: Tokenizer, text: str) -> int:
    return len(encode(tk, text))


def pieces(tk: Tokenizer, text: str) -> list[Piece]:
    """The tokens of ``text`` with the exact bytes each one stands for."""
    out = []
    for tid in encode(tk, text):
        token = tk.id_to_token(tid)
        out.append(Piece(tid, bytes(_U2B[ch] for ch in token)))
    return out


def utf8_bytes(text: str) -> int:
    return len(text.encode("utf-8"))
