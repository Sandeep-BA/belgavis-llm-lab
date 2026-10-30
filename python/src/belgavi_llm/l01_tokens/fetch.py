"""Download real tokenizer vocabularies and convert them to Hugging Face ``tokenizer.json``.

Author: Sandeep Belgavi.

The vocabularies (``r50k_base``, ``cl100k_base``, ``o200k_base``) are the public BPE tables behind
GPT-2, GPT-4 and GPT-4o. They ship inside the MIT-licensed ``jtokkit`` jar on Maven Central, which
we download once, check against a pinned SHA-256, and convert into the format both ``tokenizers``
(Python) and DJL (Java) load. Nothing here is committed to git: the cache dir is git-ignored.

    python -m belgavi_llm.l01_tokens.fetch            # -> data/cache/l01/tokenizers/*.json
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, pre_tokenizers
from tokenizers.models import BPE

JTOKKIT_URL = "https://repo1.maven.org/maven2/com/knuddels/jtokkit/1.1.0/jtokkit-1.1.0.jar"
JTOKKIT_SHA256 = "1501ce0259ab897c6746ccfafa1d208acd404fb17e1ac62e157172f2678b1183"

# Pinned hashes of the extracted tables; they equal the hashes tiktoken itself checks.
SPECS: dict[str, dict[str, str]] = {
    "r50k_base": {
        "sha256": "306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930",
        "pat": r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
    },
    "cl100k_base": {
        "sha256": "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
        "pat": r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"""
        r"""| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""",
    },
    "o200k_base": {
        "sha256": "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d",
        "pat": "|".join(
            [
                r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"""
                r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
                r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"""
                r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
                r"""\p{N}{1,3}""",
                r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
                r"""\s*[\r\n]+""",
                r"""\s+(?!\S)""",
                r"""\s+""",
            ]
        ),
    },
}

DEFAULT_DIR = Path("data/cache/l01/tokenizers")


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's reversible byte -> printable-character map (what byte-level BPE vocabularies use)."""
    keep = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    chars = keep[:]
    extra = 0
    for b in range(256):
        if b not in keep:
            keep.append(b)
            chars.append(256 + extra)
            extra += 1
    return dict(zip(keep, map(chr, chars), strict=True))


def parse_tiktoken(data: bytes) -> dict[bytes, int]:
    ranks: dict[bytes, int] = {}
    for line in data.splitlines():
        if line:
            token, rank = line.split()
            ranks[base64.b64decode(token)] = int(rank)
    return ranks


def _split_token(ranks: dict[bytes, int], token: bytes, rank: int) -> tuple[bytes, bytes]:
    """Replay BPE on ``token`` using only merges ranked below ``rank``; two parts must remain."""
    parts = [bytes([b]) for b in token]
    while len(parts) > 2:
        best = None
        for i in range(len(parts) - 1):
            r = ranks.get(parts[i] + parts[i + 1])
            if r is not None and r < rank and (best is None or r < best[0]):
                best = (r, i)
        if best is None:
            break
        i = best[1]
        parts[i : i + 2] = [parts[i] + parts[i + 1]]
    if len(parts) != 2:
        raise ValueError(f"cannot split {token!r} into two merges")
    return parts[0], parts[1]


def build_tokenizer(ranks: dict[bytes, int], pat: str) -> Tokenizer:
    """Byte-level BPE ``Tokenizer`` equivalent to a tiktoken table."""
    b2u = bytes_to_unicode()

    def enc(b: bytes) -> str:
        return "".join(b2u[x] for x in b)

    vocab = {enc(tok): r for tok, r in ranks.items()}
    merges = []
    for tok, r in sorted(ranks.items(), key=lambda kv: kv[1]):
        if len(tok) > 1:
            left, right = _split_token(ranks, tok, r)
            merges.append((enc(left), enc(right)))
    tk = Tokenizer(BPE(vocab=vocab, merges=merges))
    tk.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(pat), behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tk.decoder = decoders.ByteLevel()
    return tk


def _download(url: str, attempts: int = 5) -> bytes:
    """GET with a User-Agent and exponential backoff (Maven Central answers 429 when busy)."""
    req = urllib.request.Request(url, headers={"User-Agent": "belgavis-llm-lab/0.1"})  # noqa: S310
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 - fixed https URL
                return resp.read()
        except urllib.error.HTTPError as err:
            if err.code not in (429, 503) or attempt == attempts - 1:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def _read_jar(url: str, sha256: str, keep: Path | None = None) -> zipfile.ZipFile:
    blob = _download(url)
    got = hashlib.sha256(blob).hexdigest()
    if got != sha256:
        raise RuntimeError(f"jtokkit jar hash mismatch: {got}")
    if keep is not None:
        keep.write_bytes(blob)
    return zipfile.ZipFile(io.BytesIO(blob))


def fetch_all(out_dir: Path = DEFAULT_DIR, keep_jar: bool = False) -> list[Path]:
    """Download, verify and convert all tokenizers; returns the written ``tokenizer.json`` paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    jar = _read_jar(JTOKKIT_URL, JTOKKIT_SHA256, out_dir / "jtokkit.jar" if keep_jar else None)
    written = []
    for name, spec in SPECS.items():
        data = jar.read(f"com/knuddels/jtokkit/{name}.tiktoken")
        got = hashlib.sha256(data).hexdigest()
        if got != spec["sha256"]:
            raise RuntimeError(f"{name}: table hash mismatch: {got}")
        tk = build_tokenizer(parse_tiktoken(data), spec["pat"])
        path = out_dir / f"{name}.json"
        tk.save(str(path))
        written.append(path)
        print(f"wrote {path} ({len(tk.get_vocab())} tokens)")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--keep-jar", action="store_true", help="keep the jar for the parity test")
    args = ap.parse_args()
    fetch_all(args.out, args.keep_jar)


if __name__ == "__main__":
    main()
