"""A tiny byte-level BPE tokenizer for offline tests. Author: Sandeep Belgavi.

Trained in a second on a few synthetic English lines, so it knows common English pieces and falls
back to raw bytes for Hindi and Kannada. The saved file in ``testdata/l01/toy-tokenizer.json`` is
read by both the Python and the Java tests; regenerate it with

    python -m belgavi_llm.l01_tokens.toy testdata/l01
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from tokenizers import Tokenizer, decoders, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

CORPUS = [
    "My book order has not arrived. Please check the status.",
    "I cannot sign in after resetting my password.",
    "The team will reply within one working day.",
    "Thank you for your patience. We are looking into the problem.",
    "Is CASE-1042 fixed? Not yet. It is open and with the team.",
] * 20

SENTENCES = [
    "My book order has not arrived.",
    "Is CASE-1042 fixed?",
    "The team will reply within one working day.",
    "ಧನ್ಯವಾದಗಳು",
    "धन्यवाद",
    "",
]


def train() -> Tokenizer:
    tk = Tokenizer(BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    trainer = BpeTrainer(
        vocab_size=400, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False
    )
    tk.train_from_iterator(CORPUS, trainer)
    return tk


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "testdata/l01")
    out.mkdir(parents=True, exist_ok=True)
    tk = train()
    tk.save(str(out / "toy-tokenizer.json"))
    expected = {s: tk.encode(s, add_special_tokens=False).ids for s in SENTENCES}
    (out / "toy-expected.json").write_text(
        json.dumps(expected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
