"""CPU walk-through of lesson 2. Author: Sandeep Belgavi.

    python -m belgavi_llm.l02_transformer.demo [--real [MODEL]] [--device cpu|cuda]

Prints each stage of the toy transformer, the temperature table, and the NumPy-vs-PyTorch check.
Without --real the PyTorch check uses a tiny random-weight GPT-2 built offline.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from .tiny import VOCAB, TinyLM, sample, softmax

np.set_printoptions(precision=4, suppress=True, linewidth=100)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", nargs="?", const="default",
                    help="check a pretrained GPT-2 style model")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    t0 = time.perf_counter()

    ids = [VOCAB.index(w) for w in ("the", "cat", "sat", "on", "the")]
    print("tokens:", [VOCAB[i] for i in ids], "->", ids)
    m = TinyLM()
    r = m.forward(ids)
    print("\n1. embeddings (one row per token):\n", r["x"])
    print("\n2. causal attention weights (row = who is looking, column = who it looks at):\n",
          r["weights"])
    print("\n3. after attention + residual:\n", r["h1"])
    print("\n4. MLP output:\n", r["mlp"])
    p = m.next_token_probs(ids)
    print("\n5. next-token distribution after 'the cat sat on the':")
    for w, q in zip(VOCAB, p, strict=True):
        print(f"   {w:>4}  {q:.4f}")
    print(f"   sum = {p.sum():.4f}")

    print("\nTemperature is one division (logits / T):")
    logits = r["logits"][-1]
    for t in (0.25, 0.5, 1.0, 2.0, 4.0):
        q = softmax(logits, t)
        print(f"   T={t:<4} top={VOCAB[int(q.argmax())]} p_top={q.max():.4f} "
              f"sample(u=0.5)={VOCAB[sample(q, 0.5)]}")

    from .gpt2_check import DEFAULT_REAL_MODEL, load_model, max_abs_diff, random_tiny_model

    if args.real:
        name = DEFAULT_REAL_MODEL if args.real == "default" else args.real
        model, label = load_model(name), name
    else:
        model, label = random_tiny_model(), "tiny random-weight GPT-2 (offline)"
    check_ids = [3, 1, 4, 1, 5, 9, 2, 6]
    d = max_abs_diff(model, check_ids, args.device)
    print(f"\nNumPy vs PyTorch, layer 0 of {label} on {args.device}: max |diff| = {d:.2e}")
    print(f"elapsed {time.perf_counter() - t0:.2f} s")


if __name__ == "__main__":
    main()
