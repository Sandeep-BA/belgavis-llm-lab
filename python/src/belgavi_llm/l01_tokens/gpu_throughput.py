"""How fast does a model read and write tokens? GPU step of lesson 1. Author: Sandeep Belgavi.

Tokens are the unit you pay in; this measures the unit you *wait* in. It loads one small model,
times prefill (reading a prompt) and decode (writing tokens) and prints tokens per second.
Without a CUDA GPU it prints a notice and exits 0, so it never breaks a CPU run.

    pip install torch "transformers==5.20.2"        # transformers pin: see docs/01-tokens.md
    python -m belgavi_llm.l01_tokens.gpu_throughput --model Qwen/Qwen2.5-0.5B-Instruct

Measured on an RTX 4060: see 'More measurements' in docs/01-tokens.md.
"""

from __future__ import annotations

import argparse
import sys
import time

from .samples import LANGUAGES, paragraph


def cuda_ready() -> tuple[bool, str]:
    try:
        import torch
    except ImportError:
        return False, "torch is not installed"
    if not torch.cuda.is_available():
        return False, "no CUDA GPU found"
    return True, torch.cuda.get_device_name(0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--new-tokens", type=int, default=128)
    args = ap.parse_args(argv)

    ok, detail = cuda_ready()
    if not ok:
        print(f"GPU step skipped: {detail}. Use the CPU path or the notebook.")
        return 0

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to("cuda")
    model.eval()
    print(f"device: {detail}   model: {args.model}   torch: {torch.__version__}")
    print("| language | prompt tokens | prefill ms | decode tokens/s |")
    print("|---|---:|---:|---:|")
    for lang in LANGUAGES:
        ids = tok(paragraph(lang), return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            model.generate(ids, max_new_tokens=4, do_sample=False)  # warm-up
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(ids)
            torch.cuda.synchronize()
            prefill_ms = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            n = args.new_tokens
            out = model.generate(ids, max_new_tokens=n, min_new_tokens=n, do_sample=False)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - t0
        new = out.shape[1] - ids.shape[1]
        print(f"| {lang} | {ids.shape[1]} | {prefill_ms:.1f} | {new / seconds:.1f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
