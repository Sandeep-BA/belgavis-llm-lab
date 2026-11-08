"""NumPy attention vs PyTorch on a GPT-2 style first layer. Author: Sandeep Belgavi.

``numpy_block`` re-implements layer 0 of a GPT-2 style model with the functions from ``tiny``
(multi-head by looping the single head). ``torch_block`` asks transformers/PyTorch for the same
layer. Two entry points:

* ``random_tiny_model()``: a tiny GPT-2 *architecture* with seeded random weights, built offline.
* ``load_model(name)``: a real pretrained checkpoint (downloads; used by the opt-in tests and demo).
"""

from __future__ import annotations

import numpy as np

from .tiny import gelu, layer_norm, softmax

DEFAULT_REAL_MODEL = "sshleifer/tiny-gpt2"  # check its license on the model card


def random_tiny_model(seed: int = 0, n_embd: int = 16, n_head: int = 2):
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(seed)
    cfg = GPT2Config(n_embd=n_embd, n_head=n_head, n_layer=1, n_positions=32, vocab_size=50,
                     bos_token_id=0, eos_token_id=0)
    cfg._attn_implementation = "eager"
    return GPT2LMHeadModel(cfg).eval()


def load_model(name: str = DEFAULT_REAL_MODEL):
    from transformers import GPT2LMHeadModel

    return GPT2LMHeadModel.from_pretrained(name, attn_implementation="eager").eval()


def _w(t):
    return t.detach().cpu().double().numpy()


def _ln(x, module):
    # HF's LayerNorm has learned scale and shift; ours is the bare normalisation.
    return layer_norm(x, module.eps) * _w(module.weight) + _w(module.bias)


def numpy_block(layer, x: np.ndarray, n_head: int) -> np.ndarray:
    """GPT-2 block: x + attn(ln_1(x)), then + mlp(ln_2(.)). `layer` supplies the weights only."""
    d = x.shape[-1]
    dh = d // n_head
    h = _ln(x, layer.ln_1)
    qkv = h @ _w(layer.attn.c_attn.weight) + _w(layer.attn.c_attn.bias)  # Conv1D: (in, out)
    q, k, v = np.split(qkv, 3, axis=-1)
    heads = []
    for i in range(n_head):  # one single-head attention per slice of the width
        s = slice(i * dh, (i + 1) * dh)
        sc = q[:, s] @ k[:, s].T / np.sqrt(dh)
        sc = np.where(np.triu(np.ones(sc.shape, dtype=bool), 1), -np.inf, sc)
        heads.append(softmax(sc) @ v[:, s])
    a = np.concatenate(heads, axis=-1) @ _w(layer.attn.c_proj.weight) + _w(layer.attn.c_proj.bias)
    x = x + a
    h = _ln(x, layer.ln_2)
    m = gelu(h @ _w(layer.mlp.c_fc.weight) + _w(layer.mlp.c_fc.bias))
    return x + m @ _w(layer.mlp.c_proj.weight) + _w(layer.mlp.c_proj.bias)


def torch_block(model, ids: list[int], device: str = "cpu"):
    """Run the full PyTorch model and capture layer 0's input and output (float64).

    The hooks matter: calling ``h[0]`` by hand skips the causal mask, which the model builds
    itself in ``forward``. Capturing during a real forward keeps PyTorch's own masking.
    """
    import torch

    model = model.to(device).double()
    seen = {}

    def grab(_module, args, kwargs, out):
        x = args[0] if args else kwargs["hidden_states"]
        seen["in"] = x[0].detach().cpu().numpy()
        seen["out"] = (out[0] if isinstance(out, tuple) else out)[0].detach().cpu().numpy()

    handle = model.transformer.h[0].register_forward_hook(grab, with_kwargs=True)
    try:
        with torch.no_grad():
            model(torch.tensor([ids], device=device))
    finally:
        handle.remove()
    return seen["in"], seen["out"]


def max_abs_diff(model, ids: list[int], device: str = "cpu") -> float:
    x, expected = torch_block(model, ids, device)
    got = numpy_block(model.transformer.h[0], x, model.config.n_head)
    return float(np.abs(got - expected).max())
