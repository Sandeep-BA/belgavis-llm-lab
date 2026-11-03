"""A toy transformer block in plain NumPy. Author: Sandeep Belgavi.

Everything is deterministic and uses synthetic data: the weights come from an integer formula, so
the Java version in ``dev.belgavi.llm.l02`` produces the same numbers (to about 1e-9).
Pipeline: token ids -> embeddings -> one causal self-attention head -> MLP -> logits -> softmax.
"""

from __future__ import annotations

import math

import numpy as np

VOCAB = ["the", "cat", "sat", "on", "mat", "."]
D_MODEL = 4
D_HIDDEN = 8


def weight(seed: int, rows: int, cols: int) -> np.ndarray:
    """Deterministic weight matrix: entry (i, j) = ((7i + 13j + 5 seed) mod 11 - 5) / 10."""
    w = np.empty((rows, cols))
    for i in range(rows):
        for j in range(cols):
            w[i, j] = ((7 * i + 13 * j + 5 * seed) % 11 - 5) / 10.0
    return w


def embed(ids: list[int], table: np.ndarray) -> np.ndarray:
    """Embedding lookup: row `id` of the table, one vector per token."""
    return table[ids]


def softmax(z: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """softmax(z / T) over the last axis. Temperature is just the one division."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0 (use argmax for 'T = 0')")
    s = z / temperature
    s = s - s.max(axis=-1, keepdims=True)  # shift for numerical stability; result is unchanged
    e = np.exp(s)
    return e / e.sum(axis=-1, keepdims=True)


def causal_mask(n: int) -> np.ndarray:
    """True where a token may NOT look: strictly above the diagonal (the future)."""
    return np.triu(np.ones((n, n), dtype=bool), k=1)


def attention_head(x, wq, wk, wv, causal: bool = True):
    """One self-attention head. Returns (output, attention weights)."""
    q, k, v = x @ wq, x @ wk, x @ wv
    scores = q @ k.T / math.sqrt(k.shape[-1])
    if causal:
        scores = np.where(causal_mask(len(x)), -np.inf, scores)
    weights = softmax(scores)
    return weights @ v, weights


def gelu(x):
    """GELU, tanh approximation (the one GPT-2 uses)."""
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def mlp(x, w1, b1, w2, b2):
    """The MLP block: expand, nonlinearity, project back. Applied to each token on its own."""
    return gelu(x @ w1 + b1) @ w2 + b2


def layer_norm(x, eps: float = 1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


class TinyLM:
    """Embedding table + one block (attention + MLP, both with residual) + output projection."""

    def __init__(self):
        n = len(VOCAB)
        self.table = weight(1, n, D_MODEL)
        self.wq, self.wk, self.wv = (weight(s, D_MODEL, D_MODEL) for s in (2, 3, 4))
        self.wo = weight(5, D_MODEL, D_MODEL)
        self.w1, self.b1 = weight(6, D_MODEL, D_HIDDEN), np.zeros(D_HIDDEN)
        self.w2, self.b2 = weight(7, D_HIDDEN, D_MODEL), np.zeros(D_MODEL)
        self.unembed = self.table.T  # tied embeddings: logits = h @ table.T

    def forward(self, ids: list[int]):
        """Return a dict with every intermediate, so each step can be inspected."""
        x = embed(ids, self.table)
        attn, weights = attention_head(x, self.wq, self.wk, self.wv)
        h1 = x + attn @ self.wo
        m = mlp(h1, self.w1, self.b1, self.w2, self.b2)
        h2 = h1 + m
        logits = h2 @ self.unembed
        return {"x": x, "attn": attn, "weights": weights, "h1": h1, "mlp": m, "h2": h2,
                "logits": logits}

    def next_token_probs(self, ids: list[int], temperature: float = 1.0) -> np.ndarray:
        return softmax(self.forward(ids)["logits"][-1], temperature)


def sample(probs: np.ndarray, u: float) -> int:
    """Inverse-CDF sampling from an explicit uniform draw u in [0, 1), so it is testable."""
    c = 0.0
    for i, p in enumerate(probs):
        c += p
        if u < c:
            return i
    return len(probs) - 1
