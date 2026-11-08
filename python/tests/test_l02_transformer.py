"""Lesson 2 tests: fixed numbers shared with the Java tests. Author: Sandeep Belgavi."""

import math
import os

import pytest

np = pytest.importorskip("numpy")

from belgavi_llm.l02_transformer.tiny import (  # noqa: E402
    VOCAB,
    TinyLM,
    attention_head,
    causal_mask,
    sample,
    softmax,
)

X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
EYE = np.eye(2)


def test_attention_by_hand():
    """Wq = Wk = Wv = I: scores are x x^T / sqrt(2), then a causal softmax per row."""
    out, w = attention_head(X, EYE, EYE, EYE)
    e = math.exp(1 / math.sqrt(2))
    assert w[0] == pytest.approx([1, 0, 0])
    assert w[1] == pytest.approx([1 / (1 + e), e / (1 + e), 0])
    assert w[1, 0] == pytest.approx(0.330238, abs=1e-6)
    assert w[2] == pytest.approx([0.248255, 0.248255, 0.503490], abs=1e-6)
    assert out[2] == pytest.approx([0.751745, 0.751745], abs=1e-6)


def test_causal_mask_blocks_the_future():
    assert causal_mask(3).tolist() == [[False, True, True], [False, False, True], [False] * 3]
    _, w = attention_head(X, EYE, EYE, EYE)
    assert np.triu(w, 1).sum() == 0
    # Changing a later token must not change an earlier token's output.
    x2 = X.copy()
    x2[2] = [5.0, -3.0]
    assert attention_head(x2, EYE, EYE, EYE)[0][:2] == pytest.approx(
        attention_head(X, EYE, EYE, EYE)[0][:2])


def test_without_mask_rows_see_everything():
    _, w = attention_head(X, EYE, EYE, EYE, causal=False)
    assert (w > 0).all()


def test_softmax_temperature_is_one_division():
    z = np.array([2.0, 1.0, 0.1])
    assert softmax(z) == pytest.approx([0.659001, 0.242433, 0.098566], abs=1e-6)
    assert softmax(z, 0.5) == pytest.approx([0.863777, 0.116900, 0.019323], abs=1e-6)
    assert softmax(z, 2.0) == pytest.approx([0.501688, 0.304289, 0.194023], abs=1e-6)
    assert softmax(z, 0.5) == pytest.approx(softmax(z / 0.5))
    with pytest.raises(ValueError):
        softmax(z, 0.0)


def test_softmax_stable_for_huge_logits():
    assert softmax(np.array([1000.0, 1000.0])) == pytest.approx([0.5, 0.5])


def test_sample_inverse_cdf():
    p = np.array([0.5, 0.3, 0.2])
    assert [sample(p, u) for u in (0.0, 0.49, 0.5, 0.79, 0.8, 0.999)] == [0, 0, 1, 1, 2, 2]


def test_tiny_lm_fixed_numbers():
    m = TinyLM()
    ids = [0, 1, 2, 3, 0]
    r = m.forward(ids)
    assert r["weights"][1] == pytest.approx([0.50175, 0.49825, 0, 0, 0], abs=1e-5)
    assert r["logits"][-1] == pytest.approx(
        [0.626714, -0.183847, 0.039615, -0.093019, -0.214123, 0.009338], abs=1e-5)
    p = m.next_token_probs(ids)
    assert p.sum() == pytest.approx(1.0)
    assert p == pytest.approx(
        [0.289173, 0.128569, 0.160762, 0.140793, 0.124735, 0.155968], abs=1e-5)
    assert VOCAB[int(p.argmax())] == "the"


def test_toy_head_matches_torch():
    torch = pytest.importorskip("torch")
    x = torch.tensor(X)
    s = x @ x.T / math.sqrt(2)
    s = s.masked_fill(torch.triu(torch.ones(3, 3, dtype=torch.bool), 1), float("-inf"))
    expected = torch.softmax(s, -1) @ x
    out, _ = attention_head(X, EYE, EYE, EYE)
    assert out == pytest.approx(expected.numpy(), abs=1e-12)


def test_numpy_layer_matches_pytorch_gpt2_layer0():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from belgavi_llm.l02_transformer.gpt2_check import max_abs_diff, random_tiny_model

    assert max_abs_diff(random_tiny_model(), [3, 1, 4, 1, 5, 9, 2, 6]) < 1e-9


@pytest.mark.gpu
def test_numpy_layer_matches_pytorch_on_cuda():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA GPU")
    from belgavi_llm.l02_transformer.gpt2_check import max_abs_diff, random_tiny_model

    assert max_abs_diff(random_tiny_model(), [3, 1, 4, 1, 5, 9, 2, 6], "cuda") < 1e-9


@pytest.mark.download
@pytest.mark.skipif(os.environ.get("BELGAVI_RUN_DOWNLOAD") != "1",
                    reason="downloads a model; set BELGAVI_RUN_DOWNLOAD=1")
def test_numpy_layer_matches_pretrained_checkpoint():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from belgavi_llm.l02_transformer.gpt2_check import load_model, max_abs_diff

    assert max_abs_diff(load_model(), [3, 1, 4, 1, 5, 9, 2, 6]) < 1e-6
