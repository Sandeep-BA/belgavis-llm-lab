# Lesson 2: what a transformer actually computes

By Sandeep Belgavi. Everything here runs on a CPU in seconds. Data is synthetic. No GPU is needed.

By the end you can say, with numbers you computed yourself, what happens between "the cat sat on the" and
"the next word is a probability distribution".

## The five ideas

1. **Embeddings.** Each token id selects one row of a table: a vector of numbers. That vector is all the model
   knows about the token at the start.
2. **One self-attention head.** Each token makes a *query*, a *key* and a *value* by multiplying its vector by
   three weight matrices (`Wq`, `Wk`, `Wv`). Token *i* scores every token *j* with `q_i · k_j / sqrt(d)`; a softmax
   turns the scores into weights that sum to 1; the output is the weighted sum of the values.
3. **Causal masking.** Before the softmax, scores for later tokens are set to minus infinity, so their weight is
   exactly 0. A token can only read itself and what came before. That is what lets the model be trained to predict
   the next token, and it is why the first row of the weights table is always `[1, 0, 0, ...]`.
4. **The MLP block.** Applied to each token on its own: expand to a wider vector, apply a nonlinearity (GELU),
   project back. Attention moves information *between* tokens; the MLP transforms it *within* a token. Both are
   added back to the input (a *residual* connection).
5. **The next token is a distribution.** The last token's final vector is multiplied by the embedding table
   (tied embeddings) to give one score ("logit") per vocabulary word. A softmax turns the logits into
   probabilities. **Temperature is one division**: `softmax(logits / T)`. T below 1 sharpens the distribution,
   above 1 flattens it. It never changes which word is most likely.

### Attention by hand

Take three tokens `x = [[1,0],[0,1],[1,1]]` and let `Wq = Wk = Wv = I` (the identity), so `Q = K = V = x` and
`d = 2`.

- Scores `x xᵀ / sqrt(2)`: `[[1,0,1],[0,1,1],[1,1,2]] / 1.4142`.
- Row 0 can see only itself: weights `[1, 0, 0]`.
- Row 1 sees tokens 0 and 1: scores `0` and `0.7071`, so `softmax = [1/(1+e^0.7071), e^0.7071/(1+e^0.7071)]
  = [0.3302, 0.6698]`.
- Row 2 sees all three: scores `0.7071, 0.7071, 1.4142` give weights `[0.2483, 0.2483, 0.5035]`.
- Output of row 2 = `0.2483·[1,0] + 0.2483·[0,1] + 0.5035·[1,1] = [0.7517, 0.7517]`.

Both languages assert exactly these numbers (`test_attention_by_hand`, `attentionByHand`).

### Temperature by hand

Logits `[2.0, 1.0, 0.1]`:

| T | probabilities |
|---|---|
| 1.0 | 0.6590, 0.2424, 0.0986 |
| 0.5 | 0.8638, 0.1169, 0.0193 |
| 2.0 | 0.5017, 0.3043, 0.1940 |

`T = 0.5` is the same as doubling the logits: `softmax([4.0, 2.0, 0.2])`. A test asserts that equality.
`T = 0` would divide by zero, so the code rejects it; "greedy" decoding means taking the argmax instead.

## The code

| What | Python (NumPy) | Java (plain `double[][]`, no ML library) |
|---|---|---|
| Embedding, head, mask, MLP, tied output | `python/src/belgavi_llm/l02_transformer/tiny.py` | `java/src/main/java/dev/belgavi/llm/l02/TinyLm.java`, `Attention.java` |
| Softmax with temperature, sampling | `tiny.py` (`softmax`, `sample`) | `Sampler.java` |
| NumPy checked against PyTorch | `gpt2_check.py` | not applicable (no PyTorch in Java) |
| Walk-through that prints every stage | `python -m belgavi_llm.l02_transformer.demo` | `Lesson02Demo.java` |
| Tests | `python/tests/test_l02_transformer.py` | `java/src/test/java/dev/belgavi/llm/l02/AttentionTest.java` |

The toy model has no random numbers: the weight at row *i*, column *j* is `((7i + 13j + 5·seed) mod 11 − 5) / 10`.
Python and Java therefore produce the same numbers, and the tests assert the same fixed values in both.

What the toy leaves out, on purpose: positional information (so the two "the" tokens get identical embeddings),
several heads, layer norm, several layers. The PyTorch check below includes layer norm and several heads.

## Verified against PyTorch

`gpt2_check.py` re-implements layer 0 of a GPT-2 style model (layer norm, multi-head causal attention, MLP,
residuals) using the same `softmax`, `gelu` and `layer_norm` as the toy, then compares it with the output of
layer 0 from PyTorch/transformers.

Gotcha: in transformers 5.20.2, calling the block object (`model.transformer.h[0]`) by hand does **not** apply
the causal mask; the model builds the mask in its own `forward`. The check therefore captures layer 0's input
and output with a forward hook during a real `model(...)` call. Calling the block directly gave a difference of
0.0085 in our first attempt and matched only a *non-causal* NumPy version.

- Offline, always run: a tiny GPT-2 *architecture* with seeded random weights (`random_tiny_model`).
- Opt-in: a real pretrained checkpoint, `BELGAVI_RUN_DOWNLOAD=1 pytest -q python -k pretrained`. The default
  model id is `sshleifer/tiny-gpt2`; check its license on the model card before you rely on it. This check has
  **not been run** for this lesson, so no pretrained-checkpoint result is claimed: random weights, real
  architecture.

## Run it: three tiers

Pinned versions, checked on 2026-10-22 by installing them from PyPI (<https://pypi.org/project/numpy/2.4.6/>,
<https://pypi.org/project/torch/2.14.1/>, <https://pypi.org/project/transformers/5.20.2/>):
**numpy 2.4.6, torch 2.14.1, transformers 5.20.2**, Python 3.11+, Java 21+, Maven 3.9. The same pins are the
`l02` extra in `python/pyproject.toml` (torch comes separately, from the CPU-only index).

### CPU-only (laptop, any OS): well under 10 minutes

```sh
python3 -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -e "python[dev,l01,l02]"                 # numpy 2.4.6, transformers 5.20.2, pytest
python -m belgavi_llm.l02_transformer.demo           # prints every stage
pytest -q python                                     # lessons 1 and 2: 34 passed, 7 skipped
mvn -B -ntp -f java/pom.xml test
mvn -B -ntp -q -f java/pom.xml compile
java -cp java/target/classes dev.belgavi.llm.l02.Lesson02Demo
```

The 7 skips: 4 lesson-1 tests wait for lesson 1's `fetch` step, and 3 need a CUDA GPU or a model download.
Lesson 2 on its own is `pytest -q python -k l02`: 9 passed, 2 skipped.

Measured on a Windows 11 laptop (CPU only, Python 3.13.5, the pins above, JDK 21+, Maven 3.9.12, other programs
running): the Python demo takes about 10 to 20 s, almost all of it `import torch`; `pytest -q python` about
12 s; the Java demo about 1 s. Install time depends on your network and is not counted. On a 4-core x86-64
Linux container with the same pins the demo took 3.6 s and the Java demo 0.07 s.

The last digits of `max |diff|` depend on the platform and the PyTorch build: 6.94e-18 on the Windows laptop
(torch 2.14.1 here, and 2.11.0 in an earlier run), 1.39e-17 on the Linux container. Either way it is
floating-point rounding.

Python output:

```text
tokens: ['the', 'cat', 'sat', 'on', 'the'] -> [0, 1, 2, 3, 0]

1. embeddings (one row per token):
 [[ 0.   0.2  0.4 -0.5]
 [-0.4 -0.2  0.   0.2]
 [ 0.3  0.5 -0.4 -0.2]
 [-0.1  0.1  0.3  0.5]
 [ 0.   0.2  0.4 -0.5]]

2. causal attention weights (row = who is looking, column = who it looks at):
 [[1.     0.     0.     0.     0.    ]
 [0.5017 0.4983 0.     0.     0.    ]
 [0.3451 0.3447 0.3102 0.     0.    ]
 [0.2496 0.2503 0.2528 0.2473 0.    ]
 [0.1916 0.1976 0.2135 0.2057 0.1916]]

3. after attention + residual:
 [[-0.079   0.392   0.544  -0.712 ]
 [-0.3818 -0.1664  0.0314  0.0966]
 [ 0.3239  0.5591 -0.341  -0.2971]
 [-0.0462  0.1126  0.3146  0.4333]
 [ 0.0298  0.2459  0.4387 -0.593 ]]

4. MLP output:
 [[-0.0889  0.104   0.2293 -0.0732]
 [ 0.0537  0.1322 -0.039  -0.0741]
 [-0.1306 -0.1941  0.0602 -0.0176]
 [ 0.0518  0.0328 -0.0776  0.0333]
 [-0.0505  0.0882  0.1776 -0.0337]]

5. next-token distribution after 'the cat sat on the':
    the  0.2892
    cat  0.1286
    sat  0.1608
     on  0.1408
    mat  0.1247
      .  0.1560
   sum = 1.0000

Temperature is one division (logits / T):
   T=0.25 top=the p_top=0.7633 sample(u=0.5)=the
   T=0.5  top=the p_top=0.4503 sample(u=0.5)=cat
   T=1.0  top=the p_top=0.2892 sample(u=0.5)=sat
   T=2.0  top=the p_top=0.2222 sample(u=0.5)=sat
   T=4.0  top=the p_top=0.1929 sample(u=0.5)=sat

NumPy vs PyTorch, layer 0 of tiny random-weight GPT-2 (offline) on cpu: max |diff| = 6.94e-18
elapsed 10.09 s
```

Java output (the numbers agree with Python to the digits shown):

```text
tokens: [the, cat, sat, on, the]

1. embeddings
    0.0000  0.2000  0.4000 -0.5000
   -0.4000 -0.2000  0.0000  0.2000
    0.3000  0.5000 -0.4000 -0.2000
   -0.1000  0.1000  0.3000  0.5000
    0.0000  0.2000  0.4000 -0.5000

2. causal attention weights
    1.0000  0.0000  0.0000  0.0000  0.0000
    0.5017  0.4983  0.0000  0.0000  0.0000
    0.3451  0.3447  0.3102  0.0000  0.0000
    0.2496  0.2503  0.2528  0.2473  0.0000
    0.1916  0.1976  0.2135  0.2057  0.1916

3. after attention + residual
   -0.0790  0.3920  0.5440 -0.7120
   -0.3818 -0.1664  0.0314  0.0966
    0.3239  0.5591 -0.3410 -0.2971
   -0.0462  0.1126  0.3146  0.4333
    0.0298  0.2459  0.4387 -0.5930

4. MLP output
   -0.0889  0.1040  0.2293 -0.0732
    0.0537  0.1322 -0.0390 -0.0741
   -0.1306 -0.1941  0.0602 -0.0176
    0.0518  0.0328 -0.0776  0.0333
   -0.0505  0.0882  0.1776 -0.0337

5. next-token distribution after 'the cat sat on the':
    the  0.2892
    cat  0.1286
    sat  0.1608
     on  0.1408
    mat  0.1247
      .  0.1560
   sum = 1.0000

Temperature is one division (logits / T):
   T=0.25 top=the p_top=0.7633 sample(u=0.5)=the
   T=0.5  top=the p_top=0.4503 sample(u=0.5)=cat
   T=1.0  top=the p_top=0.2892 sample(u=0.5)=sat
   T=2.0  top=the p_top=0.2222 sample(u=0.5)=sat
   T=4.0  top=the p_top=0.1929 sample(u=0.5)=sat
```

The tests on their own: `mvn -B -ntp -f java/pom.xml test -Dtest=AttentionTest` ends with

```text
Tests run: 7, Failures: 0, Errors: 0, Skipped: 0
BUILD SUCCESS
```

### GPU (NVIDIA, 8 GB; Windows through WSL2)

The lesson does not need a GPU. The only GPU step runs the same PyTorch check on CUDA:

```sh
python -m belgavi_llm.l02_transformer.demo --device cuda
pytest -q python -m gpu      # skipped cleanly when torch.cuda.is_available() is False
```

Install the default (CUDA) PyTorch wheel (`pip install torch==2.14.1`) instead of the CPU one. Measured on an RTX 4060
Laptop GPU (8 GB, native Windows, 2026-10-23): `python -m belgavi_llm.l02_transformer.demo --device cuda` printed
`max |diff|` 1.39e-17 (layer 0, tiny GPT-2), in 22.6 s with a 1,040 MiB peak for the whole card (about 903 MiB
of it was held by other programs before the run).

### Free cloud GPU (Colab or Kaggle)

Open `notebooks/02-transformer.ipynb` and run all cells. It installs the pinned versions, runs the demo on
the GPU runtime if there is one (otherwise on CPU), and runs the tests. It does not need a GPU. The notebook
has **not been measured on a Colab T4 yet**, so no Colab number is claimed here.

## Exercises

1. Remove the causal mask (`causal=False`) and print the weights. Which entries become non-zero?
2. Change `Wq` for the toy head so token 2 attends mostly to token 0. What does that do to its output?
3. Plot `p_top` against temperature from 0.1 to 5. Where does it cross 0.5?
4. Add a second head by running `attention_head` twice with different weights and concatenating the outputs.

## Sources

- Attention formula: Vaswani et al., "Attention Is All You Need" (2017), <https://arxiv.org/abs/1706.03762>.
- GPT-2 block layout and the tanh GELU (`gelu_new`): the `transformers` GPT-2 implementation at the pinned
  version; the tests check our NumPy version against it.
