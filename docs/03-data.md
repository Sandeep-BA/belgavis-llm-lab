# Lesson 3: Data: JSONL, cleaning, and splits without leakage

By Sandeep Belgavi.

You will build a small instruction dataset, clean it, remove duplicates, split it three ways, and then **measure how a bad split inflates your score**. Everything here is synthetic (help-desk cases of Belgavi's Book Store, a fictional online bookstore, numbered `CASE-xxxx`), runs offline, and needs no GPU.

| | Java (primary) | Python |
|---|---|---|
| Code | `java/src/main/java/dev/belgavi/llm/l03/` | `python/src/belgavi_llm/l03_data/` |
| Tests | `java/src/test/java/dev/belgavi/llm/l03/Lesson03Test.java` | `python/tests/test_l03_data.py` |
| Does | JSONL reader/writer (Jackson), validator, cleaner, exact + near dedup, split by row / by group, leakage checker | the same, plus MinHash + LSH near-duplicate search, a synthetic case generator, a naive-Bayes baseline and the leakage demo |

## 1. The format: chat-style instruction JSONL

One JSON object per line. `messages` is the conversation the model trains on; `source` says where the row came from; `label` is what the classifier baseline predicts.

```json
{"id": "CASE-1001", "source": "cust-001", "label": "shipping",
 "messages": [
   {"role": "system",    "content": "You are the help-desk assistant for Belgavi's Book Store. Answer briefly and politely."},
   {"role": "user",      "content": "Tanishka-1 laptop E766: the for parcel help receipt we code library paperback arrived today"},
   {"role": "assistant", "content": "We have asked the courier for a fresh tracking update on your books and will email the result."}]}
```

The validator (`Validator.java`, `jsonl.validate_row`) rejects: a missing `id`/`source`/`label`; an empty `messages` list; a role outside `system`/`user`/`assistant`; empty or over-long content; no user turn; a last turn that is not the assistant's; and duplicate ids. Fix these *before* training: a trainer that meets a bad row either crashes an hour in or quietly learns from it.

## 2. Cleaning and duplicates

- **Clean** (`clean`): Unicode NFKC, collapse whitespace, mask e-mail addresses with `<email>`, then drop rows that still fail validation. Case is kept.
- **Exact duplicates**: same text after lower-casing and collapsing whitespace.
- **Near duplicates**: rows that differ by a word or two. Python uses MinHash (64 permutations) with LSH banding (16 bands) to find candidates cheaply, then confirms each with the exact Jaccard of word 3-gram shingles (threshold 0.7). Java compares shingle sets directly, which is simple and fine at this size.

Duplicates matter twice: they waste training steps, and when one copy lands in train and the other in test, the model is graded on something it memorised.

## 3. Splits: by source, not by row

Customers open several cases each, about the same recurring problem (a pre-order charged twice, an e-book that will not download), with their own customer handle (`Tanishka-1`), device and error code. If you shuffle **rows**, a customer's cases end up on both sides of the line and the model can recognise the customer instead of learning the problem. Split by **source**: every case from a customer goes to exactly one of train / validation / test.

```text
by row:    cust-007 -> train, train, test, train, test      (leaks)
by source: cust-007 -> train, train, train, train, train    (clean)
```

Deduplication alone does not fix this: two cases from the same customer are rarely near-duplicates, yet they share the customer's vocabulary. That is why the leakage checker looks at **sources** as well as text.

## 4. The leakage demo, measured

`demo.py` generates 2,000 clean rows (plus 160 dirty ones for the cleaner to find), cleans and de-duplicates them, and trains a tiny multinomial naive-Bayes case classifier (standard library only) on two splits of the same data, 10 seeds each. The model is deliberately weak: the point is the split.

Run it (CPU, standard library, no install beyond the package):

```sh
PYTHONPATH=python/src python3 -m belgavi_llm.l03_data.demo --out data/cache/l03
```

Output, measured on an Intel Core i9-13900HX laptop (24 cores, 16 GB RAM, Windows 11), Python 3.13.5, 2026-10-26, wall time 4.1 s (other jobs were running on the same machine):

```text
{
  "raw_rows": 2160,
  "invalid_rows": 80,
  "dropped_by_clean": 80,
  "exact_duplicates_removed": 40,
  "near_duplicates_removed": 40,
  "clean_rows": 2000,
  "sources": 230,
  "row_split_overlap_test": {
    "rows": 400,
    "shared_sources": 194,
    "rows_from_shared_sources": 400,
    "exact_text_overlap": 0,
    "near_text_overlap": 0,
    "leaky_fraction": 1.0
  },
  "group_split_overlap_test": {
    "rows": 414,
    "shared_sources": 0,
    "rows_from_shared_sources": 0,
    "exact_text_overlap": 0,
    "near_text_overlap": 0,
    "leaky_fraction": 0.0
  },
  "accuracy_row_split": {
    "mean": 0.8273,
    "stdev": 0.0156
  },
  "accuracy_group_split": {
    "mean": 0.7155,
    "stdev": 0.0166
  },
  "seeds": 10,
  "seconds": 4.1
}
```

What it says:

- **Split by row: 82.7 % accuracy.** Every one of the 400 test rows comes from a customer that is also in train (`leaky_fraction` 1.0).
- **Split by source: 71.6 % accuracy.** No shared sources. This is the honest number, about 11 points lower, for the same data and the same model.
- Text overlap is **0** in both: no exact or near-copy text crossed the line. The leak is entirely through the *source*, which is why a text-only check would have said "all clear".

Caveats, so you do not over-read it: the data is synthetic and the generator was written to behave like repeat-customer help-desk data (70 % of a customer's cases share one category, plus customer-specific tokens). The size of the gap depends on that design; the *direction* is what real data shows too. The stdev is across 10 split seeds of one generated dataset.

## 5. Run it: three tiers

This lesson has no GPU step, so the three tiers run the same commands and give the same numbers. We state that rather than invent a GPU speed-up.

### CPU only (about 4 seconds; budget is 10 minutes)

```sh
# Python
python3 -m venv .venv && .venv/bin/pip install -e "python[dev]"
.venv/bin/pytest python/tests/test_l03_data.py      # 13 passed
.venv/bin/python -m belgavi_llm.l03_data.demo --out data/cache/l03

# Java 21 + Maven
mvn -B -ntp -f java/pom.xml test
mvn -B -ntp -q -f java/pom.xml compile exec:java \
    -Dexec.mainClass=dev.belgavi.llm.l03.DatasetBuilder \
    -Dexec.args="data/cache/l03/raw.jsonl data/cache/l03/java"
```

The lesson-3 tests need nothing beyond the standard library and pytest. The whole Python suite (lessons 1 to 3,
installed as in the main README with `python[dev,l01,l02]`) gives `47 passed, 7 skipped`: the 7 skips are lesson 1's
`fetch` step and the GPU or download checks of lessons 1 and 2. The Java suite gives `Tests run: 35, Failures: 0,
Errors: 0, Skipped: 1`; `mvn -B -ntp -f java/pom.xml test -Dtest=Lesson03Test` runs this lesson's 7.

The Java run reads the `raw.jsonl` the Python demo wrote and prints (measured on the same laptop, Java 21.0.12.1, Maven 3.9.12, 13 s including compilation):

```text
raw=2160 invalid=80 exact_dups=40 near_dups=40 clean=2000
train=1384 validation=195 test=421
group split vs test: Report[rows=421, sharedSources=0, rowsFromSharedSources=0, exactTextOverlap=0, nearTextOverlap=0, leakyFraction=0.0]
row split   vs test: Report[rows=400, sharedSources=194, rowsFromSharedSources=400, exactTextOverlap=0, nearTextOverlap=0, leakyFraction=1.0]
```

Both languages agree on the cleaning counts (2,160 raw, 80 invalid, 40 exact, 40 near, 2,000 clean). The split sizes differ slightly because each language shuffles with its own random generator.

### NVIDIA GPU (8 GB; Windows through WSL2)

Same commands as the CPU tier, inside Linux or WSL2. Nothing in this lesson uses the GPU, so there is no GPU number. Later lessons reuse the data format and splits built here.

### Free cloud GPU (Colab or Kaggle)

Open `notebooks/03-data.ipynb` in Colab or Kaggle. It clones the public `belgavis-llm-lab` repo, installs the package and runs the same demo; a GPU runtime is not required.

## 6. Versions and sources

- Java 21, Maven Surefire 3.6.0, JUnit Jupiter 5.14.4, Jackson Databind 2.22.4 (pinned in `java/pom.xml`).
- Python >= 3.11. The lesson code uses only the standard library; tests use pytest 9.1.1 and ruff 0.17.2 (PyPI latest on the day of the run).
- MinHash and LSH banding: Broder (1997), "On the resemblance and containment of documents"; Leskovec, Rajaraman, Ullman, *Mining of Massive Datasets*, chapter 3 (free online).

## Exercises

1. Change `near_threshold` in the demo to 0.3. What does the overlap report start catching?
2. Make the generator give every customer a different main category (remove the 70 % rule). Does the gap between the two splits shrink? Why?
3. Add a third split strategy: by time (oldest 70 % train). Which sources does it leak?
