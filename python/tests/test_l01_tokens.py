"""Lesson 1 tests: offline, CPU only. Author: Sandeep Belgavi.

Tests marked ``download`` need ``python -m belgavi_llm.l01_tokens.fetch`` to have run once and are
skipped when the cache is missing.
"""

import json
from pathlib import Path

import pytest

from belgavi_llm.l01_tokens import fetch
from belgavi_llm.l01_tokens.compare import markdown_table, measure
from belgavi_llm.l01_tokens.cost import (
    ContextWindow,
    Pricing,
    Step,
    agent_run_cost,
    turns_until_full,
)
from belgavi_llm.l01_tokens.counting import count_tokens, encode, load, load_named, pieces
from belgavi_llm.l01_tokens.samples import LANGUAGES, PARALLEL, paragraph
from belgavi_llm.l01_tokens.toy import SENTENCES, train

REPO = Path(__file__).resolve().parents[2]
TOY = REPO / "testdata" / "l01" / "toy-tokenizer.json"
EXPECTED = REPO / "testdata" / "l01" / "toy-expected.json"
CACHE = REPO / fetch.DEFAULT_DIR

needs_cache = pytest.mark.skipif(
    not (CACHE / "cl100k_base.json").is_file(), reason="run the fetch step first"
)


# --- shared fixture: the Java tests read the same two files ---
def test_toy_tokenizer_matches_shared_expected_ids():
    tk = load(TOY)
    expected = json.loads(EXPECTED.read_text(encoding="utf-8"))
    assert set(expected) == set(SENTENCES)
    for text, ids in expected.items():
        assert encode(tk, text) == ids, text


def test_toy_training_is_reproducible():
    assert encode(train(), SENTENCES[1]) == encode(load(TOY), SENTENCES[1])


def test_pieces_round_trip_to_original_bytes():
    tk = load(TOY)
    for text in SENTENCES:
        assert b"".join(p.data for p in pieces(tk, text)) == text.encode("utf-8")


def test_unseen_script_costs_more_tokens_per_character():
    tk = load(TOY)
    english = "The team will reply within one working day."
    kannada = "ತಂಡವು ಒಂದು ಕೆಲಸದ ದಿನದೊಳಗೆ ಉತ್ತರಿಸುತ್ತದೆ."
    assert count_tokens(tk, kannada) / len(kannada) > count_tokens(tk, english) / len(english)


def test_empty_text_is_zero_tokens():
    assert count_tokens(load(TOY), "") == 0


def test_partial_utf8_piece_is_shown_as_hex():
    tk = load(TOY)
    shown = [p.show() for p in pieces(tk, "ಧ")]
    assert any(s.startswith("<0x") for s in shown) or "".join(shown) == "ಧ"


# --- samples ---
def test_parallel_samples_are_complete():
    assert all(set(row) == set(LANGUAGES) for row in PARALLEL)
    assert all(row[lang].strip() for row in PARALLEL for lang in LANGUAGES)


# --- fetch / converter ---
def test_bytes_to_unicode_is_a_bijection_over_all_bytes():
    m = fetch.bytes_to_unicode()
    assert sorted(m) == list(range(256))
    assert len(set(m.values())) == 256


def test_converter_rebuilds_a_tiny_table():
    # a, b, c, then merges "ab" (3) and "abc" (4): the converter must replay them as merges
    ranks = {b"a": 0, b"b": 1, b"c": 2, b"ab": 3, b"abc": 4}
    assert fetch._split_token(ranks, b"abc", 4) == (b"ab", b"c")
    with pytest.raises(ValueError):
        fetch._split_token({b"x": 0, b"y": 1, b"z": 2}, b"xyz", 5)


def test_parse_tiktoken_decodes_base64_rows():
    assert fetch.parse_tiktoken(b"YQ== 0\nYWI= 1\n") == {b"a": 0, b"ab": 1}


# --- cost arithmetic ---
def test_pricing_cost():
    p = Pricing(input_per_m=2.0, output_per_m=10.0)
    assert p.cost(1_000_000) == pytest.approx(2.0)
    assert p.cost(500_000, 100_000) == pytest.approx(1.0 + 1.0)


def test_context_window_fit_and_percent():
    w = ContextWindow(size=1000, reserved_for_output=200)
    assert w.usable_for_input == 800
    assert w.fits(800) and not w.fits(801)
    assert w.percent_used(250) == pytest.approx(25.0)


def test_turns_until_full():
    w = ContextWindow(size=1000, reserved_for_output=100)
    assert turns_until_full(w, fixed_prefix=300, tokens_per_turn=50) == 12
    assert turns_until_full(w, fixed_prefix=950, tokens_per_turn=50) == 0
    with pytest.raises(ValueError):
        turns_until_full(w, 0, 0)


def test_agent_run_resends_history_every_call():
    run = agent_run_cost(
        prefix_tokens=100,
        steps=[Step(output_tokens=10, tool_result_tokens=40), Step(output_tokens=20)],
        pricing=Pricing(1_000_000, 1_000_000),  # 1 USD per token, so cost == token count
    )
    assert run.per_call_input == [100, 150]
    assert run.input_tokens == 250 and run.output_tokens == 30
    assert run.peak_context_tokens == 170
    assert run.cost == pytest.approx(280.0)


def test_agent_run_input_grows_faster_than_steps():
    steps = [Step(output_tokens=20, tool_result_tokens=200)] * 10
    run = agent_run_cost(500, steps, Pricing(1.0, 1.0))
    assert run.input_tokens > 10 * 500 * 2  # far more than 10 independent calls of the prefix


def test_agent_run_reports_window_overflow():
    steps = [Step(output_tokens=10, tool_result_tokens=900)] * 3
    run = agent_run_cost(100, steps, Pricing(1.0, 1.0), ContextWindow(size=1000))
    assert not run.fits_window


def test_agent_run_rejects_bad_input():
    with pytest.raises(ValueError):
        agent_run_cost(10, [], Pricing(1, 1))


# --- measured comparison (needs the fetch step) ---
def test_missing_cache_explains_the_fix(tmp_path):
    with pytest.raises(FileNotFoundError, match="l01_tokens.fetch"):
        load_named("cl100k_base", tmp_path)


@pytest.mark.download
@needs_cache
def test_real_tokenizers_hindi_and_kannada_cost_more_than_english():
    table = measure({n: load_named(n, CACHE) for n in fetch.SPECS})
    for name, row in table.items():
        assert row["Hindi"] > row["English"], name
        assert row["Kannada"] > row["English"], name
    assert "| cl100k_base |" in markdown_table(table)


@pytest.mark.download
@needs_cache
def test_newer_vocabularies_are_never_worse_on_the_samples():
    t = measure({n: load_named(n, CACHE) for n in fetch.SPECS})
    for lang in LANGUAGES:
        assert t["o200k_base"][lang] <= t["cl100k_base"][lang] <= t["r50k_base"][lang]


@pytest.mark.download
@needs_cache
def test_converted_tokenizer_matches_tiktoken():
    tiktoken = pytest.importorskip("tiktoken")
    import zipfile

    jar = Path(CACHE / "jtokkit.jar")
    if not jar.is_file():
        pytest.skip("parity check needs the jar kept with --keep-jar")
    zf = zipfile.ZipFile(jar)
    spec = fetch.SPECS["cl100k_base"]
    ranks = fetch.parse_tiktoken(zf.read("com/knuddels/jtokkit/cl100k_base.tiktoken"))
    enc = tiktoken.Encoding("cl100k", pat_str=spec["pat"], mergeable_ranks=ranks, special_tokens={})
    tk = load_named("cl100k_base", CACHE)
    for lang in LANGUAGES:
        assert enc.encode(paragraph(lang)) == encode(tk, paragraph(lang))


def test_long_text_is_never_truncated():
    tk = load(TOY)
    sentence = "My book order has not arrived. "
    assert count_tokens(tk, sentence * 400) == 400 * count_tokens(tk, sentence)


def test_compare_extra_tokenizer_column(tmp_path, capsys, monkeypatch):
    from belgavi_llm.l01_tokens import compare

    if not (CACHE / "cl100k_base.json").is_file():
        pytest.skip("run the fetch step first")
    monkeypatch.setattr("sys.argv", ["compare", "--cache", str(CACHE), "--extra", f"toy={TOY}"])
    compare.main()
    assert "| toy |" in capsys.readouterr().out


def test_gpu_step_skips_cleanly_without_a_gpu(capsys):
    from belgavi_llm.l01_tokens import gpu_throughput

    ok, _ = gpu_throughput.cuda_ready()
    if ok:
        pytest.skip("this machine has a GPU; the gpu-marked test covers it")
    assert gpu_throughput.main([]) == 0
    assert "skipped" in capsys.readouterr().out


@pytest.mark.gpu
def test_gpu_throughput_runs_on_cuda():
    from belgavi_llm.l01_tokens import gpu_throughput

    if not gpu_throughput.cuda_ready()[0]:
        pytest.skip("needs a CUDA GPU")
    assert gpu_throughput.main(["--new-tokens", "16"]) == 0
