"""Author: Sandeep Belgavi."""

from belgavi_llm.l03_data import baseline, clean, dedup, demo, jsonl, leakage, split, synth


def _ok_row(i="CASE-1", src="c1", text="login fails again", label="login"):
    return {"id": i, "source": src, "label": label, "messages": [
        {"role": "user", "content": text}, {"role": "assistant", "content": "ok"}]}


def test_generator_is_deterministic_and_synthetic():
    a, b = synth.generate(300), synth.generate(300)
    assert a == b
    assert all(r["id"].startswith("CASE-") and r["source"].startswith("cust-") for r in a)


def test_generator_row_count_and_dirt():
    rows = synth.generate(2000)
    assert len(rows) > 2000
    assert len(jsonl.validate(rows)) > 0
    assert len(synth.generate(2000, dirty=False)) == 2000


def test_validate_flags_problems():
    assert jsonl.validate_row(_ok_row()) == []
    bad = _ok_row()
    bad["messages"][0]["role"] = "customer"
    assert any("role" in e for e in jsonl.validate_row(bad))
    assert "messages must be a non-empty list" in jsonl.validate_row({**_ok_row(), "messages": []})
    assert "duplicate id" in jsonl.validate([_ok_row(), _ok_row()])["CASE-1"]
    last_user = _ok_row()
    last_user["messages"].reverse()
    assert any("last message" in e for e in jsonl.validate_row(last_user))


def test_roundtrip(tmp_path):
    rows = synth.generate(50, dirty=False)
    assert jsonl.write_jsonl(tmp_path / "a" / "x.jsonl", rows) == 50
    assert jsonl.read_jsonl(tmp_path / "a" / "x.jsonl") == rows


def test_read_reports_line_number(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"a": 1}\n{oops\n')
    try:
        jsonl.read_jsonl(p)
    except ValueError as e:
        assert ":2:" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_clean_normalises_and_drops():
    r = _ok_row(text="  Mail  me at a.b@x.org  ")
    kept, dropped = clean.clean([r, _ok_row("CASE-2", text="   ")])
    assert dropped["invalid"] == 1
    assert jsonl.user_text(kept[0]) == "Mail me at <email>"
    assert r["messages"][0]["content"].startswith("  ")  # input untouched


def test_exact_and_near_dedup():
    rows = [_ok_row("1", text="the app crashes when I export my report to pdf today"),
            _ok_row("2", text="THE APP crashes  when I export my report to pdf today"),
            _ok_row("3", text="the app crashes when I export my report to pdf today please"),
            _ok_row("4", text="my receipt shows two charges for the same month")]
    rows, n_exact = dedup.exact_dedup(rows)
    assert n_exact == 1
    rows, n_near = dedup.near_dedup(rows)
    assert n_near == 1 and [r["id"] for r in rows] == ["1", "4"]


def test_minhash_estimates_jaccard():
    a = dedup.shingles("a b c d e f g h i j k l")
    b = dedup.shingles("a b c d e f g h i j k x")
    mh = dedup.MinHasher(256)
    sa, sb = mh.signature(a), mh.signature(b)
    est = sum(x == y for x, y in zip(sa, sb, strict=True)) / 256
    assert abs(est - dedup.jaccard(a, b)) < 0.12


def test_split_by_group_never_shares_a_source():
    rows = synth.generate(600, dirty=False)
    tr, va, te = split.split_by_group(rows)
    srcs = [{r["source"] for r in part} for part in (tr, va, te)]
    assert not (srcs[0] & srcs[1]) and not (srcs[0] & srcs[2]) and not (srcs[1] & srcs[2])
    assert len(tr) + len(va) + len(te) == len(rows)


def test_split_by_row_leaks_and_leakage_report_sees_it():
    rows = synth.generate(600, dirty=False)
    tr, _, te = split.split_by_row(rows)
    assert leakage.overlap(tr, te)["shared_sources"] > 0
    tr, _, te = split.split_by_group(rows)
    assert leakage.overlap(tr, te)["leaky_fraction"] == 0.0


def test_leakage_report_counts_exact_and_near_text():
    tr = [_ok_row("1", "c1", "the app crashes when I export my report to pdf today")]
    te = [_ok_row("2", "c2", "The app crashes when I export my report to pdf today"),
          _ok_row("3", "c3", "the app crashes when I export my report to pdf now"),
          _ok_row("4", "c4", "pre-order charged twice")]
    o = leakage.overlap(tr, te, near_threshold=0.5)
    assert (o["exact_text_overlap"], o["near_text_overlap"]) == (1, 1)


def test_naive_bayes_learns():
    rows = [_ok_row(str(i), "c", "password reset locked", "login") for i in range(5)] + \
           [_ok_row(str(9 + i), "c", "gift-card charged card", "billing") for i in range(5)]
    nb = baseline.NaiveBayes().fit(rows)
    assert nb.predict(_ok_row(text="my password is locked")) == "login"


def test_demo_is_deterministic_and_leaky_split_scores_higher():
    a = demo.run(rows_n=800, seeds=3)
    assert a == {**demo.run(rows_n=800, seeds=3), "seconds": a["seconds"]}
    assert a["clean_rows"] == 800
    assert a["accuracy_row_split"]["mean"] > a["accuracy_group_split"]["mean"]
    assert a["group_split_overlap_test"]["shared_sources"] == 0
