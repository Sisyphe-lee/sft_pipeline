from data_pipeline import label


def test_pattern_files_are_complete_and_decoupled(tmp_path):
    common, patterns = label.load_patterns()
    assert len(patterns) == 12
    for name, p in patterns.items():
        assert p["name"] == name and p["places"] and p["questions"] and p["steps"]
        assert all(isinstance(q["expect"], bool) for q in p["questions"])
        assert all(s.get("then") in ("<", "<=") for s in p["steps"][:-1])
        text = label.pattern_text(p)
        assert all(f"`{q['id']}`" in text for q in p["questions"]) and all(f"`{s['id']}`" in text for s in p["steps"])
    # editing one pattern file changes that pattern's fingerprint only
    for path in label.PATTERN_DIR.iterdir():
        (tmp_path / path.name).write_text(path.read_text() + ("\n# edited\n" if path.name == "C1.yaml" else ""))
    _, edited = label.load_patterns(tmp_path)
    assert [n for n in patterns if patterns[n]["sha"] != edited[n]["sha"]] == ["C1"]


def test_a_label_needs_answers_steps_order_and_a_command():
    c1 = label.load_patterns()[1]["C1"]
    kinds = {n: "RUN" if n % 2 else "SAY" for n in range(1, 50)}
    row = dict(start=15, end=19, steps=dict(result_step=13, check_step=15, finding_step=19),
               answers=dict(as_expected=False, message_gives_cause=False, ran_a_check=True, check_can_answer=True, changed_before_check=False, stated_finding=True))
    assert label.label_gaps(c1, row, kinds, []) == []
    assert label.label_gaps(c1, dict(row, answers=dict(row["answers"], check_can_answer=False)), kinds, []) == ["check_can_answer is not true"]
    assert label.label_gaps(c1, dict(row, steps=dict(row["steps"], check_step=16)), kinds, []) == ["check_step is not a RUN step"]
    assert label.label_gaps(c1, dict(row, steps=dict(row["steps"], check_step=13)), kinds, []) == ["result_step not < check_step", "check_step outside the span"]
    assert label.label_gaps(c1, dict(row, steps={}), kinds, [])[0] == "result_step missing"
    assert "no span" in label.label_gaps(c1, dict(row, start=None), kinds, [])


def test_a_probe_must_be_a_submit_command():
    a3 = label.load_patterns()[1]["A3"]
    kinds = {n: "RUN" for n in range(1, 50)}
    row = dict(start=10, end=12, steps=dict(question_step=10, submit_step=11, answer_step=12),
               answers=dict(judge_only_fact=True, step_on_known_trend=False, question_before=True, answer_used=True, left_on_probe=False))
    assert label.label_gaps(a3, row, kinds, [11]) == []
    assert label.label_gaps(a3, row, kinds, [30]) == ["submit_step is not a submit command"]
