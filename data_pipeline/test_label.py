from data_pipeline import label


def test_pattern_files_are_complete_and_decoupled(tmp_path):
    common, patterns = label.load_patterns()
    assert len(patterns) == 12
    for name, p in patterns.items():
        assert p["name"] == name and p["places"] and p["questions"] and p["steps"]
        assert all(isinstance(q["expect"], bool) for q in p["questions"])
        assert all(s.get("then") in ("<", "<=") for s in p["steps"][:-1])
        ids = [s["id"] for s in p["steps"]]
        assert p["span"]["from"] in ids and p["span"]["to"] in ids and ids.index(p["span"]["from"]) <= ids.index(p["span"]["to"]) and p["span"]["max"] >= 1
        text = label.pattern_text(p)
        assert all(f"`{q['id']}`" in text for q in p["questions"]) and all(f"`{i}`" in text for i in ids)
    # what the model reads changed in one pattern: only that pattern is asked again
    for path in label.PATTERN_DIR.iterdir():
        text = path.read_text()
        if path.name == "C1.yaml":
            text = text.replace("it says what the check showed", "it states what the check showed")
        if path.name == "D2.yaml":  # a rule the program applies: no new answers needed
            text = text.replace("max: 6", "max: 3")
        (tmp_path / path.name).write_text(text)
    _, edited = label.load_patterns(tmp_path)
    assert [n for n in patterns if patterns[n]["sha"] != edited[n]["sha"]] == ["C1"]
    assert edited["D2"]["span"]["max"] == 3


def test_a_label_needs_answers_steps_order_a_command_and_a_short_span():
    c1 = label.load_patterns()[1]["C1"]
    kinds = {n: "RUN" if n % 2 else "SAY" for n in range(1, 50)}
    row = dict(steps=dict(result_step=13, check_step=15, finding_step=19),
               answers=dict(as_expected=False, message_gives_cause=False, ran_a_check=True, check_can_answer=True, changed_before_check=False, stated_finding=True))
    assert label.decide(c1, row, kinds, []) == ([15, 16, 17, 18, 19], [])
    assert label.decide(c1, dict(row, answers=dict(row["answers"], check_can_answer=False)), kinds, []) == ([], ["check_can_answer is not true"])
    assert label.decide(c1, dict(row, steps=dict(row["steps"], check_step=16)), kinds, [])[1] == ["check_step is not a RUN step"]
    assert label.decide(c1, dict(row, steps=dict(row["steps"], check_step=13)), kinds, [])[1] == ["result_step not < check_step"]
    assert label.decide(c1, dict(row, steps={}), kinds, [])[1][0] == "result_step missing"
    assert label.decide(c1, dict(row, steps=dict(row["steps"], finding_step=40)), kinds, []) == ([], ["more than 8 steps from check_step to finding_step"])


def test_a_probe_must_be_a_submit_command():
    a3 = label.load_patterns()[1]["A3"]
    kinds = {n: "RUN" for n in range(1, 50)}
    row = dict(steps=dict(question_step=10, submit_step=11, answer_step=12),
               answers=dict(judge_only_fact=True, step_on_known_trend=False, question_before=True, answer_used=True, left_on_probe=False))
    assert label.decide(a3, row, kinds, [11]) == ([10, 11, 12], [])
    assert label.decide(a3, row, kinds, [30])[1] == ["submit_step is not a submit command"]
