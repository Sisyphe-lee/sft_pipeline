from data_pipeline import label


def test_pattern_files_are_complete_and_decoupled(tmp_path):
    common, patterns = label.load_patterns()
    assert len(patterns) == 12
    for name, p in patterns.items():
        assert p["name"] == name and p["places"] and p["questions"] and p["steps"]
        assert all(isinstance(q["expect"], bool) for q in p["questions"])
        assert all(s.get("then") in ("<", "<=") for s in p["steps"][:-1])
        ids = [s["id"] for s in p["steps"]]
        span = p["span"]
        assert set(span["steps"]) <= set(ids) if "steps" in span else ids.index(span["from"]) <= ids.index(span["to"]) and span["max"] >= 1
        text = label.pattern_text(p)
        assert all(f"`{q['id']}`" in text for q in p["questions"]) and all(f"`{i}`" in text for i in ids)
    # what the model reads changed in one pattern: only that pattern is asked again
    for path in label.PATTERN_DIR.iterdir():
        text = path.read_text()
        if path.name == "C1.yaml":
            text = text.replace("so nothing is left to find out", "so nothing remains to find out")
        if path.name == "D1.yaml":  # rules the program applies: no new answers needed
            text = text.replace("max: 4", "max: 2").replace("within: 2", "within: 1")
        (tmp_path / path.name).write_text(text)
    _, edited = label.load_patterns(tmp_path)
    assert [n for n in patterns if patterns[n]["sha"] != edited[n]["sha"]] == ["C1"]
    assert edited["D1"]["span"]["max"] == 2 and edited["D1"]["steps"][1]["within"] == 1


def test_a_label_needs_answers_steps_order_a_command_and_a_short_span():
    patterns = label.load_patterns()[1]
    c1 = patterns["C1"]
    kinds = {n: "RUN" if n % 2 else "SAY" for n in range(1, 50)}
    row = dict(steps=dict(result_step=13, check_step=15, finding_step=18),
               answers=dict(against_expectation=True, message_gives_cause=False, ran_a_check=True, check_can_answer=True, changed_before_check=False,
                            finding_from_check=True, next_follows=True))
    assert label.decide(c1, row, kinds, []) == ([15, 16, 17, 18], [])
    assert label.decide(c1, dict(row, answers=dict(row["answers"], check_can_answer=False)), kinds, []) == ([], ["check_can_answer is not true"])
    assert label.decide(c1, dict(row, steps=dict(row["steps"], check_step=16)), kinds, [])[1] == ["check_step is not a RUN step"]
    assert label.decide(c1, dict(row, steps=dict(row["steps"], check_step=13)), kinds, [])[1][0] == "result_step not < check_step"
    assert label.decide(c1, dict(row, steps={}), kinds, [])[1][0] == "result_step missing"
    assert label.decide(c1, dict(row, steps=dict(row["steps"], finding_step=40)), kinds, []) == ([], ["more than 4 steps from check_step to finding_step"])
    assert label.decide(c1, dict(row, steps=dict(row["steps"], result_step=9)), kinds, []) == ([], ["check_step more than 3 steps after result_step"])
    # C2 marks the trial and the launch themselves, not the steps between them
    c2 = patterns["C2"]
    row = dict(steps=dict(trial_step=21, launch_step=25), answers=dict(costly=True, small_version_first=True, trial_can_tell=True, trial_result_used=True))
    assert label.decide(c2, row, kinds, []) == ([21, 25], [])
    assert label.decide(c2, dict(row, steps=dict(trial_step=11, launch_step=25)), kinds, []) == ([], ["launch_step more than 6 steps after trial_step"])
    # D2 needs time between naming an idea and taking it up again
    d2 = patterns["D2"]
    row = dict(steps=dict(named_step=5, reopen_step=30, test_step=31), answers={q["id"]: q["expect"] for q in d2["questions"]})
    assert label.decide(d2, row, kinds, []) == ([30, 31], [])
    assert label.decide(d2, dict(row, steps=dict(named_step=25, reopen_step=30, test_step=31)), kinds, []) == ([], ["reopen_step less than 10 steps after named_step"])


def test_a_probe_must_be_a_submit_command():
    a3 = label.load_patterns()[1]["A3"]
    kinds = {n: "RUN" for n in range(1, 50)}
    row = dict(steps=dict(question_step=10, submit_step=11, answer_step=12),
               answers=dict(fact_about_judge=True, candidate=False, question_before=True, answer_used=True, left_on_probe=False))
    assert label.decide(a3, row, kinds, [11]) == ([10, 11, 12], [])
    assert label.decide(a3, row, kinds, [30])[1] == ["submit_step is not a submit command"]
