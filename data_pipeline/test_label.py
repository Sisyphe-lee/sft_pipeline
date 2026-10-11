import re
from pathlib import Path

from data_pipeline import label

SHEET = (Path(__file__).parents[1] / "patterns/label-sheet.md").read_text()


def section(pattern):
    return SHEET.split(f"### {pattern} ")[1].split("###")[0]


def test_sheet_and_program_ask_for_the_same_answers_and_steps():
    for pattern, spec in label.EVIDENCE.items():
        text = section(pattern)
        asked = {name: value == "true" for name, value in re.findall(r"^- `(\w+)` \[(true|false)\]", text, re.M)}
        assert asked == spec["need"], pattern
        for step in spec["chain"][::2]:
            assert f"`{step}`" in text, (pattern, step)


def test_a_label_needs_answers_steps_order_and_a_command():
    kinds = {n: "RUN" if n % 2 else "SAY" for n in range(1, 50)}
    row = dict(pattern="C1", start=15, end=19, steps=dict(result_step=13, check_step=15, finding_step=19),
               answers=dict(as_expected=False, message_gives_cause=False, ran_a_check=True, check_can_answer=True, changed_before_check=False, stated_finding=True))
    assert label.evidence_gaps(row, kinds, []) == []
    assert label.evidence_gaps(dict(row, answers=dict(row["answers"], check_can_answer=False)), kinds, []) == ["check_can_answer is not true"]
    assert label.evidence_gaps(dict(row, steps=dict(row["steps"], check_step=16)), kinds, []) == ["check_step is not a RUN step"]
    assert label.evidence_gaps(dict(row, steps=dict(row["steps"], check_step=13)), kinds, []) == ["result_step not < check_step", "check_step outside the span"]
    assert label.evidence_gaps(dict(row, steps={}), kinds, [])[0] == "result_step missing"


def test_a_probe_must_be_a_submit_command():
    kinds = {n: "RUN" for n in range(1, 50)}
    row = dict(pattern="A3", start=10, end=12, steps=dict(question_step=10, submit_step=11, answer_step=12),
               answers=dict(judge_only_fact=True, step_on_known_trend=False, question_before=True, answer_used=True, left_on_probe=False))
    assert label.evidence_gaps(row, kinds, [11]) == []
    assert label.evidence_gaps(row, kinds, [30]) == ["submit_step is not a submit command"]
