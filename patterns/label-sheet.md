# Label sheet (draft 1, 2026-10-11)

You read part of the log of one run: an agent working alone for hours on one task that a judge scores. You are given places in the log and, for each place, behaviour patterns to look for there. For each pattern you answer a few yes/no questions and name the steps that show the behaviour. A program turns your answers into a label: the steps are used as training examples of the behaviour only when every answer and every step holds. You do not give a verdict.

A wrong example costs more than a missed one. When you cannot tell, give the answer that speaks against the behaviour, and leave a step `null` when you have not found it.

## The run

- Every step has a number (`#N`) and is one reply of the agent. Its head says `RUN` when the reply ran a command and `SAY` when it did not. Inside a step, `THINK` is the agent's reasoning, `SAY` is what it wrote, and `$` is the command with the output that came back. The reasoning and the text of a step were written before its command ran: they react to the output of earlier steps, not to their own. Long texts are cut in the middle.
- The agent submits with `sforge-submit`. The judge returns a score, sometimes with details. Only the best submission counts. A harness also submits the current files every 30 minutes (`auto-N`); the agent is not shown those results.
- A `[judge record ...]` note on a step was added by us. The agent knows only what its command printed.
- Some tasks are games played through a server: the score is the game's own score and there are no submissions.

## Material

- HEAD: facts about the run and the list of judged rounds.
- TIMELINE: another reader's account of the whole run. Use it to learn what happened outside the steps you see. It can be wrong; the log decides.
- LOG: the steps around the place.
- PACKET: the places. Each has `id`, `type`, `step`, `steps` (where to start reading), `patterns` and `facts`. Places of type `start`, `submit`, `result` and `end` were found by rule; the others were reported by the other reader, whose notes are in `facts`.

## What to write

For every place and every pattern listed for it, one line:

1. Answer every question of the pattern with `true` or `false`.
2. If the answers describe the behaviour (each has the value shown in brackets), fill in the steps. A step marked RUN must be a `RUN` step: a command the agent ran. A sentence in which it explains, estimates or reasons is not one.
3. `start` and `end`: the first and last step of the agent's own behaviour, tight. They hold the behaviour and little else; the step that only shows the situation (the result, the failure) stays outside when the behaviour begins later.

The answers must be about this place. If the place was located wrongly (the facts say one thing, the log another), answer by the log.

## The patterns

### A1 First submission early (place: `start`)
- `improved_before_first_submit` [false]: before the first submission the agent tuned its solution or compared variants of it.
- `read_answer` [true]: it read what the judge returned and said something about it.
- Steps: `submit_step` (RUN, the first submit command) ≤ `read_step`.

### L1 A local number before submitting (place: `submit`)
- `meant_to_score` [true]: the submission is meant to score. False for a first look at the judge and for a submission made to learn something.
- `measured_this_version` [true]: a local run printed a score-like number for exactly the version that is submitted.
- `changed_after_measuring` [false]: a file of the solution changed between that run and the submit command.
- `number_says_submit` [true]: the number is at least as good as the best so far, or the agent gives it as the reason to submit.
- Steps: `measure_step` (RUN, prints the number) < `submit_step` (RUN).

### A4 One change per submission (place: `submit`)
- `changed_and_meant_to_compare` [true]: the submission differs from the last judged version and the agent will read the score to learn whether the change helped. False for the first submission, for unchanged files, and for a submission made to learn a fact about the judge.
- `one_idea_or_each_measured` [true]: everything changed since the last judged version implements one idea, or every change was measured on its own beforehand.
- Steps: `first_edit_step` ≤ `submit_step` (RUN).

### A2 Use the parts of the judge's answer (place: `result`)
- `saw_parts` [true]: the command printed more than a total: results per test or dataset, sub-scores, named failures.
- `planned_before_result` [false]: the work that follows was already planned or begun before this result.
- `part_names_target` [true]: the agent names one part of the answer, and its next piece of work is on exactly that part.
- Steps: `result_step` < `work_step` (the first step of that work).

### C1 Check the reason of an unexpected result before changing anything (places: `result`, `unexpected`)
- `as_expected` [false]: the result is what the agent expected, or one more step of a trend it has already seen, or the success of what it just tried.
- `message_gives_cause` [false]: the output itself says what is wrong (an error message, a refusal that names its reason), so nothing is left to find out.
- `ran_a_check` [true]: after the result the agent ran a command whose purpose was to find the reason. Reading the same output again, or reasoning, is not a check.
- `check_can_answer` [true]: the check's output bears on the reason: it could have confirmed or ruled out a cause.
- `changed_before_check` [false]: it changed the solution or submitted again before the check.
- `stated_finding` [true]: it says what the check showed, and its next step follows from that.
- Steps: `result_step` < `check_step` (RUN) ≤ `finding_step`.
- Also `"source"`: `"judge"` when the unexpected result is a judge answer, `"local"` otherwise.

### A3 Ask the judge with a probe submission (places: `probe`, `result`)
- `judge_only_fact` [true]: there is an open question that only the judge can answer. A question that a local run, a local server or the files can answer is not one.
- `step_on_known_trend` [false]: the submission is the next step along a trend the judge has already shown (one more value in the same direction).
- `question_before` [true]: the agent says what it wants to learn before it submits.
- `answer_used` [true]: it reads the answer and its next decision follows from it.
- `left_on_probe` [false]: the submitted files stay on the probe version afterwards.
- Steps: `question_step` ≤ `submit_step` (RUN, a submit command) ≤ `answer_step`.

### B1 Back to the best version (places: `result`, `end`)
- `files_not_best` [true]: at this point the files a submission would take are not the best judged version.
- `built_on_worse` [false]: it made further changes on top of the worse version before going back.
- `best_back` [true]: it puts the best version back.
- Steps: `restore_step` (RUN, the command that puts it back).

### D4 Leave a direction that failed twice (place: `second_failure`)
- `same_direction_twice` [true]: one direction or premise that you can name had at least two attempts without improvement by this step.
- `results_moving` [false]: each attempt was guided by new evidence and the results were improving.
- `more_than_one_further` [false]: after this step the agent makes two or more further attempts in the direction.
- `came_back_later` [false]: it returns to the direction later without new evidence.
- Steps: `first_failure_step` < `second_failure_step` ≤ `leave_step` (the first step of the other work).

### D1 On leaving a direction, gather new evidence first (place: `second_failure`)
- `left_direction` [true]: a direction failed at least twice and the agent leaves it for good.
- `new_kind_of_evidence` [true]: the first thing after leaving is a command that gets a kind of evidence it had not used so far: single cases, a profile, the task or scoring text, a probe of the judge, a test of why the direction failed.
- `used_for_next_change` [true]: it says what the evidence shows, and the next change was not chosen before it.
- Steps: `leave_step` ≤ `evidence_step` (RUN) ≤ `finding_step`.

### D2 Reopen an idea that was set aside (places: `set_aside`, `end`)
- `named_and_untested` [true]: the agent named something it could do and did not do it.
- `tested_soon_after` [false]: it tested the thing soon after, before any stall.
- `stalled_after` [true]: afterwards several attempts brought no new best, or it said nothing was left to try.
- `reopened_same_idea` [true]: it went back to that same idea, not to a new one.
- `test_reached_idea` [true]: the test it ran tried the idea itself and gave an answer about it. A test that failed for another reason first does not count.
- Steps: `named_step` < `stall_step` ≤ `test_step` (RUN).

### D3 Keep a record of what was tried, and look back at it (places: `record`, `redo`, `end`)
- `record_file` [true]: there is a file that holds results of earlier experiments: what was tried and the number or outcome it gave. A notes file, a results table and a log that the agent's scripts append to all count. The judge's submission list and plain copies of versions do not.
- `read_back_later` [true]: in a later command the agent reads that file to see results of earlier experiments. Looking at the output of the run it has just made is not a look back.
- `read_feeds_decision` [true]: what it read is used in the choice of what to do next.
- Steps: `write_step` (RUN) < `read_step` (RUN) ≤ `decision_step`. Also `"file"`: the name of the file.
- `start` and `end`: the read-back and the decision.

### C2 A small trial before something expensive (place: `expensive_start`)
- `costly` [true]: the thing started runs five minutes or more, or is a large implementation resting on an unchecked idea. A submission is not such a start.
- `small_version_first` [true]: before it, the agent ran a smaller version of this same thing (a short run, a small input, a timing measurement, a tiny case against a known answer). An earlier experiment with another purpose is not a trial of this one.
- `trial_can_tell` [true]: the trial could have shown that the idea or the run would fail. A syntax or import check cannot.
- `trial_result_used` [true]: the result is what the agent goes ahead, adjusts or cancels by.
- Steps: `trial_step` (RUN) < `launch_step` (RUN).

## Output

One JSON object per line and nothing else, a line for every pattern of every place:

    {"moment": "m012", "pattern": "C1", "answers": {"as_expected": false, "message_gives_cause": false, "ran_a_check": true, "check_can_answer": true, "changed_before_check": false, "stated_finding": true}, "steps": {"result_step": 113, "check_step": 115, "finding_step": 118}, "start": 115, "end": 118, "source": "local", "why": "at most 25 words", "quote": "at most 25 words copied from the log"}

`answers` holds every question of the pattern. When the answers do not describe the behaviour: `"steps": {}`, `"start": null`, `"end": null`.
