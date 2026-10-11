# Timeline protocol for the whole-run reader (draft 2, 2026-10-11)

You will read the complete log of one run: an agent working alone for hours on one task that a judge scores. Write down what happened as a timeline, and point at the moments listed below. You do not judge whether the agent did well. A second reader will look at every moment you report and judge it there. A moment you leave out is never looked at, so leave none out; an extra moment costs little.

## What you read

- Every step has a number (`#N`) and is one reply of the agent. Its head says `RUN` when the reply ran a command and `SAY` when it did not. Inside a step, `THINK` is the agent's reasoning, `SAY` is what it wrote, and `$` is the command with the output that came back. The reasoning and the text of a step were written before its command ran: they react to the output of earlier steps, not to their own. Long texts are cut in the middle.
- The agent submits with `sforge-submit`. The judge returns a score, sometimes with details. Only the best submission counts. A harness also submits the current files every 30 minutes (`auto-N`); the agent is not shown those results.
- `JUDGED ROUNDS` at the top lists the judge's own records. A `[judge record ...]` note on a step was added by us. The agent knows only what its command printed.

## What you write

One JSON object and nothing else:

    {"phases": [...], "directions": [...], "moments": [...]}

### phases

5 to 15 entries that cover the whole run in order:

    {"steps": [first, last], "what": "one sentence: what the agent worked on and what came of it"}

### directions

A direction is one kind of change the agent tried in order to raise the score: one algorithm idea, one component, one family of parameters. It is not the whole task and not a single edit. List every direction in the order it first appears:

    {"id": "d1", "name": "at most 10 words",
     "attempts": [{"steps": [first, last], "what": "what was changed, at most 15 words",
                   "measured": "judge" | "local" | "none",
                   "result": "better" | "same" | "worse" | "broken" | "unknown",
                   "number": "the measured value and what it is compared with, as printed"}],
     "end": "kept" | "dropped" | "open"}

- An attempt is one change of that kind together with the measurement that tells whether it helped. `same` covers differences the agent itself treats as noise.
- `kept`: it is part of the final solution. `dropped`: the agent stopped working on it and it is not in the final solution. `open`: still being worked on when the log ends.

### moments

Report every moment of the types below. Each is

    {"type": "...", "step": N, "steps": [first, last], "direction": "dK" or null,
     "what": "one sentence in your own words", "quote": "at most 25 words copied from the log"}

`step` is the single step where the moment begins. `steps` is the range a reader should look at to see what the agent did about it: usually from `step` to 10 to 30 steps later, shorter when another moment starts.

1. `unexpected`: a result that does not fit what the agent expected of its solution or of the judge. The score or a local number got worse after a change meant to improve it. Nothing changed where a change was expected. Two measurements of what should be the same thing differ a lot (judge and local, or two runs of one version). A test of its own idea fails (a check against a slow reference, a stress test). A zero or a failure from the judge that it had not predicted. The result may come from the judge or from a local run. `step` is where the result appears. Report it whether or not the agent comments on it.
   Do not report a plain slip: a syntax error, a missing name or file, a wrong command, a crash whose message says what is wrong and which is fixed in the next step or two. Do not report a result that only continues a trend the agent has already seen.

2. `second_failure`: the step at which one direction has failed for the second time, that is, where its second attempt with result `same`, `worse` or `broken` becomes known. One moment per direction: the second failure, not each later one. `steps` runs from there until the agent is clearly working on something else, or 40 steps, whichever comes first. Add `"earlier": N`, the step of the first failed attempt.
   Report it one level up as well. Several directions often rest on one premise (for example "a better fit on the public data will raise the judge's score", or "the remaining failures are in component X"). When results have contradicted such a premise twice, report one more `second_failure` with `"direction": null` and `"premise": "the premise, at most 15 words"`, at the step of the second contradiction. Add `"left_at": N`, the step where the agent first works on something that does not rest on the premise, or `null` if it never does.

3. `set_aside`: the agent names an idea and decides not to test it, giving an estimate or a doubt ("small gain", "too risky", "too complex", "no time", "later", "not worth it"). An item that is simply next on its plan does not count. Add `"reopened_at": N` if the agent later does test this idea (the step where it starts), else `"reopened_at": null`.

4. `expensive_start`: the agent commits a large amount of time or work in one go. A run that takes five minutes or longer (training, a large search, a full build, a long optimisation), or a batch of many runs. Or the start of a large piece of implementation (a new algorithm, a rewrite of a large part of the solution) that rests on an idea it has not yet checked. `step` is the command or the first edit that starts it. `steps` begins early enough to show any small trial made before it (a short run, a small input, a timing test, a check of the idea on tiny cases).
   A submission is not an `expensive_start`, however long the judge takes. A test suite or benchmark of a minute or two is not either.

5. `record`: the agent writes down what it tried and what resulted, in a place it can read later (a notes file, a results table, a log it keeps by hand, a table of experiments in its own text), or it reads such a record back. Say in `what` whether it is a write or a read-back. The judge's submission list, a saved output file, a backup copy of a file and a final report for the user are not records.

6. `redo`: the agent repeats a local experiment or a change it already tried (the same change measured the same way), apparently without knowing the earlier result, or it rebuilds something it had already built and lost. Add `"earlier": N`. Submitting unchanged files again is not a `redo`.

7. `probe`: the agent makes a submission that is built to answer a question about the judge or the hidden data (how a score is computed, where a limit lies, which of two explanations is right), not to score. The mark of a probe is that the submitted version differs from the agent's best attempt on purpose, or that the agent says what each outcome would tell it. `step` is where it states the question or starts building the probe; `steps` runs through the step where it reads the answer. A normal submission of the current solution is not a probe, even when the agent says it wants to see the score or the details.

8. `restore`: the agent puts the solution back to an earlier version because a later version scored or measured worse, or so that the files a submission would take are its best version. Undoing a broken edit a step or two after making it is not a `restore`.

## Rules

- A step number must be a number that appears in the log. Check each one before you write it.
- Copy quotes exactly.
- Do not skip a moment because it looks unimportant or because the agent handled it well. Both outcomes are needed.
- If a type never occurs, report none of that type. Invent nothing.
- No text outside the JSON object.
