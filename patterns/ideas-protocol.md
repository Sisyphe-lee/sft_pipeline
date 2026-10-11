# Ideas-and-records protocol for the whole-run reader, second pass (draft 1, 2026-10-11)

You will read the complete log of one run: an agent working alone for hours on one task that a judge scores. This pass looks for three things only: what the agent named and did not do, the records it kept, and work it repeated. You do not judge whether the agent did well. A second reader will look at every moment you report. A moment you leave out is never looked at, so leave none out; an extra moment costs little.

## What you read

- Every step has a number (`#N`) and is one reply of the agent. Its head says `RUN` when the reply ran a command and `SAY` when it did not. Inside a step, `THINK` is the agent's reasoning, `SAY` is what it wrote, and `$` is the command with the output that came back. The reasoning and the text of a step were written before its command ran: they react to the output of earlier steps, not to their own. Long texts are cut in the middle.
- The agent submits with `sforge-submit`. The judge returns a score, sometimes with details. Only the best submission counts. A harness also submits the current files every 30 minutes (`auto-N`); the agent is not shown those results.
- `JUDGED ROUNDS` at the top lists the judge's own records. A `[judge record ...]` note on a step was added by us. The agent knows only what its command printed.

## How to read

Go through everything the agent wrote, in order: every `SAY`, every `THINK`, and the comments inside its commands. Do not skim the long middle of the run; most of what this pass looks for is there.

## What you write

One JSON object and nothing else:

    {"moments": [...]}

Each moment is

    {"type": "...", "step": N, "steps": [first, last], "what": "one sentence in your own words", "quote": "at most 25 words copied from the log"}

with the extra fields named below.

1. `set_aside`: the agent names something it could do and then does not do it in the steps that follow. It may be an idea for raising the score, a different method, a check, or a test of one of its own assumptions. It may give a reason ("small gain", "too risky", "too complex", "no time", "later", "circular", "but first ...") or give none and simply go on with something else. Something it does within the next ten steps does not count; a list of next steps that it then works through does not count.
   - `step`: where it names the thing. `steps`: from there to the step where it turns to something else.
   - `"kind"`: `"check"` if the thing would test an assumption or explain a result; `"change"` if it would be a change to the solution.
   - `"reason"`: its reason, copied, or `"none"`.
   - `"reopened_at"`: the step where it later does the thing after all, or `null`.

2. `record`: the agent writes down what it tried and what resulted, in a place it can read later (a notes file, a results table, a log it keeps by hand, a table of experiments in its own text), or it reads such a record back. Say in `what` whether it is a write or a read-back, and what the record holds. The judge's submission list, a saved output file, a backup copy of a file and a final report for the user are not records.

3. `redo`: the agent repeats a local experiment or a change it already tried (the same thing measured the same way), apparently without knowing the earlier result, or it rebuilds something it had already built and lost. Add `"earlier": N`. Submitting unchanged files again is not a `redo`.

## Rules

- A step number must be a number that appears in the log. Check each one before you write it.
- Copy quotes exactly.
- Report a thing each time it is named and dropped, even if it was dropped before.
- If a type never occurs, report none of that type. Invent nothing.
- No text outside the JSON object.
