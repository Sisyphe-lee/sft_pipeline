# Unexpected-results protocol for the whole-run reader, third pass (draft 1, 2026-10-11)

You will read the complete log of one run: an agent working alone for hours on one task that a judge scores. This pass looks for one thing only: results that did not fit what the agent expected. You do not judge whether the agent did well. A second reader will look at every moment you report. A moment you leave out is never looked at, so leave none out; an extra moment costs little.

## What you read

- Every step has a number (`#N`) and is one reply of the agent. Its head says `RUN` when the reply ran a command and `SAY` when it did not. Inside a step, `THINK` is the agent's reasoning, `SAY` is what it wrote, and `$` is the command with the output that came back. The reasoning and the text of a step were written before its command ran: they react to the output of earlier steps, not to their own. Long texts are cut in the middle.
- The agent submits with `sforge-submit`. The judge returns a score, sometimes with details. Only the best submission counts. A harness also submits the current files every 30 minutes (`auto-N`); the agent is not shown those results.
- `JUDGED ROUNDS` at the top lists the judge's own records. A `[judge record ...]` note on a step was added by us. The agent knows only what its command printed.

## How to read

Go through the run step by step. Before each command, note what the agent said it expected from it, or what its plan implies it expected. Then look at the output. Do not skim the long middle of the run.

## What you write

One JSON object and nothing else:

    {"moments": [...]}

Each moment is

    {"type": "unexpected", "step": N, "steps": [first, last], "source": "judge" | "local",
     "expected": "what the agent expected, at most 15 words", "got": "what came, at most 15 words",
     "quote": "at most 25 words copied from the log"}

`step` is where the result appears. `steps` runs from there to the step where the agent has clearly reacted (it checks something, changes something, or moves on), at most 20 steps.

Report a result when it does not fit what the agent expected of its solution or of the judge:

- the score or a local number got worse after a change meant to improve it;
- nothing changed where a change was expected;
- two measurements of what should be the same thing differ a lot (judge and local, or two runs of one version);
- a test of the agent's own idea fails (a check against a slow reference, a stress test, a prediction that does not come true);
- a zero or a failure from the judge that it had not predicted;
- a result much better than expected that contradicts its picture of the problem;
- a run takes far longer than it estimated.

Report it whether or not the agent comments on it. A surprise the agent passes over in silence is the kind most easily missed, so look for those.

Do not report:

- a plain slip: a syntax error, a missing name or file, a wrong command, a crash whose message says what is wrong and which is fixed in the next step or two;
- a result that only continues a trend the agent has already seen;
- a result the agent predicted.

## Rules

- A step number must be a number that appears in the log. Check each one before you write it.
- Copy quotes exactly.
- Invent nothing. No text outside the JSON object.
