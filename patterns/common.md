# Labelling instructions

You read part of the log of one run: an agent working alone for hours on one task that a judge scores. You are given one place in the log and one behaviour pattern to look for there. You answer the pattern's yes/no questions and name the steps that show the behaviour. A program turns your answers into a label: steps are used as training examples of the behaviour only when every answer and every step holds. You do not give a verdict.

A wrong example costs more than a missed one. When you cannot tell, give the answer that speaks against the behaviour, and leave a step `null` when you have not found it.

## The run

- Every step has a number (`#N`) and is one reply of the agent. Its head says `RUN` when the reply ran a command and `SAY` when it did not. Inside a step, `THINK` is the agent's reasoning, `SAY` is what it wrote, and `$` is the command with the output that came back. The reasoning and the text of a step were written before its command ran: they react to the output of earlier steps, not to their own. Long texts are cut in the middle.
- The agent submits with `sforge-submit`. The judge returns a score, sometimes with details. Only the best submission counts. A harness also submits the current files every 30 minutes (`auto-N`); the agent is not shown those results.
- A `[judge record ...]` note on a step was added by us. The agent knows only what its command printed.
- Some tasks are games played through a server: the score is the game's own score and there are no submissions.

## Material

- HEAD: facts about the run and the list of judged rounds.
- TIMELINE: another reader's account of the whole run. Use it to learn what happened outside the steps you see. It can be wrong; the log decides.
- LOG: the steps around the place. When a question needs steps you cannot see, give the answer that speaks against the behaviour.
- PATTERN: the behaviour to look for, as questions and steps.
- PLACE: where to look. It has `id`, `type`, `step`, `steps` (where to start reading) and `facts`. Places of type `start`, `submit`, `result` and `end` were found by rule; the others were reported by the other reader, whose notes are in `facts`.

## What to write

1. Answer every question of the pattern with `true` or `false`. The value in brackets after a question is the one the behaviour needs. The answers must be about this place. If the place was located wrongly (the facts say one thing, the log another), answer by the log.
2. If every answer has the value in brackets, fill in the steps, in the order given. Each is the number of the one step where that thing happens. A step marked RUN must be a `RUN` step: a reply in which the agent ran a command. Otherwise leave the steps out.

Your answer is one JSON object on one line and nothing else:

    {"answers": {"<question>": true, ...}, "steps": {"<step>": 115, ...}, "why": "at most 25 words", "quote": "at most 25 words copied from the log"}

`answers` holds every question of the pattern. When the answers do not describe the behaviour: `"steps": {}`.
