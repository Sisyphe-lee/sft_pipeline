# sft-pipeline

SFT data and training for the "Learn from Experience" project. Rollouts of a source model (Kimi-K3 under Codex) on
EdgeBench tasks are turned into training rows; a labelling model marks the steps that show a behaviour pattern; the
student is trained with loss on those steps only. The control is the same number of random steps from the same rollouts.

Rollouts, the Responses bridge that captures the model exchanges, and the evaluation of a checkpoint are in the
EdgeBench repository. This repository starts from the captured exchanges.

## Layout

| Path | What |
|---|---|
| `data_pipeline/export_capture.py` | stage 1: captured exchanges → one row per model reply |
| `data_pipeline/rollout.py` | stage 2: rows → numbered transcript and rule-located places |
| `data_pipeline/label.py` | stages 3–6: readings, places, labels (`read`, `places`, `judge`, `labels`) |
| `data_pipeline/select_rows.py` | stage 7: labels → training manifests (pattern arm, control arm) |
| `data_pipeline/manifest.py` | manifest format shared with the trainer |
| `scripts/label_rollouts.sh` | runs stages 3–6 for every rendered rollout in a directory |
| `patterns/label-sheet.md` | the 12 patterns as yes/no questions and steps: what the labelling model is asked |
| `patterns/{timeline,ideas,surprises}-protocol.md` | instructions for the three readings |
| `training/`, `jobs/`, `configs/` | trainer, job templates and recipes (copied from physicalagent-sft-pipeline, see State) |

## Directories used by a run

Nothing here is committed. Three directories, anywhere on disk:

- `ROWS/`: `<trajectory id>.jsonl`, written by stage 1. This is the training data; it is never edited afterwards.
- `WORK/<trajectory id>/`: everything the labelling of one rollout needs and produces (stages 2–5).
- `MANIFESTS/`: the selection manifests (stage 7), which point into `ROWS/` by relative path.

A trajectory id is the run id the harness gave the rollout, for example `full50-max-20260909-230641-borden_source_inversion`.

## Stages

Stages 1, 2, 6 and 7 need only Python's standard library. Stages 3 and 5 call an OpenAI-compatible chat endpoint of
the labelling model and are standard library only too, so they run on the cluster, which has no internet.

### 1. Export

    python3 data_pipeline/export_capture.py CAPTURE_DIR [CAPTURE_DIR ...] --out ROWS [--only TEXT]

- `CAPTURE_DIR`: a bridge capture directory with `resp_<id>/{request,upstream-request,upstream-response}.json.gz`
  (EdgeBench `scripts/kimi_responses_bridge.py --capture-dir`). Several directories are merged (one per bridge).
- `--only TEXT`: export only rollouts whose capture id contains TEXT (a task name or a run id).
- Output: `ROWS/<trajectory id>.jsonl` and one summary line per rollout on stdout (`rows`, `skipped`).

A row is one model reply:

    {"prompt": [chat messages the model saw],
     "completion": {"role": "assistant", "content": ..., "reasoning_content": ..., "tool_calls": [...]},
     "tools": [...],
     "metadata": {"task": {"id": ..., "seed": 0}, "trajectory_id": ..., "turn": 1.., "time": epoch seconds,
                  "seconds": ..., "exchange": "resp_...", "finish": ..., "context": "start|continued|rewritten",
                  "kind": "turn|no_tools", "reasoning_effort": ..., "usage": {...}}}

Every reply with HTTP status 200 is kept, whatever the outcome of the run. `context: rewritten` marks the first reply
after the scaffold replaced the conversation with a summary. Earlier turns inside `prompt` get their reasoning text
back from the replies captured before them. One rollout of 200 replies is about 80 MB.

### 2. Render

    python3 data_pipeline/rollout.py ROWS/<t>.jsonl --run-dir TASK_DIR --tasks-dir TASKS --out WORK/<t> [--cap 3000]

- `--run-dir`: the harness's directory of this run (`.../harness/runs/<run id>/<task>/` with `run_history.json`,
  `submissions/`, `final_result.json`). Without it the transcript has no judge records and no submission places.
- `--tasks-dir`: EdgeBench task specifications (`<task>.json`), for the score direction. Default: maximise.
- `--cap`: characters kept of one reasoning text, command or output.
- Output: `transcript.txt` (a head, a line of `=`, the steps), `head.txt`, `points.json`.

Step `#N` is row N: its reasoning (`THINK`), text (`SAY`), command (`$`) and the output that came back. `points.json`
holds the places a rule finds: `start`, each `submit` command of the agent, the step where its `result` is read,
and `end`.

### 3–6. Readings, places, labels

All four at once, for every rendered rollout under WORK:

    KIMI_URLS="http://host1:30000/v1 http://host2:30000/v1" PER_URL=2 scripts/label_rollouts.sh WORK labels.jsonl

`KIMI_URLS` defaults to `KIMI_BASE_URL_1..4` from `/personal/kimi.env`. `PER_URL` is the number of concurrent
labelling requests per address. Finished readings and packets are skipped on a second run. Logs: `WORK/logs/`.
On the cluster run it inside tmux. The single steps:

    python3 data_pipeline/label.py read WORK/<t> --protocol patterns/timeline-protocol.md --url URL --name timeline
    python3 data_pipeline/label.py read WORK/<t> --protocol patterns/ideas-protocol.md --url URL --name ideas --context timeline.json
    python3 data_pipeline/label.py read WORK/<t> --protocol patterns/surprises-protocol.md --url URL --name surprises --context timeline.json

- The labelling model reads the whole transcript and reports places the rules cannot find (an unexpected result, a
  second failure of one direction, an idea set aside, an expensive start, a record written or read, a probe).
- `--url`: base URL ending in `/v1`; the model id is taken from `/v1/models`.
- `--name`: the output file `WORK/<t>/<name>.json`; `<name>.raw.json` keeps the raw answer and the reasoning.
- `--segment-chars` (1200000): a longer transcript is read in parts of about this size; `--context` gives the later
  parts another reading's account of the earlier ones.
- `--max-tokens` (40000), `--tries` (3), `--temperature` (0), `--force` (read again although the output exists).

      python3 data_pipeline/label.py places WORK/<t> --timeline ideas.json surprises.json timeline.json [--batch 1]

- Merges the rule places and the readers' places into `places.json` and cuts `packets/NNNN.json`, `--batch` places per
  request. Each place names the patterns to look for there.

      python3 data_pipeline/label.py judge WORK/<t> [WORK/<u> ...] --sheet patterns/label-sheet.md --url URL [URL ...] --per-url 2 --name judged

- One request per packet: the sheet, the head, `timeline.json`, the steps around the place (`--window-chars`, 150000)
  and the packet. The model answers the sheet's questions and names steps; it gives no verdict.
- `label.py` then decides: a line becomes `"verdict": "done"` only when every answer has the value the sheet asks for,
  the steps are given in the required order, the steps that must be commands are `RUN` steps, a submit step is a real
  submit command, and the steps lie inside the span. Otherwise `"verdict": "no"` with the reasons in `gaps`.
- Output: `WORK/<t>/judged/NNNN.jsonl` and `judged/logs/NNNN.raw.json` (model, usage, sheet SHA-256, raw answer).
- `--max-tokens` (30000, raised by half after an answer that was cut), `--tries` (3), `--timeline` (timeline.json).

      python3 data_pipeline/label.py labels WORK/* --name judged --out labels.jsonl

- One record per labelled span: `task`, `trajectory_id`, `pattern`, `start`, `end`, `steps`, the place it came from,
  the model's `answers` and `evidence` steps, `model`, `sheet_sha256`.

### 7. Select

    python3 data_pipeline/select_rows.py --rows ROWS --labels labels.jsonl --out MANIFESTS/<name> [--patterns C1 D3 ...] [--seed 0]

- Output: `MANIFESTS/<name>-pattern.json` (every row a label covers, once) and `MANIFESTS/<name>-control.json` (from
  each rollout the same number of rows, drawn at random among its ordinary turns with `--seed`).
- `--patterns`: use only these patterns.
- A manifest lists rows by source file, byte offset and length; `selection` records the counts per pattern and per
  rollout and the SHA-256 of the labels file. Rows that are not selected still appear as history inside later rows.

### 8. Train

`training/train_sft.py` reads a manifest and a recipe (`configs/recipes/`); jobs are generated from `jobs/templates/`
and submitted with `robbyctl`. See State: this part is copied and not yet adapted.

## Labels

A label is binary: a step is selected for a pattern or it is not. The twelve patterns are A1, L1, A4, A2, C1, A3,
B1, D4, D1, D2, D3, C2; `patterns/label-sheet.md` defines each by its questions. To change what counts, edit the sheet
and `EVIDENCE` in `data_pipeline/label.py` together; `python -m pytest -q data_pipeline/test_label.py` checks that
they agree.

## State (2026-10-11)

- Stages 1, 2 and 7 were run on one captured Kimi rollout (212 rows); the manifests load with `data_pipeline/manifest.py`.
- Stages 3 to 6 are ported from EdgeBench's `scripts/pattern_timeline.py`, where they were tried on seven rollouts
  against Opus judgements (EdgeBench `patterns/trial-report.md`).
- `training/`, `jobs/`, `configs/` are copied unchanged from physicalagent-sft-pipeline, branch `feat/flashnext-sft`,
  commit `dc7854c`. Not yet done: removing the image and robot-task parts (`training/vlm_collator.py` imports
  `physical_agent.artifact_refs`, which is not copied, so the trainer does not import yet), and a first run. No
  environment with the training dependencies exists on the workstation, so the `training/` tests were not run here.
- The Kimi rollouts captured in September 2026 serve to develop the pipeline but not to train: the model hardly
  reasoned in them (EdgeBench `docs/kimi-serving.md`). Training rollouts need the bridge with reasoning history.
- Not written yet: choosing rollouts by final result for the control arm, a validation split, and an audit step
  that samples labelled rows for a second opinion.
