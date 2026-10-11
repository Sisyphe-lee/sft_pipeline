# sft-pipeline

SFT data and training for the "Learn from Experience" project: take rollouts of a source model on EdgeBench tasks,
label the steps that show a behaviour pattern, and train a student with loss on those steps only. The control is the
same number of random steps from the same rollouts.

Rollouts are made in the EdgeBench repository (harness, Responses bridge, capture). This repository starts from the
captured model exchanges.

## Stages

| # | Stage | Command | Output |
|---|---|---|---|
| 1 | Export | `data_pipeline/export_capture.py CAPTURE_DIR --out ROWS` | `ROWS/<trajectory>.jsonl`: one row per model reply (`prompt`, `completion`, `tools`, `metadata`) |
| 2 | Render | `data_pipeline/rollout.py ROWS/<t>.jsonl --run-dir TASK_DIR --tasks-dir DIR --out WORK/<t>` | `transcript.txt` (step N = row N), `points.json` (start, submissions, results, end) |
| 3 | Read | `data_pipeline/label.py read WORK/<t> --protocol patterns/<x>-protocol.md --url URL --name <x>` (timeline, ideas, surprises) | `<x>.json`: a reader model's account and the places it reports |
| 4 | Places | `data_pipeline/label.py places WORK/<t> --timeline ideas.json surprises.json timeline.json` | `places.json`, `packets/` |
| 5 | Label | `data_pipeline/label.py judge WORK/<t> --sheet patterns/label-sheet.md --url URL ... --name judged` | per place: the model's answers and steps; a span is labelled only when they satisfy the sheet |
| 6 | Collect | `data_pipeline/label.py labels WORK/* --name judged --out labels.jsonl` | one record per labelled span |
| 7 | Select | `data_pipeline/select_rows.py --rows ROWS --labels labels.jsonl --out MANIFESTS/<name>` | `<name>-pattern.json`, `<name>-control.json` (trainer manifests; rows by file and byte offset) |
| 8 | Train | `training/train_sft.py` with a recipe from `configs/recipes/` and a manifest | checkpoints |

Stages 1, 2, 6 and 7 need only Python's standard library and run anywhere. Stages 3 and 5 call an OpenAI-compatible
endpoint of the labelling model (Kimi-K3 on the cluster; addresses from `/personal/kimi.env`) and are also standard
library only. Evaluation of a checkpoint is an EdgeBench run and lives in that repository.

A label is binary: a step is selected for a pattern or it is not. `patterns/label-sheet.md` asks the model yes/no
questions and step numbers per pattern; `label.py` (`EVIDENCE`) decides. `data_pipeline/test_label.py` checks that the
sheet and the program agree.

## State (2026-10-11)

- Stages 1, 2 and 7 were run on one captured Kimi rollout (212 rows); the manifests load with `data_pipeline/manifest.py`.
- Stages 3 to 6 are ported from EdgeBench's `scripts/pattern_timeline.py`, where they were tried on seven rollouts.
  They have not yet been run on a transcript rendered by stage 2.
- `training/`, `jobs/`, `configs/` are copied unchanged from physicalagent-sft-pipeline, branch `feat/flashnext-sft`,
  commit `dc7854c`. Not yet done: removing the image and robot-task parts (`training/vlm_collator.py` imports
  `physical_agent.artifact_refs`, which is not copied, so the trainer does not import yet), and a first run.
  No environment with the training dependencies exists on the workstation, so `training/` tests were not run here.
- The Kimi rollouts captured in September 2026 can be used to develop the pipeline but not to train: the model
  hardly reasoned in them (see EdgeBench `docs/kimi-serving.md`).

Tests: `python -m pytest -q data_pipeline/test_label.py`.
