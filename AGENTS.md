# Working notes

- Read `README.md` first. Rollouts, the bridge and evaluation are in the EdgeBench repository; do not copy them here.
- One row is one model reply. Export keeps every reply; selection only writes manifests and never copies or edits rows.
- A label is binary. Change what counts as a pattern in `patterns/label-sheet.md` and `EVIDENCE` in
  `data_pipeline/label.py` together; `data_pipeline/test_label.py` must pass.
- Pattern and sheet texts contain no task detail. Held-out tasks are not used for labelling data, training or tuning.
- Stages that call the labelling model must stay standard-library only (the cluster has no internet), resume by
  skipping finished packets, and record the model and the sheet's SHA-256 with each answer.
- Kimi addresses come from `/personal/kimi.env`; the endpoints are shared, keep to the concurrency the user gives.
- `training/` follows the conventions of physicalagent-sft-pipeline (policy, manifest and recipe are separate; model,
  data root and output paths are given on the command line). Keep it minimal: no image or robot-task code.
- Data, weights, outputs and generated job files are not committed.
