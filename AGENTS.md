# Working notes

- Read `README.md` first: it lists the stages, the command of each and its arguments. Keep it current when a
  command or an argument changes. Rollouts, the bridge and evaluation are in the EdgeBench repository
  (`/ssd0/foke/EdgeBench`, branch `lcy-dev`); do not copy them here.
- Two entries. `data_pipeline/annotate.py --config configs/labelling/<x>.yaml` is the offline labelling by the served
  model; it is done once per rollout and its label files are reused. `data_pipeline/select_rows.py --policy
  configs/data_policies/<x>.yaml` turns labels into a manifest without calling a model. What differs between
  selections goes into the policy file, not into new scripts or flags. Selecting must never trigger labelling.
  Step N of a transcript is row N of the export; never renumber steps.
- The README is in Chinese and short: what the stages are, the one command, its arguments, the policy keys.
- One row is one model reply. Export keeps every reply; selection only writes manifests and never copies or edits rows.
- A label is binary. Change what counts as a pattern in `patterns/label-sheet.md` and `EVIDENCE` in
  `data_pipeline/label.py` together; `data_pipeline/test_label.py` must pass.
- Pattern and sheet texts contain no task detail. Held-out tasks are not used for labelling data, training or tuning.
- Stages that call the labelling model must stay standard-library only (the cluster has no internet), resume by
  skipping finished packets, and record the model and the sheet's SHA-256 with each answer.
- Kimi addresses come from `/personal/kimi.env`; the endpoints are shared, keep to the concurrency the user gives
  (2 requests per address as of 2026-10-11). Check `http://<host>:30000/get_load` before a large run.
- Remote work: the cluster has no internet. Code reaches it by git, other files through OSS. Jobs longer than a
  minute run in tmux with logs on disk. A waiting loop on a tmux session must use the exact name
  (`tmux has-session -t =name`); a bare name also matches longer names.
- The tool call ids of Kimi (`exec_command:0`) are numbered per conversation, not unique across rollouts. Anything
  that keys on them must also key on the rollout.
- `training/` follows the conventions of physicalagent-sft-pipeline (policy, manifest and recipe are separate; model,
  data root and output paths are given on the command line). Keep it minimal: no image or robot-task code.
- Data, weights, outputs and generated job files are not committed.
