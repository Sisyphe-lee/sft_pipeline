# Working notes

- Read `README.md` first: it lists the stages, the command of each and its arguments. Keep it current when a
  command or an argument changes. Rollouts, the bridge and evaluation are in the EdgeBench repository
  (`/ssd0/foke/EdgeBench`, branch `lcy-dev`); do not copy them here.
- Two entries. `data_pipeline/annotate.py --config configs/labelling/<x>.yaml` is the offline preparation: export,
  length measuring with the student's tokenizer (`--student`) and labelling by the served model; it is done once per
  rollout and its files are reused. `data_pipeline/select_rows.py --policy configs/data_policies/<x>.yaml` turns
  labels into a manifest without calling a model. What differs between selections goes into the policy file, not
  into new scripts or flags. Selecting must never trigger labelling. Step N of a transcript is row N of the export;
  never renumber steps.
- Rows are stored in the shape the trainer and the student's chat template take (`export_capture.for_student`: one
  system message, tool call arguments as objects, `completion` a one-message list, `images: []`). The bridge that
  serves the student must do the same to live requests (`--single-system` in EdgeBench's bridge). After any change
  to the export, run `training/tools/check_collate.py` on a manifest: it uses the trainer's own reader and collator.
- A row can carry loss only if the trainer accepts it: selection leaves out replies without reasoning (the collator's
  prefix check fails on them), rows `measure_rows.py` marks not ok, and rows over the policy's `max_tokens`. The
  control arm draws from the same usable rows.
- The README is in Chinese and short: what the stages are, the one command, its arguments, the policy keys.
- One row is one model reply. Export keeps every reply; selection only writes manifests and never copies or edits rows.
- A label is binary and decided by the program (`decide` in `data_pipeline/label.py`), not by the model.
  A pattern is one file, `patterns/<name>.yaml`. One request asks about neighbouring places and all their patterns
  (`requests` in `label.py`), but every answer is stored by place and pattern under the pattern's fingerprint. Keep
  patterns decoupled: nothing in one pattern file may depend on another, and editing one must not invalidate the
  labels of the others. `patterns/common.md` is shared; editing it relabels everything, so change it rarely. Prefer
  a program rule (`skip`, `within`, `apart`, `span`) to a new question when it does the job: it costs no relabelling. A stored answer records the place it was given for, and
  place ids are `<type>-<step>`, so answers survive places being found anew.
- Check label quality by reading sampled labels against the log, not by counts. `Kimi` is lenient: a question that
  asks two things gets a yes when the first half holds, and "only the judge can tell" was read as true of every
  submission. Ask one thing per question and say what does not count.
- Labelling time is Kimi's output tokens, almost all reasoning (about 100 tokens a second per request), not the
  number of requests; measure it in slot-minutes from `work/<rollout>/requests/`. Measured on 2026-10-12: grouping
  saves 2.4x, `reasoning_effort: low` another 3x (Kimi takes only low, high and max; none switches thinking off).
  Telling the model to stop at the first contrary answer does not shorten its reasoning. Readings at low effort
  report a third fewer places. B1 needs high effort. Check any further speed-up against stored answers of the same
  questions, and against the same setting asked twice: two runs of one setting already differ in about a fifth of
  the labels, and a label only one run gives is wrong about twice as often as one both give.
- Pattern and sheet texts contain no task detail. Held-out tasks are not used for labelling data, training or tuning.
- Stages that call the labelling model must stay standard-library only (the cluster has no internet), resume by
  skipping finished packets, and record the model and the sheet's SHA-256 with each answer.
- Kimi addresses come from `/personal/kimi.env`; the endpoints are shared, keep to the concurrency the user gives
  (2 requests per address as of 2026-10-12). Check `http://<host>:30000/get_load` before a large run.
- Remote work: the cluster has no internet. This repository has no remote yet: it reaches the cluster as a tar
  through OSS (EdgeBench `scripts/transfer_via_oss.py`), other files likewise. Jobs longer than a
  minute run in tmux with logs on disk. A waiting loop on a tmux session must use the exact name
  (`tmux has-session -t =name`; `send-keys` needs `-t '=name:'`); a bare name also matches longer names. `grep`
  between a job and its log file buffers the output: use `grep --line-buffered`.
- The tool call ids of Kimi (`exec_command:0`) are numbered per conversation, not unique across rollouts. Anything
  that keys on them must also key on the rollout.
- `training/` follows the conventions of physicalagent-sft-pipeline (policy, manifest and recipe are separate; model,
  data root and output paths are given on the command line). Keep it minimal: no image or robot-task code.
- Data, weights, outputs and generated job files are not committed.
