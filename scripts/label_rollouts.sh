#!/bin/bash
# Stages 3 to 6 for every rendered rollout under WORK: three readings, places, labels, one labels file.
#   label_rollouts.sh WORK LABELS.jsonl
# Kimi addresses: KIMI_URLS="http://host:30000/v1 ..." or, when unset, KIMI_BASE_URL_1..4 from /personal/kimi.env.
# PER_URL (default 2) is the number of concurrent labelling requests per address. Finished readings and packets are
# skipped, so the script can be run again after an interruption. Logs go to WORK/logs.
set -u
work=$(realpath "$1"); labels=$2
here=$(cd "$(dirname "$0")/.." && pwd)
if [ -z "${KIMI_URLS:-}" ]; then
  source /personal/kimi.env
  KIMI_URLS="$KIMI_BASE_URL_1 ${KIMI_BASE_URL_2:-} ${KIMI_BASE_URL_3:-} ${KIMI_BASE_URL_4:-}"
fi
urls=($KIMI_URLS)
label="python3 $here/data_pipeline/label.py"
mkdir -p "$work/logs"
dirs=()
for d in "$work"/*/; do
  [ -f "$d/transcript.txt" ] || continue
  dirs+=("$d"); name=$(basename "$d")
  # the timeline first: the other two readings of a long log use it for what happened in the earlier parts
  $label read "$d" --protocol "$here/patterns/timeline-protocol.md" --url "${urls[0]}" --name timeline >> "$work/logs/$name-read.log" 2>&1
  $label read "$d" --protocol "$here/patterns/ideas-protocol.md" --url "${urls[0]}" --name ideas --context timeline.json >> "$work/logs/$name-read.log" 2>&1 &
  $label read "$d" --protocol "$here/patterns/surprises-protocol.md" --url "${urls[1 % ${#urls[@]}]}" --name surprises --context timeline.json >> "$work/logs/$name-read.log" 2>&1 &
  wait
  [ -f "$d/places.json" ] || $label places "$d" --timeline ideas.json surprises.json timeline.json >> "$work/logs/$name-read.log" 2>&1
done
$label judge "${dirs[@]}" --sheet "$here/patterns/label-sheet.md" --url "${urls[@]}" --per-url "${PER_URL:-2}" --name judged >> "$work/logs/judge.log" 2>&1
$label labels "${dirs[@]}" --name judged --out "$labels" | tee -a "$work/logs/judge.log"
