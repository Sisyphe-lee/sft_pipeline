#!/usr/bin/env python3
"""Turn the model exchanges a bridge captured into SFT rows: one JSONL file per rollout, one row per model reply.

Used by clean.py; on its own: export_capture.py CAPTURE_DIR [CAPTURE_DIR ...] --out DIR [--only SUBSTRING]

CAPTURE_DIR holds `resp_<id>/{request,upstream-request,upstream-response}.json.gz` as written by EdgeBench's
`scripts/kimi_responses_bridge.py --capture-dir`. Exchanges are grouped by the capture id the harness sent
(`sforge.run.<task>.<run id>`) and ordered by time. A row is

    {"prompt": [chat messages the model saw], "completion": {the assistant message it returned}, "tools": [...],
     "metadata": {"task": {"id": ...}, "trajectory_id": ..., "turn": 1.., "time": epoch seconds, ...}}

Every reply is kept, whatever the run's outcome; choosing rows is the job of the selection step. Earlier turns in
`prompt` get their reasoning text back from the replies captured before (captures made before the bridge kept
reasoning in the history lack it). Standard library only.
"""
import argparse
import gzip
import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def turn_key(message):
    """Identifies an assistant turn inside one rollout: its first tool call, else its text."""
    calls = message.get("tool_calls") or []
    if calls:
        function = calls[0].get("function") or {}
        text = json.dumps([calls[0].get("id"), function.get("name"), function.get("arguments")], ensure_ascii=False)
    else:
        content = message.get("content")
        text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


def load(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def scan(roots, only):
    """{capture id: [(time, exchange dir)]}. Reads every request once, in threads: a capture tree holds tens of thousands."""
    def one(path):
        try:
            request = load(os.path.join(path, "request.json.gz"))
        except (OSError, ValueError):
            return None
        return request.get("capture_id") or "unknown", request.get("timestamp") or 0, path

    paths = [entry.path for root in roots for entry in os.scandir(root) if entry.name.startswith("resp_")]
    found = {}
    with ThreadPoolExecutor(16) as pool:
        for hit in pool.map(one, paths):
            if hit and (not only or only in hit[0]):
                found.setdefault(hit[0], []).append(hit[1:])
    return found


def identity(ident):
    """(task, trajectory id) of a capture id `sforge.run.<task>.<run id>`."""
    match = re.match(r"sforge\.run\.([^.]+)\.(.+)$", ident)
    return (match.group(1), match.group(2)) if match else (ident, ident)


def export(ident, exchanges, out):
    task, run = identity(ident)
    reasoning, rows, skipped, before = {}, 0, {}, None
    target = out / f"{run}.jsonl"
    with open(target.with_suffix(".tmp"), "w", encoding="utf-8") as stream:
        for began, path in sorted(exchanges):
            try:
                sent = load(os.path.join(path, "upstream-request.json.gz"))
                answer = load(os.path.join(path, "upstream-response.json.gz"))
                if answer.get("status_code") != 200:
                    raise ValueError(f"status {answer.get('status_code')}")
                body = json.loads(answer["body"])
                choice = body["choices"][0]
                message = choice["message"]
            except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
                reason = type(error).__name__ if not isinstance(error, ValueError) else str(error)[:40]
                skipped[reason] = skipped.get(reason, 0) + 1
                continue
            prompt = sent["messages"]
            for earlier in prompt:
                if earlier.get("role") == "assistant" and not earlier.get("reasoning_content") and turn_key(earlier) in reasoning:
                    earlier["reasoning_content"] = reasoning[turn_key(earlier)]
            completion = {"role": "assistant", "content": message.get("content") or ""}
            if isinstance(message.get("reasoning_content"), str) and message["reasoning_content"].strip():
                completion["reasoning_content"] = message["reasoning_content"]
                reasoning[turn_key(message)] = message["reasoning_content"]
            if message.get("tool_calls"):
                completion["tool_calls"] = message["tool_calls"]
            rows += 1
            usage = body.get("usage") or {}
            # the history normally grows by this turn and its tool results; a shorter one means the scaffold rewrote it (compaction)
            context = "start" if before is None else ("rewritten" if len(prompt) < before else "continued")
            before = len(prompt)
            row = dict(prompt=prompt, completion=completion, tools=sent.get("tools") or [], metadata=dict(
                task=dict(id=task, seed=0), trajectory_id=run, turn=rows, time=began, seconds=round((answer.get("timestamp") or began) - began, 2),
                exchange=os.path.basename(path), finish=choice.get("finish_reason"), context=context,
                kind="turn" if sent.get("tools") else "no_tools", reasoning_effort=sent.get("reasoning_effort"),
                usage={k: usage.get(k) for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens")}))
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(target.with_suffix(".tmp"), target)
    return dict(trajectory_id=run, task=task, file=target.name, rows=rows, skipped=skipped)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("capture", nargs="+")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--only", help="export only capture ids that contain this text")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for ident, exchanges in sorted(scan(args.capture, args.only).items()):
        print(json.dumps(export(ident, exchanges, args.out), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
