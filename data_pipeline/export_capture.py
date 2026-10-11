#!/usr/bin/env python3
"""Turn the model exchanges a bridge captured into SFT rows: one JSONL file per rollout, one row per model reply.

Used by clean.py; on its own: export_capture.py CAPTURE_DIR [CAPTURE_DIR ...] --out DIR [--only SUBSTRING]

CAPTURE_DIR holds `resp_<id>/{request,upstream-request,upstream-response}.json.gz` as written by EdgeBench's
`scripts/kimi_responses_bridge.py --capture-dir`. Exchanges are grouped by the capture id the harness sent
(`sforge.run.<task>.<run id>`) and ordered by time. A row is

    {"prompt": [chat messages the model saw], "completion": [the assistant message it returned], "tools": [...],
     "images": [], "metadata": {"task": {"id": ...}, "trajectory_id": ..., "turn": 1.., "time": epoch seconds, ...}}

which is what the trainer reads (training/vlm_collator.py). Every reply is kept, whatever the run's outcome; choosing
rows is the job of the selection step. Earlier turns in `prompt` get their reasoning text back from the replies
captured before (captures made before the bridge kept reasoning in the history lack it).

The messages are stored the way the student's chat template accepts them, see `for_student`: the same must be done
to the requests the student receives when it runs under the scaffold. Standard library only.
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


def call_arguments(text):
    """Tool call arguments as an object: chat templates render them key by key. Text that is no JSON object is kept whole."""
    if isinstance(text, dict):
        return text
    try:
        value = json.loads(text) if text else {}
    except ValueError:
        return {"_unparsed": text}
    return value if isinstance(value, dict) else {"_unparsed": text}


def for_student(messages):
    """(messages, changes) as a chat template with one system block accepts them.

    Codex sends its instructions and a permissions note as two system messages: the leading system messages become
    one, a later one becomes a user message. Tool call arguments become objects. Images become a text mark: the rows
    are text only."""
    out, changes, head = [], {}, True
    for message in messages:
        message = dict(message)
        if isinstance(message.get("content"), list):
            parts = []
            for part in message["content"]:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text") or "")
                else:
                    parts.append("[image]")
                    changes["images"] = changes.get("images", 0) + 1
            message["content"] = "\n".join(parts)
        if message.get("role") == "system":
            if head and out:
                out[0]["content"] = f"{out[0]['content']}\n\n{message.get('content') or ''}"
                changes["system_merged"] = changes.get("system_merged", 0) + 1
                continue
            if not head:
                message["role"] = "user"
                changes["system_later"] = changes.get("system_later", 0) + 1
        else:
            head = False
        if message.get("tool_calls"):
            message["tool_calls"] = [dict(call, function=dict(call.get("function") or {}, arguments=call_arguments((call.get("function") or {}).get("arguments"))))
                                     for call in message["tool_calls"]]
            if any("_unparsed" in call["function"]["arguments"] for call in message["tool_calls"]):
                changes["unparsed_calls"] = changes.get("unparsed_calls", 0) + 1
        out.append(message)
    return out, changes


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
            prompt, changes = for_student(prompt)
            completion = for_student([completion])[0][0]
            holds = [name for name, has in (("reasoning", completion.get("reasoning_content")), ("text", completion["content"].strip()),
                                             ("call", completion.get("tool_calls"))) if has]
            if any("_unparsed" in call["function"]["arguments"] for call in completion.get("tool_calls") or []):
                holds.append("unparsed_call")
            # the history normally grows by this turn and its tool results; a shorter one means the scaffold rewrote it (compaction)
            context = "start" if before is None else ("rewritten" if len(prompt) < before else "continued")
            before = len(prompt)
            row = dict(prompt=prompt, completion=[completion], tools=sent.get("tools") or [], images=[], metadata=dict(
                task=dict(id=task, seed=0), trajectory_id=run, turn=rows, time=began, seconds=round((answer.get("timestamp") or began) - began, 2),
                exchange=os.path.basename(path), finish=choice.get("finish_reason"), context=context,
                kind="turn" if sent.get("tools") else "no_tools", holds=holds, changes=changes, reasoning_effort=sent.get("reasoning_effort"),
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
