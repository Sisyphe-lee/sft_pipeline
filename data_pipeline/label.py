#!/usr/bin/env python3
"""Find and label the steps of a rendered rollout that show a behaviour pattern (patterns/README.md).

    label.py read DIR --protocol patterns/timeline-protocol.md --url URL       # DIR/timeline.json: a reader model's account, long logs in parts
    label.py places DIR --timeline ideas.json surprises.json timeline.json     # DIR/places.json, DIR/packets/: where to look, from rules and readers
    label.py judge DIR [DIR ...] --sheet patterns/label-sheet.md --url URL [URL ...] --name judged   # the model answers the sheet's questions per place
    label.py labels DIR [DIR ...] --name judged --out labels.jsonl             # one record per labelled span, for the selection step

DIR is written by rollout.py. A span gets a label only when the model's answers and steps satisfy the sheet
(EVIDENCE below mirrors the sheet; test_label.py checks that they agree). Standard library only: it runs where the
model is served.
"""
import argparse
import hashlib
import json
import re
import sys
import threading
import time
import urllib.request
from pathlib import Path

# reader moment type -> patterns to judge there
READER = {
    "unexpected": ["C1"],
    "second_failure": ["D4", "D1"],
    "set_aside": ["D2"],
    "expensive_start": ["C2"],
    "record": ["D3"],
    "redo": ["D3"],
    "probe": ["A3"],
}
# rule point kind -> patterns to judge there
RULE = {
    "start": ["A1"],
    "submit": ["L1", "A4"],
    "result": ["A2", "C1", "A3", "B1"],
    "end": ["B1", "D3", "D2"],
}


def parse_json(text):
    """The JSON object in a model answer: fenced or bare."""
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    body = m.group(1) if m else text[text.find("{"): text.rfind("}") + 1]
    return json.loads(body, strict=False)  # models put raw newlines inside strings


def ask(base, model, protocol, text, args):
    """One request to the reader model: (meta, raw message)."""
    body = dict(model=model, temperature=args.temperature, max_tokens=args.max_tokens, messages=[
        dict(role="system", content=protocol), dict(role="user", content=text)])
    body.update(json.loads(args.extra))
    began = time.time()
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=args.timeout) as resp:
        answer = json.load(resp)
    msg = answer["choices"][0]["message"]
    meta = dict(url=base, model=model, seconds=round(time.time() - began), usage=answer.get("usage"),
                finish=answer["choices"][0].get("finish_reason"), temperature=args.temperature, extra=json.loads(args.extra),
                protocol=str(args.protocol))
    return meta, msg


def step_starts(body):
    """[(step number, position in the text)] of the log."""
    starts, want = [], None
    for m in re.finditer(r"^#(\d+) \S+ [A-Z]", body, re.M):
        if want is None or int(m.group(1)) == want:  # a line inside an output can look like a step header; the numbers must run on
            starts.append((int(m.group(1)), m.start()))
            want = int(m.group(1)) + 1
    return starts


def segments(body, limit):
    """Cut the log into parts of about equal size, at step boundaries: [(first step, last step, text)]."""
    starts = step_starts(body)
    count = max(1, -(-len(body) // limit)) if limit else 1
    size, parts, begin = len(body) / count, [], 0
    for k in range(1, count + 1):
        end = len(starts) if k == count else next(i for i, (_, pos) in enumerate(starts) if pos >= size * k)
        text = body[starts[begin][1]: starts[end][1] if end < len(starts) else len(body)]
        parts.append((starts[begin][0], starts[end - 1][0], text))
        begin = end
    return parts


def merge(account, part):
    """Add one part's answer to the account so far. An entry repeated with the same id (direction) or type and step (moment) replaces the earlier one."""
    account.setdefault("phases", []).extend(part.get("phases", []))
    for key, ident in (("directions", lambda x: x.get("id")), ("moments", lambda x: (x.get("type"), x.get("step")))):
        held = account.setdefault(key, [])
        for item in part.get(key, []):
            old = next((i for i, x in enumerate(held) if ident(x) == ident(item)), None)
            if old is None:
                held.append(item)
            else:
                held[old] = item
    return account


def read(args):
    out = Path(args.dir)
    target = out / (args.name + ".json")
    if target.exists() and not args.force:
        print(f"{out.name}: {target.name} exists")
        return
    protocol = Path(args.protocol).read_text()
    head, _, body = (out / "transcript.txt").read_text().partition("\n" + "=" * 100 + "\n")
    base = args.url.rstrip("/")
    with urllib.request.urlopen(base + "/models", timeout=30) as resp:
        model = json.load(resp)["data"][0]["id"]
    parts = segments(body, args.segment_chars)
    context = json.loads((out / args.context).read_text()) if args.context else {}
    account, metas = {}, []
    for k, (lo, hi, text) in enumerate(parts, 1):
        saved = out / f"{args.name}.part{k}.json"
        if len(parts) > 1 and saved.exists() and not args.force:  # a long run is read in parts; a finished part is not asked again
            done = json.loads(saved.read_text())
            metas.append(done.pop("_meta", {}))
            merge(account, done)
            continue
        if len(parts) == 1:
            user = head + "\n" + "=" * 100 + "\n" + text + "\n" + "=" * 100 + "\nEnd of the log. Write the JSON object now, as the instructions say."
        else:
            sofar = dict(account)
            if context:  # another pass's account of the earlier parts: what happened before this part
                sofar["phases_before"] = [p for p in context.get("phases", []) if p.get("steps", [lo])[0] < lo]
                sofar["directions_before"] = [dict(d, attempts=[a for a in d.get("attempts", []) if a.get("steps", [lo])[0] < lo])
                                              for d in context.get("directions", [])]
            user = (head + "\n" + "=" * 100 + f"\nThe log is too long to read at once. This is part {k} of {len(parts)}: steps #{lo} to #{hi}.\n"
                    + ("" if k == 1 else "Your account of the earlier parts:\n" + json.dumps(sofar, ensure_ascii=False) + "\n")
                    + "=" * 100 + "\n" + text + "\n" + "=" * 100 + f"\nEnd of part {k}"
                    + (" and of the log" if k == len(parts) else "; the run goes on after it") + ".\n"
                    "Write the JSON object for this part only, as the instructions say: what happened in these steps and the moments that begin in them. "
                    "Use what you know of the earlier parts (an experiment repeated, an idea named earlier and taken up now, a direction that had already failed). "
                    "Keep the ids of earlier directions; give a direction again, complete, when this part adds attempts to it or changes how it ended. "
                    "If something in this part changes a field of an earlier moment (an idea set aside earlier is tested now, a premise is left now), "
                    "give that moment again with its type and step unchanged and the field filled in.")
        for attempt in range(1, args.tries + 1):
            began = time.time()
            try:
                meta, msg = ask(base, model, protocol, user, args)
                meta.update(attempt=attempt, part=[k, len(parts), lo, hi], max_tokens=args.max_tokens)
                (out / (args.name + (f".part{k}" if len(parts) > 1 else "") + ".raw.json")).write_text(
                    json.dumps(dict(meta=meta, content=msg.get("content"), reasoning=msg.get("reasoning_content")), ensure_ascii=False, indent=1))
                if meta["finish"] == "length":  # the reasoning used up the allowance before the answer was complete
                    args.max_tokens = args.max_tokens * 3 // 2
                    raise ValueError(f"answer cut at max_tokens; next attempt gets {args.max_tokens}")
                data = parse_json(msg.get("content") or "")
                break
            except Exception as error:  # noqa: BLE001 - log and retry whatever the server or the parser raised
                print(f"{out.name}: part {k} attempt {attempt} failed after {round(time.time() - began)}s: {type(error).__name__} {error}", flush=True)
                time.sleep(20)
        else:
            sys.exit(1)
        if len(parts) > 1:
            saved.write_text(json.dumps(dict(data, _meta=meta), ensure_ascii=False, indent=1))
        metas.append(meta)
        merge(account, data)
        print(f"{out.name}: part {k}/{len(parts)} ok in {meta['seconds']}s, usage {meta['usage']}, {len(data.get('moments', []))} moments", flush=True)
    account = {k: v for k, v in account.items() if v or k == "moments"}
    account["_meta"] = metas[0] if len(metas) == 1 else dict(parts=metas)
    target.write_text(json.dumps(account, ensure_ascii=False, indent=1))


# What a label needs (patterns/label-sheet.md): answers that must have a value, steps that must be given in this order,
# which of them must be RUN steps, be submit commands, and lie inside the span.
EVIDENCE = {
    "A1": dict(need=dict(improved_before_first_submit=False, read_answer=True), chain=["submit_step", "<=", "read_step"], run=["submit_step"], submit=["submit_step"]),
    "L1": dict(need=dict(meant_to_score=True, measured_this_version=True, changed_after_measuring=False, number_says_submit=True),
               chain=["measure_step", "<", "submit_step"], run=["measure_step", "submit_step"], submit=["submit_step"], inside=["measure_step", "submit_step"]),
    "A4": dict(need=dict(changed_and_meant_to_compare=True, one_idea_or_each_measured=True), chain=["first_edit_step", "<=", "submit_step"],
               run=["submit_step"], submit=["submit_step"]),
    "A2": dict(need=dict(saw_parts=True, planned_before_result=False, part_names_target=True), chain=["result_step", "<", "work_step"], inside=["work_step"]),
    "C1": dict(need=dict(as_expected=False, message_gives_cause=False, ran_a_check=True, check_can_answer=True, changed_before_check=False, stated_finding=True),
               chain=["result_step", "<", "check_step", "<=", "finding_step"], run=["check_step"], inside=["check_step", "finding_step"]),
    "A3": dict(need=dict(judge_only_fact=True, step_on_known_trend=False, question_before=True, answer_used=True, left_on_probe=False),
               chain=["question_step", "<=", "submit_step", "<=", "answer_step"], run=["submit_step"], submit=["submit_step"], inside=["submit_step"]),
    "B1": dict(need=dict(files_not_best=True, built_on_worse=False, best_back=True), chain=["restore_step"], run=["restore_step"], inside=["restore_step"]),
    "D4": dict(need=dict(same_direction_twice=True, results_moving=False, more_than_one_further=False, came_back_later=False),
               chain=["first_failure_step", "<", "second_failure_step", "<=", "leave_step"]),
    "D1": dict(need=dict(left_direction=True, new_kind_of_evidence=True, used_for_next_change=True),
               chain=["leave_step", "<=", "evidence_step", "<=", "finding_step"], run=["evidence_step"], inside=["evidence_step", "finding_step"]),
    "D2": dict(need=dict(named_and_untested=True, tested_soon_after=False, stalled_after=True, reopened_same_idea=True, test_reached_idea=True),
               chain=["named_step", "<", "stall_step", "<=", "test_step"], run=["test_step"], inside=["test_step"]),
    "D3": dict(need=dict(record_file=True, read_back_later=True, read_feeds_decision=True),
               chain=["write_step", "<", "read_step", "<=", "decision_step"], run=["write_step", "read_step"], inside=["read_step"]),
    "C2": dict(need=dict(costly=True, small_version_first=True, trial_can_tell=True, trial_result_used=True),
               chain=["trial_step", "<", "launch_step"], run=["trial_step", "launch_step"], inside=["trial_step", "launch_step"]),
}


def evidence_gaps(row, kinds, submits):
    """Why a line's own answers and steps do not make a label: a list, empty when they do."""
    spec = EVIDENCE.get(row.get("pattern"))
    if not spec:
        return ["unknown pattern"]
    answers = row.get("answers") if isinstance(row.get("answers"), dict) else {}
    steps = row.get("steps") if isinstance(row.get("steps"), dict) else {}
    gaps = [f"{k} is not {str(v).lower()}" for k, v in spec["need"].items() if answers.get(k) is not v]
    if spec.get("either") and not any(answers.get(k) is True for k in spec["either"]):
        gaps.append("neither " + " nor ".join(spec["either"]))
    names = spec["chain"][::2]
    given = {}
    for name in names:
        n = steps.get(name)
        if isinstance(n, int) and not isinstance(n, bool) and n in kinds:
            given[name] = n
        elif not (n is None and name in spec.get("optional", [])):
            gaps.append(f"{name} missing")
    for a, op, b in zip(names, spec["chain"][1::2], names[1:]):
        if a in given and b in given and not (given[a] < given[b] if op == "<" else given[a] <= given[b]):
            gaps.append(f"{a} not {op} {b}")
    gaps += [f"{n} is not a RUN step" for n in spec.get("run", []) if n in given and kinds[given[n]] != "RUN"]
    gaps += [f"{n} is not a submit command" for n in spec.get("submit", []) if n in given and not any(abs(given[n] - x) <= 2 for x in submits)]
    if isinstance(row.get("start"), int) and isinstance(row.get("end"), int):
        gaps += [f"{n} outside the span" for n in spec.get("inside", []) if n in given and not row["start"] <= given[n] <= row["end"]]
    else:
        gaps.append("no span")
    return gaps


def window(body, lo, hi, limit):
    """The log if it fits in `limit` characters, else the steps lo..hi with as much around them as fits (more before than after):
    (first step, last step, text)."""
    starts = step_starts(body)
    if not limit or len(body) <= limit:
        return starts[0][0], starts[-1][0], body
    pos = [p for _, p in starts] + [len(body)]
    a = next((i for i, (n, _) in enumerate(starts) if n >= lo), len(starts) - 1)
    b = max(a, max((i for i, (n, _) in enumerate(starts) if n <= hi), default=a))
    while b > a and pos[b + 1] - pos[a] > limit:
        b -= 1
    spare = max(0, limit - (pos[b + 1] - pos[a]))
    begin = max(0, pos[a] - spare * 6 // 10)
    end = min(len(body), begin + limit)
    begin = max(0, end - limit)
    a = next(i for i, p in enumerate(pos) if p >= begin)
    b = max(a, max(i for i in range(len(starts)) if pos[i + 1] <= end))
    return starts[a][0], starts[b][0], body[pos[a]:pos[b + 1]]


def judge(args):
    """Judge the packets of rendered rollouts with the served model. It has no file tools, so each request carries the
    sheet, the head, the other reader's account, the log (or the part of it around the packet) and the packet."""
    sheet = Path(args.sheet[0]).read_text()
    for more in args.sheet[1:]:  # sheets that the first one refers to, or that add to it
        sheet += "\n\n" + "=" * 100 + "\n" + Path(more).name + "\n" + "=" * 100 + "\n\n" + Path(more).read_text()
    args.protocol = " ".join(args.sheet)
    urls = [u.rstrip("/") for u in args.url]
    models = {}
    for u in urls:
        with urllib.request.urlopen(u + "/models", timeout=30) as resp:
            models[u] = json.load(resp)["data"][0]["id"]
    jobs = []
    for d in args.dirs:
        out = Path(d)
        (out / args.name / "logs").mkdir(parents=True, exist_ok=True)
        for packet in sorted((out / args.packets).glob("*.json")):
            if not (out / args.name / (packet.stem + ".jsonl")).exists():
                jobs.append((out, packet))
    print(f"{len(jobs)} packets to judge on {len(urls)} URLs x {args.per_url}", flush=True)
    lock, busy = threading.Lock(), {}

    def take(me, last):  # a worker stays on one rollout while it can: the server keeps the log it has just read
        with lock:
            others = {d for w, d in busy.items() if w != me}
            job = (next((j for j in jobs if j[0] == last), None) or next((j for j in jobs if j[0] not in others), None)
                   or (jobs[0] if jobs else None))
            if job:
                jobs.remove(job)
                busy[me] = job[0]
            else:
                busy.pop(me, None)
            return job

    def one(base, out, packet):
        items = json.loads(packet.read_text())
        key = "moment" if "patterns" in items[0] else "item"
        wanted = {(i["id"], p) for i in items for p in i.get("patterns", [i.get("pattern")])}
        near = [n for i in items for n in [i.get("step"), i.get("start"), i.get("end"), (i.get("located") or {}).get("step")] + list(i.get("steps") or [])
                if isinstance(n, int)]
        head, _, body = (out / "transcript.txt").read_text().partition("\n" + "=" * 100 + "\n")
        lo, hi, text = window(body, min(near), max(near), args.window_chars)
        points = json.loads((out / "points.json").read_text())
        first, last = points["steps"]
        cut = (lo, hi) != (first, last)
        kinds = {int(n): kind for n, kind in re.findall(r"^#(\d+) \S+ ([A-Z]+)", body, re.M)}
        submits = [x["step"] for x in points["points"] if x["kind"] == "submit"]
        user = ("You have no file tools here. Everything the instructions call material is in this message: HEAD (head.txt), TIMELINE "
                f"({args.timeline}), LOG (the steps) and PACKET (the batch or packet file to judge).\n"
                + (f"The log is too long to show whole. You get steps #{lo} to #{hi} of #{first} to #{last}; the timeline says what happened "
                   "outside them. Judge by the steps you can see: when a verdict would need steps you cannot see, give the lower verdict and say so in `why`.\n" if cut else "")
                + "=" * 100 + "\nHEAD\n" + head + "\n" + "=" * 100 + "\nTIMELINE\n" + (out / args.timeline).read_text() + "\n"
                + "=" * 100 + "\nLOG\n" + text + "\n" + "=" * 100 + "\nPACKET\n" + json.dumps(items, ensure_ascii=False, indent=1) + "\n" + "=" * 100
                + "\nWrite the output lines as your answer: one JSON object per line and nothing else, a line for every "
                + ("pattern of every moment" if key == "moment" else "span") + " in the packet."
                + ("" if key == "moment" else ' Each line names its span with "item" (the span\'s id) where a moment line would have "moment".'))
        limit, best = args.max_tokens, None
        for attempt in range(1, args.tries + 1):
            began = time.time()
            try:
                local = argparse.Namespace(**dict(vars(args), max_tokens=limit))
                meta, msg = ask(base, models[base], sheet, user, local)
                meta.update(attempt=attempt, steps_shown=[lo, hi], max_tokens=limit, sheet_sha256=hashlib.sha256(sheet.encode()).hexdigest())
                (out / args.name / "logs" / f"{packet.stem}.raw.json").write_text(
                    json.dumps(dict(meta=meta, content=msg.get("content"), reasoning=msg.get("reasoning_content")), ensure_ascii=False, indent=1))
                if meta["finish"] == "length":
                    limit = limit * 3 // 2
                    raise ValueError(f"answer cut at max_tokens; next attempt gets {limit}")
                rows = []
                for line in (msg.get("content") or "").splitlines():
                    line = line.strip().rstrip(",")
                    if line.startswith("{"):
                        try:
                            row = json.loads(line, strict=False)
                            row.setdefault("verdict", "done")  # the sheet asks for answers and steps, not for a verdict
                            row["pattern"], row["verdict"]
                            rows.append(row)
                        except (ValueError, KeyError, TypeError):
                            pass
                for row in rows:  # the label: `done` only when the line's own answers and steps carry it
                    gaps = evidence_gaps(row, kinds, submits) if row["verdict"] == "done" else []
                    if gaps:
                        row.update(verdict="no", gaps=gaps)
                missing = wanted - {(r.get(key), r["pattern"]) for r in rows}
                if best is None or len(missing) < best[0]:
                    best = (len(missing), rows, meta)
                if missing:
                    raise ValueError(f"{len(missing)} of {len(wanted)} lines missing")
                break
            except Exception as error:  # noqa: BLE001 - log and retry whatever the server or the parser raised
                print(f"{out.name} {packet.stem}: attempt {attempt} failed after {round(time.time() - began)}s: {type(error).__name__} {error}", flush=True)
                time.sleep(20)
        if best is None:
            return
        (out / args.name / (packet.stem + ".jsonl")).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in best[1]))
        print(f"done {out.name} {packet.stem}: {len(best[1])} lines, {best[0]} missing, {best[2]['seconds']}s, usage {best[2]['usage']}", flush=True)

    def worker(me, base):
        last = None
        while True:
            job = take(me, last)
            if not job:
                return
            last = job[0]
            one(base, *job)

    threads = [threading.Thread(target=worker, args=(f"{u}#{k}", u)) for u in urls for k in range(args.per_url)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def places(args):
    out = Path(args.dir)
    rule = json.loads((out / "points.json").read_text())
    timeline = dict(moments=[])
    for name in args.timeline:  # several reader passes: a moment of the same type within two steps of one from an earlier pass is the same moment
        earlier = list(timeline["moments"])
        for m in json.loads((out / name).read_text()).get("moments", []):
            try:
                twin = any(x.get("type") == m.get("type") and abs(int(x["step"]) - int(m["step"])) <= 2 for x in earlier)
            except (KeyError, TypeError, ValueError):
                continue
            if not twin:
                timeline["moments"].append(m)
    first, last = rule["steps"]

    def clamp(n):
        return max(first, min(last, int(n)))

    rows = []
    for p in rule["points"]:
        if p["kind"] not in RULE:
            continue
        facts = {k: p[k] for k in ("round", "score", "best_before", "mark", "first_submit_min", "note", "matched_by", "located_by") if k in p}
        rows.append(dict(src="rule", type=p["kind"], step=p["step"], steps=p["steps"], patterns=RULE[p["kind"]], facts=facts))
    restores = sorted(int(m["step"]) for m in timeline.get("moments", []) if m.get("type") == "restore" and str(m.get("step", "")).isdigit())
    for row in rows:  # where the reader saw the files put back: a fact for B1, which is judged at results and at the end
        later = [n for n in restores if n > row["step"]]
        if row["type"] == "result" and later:
            row["facts"]["next_restore_seen_by_reader"] = later[0]
        if row["type"] == "end":
            row["facts"]["restores_seen_by_reader"] = restores
    for m in timeline.get("moments", []):
        if m.get("type") not in READER:
            continue
        try:
            step = clamp(m["step"])
            lo, hi = (clamp(x) for x in (m.get("steps") or [step, step])[:2])
        except (KeyError, TypeError, ValueError):
            continue
        facts = {k: m[k] for k in ("direction", "premise", "what", "source", "expected", "got", "quote", "earlier", "reopened_at", "left_at", "kind", "reason") if m.get(k)}
        results = [r for r in rows if r["src"] == "rule" and r["type"] == "result"]
        if m["type"] == "unexpected":  # the judge's answer itself is the surprise: one moment, not two
            same = next((r for r in results if abs(r["step"] - step) <= 1), None)
        else:  # a probe is judged at the result of its submission
            same = next((r for r in results if min(lo, step) <= r["step"] <= max(hi, step) + 1), None) if m["type"] == "probe" else None
        if same:
            same["facts"].setdefault("reader_notes", []).append(dict(facts, type=m["type"], step=step))
            continue
        rows.append(dict(src="reader", type=m["type"], step=step, steps=[min(lo, step), max(hi, step)], patterns=READER[m["type"]], facts=facts))
    rows.sort(key=lambda x: (x["step"], x["src"] != "rule"))
    for k, row in enumerate(rows, 1):
        row["id"] = f"m{k:03d}"
    (out / "places.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    batches = out / "packets"
    batches.mkdir(exist_ok=True)
    for old in batches.glob("*.json"):
        old.unlink()
    for b in range(0, len(rows), args.batch):
        (batches / f"{b // args.batch + 1:04d}.json").write_text(json.dumps(rows[b:b + args.batch], ensure_ascii=False, indent=1))
    kinds = {}
    for row in rows:
        kinds[f"{row['src']}:{row['type']}"] = kinds.get(f"{row['src']}:{row['type']}", 0) + 1
    print(f"{out.name}: {len(rows)} places in {(len(rows) + args.batch - 1) // args.batch} packets {kinds}")


def labels(args):
    """Collect the labelled spans of rendered rollouts into one JSONL file: the input of the selection step."""
    count, per = 0, {}
    with open(args.out, "w", encoding="utf-8") as stream:
        for d in args.dirs:
            out = Path(d)
            points = json.loads((out / "points.json").read_text())
            index = {m["id"]: m for m in json.loads((out / "places.json").read_text())}
            seen = set()
            for path in sorted((out / args.name).glob("*.jsonl")):
                meta = json.loads((out / args.name / "logs" / (path.stem + ".raw.json")).read_text())["meta"]
                for line in path.read_text().splitlines():
                    row = json.loads(line)
                    key = (row["pattern"], row.get("start"), row.get("end"))
                    if row["verdict"] != "done" or key in seen:  # the same span is often reached from two places
                        continue
                    seen.add(key)
                    place = index.get(row.get("moment"), {})
                    stream.write(json.dumps(dict(
                        task=points["task"], trajectory_id=points["trajectory_id"], pattern=row["pattern"], start=row["start"], end=row["end"],
                        steps=list(range(row["start"], row["end"] + 1)), place=dict(id=row.get("moment"), type=place.get("type"), step=place.get("step")),
                        answers=row.get("answers"), evidence=row.get("steps"), why=row.get("why"), quote=row.get("quote"),
                        model=meta.get("model"), sheet_sha256=meta.get("sheet_sha256")), ensure_ascii=False) + "\n")
                    count += 1
                    per[row["pattern"]] = per.get(row["pattern"], 0) + 1
    print(f"{count} labelled spans in {len(args.dirs)} rollouts: {dict(sorted(per.items()))}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("read")
    p.add_argument("dir")
    p.add_argument("--protocol", required=True)
    p.add_argument("--url", required=True, help="OpenAI-compatible base URL ending in /v1")
    p.add_argument("--name", default="timeline")
    p.add_argument("--max-tokens", type=int, default=40000)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("--tries", type=int, default=3)
    p.add_argument("--extra", default="{}", help="JSON merged into the request body")
    p.add_argument("--force", action="store_true")
    p.add_argument("--segment-chars", type=int, default=1200000, help="read a longer log in parts of about this size; 0 reads it whole")
    p.add_argument("--context", help="another pass's output inside DIR (timeline.json), shown with the later parts of a long log")
    p.set_defaults(run=read)
    p = sub.add_parser("places")
    p.add_argument("dir")
    p.add_argument("--timeline", nargs="+", default=["timeline.json"], help="reader output files inside DIR; earlier files win")
    p.add_argument("--batch", type=int, default=1, help="places per request")
    p.set_defaults(run=places)
    p = sub.add_parser("judge")
    p.add_argument("dirs", nargs="+")
    p.add_argument("--sheet", nargs="+", required=True, help="patterns/label-sheet.md")
    p.add_argument("--url", nargs="+", required=True, help="OpenAI-compatible base URLs ending in /v1")
    p.add_argument("--per-url", type=int, default=1, help="concurrent requests per URL")
    p.add_argument("--name", required=True, help="output directory inside each DIR")
    p.add_argument("--packets", default="packets", help="packet directory inside each DIR")
    p.add_argument("--timeline", default="timeline.json")
    p.add_argument("--window-chars", type=int, default=150000, help="the log is shown as the steps around the place, about this size")
    p.add_argument("--max-tokens", type=int, default=30000)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("--tries", type=int, default=3)
    p.add_argument("--extra", default="{}", help="JSON merged into the request body")
    p.set_defaults(run=judge)
    p = sub.add_parser("labels")
    p.add_argument("dirs", nargs="+")
    p.add_argument("--name", required=True, help="directory of judge output inside each DIR")
    p.add_argument("--out", required=True)
    p.set_defaults(run=labels)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
