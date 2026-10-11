#!/usr/bin/env python3
"""The labelling steps that annotate.py runs on a rendered rollout (DIR, written by rollout.py).

    label.py read DIR --protocol patterns/timeline-protocol.md --url URL    # DIR/timeline.json: a reader model's account, long logs in parts
    label.py places DIR --timeline ideas.json surprises.json timeline.json  # DIR/places.json: where to look, from rules and readings
    label.py judge DIR [DIR ...] --url URL [URL ...]                        # DIR/labels/<pattern>-<sha>/<place>.json
    label.py labels DIR [DIR ...] --out labels.jsonl                        # the labels of the current pattern versions

A pattern is one file, patterns/<name>.yaml. One request asks about neighbouring places and all the patterns looked
for there; each answer is kept by place and pattern under the pattern's fingerprint, so after an edit of one pattern
file only that pattern is asked again.
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

PATTERN_DIR = Path(__file__).resolve().parents[1] / "patterns"
RULE_PLACES = ("start", "submit", "result", "end")  # found by rollout.py; every other place type comes from a reading
REACH = 10  # places further apart than this many steps are not asked about in one request


def load_patterns(directory=PATTERN_DIR):
    """(shared instructions, {name: pattern}). A pattern is one YAML file: the place types where it is looked for, its
    yes/no questions with the answer the behaviour needs, its steps in order, and which of the steps carry the label
    (`span`). `sha` is the fingerprint of what the model reads (the shared instructions and this pattern's questions,
    steps and notes): editing those asks the model again, for this pattern only. `places` and `skip` (where it is asked, see
    asked_at), `effort` (the reasoning effort of its requests, when not the configured one) and `submit`, `within` and `apart` (a step lies
    at most, or at least, that many steps after the one before it) and `span` (`from`/`to`/`max`: the steps between two named steps, dropped
    when longer than `max`; or `steps`: just the named steps) are applied by the program when labels are collected, so
    changing them needs no new answers."""
    import yaml
    common = (directory / "common.md").read_text()
    patterns = {}
    for path in sorted(directory.glob("*.yaml")):
        text = path.read_text()
        pattern = yaml.safe_load(text)
        pattern["sha"] = hashlib.sha256((common + "\n" + pattern_text(pattern)).encode()).hexdigest()[:8]
        patterns[pattern["name"]] = pattern
    return common, patterns


def pattern_text(pattern):
    """A pattern as the labelling model reads it."""
    lines = [f"{pattern['name']}: {pattern['title']}", "Questions:"]
    lines += [f"- `{q['id']}` [{str(q['expect']).lower()}]: {q['text']}" for q in pattern["questions"]]
    chain = ""
    for step in pattern["steps"]:
        notes = ", ".join(x for x in ("RUN" if step.get("run") else "", step.get("text", "")) if x)
        chain += f"`{step['id']}`" + (f" ({notes})" if notes else "") + (f" {'<' if step['then'] == '<' else '≤'} " if step.get("then") else "")
    lines.append("Steps: " + chain + ".")
    return "\n".join(lines + [f"- {note}" for note in pattern.get("notes", [])])


def decide(pattern, row, kinds, submits):
    """(labelled steps, gaps). The steps carry the label when the model's answers and steps hold; else gaps says why not."""
    answers = row.get("answers") if isinstance(row.get("answers"), dict) else {}
    steps = row.get("steps") if isinstance(row.get("steps"), dict) else {}
    gaps = [f"{q['id']} is not {str(q['expect']).lower()}" for q in pattern["questions"] if answers.get(q["id"]) is not q["expect"]]
    given = {}
    for step in pattern["steps"]:
        n = steps.get(step["id"])
        if isinstance(n, int) and not isinstance(n, bool) and n in kinds:
            given[step["id"]] = n
            if step.get("run") and kinds[n] != "RUN":
                gaps.append(f"{step['id']} is not a RUN step")
            if step.get("submit") and not any(abs(n - x) <= 2 for x in submits):
                gaps.append(f"{step['id']} is not a submit command")
        else:
            gaps.append(f"{step['id']} missing")
    for a, b in zip(pattern["steps"], pattern["steps"][1:]):
        if a["id"] in given and b["id"] in given:
            if not (given[a["id"]] < given[b["id"]] if a["then"] == "<" else given[a["id"]] <= given[b["id"]]):
                gaps.append(f"{a['id']} not {a['then']} {b['id']}")
            elif b.get("within") and given[b["id"]] - given[a["id"]] > b["within"]:
                gaps.append(f"{b['id']} more than {b['within']} steps after {a['id']}")  # too far apart to be one piece of behaviour
            elif b.get("apart") and given[b["id"]] - given[a["id"]] < b["apart"]:
                gaps.append(f"{b['id']} less than {b['apart']} steps after {a['id']}")  # too close: the behaviour needs time in between
    span = pattern["span"]
    if "steps" in span:  # the named steps themselves carry the label, not what lies between them
        marked = [given.get(name) for name in span["steps"]]
        return ([] if gaps or None in marked else sorted(set(marked))), gaps
    lo, hi = given.get(span["from"]), given.get(span["to"])
    if lo is not None and hi is not None and hi - lo + 1 > span["max"]:
        gaps.append(f"more than {span['max']} steps from {span['from']} to {span['to']}")  # too long to be this behaviour and little else
    return ([] if gaps or lo is None or hi is None else list(range(lo, hi + 1))), gaps


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


def asked_at(pattern, place):
    """Whether the pattern is looked for at the place: at the place types it names, unless the facts of the place rule
    the behaviour out (`skip`: {fact: [values]}). A result where a reader noted a moment counts as a place of that
    moment's type too: C1 names `unexpected`, so of the results it is asked only where a reader saw one."""
    facts = place.get("facts") or {}
    kinds = {place["type"]} | {note.get("type") for note in facts.get("reader_notes", [])}
    if not kinds & set(pattern["places"]):
        return False
    return not any(fact in facts and facts[fact] in values for fact, values in (pattern.get("skip") or {}).items())


def requests(out, places, patterns, wanted, most):
    """The questions still open for one rollout, grouped into requests: [[(place, pattern), ...], ...]. The two places
    of one submission (the command and its result) are always asked together, with all their patterns. Neighbours join
    while a request holds at most `most` (place, pattern) pairs and its places lie within REACH steps. A pattern that
    names its own reasoning `effort` is asked apart from the others, in requests of that effort."""
    units = {}
    for place in sorted(places, key=lambda p: p["step"]):
        submission = place["facts"].get("round") if place["type"] in ("submit", "result") else None
        for name, p in patterns.items():
            if name in wanted and asked_at(p, place) and answer_of(out, place, p) is None:
                units.setdefault((p.get("effort"), submission or place["id"]), []).append((place, p))
    grouped = []
    for effort in {key[0] for key in units}:
        last = None
        for unit in (u for key, u in units.items() if key[0] == effort):
            if last and len(last) + len(unit) <= most and unit[-1][0]["step"] - last[0][0]["step"] <= REACH:
                last.extend(unit)
            else:
                last = list(unit)
                grouped.append(last)
    return sorted(grouped, key=lambda pairs: pairs[0][0]["step"])


def answer_of(out, place, pattern):
    """The stored answer for this place and this version of the pattern, or None. An answer records the place it was
    given for; one that was given for another place (places found anew after a change of the rules) does not count."""
    path = out / "labels" / f"{pattern['name']}-{pattern['sha']}" / f"{place['id']}.json"
    if not path.exists():
        return None
    row = json.loads(path.read_text())
    return row if row.get("at", [place["type"], place["step"]]) == [place["type"], place["step"]] else None


def judge(args):
    """Ask the served model every (place, pattern) of rendered rollouts that has no answer yet for the pattern's current
    version. One request holds the shared instructions, the head, the other reader's account, the steps around its
    places, the patterns and the places. An answer goes to DIR/labels/<pattern>-<sha>/<place id>.json; what the model
    wrote for the whole request (its reasoning, the time and the tokens) goes to DIR/requests/."""
    common, patterns = load_patterns()
    wanted = set(args.patterns or patterns)
    args.protocol = "patterns/common.md"
    urls = [u.rstrip("/") for u in args.url]
    models = {}
    for u in urls:
        with urllib.request.urlopen(u + "/models", timeout=30) as resp:
            models[u] = json.load(resp)["data"][0]["id"]
    jobs = []
    for d in args.dirs:
        out = Path(d)
        jobs += [(out, pairs) for pairs in requests(out, json.loads((out / "places.json").read_text()), patterns, wanted, args.together)]
    print(f"{sum(len(pairs) for _, pairs in jobs)} questions to ask in {len(jobs)} requests on {len(urls)} URLs x {args.per_url}", flush=True)
    lock, busy, shared = threading.Lock(), {}, {}

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

    def material(out):
        with lock:
            if out not in shared:
                head, _, body = (out / "transcript.txt").read_text().partition("\n" + "=" * 100 + "\n")
                points = json.loads((out / "points.json").read_text())
                shared[out] = dict(head=head, body=body, timeline=(out / args.timeline).read_text(), first=points["steps"][0], last=points["steps"][1],
                                   kinds={int(n): kind for n, kind in re.findall(r"^#(\d+) \S+ ([A-Z]+)", body, re.M)},
                                   submits=[x["step"] for x in points["points"] if x["kind"] == "submit"])
            return shared[out]

    def one(base, out, pairs):
        m = material(out)
        limit = args.max_tokens
        for attempt in range(1, args.tries + 1):
            at = {}  # place id -> (place, the patterns asked there)
            for place, pattern in pairs:
                at.setdefault(place["id"], (place, []))[1].append(pattern)
            near = [n for place, _ in at.values() for n in [place.get("step")] + list(place.get("steps") or []) if isinstance(n, int)]
            lo, hi, text = window(m["body"], min(near), max(near), args.window_chars)
            asked = sorted({pattern["name"]: pattern for _, pattern in pairs}.values(), key=lambda p: p["name"])
            shown = [dict({k: v for k, v in place.items() if k != "patterns"}, patterns=[p["name"] for p in ps]) for place, ps in at.values()]
            # the patterns and the places come last: requests for neighbouring places share everything before them
            user = ("=" * 100 + "\nHEAD\n" + m["head"] + "\n" + "=" * 100 + "\nTIMELINE\n" + m["timeline"] + "\n" + "=" * 100
                    + f"\nLOG (steps #{lo} to #{hi} of #{m['first']} to #{m['last']})\n" + text + "\n" + "=" * 100
                    + "\nPATTERNS\n" + "\n\n".join(pattern_text(p) for p in asked) + "\n" + "=" * 100
                    + "\nPLACES\n" + json.dumps(shown, ensure_ascii=False, indent=1) + "\n" + "=" * 100
                    + f"\nWrite your answer now: one JSON object on one line, with {len(pairs)} entries in `results`.")
            began = time.time()
            extra = dict(json.loads(args.extra), **({"reasoning_effort": asked[0]["effort"]} if asked[0].get("effort") else {}))
            try:
                meta, msg = ask(base, models[base], common, user, argparse.Namespace(**dict(vars(args), max_tokens=limit, extra=json.dumps(extra))))
                if meta["finish"] == "length":
                    if len(at) > 1:  # too much to think about at once: ask about the places in two halves
                        ids = list(at)[:len(at) // 2]
                        print(f"{out.name} {'+'.join(at)}: cut at max_tokens after {meta['seconds']}s; asking in two parts", flush=True)
                        one(base, out, [x for x in pairs if x[0]["id"] in ids])
                        one(base, out, [x for x in pairs if x[0]["id"] not in ids])
                        return
                    limit = limit * 3 // 2
                    raise ValueError(f"answer cut at max_tokens; next attempt gets {limit}")
                rows = parse_json(msg.get("content") or "").get("results")
                rows = {(str(row.get("place")), str(row.get("pattern"))): row for row in rows if isinstance(row, dict) and isinstance(row.get("answers"), dict)}
                name = f"{time.strftime('%m%d-%H%M%S')}-{pairs[0][0]['id']}-{pairs[0][1]['name']}-{len(pairs)}.json"
                (out / "requests").mkdir(exist_ok=True)
                (out / "requests" / name).write_text(json.dumps(dict(
                    asked=[[place["id"], pattern["name"]] for place, pattern in pairs], answered=len(rows), meta=dict(meta, attempt=attempt, steps_shown=[lo, hi]),
                    content=msg.get("content"), reasoning=msg.get("reasoning_content")), ensure_ascii=False, indent=1))
                found = 0
                for place, pattern in list(pairs):
                    row = rows.get((place["id"], pattern["name"]))
                    if row is None:
                        continue
                    found += not decide(pattern, row, m["kinds"], m["submits"])[1]  # for the log only; labels() decides again from the stored answer
                    record = dict(place=place["id"], at=[place["type"], place["step"]], pattern=pattern["name"], sha=pattern["sha"],
                                  answers=row.get("answers"), steps=row.get("steps"), why=row.get("why"), quote=row.get("quote"),
                                  **{k: row[k] for k in ("source", "file") if k in row},
                                  meta=dict(model=meta["model"], url=base, request=name, asked_together=len(pairs), steps_shown=[lo, hi]))
                    target = out / "labels" / f"{pattern['name']}-{pattern['sha']}" / f"{place['id']}.json"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.with_suffix(".tmp").write_text(json.dumps(record, ensure_ascii=False, indent=1))
                    target.with_suffix(".tmp").replace(target)
                    pairs.remove((place, pattern))
                usage = meta["usage"] or {}
                print(f"done {out.name} {'+'.join(at)}: {len(rows)} answers, {found} labels, {meta['seconds']}s, "
                      f"{usage.get('prompt_tokens')}+{usage.get('completion_tokens')} tokens", flush=True)
                if not pairs:
                    return
                raise ValueError(f"{len(pairs)} of the questions were not answered")  # asked again, alone
            except Exception as error:  # noqa: BLE001 - log and retry whatever the server or the parser raised
                print(f"{out.name} {'+'.join(at)}: attempt {attempt} failed after {round(time.time() - began)}s: {type(error).__name__} {error}", flush=True)
                time.sleep(20 if "not answered" not in str(error) else 0)

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
    patterns = load_patterns()[1]
    at = {kind for pattern in patterns.values() for kind in pattern["places"]} | {"submit", "result"}  # the place types to keep
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
        if p["kind"] not in at:
            continue
        facts = {k: p[k] for k in ("round", "score", "best_before", "mark", "first_submit_min", "note", "matched_by", "located_by") if k in p}
        rows.append(dict(src="rule", type=p["kind"], step=p["step"], steps=p["steps"], facts=facts))
    restores = sorted(int(m["step"]) for m in timeline.get("moments", []) if m.get("type") == "restore" and str(m.get("step", "")).isdigit())
    for row in rows:  # where the reader saw the files put back: a fact for B1, which is judged at results and at the end
        later = [n for n in restores if n > row["step"]]
        if row["type"] == "result" and later:
            row["facts"]["next_restore_seen_by_reader"] = later[0]
        if row["type"] == "end":
            row["facts"]["restores_seen_by_reader"] = restores
    for m in timeline.get("moments", []):
        if m.get("type") not in at or m.get("type") in RULE_PLACES:
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
        rows.append(dict(src="reader", type=m["type"], step=step, steps=[min(lo, step), max(hi, step)], facts=facts))
    rows.sort(key=lambda x: (x["step"], x["src"] != "rule"))
    taken = set()
    for row in rows:  # the id says what the place is, so it stays the same when other places come or go
        row["id"] = f"{row['type']}-{row['step']:04d}"
        while row["id"] in taken:
            row["id"] += "b"
        taken.add(row["id"])
    (out / "places.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    kinds = {}
    for row in rows:
        kinds[f"{row['src']}:{row['type']}"] = kinds.get(f"{row['src']}:{row['type']}", 0) + 1
    print(f"{out.name}: {len(rows)} places, {sum(asked_at(p, r) for r in rows for p in patterns.values())} questions {kinds}")


def labels(args):
    """Decide the labels from the stored answers, by the current pattern files, and write them to one JSONL file."""
    _, patterns = load_patterns()
    count, per, missing = 0, {}, 0
    with open(args.out, "w", encoding="utf-8") as stream:
        for d in args.dirs:
            out = Path(d)
            points = json.loads((out / "points.json").read_text())
            body = (out / "transcript.txt").read_text().partition("\n" + "=" * 100 + "\n")[2]
            kinds = {int(n): kind for n, kind in re.findall(r"^#(\d+) \S+ ([A-Z]+)", body, re.M)}
            submits = [x["step"] for x in points["points"] if x["kind"] == "submit"]
            seen = set()
            for place in json.loads((out / "places.json").read_text()):
                for name in (n for n, p in patterns.items() if asked_at(p, place)):
                    row = answer_of(out, place, patterns[name])
                    if row is None:
                        missing += 1
                        continue
                    steps, _ = decide(patterns[name], row, kinds, submits)
                    if not steps or (name, steps[0], steps[-1]) in seen:  # the same steps are often reached from two places
                        continue
                    seen.add((name, steps[0], steps[-1]))
                    stream.write(json.dumps(dict(
                        task=points["task"], trajectory_id=points["trajectory_id"], pattern=name, pattern_sha=row["sha"], steps=steps,
                        place=dict(id=place["id"], type=place["type"], step=place["step"]), answers=row["answers"], evidence=row["steps"],
                        why=row.get("why"), quote=row.get("quote"), model=row["meta"]["model"]), ensure_ascii=False) + "\n")
                    count += 1
                    per[name] = per.get(name, 0) + 1
    print(f"{count} labels in {len(args.dirs)} rollouts: {dict(sorted(per.items()))}" + (f"; {missing} questions not answered yet" if missing else ""))


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
    p.set_defaults(run=places)
    p = sub.add_parser("judge")
    p.add_argument("dirs", nargs="+")
    p.add_argument("--url", nargs="+", required=True, help="OpenAI-compatible base URLs ending in /v1")
    p.add_argument("--per-url", type=int, default=1, help="concurrent requests per URL")
    p.add_argument("--patterns", nargs="*", help="ask only about these patterns (default: all)")
    p.add_argument("--together", type=int, default=8, help="at most this many (place, pattern) pairs in one request; one submission is never split")
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
    p.add_argument("--out", required=True)
    p.set_defaults(run=labels)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
