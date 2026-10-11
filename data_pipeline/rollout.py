#!/usr/bin/env python3
"""One exported rollout as numbered steps, with the judge's records and the places a rule can find.

    rollout.py ROWS.jsonl --run-dir TASK_DIR [--tasks-dir DIR] --out DIR

Step N is row N of the export (one model reply): its reasoning, what it wrote, the command it ran and the output
that came back. So a step number is a row number, and a label on a step selects that row for training.

Writes DIR/transcript.txt (head, a line of `=`, the steps), DIR/head.txt and DIR/points.json (the rule-located
places: start, every submission of the agent, the judge's answer to it, the end). TASK_DIR is the harness's
directory of the run (run_history.json, submissions/, final_result.json); without it there are no judge records.
Standard library only.
"""
import argparse
import calendar
import json
import re
import time
from pathlib import Path

SUBMIT = re.compile(r"sforge-submit(?![\w-])(?!\s+--list)")
WALL = re.compile(r"^Wall time: ([\d.]+) seconds", re.M)


def load_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def epoch(stamp):
    """Seconds since 1970 of an ISO time in UTC, or of a number; None when unreadable."""
    if isinstance(stamp, (int, float)):
        return float(stamp)
    try:
        return calendar.timegm(time.strptime(str(stamp)[:19], "%Y-%m-%dT%H:%M:%S"))
    except (ValueError, TypeError):
        return None


def text_of(content):
    if isinstance(content, str):
        return content
    return "\n".join(part.get("text", "") for part in content or [] if isinstance(part, dict))


def command(call):
    function = call.get("function") or {}
    args = function.get("arguments") or {}
    if function.get("name") == "exec_command" and "cmd" in args:
        return str(args["cmd"])
    return f"[{function.get('name')}] " + (args["_unparsed"] if "_unparsed" in args else json.dumps(args, ensure_ascii=False))


class Rollout:
    def __init__(self, rows_path, run_dir=None, tasks_dir=None):
        self.steps, self.first_messages = [], []
        with open(rows_path, encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                prompt, meta, reply = row["prompt"], row["metadata"], row["completion"][0]
                last = max((i for i, m in enumerate(prompt) if m.get("role") == "assistant"), default=-1)
                arrived = prompt[last + 1:]  # what came back since the previous reply: tool results, messages to the agent
                if self.steps:
                    self.steps[-1]["out"] = "\n".join(text_of(m.get("content")) for m in arrived if m.get("role") == "tool")
                else:
                    self.first_messages = [text_of(m.get("content")) for m in arrived if m.get("role") == "user"]
                told = [text_of(m.get("content")) for m in arrived if m.get("role") == "user"] if self.steps else []
                calls = reply.get("tool_calls") or []
                self.steps.append(dict(n=meta["turn"], t=meta["time"], think=reply.get("reasoning_content") or "", say=text_of(reply.get("content")),
                                       cmd="\n".join(command(c) for c in calls), has_call=bool(calls), out="", told=told,
                                       rewritten=meta.get("context") == "rewritten", prompt_tokens=(meta.get("usage") or {}).get("prompt_tokens")))
                self.task, self.trajectory = meta["task"]["id"], meta["trajectory_id"]
        for s in self.steps:
            wall = WALL.search(s["out"])
            s["dur"] = float(wall.group(1)) if wall else None
        self.dir = Path(run_dir) if run_dir else None
        spec = load_json(Path(tasks_dir) / f"{self.task}.json", {}) if tasks_dir else {}
        self.minimize = spec.get("judge", {}).get("score_direction", "maximize") == "minimize"
        self.final = load_json(self.dir / "final_result.json", {}) if self.dir else {}
        self.game = bool(self.dir and (self.dir / "game_history.json").exists())
        self.start = self.steps[0]["t"]
        self.subs = self.submissions() if self.dir else []

    def elapsed(self, t):
        return None if t is None else t - self.start

    def submissions(self):
        """Judged rounds in time order, each marked against the best before it. Times come from the judge report
        (`submitted_at`), else from `evolve_state.json`. Rounds outside the time of this rollout are dropped."""
        history = load_json(self.dir / "run_history.json", {}) or {}
        state = {s.get("round"): s.get("at") for s in (load_json(self.dir / "evolve_state.json", {}) or {}).get("submissions", [])}
        rows = []
        for entry in history.get("entries", []):
            if entry.get("type", "submission") != "submission":
                continue
            name = str(entry.get("round", ""))
            report = load_json(self.dir / "submissions" / name / "report.json", {}) or {}
            at = epoch(report.get("submitted_at") or state.get(name))
            rows.append(dict(round=name, kind="agent" if name.startswith("agent") else "auto", at=at,
                             score=entry.get("score") if entry.get("valid", True) else None, order=len(rows)))
        last_use = {r["round"]: r["order"] for r in rows}
        end = self.steps[-1]["t"] + 3600
        rows = [r for r in rows if r["at"] and last_use[r["round"]] == r["order"] and self.start - 5 <= r["at"] <= end]
        rows.sort(key=lambda r: r["at"])
        best = None
        for r in rows:
            score = r.get("score")
            r["el"], r["best_before"] = self.elapsed(r["at"]), best
            if score is None:
                r["mark"] = "no_score"
            elif best is None or (score < best if self.minimize else score > best):
                r["mark"], best = "new_best", score
            elif score == best:
                r["mark"] = "equal_best"
            else:
                r["mark"] = "below_best"
        return rows

    def points(self):
        """The places a rule can find. `steps` is the range to show a reader."""
        out = []
        agent = [r for r in self.subs if r["kind"] == "agent"]
        # A submit command returns to the agent after a few seconds while the judge still runs; the agent then waits, and
        # reads the answer in a later step. So a round belongs to the last submit command before it, and its answer is
        # read in the step that was running when the judge finished.
        submits = [i for i, s in enumerate(self.steps) if s["has_call"] and SUBMIT.search(s["cmd"])]
        pairs, used = [], set()
        for r in agent:
            started = [i for i, s in enumerate(self.steps) if s["t"] <= r["at"]]
            if not started:
                continue
            running = self.steps[started[-1]]
            if running["t"] + (running["dur"] or 0) + 2 < r["at"] and started[-1] + 1 < len(self.steps):
                started.append(started[-1] + 1)  # that command had already returned: the next step is the one that sees the answer
            # rounds and commands both come in order of time: a command before the one the previous round used is not
            # this round's (it was refused, for example during a cooldown)
            before = [i for i in submits if self.steps[i]["t"] <= r["at"] + 5 and i > max(used, default=-1)]
            if before:
                used.add(before[-1])
                pairs.append((before[-1], max(before[-1], started[-1]), r, None))
            else:  # submitted from the agent's own script: the step running when the judge received it
                pairs.append((started[-1], started[-1], r, "time: no submit command of its own, this is the step running when the judge received the submission"))
        pairs.sort(key=lambda pair: (pair[0], pair[2]["at"]))
        last = len(self.steps) - 1

        def add(kind, at, lo, hi, **facts):
            out.append(dict(id=f"{kind}-{len([p for p in out if p['kind'] == kind]) + 1}", kind=kind, step=self.steps[at]["n"],
                            steps=[self.steps[max(0, lo)]["n"], self.steps[min(last, hi)]["n"]], **facts))

        if pairs:
            i, _, r, _ = pairs[0]
            add("start", i, 0, i, first_submit_min=round(r["el"] / 60, 1))
        elif not self.game:
            add("start", last, max(0, last - 60), last, note="no judged agent submission in the whole run")
        for j, (i, read, r, how) in enumerate(pairs):
            nxt = pairs[j + 1][0] if j + 1 < len(pairs) else last
            facts = dict(round=r["round"], score=r["score"], best_before=r["best_before"], mark=r["mark"])
            if how:
                facts["located_by"] = how
            prev = pairs[j - 1][0] + 1 if j else 0
            add("submit", i, max(prev, i - 25), i, **facts)  # the submission itself: what went into it
            add("result", read, read, max(read, min(nxt, read + 25)), submit_step=self.steps[i]["n"], **facts)  # where the answer is read, and what follows
        add("end", last, max(0, last - 40), last)
        return out

    def window(self, lo, hi, cap=3000, tags=None):
        """The steps lo..hi as text; long reasoning, commands and outputs are cut in the middle."""
        def cut(text):
            return text if len(text) <= cap else text[: cap * 2 // 3] + f"\n…[{len(text) - cap} chars omitted]…\n" + text[-cap // 3:]

        blocks = []
        for s in self.steps:
            if not lo <= s["n"] <= hi:
                continue
            el = int(self.elapsed(s["t"]))
            took = f" ({s['dur']:.0f}s)" if s["dur"] else ""
            head = f"#{s['n']} {el // 3600}:{el % 3600 // 60:02d}:{el % 60:02d} {'RUN' + took if s['has_call'] else 'SAY'}{(tags or {}).get(s['n'], '')}"
            parts = [head]
            if s["rewritten"]:
                parts.append("[the scaffold replaced the earlier conversation with a summary before this step]")
            parts += [f"MESSAGE TO AGENT\n{cut(m)}" for m in s["told"]]
            if s["think"]:
                parts.append("THINK\n" + cut(s["think"]))
            if s["say"]:
                parts.append("SAY\n" + cut(s["say"]))
            if s["has_call"]:
                parts.append("$ " + cut(s["cmd"]) + "\n" + cut(s["out"]))
            blocks.append("\n".join(parts))
        return "\n\n".join(blocks)

    def render(self, out, cap=3000):
        points = self.points()
        tags, located = {}, {}
        for p in points:
            if p["kind"] == "result":
                located[p["round"]] = p["submit_step"]
                before = "none" if p["best_before"] is None else p["best_before"]
                note = "; placed by time, the submission came from a script or a background command" if p.get("located_by") else ""
                tags[p["step"]] = tags.get(p["step"], "") + f"   [judge record {p['round']}: score {p['score']}, {p['mark']}, best before {before}{note}]"

        def clock(el):
            return "-:--" if el is None else "%d:%02d" % (int(el) // 3600, int(el) % 3600 // 60)

        rounds = "\n".join(f"  {x['round']:<10} {clock(x['el'])}  score {x['score']}  {x['mark']}"
                           + (f"  submit command at #{located[x['round']]}" if x["round"] in located else "") for x in self.subs)
        hours = (self.steps[-1]["t"] - self.start) / 3600
        head = (f"RUN of one agent on one task. The score is {'minimised' if self.minimize else 'maximised'}. {len(self.steps)} steps, {hours:.1f} hours.\n"
                "A step is one reply of the agent. Its head says RUN when the reply ran a command and SAY when it did not. Inside a step:\n"
                "THINK is the agent's reasoning, SAY is what it wrote, `$` is the command with its output. Long texts are cut in the middle.\n"
                + ("This task is a game played through a server: the score is the game's own score, and there are no submissions.\n" if self.game else "")
                + "JUDGED ROUNDS, from the judge's own records. `agent-N` is a submission the agent made; `auto-N` is a submission the harness\n"
                "made of the current files, whose result the agent is not shown. A [judge record ...] note on a step is added by us:\n"
                "the agent saw only what the command printed.\n" + rounds + "\n"
                + "THE TASK AS GIVEN TO THE AGENT (cut):\n" + "\n".join(m[:6000] for m in self.first_messages[-1:]) + "\n")
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        first, last = self.steps[0]["n"], self.steps[-1]["n"]
        (out / "transcript.txt").write_text(head + "\n" + "=" * 100 + "\n\n" + self.window(first, last, cap=cap, tags=tags) + "\n")
        (out / "head.txt").write_text(head)
        (out / "points.json").write_text(json.dumps(dict(task=self.task, trajectory_id=self.trajectory, steps=[first, last], points=points),
                                                    ensure_ascii=False, indent=1))
        return points


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("rows")
    parser.add_argument("--run-dir", help="the harness's directory of this run, for the judge's records")
    parser.add_argument("--tasks-dir", help="EdgeBench task specifications, for the score direction")
    parser.add_argument("--out", required=True)
    parser.add_argument("--cap", type=int, default=3000, help="characters kept of one reasoning text, command or output")
    args = parser.parse_args()
    rollout = Rollout(args.rows, args.run_dir, args.tasks_dir)
    points = rollout.render(args.out, args.cap)
    size = (Path(args.out) / "transcript.txt").stat().st_size
    print(f"{rollout.trajectory}: {len(rollout.steps)} steps, {len(points)} rule points, {len(rollout.subs)} judged rounds, transcript {size // 1024}K")


if __name__ == "__main__":
    main()
