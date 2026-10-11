"""The whole of annotate.py and select_rows.py against a stand-in for the labelling model."""
import gzip
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

REPO = Path(__file__).parents[1]


class Model(BaseHTTPRequestHandler):
    asked = []

    def log_message(self, *args):
        pass

    def reply(self, body):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.reply({"data": [{"id": "stand-in"}]})

    def do_POST(self):
        sent = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        user = sent["messages"][1]["content"]
        if "PATTERN\n" in user:  # a labelling question: say yes to C1 with a check at step 3, no to everything else
            name = user.split("PATTERN\n")[1].split(":")[0]
            Model.asked.append(name)
            answer = {"answers": {}, "steps": {}}
            if name == "C1":
                answer = {"answers": {"as_expected": False, "message_gives_cause": False, "ran_a_check": True, "check_can_answer": True,
                                      "changed_before_check": False, "stated_finding": True},
                          "steps": {"result_step": 2, "check_step": 3, "finding_step": 4}}
        else:  # a reading: report one unexpected result at step 2
            answer = {"phases": [], "directions": [], "moments": [{"type": "unexpected", "step": 2, "steps": [2, 4], "what": "worse"}]}
        self.reply({"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(answer)}}], "usage": {}})


def capture(root, turns):
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "task"}]
    for k in range(turns):
        reply = {"role": "assistant", "content": f"say {k}", "reasoning_content": f"think {k}",
                 "tool_calls": [{"id": f"exec_command:{k}", "type": "function", "function": {"name": "exec_command", "arguments": json.dumps({"cmd": f"run {k}"})}}]}
        folder = root / f"resp_{k:03d}"
        folder.mkdir(parents=True)
        for name, value in (("request", {"timestamp": 1000 + 60 * k, "capture_id": "sforge.run.toy.run1-toy", "body": {}}),
                            ("upstream-request", {"messages": list(messages), "tools": [{"type": "function", "function": {"name": "exec_command"}}]}),
                            ("upstream-response", {"timestamp": 1001 + 60 * k, "status_code": 200, "body": json.dumps(
                                {"choices": [{"finish_reason": "tool_calls", "message": reply}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}})})):
            with gzip.open(folder / f"{name}.json.gz", "wt") as stream:
                json.dump(value, stream)
        messages += [dict(reply, reasoning_content=None), {"role": "tool", "tool_call_id": f"exec_command:{k}", "content": f"Wall time: 1.0 seconds\nOutput:\nout {k}"}]


def test_annotate_then_select(tmp_path):
    capture(tmp_path / "capture", 5)
    server = HTTPServer(("127.0.0.1", 0), Model)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    env = {"KIMI_URLS": f"http://127.0.0.1:{server.server_port}/v1", "PATH": "/usr/bin:/bin"}
    annotate = [sys.executable, str(REPO / "data_pipeline/annotate.py"), "--config", str(REPO / "configs/labelling/kimi_v1.yaml"),
                "--capture", str(tmp_path / "capture"), "--runs", str(tmp_path / "runs"), "--data", str(tmp_path / "data")]
    first = subprocess.run(annotate, env=env, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    labels = [json.loads(line) for line in (tmp_path / "data/labels.jsonl").read_text().splitlines()]
    assert [(x["pattern"], x["steps"]) for x in labels] == [("C1", [3, 4])]
    row = json.loads((tmp_path / "data/rows/run1-toy.jsonl").read_text().splitlines()[2])
    assert row["metadata"]["turn"] == 3 and row["completion"]["reasoning_content"] == "think 2"
    assert [m.get("reasoning_content") for m in row["prompt"] if m["role"] == "assistant"] == ["think 0", "think 1"]  # put back from earlier replies
    # a second run asks nothing
    asked = len(Model.asked)
    assert subprocess.run(annotate, env=env, capture_output=True, text=True).returncode == 0 and len(Model.asked) == asked
    for policy, turns in (("pattern_all", [3, 4]), ("all_turns", [1, 2, 3, 4, 5])):
        out = tmp_path / f"{policy}.json"
        done = subprocess.run([sys.executable, str(REPO / "data_pipeline/select_rows.py"), "--policy", str(REPO / f"configs/data_policies/{policy}.yaml"),
                               "--rows", str(tmp_path / "data/rows"), "--labels", str(tmp_path / "data/labels.jsonl"), "--output", str(out)],
                              capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        assert [r["source_line"] for r in json.loads(out.read_text())["train"]] == turns
    control = tmp_path / "control.json"
    fixes = tmp_path / "fixes.jsonl"
    fixes.write_text(json.dumps({"trajectory_id": "run1-toy", "steps": [4]}) + "\n")
    subprocess.run([sys.executable, str(REPO / "data_pipeline/select_rows.py"), "--policy", str(REPO / "configs/data_policies/control_random.yaml"),
                    "--rows", str(tmp_path / "data/rows"), "--labels", str(tmp_path / "data/labels.jsonl"), "--exclude", str(fixes), "--output", str(control)], check=True)
    assert len(json.loads(control.read_text())["train"]) == 1  # one labelled step is left after the correction, so one random row
    server.shutdown()
