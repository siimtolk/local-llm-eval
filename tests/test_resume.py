"""Offline tests for resuming unfinished benchmarks (stdlib only, no network).

bench/harness.py Checkpoint: a section cut off after some requests reruns only the rest. pipeline/resume.py:
missing sections, scan after a test, finish (complete / still missing / dropped without progress).
pipeline/runner.py: resume entries join the queue only with weights, one session per model, and a resume
run updates the queue and the candidate's summary. tools/bench_bg.sh: start / wait / stop at --stop-at.
Run from the repo root: python3 -m unittest tests.test_resume -v
"""
import atexit
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import tomllib
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
TMP = Path(tempfile.mkdtemp(prefix="lle-test-resume-"))
atexit.register(shutil.rmtree, TMP, ignore_errors=True)
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(1, str(ROOT / "pipeline"))

import candidates  # noqa: E402
import config  # noqa: E402
import harness  # noqa: E402
import nightjobs  # noqa: E402
import resume  # noqa: E402
import runner  # noqa: E402

MAIN = "model-27b_q4km_nospec_2gpu_128k"
CORE = resume.sections_of("cases.py")


class FakeServer:
    """urlopen stand-in: answers with the request's last message, counts calls, fails after `fail_after`."""

    def __init__(self, fail_after=None):
        self.calls, self.fail_after = 0, fail_after

    def __call__(self, req, timeout=None):
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise ConnectionError("killed")
        self.calls += 1
        body = json.loads(req.data)
        out = {"choices": [{"message": {"content": "echo " + body["messages"][-1]["content"]}, "finish_reason": "stop"}]}
        return mock.Mock(read=lambda: json.dumps(out).encode())


def section(client, reps):  # two prompts x reps, like a fixture section
    return [{"case": p, "ok": client.chat([{"role": "user", "content": p}])["choices"][0]["message"]["content"] == f"echo {p}"}
            for p in ("a", "b") for _ in range(reps)]


class CheckpointTest(unittest.TestCase):
    def test_cut_off_section_resumes_with_only_the_rest(self):
        path = TMP / "cp" / "x.jsonl"
        client = harness.Client("http://h", "k")
        client.checkpoint = harness.Checkpoint(path)
        with mock.patch.object(harness.urllib.request, "urlopen", FakeServer(fail_after=4)):
            with self.assertRaises(ConnectionError):
                section(client, 3)  # 6 requests; killed after 4
        client.checkpoint = cp = harness.Checkpoint(path)
        self.assertEqual(cp.stored, 4)
        server = FakeServer()
        with mock.patch.object(harness.urllib.request, "urlopen", server):
            rows = section(client, 3)
        self.assertEqual(server.calls, 2)  # only the two unfinished requests
        self.assertEqual(cp.hits, 4)
        self.assertTrue(all(r["ok"] for r in rows) and len(rows) == 6)
        cp.discard()
        self.assertFalse(path.exists())

    def test_errors_are_not_stored_and_a_cut_off_line_is_ignored(self):
        path = TMP / "cp2" / "x.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text('{"key": "abc", "out": {"x": 1}}\n{"key": "trunc')
        self.assertEqual(harness.Checkpoint(path).stored, 1)
        err = harness.urllib.error.HTTPError("u", 400, "bad", {}, None)
        err.read = lambda: b"too long"
        client = harness.Client("http://h", "k")
        client.checkpoint = cp = harness.Checkpoint(TMP / "cp3" / "x.jsonl")
        with mock.patch.object(harness.urllib.request, "urlopen", side_effect=err):
            self.assertIn("error", client.chat([{"role": "user", "content": "x"}]))
        self.assertFalse(cp.path.exists())


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(dir=TMP))
        with open(ROOT / "local.example.toml", "rb") as f:
            conf = tomllib.load(f)
        conf["paths"].update(store=str(self.dir / "state" / "candidates.json"), runs=str(self.dir / "state" / "runs"), state=str(self.dir / "state"),
                             experiments=str(self.dir / "state" / "experiments.json"),
                             results=str(self.dir / "results"), serving=str(self.dir / "serving"),
                             models=str(self.dir / "models"))
        p = mock.patch.object(config, "load", return_value=conf)
        p.start()
        self.addCleanup(p.stop)
        for target in ("due", "finish"):
            q = mock.patch.object(nightjobs, target, return_value=[] if target == "due" else None)
            q.start()
            self.addCleanup(q.stop)
        (self.dir / "results").mkdir()

    def write(self, label, sections, extra=(), **more):
        res = {"label": label, "suite": "2026-10-18", "reps": 3, "temperature": None, "request_extra": {}, **more}
        res.update({s: [{"case": s, "ok": True}] for s in sections})
        if extra:
            res["extra"] = {s: [{"case": s, "ok": True}] for s in extra}
        path = self.dir / "results" / f"{label}.json"
        path.write_text(json.dumps(res))
        return path


class ResumeTest(Base):
    def test_missing_counts_speed_core_extras_and_own_traffic(self):
        self.write(MAIN, ["speed"] + CORE, extra=["robust"], own_traffic={"replay": []})
        self.assertEqual(resume.missing(MAIN), ["longagent", "estonian_free", "cron_replay", "ctx"])
        self.assertEqual(resume.missing(MAIN + "_nothink"), list(resume.NOTHINK))  # never written

    def test_scan_queues_main_and_nothink_of_a_model_test(self):
        t0 = time.time() - 5
        self.write(MAIN, ["speed"] + CORE)
        self.write(MAIN + "_nothink", list(resume.NOTHINK))
        self.write("model-27b_side_ab", ["speed"])  # no core sections: not a main label
        queued = resume.scan((self.dir / "results").glob("model-27b_*.json"), t0, "model", "a/M", "model-27b")
        self.assertEqual(queued, [MAIN])
        e = resume.queued()[MAIN]
        self.assertEqual((e["model"], e["kind"], e["sections"][:2]), ("model-27b", "model", ["robust", "longagent"]))

    def test_finish_complete_still_missing_and_stalled(self):
        self.write(MAIN, ["speed"] + CORE)
        resume.add(MAIN)
        before = (resume.missing(MAIN), 0)
        self.write(MAIN, ["speed"] + CORE, extra=["robust", "longagent", "estonian_free"])
        self.assertEqual(resume.finish(MAIN, before)[0], "queued")
        self.assertEqual(resume.queued()[MAIN]["stalled"], 0)
        before = (resume.missing(MAIN), 0)
        for _ in range(resume.MAX_STALLED - 1):
            self.assertEqual(resume.finish(MAIN, before)[0], "queued")
        self.assertEqual(resume.finish(MAIN, before)[0], "dropped")
        self.assertNotIn(MAIN, resume.queued())
        resume.add(MAIN, sections=["replay"])
        self.write(MAIN, ["speed"] + CORE, extra=["robust", "longagent", "estonian_free"],
                   own_traffic={"replay": [], "cron_replay": [], "ctx": []})
        self.assertEqual(resume.finish(MAIN, (["replay"], 0))[0], "done")

    def test_add_refuses_unknown_sections_and_complete_labels(self):
        self.assertFalse(resume.add(MAIN, sections=["nope"])[0])
        self.write(MAIN + "_nothink", list(resume.NOTHINK))
        self.assertFalse(resume.add(MAIN + "_nothink")[0])

    def test_bench_command_keeps_the_labels_own_settings(self):
        self.write(MAIN, ["speed"], reps=2, temperature=0.7, request_extra={"reasoning_effort": "low"})
        cmd = resume.bench_command(MAIN, ["robust", "ctx"], 8085, "/k")
        for part in ("--only robust,ctx", "--reps 2", "--temperature 0.7", "\"reasoning_effort\": \"low\""):
            self.assertIn(part, cmd)
        self.assertIn("--thinking off", resume.bench_command("m_x_nothink", ["tools"], 8085, "/k"))


class RunnerTest(Base):
    def setUp(self):
        super().setUp()
        self.write(MAIN, ["speed"] + CORE)
        resume.add(MAIN, model="model-27b")
        resume.add(MAIN + "_nothink", model="model-27b")

    def queue(self):
        with candidates.store(readonly=True) as data:
            return runner.ready_queue(data, 10 ** 6, None, (), resume.queued())

    def test_waits_for_weights_then_one_session_per_model(self):
        self.assertEqual(self.queue(), [])
        (self.dir / "models" / "model-27b").mkdir(parents=True)
        (self.dir / "models" / "model-27b" / "w.gguf").write_text("x")
        q = self.queue()
        self.assertEqual([(k, key) for _, _, k, key in q], [("resume", MAIN)])
        prompt = runner.prompt_for(MAIN, resume.queued()[MAIN], runner.deadline())
        self.assertIn("RESUME.md", prompt)
        self.assertIn(f"--label {MAIN}_nothink", prompt)
        self.assertIn("$BENCH_STOP_AT", prompt)

    def test_the_nightly_loop_runs_a_resume_entry_before_night_jobs(self):
        """model_test itself (not only the dry run) must see the resume store: the first night after PR #11 it did not."""
        (self.dir / "models" / "model-27b").mkdir(parents=True)
        (self.dir / "models" / "model-27b" / "w.gguf").write_text("x")
        resume.add(MAIN, model="model-27b", priority=10)  # as inkling / glimmer were queued
        ran = []
        job = [(-5, "", "job", "log-triage")]

        def run_job(key, end):
            ran.append(("job", key))
            if len(ran) > 3:
                raise RuntimeError("the loop keeps taking the job: the resume entry was never picked")
        with mock.patch.object(runner, "test_resume", side_effect=lambda key, end, early=False: ran.append(("resume", key)) or True), \
                mock.patch.object(runner.nightjobs, "due", return_value=job), \
                mock.patch.object(runner.nightjobs, "run_job", side_effect=run_job), \
                mock.patch.object(runner, "hold_for_builds"):
            runner.model_test(now=True, end=runner.deadline())
        self.assertEqual(ran[0], ("resume", MAIN))

    def test_resume_run_updates_queue_and_candidate_summary(self):
        with candidates.store() as data:
            data["candidates"]["a/M"] = {"model": "model-27b", "status": "tested", "summary": "271/291", "reported": True}

        def agent(*a, **k):  # the agent finishes the main label and records both
            self.write(MAIN, ["speed"] + CORE, extra=["robust", "longagent", "estonian_free"],
                       own_traffic={"replay": [], "cron_replay": [], "ctx": []})
            resume.mark(MAIN, recorded="x", summary="robust 64/72")
            resume.mark(MAIN + "_nothink", recorded="x", summary="not run: out of time")
            return {"result": "ok"}, "", False

        with mock.patch.object(runner, "agent", side_effect=agent), mock.patch.object(runner.subprocess, "run"), \
                mock.patch.object(runner, "production_ensure"):
            self.assertFalse(runner.test_resume(MAIN, runner.deadline()))
        runs = resume.queued()
        self.assertNotIn(MAIN, runs)
        self.assertEqual(runs[MAIN + "_nothink"]["stalled"], 1)
        with candidates.store(readonly=True) as data:
            c = data["candidates"]["a/M"]
        self.assertFalse(c["reported"])
        self.assertIn("robust 64/72", c["summary"])


class BenchBgTest(unittest.TestCase):
    SH = ROOT / "tools" / "bench_bg.sh"

    def run_sh(self, *args, env):
        return subprocess.run([str(self.SH), *args], capture_output=True, text=True, env=env, timeout=120)

    def test_finishes_and_stops_at_the_stop_time(self):
        d = Path(tempfile.mkdtemp(dir=TMP))
        env = {**os.environ, "LLE_BENCH_BG_DIR": str(d / "bg"), "LLE_CONFIG": str(ROOT / "local.example.toml")}
        key = d / "key"
        key.write_text("k")
        # a server that accepts and never answers: the bench hangs in speed until it is stopped
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        sock.listen(5)
        self.addCleanup(sock.close)
        args = ["--label", "t", "--results-dir", str(d / "r"), "--checkpoint-dir", str(d / "cp"), "--api-key-file", str(key)]
        r = self.run_sh("start", str(d / "a.log"), *args, "--only", "nosuchsection", "--base-url", "http://127.0.0.1:1", env=env)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        r = self.run_sh("wait", "--max", "60", env=env)
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn("exit 0", r.stdout)
        r = self.run_sh("start", str(d / "b.log"), *args, "--only", "speed", "--base-url", f"http://127.0.0.1:{sock.getsockname()[1]}", env=env)
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertEqual(self.run_sh("start", str(d / "c.log"), *args, env=env).returncode, 1)  # one at a time
        r = self.run_sh("wait", "--max", "2", env=env)
        self.assertEqual(r.returncode, 3, r.stdout)
        r = self.run_sh("wait", "--stop-at", time.strftime("%Y-%m-%d %H:%M"), env=env)
        self.assertEqual(r.returncode, 4, r.stdout)
        self.assertEqual(self.run_sh("status", env=env).stdout.split()[:2], ["bench", "finished,"])


if __name__ == "__main__":
    unittest.main()
