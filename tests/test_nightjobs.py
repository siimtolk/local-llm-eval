"""Offline tests for pipeline/nightjobs.py: due/priority order, JSON parsing, runner integration.
Run from the repo root: python3 -m unittest tests.test_nightjobs -v"""
import json
import subprocess
import sys
import tempfile
import tomllib
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

import config  # noqa: E402
import nightjobs  # noqa: E402
import runner  # noqa: E402


class TestNightJobs(unittest.TestCase):
    def setUp(self):
        with open(ROOT / "local.example.toml", "rb") as f:
            conf = tomllib.load(f)
        self.tmp = Path(tempfile.mkdtemp(prefix="lle-test-nightjobs-"))
        conf["paths"]["state"] = str(self.tmp)
        # production_model_id() and llm() read the API key file before anything else; give them a real one
        self.key_file = self.tmp / "api_key.txt"
        self.key_file.write_text("test-key")
        conf["serving"]["api_key_file"] = str(self.key_file)
        p = mock.patch.object(config, "load", return_value=conf)
        p.start()
        self.addCleanup(p.stop)
        q = mock.patch.object(nightjobs, "STATE", self.tmp / "nightjobs")
        q.start()
        self.addCleanup(q.stop)

    @staticmethod
    def _resp(payload):
        """A fake urlopen response: .read() yields payload as JSON bytes."""
        class R:
            def read(self):
                return json.dumps(payload).encode()
        return R()

    def test_all_jobs_due_when_never_run_in_priority_order(self):
        order = [k for *_, k in nightjobs.due(10 ** 6)]
        self.assertEqual(order[0], "log-triage")
        self.assertEqual(order[-1], "trend-synth")
        self.assertEqual(len(order), len(nightjobs.JOBS))

    def test_job_that_does_not_fit_is_skipped_and_tried_today_is_not_due(self):
        self.assertNotIn("regression", [k for *_, k in nightjobs.due(10 * 60)])
        state = {"log-triage": {"tried": nightjobs.today()}}
        self.assertNotIn("log-triage", [k for *_, k in nightjobs.due(10 ** 6, state)])

    def test_weekly_job_waits_for_its_cadence_and_overdue_raises_priority(self):
        three = (datetime.now() - nightjobs.timedelta(days=3)).strftime("%Y-%m-%d")
        ten = (datetime.now() - nightjobs.timedelta(days=10)).strftime("%Y-%m-%d")
        state = {"idea-mining": {"tried": three}, "bug-hunt": {"tried": ten}}
        due = {k: p for p, _, _, k in nightjobs.due(10 ** 6, state)}
        self.assertNotIn("idea-mining", due)
        self.assertEqual(due["bug-hunt"], -(nightjobs.JOBS["bug-hunt"]["prio"] + 9))

    def test_jobs_join_the_runner_queue_after_higher_priority_candidates(self):
        cands = {"candidates": {"a/Model-GGUF": {"status": "downloaded", "model": "model-27b", "priority": 9, "brief": "2026-10-01"}}}
        ready = runner.ready_queue(cands, 10 ** 6, None, nightjobs.due(10 ** 6))
        self.assertEqual(ready[0][2:], ("model", "a/Model-GGUF"))
        self.assertEqual(ready[1][2:], ("job", "log-triage"))

    def test_jparse(self):
        self.assertEqual(nightjobs.jparse('text ```json\n[{"a": 1}]\n``` more'), [{"a": 1}])
        self.assertEqual(nightjobs.jparse('Sure: {"real": true} done'), {"real": True})
        self.assertIsNone(nightjobs.jparse("no json here"))

    def test_log_ignore_drops_known_noise(self):
        self.assertTrue(nightjobs._ignored("Unable to get default sink"))
        self.assertTrue(nightjobs._ignored("gsd_power_backlight_percentage_to_abs: assertion 'max > min' failed"))
        self.assertFalse(nightjobs._ignored("llama-server: CUDA error: out of memory"))

    def test_timer_calendars(self):
        with mock.patch.object(nightjobs, "sh", return_value="*-*-* 01:00:00"):
            self.assertEqual(nightjobs.timer_calendars("see `x.timer` here"), "x.timer: *-*-* 01:00:00")

    def test_production_model_id_reads_first_id_from_v1_models(self):
        with mock.patch("urllib.request.urlopen") as up:
            up.return_value = self._resp({"data": [{"id": "qwen3.8-27b"}, {"id": "other"}]})
            self.assertEqual(nightjobs.production_model_id(), "qwen3.8-27b")

    def test_production_model_id_falls_back_when_server_unreachable(self):
        with mock.patch("urllib.request.urlopen", side_effect=RuntimeError("production down")):
            self.assertEqual(nightjobs.production_model_id(), "production")

    def test_production_model_id_falls_back_when_no_models_listed(self):
        with mock.patch("urllib.request.urlopen") as up:
            up.return_value = self._resp({"data": []})
            self.assertEqual(nightjobs.production_model_id(), "production")

    def test_llm_sends_production_model_id_in_request_body(self):
        captured = {}
        def fake_urlopen(req, timeout=None):
            if req.full_url.endswith("/v1/models"):
                return self._resp({"data": [{"id": "qwen3.8-27b"}]})
            captured["body"] = json.loads(req.data.decode())
            return self._resp({"choices": [{"message": {"content": "ok"}}]})
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            self.assertEqual(nightjobs.llm("hi"), "ok")
        self.assertEqual(captured["body"]["model"], "qwen3.8-27b")
        self.assertIn("messages", captured["body"])
        self.assertIn("max_tokens", captured["body"])

    def test_llm_uses_fallback_model_when_models_endpoint_fails(self):
        captured = {}
        def fake_urlopen(req, timeout=None):
            if req.full_url.endswith("/v1/models"):
                raise RuntimeError("no /v1/models")
            captured["body"] = json.loads(req.data.decode())
            return self._resp({"choices": [{"message": {"content": "ok"}}]})
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            self.assertEqual(nightjobs.llm("hi"), "ok")
        self.assertEqual(captured["body"]["model"], "production")

    def test_regression_passes_model_to_bench(self):
        calls = {}
        def fake_run(cmd, **kwargs):
            calls["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")
        ctx = nightjobs.Ctx(config.now() + nightjobs.timedelta(minutes=40), 40)
        with mock.patch.object(nightjobs, "production_model_id", return_value="qwen3.8-27b"), \
                mock.patch("subprocess.run", side_effect=fake_run):
            res = nightjobs.regression(ctx)
        cmd = calls["cmd"]
        self.assertIn("--model", cmd)
        self.assertEqual(cmd[cmd.index("--model") + 1], "qwen3.8-27b")
        self.assertTrue(res["notable"])  # bench.py failed -> flagged


if __name__ == "__main__":
    unittest.main()
