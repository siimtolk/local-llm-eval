"""Offline tests for pipeline/nightjobs.py: due/priority order, JSON parsing, runner integration.
Run from the repo root: python3 -m unittest tests.test_nightjobs -v"""
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
        p = mock.patch.object(config, "load", return_value=conf)
        p.start()
        self.addCleanup(p.stop)
        q = mock.patch.object(nightjobs, "STATE", self.tmp / "nightjobs")
        q.start()
        self.addCleanup(q.stop)

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


if __name__ == "__main__":
    unittest.main()
