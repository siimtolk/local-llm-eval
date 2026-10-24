"""Offline tests for the deferred agent exit in pipeline/runner.py (stdlib only, no network).

On a settings dict from local.example.toml with the stores in a temp dir (config.load patched),
runner.agent is monkeypatched to return a deferral (or a normal unfinished run) and the store is
checked: a deferred candidate ends `downloaded` (no summary), an unfinished one still ends
`failed`. Run from the repo root: python3 -m unittest tests.test_deferred -v
"""
import atexit
import shutil
import sys
import tempfile
import tomllib
import unittest
from datetime import timedelta
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
TMP = Path(tempfile.mkdtemp(prefix="lle-test-deferred-"))
atexit.register(shutil.rmtree, TMP, ignore_errors=True)
sys.path.insert(0, str(ROOT / "pipeline"))

import candidates  # noqa: E402
import config  # noqa: E402
import runner  # noqa: E402

REPO = "a/Model-GGUF"


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(dir=TMP))
        with open(ROOT / "local.example.toml", "rb") as f:
            conf = tomllib.load(f)
        conf["paths"].update(store=str(self.dir / "state" / "candidates.json"),
                             state=str(self.dir / "state"),
                             experiments=str(self.dir / "state" / "experiments.json"),
                             results=str(self.dir / "results"), serving=str(self.dir / "serving"))
        self.conf = conf
        patch = mock.patch.object(config, "load", return_value=conf)
        patch.start()
        self.addCleanup(patch.stop)
        with candidates.store() as data:
            data["candidates"][REPO] = {"name": "Model", "status": "downloaded",
                                        "model": "model-27b", "brief": "2026-10-01"}

    def cands(self):
        with candidates.store(readonly=True) as data:
            return data["candidates"]

    def run_test_one(self, agent_return):
        with mock.patch.object(runner, "agent", return_value=agent_return), \
                mock.patch.object(runner.subprocess, "run"), \
                mock.patch.object(runner, "production_ensure"), \
                mock.patch.object(runner, "stop_cmd", return_value=["true"]), \
                mock.patch.object(runner, "log"):
            return runner.test_one(REPO, config.now() + timedelta(hours=2))


class TestDeferred(Base):
    def test_deferred_run_goes_back_to_downloaded(self):
        deferred = self.run_test_one((None, "exit 75: provider usage limit reached", True))
        c = self.cands()[REPO]
        self.assertTrue(deferred)
        self.assertEqual(c["status"], "downloaded")
        self.assertNotIn("summary", c)  # not tested, so nothing to report yet

    def test_deferred_run_stops_the_model_test_run(self):
        with mock.patch.object(runner, "test_one", return_value=True) as one:
            runner.model_test(now=True, end=config.now() + timedelta(hours=2))
        self.assertEqual(one.call_count, 1)  # the loop returned after the deferral

    def test_unfinished_run_still_failed(self):
        deferred = self.run_test_one((None, "stopped after 60 min", False))
        c = self.cands()[REPO]
        self.assertFalse(deferred)
        self.assertEqual(c["status"], "failed")
        self.assertIn("stopped after 60 min", c["summary"])

    def test_deferred_exit_default_is_75(self):
        self.assertEqual(self.conf["agent"]["deferred_exit"], 75)


if __name__ == "__main__":
    unittest.main()
