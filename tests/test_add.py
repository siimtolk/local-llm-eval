"""Offline tests for `candidates.py add` and fetch.py's test_now hand-off (stdlib only, no network).

Run from the repo root: python3 -m unittest discover -s tests -v
"""
import atexit
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
TMP = Path(tempfile.mkdtemp(prefix="lle-test-add-"))
atexit.register(shutil.rmtree, TMP, ignore_errors=True)
SERVING = TMP / "serving"
for d in ("models", "models.d", "logs"):
    (SERVING / d).mkdir(parents=True)
CONF = TMP / "local.toml"
lines = []
for line in (ROOT / "local.example.toml").read_text().splitlines():
    key = line.split("=")[0].strip()
    if key == "store":
        line = f'store = "{TMP / "state" / "candidates.json"}"'
    elif key == "serving" and line.startswith("serving"):
        line = f'serving = "{SERVING}"'
    lines.append(line)
CONF.write_text("\n".join(lines) + "\n")
os.environ["LLE_CONFIG"] = str(CONF)
sys.path.insert(0, str(ROOT / "pipeline"))

import candidates  # noqa: E402
import config  # noqa: E402
import fetch  # noqa: E402

REPO = "someone/Model-27B-GGUF"
GB = 1e9
FITTING = {"Model-27B-Q4_K_M.gguf": 16.2 * GB, "Model-27B-Q8_0.gguf": 28.7 * GB, "README.md": 1000}
TOO_BIG = {"Huge-400B-Q4_K_M.gguf": 230 * GB}  # above machine.offload_gb: no active-parameter lookup
LOW = {"level": "low", "reasons": [], "checked": "2026-10-24T00:00:00+00:00"}


def store():
    path = config.path("store")
    return json.loads(path.read_text()) if path.exists() else {"candidates": {}, "read": []}


def put(repo, **record):
    with candidates.store() as data:
        data["candidates"][repo] = {"name": repo.split("/")[1], "link": "", "brief": "2026-10-01",
                                    "updated": candidates.now(), "summary": "", "reported": True} | record


class Base(unittest.TestCase):
    def setUp(self):
        config.path("store").unlink(missing_ok=True)
        self.listing = dict(FITTING)
        self.risk = dict(LOW)
        patches = [mock.patch.object(fetch, "all_files", side_effect=lambda repo: self.listing),
                   mock.patch.object(fetch, "vllm_pick", return_value=("", 0, [], "no safetensors")),
                   mock.patch("risk.safe_assess", side_effect=lambda repo: self.risk),
                   mock.patch("params.active_b", return_value=None)]
        self.mocks = [p.start() for p in patches]
        for p in patches:
            self.addCleanup(p.stop)


class TestAdd(Base):
    def cli(self, *args):
        out = io.StringIO()
        with mock.patch.object(sys, "argv", ["candidates.py", "add", *args]), redirect_stdout(out):
            with self.assertRaises(SystemExit) as ex:
                candidates.main()
        return ex.exception.code, out.getvalue().strip()

    def test_fitting_gguf_is_queued(self):
        code, line = self.cli(REPO)
        self.assertEqual(code, 0, line)
        self.assertIn("Q4_K_M 16.2 GB", line)
        c = store()["candidates"][REPO]
        self.assertEqual((c["status"], c["priority"], c["picked_by"], c["quant"], c["size_gb"]),
                         ("pending", 1, "add", "Q4_K_M", 16.2))
        self.assertTrue(c["reported"])
        self.assertNotIn("backend", c)
        self.assertNotIn("test_now", c)
        self.assertEqual(c["risk"]["level"], "low")
        self.assertRegex(c["brief"], r"^\d{4}-\d{2}-\d{2}$")

    def test_too_big_is_refused(self):
        self.listing = dict(TOO_BIG)
        code, line = self.cli("someone/Huge-400B-GGUF")
        self.assertEqual(code, 1)
        self.assertIn("Not queued", line)
        self.assertEqual(store()["candidates"], {})

    def test_invalid_repo_is_refused_before_network(self):
        code, line = self.cli("../x")
        self.assertEqual(code, 1)
        self.assertIn("Not a Hugging Face repo id", line)
        self.mocks[0].assert_not_called()

    def test_capy_is_refused(self):
        code, line = self.cli(REPO, "--backend", "capy")
        self.assertEqual(code, 1)
        self.assertIn("capy", line)
        self.assertEqual(store()["candidates"], {})

    def test_already_pending_is_refused_with_status(self):
        put(REPO, status="downloading")
        code, line = self.cli(REPO)
        self.assertEqual(code, 1)
        self.assertIn("downloading", line)
        self.assertEqual(store()["candidates"][REPO]["status"], "downloading")

    def test_rejected_is_reset_to_pending(self):
        put(REPO, status="rejected", model="model-27b", summary="old verdict", vllm_checked="x",
            quant_override="Q4_K_M", tested_at="2026-10-02", picked_by="poll-1")
        code, line = self.cli(REPO)
        self.assertEqual(code, 0, line)
        c = store()["candidates"][REPO]
        self.assertEqual((c["status"], c["picked_by"], c["summary"]), ("pending", "add", ""))
        self.assertNotIn("model", c)
        self.assertNotIn("vllm_checked", c)
        self.assertEqual((c["quant_override"], c["tested_at"]), ("Q4_K_M", "2026-10-02"))

    def test_now_sets_test_now(self):
        code, line = self.cli(REPO, "--now")
        self.assertEqual(code, 0, line)
        self.assertIn("/stopbkgtasks", line)
        self.assertIs(store()["candidates"][REPO]["test_now"], True)

    def test_priority_is_stored(self):
        self.assertEqual(self.cli(REPO, "--priority", "5")[0], 0)
        self.assertEqual(store()["candidates"][REPO]["priority"], 5)

    def test_no_gguf_falls_back_to_vllm(self):
        self.listing = {"model.safetensors": 20 * GB, "config.json": 100}
        self.mocks[1].return_value = ("nvfp4", 20.0, ["model.safetensors", "config.json"], "")
        code, line = self.cli("someone/Model-27B-NVFP4")
        self.assertEqual(code, 0, line)
        c = store()["candidates"]["someone/Model-27B-NVFP4"]
        self.assertEqual((c["backend"], c["quant"], c["status"]), ("vllm", "nvfp4", "pending"))

    def test_wontload_is_refused(self):
        self.risk = {"level": "wontload", "reasons": ["non-standard tensor type"]}
        code, line = self.cli(REPO)
        self.assertEqual(code, 1)
        self.assertIn("will not load", line)
        self.assertEqual(store()["candidates"], {})

    def test_high_risk_is_queued_and_shown(self):
        self.risk = {"level": "high", "reasons": ["little-known uploader"]}
        code, line = self.cli(REPO)
        self.assertEqual(code, 0, line)
        self.assertIn("high risk", line)

    def test_listing_failure_is_refused(self):
        self.mocks[0].side_effect = OSError("HTTP Error 404")
        code, line = self.cli(REPO)
        self.assertEqual(code, 1)
        self.assertIn("404", line)


class TestFetch(Base):
    """fetch.main with the download, bkg-start and the chat mocked."""

    def setUp(self):
        super().setUp()
        self.calls = []
        self.during_download = None
        patches = [mock.patch.object(fetch.subprocess, "run", side_effect=self.fake_run),
                   mock.patch.object(fetch.shutil, "disk_usage", return_value=mock.Mock(free=1e15)),
                   mock.patch.object(fetch, "add_sibling"),
                   mock.patch.object(fetch.notify, "send"),
                   mock.patch.object(sys, "argv", ["fetch.py"])]
        mocks = [p.start() for p in patches]
        self.send = mocks[3]
        for p in patches:
            self.addCleanup(p.stop)

    def fake_run(self, argv, **kw):
        self.calls.append(argv)
        if "download" in argv and self.during_download:
            hook, self.during_download = self.during_download, None
            hook()
        return subprocess.CompletedProcess(argv, 0, stdout="Background tasks started.\n", stderr="")

    def bkg_starts(self):
        return [a for a in self.calls if "bkg-start" in a]

    def main(self):
        with redirect_stdout(io.StringIO()):
            fetch.main()

    def test_test_now_starts_bkg_once_after_download(self):
        with redirect_stdout(io.StringIO()):
            self.assertTrue(candidates.add("a/One-GGUF", now_=True)[0])
            self.assertTrue(candidates.add("b/Two-GGUF", now_=True)[0])
        self.main()
        data = store()["candidates"]
        self.assertEqual([data[r]["status"] for r in ("a/One-GGUF", "b/Two-GGUF")], ["downloaded"] * 2)
        self.assertFalse(any(data[r].get("test_now") for r in data))
        self.assertEqual(len(self.bkg_starts()), 1)
        self.send.assert_called_once()
        self.assertIn("Background tasks started.", self.send.call_args[0][0])

    def test_no_test_now_no_bkg(self):
        with redirect_stdout(io.StringIO()):
            candidates.add("a/One-GGUF")
        self.main()
        self.assertEqual(store()["candidates"]["a/One-GGUF"]["status"], "downloaded")
        self.assertEqual(self.bkg_starts(), [])
        self.send.assert_not_called()

    def test_dry_run_changes_nothing(self):
        with redirect_stdout(io.StringIO()):
            candidates.add("a/One-GGUF", now_=True)
        with mock.patch.object(sys, "argv", ["fetch.py", "--dry-run"]):
            self.main()
        self.assertEqual(store()["candidates"]["a/One-GGUF"]["status"], "pending")
        self.assertEqual(self.calls, [])

    def test_repo_added_during_a_download_is_taken_in_the_same_run(self):
        put("a/One-GGUF", status="pending")
        self.during_download = lambda: candidates.add("b/Late-GGUF", now_=True)
        self.main()
        data = store()["candidates"]
        self.assertEqual(data["b/Late-GGUF"]["status"], "downloaded")
        self.assertEqual(len(self.bkg_starts()), 1)

    def test_test_now_may_take_one_slot_over_max_ready(self):
        for i in range(fetch.MAX_READY):
            put(f"x/Ready{i}-GGUF", status="downloaded", model=f"ready{i}")
        put("a/Plain-GGUF", status="pending")
        with redirect_stdout(io.StringIO()):
            candidates.add("b/Now-GGUF", now_=True)
        self.main()
        data = store()["candidates"]
        self.assertEqual(data["b/Now-GGUF"]["status"], "downloaded")
        self.assertEqual(data["a/Plain-GGUF"]["status"], "pending")


if __name__ == "__main__":
    unittest.main()
