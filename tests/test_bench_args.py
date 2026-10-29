"""Offline tests for the bench/bench.py argument guards (stdlib only, no network).

An empty --cases and an empty --api-key-file used to blow up with a traceback
(IndexError / IsADirectoryError); they now fail early through argparse with a
clear one-line error (exit code 2). Run from the repo root:
    python3 -m unittest tests.test_bench_args -q
"""
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "bench" / "bench.py"


def _run(args):
    """Run bench.py with the given arguments and return the completed process."""
    return subprocess.run(
        [sys.executable, str(BENCH), *args],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


class BenchArgsTest(unittest.TestCase):
    def test_empty_cases(self):
        proc = _run(["--label", "t", "--cases", ""])
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("--cases names no file", proc.stderr)

    def test_empty_api_key_file(self):
        proc = _run(["--label", "t", "--api-key-file", ""])
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("no API key file", proc.stderr)


if __name__ == "__main__":
    unittest.main()
