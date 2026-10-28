"""Offline tests for tools/leak_check.sh (stdlib only, no network).

Builds a throwaway git repository, copies tools/leak_check.sh into it, and checks
that the generic sandbox path /home/user/ is ignored while every other home
directory is still flagged. Run from the repo root:
    python3 -m unittest tests.test_leak_check -q
"""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEAK_CHECK = ROOT / "tools" / "leak_check.sh"
# Built at runtime so this file does not trip the leak check itself.
OTHER_HOME = "/" + "home/alice/x"


def _git(args, cwd):
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed: {proc.stderr}")
    return proc


def _new_repo(root):
    """Fresh git repo in `root` with an initial commit and the leak check copied in."""
    _git(["init", "-q"], root)
    _git(["config", "user.email", "leakcheck@example.invalid"], root)
    _git(["config", "user.name", "Leak Check Test"], root)
    (root / "README.md").write_text("# sandbox repo\n", encoding="utf-8")
    _git(["add", "README.md"], root)
    _git(["commit", "-q", "-m", "initial"], root)
    (root / "tools").mkdir(exist_ok=True)
    shutil.copy(LEAK_CHECK, root / "tools" / "leak_check.sh")


class TestLeakCheck(unittest.TestCase):
    def test_home_user_ignored_other_home_flagged(self):
        with tempfile.TemporaryDirectory(prefix="lle-test-leakcheck-") as tmp:
            root = Path(tmp)
            _new_repo(root)

            # 1. A file with only the generic sandbox path /home/user/ must pass.
            (root / "sandbox.txt").write_text(
                "cd /home/user/project && python3 -m unittest -q\n", encoding="utf-8"
            )
            _git(["add", "sandbox.txt"], root)
            ok = subprocess.run(
                ["bash", "tools/leak_check.sh"],
                cwd=root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertEqual(
                ok.returncode,
                0,
                f"/home/user/ should be ignored; rc={ok.returncode}\n"
                f"{ok.stdout}\n{ok.stderr}",
            )

            # 2. A file with another home directory must be flagged.
            (root / "other.txt").write_text(f"data in {OTHER_HOME}\n", encoding="utf-8")
            _git(["add", "other.txt"], root)
            bad = subprocess.run(
                ["bash", "tools/leak_check.sh"],
                cwd=root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertEqual(
                bad.returncode,
                1,
                f"{OTHER_HOME} should be flagged; rc={bad.returncode}\n"
                f"{bad.stdout}\n{bad.stderr}",
            )


if __name__ == "__main__":
    unittest.main()
