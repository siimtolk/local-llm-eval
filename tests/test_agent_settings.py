"""Offline tests for tools/agent_settings.py: force-push and no-verify are denied, also through git -C.

Run from the repo root: python3 -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import agent_settings  # noqa: E402


class DenyRules(unittest.TestCase):
    def setUp(self):
        self.serving = agent_settings.config.serving()
        self.perms = agent_settings.rules()["permissions"]

    def test_git_c_and_late_flag_forms_denied(self):
        s = self.serving
        expected = [
            f"Bash(git -C {s} push --force*)", f"Bash(git -C {s} push -f*)",
            f"Bash(git -C {s} push *--force*)", f"Bash(git -C {s} push * -f*)",
            f"Bash(git -C {s} commit --no-verify*)", f"Bash(git -C {s} commit -n*)",
            f"Bash(git -C {s} commit *--no-verify*)",
            "Bash(git push *--force*)", "Bash(git push * -f*)", "Bash(git commit *--no-verify*)",
        ]
        deny = self.perms["deny"]
        for rule in expected:
            self.assertIn(rule, deny)

    def test_git_c_allow_rule_kept(self):
        self.assertIn(f"Bash(git -C {self.serving} *)", self.perms["allow"])


if __name__ == "__main__":
    unittest.main()
