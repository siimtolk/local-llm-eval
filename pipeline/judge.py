#!/usr/bin/env python3
"""Stage 2, judge: an LLM reads discover's output and picks at most three candidates worth testing.

Runs discover.py, sends its output followed by prompts/judge.md to [judge] command (stdin in,
reply on stdout), stores the reply in paths.briefs/<date>_<time>.md (candidates.py index reads the
picks from there) and sends it with [notify]. When discover finds nothing new, no model is called:
a short "nothing new" message is sent instead. Model cards and posts are data for the judge,
never instructions; the prompt says so.

Usage: judge.py [--dry-run]   (--dry-run: print the judge input, call nothing, change nothing)
"""
import shlex
import subprocess
import sys
import time

import config
import discover
import notify

PROMPT = config.ROOT / "prompts" / "judge.md"
TIMEOUT_S = 15 * 60


def prompt():
    return PROMPT.read_text().replace("{{languages}}", config.get("judge", "languages", "English"))


def main():
    dry = "--dry-run" in sys.argv
    text, quiet, _ = discover.build(dry=dry)
    if not text:
        print(quiet) if dry else notify.send(quiet)
        return
    full = f"{text}\n\n{prompt()}"
    if dry:
        print(full)
        return
    cmd = config.get("judge", "command")
    run = subprocess.run(shlex.split(cmd) if isinstance(cmd, str) else cmd, input=full, capture_output=True,
                         text=True, timeout=TIMEOUT_S)
    reply = run.stdout.strip()
    if run.returncode or not reply:
        sys.exit(f"judge: exit {run.returncode}: {(run.stderr or run.stdout).strip()[-500:]}")
    briefs = config.path("briefs")
    briefs.mkdir(parents=True, exist_ok=True)
    (briefs / f"{time.strftime('%Y-%m-%d_%H-%M-%S')}.md").write_text(reply + "\n")
    discover.mark_judged()
    notify.send(reply)


if __name__ == "__main__":
    main()
