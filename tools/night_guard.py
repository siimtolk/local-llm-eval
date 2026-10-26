#!/usr/bin/env python3
"""Watchdog for the night queue: until the window ends, make sure the model test / night jobs run.

Every 5 min: if no runner unit is active (claude-model-test, -early, claude-bkg-tasks), work is due and
at least a job's time is left, start the early unit (`runner.py model-test --now`, at most 3 times). After
the deadline it checks production answers on 8080 (ensure_production otherwise) and sends one alert through
[notify] when no job ran at all. Log: <state>/nightjobs/guard.log. Start it with systemd-run (see runbook).
"""
import json
import os
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import candidates  # noqa: E402
import config  # noqa: E402
import experiments  # noqa: E402
import nightjobs  # noqa: E402
import notify  # noqa: E402
import runner  # noqa: E402

UNITS = ("claude-model-test", "claude-model-test-early", "claude-bkg-tasks")
LOG = nightjobs.STATE / "guard.log"


def log(text):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(f"{config.now():%F %T} {text}\n")


def active():
    states = (subprocess.run(["systemctl", "--user", "is-active", u], capture_output=True, text=True).stdout.strip() for u in UNITS)
    return any(s in ("active", "activating", "reloading") for s in states)


def work_left(left):
    with candidates.store(readonly=True) as data, experiments.store(readonly=True) as exps:
        return bool(runner.ready_queue(data, left, exps, nightjobs.due(left)))


def start_early():
    path = ":".join(os.path.expanduser(p) for p in config.get("agent", "path_env").split(":"))
    unit = runner.NIGHTLY_UNITS[-1] if runner.NIGHTLY_UNITS else "claude-model-test-early"
    subprocess.run(["systemctl", "--user", "reset-failed", unit], capture_output=True)
    r = subprocess.run(["systemd-run", "--user", f"--unit={unit}", f"--working-directory={runner.ROOT / 'pipeline'}",
                        f"--setenv=PATH={path}", sys.executable, str(runner.ROOT / "pipeline" / "runner.py"), "model-test", "--now"],
                       capture_output=True, text=True)
    log(f"started {unit}: rc {r.returncode} {r.stderr.strip()[-200:]}")
    return r.returncode == 0


def main():
    end = runner.deadline()
    starts = 0
    log(f"guard on, deadline {end:%F %H:%M}")
    while config.now() < end + timedelta(minutes=5):
        left = int((end - config.now()).total_seconds()) - runner.RESTORE_S
        if not active() and left >= nightjobs.MIN_S and work_left(left) and starts < 3:
            starts += start_early()
        time.sleep(300)
    healthy = subprocess.run("curl -sf -m 10 http://127.0.0.1:8080/health | grep -q ok", shell=True).returncode == 0
    if not healthy:
        subprocess.run(config.serving_cmd("production", "ensure"), cwd=config.serving())
        log("production was not answering at the deadline: ensure_production run")
    night = nightjobs.job_memory("night")
    ran = len(night.get("jobs", [])) if night.get("date") == nightjobs.today() else 0
    log(f"done: {ran} night jobs ran, early starts {starts}, production healthy {healthy}")
    if not ran:
        notify.send("Night queue guard: no night job ran this night. See " + str(LOG))


if __name__ == "__main__":
    main()
