#!/usr/bin/env python3
"""Unfinished benchmark runs: the sections a result file still lacks, queued for the next night.

A benchmark has no length cap: it runs until it is done or the night's stop time comes (bench/NIGHTLY.md
step 6, tools/bench_bg.sh). Whatever it did not reach is continued the next night: after every model test
and experiment, runner.py calls scan() on the result files that run wrote, and each label that still lacks
sections of the full suite gets an entry here. runner.py model-test takes the entries like candidates
(kind "resume", priority as on the candidates' scale): the agent follows bench/RESUME.md, serving the same
model again and running bench.py --only <the missing sections> with the same label; bench.py's per-request
checkpoint continues a section that was cut off in the middle. When nothing is missing the entry is done;
an entry that makes no progress for MAX_STALLED runs is dropped with a log line (e.g. a section that is
skipped on this machine every time).

The full suite of a label is speed plus every section of the default case files (bench/cases.py,
cases_v2.py, cases_private.py); of a `_nothink` label, NOTHINK. Present = stored in the result file (top
level, `extra` or `own_traffic`); a section bench.py listed under `skipped` still counts as missing.

  resume.py list
  resume.py missing <label>                         the sections <label> lacks
  resume.py add <label> [--sections a,b] [--model M] [--priority N] [--backend B]
                                                    queue a label by hand (default: what it lacks)
  resume.py done <label> '<summary>'                the agent's record at the end of a resume run
  resume.py drop <label>
"""
import ast
import fcntl
import json
import re
import shlex
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import config

ROOT = config.ROOT
CASE_FILES = ("cases.py", "cases_v2.py", "cases_private.py")
NOTHINK = ("tools", "extract", "instruct", "judge", "estonian")  # bench/NIGHTLY.md step 6, thinking off
LABEL = re.compile(r"[a-z0-9][a-z0-9._-]{1,150}")
MAX_STALLED = 3
PRIORITY = 1  # like `candidates.py add`: before judge picks (0)


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sections_of(case_file):
    """Section names of a case file in run order, read from its SECTIONS dict (the file is not imported)."""
    tree = ast.parse((ROOT / "bench" / case_file).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "SECTIONS" for t in node.targets) \
                and isinstance(node.value, ast.Dict):
            return [k.value for k in node.value.keys if isinstance(k, ast.Constant)]
    return []


def suite_sections():
    return [s for f in CASE_FILES for s in sections_of(f)]


def expected(label):
    return list(NOTHINK) if label.endswith("_nothink") else ["speed"] + suite_sections()


def result_path(label):
    return config.path("results") / f"{label}.json"


def present(res):
    have = {k for k in res if k not in ("extra", "own_traffic")} | set(res.get("extra") or {}) | set(res.get("own_traffic") or {})
    return have


def missing(label):
    p = result_path(label)
    res = json.loads(p.read_text()) if p.exists() else {}
    have = present(res if isinstance(res, dict) else {})
    return [s for s in expected(label) if s not in have]


def checkpointed(label):
    """Finished requests kept in bench.py's checkpoints for this label (progress inside a cut-off section)."""
    d = config.path("state") / "bench-partial" / label
    return sum(len(p.read_text().splitlines()) for p in d.glob("*.jsonl")) if d.is_dir() else 0


def store_path():
    return config.path("state") / "resume.json"


@contextmanager
def store(readonly=False):
    p = store_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH if readonly else fcntl.LOCK_EX)
        data = json.loads(p.read_text()) if p.exists() else {}
        data.setdefault("runs", {})
        yield data
        if not readonly:
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False))
            tmp.replace(p)


def model_of(label):
    """(kind, model or experiment id) from a label: exp-<id>[...] or <model>_<quant>_..."""
    base = label.removesuffix("_nothink")
    if base.startswith("exp-"):
        return "experiment", base[4:].split("_")[0]
    return "model", base.split("_")[0]


def add(label, sections=None, model=None, priority=PRIORITY, backend=None, source="hand"):
    """Queue a label (or update its entry). Returns (ok, text)."""
    if not LABEL.fullmatch(label or "") or ".." in label:
        return False, f"not a result label: {label!r}"
    known = set(expected(label))
    sections = list(sections or missing(label))
    bad = [s for s in sections if s not in known]
    if bad:
        return False, f"not sections of {label}'s suite: {', '.join(bad)}"
    if not sections:
        return False, f"{label} lacks nothing"
    kind, guess = model_of(label)
    with store() as data:
        e = data["runs"].get(label) or {"created": now(), "stalled": 0, "tries": 0}
        e.update(label=label, kind=kind, model=model or e.get("model") or guess, sections=sections,
                 priority=int(priority), backend=backend or e.get("backend") or "", source=source, updated=now())
        data["runs"][label] = e
    return True, f"queued {label}: {', '.join(sections)}"


def scan(results, since_ts, kind, key, model, backend="", priority=PRIORITY):
    """After a test: queue the labels among results (result file paths) written since since_ts (epoch) that
    lack sections. A model test's main label (the newest such file with core sections) also brings its
    _nothink label, written or not. Returns the queued labels."""
    core = set(sections_of(CASE_FILES[0]))
    fresh = [p for p in results if p.exists() and p.stat().st_mtime >= since_ts]
    labels = set()
    for p in fresh:
        try:
            res = json.loads(p.read_text())
        except ValueError:
            continue
        if not isinstance(res, dict) or not res.get("suite"):
            continue
        if p.stem.endswith("_nothink") or present(res) & core:
            labels.add(p.stem)
    if kind == "model":
        mains = [l for l in labels if not l.endswith("_nothink")]
        labels |= {f"{l}_nothink" for l in mains}
    out = []
    for label in sorted(labels):
        if missing(label) and add(label, model=model, priority=priority, backend=backend, source=f"{kind} {key}")[0]:
            out.append(label)
    return out


def queued():
    with store(readonly=True) as data:
        return {k: dict(v) for k, v in data["runs"].items()}


def mark(label, **fields):
    with store() as data:
        if label in data["runs"]:
            data["runs"][label].update(fields, updated=now())


def finish(label, before):
    """After a resume run: drop the entry when nothing is missing, else count a run without progress.
    before: (missing sections, checkpointed requests) at the start. Returns (state, text) with state done,
    queued or dropped."""
    left, cp = missing(label), checkpointed(label)
    with store() as data:
        e = data["runs"].get(label)
        if e is None:
            return "done", f"{label}: no entry"
        e.pop("running", None)
        e["tries"] = e.get("tries", 0) + 1
        if not left:
            del data["runs"][label]
            return "done", f"{label}: complete"
        progress = len(left) < len(before[0]) or cp > before[1]
        e["stalled"] = 0 if progress else e.get("stalled", 0) + 1
        e.update(sections=left, updated=now())
        if e["stalled"] >= MAX_STALLED:
            del data["runs"][label]
            return "dropped", f"{label}: no progress in {MAX_STALLED} runs, dropped; still missing {', '.join(left)}"
        return "queued", f"{label}: still missing {', '.join(left)} (next night)"


def bench_command(label, sections, test_port, api_key_file):
    """The bench.py command that continues label: the same reps, temperature and request extras as its result file."""
    p = result_path(label)
    res = json.loads(p.read_text()) if p.exists() else {}
    cmd = ["bench/bench.py", "--label", label, "--only", ",".join(sections),
           "--base-url", f"http://127.0.0.1:{test_port}", "--api-key-file", str(api_key_file),
           "--reps", str(res.get("reps") or 3)]
    if res.get("temperature") is not None:
        cmd += ["--temperature", str(res["temperature"])]
    if res.get("request_extra"):
        cmd += ["--request-extra", json.dumps(res["request_extra"], ensure_ascii=False)]
    elif label.endswith("_nothink"):
        cmd += ["--thinking", "off"]
    return shlex.join(cmd)


def main():
    a = sys.argv[1:]
    if a == ["list"]:
        runs = queued()
        for label, e in sorted(runs.items(), key=lambda kv: (-kv[1].get("priority", 0), kv[1].get("created", ""))):
            print(f"{label}: {', '.join(e['sections'])} (model {e['model']}, priority {e.get('priority', 0)}, "
                  f"tries {e.get('tries', 0)}, stalled {e.get('stalled', 0)}{', running' if e.get('running') else ''})")
        print(f"{len(runs)} queued" if runs else "nothing queued")
    elif len(a) == 2 and a[0] == "missing":
        print(", ".join(missing(a[1])) or "nothing missing")
    elif len(a) >= 2 and a[0] == "add":
        opts = dict(zip(a[2::2], a[3::2]))
        if set(opts) - {"--sections", "--model", "--priority", "--backend"} or len(a[2:]) % 2:
            sys.exit(__doc__)
        ok, text = add(a[1], [s for s in opts.get("--sections", "").split(",") if s] or None, opts.get("--model"),
                       int(opts.get("--priority", PRIORITY)), opts.get("--backend"))
        print(text)
        sys.exit(0 if ok else 1)
    elif len(a) == 3 and a[0] == "done":
        mark(a[1], recorded=now(), summary=a[2][:800])
        print(f"recorded {a[1]}")
    elif len(a) == 2 and a[0] == "drop":
        with store() as data:
            print(f"dropped {a[1]}" if data["runs"].pop(a[1], None) else f"{a[1]} not queued")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
