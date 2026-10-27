#!/usr/bin/env python3
"""Experiments: anything tested against production with the bench suite and a pass rule in code.

Three types. A `model` experiment is a candidate of the candidates store (candidates.py, unchanged:
add, fetch, votes and the report work as before). `config` (e.g. a KV cache type, sampling, a newer
llama.cpp build) and `engine` (another backend) experiments live in their own store,
[paths] experiments, under the same lock helper (candidates.locked_json). They run on the test
port with the same suite (bench/EXPERIMENT.md) and never change production.

Status: planned -> approved -> queued -> running -> beneficial | not-beneficial | failed, plus dropped.
A candidate shows as a model experiment with its status mapped (UNIFIED).

The verdict of a config/engine experiment is computed here, never by the agent: `verdict` reads
the variant's result JSON and the [production] baseline's, computes the metrics and applies the
spec's pass rule (min_delta / max_delta: variant - baseline, min_ratio / max_ratio: variant /
baseline). Results of different suites are never compared: the verdict is then `failed`.

Usage: experiments.py list [--json]
       experiments.py show <id|repo> [--json]
       experiments.py add --spec-file <json>               (validated, stored as planned; prints the id)
       experiments.py add --type model <repo> [--backend B] [--now]   (candidates.add)
       experiments.py approve <id>                         (planned -> approved)
       experiments.py drop <id> ['<reason>']
       experiments.py queue                                (every approved experiment -> queued)
       experiments.py set <id> <status> ['<summary>']      (for the agent; not the verdict statuses)
       experiments.py verdict <id> <result label>          (metrics, pass rule, final status)

After a final status a note goes to the wiki inbox ([wiki] inbox, optional) and
[experiments] verdict_command, if set, runs with the id (a repo, for a model) as its last argument.
"""
import json
import math
import re
import shlex
import subprocess
import sys
from pathlib import Path

import candidates
import config
from texts import t

TYPES = ("config", "engine")
STATUSES = ("planned", "approved", "queued", "running", "beneficial", "not-beneficial", "failed", "dropped")
FINAL = ("beneficial", "not-beneficial", "failed")
VERDICT_ONLY = ("beneficial", "not-beneficial")  # only `verdict` sets these
UNIFIED = {"proposed": "planned", "pending": "approved", "downloading": "approved", "downloaded": "queued",
           "testing": "running", "tested": "beneficial", "rejected": "not-beneficial", "skipped": "dropped",
           "failed": "failed"}
MODEL_FINAL = ("tested", "rejected", "failed")
SLUG = re.compile(r"[a-z0-9-]{3,40}")
ID = re.compile(r"\d{8}-[a-z0-9-]{3,40}")
LABEL = re.compile(r"[A-Za-z0-9][\w.+-]{0,200}")
# bench/cases.py SECTIONS and bench.py speed() tests, the names a result JSON has
CATEGORIES = ("tools", "agent", "reason", "extract", "code", "hard", "instruct", "judge", "estonian", "uncensored")
SPEED_TESTS = ("prose", "tool_json", "long_prompt", "parallel", "alternating", "deep_decode")
# bench/cases_v2.py sections are stored under the result's "extra" key, bench/cases_private.py sections (your own traffic)
# under "own_traffic"; both stay outside the core totals
EXTRA_SECTIONS = ("robust", "longagent", "estonian_free", "vision")
TRAFFIC_SECTIONS = ("replay", "cron_replay", "ctx")
KNOWN_METRICS = ({"passed"} | {f"passed.{c}" for c in CATEGORIES} | {f"extra.{s}" for s in EXTRA_SECTIONS}
                 | {f"own_traffic.{s}" for s in TRAFFIC_SECTIONS}
                 | {f"{m}.{s}" for m in ("tok_s", "ttft") for s in SPEED_TESTS})
OPS = ("min_delta", "max_delta", "min_ratio", "max_ratio")
VARIANT_KEYS = ("llama_cpp", "server_args_add", "server_args_remove", "request_extra", "backend", "notes")
BACKENDS = ("llamacpp", "vllm", "capy")
SLOW_BACKENDS = ("vllm", "capy")  # a first start compiles kernels: the longer test slot


def store_path():
    """[paths] experiments; a local.toml from before experiments: <paths.state>/experiments.json."""
    return config.path("experiments") if config.get("paths", "experiments") else config.path("state") / "experiments.json"


def store(readonly=False):
    """The experiments store {"experiments": {id: record}} under its lock; saved on exit."""
    return candidates.locked_json(store_path(), {"experiments": {}}, readonly)


# --- pure functions -------------------------------------------------------------------------------

def unified_status(c):
    """A candidate's status as a model experiment's."""
    return UNIFIED.get(c.get("status"), "failed")


def _runs(test):
    """The run dicts of one speed test, whatever its shape (list, {"runs"}, {"turns"}, {depth: run})."""
    if isinstance(test, list):
        return [r for r in test if isinstance(r, dict)]
    if isinstance(test, dict):
        for key in ("runs", "turns"):
            if isinstance(test.get(key), list):
                return [r for r in test[key] if isinstance(r, dict)]
        return [r for r in test.values() if isinstance(r, dict)]
    return []


def _mean(values):
    values = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)]
    return sum(values) / len(values) if values else None


def metrics(result):
    """Every metric of a result JSON (the shape bench.py writes): passed, passed.<category>,
    extra.<section> / own_traffic.<section> (passed cases of a section under "extra" / "own_traffic"), tok_s.<speed test>, ttft.<speed test>.
    A test without numbers has no entry."""
    out, total = {}, 0
    for name, rows in result.items():
        if isinstance(rows, list) and rows and all(isinstance(r, dict) and "ok" in r for r in rows):
            out[f"passed.{name}"] = n = sum(1 for r in rows if r["ok"] is True)
            total += n
    if out:
        out["passed"] = total
    for group in ("extra", "own_traffic"):  # these sections never count in "passed"
        for name, rows in (result.get(group) or {}).items():
            if isinstance(rows, list) and rows and all(isinstance(r, dict) and "ok" in r for r in rows):
                out[f"{group}.{name}"] = sum(1 for r in rows if r["ok"] is True)
    for name, test in (result.get("speed") or {}).items():
        for field in ("tok_s", "ttft"):
            value = _mean(r.get(field) for r in _runs(test))
            if value is not None:
                out[f"{field}.{name}"] = value
    return out


def rule_text(rule):
    return " and ".join(f"{op} {rule[op]:g}" for op in OPS if op in rule)


def check(rule, variant_metrics, baseline_metrics):
    """(passed, text) for one pass-rule condition {"metric", "min_delta" | "max_delta" | "min_ratio" |
    "max_ratio": number, ...}; every operator given must hold. A missing value fails."""
    name = rule["metric"]
    v, b = variant_metrics.get(name), baseline_metrics.get(name)
    if v is None or b is None:
        return False, f"{name}: no value ({'variant' if v is None else 'baseline'})"
    ok, parts = True, []
    if "min_delta" in rule or "max_delta" in rule:
        delta = v - b
        ok &= delta >= rule.get("min_delta", -math.inf) and delta <= rule.get("max_delta", math.inf)
        parts.append(f"delta {delta:+.4g}")
    if "min_ratio" in rule or "max_ratio" in rule:
        ratio = v / b if b else None
        ok &= ratio is not None and rule.get("min_ratio", -math.inf) <= ratio <= rule.get("max_ratio", math.inf)
        parts.append(f"ratio {ratio:.4g}" if ratio is not None else "ratio n/a (baseline 0)")
    return bool(ok), f"{name} {v:.4g} vs {b:.4g}: {', '.join(parts)} ({rule_text(rule)})"


def evaluate(spec, variant, baseline):
    """(status, verdict dict, reason) for a spec against two result dicts."""
    names = list(dict.fromkeys(list(spec.get("metrics") or []) + [r["metric"] for r in spec["pass_rule"]]))
    if variant.get("suite") != baseline.get("suite"):
        return "failed", {"metrics": {}, "rules": {}, "passed": [], "failed": []}, t("exp.suite_differs")
    vm, bm = metrics(variant), metrics(baseline)
    verdict = {"metrics": {n: [vm.get(n), bm.get(n)] for n in names}, "rules": {}, "passed": [], "failed": []}
    missing = [n for n in names if vm.get(n) is None or bm.get(n) is None]
    for i, rule in enumerate(spec["pass_rule"]):
        key = rule["metric"] if rule["metric"] not in verdict["rules"] else f"{rule['metric']}#{i + 1}"
        ok, text = check(rule, vm, bm)
        verdict["rules"][key] = text
        verdict["passed" if ok else "failed"].append(key)
    if missing:
        return "failed", verdict, t("exp.missing_metric", names=", ".join(missing))
    return ("beneficial" if not verdict["failed"] else "not-beneficial"), verdict, ""


def _number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _arg_list(x):
    return isinstance(x, list) and all(isinstance(a, str) and a and not re.search(r"[\n\r\0]", a) for a in x)


def validate(spec):
    """A refusal line, or "" for a well-formed config/engine spec."""
    if not isinstance(spec, dict):
        return t("exp.bad_spec", why="not a JSON object")
    kind = spec.get("type")
    if kind == "model":
        return t("exp.model_spec")
    if kind not in TYPES:
        return t("exp.bad_spec", why=f"type must be one of {', '.join(TYPES)}")
    if not isinstance(spec.get("slug"), str) or not SLUG.fullmatch(spec["slug"]):
        return t("exp.bad_slug", slug=str(spec.get("slug"))[:60])
    if not isinstance(spec.get("title"), str) or not spec["title"].strip() or len(spec["title"]) > 200:
        return t("exp.bad_spec", why="title: a string of 1-200 characters")
    if not isinstance(spec.get("hypothesis", ""), str):
        return t("exp.bad_spec", why="hypothesis: a string")
    variant = spec.get("variant")
    if not isinstance(variant, dict) or set(variant) - set(VARIANT_KEYS):
        return t("exp.bad_spec", why=f"variant: an object with keys from {', '.join(VARIANT_KEYS)}")
    build = variant.get("llama_cpp", "")
    if not isinstance(build, str) or ".." in build or build.startswith(("/", "~", "\\")) or (build and not re.fullmatch(r"[\w.+/-]{1,200}", build)):
        return t("exp.bad_path", path=str(build)[:200])
    for key in ("server_args_add", "server_args_remove"):
        if not _arg_list(variant.get(key, [])):
            return t("exp.bad_arg", field=key)
    if not isinstance(variant.get("request_extra", {}), dict) or not isinstance(variant.get("notes", ""), str):
        return t("exp.bad_spec", why="variant.request_extra: an object, variant.notes: a string")
    if "backend" in variant and (kind != "engine" or variant["backend"] not in BACKENDS):
        return t("exp.bad_backend", backends=", ".join(BACKENDS))
    if kind == "engine" and not (variant.get("backend") or build):
        return t("exp.bad_spec", why="an engine experiment names variant.backend or variant.llama_cpp")
    names = spec.get("metrics", [])
    if not isinstance(names, list) or not all(isinstance(m, str) for m in names):
        return t("exp.bad_spec", why="metrics: a list of names")
    rules = spec.get("pass_rule")
    if not isinstance(rules, list) or not rules:
        return t("exp.bad_rule", why="pass_rule: a non-empty list")
    for rule in rules:
        if not isinstance(rule, dict) or not isinstance(rule.get("metric"), str) or set(rule) - {"metric", *OPS} \
                or not any(op in rule for op in OPS) or not all(_number(rule[op]) for op in OPS if op in rule):
            return t("exp.bad_rule", why=json.dumps(rule, ensure_ascii=False)[:200])
        names = names + [rule["metric"]]
    unknown = [m for m in names if m not in KNOWN_METRICS]
    if unknown:
        return t("exp.bad_metric", names=", ".join(sorted(set(unknown)))[:200])
    if not all(isinstance(c, str) for c in spec.get("checks", [])) or not isinstance(spec.get("checks", []), list):
        return t("exp.bad_spec", why="checks: a list of strings")
    cost = spec.get("cost", {})
    if not isinstance(cost, dict) or not all(_number(v) for v in cost.values()):
        return t("exp.bad_spec", why="cost: numbers")
    if not isinstance(spec.get("priority", 0), int) or isinstance(spec.get("priority"), bool):
        return t("exp.bad_spec", why="priority: an integer")
    return ""


# --- the unified view -----------------------------------------------------------------------------

def model_view(repo, c):
    return {"id": repo, "type": "model", "status": unified_status(c), "title": c.get("name", repo),
            "priority": c.get("priority", 0), "created": c.get("brief", ""), "updated": c.get("updated", ""),
            "summary": c.get("summary", ""), "store_status": c.get("status"), "backend": c.get("backend", "llamacpp")}


def everything():
    """[(id, record)] of both stores, models first, in store order."""
    with candidates.store(readonly=True) as cands, store(readonly=True) as data:
        return ([(repo, model_view(repo, c)) for repo, c in cands["candidates"].items()]
                + [(i, dict(e)) for i, e in data["experiments"].items()])


# --- changes --------------------------------------------------------------------------------------

def add_spec(spec):
    """(True, id) or (False, refusal line)."""
    why = validate(spec)
    if why:
        return False, why
    eid = f"{config.now():%Y%m%d}-{spec['slug']}"
    variant = {k: spec["variant"][k] for k in VARIANT_KEYS if k in spec["variant"]}
    record = {"id": eid, "type": spec["type"], "title": spec["title"].strip(), "hypothesis": spec.get("hypothesis", ""),
              "variant": variant, "metrics": spec.get("metrics") or ["passed"], "pass_rule": spec["pass_rule"],
              "checks": spec.get("checks", []), "cost": spec.get("cost", {}), "status": "planned",
              "priority": spec.get("priority", 0), "source": str(spec.get("source", ""))[:200],
              "created": candidates.now(), "updated": candidates.now(), "result": "", "verdict": {}, "summary": ""}
    with store() as data:
        if eid in data["experiments"]:
            return False, t("exp.exists", id=eid)
        data["experiments"][eid] = record
    return True, eid


def transition(eid, to, allowed, **fields):
    """Move one experiment to status `to` from one of `allowed`: (True, record) or (False, refusal)."""
    with store() as data:
        e = data["experiments"].get(eid)
        if e is None:
            return False, t("exp.unknown", id=eid[:100])
        if e["status"] not in allowed:
            return False, t("exp.bad_move", id=eid, status=e["status"], to=to)
        e.update(fields, status=to, updated=candidates.now())
        record = dict(e)
    if to in FINAL:
        finished(record)
    return True, record


def queue():
    """Every approved experiment -> queued; their ids."""
    with store() as data:
        ids = [i for i, e in data["experiments"].items() if e["status"] == "approved"]
        for i in ids:
            data["experiments"][i].update(status="queued", updated=candidates.now())
    return ids


def verdict(eid, label):
    """Compute and store the verdict of a config/engine experiment from results/<label>.json and the
    baseline: (True, one-line summary) or (False, refusal)."""
    if not LABEL.fullmatch(label or "") or ".." in label:
        return False, t("exp.bad_label", label=str(label)[:100])
    with store(readonly=True) as data:
        e = data["experiments"].get(eid)
    if e is None:
        return False, t("exp.unknown", id=eid[:100])
    if e["status"] in ("planned", "dropped"):
        return False, t("exp.bad_move", id=eid, status=e["status"], to="verdict")
    results, baseline = config.path("results"), config.get("production", "baseline", "")
    try:
        variant = json.loads((results / f"{label}.json").read_text())
        base = json.loads((results / f"{baseline}.json").read_text()) if LABEL.fullmatch(baseline or "") else None
    except (OSError, ValueError) as err:
        variant, base, why = None, None, t("exp.no_result", err=str(err)[:200])
    else:
        why = "" if base is not None else t("exp.no_result", err=f"baseline {baseline!r}")
    if why:
        status, v = "failed", {"metrics": {}, "rules": {}, "passed": [], "failed": []}
    else:
        status, v, why = evaluate(e, variant, base)
    line = t("exp.verdict", id=eid, status=status, passed=len(v["passed"]), total=len(v["passed"]) + len(v["failed"]),
             why=f" — {why}" if why else "")
    summary = (why or "; ".join(v["rules"].values()))[:1000]
    with store() as data:
        data["experiments"][eid].update(status=status, verdict=v, result=label, summary=summary,
                                        updated=candidates.now())
        record = dict(data["experiments"][eid])
    finished(record)
    return True, line


# --- reporting ------------------------------------------------------------------------------------

def verdict_hook(key):
    """[experiments] verdict_command <key>: argument list, no shell, 60 s; failures are logged, never raised."""
    cmd = config.get("experiments", "verdict_command", []) or []
    argv = shlex.split(cmd) if isinstance(cmd, str) else [str(a) for a in cmd]
    if not argv:
        return
    argv[0] = str(Path(argv[0]).expanduser())
    try:
        run = subprocess.run(argv + [key], capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
        if run.returncode:
            print(f"experiments: verdict_command exit {run.returncode}: {(run.stderr or run.stdout).strip()[-300:]}",
                  file=sys.stderr)
    except (OSError, subprocess.SubprocessError) as err:
        print(f"experiments: verdict_command failed: {err}", file=sys.stderr)


def finished(record):
    """A config/engine experiment reached a final status: wiki note and verdict_command."""
    try:
        import wiki_note
        wiki_note.send_experiment(record)
    except OSError as err:
        print(f"experiments: wiki note failed: {err}", file=sys.stderr)
    verdict_hook(record["id"])


def model_finished(repo, status):
    """candidates.py set calls this: a model experiment reports back too (its wiki note comes from report.py)."""
    if status in MODEL_FINAL:
        verdict_hook(repo)


def line(eid, e):
    return f"{eid:50} {e['type']:6} {e['status']:14} {e.get('title', '')}"


def main():
    args = sys.argv[1:]
    as_json = "--json" in args
    rest = [a for a in args if a != "--json"]
    if rest == ["list"]:
        items = everything()
        if as_json:
            print(json.dumps(dict(items), ensure_ascii=False, indent=1))
        for eid, e in [] if as_json else items:
            print(line(eid, e))
    elif len(rest) == 2 and rest[0] == "show":
        found = dict(everything()).get(rest[1])
        if found is None:
            sys.exit(t("exp.unknown", id=rest[1][:100]))
        if found["type"] == "model":
            with candidates.store(readonly=True) as cands:
                found = found | {"candidate": cands["candidates"][rest[1]]}
        print(json.dumps(found, ensure_ascii=False, indent=1) if as_json else
              "\n".join(f"{k}: {json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v}"
                        for k, v in found.items()))
    elif args[:2] == ["add", "--spec-file"] and len(args) == 3:
        try:
            spec = json.loads(Path(args[2]).read_text())
        except (OSError, ValueError) as err:
            sys.exit(t("exp.bad_spec", why=str(err)[:200]))
        ok, out = add_spec(spec)
        print(out)
        sys.exit(0 if ok else 1)
    elif args[:3] == ["add", "--type", "model"]:
        import argparse
        parser = argparse.ArgumentParser(prog="experiments.py add --type model")
        parser.add_argument("repo")
        parser.add_argument("--backend", default="llamacpp")
        parser.add_argument("--now", action="store_true")
        opts = parser.parse_args(args[3:])
        ok, out = candidates.add(opts.repo, opts.backend, now_=opts.now)
        print(out)
        sys.exit(0 if ok else 1)
    elif len(args) == 2 and args[0] == "approve":
        ok, out = transition(args[1], "approved", ("planned",))
        print(t("exp.moved", id=args[1], status="approved") if ok else out)
        sys.exit(0 if ok else 1)
    elif len(args) in (2, 3) and args[0] == "drop":
        ok, out = transition(args[1], "dropped", ("planned", "approved", "queued"),
                             **({"summary": args[2].strip()[:1000]} if len(args) == 3 else {}))
        print(t("exp.moved", id=args[1], status="dropped") if ok else out)
        sys.exit(0 if ok else 1)
    elif args == ["queue"]:
        print(t("exp.queued", n=len(ids := queue()), ids=", ".join(ids) or "-"))
    elif len(args) in (3, 4) and args[0] == "set":
        if args[2] not in STATUSES or args[2] in VERDICT_ONLY:
            sys.exit(t("exp.set_status", status=args[2][:30]))
        ok, out = transition(args[1], args[2], STATUSES,
                             **({"summary": args[3].strip()[:1000]} if len(args) == 4 else {}))
        if not ok:
            sys.exit(out)
    elif len(args) == 3 and args[0] == "verdict":
        ok, out = verdict(args[1], args[2])
        print(out)
        sys.exit(0 if ok else 1)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
