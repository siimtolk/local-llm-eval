#!/usr/bin/env python3
"""The data for the live results board on the blog, from results/*.json.

    tools/live_board.py                 print a summary of what the board would show
    tools/live_board.py --write         write <board.blog_repo>/_data/results.json
    tools/live_board.py --publish       --write, then commit and push the blog repo if the data changed

One point per model setup: full runs of the newest suite only (a run with fewer graded cases than
the biggest one, a speed-only run, `_nothink` passes and `exp-` experiments are not points; the exception
is a full run of an `engine` experiment, another backend serving a model, which is a point named by the
experiment's title), and of
several runs of one setup (reruns `_r2`, other llama.cpp builds `_b11707`, `-2` re-tests) the latest
stands for it; the earlier ones stay in the data with `latest: false`. Quality is graded runs
passed with Estonian and the uncensored set left out, as in the articles. The extra sections (estonian_free,
vision, robust, longagent under the result's "extra" key; replay, cron_replay, ctx under "own_traffic") are shown as
their own columns and never count in that percentage; a setup that ran them also gets a second total, graded_all / pct_all.

Display names and kinds come from tools/board_names.json (key -> {name, kind}); a setup without an
entry gets a name made from its label. An extras-only experiment (see extras_targets) is merged into the latest earlier run of the same model and effort. The blog repo is `[board] blog_repo` in local.toml.
"""
import argparse
import json
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from results_table import sections  # noqa: E402

NAMES = Path(__file__).resolve().parent / "board_names.json"
EXTRAS_FOR = Path(__file__).resolve().parent / "board_extras.json"  # manual override of extras_targets(): {extras-only result label: core result label}
NOT_GRADED = {"uncensored"}
EXTRA_SECTIONS = ("estonian_free", "vision", "robust", "longagent")  # result key "extra": the public, synthetic sections
TRAFFIC_SECTIONS = ("replay", "cron_replay", "ctx")                  # result key "own_traffic": built from the owner's own agent history


def setup_key(label):
    """The model setup a label belongs to: reruns, suite tags and llama.cpp builds do not make a new one."""
    k = re.sub(r"_suite\d{4}-\d\d-\d\d", "", label)
    k = re.sub(r"_b\d+(?=_|$)", "", k)
    k = re.sub(r"_r\d+(?=_|$)", "", k)
    k = re.sub(r"_(depth|clientspeed)(?=_|$)", "", k)
    head, _, rest = k.partition("_")
    return re.sub(r"-\d+$", "", head) + (("_" + rest) if rest else "")


def guess_kind(label):
    low = label.lower()
    if "capy" in low:
        return "capy"
    if "vllm" in low:
        return "vllm"
    return "other"


def pretty(label):
    head = setup_key(label).split("_")[0]
    return head.replace("-", " ")


def med(rows, field):
    vals = [r[field] for r in rows if isinstance(r, dict) and r.get(field)]
    return round(statistics.median(vals), 1) if vals else None


def speed(res):
    sp = res.get("speed") or {}
    turns = (sp.get("alternating") or {}).get("turns") or []
    follow = [t["ttft"] for t in turns if t.get("turn", 0) >= 1 and t.get("ttft") is not None]
    deep = sp.get("deep_decode") or {}
    agg = (sp.get("parallel") or {}).get("aggregate_tok_s")
    return {
        "prose": med(sp.get("prose", []), "tok_s"),
        "tool_json": med(sp.get("tool_json", []), "tok_s"),
        "prompt_30k": med(sp.get("long_prompt", []), "prompt_tok_s"),
        "parallel": round(agg, 1) if agg else None,
        "followup_ttft": round(statistics.mean(follow), 2) if follow else None,
        "decode_30k": round(deep["30k"]["tok_s"], 1) if deep.get("30k") else None,
        "decode_90k": round(deep["90k"]["tok_s"], 1) if deep.get("90k") else None,
    }


SPEC = {"mtp": "MTP drafting", "dflash": "DFlash drafting", "nospec": "no drafting"}


def derive_details(key):
    """A short serving summary from the setup key (<model>_<quant>_<spec>_<gpus>_<context>; the quant is part of the name), used when board_names.json has no `detail`."""
    out = []
    for p in key.split("_")[1:]:
        if p in SPEC:
            out.append(SPEC[p])
        elif re.fullmatch(r"\d+gpu", p):
            out.append(p[:-3] + " GPUs")
        elif re.fullmatch(r"\d+k", p):
            out.append(p + " context")
        elif re.fullmatch(r"vllm[\d.]+", p):
            out.append("vLLM " + p[4:])
        elif p == "tp2":
            out.append("2 GPUs")
    return " · ".join(out)


def display_names(meta, fallback):
    """(full name, short name) of a setup: `backend · model (variant) · quant · uncensoring method · author` from the
    board_names.json fields, and `model (backend if not llama.cpp, variant)` for the chart and tiles; a setup with only a
    `name` (or none) uses that for both."""
    if "model" not in meta:
        return meta.get("name") or fallback, meta.get("short") or meta.get("name") or fallback
    var = meta.get("variant")
    model = meta["model"] + (f" ({var})" if var else "")
    full = " · ".join(x for x in (meta.get("backend"), model, meta.get("quant"), meta.get("method"), meta.get("author")) if x)
    extra = [x for x in (meta["backend"] if meta.get("backend") not in (None, "llama.cpp") else None, var) if x]
    return full, meta["model"] + (f" ({', '.join(extra)})" if extra else "")


def totals(res, skip):
    sec = {k: v for k, v in sections(res).items() if k not in skip}
    return (sum(p for p, _ in sec.values()), sum(n for _, n in sec.values())) if sec else (0, 0)


def passed(rows):
    return [sum(1 for r in rows if r.get("ok")), len(rows)]


def extra_scores(res):
    """The sections kept outside the core totals: {"extra": {section: [passed, runs]}, "own_traffic": [passed, runs] | None,
    "own_traffic_parts": {section: [passed, runs]}, "section_versions": {...}}. The board shows them next to the core
    areas and never add them to graded / pct; a setup that ran them also has graded_all / pct_all (base areas plus these)."""
    def rows_of(key, names):
        d = res.get(key) or {}
        return {n: passed(d[n]) for n in names if isinstance(d.get(n), list) and d[n] and all(isinstance(r, dict) and "ok" in r for r in d[n])}
    extra, parts = rows_of("extra", EXTRA_SECTIONS), rows_of("own_traffic", TRAFFIC_SECTIONS)
    versions = res.get("section_versions") or {}
    return {"extra": extra,
            "own_traffic": [sum(p[0] for p in parts.values()), sum(p[1] for p in parts.values())] if parts else None,
            "own_traffic_parts": parts,
            "section_versions": {n: versions[n] for n in list(extra) + list(parts) if n in versions}}


def weights_urls(confs_dir):
    """{serving config name: Hugging Face page} from HF_REPO in models.d/<name>.conf.sh."""
    out = {}
    for conf in Path(confs_dir).glob("*.conf.sh") if confs_dir else []:
        m = re.search(r'^HF_REPO="?([\w.-]+/[\w.-]+)"?', conf.read_text(), re.M)
        if m:
            out[conf.name.removesuffix(".conf.sh")] = "https://huggingface.co/" + m.group(1)
    return out


def depth_runs(results_dir):
    """{setup key: deep_decode of its latest run that has one}: a setup's long-context decode is sometimes measured on its own
    (<label>_depth.json, like production's), not in the full run."""
    best = {}
    for path in sorted(results_dir.glob("*.json")):
        if path.name.startswith("_") or path.stem.endswith("_nothink"):
            continue
        try:
            res = json.loads(path.read_text())
        except ValueError:
            continue
        deep = (res.get("speed") or {}).get("deep_decode") if isinstance(res, dict) else None
        if isinstance(deep, dict) and any(isinstance(v, dict) and v.get("tok_s") for v in deep.values()):
            key, started = setup_key(path.stem), res.get("started") or ""
            if key not in best or started > best[key][0]:
                best[key] = (started, deep)
    return {k: v[1] for k, v in best.items()}


def merge_extras(res, extras_path):
    """res with the extra / own_traffic sections of an extras-only run added; sections res already has stay."""
    if not extras_path.exists():
        return res
    ex = json.loads(extras_path.read_text())
    res = dict(res)
    for key in ("extra", "own_traffic", "section_versions"):
        res[key] = {**(ex.get(key) or {}), **(res.get(key) or {})}
    return res


def collect(results_dir, baseline, confs_dir=None, engine_titles=None, extras_for=None):
    """engine_titles: {result label: title} of the engine experiments, the only `exp-` results shown."""
    engine_titles = engine_titles or {}
    names = json.loads(NAMES.read_text()) if NAMES.exists() else {}
    extras_for = {**(extras_for or {}), **(json.loads(EXTRAS_FOR.read_text()) if EXTRAS_FOR.exists() else {})}
    urls = weights_urls(confs_dir)
    depth = depth_runs(results_dir)
    runs = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name.startswith("_") or path.stem.endswith("_nothink") \
                or (path.name.startswith("exp-") and path.stem not in engine_titles):
            continue
        res = json.loads(path.read_text())
        if not isinstance(res, dict) or not res.get("suite"):
            continue
        for src in [k for k, v in extras_for.items() if v == path.stem]:  # extras-only runs filed under this setup
            res = merge_extras(res, results_dir / f"{src}.json")
        ok, n = totals(res, NOT_GRADED)  # graded incl. Estonian, for the "full run" test
        runs.append((path, res, n))
    if not runs:
        return None
    suite = max(r[1]["suite"] for r in runs)
    in_suite = [r for r in runs if r[1]["suite"] == suite]
    full_n = max(n for _, _, n in in_suite)
    out, skipped = [], len(runs) - len(in_suite)
    for path, res, n in in_suite:
        sp = speed(res)
        if sp["decode_30k"] is None and sp["decode_90k"] is None and setup_key(path.stem) in depth:
            sp.update({k: v for k, v in speed({"speed": {"deep_decode": depth[setup_key(path.stem)]}}).items() if k.startswith("decode_")})
        if n < full_n or sp["prose"] is None:
            continue
        sec = sections(res)
        ok, tot = totals(res, NOT_GRADED | {"estonian"})
        more = extra_scores(res)  # the newer sections: a second total, graded_all, for a setup that ran them
        add = [more["extra"][n] for n in more["extra"]] + ([more["own_traffic"]] if more["own_traffic"] else [])
        all_ok, all_tot = ok + sum(p for p, _ in add), tot + sum(n for _, n in add)
        nothink = None
        nt = path.with_name(f"{path.stem}_nothink.json")
        if nt.exists():
            a, b = totals(json.loads(nt.read_text()), NOT_GRADED)
            nothink = [a, b] if b else None
        key = setup_key(path.stem)
        meta = names.get(key, {})
        name, short = display_names(meta, engine_titles.get(path.stem) or pretty(path.stem))
        out.append({
            "key": key, "label": path.stem, "url": meta.get("url") or urls.get(path.stem.split("_")[0]) or urls.get(key.split("_")[0]), "name": name, "short": short,
            "title": meta["model"] + (f" ({meta['variant']})" if meta.get("variant") else "") if "model" in meta else short,
            "backend": meta.get("backend") or ("vLLM" if guess_kind(path.stem) == "vllm" else "capy-flash" if guess_kind(path.stem) == "capy" else "llama.cpp"),
            "traits": " · ".join(x for x in (meta.get("quant"), meta.get("method"), meta.get("author")) if x),
            "kind": meta.get("kind") or guess_kind(path.stem),
            "detail": meta.get("detail") or derive_details(key),
            "sections": {k: list(v) for k, v in sec.items() if k not in NOT_GRADED | {"estonian"}},
            "date": (res.get("started") or "")[:10], "started": res.get("started") or "",
            "production": path.stem == baseline,
            "graded": [ok, tot], "pct": round(100 * ok / tot, 1),
            "graded_all": [all_ok, all_tot] if add else None, "pct_all": round(100 * all_ok / all_tot, 1) if add else None,
            "estonian": list(sec["estonian"]) if "estonian" in sec else None,
            "uncensored": list(sec["uncensored"]) if "uncensored" in sec else None,
            "nothink": nothink,
            **more,
            "minutes": round(res["total_seconds"] / 60, 1) if res.get("total_seconds") else None,
            **sp,
        })
    best = {}
    for m in out:  # production's baseline stands for its setup, otherwise the latest start
        cur = best.get(m["key"])
        if cur is None or (m["production"], m["started"]) > (cur["production"], cur["started"]):
            best[m["key"]] = m
    for m in out:
        m["latest"] = best[m["key"]] is m
    out.sort(key=lambda m: m["started"], reverse=True)
    return {"suite": suite, "graded_cases": full_n, "older_suite_runs": skipped,
            "production": next((m["label"] for m in out if m["production"]), None), "models": out}


def check(data, old):
    """Sanity check before anything is written or pushed: the problems found, empty when the data looks right."""
    bad, pts = [], [m for m in data["models"] if m["latest"]]
    if not pts:
        return ["no setups to show"]
    if not data["production"]:
        bad.append("production's baseline run is not among the results")
    for m in pts:
        if not (0 <= m["pct"] <= 100) or not m["graded"][1] or m["graded"][0] > m["graded"][1]:
            bad.append(f"{m['label']}: graded {m['graded']} / {m['pct']}% is impossible")
        if not (0 < m["prose"] < 1000):
            bad.append(f"{m['label']}: prose speed {m['prose']} tok/s is outside 0-1000")
        for f in ("tool_json", "prompt_30k", "parallel"):
            if m[f] is not None and m[f] <= 0:
                bad.append(f"{m['label']}: {f} is {m[f]}")
        for n, p in {**m["extra"], **m["own_traffic_parts"]}.items():
            if not p[1] or p[0] > p[1]:
                bad.append(f"{m['label']}: {n} {p} is impossible")
    if len({m["key"] for m in pts}) != len(pts):
        bad.append("two points for one setup")
    if old and old.get("suite") == data["suite"] and len(pts) < 0.8 * len([m for m in old["models"] if m["latest"]]):
        bad.append(f"{len(pts)} setups, down from {len([m for m in old['models'] if m['latest']])} on the same suite")
    return bad


def engine_experiment_titles():
    """{result label: title} of the engine experiments (<state>/experiments.json), whose full runs the board shows."""
    import experiments
    try:
        data = json.loads(experiments.store_path().read_text())["experiments"]
    except (OSError, ValueError, KeyError):
        return {}
    return {f"exp-{eid}": e["title"] for eid, e in data.items() if e.get("type") == "engine" and e.get("title")}


def extras_targets(results_dir):
    """{extras-only result label: core result label} for the extras-only experiments (EXTRAS ONLY in their notes):
    the latest full run of the same served model (`Serve model name <slug>` in the notes), with the same reasoning
    effort (a request_extra reasoning_effort of low matches `efflow` files, none matches files without `eff`).
    board_extras.json overrides this. An experiment with no earlier run of its model stays off the board."""
    import experiments
    try:
        data = json.loads(experiments.store_path().read_text())["experiments"]
    except (OSError, ValueError, KeyError):
        return {}
    out = {}
    for eid, e in data.items():
        v = e.get("variant") or {}
        notes = v.get("notes") or ""
        slug = re.search(r"Serve model name (\S+?)[ ,;(]", notes)
        label = f"exp-{eid}"
        if "EXTRAS ONLY" not in notes or not slug or not (results_dir / f"{label}.json").exists():
            continue
        effort = (v.get("request_extra") or {}).get("reasoning_effort")
        cands = []
        for path in results_dir.glob(f"{slug.group(1)}_*.json"):
            if path.stem.endswith("_nothink"):
                continue
            tail = path.stem.rsplit("_", 1)[-1]
            if effort == "low" and tail != "efflow" or effort is None and tail.startswith("eff"):
                continue
            try:
                res = json.loads(path.read_text())
            except ValueError:
                continue
            if isinstance(res, dict) and res.get("suite") and sections(res):
                cands.append((res.get("started") or "", path.stem))
        if cands:
            out[label] = max(cands)[1]
    return out


def build():
    import config
    data = collect(config.path("results"), config.get("production", "baseline"),
                   config.path("model_configs", config.serving()), engine_experiment_titles(),
                   extras_targets(config.path("results")))
    if data is None:
        sys.exit("no results")
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()
    data = build()
    pts = [m for m in data["models"] if m["latest"]]
    print(f"suite {data['suite']}: {len(pts)} setups, {len(data['models'])} runs, "
          f"{data['older_suite_runs']} older-suite files not shown")
    for m in pts:
        print(f"  {m['name'][:38]:38s} {m['pct']:5.1f}%  {m['prose']:6.1f} tok/s  {m['kind']:6s} {m['date']}"
              f"{'  (production)' if m['production'] else ''}")
    if not (args.write or args.publish):
        return
    import config
    repo = Path(config.get("board", "blog_repo", "")).expanduser()
    if not (repo / "_config.yml").exists():
        sys.exit("[board] blog_repo is not set in local.toml (the blog repo checkout)")
    target = repo / "_data" / "results.json"
    target.parent.mkdir(exist_ok=True)
    old = json.loads(target.read_text()) if target.exists() else None
    problems = check(data, old)
    if problems:
        sys.exit("sanity check failed, nothing written:\n  " + "\n  ".join(problems))
    print("sanity check ok")
    same = bool(old) and {k: v for k, v in old.items() if k != "generated"} == data
    dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--", "_data/results.json"],
                           capture_output=True, text=True).stdout.strip()
    if same and not (args.publish and dirty):  # --publish also commits a file that was written but never committed
        print("unchanged")
        return
    if not same:
        data = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), **data}
        target.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
        print(f"wrote {target.name}")
    if args.publish:
        git = ["git", "-C", str(repo)]
        subprocess.run(git + ["pull", "--rebase", "--autostash", "-q"], check=True)
        subprocess.run(git + ["add", "_data/results.json"], check=True)
        latest = max(pts, key=lambda m: m["started"])
        subprocess.run(git + ["commit", "-q", "-m", f"Live results: {latest['name']}, {len(pts)} setups on suite {data['suite']}"], check=True)
        subprocess.run(git + ["push", "-q"], check=True)
        print("pushed")


if __name__ == "__main__":
    main()
