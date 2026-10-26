#!/usr/bin/env python3
"""The data for the live results board on the blog, from results/*.json.

    tools/live_board.py                 print a summary of what the board would show
    tools/live_board.py --write         write <board.blog_repo>/_data/results.json
    tools/live_board.py --publish       --write, then commit and push the blog repo if the data changed

One point per model setup: full runs of the newest suite only (a run with fewer graded cases than
the biggest one, a speed-only run, `_nothink` passes and `exp-` experiments are not points), and of
several runs of one setup (reruns `_r2`, other llama.cpp builds `_b11707`, `-2` re-tests) the latest
stands for it; the earlier ones stay in the data with `latest: false`. Quality is graded runs
passed with Estonian and the uncensored set left out, as in the articles.

Display names and kinds come from tools/board_names.json (key -> {name, kind}); a setup without an
entry gets a name made from its label. The blog repo is `[board] blog_repo` in local.toml.
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
NOT_GRADED = {"uncensored"}


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


def totals(res, skip):
    sec = {k: v for k, v in sections(res).items() if k not in skip}
    return (sum(p for p, _ in sec.values()), sum(n for _, n in sec.values())) if sec else (0, 0)


def collect(results_dir, baseline):
    names = json.loads(NAMES.read_text()) if NAMES.exists() else {}
    runs = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name.startswith(("_", "exp-")) or path.stem.endswith("_nothink"):
            continue
        res = json.loads(path.read_text())
        if not isinstance(res, dict) or not res.get("suite"):
            continue
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
        if n < full_n or sp["prose"] is None:
            continue
        sec = sections(res)
        ok, tot = totals(res, NOT_GRADED | {"estonian"})
        nothink = None
        nt = path.with_name(f"{path.stem}_nothink.json")
        if nt.exists():
            a, b = totals(json.loads(nt.read_text()), NOT_GRADED)
            nothink = [a, b] if b else None
        key = setup_key(path.stem)
        meta = names.get(key, {})
        out.append({
            "key": key, "label": path.stem, "name": meta.get("name") or pretty(path.stem),
            "kind": meta.get("kind") or guess_kind(path.stem),
            "date": (res.get("started") or "")[:10], "started": res.get("started") or "",
            "production": path.stem == baseline,
            "graded": [ok, tot], "pct": round(100 * ok / tot, 1),
            "estonian": list(sec["estonian"]) if "estonian" in sec else None,
            "uncensored": list(sec["uncensored"]) if "uncensored" in sec else None,
            "nothink": nothink,
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
    if len({m["key"] for m in pts}) != len(pts):
        bad.append("two points for one setup")
    if old and old.get("suite") == data["suite"] and len(pts) < 0.8 * len([m for m in old["models"] if m["latest"]]):
        bad.append(f"{len(pts)} setups, down from {len([m for m in old['models'] if m['latest']])} on the same suite")
    return bad


def build():
    import config
    data = collect(config.path("results"), config.get("production", "baseline"))
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
