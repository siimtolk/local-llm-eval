#!/usr/bin/env python3
"""The README results table from results/*.json: the latest run per model.

    tools/results_table.py            print the table (Markdown)
    tools/results_table.py --write    replace the table between the markers in README.md
    tools/results_table.py --results DIR

A model is the label up to its first "_" (`<model>_<quant>_<spec>_...`); its latest run is the
newest suite's most complete run, started last (`_nothink` passes and files starting with "_"
are not runs of their own). Columns: graded
runs passed (every graded section except uncensored, Estonian included), Estonian, uncensored
answered, the thinking-off pass (`<label>_nothink.json`), and single-request prose speed (median
tok/s as bench.py timed it). Totals are comparable only within one suite version.
"""
import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
START, END = "<!-- results-table:start -->", "<!-- results-table:end -->"
NOT_GRADED = {"uncensored"}


def sections(res):
    """{section: (passed, runs)} for every graded section in a result file."""
    out = {}
    for name, rows in res.items():
        if isinstance(rows, list) and rows and isinstance(rows[0], dict) and "ok" in rows[0]:
            out[name] = (sum(bool(r["ok"]) for r in rows), len(rows))
    return out


def frac(pair):
    return f"{pair[0]}/{pair[1]}" if pair else "–"


def prose(res):
    vals = [r["tok_s"] for r in (res.get("speed") or {}).get("prose", []) if r.get("tok_s")]
    return f"{statistics.median(vals):.1f}" if vals else "–"


def rank(res):
    """The run that stands for its model: the newest suite, then the most graded runs (a rerun of one
    section or a speed-only run does not replace a full one), then the latest start."""
    graded = sum(n for k, (_, n) in sections(res).items() if k not in NOT_GRADED)
    return res.get("suite") or "", graded, res.get("started", "")


def rows(results):
    runs = {}
    for path in sorted(results.glob("*.json")):
        if path.name.startswith(("_", "exp-")) or path.stem.endswith("_nothink"):  # exp-: experiments, not models
            continue
        res = json.loads(path.read_text())
        label = res.get("label") or path.stem
        model = label.split("_", 1)[0]
        if model not in runs or rank(res) > rank(runs[model][1]):
            runs[model] = (path, res)
    out = []
    for model, (path, res) in sorted(runs.items(), key=lambda kv: kv[1][1].get("started", ""), reverse=True):
        sec = sections(res)
        graded = [v for k, v in sec.items() if k not in NOT_GRADED]
        total = (sum(p for p, _ in graded), sum(n for _, n in graded)) if graded else None
        nothink_path = path.with_name(f"{path.stem}_nothink.json")
        nothink = None
        if nothink_path.exists():
            ns = [v for k, v in sections(json.loads(nothink_path.read_text())).items() if k not in NOT_GRADED]
            nothink = (sum(p for p, _ in ns), sum(n for _, n in ns)) if ns else None
        out.append({"model": model, "label": path.stem, "date": (res.get("started") or "")[:10],
                    "suite": res.get("suite") or "–", "graded": frac(total), "estonian": frac(sec.get("estonian")),
                    "uncensored": frac(sec.get("uncensored")), "nothink": frac(nothink), "prose": prose(res)})
    return out


def table(results):
    lines = ["| Model | Run | Date | Suite | Graded | Estonian | Uncensored | Thinking off | Prose tok/s |",
             "|---|---|---|---|---:|---:|---:|---:|---:|"]
    for r in rows(results):
        lines.append(f"| {r['model']} | `{r['label']}` | {r['date']} | {r['suite']} | {r['graded']} | {r['estonian']} | "
                     f"{r['uncensored']} | {r['nothink']} | {r['prose']} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--write", action="store_true", help="update README.md between the markers")
    args = ap.parse_args()
    text = table(Path(args.results))
    if not args.write:
        print(text)
        return
    readme = ROOT / "README.md"
    old = readme.read_text()
    if START not in old or END not in old:
        raise SystemExit(f"results_table: no {START} ... {END} markers in README.md")
    head, rest = old.split(START, 1)
    readme.write_text(f"{head}{START}\n{text}\n{END}{rest.split(END, 1)[1]}")
    print(f"results_table: {text.count(chr(10)) - 1} models -> README.md")


if __name__ == "__main__":
    main()
