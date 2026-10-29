"""Hand a finished test result to a shared LLM wiki: one note file per candidate or experiment in [wiki] inbox.

Optional: with [wiki] inbox empty (the default) nothing is written. The note uses the inbox format of
a wiki whose steward merges notes into pages (frontmatter: title, kind, agent, created, topic,
sources). Only real test outcomes (tested, failed; rejected is an old name for tested) are sent; skipped
proposals are noise. The notes carry results, never a verdict: the owner decides.
"""
import json
import re
from datetime import datetime
from pathlib import Path

import config

KINDS = ("tested", "rejected", "failed")


def send(repo, c):
    inbox = config.get("wiki", "inbox", "")
    if not inbox or c.get("status") not in KINDS:
        return
    folder = Path(inbox).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone()
    slug = re.sub(r"[^a-z0-9]+", "-", c["name"].lower()).strip("-")[:50]
    path = folder / f"{stamp:%Y-%m-%d-%H%M}-{slug}.md"
    sources = [str(config.path("store")), str(config.path("results")), c.get("link", repo)]
    # JSON strings are valid YAML scalars, so the frontmatter needs no YAML library.
    meta = {"title": f"Model test: {c['name']} {c['status']}", "kind": "result", "agent": "local-llm-eval",
            "created": f"{stamp:%Y-%m-%d %H:%M}", "topic": config.get("wiki", "topic", "") or None,
            "sources": sources}
    head = "\n".join(f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in meta.items())
    body = (f"Candidate `{repo}` — status **{c['status']}** ({c.get('updated', '')}).\n\n"
            f"Size/fit: {c.get('size', '?')}\n\nSummary from the nightly test:\n\n{c.get('summary') or '(none)'}\n")
    path.write_text(f"---\n{head}\n---\n\n{body}", encoding="utf-8")


def send_experiment(rec):
    """A config/engine experiment's final status (experiments.FINAL) as one inbox note."""
    inbox = config.get("wiki", "inbox", "")
    if not inbox or rec.get("status") not in ("done", "failed"):
        return
    folder = Path(inbox).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone()
    path = folder / f"{stamp:%Y-%m-%d-%H%M}-exp-{rec['id']}.md"
    from experiments import store_path  # here: experiments imports this module
    sources = [str(store_path())]
    if rec.get("result"):
        sources.append(str(config.path("results") / f"{rec['result']}.json"))
    meta = {"title": f"Experiment: {rec['title']} {rec['status']}", "kind": "result", "agent": "local-llm-eval",
            "created": f"{stamp:%Y-%m-%d %H:%M}", "topic": config.get("wiki", "topic", "") or None,
            "sources": sources}
    head = "\n".join(f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in meta.items())
    def cell(x):
        return "" if x is None else f"{x:.4g}"
    rows = "".join(f"| {name} | {cell(r.get('variant'))} | {cell(r.get('baseline'))} | {cell(r.get('delta'))} | {cell(r.get('ratio'))} |\n"
                   for name, r in (rec.get("comparison") or {}).items())
    body = (f"{rec['type'].capitalize()} experiment `{rec['id']}` — status **{rec['status']}** ({rec.get('updated', '')}), "
            f"measured against the production baseline. No verdict: the owner decides.\n\n"
            f"Hypothesis: {rec.get('hypothesis') or '(none)'}\n\nVariant: `{json.dumps(rec.get('variant', {}), ensure_ascii=False)}`\n\n"
            + (f"| metric | variant | baseline | delta | ratio |\n|---|---|---|---|---|\n{rows}\n" if rows else "")
            + f"Summary:\n\n{rec.get('summary') or '(none)'}\n")
    path.write_text(f"---\n{head}\n---\n\n{body}", encoding="utf-8")
