"""Hand a finished test result to a shared LLM wiki: one note file per candidate in [wiki] inbox.

Optional: with [wiki] inbox empty (the default) nothing is written. The note uses the inbox format of
a wiki whose steward merges notes into pages (frontmatter: title, kind, agent, created, topic,
sources). Only real test outcomes (tested, rejected, failed) are sent; skipped proposals are noise.
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
