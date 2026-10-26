#!/usr/bin/env python3
"""The candidates store (paths.store): every model the loop has seen, and its test status.

The store is data only: every repo id and model name read from it is checked before use. Other
programs may write it too (e.g. a chat bot recording poll votes); they must take the same lock
file (<store>.lock, fcntl.flock).

Status: proposed (offered in the report's poll, not ticked yet) -> pending -> downloading -> downloaded
        -> testing -> tested | rejected | skipped | failed

Usage: candidates.py list
       candidates.py next                                    (oldest pending as JSON, or nothing)
       candidates.py set <repo> <status> ['<summary for the morning report>']
                                                             (tested/rejected/failed: also [experiments] verdict_command)
       candidates.py index                                   (store what the latest judge briefs recommended)
       candidates.py vote <poll id> '<JSON list of ticked options>'   (from the chat bridge)
       candidates.py votes <inbox file>                       (apply and clear queued votes)
       candidates.py add <repo> [--backend llamacpp|vllm] [--priority N] [--now]
                                                             (queue a repo for the model test)

`add` is the supported way for other programs (e.g. a chat command) to queue a Hugging Face repo:
it checks that the repo exists and has weights that fit this machine (fetch's rules), runs the
risk check and stores it as pending with priority N (default 1: before judge picks at 0 and vLLM
builds at -1). It prints one result line and exits 0, or why it refused and exits 1. --now:
fetch.py runs `runner.py bkg-start` after the download (the whole downloaded queue gets tested).

A chat bridge that cannot run this repo (e.g. it lives in another container) can queue its votes
instead: one JSON line {"poll": <poll id>, "selected": [<ticked options>]} per vote, appended to an
inbox file under the lock <inbox>.lock; `votes` applies them in order, e.g. from a systemd path
unit that watches the file.
"""
import fcntl
import json
import re
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import config
from texts import t

STATUSES = ("proposed", "pending", "downloading", "downloaded", "testing", "tested", "rejected", "skipped", "failed")
REPO_ID = re.compile(r"[A-Za-z0-9][\w.-]{0,95}/[A-Za-z0-9][\w.-]{0,95}")
MODEL_NAME = re.compile(r"[a-z0-9][a-z0-9.-]{1,60}")
HF_LINK = re.compile(r"https://huggingface\.co/([A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*)")
PROPOSED_DAYS = 7
BUSY = ("pending", "downloading", "downloaded", "testing")
BACKENDS = ("llamacpp", "vllm")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def valid_repo(repo):
    return bool(REPO_ID.fullmatch(repo or "")) and ".." not in repo


@contextmanager
def locked_json(path, empty, readonly=False):
    """The JSON file's data under its lock (<file>.lock); written back on exit unless readonly."""
    if readonly:
        yield json.loads(path.read_text()) if path.exists() else empty
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(path.read_text()) if path.exists() else empty
        yield data
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False))
        tmp.replace(path)


def store(readonly=False):
    """The store dict {"candidates": {repo: record}, "read": [...]} under its lock; saved on exit
    (readonly: a snapshot, nothing written)."""
    return locked_json(config.path("store"), {"candidates": {}, "read": []}, readonly)


def polls():
    return locked_json(config.path("polls"), {"polls": {}})


def update(repo, **fields):
    with store() as data:
        c = data["candidates"].get(repo)
        if c is None:
            raise KeyError(repo)
        c.update(fields, updated=now(), reported=False)


def recommended(text):
    """(name, size line, repo, link) for each block of a judge brief (a whole reply, or a run log
    whose reply follows a '## Response' heading)."""
    from short_links import expand
    response = text.split("\n## Response", 1)[1] if "\n## Response" in text else text
    for block in re.split(r"\n\s*\n", response):
        lines = [l.strip() for l in block.strip().splitlines() if l.strip()]
        title = re.fullmatch(r"\*\*(.+?)\*\*", lines[0]) if lines else None
        link = next((m for l in lines if (m := HF_LINK.match(expand(l.split(" ")[0])))), None)
        if title and link:
            repo = link.group(1).removesuffix(".git")
            yield title.group(1), (lines[1] if len(lines) > 1 else ""), repo, f"https://huggingface.co/{repo}"


def index():
    """Copy each repo the judge briefs (paths.briefs/*.md) recommended into the store, once, as
    "proposed", with a risk check (risk.py)."""
    from risk import safe_assess
    briefs = config.path("briefs")
    with store() as data:
        read = set(data["read"])
        new = []
        for path in sorted(briefs.glob("*.md")) if briefs.is_dir() else []:
            if path.name in read:
                continue
            for name, size, repo, link in recommended(path.read_text(errors="replace")):
                if repo not in data["candidates"] and valid_repo(repo):
                    data["candidates"][repo] = {
                        "name": name, "size": size, "link": link, "brief": path.name[:10],
                        "status": "proposed", "updated": now(), "summary": "", "reported": True}
                    new.append(repo)
            data["read"].append(path.name)
        data["read"] = data["read"][-200:]
    for repo in new:  # network calls outside the lock
        risk = safe_assess(repo)
        with store() as data:
            data["candidates"][repo]["risk"] = risk


def expire():
    """Proposed candidates nobody ticked within PROPOSED_DAYS are dropped (quietly)."""
    with store() as data:
        for c in data["candidates"].values():
            if c["status"] == "proposed" and datetime.now(timezone.utc) - datetime.fromisoformat(c["updated"]) > timedelta(days=PROPOSED_DAYS):
                c.update(status="skipped", updated=now(), reported=True,
                         summary=c["summary"] or t("store.expired", days=PROPOSED_DAYS))


def vote(poll_id, selected):
    """A tick or untick in a model poll: ticked proposed -> pending, unticked pending (not started) -> proposed."""
    with polls() as reg:
        poll = reg["polls"].get(poll_id)
        if not poll:
            return
        options = poll["options"]  # option text -> repo
        if selected and not set(selected) <= set(options) | {poll.get("none", "")}:
            return  # undecodable after a bridge restart: keep the state as it is
        poll["selected"], poll["voted_at"] = selected, now()
    ticked = {options[o] for o in selected if o in options}
    with store() as data:
        for repo in options.values():
            c = data["candidates"].get(repo)
            if not c:
                continue
            if repo in ticked and c["status"] == "proposed":
                c.update(status="pending", updated=now(), picked_by=poll_id)
            elif repo not in ticked and c["status"] == "pending" and c.get("picked_by") == poll_id:
                c.update(status="proposed", updated=now())


def votes(inbox):
    """Apply the votes queued in the inbox file (JSON lines), oldest first, then empty it."""
    inbox = Path(inbox)
    with open(f"{inbox}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        lines = inbox.read_text().splitlines() if inbox.exists() else []
        for line in lines:
            try:
                v = json.loads(line)
                poll, selected = v["poll"], v["selected"]
            except (ValueError, KeyError, TypeError):
                print(f"candidates: bad vote line {line[:200]!r}", file=sys.stderr)
                continue
            if isinstance(poll, str) and isinstance(selected, list) and all(isinstance(o, str) for o in selected):
                vote(poll, selected)
        if lines:
            inbox.write_text("")


def add(repo, backend="llamacpp", priority=1, now_=False):
    """Queue a repo for the model test: (True, result line) or (False, why it was refused)."""
    if not valid_repo(repo):
        return False, t("add.bad_repo", repo=repo[:200])
    if backend == "capy":
        return False, t("add.capy")
    if backend not in BACKENDS:
        return False, t("add.bad_backend", backend=backend[:50])
    with store(readonly=True) as data:
        old = dict(data["candidates"].get(repo) or {})
    if old.get("status") in BUSY:
        return False, t("add.busy", repo=repo, status=old["status"])
    import fetch  # here: fetch imports this module
    from risk import safe_assess
    try:  # network calls outside the lock, as in index()
        full = fetch.all_files(repo)
        chosen = fetch.pick(repo, {p: s for p, s in full.items() if p.lower().endswith(".gguf")},
                            old.get("quant_override", "")) if backend == "llamacpp" else None
        if chosen:
            quant, parts, _, fit = chosen
            gb = sum(full[p] for p in parts) / 1e9
            why = "" if fit else t("fetch.too_big", tag=quant, gb=gb)
        else:  # no GGUF (or --backend vllm): a 4-bit vLLM checkpoint, as fetch.main does
            backend = "vllm"
            quant, gb, _, why = fetch.vllm_pick(repo, full)
            why = why and t("fetch.vllm_unfit", why=why)
    except (OSError, ValueError) as err:
        return False, t("add.listing", repo=repo, err=str(err)[:200])
    if why:
        return False, t("add.unfit", repo=repo, why=why)
    risk = safe_assess(repo)
    if backend == "vllm" and risk.get("level") == "wontload":  # no GGUF header to read: not a reason for a safetensors build
        risk["level"] = "low"
    if risk.get("level") == "wontload":
        return False, t("add.wontload", repo=repo, why="; ".join(risk.get("reasons", []))[:300])
    with store() as data:
        c = data["candidates"].get(repo)
        if c and c["status"] in BUSY:  # queued by someone else meanwhile
            return False, t("add.busy", repo=repo, status=c["status"])
        if c is None:
            c = data["candidates"][repo] = {"name": repo.split("/")[1], "link": f"https://huggingface.co/{repo}"}
        for key in ("model", "vllm_checked", "backend", "test_now"):  # a re-test: history kept, quant_override too
            c.pop(key, None)
        c.update(brief=config.now().strftime("%Y-%m-%d"), status="pending", updated=now(), summary="",
                 reported=True, priority=priority, picked_by="add", size_gb=round(gb, 1), quant=quant, risk=risk,
                 **({"backend": "vllm"} if backend == "vllm" else {}), **({"test_now": True} if now_ else {}))
    level = risk.get("level")
    label = t("risk.high_short") if level == "high" else t(f"risk.{level}") if level in ("low", "medium") else t("risk.none")
    return True, t("add.queued", repo=repo, quant=quant, gb=gb, risk=label, when=t("add.now" if now_ else "add.later"))


def main():
    args = sys.argv[1:]
    if args[:1] == ["add"]:
        import argparse
        parser = argparse.ArgumentParser(prog="candidates.py add")
        parser.add_argument("repo")
        parser.add_argument("--backend", default="llamacpp")
        parser.add_argument("--priority", type=int, default=1)
        parser.add_argument("--now", action="store_true")
        opts = parser.parse_args(args[1:])
        ok, line = add(opts.repo, opts.backend, opts.priority, opts.now)
        print(line)
        sys.exit(0 if ok else 1)
    if args == ["list"]:
        with store(readonly=True) as data:
            for repo, c in data["candidates"].items():
                print(f"{c['status']:11} {c.get('brief', '')} {repo} {c.get('model', '')} {c.get('quant', '')}")
    elif args == ["next"]:
        with store(readonly=True) as data:
            pending = [(c.get("brief", ""), repo, c) for repo, c in data["candidates"].items() if c["status"] == "pending"]
            if pending:
                _, repo, c = min(pending)
                print(json.dumps({"repo": repo} | c, ensure_ascii=False))
    elif len(args) in (3, 4) and args[0] == "set" and args[2] in STATUSES and valid_repo(args[1]):
        try:
            update(args[1], status=args[2], **({"summary": args[3].strip()[:1000]} if len(args) == 4 else {}))
        except KeyError:
            sys.exit(f"candidates: no candidate {args[1]}")
        import experiments  # here: experiments imports this module
        experiments.model_finished(args[1], args[2])  # [experiments] verdict_command, on a final status
    elif args == ["index"]:
        index()
    elif len(args) == 3 and args[0] == "vote":
        vote(args[1], json.loads(args[2]))
    elif len(args) == 2 and args[0] == "votes":
        votes(args[1])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
