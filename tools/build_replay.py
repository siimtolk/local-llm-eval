#!/usr/bin/env python3
"""Build the fixtures of bench/cases_private.py from a Hermes state.db.

    tools/build_replay.py --state-db STATE.db --out PRIVATE_DIR [--hermes-dir CHECKOUT | --tools-json FILE]
                          [--replay 60] [--cron 12] [--redact FILE] [--seed 1] [--max-chars 120000]

Writes <out>/replay.jsonl, cron.jsonl, tools.json, prefix.json and systems/<hash>.txt. The output is made of real
transcripts: keep <out> outside tracked folders (it is `[bench] private_dir`, which .gitignore should cover) and never publish it.

replay   One fixture per decision point of a kanban / WhatsApp / API session: the messages before one assistant turn, and
         what the agent did next (its first tool call, or "final" when it answered). Points are sampled across tools, at most
         three per session, so a few long sessions do not dominate.
cron     Sessions of scheduled one-shot jobs (at most two API calls, a long input): the prompt and the reply that went out.

--redact FILE: one regex per line, replaced by [redacted] everywhere (names, addresses, ...). Secret-looking tokens are always removed.
"""
import argparse
import json
import math
import random
import re
import sqlite3
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

SECRET = re.compile(r"(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}"
                    r"|xox[bp]-[0-9A-Za-z-]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----)")
SESSION_SOURCES = ("kanban", "whatsapp", "api_server")


def scrubber(redact_file):
    pats = [SECRET]
    if redact_file:
        pats += [re.compile(l.strip()) for l in Path(redact_file).read_text().splitlines() if l.strip() and not l.startswith("#")]

    def scrub(x):
        if isinstance(x, str):
            for p in pats:
                x = p.sub("[redacted]", x)
            return x
        if isinstance(x, list):
            return [scrub(i) for i in x]
        if isinstance(x, dict):
            return {k: scrub(v) for k, v in x.items()}
        return x
    return scrub


def dump_tools(hermes_dir):
    code = "import json,model_tools;print('@@'+json.dumps(model_tools.get_tool_definitions(quiet_mode=True)))"
    out = subprocess.run([str(Path(hermes_dir) / ".venv/bin/python"), "-c", code], cwd=hermes_dir, capture_output=True, text=True, timeout=120).stdout
    line = next((l for l in out.splitlines() if l.startswith("@@")), None)
    if not line:
        sys.exit("could not dump the tool schemas from the Hermes checkout; pass --tools-json instead")
    return json.loads(line[2:])


def to_message(row):
    """A stored message as an API message, or None for rows that are not part of the conversation."""
    role = row["role"]
    if role == "user":
        return {"role": "user", "content": row["content"] or ""}
    if role == "tool":
        return {"role": "tool", "tool_call_id": row["tool_call_id"] or "", "content": row["content"] or ""}
    if role == "assistant":
        msg = {"role": "assistant", "content": row["content"] or ""}
        if row["tool_calls"]:
            calls = json.loads(row["tool_calls"])
            msg["tool_calls"] = [{"id": c["id"], "type": "function", "function": {"name": c["function"]["name"], "arguments": c["function"].get("arguments") or "{}"}} for c in calls]
        return msg
    return None


def session_messages(db, sid):
    rows = db.execute("select * from messages where session_id=? and active=1 order by coalesce(display_order, id), id", (sid,)).fetchall()
    return [(m, r) for r in rows if (m := to_message(r))]


def decision_points(db, args):
    out = []
    marks = ",".join("?" * len(SESSION_SOURCES))
    for s in db.execute(f"select id, source, system_prompt_hash from sessions where source in ({marks}) and api_call_count>0", SESSION_SOURCES):
        msgs = session_messages(db, s["id"])
        for i, (m, r) in enumerate(msgs):
            if m["role"] != "assistant" or i == 0 or not any(x["role"] == "user" for x, _ in msgs[:i]):
                continue
            if m.get("tool_calls"):
                call = m["tool_calls"][0]["function"]
                try:
                    ref = {"name": call["name"], "args": json.loads(call["arguments"])}
                except ValueError:
                    continue
            elif r["finish_reason"] == "stop" and m["content"].strip() and any(x["role"] == "tool" for x, _ in msgs[:i]):
                ref = {"final": True}
            else:
                continue
            prefix = [x for x, _ in msgs[:i]]
            if sum(len(json.dumps(x)) for x in prefix) > args.max_chars:
                continue
            out.append({"session": s["id"], "system": s["system_prompt_hash"], "messages": prefix, "ref": ref})
    return out


def sample(points, n, rng):
    """About n points: tool kinds capped so terminal and final answers do not crowd out the rest, three per session."""
    rng.shuffle(points)
    cap = defaultdict(lambda: max(2, math.ceil(n * 0.1)), {"terminal": math.ceil(n * 0.35), "final": math.ceil(n * 0.15)})
    kinds, per_session, chosen = Counter(), Counter(), []
    for p in points:
        kind = "final" if p["ref"].get("final") else p["ref"]["name"]
        if kinds[kind] >= cap[kind] or per_session[p["session"]] >= 3:
            continue
        kinds[kind] += 1
        per_session[p["session"]] += 1
        chosen.append(p)
        if len(chosen) >= n:
            break
    return chosen


def cron_fixtures(db, n):
    rows = db.execute("select id, title, system_prompt_hash from sessions where source='cron' and api_call_count<=2 order by started_at desc").fetchall()
    per_job, out = Counter(), []
    for s in rows:
        job = (s["title"] or "job").split(" · ")[0].strip()
        if per_job[job] >= 2:
            continue
        msgs = [m for m, _ in session_messages(db, s["id"])]
        user = next((m for m in msgs if m["role"] == "user"), None)
        final = next((m for m in reversed(msgs) if m["role"] == "assistant" and m["content"].strip() and not m.get("tool_calls")), None)
        if not user or not final or len(user["content"]) < 2000:
            continue
        per_job[job] += 1
        out.append({"id": f"cron_{job}_{per_job[job]}".replace(" ", "_"), "job": job, "system": s["system_prompt_hash"],
                    "prompt": user["content"], "ref_output": final["content"]})
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state-db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hermes-dir")
    ap.add_argument("--tools-json")
    ap.add_argument("--replay", type=int, default=60)
    ap.add_argument("--cron", type=int, default=12)
    ap.add_argument("--redact")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-chars", type=int, default=120000, help="longest allowed replay prefix, in characters")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "systems").mkdir(parents=True, exist_ok=True)
    scrub = scrubber(args.redact)
    db = sqlite3.connect(f"file:{args.state_db}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    rng = random.Random(args.seed)

    if args.tools_json:
        tools = json.loads(Path(args.tools_json).read_text())
    elif args.hermes_dir:
        tools = dump_tools(args.hermes_dir)
    else:
        sys.exit("give --hermes-dir or --tools-json for the tool schemas")
    replay = sample(decision_points(db, args), args.replay, rng)
    cron = cron_fixtures(db, args.cron)

    systems = {fx["system"] for fx in replay + cron if fx["system"]}
    for h in systems:
        row = db.execute("select prompt from system_prompts where hash=?", (h,)).fetchone()
        if row:
            (out / "systems" / f"{h}.txt").write_text(scrub(row["prompt"]))
    with open(out / "replay.jsonl", "w") as f:
        for i, fx in enumerate(replay):
            f.write(json.dumps(scrub({"id": f"rp{i:03d}_{'final' if fx['ref'].get('final') else fx['ref']['name']}", "system": fx["system"],
                                      "messages": fx["messages"], "ref": fx["ref"]}), ensure_ascii=False) + "\n")
    with open(out / "cron.jsonl", "w") as f:
        for fx in cron:
            f.write(json.dumps(scrub(fx), ensure_ascii=False) + "\n")
    (out / "tools.json").write_text(json.dumps(tools, ensure_ascii=False))
    common = Counter(fx["system"] for fx in replay if fx["system"]).most_common(1)
    if common:  # the prompt ctx uses: the one most replay points ran under
        (out / "prefix.json").write_text(json.dumps({"system": common[0][0]}))
    kinds = Counter("final" if fx["ref"].get("final") else fx["ref"]["name"] for fx in replay)
    print(f"replay: {len(replay)} points {dict(kinds)}\ncron: {len(cron)} fixtures {[fx['job'] for fx in cron]}\n"
          f"tools: {len(tools)}, system prompts: {len(systems)} -> {out}")


if __name__ == "__main__":
    main()
