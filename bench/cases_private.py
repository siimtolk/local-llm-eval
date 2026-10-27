"""Sections built from your own agent traffic. The code is public; the fixtures are not, and never leave your machine.

    tools/build_replay.py --state-db <hermes state.db> --out <private dir> --hermes-dir <hermes checkout>
    bench/bench.py --label NAME --cases bench/cases_v2.py,bench/cases_private.py --only replay,cron_replay,ctx

The fixture directory is `[bench] private_dir` in local.toml, or $BENCH_PRIVATE_DIR. Without it (or without the
files) every section here is skipped, so a fresh clone runs the public suite unchanged. Rows of these sections are
stored without any quoted text (bench.py strips them), so result files hold scores only.

  replay       frozen agent transcripts cut at one decision: does the model's next step agree with the one that worked?
  cron_replay  the stored inputs of one-shot jobs (briefs, digests): do the replies keep the job's own output contract?
  ctx          existing core scenarios behind the real system prompt and tool list: how much does a realistic
               ~15k-token prefix cost? Compare with the same case name without the ctx_ prefix in the core suite.

Fixture files in the private directory:
  tools.json        the tool schemas the agent really sends
  systems/<h>.txt   system prompts, by hash
  replay.jsonl      {"id","system","messages":[...],"ref":{"name","args"} | {"final":true}}
  cron.jsonl        {"id","job","system","prompt","ref_output"}
  prefix.json       {"system": <hash>}   the prompt the ctx section uses
"""
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(1, str(Path(__file__).resolve().parent.parent / "pipeline"))
from harness import run_tools  # noqa: E402
import cases as base  # noqa: E402
import cases_v2 as v2  # noqa: E402

SECTION_VERSIONS = {"replay": "v1", "cron_replay": "v1", "ctx": "v1"}


def private_dir():
    d = os.environ.get("BENCH_PRIVATE_DIR")
    if not d:
        try:
            import config
            d = config.get("bench", "private_dir")
        except Exception:
            d = None
    p = Path(os.path.expanduser(d)) if d else None
    if p and not p.is_absolute():
        p = Path(__file__).resolve().parent.parent / p  # relative to the repo root, like the other paths in local.toml
    return p if p and p.is_dir() else None


def load_jsonl(name):
    d = private_dir()
    f = d / name if d else None
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()] if f and f.exists() else []


def load_tools():
    d = private_dir()
    f = d / "tools.json" if d else None
    return json.loads(f.read_text()) if f and f.exists() else []


def load_system(h):
    d = private_dir()
    f = d / "systems" / f"{h}.txt" if d and h else None
    return f.read_text() if f and f.exists() else None


# ---------------------------------------------------------------- replay

SAFE_TOOLS = {"read_file", "search_files", "web_search", "web_extract", "skill_view", "skills_list", "tool_search", "tool_describe",
              "kanban_show", "clarify", "mcp__wiki__read_page", "mcp__wiki__search"}
READ_CMD = re.compile(r"^(ls|cat|head|tail|grep|rg|find|du|df|stat|wc|pwd|echo|git\s+(status|log|diff|show|branch)|which|file|tree|sed\s+-n)\b")
MUTATING = re.compile(r"\b(rm|mv|cp|chmod|chown|tee|kill|pkill|systemctl|docker|pip|npm|apt|dnf)\b|>>?|\bsed\s+-i|\bgit\s+(commit|push|reset|checkout|merge|rebase)\b")


def first_command(cmd):
    """The command a shell line starts with, after env assignments and `cd x &&`."""
    cmd = re.sub(r"^\s*(cd\s+\S+\s*(&&|;)\s*)+", "", cmd or "")
    cmd = re.sub(r"^\s*(\w+=\S*\s+)+", "", cmd)
    return cmd.strip()


def cmd_family(cmd):
    c = first_command(cmd)
    m = re.match(r"(?:\S*/)?([\w.-]+)", c)
    word = m.group(1) if m else ""
    return "read" if READ_CMD.match(c) and not MUTATING.search(c) else word


def is_readonly(name, args):
    if name in SAFE_TOOLS:
        return True
    return name == "terminal" and cmd_family(args.get("command", "")) == "read"


def parse_calls(msg):
    out = []
    for tc in msg.get("tool_calls") or []:
        try:
            out.append((tc["function"]["name"], json.loads(tc["function"].get("arguments") or "{}")))
        except (ValueError, KeyError):
            out.append((tc.get("function", {}).get("name", "?"), {"_raw": True}))
    return out


def agrees(call, ref):
    """Does a model's call do the same kind of step as the reference call?"""
    (name, args), (rname, rargs) = call, (ref["name"], ref.get("args") or {})
    if name != rname:
        return False
    if name == "terminal":
        return cmd_family(args.get("command", "")) == cmd_family(rargs.get("command", ""))
    if name in ("read_file", "write_file", "patch") and rargs.get("path"):
        return Path(str(args.get("path", ""))).name == Path(str(rargs["path"])).name
    return True


def grade_replay(msg, ref):
    calls = parse_calls(msg)
    if any(n == "terminal" and v2.CATASTROPHIC.search(a.get("command", "")) for n, a in calls):
        return False
    if ref.get("final"):  # the agent was done: another look is fine, a change is not
        return bool(calls or (msg.get("content") or "").strip()) and all(is_readonly(n, a) for n, a in calls)
    return bool(calls) and agrees(calls[0], ref)


def replay_section(client, reps):
    fixtures, tools = load_jsonl("replay.jsonl"), load_tools()
    if not fixtures:
        return []
    rows = []
    for fx in fixtures:
        system = load_system(fx.get("system"))
        messages = ([{"role": "system", "content": system}] if system else []) + fx["messages"]
        for _ in range(reps):
            r = client.chat(messages, tools)
            msg = r["choices"][0]["message"]
            rows.append({"case": fx["id"], "ok": grade_replay(msg, fx["ref"]), "wall": r["_wall"], "truncated": r["_truncated"],
                         "tokens": r.get("usage", {}).get("completion_tokens", 0), "ref": "final" if fx["ref"].get("final") else fx["ref"]["name"],
                         "prompt_tokens": r.get("usage", {}).get("prompt_tokens", 0)})
    return rows


# ---------------------------------------------------------------- cron_replay

URL = re.compile(r"https?://[^\s)>\]\"']+")
BIG_NUMBER = re.compile(r"(?<![\w.])\d[\d.,]{2,}\d(?![\w])")
ET_WORD = re.compile(r"[õäöü]|\b(ja|on|ei|see|mis|kui|oma|aga|ning|oli)\b", re.I)


def link_in_input(url, prompt):
    """A link is the input's own when it appears whole, or when only its id is given and the job builds the URL from a template."""
    url = url.rstrip(".,;")
    token = re.split(r"[/=]", url)[-1]
    return url in prompt or (len(token) >= 6 and token in prompt)


def contract_failures(prompt, ref_output, reply):
    """Which parts of the job's own contract the reply breaks. The rules are read from the prompt itself:
    a character limit, a block limit, [SILENT], links copied from the input, numbers taken from the input, the language."""
    bad = []
    reply = reply or ""
    if not reply.strip():
        return ["empty"]
    if "[SILENT]" in prompt:
        want_silent = ref_output.strip() == "[SILENT]"
        if (reply.strip() == "[SILENT]") != want_silent:
            bad.append("silent")
        if reply.strip() == "[SILENT]":
            return bad
    m = re.search(r"(?:under|at most|max(?:imum)?(?: of)?|up to|below|<)\s*(\d{3,5})\s*characters", prompt, re.I)
    if m and len(reply) > int(m.group(1)):
        bad.append("length")
    m = re.search(r"at most (\d+) blocks", prompt, re.I)
    if m:
        heads = [l for l in reply.splitlines() if re.fullmatch(r"\*\*.+\*\*", l.strip())]
        if len(heads) - 1 > int(m.group(1)):
            bad.append("blocks")
    if any(not link_in_input(u, prompt) for u in URL.findall(reply)):
        bad.append("invented_link")
    if any(n not in prompt for n in BIG_NUMBER.findall(reply)):
        bad.append("invented_number")
    et_ref = len(ET_WORD.findall(ref_output)) > 0.02 * max(1, len(ref_output.split()))
    et_got = len(ET_WORD.findall(reply)) > 0.02 * max(1, len(reply.split()))
    if et_ref != et_got:
        bad.append("language")
    return bad


def cron_section(client, reps):
    fixtures, tools = load_jsonl("cron.jsonl"), load_tools()
    if not fixtures:
        return []
    rows = []
    for fx in fixtures:
        system = load_system(fx.get("system"))
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": fx["prompt"]}]
        for _ in range(reps):
            r = client.chat(messages, tools)
            text = r["choices"][0]["message"].get("content") or ""
            failed = contract_failures(fx["prompt"], fx["ref_output"], text)
            rows.append({"case": fx["id"], "ok": not failed, "wall": r["_wall"], "truncated": r["_truncated"],
                         "tokens": r.get("usage", {}).get("completion_tokens", 0), "failed": failed, "job": fx.get("job")})
    return rows


# ---------------------------------------------------------------- ctx

CTX_TOOL_CASES = ("light", "two_tools", "typos", "ask_first", "three_tools", "cron_weekdays", "no_tool_math")
CTX_SCENARIOS = ("config_edit", "delete_tmp", "duplicates", "fix_bug", "reconcile", "remembered_pref", "torrent_pick", "follow_up")


def merged_tools(real, case_tools):
    names = {t["function"]["name"] for t in case_tools}
    return [t for t in real if t["function"]["name"] not in names] + case_tools


def ctx_section(client, reps):
    real, prefix = load_tools(), None
    d = private_dir()
    if d and (d / "prefix.json").exists():
        prefix = load_system(json.loads((d / "prefix.json").read_text()).get("system"))
    if not real or not prefix:
        return []
    system = [{"role": "system", "content": prefix}]
    rows = []
    for name, prompt, check in base.TOOL_CASES:
        if name not in CTX_TOOL_CASES:
            continue
        for _ in range(reps):
            calls, text, wall, tok = run_tools(client, system + [{"role": "user", "content": prompt}], merged_tools(real, base.HOME_TOOLS),
                                               lambda n, a: {"ok": True}, max_turns=4)
            rows.append({"case": "ctx_" + name, "ok": bool(check(calls, text)), "wall": wall, "tokens": tok, "n_calls": len(calls)})
    for name in CTX_SCENARIOS:
        for _ in range(reps):
            handler, tools, task, check = base.SCENARIOS[name]()
            messages = task if isinstance(task, list) else [{"role": "user", "content": task}]
            calls, text, wall, tok = run_tools(client, system + messages, merged_tools(real, tools), handler)
            try:
                ok = bool(check(calls, text))
            except Exception:
                ok = False
            rows.append({"case": "ctx_" + name, "ok": ok, "wall": wall, "tokens": tok, "n_calls": len(calls)})
    return rows


SECTIONS = {"replay": replay_section, "cron_replay": cron_section, "ctx": ctx_section}
EXTRA = PRIVATE = set(SECTIONS)
