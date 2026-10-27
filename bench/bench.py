#!/usr/bin/env python3
"""Speed + quality benchmark for any OpenAI-compatible endpoint (written for llama-server).

    bench/bench.py --label NAME [--base-url URL] [--api-key-file F] [--results-dir D]
                   [--reps 3] [--temperature T] [--thinking on|off] [--reasoning-effort E] [--cases FILE[,FILE]]
                   [--only speed,tools,agent,reason,extract,code,hard,instruct,judge,estonian,uncensored]
                   (extra sections: --cases bench/cases_v2.py[,bench/cases_private.py] --only robust,longagent,estonian_free,vision,replay,cron_replay,ctx)

Speed is timed on this side from streamed responses, so it works with any server; llama-server's
own timings are stored next to it when present. Quality cases (bench/cases.py, or your own file
with --cases) are graded automatically (exact answers, executed code, simulated tools), each run
--reps times because sampling is not deterministic. Without --temperature the server's default
sampling is used; without --thinking / --reasoning-effort, the server's default reasoning (as an
everyday client gets it). --thinking off sends reasoning_effort "none" and
chat_template_kwargs enable_thinking false, which covers llama-server and Qwen/Gemma-style templates.
Results go to <results>/<label>.json (paths.results in local.toml, default results/). The defaults
for --base-url and --api-key-file come from local.toml [serving] too. Stdlib only.
"""
import argparse
import importlib.util
import json
import os
import statistics
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parent.parent / "pipeline"))
import config  # noqa: E402
from harness import Client  # noqa: E402


def load_sections(path):
    spec = importlib.util.spec_from_file_location("bench_cases", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Row fields that quote a prompt or an answer. Sections a cases file lists in PRIVATE are built from
# real traffic; their rows are stored without these, so result files (which are tracked) hold scores only.
QUOTING = ("answer", "calls", "error", "text", "reply", "detail")


# ---------------------------------------------------------------- speed

def speed(client):
    out = {}
    prompts = {
        "prose": "Write about 400 words explaining how a sailboat can sail upwind. Plain paragraphs.",
        "tool_json": "Output a JSON array of 12 tool calls, each {\"name\": \"set_light\", \"arguments\": "
                     "{\"room\": <a different room>, \"brightness\": <0-100>, \"color_temp\": <2700-6500>}}. "
                     "JSON only, no commentary.",
    }
    for key, prompt in prompts.items():
        runs = []
        for _ in range(3):
            r = client.stream([{"role": "user", "content": f"{prompt}\n(request {uuid.uuid4().hex[:6]})"}], max_tokens=700)
            t = r["timings"]
            runs.append({"tok_s": r["tok_s"], "n": r["n"], "ttft": r["ttft"], "wall": r["wall"],
                         "server_tok_s": t.get("predicted_per_second"),
                         "draft_n": t.get("draft_n"), "draft_accepted": t.get("draft_n_accepted")})
        out[key] = runs
    filler = " ".join(f"Entry {i}: sensor {uuid.uuid4().hex} reported value {i * 37 % 1000}." for i in range(700))
    runs = []
    for _ in range(2):
        r = client.stream([{"role": "user", "content": f"{uuid.uuid4().hex}\n{filler}\nHow many entries are listed? "
                            "Answer with the number only."}], max_tokens=300)
        # time to first token = prompt processing (plus one token and the request's overhead)
        prompt_tok_s = r["prompt_n"] / r["ttft"] if r["prompt_n"] and r["ttft"] else None
        runs.append({"prompt_tok_s": prompt_tok_s, "prompt_n": r["prompt_n"], "ttft": r["ttft"], "wall": r["wall"],
                     "server_prompt_tok_s": r["timings"].get("prompt_per_second")})
    out["long_prompt"] = runs
    out["parallel"], out["alternating"] = parallel(client), alternating(client)
    out["deep_decode"] = deep_decode(client)
    return out


# depth label: filler entries (~43 tokens each); the 90k depth needs a slot of at least ~92k tokens
DEPTHS = {"30k": 700, "90k": 2100}


def deep_decode(client):
    """Generation speed with the context already full, as in a long agent session: a ~30k and a ~90k-token
    prompt, then a ~400-word answer. Short-prompt speed hides how decode slows down as the KV cache grows
    (and how MTP acceptance holds up). A depth the server cannot fit is recorded with its error."""
    out = {}
    for depth, entries in DEPTHS.items():
        filler = " ".join(f"Entry {i}: sensor {uuid.uuid4().hex} reported value {i * 37 % 1000}." for i in range(entries))
        msg = (f"{uuid.uuid4().hex}\n{filler}\nIgnore the list above. Write about 300 words explaining how a sailboat "
               "can sail upwind. Plain paragraphs.")
        try:
            r = client.stream([{"role": "user", "content": msg}], max_tokens=600)
        except Exception as e:  # context too small for this depth, or the server gave up
            out[depth] = {"error": f"{type(e).__name__}: {e}"[:200]}
            continue
        t = r["timings"]
        out[depth] = {"prompt_n": r["prompt_n"], "n": r["n"], "tok_s": r["tok_s"], "ttft": r["ttft"], "wall": r["wall"],
                      "server_tok_s": t.get("predicted_per_second"),
                      "draft_n": t.get("draft_n"), "draft_accepted": t.get("draft_n_accepted")}
    return out


def parallel(client):
    """Both server slots busy at once: two prose requests started together."""
    from concurrent.futures import ThreadPoolExecutor
    prompt = "Write about 350 words on how tides work. Plain paragraphs.\n(request {})"
    with ThreadPoolExecutor(2) as ex:
        rs = list(ex.map(lambda _: client.stream([{"role": "user", "content": prompt.format(uuid.uuid4().hex[:6])}], 500), range(2)))
    return {"runs": [{k: r[k] for k in ("tok_s", "n", "ttft", "wall")} for r in rs],
            "aggregate_tok_s": sum(r["n"] for r in rs) / max(r["wall"] for r in rs)}


def alternating(client):
    """Two long conversations taking turns, like an assistant switching between chats. With a slot each,
    a follow-up re-reads only its new tokens; if the cache is evicted, the whole ~9k-token history."""
    filler = lambda tag: " ".join(f"{tag} note {i}: item {uuid.uuid4().hex[:8]} costs {i * 13 % 97} euros." for i in range(600))
    convs = {c: [{"role": "user", "content": f"{filler(c)}\nRead this list. Reply only 'ready'."}] for c in "AB"}
    turns = []
    for turn in range(3):
        for c in "AB":
            r = client.stream(convs[c], 60)
            turns.append({"conv": c, "turn": turn, "ttft": r["ttft"], "prompt_n": r["prompt_n"]})
            convs[c] += [{"role": "assistant", "content": r["text"].strip() or "ready"},
                         {"role": "user", "content": f"Question {turn + 1}: how many notes were there? Reply with the number only."}]
    follow = [x["ttft"] for x in turns if x["turn"] > 0 and x["ttft"]]
    return {"turns": turns, "first_ttft": statistics.median(x["ttft"] for x in turns if x["turn"] == 0),
            "follow_up_ttft": statistics.median(follow) if follow else None}




def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True)
    ap.add_argument("--base-url", default=config.get("serving", "production_url", "http://127.0.0.1:8080"))
    ap.add_argument("--api-key-file", default=os.path.expanduser(config.get("serving", "api_key_file", "")))
    ap.add_argument("--results-dir", default=str(config.path("results")), help="default: paths.results")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--temperature", type=float, help="sent with every request; default: the server's")
    ap.add_argument("--thinking", choices=["on", "off"], help="force thinking on or off; default: the server's")
    ap.add_argument("--reasoning-effort", help="sent as reasoning_effort (e.g. low, medium, high, none)")
    ap.add_argument("--request-extra", type=json.loads, default={},
                    help="a JSON object merged into every request (an experiment's variant.request_extra)")
    ap.add_argument("--cases", default=",".join(str(Path(__file__).resolve().parent / f) for f in ("cases.py", "cases_v2.py", "cases_private.py")),
                    help="Python file defining SECTIONS (see bench/cases.py); several files, comma-separated; default: the core "
                         "suite plus the extra sections (the private ones are skipped without fixtures). A file that "
                         "defines EXTRA (a set of section names) keeps those sections under the result file's \"extra\" key, "
                         "so the headline totals of the core suite do not change")
    ap.add_argument("--only", help="comma-separated: speed and section names (default: all)")
    ap.add_argument("--model", help="sent as the request's model field (servers that require it, e.g. capy-flash)")
    args = ap.parse_args()
    # model-written code runs with these directories hidden (cases.py reads it at import)
    os.environ.setdefault("BENCH_SANDBOX_HIDE", ":".join(config.get("bench", "sandbox_hide", [])))
    files = [f for f in args.cases.split(",") if f]
    modules = [load_sections(f) for f in files]
    cases = modules[0]
    sections, extra_names, private, versions = {}, set(), set(), {}
    for m in modules:
        sections.update(m.SECTIONS)
        extra_names |= set(getattr(m, "EXTRA", ()))
        private |= set(getattr(m, "PRIVATE", ()))
        versions.update(getattr(m, "SECTION_VERSIONS", {}))
    extra = {}
    if args.thinking:
        extra["chat_template_kwargs"] = {"enable_thinking": args.thinking == "on"}
        if args.thinking == "off":
            extra["reasoning_effort"] = "none"
    if args.reasoning_effort:
        extra["reasoning_effort"] = args.reasoning_effort
    if not isinstance(args.request_extra, dict):
        ap.error("--request-extra must be a JSON object")
    extra.update(args.request_extra)
    if args.model:
        extra["model"] = args.model
    client = Client(args.base_url, Path(args.api_key_file).read_text().strip(), args.temperature, extra)
    only, t0 = (args.only or "speed," + ",".join(sections)).split(","), time.time()
    out = Path(args.results_dir) / f"{args.label}.json"
    res = json.loads(out.read_text()) if out.exists() else {}
    res.update({"label": args.label, "reps": args.reps, "temperature": args.temperature, "request_extra": extra,
                "started": time.strftime("%F %T")})
    if hasattr(modules[0], "SUITE"):  # a run of extra sections alone must not relabel the core suite of this file
        res.update({"cases": Path(files[0]).name, "suite": modules[0].SUITE})
    elif "suite" not in res:
        res.update({"cases": Path(files[0]).name, "suite": None})
    total0 = res.get("total_seconds", 0)

    def save():
        """Written after speed and after every section, so a run that is killed keeps what it finished;
        rerun the rest with --only and the same label (results are merged)."""
        res["total_seconds"] = total0 + time.time() - t0
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(res, indent=1, ensure_ascii=False))

    if "speed" in only:
        res["speed"], res["speed_method"] = speed(client), "client"  # older results: llama-server timings

        def med(k, f, fmt):
            vals = [r[f] for r in res["speed"][k] if r.get(f)]
            return format(statistics.median(vals), fmt) if vals else "n/a"

        server = lambda k, f, fmt: f" (server {med(k, f, fmt)})" if med(k, f, fmt) != "n/a" else ""
        print(f"speed: prose {med('prose', 'tok_s', '.1f')}{server('prose', 'server_tok_s', '.1f')} tok/s | "
              f"tool-json {med('tool_json', 'tok_s', '.1f')}{server('tool_json', 'server_tok_s', '.1f')} tok/s | "
              f"prompt {med('long_prompt', 'prompt_tok_s', '.0f')}{server('long_prompt', 'server_prompt_tok_s', '.0f')} tok/s",
              flush=True)
        par, alt = res["speed"]["parallel"], res["speed"]["alternating"]
        print(f"load:  2 at once {par['aggregate_tok_s']:.1f} tok/s total "
              f"({' + '.join(format(r['tok_s'] or 0, '.1f') for r in par['runs'])}) | alternating chats: "
              f"first turn {alt['first_ttft']:.1f}s, follow-ups {alt['follow_up_ttft'] or 0:.1f}s to first token", flush=True)
        deep = res["speed"]["deep_decode"]
        print("depth: " + " | ".join(
            f"decode after {d}: " + (f"{v['tok_s']:.1f} tok/s" if v.get("tok_s") else v.get("error", "n/a")[:60])
            + (f" (MTP {v['draft_accepted'] / v['draft_n']:.0%})" if v.get("draft_n") else "") for d, v in deep.items()), flush=True)
        save()
    for name, section in sections.items():
        if name not in only:
            continue
        rows = section(client, args.reps)
        if not rows:  # the section decided it does not apply (no vision, no private fixtures, no Estonian checker)
            print(f"{name:10} skipped (not applicable here)", flush=True)
            continue
        stored = [{k: v for k, v in r.items() if k not in QUOTING} for r in rows] if name in private else rows
        if name in extra_names:
            res.setdefault("extra", {})[name] = stored
            res.setdefault("section_versions", {})[name] = versions.get(name, "v1")
        else:
            res[name] = stored
        fails = sorted({r["case"] for r in rows if not r["ok"]})
        print(f"{name:10} {sum(r['ok'] for r in rows)}/{len(rows)} passed | {sum(r['wall'] for r in rows):6.0f}s total | "
              f"median {statistics.median(r['wall'] for r in rows):5.1f}s per case" + (f" | out of tokens: {sum(bool(r.get('truncated')) for r in rows)}" if any(r.get("truncated") for r in rows) else "")
              + (f" | failed: {', '.join(fails)}" if fails else ""), flush=True)
        save()
    save()
    print(f"total {res['total_seconds']:.0f}s -> {out}")


if __name__ == "__main__":
    main()
