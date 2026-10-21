#!/usr/bin/env python3
"""Stage 1, discover: new open-weight models on Hugging Face that could replace production.

Looks for new open-weight language models, and new uncensored (abliterated, heretic, ...)
versions of existing ones ([huggingface] search_words and trusted_publishers), that would run on
this machine ([machine]) and could replace the production model. Keeps only what fits the
hardware, drops what was already reported or tried, and builds the judge's input: the candidates
with model-card excerpts and r/LocalLLaMA reactions, for the judge (judge.py, stage 2) to decide
in a single turn. When nothing new fits, the judge is not needed: judge.py posts a short
"nothing new" message (what was excluded and the closest misses).

A batch counts as reported only once a judge run that got it succeeded (judge.py records that);
after a failed or killed run the same models are offered again.

Usage: discover.py [--dry-run]   prints the judge input; --dry-run leaves the state file alone
"""
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

import config
import params
from short_links import link_lines
from web import ATOM, clean, community, get, parse_date

PRODUCTION = config.get("production", "description")
HARDWARE = config.get("machine", "description")
# Repos already served or tested before the store kept track ([huggingface] tried).
TRIED = set(config.get("huggingface", "tried", []))
STATE = config.path("state") / "discover_state.json"

HF_API = ("https://huggingface.co/api/models?{}&expand[]=createdAt&expand[]=likes&expand[]=downloads"
          "&expand[]=pipeline_tag&expand[]=safetensors&expand[]=gguf&expand[]=cardData&expand[]=tags")
QUERIES = {
    "trending": "sort=trendingScore&limit=80",
    **{f"search {w}": f"search={w}&sort=trendingScore&limit=25"
       for w in config.get("huggingface", "search_words", ["uncensored", "abliterated", "heretic"])},
    # Uploaders whose new repos are uncensored versions of notable models.
    **{f"author {a}": f"author={a}&sort=createdAt&direction=-1&limit=15"
       for a in config.get("huggingface", "trusted_publishers", [])},
}
README_URL = "https://huggingface.co/{}/raw/main/README.md"
REDDIT_URL = "https://www.reddit.com/r/LocalLLaMA/top/.rss?t=day"

TEXT_PIPELINES = {"text-generation", "image-text-to-text", None}
UNCENSORED = re.compile(r"uncensored|abliterat|heretic|hauhau|derestrict|unalign|dealign", re.I)
NOT_A_MODEL = re.compile(r"lora|adapter|embed|reward|judge|classif|\bocr\b|guard|draft|eagle|mtp-head", re.I)
GB_PER_B = 0.56  # a Q4-class quant (IQ4_XS-Q4_K_M) is about 4.5 bits per weight
GB_PER_B_MOE_MIN = 0.44  # an IQ3-class quant (~3.5 bits): the smallest a large offloaded MoE is tried at
FIT_GB = config.get("machine", "fit_gb")  # weights that leave room for 128k context in VRAM
OFFLOAD_GB = config.get("machine", "offload_gb")  # a MoE this size still runs with its experts in system RAM
OFFLOAD_MAX_ACTIVE_B = config.get("machine", "max_active_b")  # more active parameters than this is too slow from RAM
MIN_PARAMS_B = config.get("machine", "min_params_b")  # smaller models are no replacement
MIN_LIKES = 10  # fewer: wait, it stays eligible until it is MAX_AGE_DAYS old
MAX_AGE_DAYS = 30
FIRST_RUN_DAYS = 7
MAX_CANDIDATES = 8
README_CHARS = 700
REDDIT_POSTS = 6
REDDIT_THREADS = 2  # each one costs about a minute of waiting
REDDIT_MATCH = re.compile(r"\b\d+(\.\d+)?B\b|uncensored|abliterat|heretic|gguf|release|quant", re.I)
SEEN_KEEP_DAYS = 120


def billions(model):
    """(total, active) parameters in billions; active is None for a dense model or when unknown."""
    name = model["id"].split("/")[-1]
    sizes = re.findall(r"(?<![\w.])(\d+(?:\.\d+)?)B(?![a-z])", name, re.I)
    # The name wins: packed quants (NVFP4, GPTQ) store two or more weights per counted value.
    total = float(sizes[0]) if sizes else None
    if not total:
        counted = (model.get("safetensors") or {}).get("total") or (model.get("gguf") or {}).get("total")
        total = counted / 1e9 if counted else None
    active = params.NAME.search(name)
    active = float(active.group(1)) if active else None
    # The name rarely says it for new releases ("Qwen3.8-Flash-Next"): then the card (or the base
    # model's) does. Looked up only for models too big for VRAM, where it decides the fit.
    if active is None and total and total * GB_PER_B > FIT_GB:
        active = params.active_b(model["id"])
    return total, active


def base_of(model):
    base = (model.get("cardData") or {}).get("base_model")
    return (base[0] if isinstance(base, list) and base else base) or None


def hf_models(name):
    out = []
    for m in json.loads(get(HF_API.format(QUERIES[name]))):
        m["via"] = name
        out.append(m)
    return out


def fit(model):
    """"fits", "offload" or None (does not fit, or is not a replacement candidate)."""
    total, active = billions(model)
    if not total or total < MIN_PARAMS_B:
        return None
    gb = total * GB_PER_B
    if gb <= FIT_GB:
        return "fits"
    # A big MoE with few active parameters runs with its experts in RAM, at a 3-bit quant if Q4 is too big.
    if active and active <= OFFLOAD_MAX_ACTIVE_B and total * GB_PER_B_MOE_MIN <= OFFLOAD_GB:
        return "offload"
    return None


def readme(model_id):
    try:
        text = get(README_URL.format(quote(model_id, safe="/"))).decode("utf8", "ignore")
    except Exception:
        return ""
    text = re.sub(r"\A---.*?\n---\s*", "", text, flags=re.S)  # YAML front matter
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)|<img[^>]*>", " ", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[#*`|>_~-]{2,}|^\s*[#>|-]+", " ", text, flags=re.M)
    return clean(text, README_CHARS)


def reddit_posts():
    root = ET.fromstring(get(REDDIT_URL))
    out = []
    for e in root.iter(f"{ATOM}entry"):
        title = clean(e.findtext(f"{ATOM}title"))
        link = next((l.get("href") for l in e.findall(f"{ATOM}link")), None)
        if title and link and REDDIT_MATCH.search(title):
            out.append({"title": title, "link": link, "summary": clean(e.findtext(f"{ATOM}content"))})
    return out[:REDDIT_POSTS]



def profile_block():
    """The owner's notes ([judge] profile_files) as a block of the judge input; "" while there are none."""
    texts = []
    for name in config.get("judge", "profile_files", []):
        try:
            texts.append(Path(name).expanduser().read_text().strip())
        except OSError:
            pass
    texts = [t for t in texts if t]
    if not texts:
        return ""
    return "\n\n".join(["MY PROFILE (use it to choose and rank; the rules in the instructions still apply)"] + texts)


def build(dry=False):
    """(judge input text or None when nothing new, quiet message, batch ids)."""
    now = datetime.now(timezone.utc)
    first_run = not STATE.exists()
    state = {"seen": {}, "pending": None} if first_run else json.loads(STATE.read_text())
    seen = state["seen"]
    models, failed = {}, []
    with ThreadPoolExecutor(8) as pool:
        futures = [(name, pool.submit(hf_models, name)) for name in QUERIES]
    for name, future in futures:
        try:
            for m in future.result():
                models.setdefault(m["id"], m)
        except Exception as exc:
            failed.append(name)
            print(f"discover: {name} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    if len(failed) == len(QUERIES):
        sys.exit("discover: every Hugging Face query failed")

    pending = state.get("pending")
    if pending and state.get("judged_ok_at", "") > pending["offered_at"]:
        for k in pending["ids"]:
            seen[k] = now.date().isoformat()
    state["pending"] = None
    oldest = now - timedelta(days=FIRST_RUN_DAYS if first_run else MAX_AGE_DAYS)
    batch, excluded, close = [], Counter(), []
    for m in models.values():
        created = parse_date(m.get("createdAt"))
        base = base_of(m)
        reason = ("already reported" if m["id"] in seen else "tested" if m["id"] in TRIED
                  else f"older than {MAX_AGE_DAYS} days" if not created or created < oldest
                  else "not text models" if m.get("pipeline_tag") not in TEXT_PIPELINES or NOT_A_MODEL.search(m["id"])
                  else "too big" if not fit(m)
                  else f"under {MIN_LIKES} likes" if (m.get("likes") or 0) < MIN_LIKES else None)
        if reason:
            excluded[reason] += 1
            if reason.startswith("under"):
                close.append((m.get("likes") or 0, m["id"].split("/", 1)[1]))
            continue
        where = fit(m)
        total, active = billions(m)
        batch.append({
            "id": m["id"], "base": base if base != m["id"] else None, "created": created.date().isoformat(),
            "params": f"{total:.0f}B" + (f" total, {active:g}B active (MoE)" if active else " dense"),
            "size": f"~{total * GB_PER_B:.0f} GB at Q4" + (f", ~{total * GB_PER_B_MOE_MIN:.0f} GB at IQ3"
                                                           if total * GB_PER_B > OFFLOAD_GB else ""), "fit": where,
            "uncensored": bool(UNCENSORED.search(m["id"]) or UNCENSORED.search(" ".join(m.get("tags") or []))),
            "gguf": "gguf" in (m.get("tags") or []) or "gguf" in m["id"].lower(),
            "likes": m.get("likes") or 0, "downloads": m.get("downloads") or 0,
        })
    # Uncensored first, then by popularity.
    batch.sort(key=lambda c: (not c["uncensored"], -c["likes"]))
    batch = batch[:MAX_CANDIDATES]

    posts = []
    if batch:
        with ThreadPoolExecutor(8) as pool:
            for c, text in zip(batch, pool.map(readme, [c["id"] for c in batch])):
                c["card"] = text
        try:
            # Threads about a candidate get their comments fetched first.
            words = {w for c in batch for w in re.split(r"[/_.-]", c["id"].lower()) if len(w) > 3}
            posts = sorted(reddit_posts(), key=lambda p: -len(words & set(re.split(r"[\s/_.-]", p["title"].lower()))))
            for post in posts[:REDDIT_THREADS]:
                community(post)
        except Exception as exc:
            print(f"discover: r/LocalLLaMA failed: {type(exc).__name__}: {exc}", file=sys.stderr)

    cutoff = (now - timedelta(days=SEEN_KEEP_DAYS)).date().isoformat()
    state["seen"] = {k: d for k, d in seen.items() if d >= cutoff}
    if batch:
        state["pending"] = {"ids": [c["id"] for c in batch], "offered_at": now.isoformat()}
    if not dry:
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps(state, indent=1))

    summary = "Excluded: " + ", ".join(f"{n} {why}" for why, n in excluded.most_common()) + "." if excluded else ""
    if close:
        summary += "\nClose: " + ", ".join(f"{name} ({likes} likes)" for likes, name in sorted(close, reverse=True)[:3]) + "."
    if not batch:
        return None, f"**Local models**\nNo new candidates today ({len(models)} checked).\n{summary}".strip(), []

    out = []
    emit = lambda text: out.append(text)  # the judge input, block by block
    emit(f"CURRENT MODEL: {PRODUCTION}\nHARDWARE: {HARDWARE}\n\nNEW CANDIDATES: {len(batch)}")
    links = link_lines([f"https://huggingface.co/{c['id']}" for c in batch])  # copied as-is into the brief
    for n, c in enumerate(batch, 1):
        fits = "fits fully in VRAM" if c["fit"] == "fits" else "runs only with MoE experts offloaded to RAM (slower)"
        lines = [
            f"\n[{n}] {c['id']}",
            f"base model: {c['base'] or '(this is the base model)'}",
            f"uploaded: {c['created']} · {c['likes']} likes · {c['downloads']} downloads",
            f"parameters: {c['params']} · size: {c['size']} · {fits}",
            f"uncensored: {'yes' if c['uncensored'] else 'no (no uncensored weights in this repo)'}",
            f"GGUF in this repo: {'yes' if c['gguf'] else 'no'}",
        ]
        if c.get("card"):
            lines.append(f"model card excerpt: {c['card']}")
        lines.append(f"link: {links['https://huggingface.co/' + c['id']]}")
        emit("\n".join(lines))
    if posts:
        emit("\n\nr/LocalLLaMA TODAY (community context; may or may not be about the candidates):")
        for post in posts:
            lines = [f"\n- {post['title']}"]
            if post["summary"]:
                lines.append(f"  summary: {post['summary']}")
            lines.append(f"  link: {post['link']}")
            for text in (post.get("community") or {}).get("comments", []):
                lines.append(f'  comment: "{text}"')
            emit("\n".join(lines))
    emit(f"\n\nEXCLUDED (copy as the brief's last lines, as-is):\n{summary}")
    if profile := profile_block():
        emit("\n\n" + profile)
    return "\n".join(out), "", [c["id"] for c in batch]


def mark_judged():
    """The judge run that got the latest batch succeeded: report those models as seen next time."""
    if STATE.exists():
        state = json.loads(STATE.read_text())
        state["judged_ok_at"] = datetime.now(timezone.utc).isoformat()
        STATE.write_text(json.dumps(state, indent=1))


def main():
    text, quiet, _ = build(dry="--dry-run" in sys.argv)
    print(text or quiet)


if __name__ == "__main__":
    main()
