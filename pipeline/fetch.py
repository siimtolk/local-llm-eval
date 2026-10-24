#!/usr/bin/env python3
"""Stage 3, fetch: download pending candidates in the daytime, ahead of the nightly test (no LLM).

Run after the morning report, once the day's picks are in the store. For each pending candidate
(oldest first) it lists the repo's files through the Hugging Face API, picks a GGUF quant that fits
(a Q4-class one first, else the largest that fits, whatever its naming; all parts of a split one)
plus a vision projector and an MTP / draft file when the repo has them, checks that it fits this
machine ([machine]) and the disk, and downloads it to <paths.models>/<model>/. Production keeps serving.

Fit (as in bench/NIGHTLY.md): weights up to machine.fit_gb fit in VRAM with two 128k slots; up to
machine.offload_gb only for a MoE with at most machine.max_active_b active parameters (experts
offloaded to RAM). At most machine.max_ready candidates wait downloaded, so the disk does not fill up.

vLLM lane: after a GGUF candidate is downloaded, Hugging Face is searched once for a 4-bit
safetensors build of the same model (NVFP4, AWQ, GPTQ; no repo code, no trust_remote_code) that
fits vLLM (weights up to machine.vllm_fit_gb). The best one is added to the store as its own
pending candidate (backend "vllm", variant_of the GGUF repo, priority -1 so it is tested after the
GGUF ones) and downloaded on a later run to models/vllm-<name>/. A pending candidate whose repo has
no GGUF but is such a checkpoint itself goes the same way. The nightly runner tests those with
bench/NIGHTLY_VLLM.md.

Highest priority first (candidates.py add queues at 1). After each pass the store is read again, so a
repo queued while this run downloads is taken in the same run. A candidate queued with `add --now`
(test_now) may take one slot over machine.max_ready; once it is downloaded, `runner.py bkg-start` hands
the GPUs to the model test (at most once per run) and the outcome goes to the chat.

Usage: fetch.py [--dry-run]   (--dry-run: pick and print, download nothing, change nothing)
"""
import json
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from urllib.parse import quote

import candidates
import config
import notify
import params
from texts import t

HF = config.get("huggingface", "cli", "hf")
MODELS = config.models_dir()
LOGS = config.path("logs", config.serving())
CONFIGS = config.path("model_configs", config.serving())
TREE = "https://huggingface.co/api/models/{}/tree/main?recursive=true"
QUANTS = ("IQ4_XS", "Q4_K_M", "Q4_K_S", "IQ4_NL", "Q4_0")  # preference order
DRAFT = re.compile(r"mtp|draft|assistant|dflash", re.I)
DRAFT_MAX = 3e9  # a drafter is small; a "...-MTP" model's own weights are not
SPLIT = re.compile(r"-\d{5}-of-\d{5}\.gguf$", re.I)
FIT_GB, OFFLOAD_GB, MAX_ACTIVE_B = (config.get("machine", k) for k in ("fit_gb", "offload_gb", "max_active_b"))
MIN_FREE_GB = config.get("machine", "min_free_gb")
MAX_READY = config.get("machine", "max_ready")  # a 5-hour window tests up to ~4-5
SEARCH = "https://huggingface.co/api/models?search={}&limit=50"
RAW = "https://huggingface.co/{}/resolve/main/{}"
# 27B NVFP4 weights of 20-22 GB served 2 x 128k with int4 KV on vLLM 0.32 on 2 x 16 GB (2026-10-21)
VLLM_FIT_GB = config.get("machine", "vllm_fit_gb")
VLLM_NAME = re.compile(r"nvfp4|fp4|awq|gptq|int4|w4a16|w4a4", re.I)
VLLM_NOT = re.compile(r"gguf|mlx|exl\d|onnx", re.I)
VLLM_METHODS = ("modelopt", "compressed-tensors", "awq", "gptq")
# base models from these orgs are the stock originals: sharing one says nothing about being the same build
OFFICIAL = {o.lower() for o in config.get("huggingface", "official_orgs", [])}
INFO = "https://huggingface.co/api/models/{}"
VLLM_SINCE = "2026-10-21"  # the lane started then; older candidates are not searched
VLLM_SKIP_EXT = (".py", ".bin", ".pt", ".pth", ".gguf", ".onnx", ".msgpack", ".h5")


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def all_files(repo):
    return {f["path"]: (f.get("lfs") or {}).get("size", f.get("size", 0))
            for f in get_json(TREE.format(quote(repo, safe="/"))) if f.get("type") == "file"}


def files(repo):
    return {p: s for p, s in all_files(repo).items() if p.lower().endswith(".gguf")}


def vllm_pick(repo, listing=None):
    """(quant label, weight GB, files to download, reason) for a vLLM-ready 4-bit safetensors checkpoint;
    reason is non-empty when the repo is not one or does not fit."""
    listing = listing if listing is not None else all_files(repo)
    weights = [p for p in listing if p.endswith(".safetensors")]
    if not weights or "config.json" not in listing:
        return "", 0, [], "no safetensors"
    if any(p.endswith(".py") for p in listing):
        return "", 0, [], "repo ships code"
    cfg = get_json(RAW.format(quote(repo, safe="/"), "config.json"))
    if cfg.get("auto_map"):
        return "", 0, [], "needs trust_remote_code"
    q = cfg.get("quantization_config") or {}
    if not q and "hf_quant_config.json" in listing:  # older modelopt exports
        q = get_json(RAW.format(quote(repo, safe="/"), "hf_quant_config.json")).get("quantization") or {}
        q.setdefault("quant_method", "modelopt")
    method = str(q.get("quant_method", "")).lower()
    if not method.startswith(VLLM_METHODS):
        return "", 0, [], f"not a 4-bit vLLM checkpoint ({method or 'unquantized'})"
    label = str(q.get("format") or q.get("quant_algo") or method)
    gb = sum(listing[p] for p in weights) / 1e9
    if gb > VLLM_FIT_GB:
        return label, gb, [], f"{gb:.0f} GB > {VLLM_FIT_GB} GB"
    keep = [p for p in listing if "/" not in p and not p.lower().endswith(VLLM_SKIP_EXT)]
    return label, gb, keep, ""


def lineage(repo):
    """Lower-case repo ids naming the build a repo is: itself and its non-official base models. Names
    alone are too generic to match on (several "Qwen3.8-27B-Uncensored" builds exist)."""
    try:
        bases = (get_json(INFO.format(quote(repo, safe="/"))).get("cardData") or {}).get("base_model") or []
    except (OSError, ValueError):
        bases = []
    keys = set()
    for r in [repo] + ([bases] if isinstance(bases, str) else list(bases)):
        if not isinstance(r, str) or "/" not in r or (r != repo and r.split("/")[0].lower() in OFFICIAL):
            continue
        keys.add(r.lower())
    return keys


def vllm_sibling(repo, taken):
    """(repo, label, GB) of the best vLLM-ready build of the GGUF candidate repo, or None. A build counts
    only when its base_model is the GGUF repo or the same non-official model (a same-named NVFP4 of
    another uncensored fine-tune does not)."""
    stem = re.sub(r"[-_.]?(gguf|i1|imatrix)(?=$|[-_.])", "", repo.split("/")[1], flags=re.I)
    parent = lineage(repo)
    found = []
    for term in dict.fromkeys((stem, re.sub(r"[-_.]mtp(?=$|[-_.])", "", stem, flags=re.I))):
        try:
            found += [m["id"] for m in get_json(SEARCH.format(quote(term)))]
        except OSError:
            pass
    names = [r for r in dict.fromkeys(found) if candidates.valid_repo(r) and r not in taken
             and VLLM_NAME.search(r) and not VLLM_NOT.search(r)]
    names.sort(key=lambda r: not re.search(r"nvfp4|fp4", r, re.I))  # NVFP4 first: native on the 5060 Ti
    for r in names[:8]:
        if not (lineage(r) - {r.lower()}) & parent:
            continue
        try:
            label, gb, keep, why = vllm_pick(r)
        except (OSError, ValueError):
            continue
        if not why:
            return r, label, gb
    return None


def add_sibling(parent, data_c):
    """Look once for a vLLM build of a downloaded GGUF candidate and queue it."""
    with candidates.store() as data:
        taken = set(data["candidates"])
        c = data["candidates"].get(parent) or {}
        if c.get("vllm_checked"):
            return
        c["vllm_checked"] = candidates.now()
    sib = vllm_sibling(parent, taken)
    if not sib:
        print(f"{parent}: no vLLM-ready build found")
        return
    repo, label, gb = sib
    print(f"{parent}: vLLM build {repo} ({label}, {gb:.1f} GB) queued")
    with candidates.store() as data:
        data["candidates"].setdefault(repo, {
            "name": f"{data_c.get('name', parent)} (vLLM {label})", "size": f"~{gb:.0f} GB · vLLM {label}",
            "link": f"https://huggingface.co/{repo}", "brief": data_c.get("brief", ""), "status": "pending",
            "updated": candidates.now(), "summary": "", "reported": True,
            "backend": "vllm", "variant_of": parent, "priority": -1})


def fetch_vllm(repo, c, dry, label="", gb=0, keep=None):
    """Download a vLLM candidate; returns True when it took a download slot."""
    if keep is None:
        label, gb, keep, why = vllm_pick(repo)
        if why:
            print(f"{repo}: {why}")
            if not dry:
                candidates.update(repo, status="skipped", summary=t("fetch.vllm_unfit", why=why))
            return False
    listing_gb = gb
    free_gb = shutil.disk_usage(MODELS).free / 1e9
    if free_gb - listing_gb < MIN_FREE_GB:
        print(f"{repo}: not enough disk ({free_gb:.0f} GB free, needs {listing_gb:.0f} GB + {MIN_FREE_GB})")
        return False
    model = model_name("vllm/vllm-" + repo.split("/")[1])
    print(f"{repo}: {model} vLLM {label} {gb:.1f} GB {len(keep)} files")
    if dry:
        return False
    candidates.update(repo, status="downloading", model=model, quant=label, backend="vllm", files=keep, size_gb=round(gb, 1))
    with open(LOGS / f"{model}-download.log", "w") as out:
        done = subprocess.run(["nice", "-n", "10", HF, "download", repo, *keep,
                               "--local-dir", str(MODELS / model)], stdout=out, stderr=out).returncode == 0
    if done:
        candidates.update(repo, status="downloaded")
    else:
        shutil.rmtree(MODELS / model, ignore_errors=True)
        candidates.update(repo, status="pending", model="")
        print(f"{repo}: download failed, see logs/{model}-download.log", file=sys.stderr)
    return True


def quant_tag(path):
    """The quant named in a file name (Q6_K, IQ3_S, BF16, Q4LynnStyle, ...), else the file stem."""
    stem = SPLIT.sub("", Path(path).name).removesuffix(".gguf")
    found = re.search(r"(?<![a-z0-9])(i?q\d+[a-z0-9_]*|pq\d_\d|bf16|f16|f32)(?![a-z0-9])", stem, re.I)
    return found.group(1) if found else stem


def pick(repo, listing, want=""):
    """(quant tag, weight files, extra files, fits) for the best weight set.

    want = a quant tag set by hand in the store (`quant_override`), for models the size rule cannot
    judge (e.g. a huge embedding table that stays memory-mapped): that set is taken as it is.

    A Q4-class quant in QUANTS order when one fits; otherwise the largest weight set (any quant or
    naming) that fits; otherwise the smallest set, with fits False. None when the repo has no GGUF weights."""
    is_draft = lambda p: bool(DRAFT.search(Path(p).name)) and listing[p] < DRAFT_MAX and "imatrix" not in p.lower()
    weights = {p: s for p, s in listing.items()
               if "mmproj" not in p.lower() and not is_draft(p) and "imatrix" not in p.lower()}
    sets = {}  # one entry per model file; the parts of a split file together
    for p in sorted(weights):
        sets.setdefault(SPLIT.sub("", p), []).append(p)
    if not sets:
        return None
    gb = lambda parts: sum(listing[p] for p in parts) / 1e9
    rank = lambda parts: next((i for i, tag in enumerate(QUANTS)
                               if re.search(rf"(^|[-_.]){tag}([-_.]|$)", Path(parts[0]).name, re.I)), len(QUANTS))
    ordered = sorted(sets.values(), key=lambda parts: (rank(parts), -gb(parts)))
    wanted = [s for s in ordered if want and quant_tag(s[0]).lower() == want.lower()]
    parts = wanted[0] if wanted else next((s for s in ordered if fits(repo, gb(s))), None)
    fit = parts is not None
    parts = parts or min(sets.values(), key=gb)
    mmproj = sorted((p for p in listing if "mmproj" in p.lower()),
                    key=lambda p: (not re.search(r"bf16|f16", p, re.I), listing[p]))[:1]
    drafts = sorted((p for p in listing if is_draft(p) and "mmproj" not in p.lower()), key=lambda p: listing[p])[:1]
    return quant_tag(parts[0]), parts, mmproj + drafts, fit


def model_name(repo):
    name = repo.split("/")[1].lower()
    name = re.sub(r"[-_.]?(gguf|i1|imatrix)(?=$|[-_.])", "", name)
    name = re.sub(r"[^a-z0-9.-]+", "-", name).strip("-.")
    if len(name) > 40:
        name = name[:41].rsplit("-", 1)[0]
    name = name or "candidate"
    taken = {p.name for p in MODELS.iterdir()} | {p.name.removesuffix(".conf.sh") for p in CONFIGS.iterdir()}
    base, n = name, 2
    while name in taken:
        name, n = f"{base}-{n}", n + 1
    return name


def fits(repo, gb):
    if gb <= FIT_GB:
        return True
    active = params.active_b(repo)  # the name, else the model card or its base model's
    return gb <= OFFLOAD_GB and bool(active) and active <= MAX_ACTIVE_B


def start_test(repo, allowed):
    """A downloaded candidate queued with `candidates.py add --now` hands the GPUs to the model test
    now (runner.py bkg-start, which tests the whole downloaded queue) and says so in the chat. Only
    when allowed (once per fetch run); the flag is cleared either way. True when it ran bkg-start."""
    with candidates.store(readonly=True) as data:
        c = data["candidates"].get(repo) or {}
    if not c.get("test_now") or c.get("status") != "downloaded":
        return False
    candidates.update(repo, test_now=False)
    if not allowed:
        return False
    try:
        run = subprocess.run([sys.executable, str(config.ROOT / "pipeline" / "runner.py"), "bkg-start"],
                             cwd=config.ROOT, capture_output=True, text=True, timeout=120)
        lines = (run.stdout + run.stderr).strip().splitlines()
    except subprocess.TimeoutExpired:
        lines = ["runner.py bkg-start: no answer within 120 s"]
    print(f"{repo}: test_now, bkg-start: {' / '.join(lines[-5:])}")
    notify.send(t("fetch.test_now", repo=repo, out="\n".join(lines[-5:])))
    return True


def main():
    dry = "--dry-run" in sys.argv
    seen, first, tested = set(), True, False
    while True:  # after each pass the store is read again: a repo queued meanwhile (candidates.py add) is taken too
        with candidates.store(readonly=True) as data:
            ready = sum(c["status"] in ("downloading", "downloaded") for c in data["candidates"].values())
            # highest priority first (add 1, judge picks 0, vLLM builds -1), then oldest brief
            todo = sorted((-int(c.get("priority", 0)), c.get("brief", ""), repo, dict(c)) for repo, c in data["candidates"].items()
                          if c["status"] == "pending" and candidates.valid_repo(repo) and repo not in seen)
            unchecked = [(repo, dict(c)) for repo, c in data["candidates"].items()
                         if c["status"] in ("downloaded", "testing", "tested", "rejected") and c.get("model")
                         and c.get("backend") != "vllm" and c.get("brief", "") >= VLLM_SINCE
                         and not c.get("vllm_checked") and candidates.valid_repo(repo)] if first else []
        if dry and first:
            print(f"dry run: {len(todo)} pending, {ready} downloading or downloaded (max {MAX_READY}), "
                  f"{len(unchecked)} GGUF candidates not yet searched for a vLLM build")
        if not dry:  # GGUF candidates downloaded earlier (rejected too: the backend is the question): look once
            for repo, c in unchecked:
                add_sibling(repo, c)
        first = False
        if not todo:
            return
        for _, _, repo, c in todo:
            seen.add(repo)
            if ready >= MAX_READY + bool(c.get("test_now")):  # one asked to be tested now may take one slot more
                continue
            try:
                if c.get("backend") == "vllm":
                    ready += fetch_vllm(repo, c, dry)
                    tested |= not dry and start_test(repo, not tested)
                    continue
                full = all_files(repo)
            except (OSError, ValueError) as exc:
                print(f"{repo}: listing failed: {exc}", file=sys.stderr)
                continue
            listing = {p: s for p, s in full.items() if p.lower().endswith(".gguf")}
            chosen = pick(repo, listing, c.get("quant_override", ""))
            if not chosen:
                try:
                    label, gb, keep, why = vllm_pick(repo, full)
                except (OSError, ValueError):
                    why = "no safetensors"
                if not why:  # no GGUF, but a vLLM-ready checkpoint itself
                    if not dry:
                        candidates.update(repo, backend="vllm", priority=-1)
                    ready += fetch_vllm(repo, c, dry, label, gb, keep)
                    tested |= not dry and start_test(repo, not tested)
                    continue
                print(f"{repo}: no GGUF weights ({why})")
                if not dry:
                    candidates.update(repo, status="skipped", summary=t("fetch.no_weights"))
                continue
            tag, parts, extras, fit = chosen
            gb = sum(listing[p] for p in parts) / 1e9
            total_gb = gb + sum(listing[p] for p in extras) / 1e9
            free_gb = shutil.disk_usage(MODELS).free / 1e9
            if not fit:
                print(f"{repo}: {gb:.1f} GB does not fit")
                if not dry:
                    candidates.update(repo, status="skipped", summary=t("fetch.too_big", tag=tag, gb=gb))
                continue
            if free_gb - total_gb < MIN_FREE_GB:
                print(f"{repo}: not enough disk ({free_gb:.0f} GB free, needs {total_gb:.0f} GB + {MIN_FREE_GB})")
                return
            model = model_name(repo)
            print(f"{repo}: {model} {tag} {gb:.1f} GB {parts + extras}")
            if dry:
                continue
            candidates.update(repo, status="downloading", model=model, quant=tag, files=parts + extras, size_gb=round(gb, 1))
            ready += 1
            with open(LOGS / f"{model}-download.log", "w") as out:
                done = subprocess.run(["nice", "-n", "10", HF, "download", repo, *parts, *extras,
                                       "--local-dir", str(MODELS / model)], stdout=out, stderr=out).returncode == 0
            if done:
                candidates.update(repo, status="downloaded")
                tested |= start_test(repo, not tested)
                add_sibling(repo, c)
            else:
                shutil.rmtree(MODELS / model, ignore_errors=True)
                candidates.update(repo, status="pending", model="")
                print(f"{repo}: download failed, see logs/{model}-download.log", file=sys.stderr)
        if dry:
            return


if __name__ == "__main__":
    main()
