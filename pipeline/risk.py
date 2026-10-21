#!/usr/bin/env python3
"""Risk check for a local model candidate before it is offered for download (no LLM).

What could go wrong when the host downloads a Hugging Face repo's GGUF and runs it with
llama.cpp, and what this looks at:

- Malicious files: Hugging Face's per-file scans (protectAI, ClamAV, VirusTotal, JFrog, which
  also checks a GGUF chat template for template injection) of the file that would be downloaded; pickle
  weights (.bin/.pt/.pkl/.ckpt) and scripts or binaries in the repo. The host downloads only
  .gguf files, so these are a signal about the uploader rather than a direct threat.
- A GGUF that will not load or targets a parser bug: its header is read with an HTTP range
  request (a few MB, not the weights) and parsed; an unreadable header, an unknown tensor type
  (a custom fork's format, e.g. PrismML's 142/143) or absurd counts mean it is not offered.
- Who uploaded it: a little-known account with a fresh, barely downloaded repo is flagged.
- The chat template inside the GGUF is shown by size only; llama.cpp renders it with its own
  Jinja engine, not Python, so it cannot run code.
- Running it: model output is data. The bench runs model-written code only inside bubblewrap
  (no network, nothing writable but a temp dir), so that is not checked per model.

level: low | medium | high (not offered) | wontload (not offered).

Usage: risk.py <repo>     prints the assessment as JSON
"""
import json
import re
import struct
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone

from texts import t

API = "https://huggingface.co/api"
UA = {"User-Agent": "local-llm-eval-risk/1"}
PICKLE = re.compile(r"\.(bin|pt|pth|pkl|pickle|ckpt|joblib|npy|npz)$", re.I)
CODE = re.compile(r"\.(py|sh|bash|exe|bat|cmd|ps1|dll|so|dylib|jar|msi|scr|vbs|js)$", re.I)
PREFERRED = ("Q4_K_M", "IQ4_XS", "Q4_K_S", "Q4_K", "Q4_0", "Q4")
MAX_GGML_TYPE = 63  # standard ggml types are below this; forks use their own high numbers
HEADER_LIMIT = 96 * 2**20


def get_json(url):
    return json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30).read())


def get_range(url, start, end):
    req = urllib.request.Request(url, headers=UA | {"Range": f"bytes={start}-{end}"})
    return urllib.request.urlopen(req, timeout=60).read()


class Short(Exception):
    pass


class Reader:
    """Reads a GGUF header from the start of the file, fetching more on demand."""

    def __init__(self, url):
        self.url, self.buf, self.pos = url, b"", 0

    def need(self, n):
        while self.pos + n > len(self.buf):
            if len(self.buf) >= HEADER_LIMIT:
                raise Short
            chunk = get_range(self.url, len(self.buf), min(len(self.buf) * 2 or 4 * 2**20, HEADER_LIMIT) - 1)
            if not chunk:
                raise Short
            self.buf += chunk

    def take(self, fmt):
        n = struct.calcsize(fmt)
        self.need(n)
        value = struct.unpack_from("<" + fmt, self.buf, self.pos)
        self.pos += n
        return value[0] if len(value) == 1 else value

    def string(self):
        n = self.take("Q")
        if n > 2**26:
            raise ValueError(f"string of {n} bytes")
        self.need(n)
        s = self.buf[self.pos:self.pos + n]
        self.pos += n
        return s.decode("utf-8", "replace")


SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def value(r, kind, keep=True):
    if kind in SCALAR:
        return r.take(SCALAR[kind])
    if kind == 8:
        return r.string()
    if kind == 9:
        item, n = r.take("I"), r.take("Q")
        if n > 2**24:
            raise ValueError(f"array of {n} items")
        items = [value(r, item, keep=False) for _ in range(n)]
        return len(items)  # arrays (vocab, merges) only by size
    raise ValueError(f"unknown value type {kind}")


def gguf_header(url):
    """{arch, name, tensors, types, template_chars, header_mb} or raises ValueError."""
    r = Reader(url)
    if r.take("4s") != b"GGUF":
        raise ValueError("not a GGUF file")
    version, n_tensors, n_kv = r.take("I"), r.take("Q"), r.take("Q")
    if version not in (2, 3) or n_tensors > 100_000 or n_kv > 100_000:
        raise ValueError(f"implausible header (v{version}, {n_tensors} tensors, {n_kv} keys)")
    meta = {}
    for _ in range(n_kv):
        key = r.string()
        meta[key] = value(r, r.take("I"))
    types = set()
    for _ in range(n_tensors):
        r.string()
        dims = r.take("I")
        if dims > 8:
            raise ValueError(f"tensor with {dims} dimensions")
        for _ in range(dims):
            r.take("Q")
        types.add(r.take("I"))
        r.take("Q")
    template = meta.get("tokenizer.chat_template", "")
    return {"arch": meta.get("general.architecture", "?"), "name": meta.get("general.name", ""),
            "tensors": n_tensors, "types": sorted(types),
            "template_chars": len(template) if isinstance(template, str) else 0,
            "header_mb": round(r.pos / 2**20, 1)}


def sample_file(files, sizes=None):
    """The weight file the host would pick: a Q4-class quant, else the largest (a small extra GGUF
    is usually a drafter, e.g. a DFlash file without "dflash" in its name)."""
    weights = [f for f in files if f.lower().endswith(".gguf") and "mmproj" not in f.lower() and "imatrix" not in f.lower()]
    for tag in PREFERRED:
        hit = [f for f in weights if re.search(rf"(^|[-_.]){tag}([-_.]|$)", f.rsplit("/", 1)[-1], re.I)]
        if hit:
            return sorted(hit)[0]
    return max(weights, key=lambda f: (sizes or {}).get(f, 0)) if weights else None


def file_scans(repo, paths):
    """{path: {scanner: status}} from Hugging Face's per-file scans (protectAI, ClamAV, VirusTotal,
    JFrog, which also checks a GGUF chat template for template injection)."""
    if not paths:
        return {}
    data = "&".join(f"paths={urllib.request.quote(p)}" for p in paths) + "&expand=true"
    req = urllib.request.Request(f"{API}/models/{repo}/paths-info/main", data.encode(), UA)
    out = {}
    for item in json.loads(urllib.request.urlopen(req, timeout=30).read()):
        status = item.get("securityFileStatus") or {}
        out[item["path"]] = {k.removesuffix("Scan"): v.get("status") for k, v in status.items() if isinstance(v, dict)}
    return out


def uploader(author):
    for kind in ("users", "organizations"):
        try:
            return get_json(f"{API}/{kind}/{author}/overview")
        except Exception:
            continue
    return {}


def assess(repo):
    reasons, level = [], "low"

    def flag(lvl, why):
        nonlocal level
        order = ["low", "medium", "high", "wontload"]
        level = max(level, lvl, key=order.index)
        reasons.append(why)

    info = get_json(f"{API}/models/{repo}?securityStatus=true")
    files = [s["rfilename"] for s in info.get("siblings", [])]
    try:
        sizes = {f["path"]: f.get("size", 0) for f in get_json(f"{API}/models/{repo}/tree/main?recursive=true")}
    except Exception:
        sizes = {}
    scan = info.get("securityRepoStatus") or {}
    sample = sample_file(files, sizes)
    mmproj = [f for f in files if "mmproj" in f.lower() and f.lower().endswith(".gguf")][:2]
    flagged = {str(f.get("path")): f.get("level") for f in scan.get("filesWithIssues") or []}
    try:
        scans = file_scans(repo, sorted(set(flagged) | set(filter(None, [sample, *mmproj]))))
    except Exception as err:
        scans = {}
        flag("medium", t("risk.scan_failed", err=err)[:160])
    for path, verdicts in scans.items():
        if bad := [name for name, status in verdicts.items() if status in ("unsafe", "malicious", "suspicious", "caution")]:
            flag("high", t("risk.scanner_warns", path=path, names=", ".join(bad)))
        elif path in flagged:
            flag("medium", t("risk.hf_flagged", path=path, level=flagged[path]))
    if not scan.get("scansDone") and not flagged:
        flag("medium", t("risk.scan_pending"))
    if bad := [f for f in files if PICKLE.search(f)]:
        flag("medium", t("risk.pickle", n=len(bad), f=bad[0]))
    if code := [f for f in files if CODE.search(f)]:
        flag("medium", t("risk.code", n=len(code), f=code[0]))

    downloads = info.get("downloads", 0)
    created = info.get("createdAt")
    age_days = (datetime.now(timezone.utc) - datetime.fromisoformat(created.replace("Z", "+00:00"))).days if created else None
    who = uploader(info.get("author") or repo.split("/")[0])
    followers, models = who.get("numFollowers", 0), who.get("numModels", 0)
    if followers < 10 and downloads < 500:
        flag("medium", t("risk.unknown_uploader", followers=followers, models=models, downloads=downloads))
    if age_days is not None and age_days < 3 and downloads < 200:
        flag("medium", t("risk.fresh", days=age_days))
    card = info.get("cardData") or {}
    if not card.get("base_model"):
        reasons.append(t("risk.no_base"))
    if info.get("gated"):
        reasons.append(t("risk.gated"))

    gguf = None
    if not sample:
        flag("wontload", t("risk.no_gguf"))
    else:
        try:
            gguf = gguf_header(f"https://huggingface.co/{repo}/resolve/main/{sample}")
            if odd := [t for t in gguf["types"] if t > MAX_GGML_TYPE]:
                flag("wontload", t("risk.odd_type", types=odd))
        except Short:
            flag("medium", t("risk.big_header", mb=HEADER_LIMIT // 2**20))
        except urllib.error.HTTPError as err:
            flag("medium", t("risk.header_http", code=err.code, gated=", gated" if info.get("gated") else ""))
        except Exception as err:
            flag("high", t("risk.header_bad", file=sample, err=err))

    return {"level": level, "reasons": reasons, "file": sample, "gguf": gguf, "downloads": downloads,
            "uploader_followers": followers, "checked": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def safe_assess(repo):
    try:
        return assess(repo)
    except Exception as err:
        return {"level": "medium", "reasons": [t("risk.failed", err=err)[:200]],
                "checked": datetime.now(timezone.utc).isoformat(timespec="seconds")}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    print(json.dumps(safe_assess(sys.argv[1]), ensure_ascii=False, indent=1))
