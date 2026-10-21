"""Active parameters of a mixture-of-experts model, for the fit checks in discover and fetch.

The repo name says it only sometimes ("...-35B-A3B"). Otherwise the model card usually does
("6B activated per token", "~3B active parameters"), on the repo itself or on its base model.
Returns billions, or None when nothing says so (then the model is treated as dense).
"""
import json
import re
import urllib.request
from functools import lru_cache
from urllib.parse import quote

API = "https://huggingface.co/api/models/{}"
README = "https://huggingface.co/{}/raw/main/README.md"
UA = {"User-Agent": "local-llm-eval"}
NAME = re.compile(r"(?<![\w.])A(\d+(?:\.\d+)?)B\b", re.I)
CARD = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*B\s*(?:parameters?\s+)?(?:active|activated)\b"
                  r"|\b(?:active|activated)\s+param(?:eter)?s?\s*[:=(]?\s*~?\s*(\d+(?:\.\d+)?)\s*B\b", re.I)


def _get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=25) as r:
        return r.read()


def from_text(text):
    m = CARD.search(text or "")
    return float(m.group(1) or m.group(2)) if m else None


@lru_cache(maxsize=256)
def active_b(repo, depth=2):
    m = NAME.search(repo.split("/")[-1])
    if m:
        return float(m.group(1))
    try:
        found = from_text(_get(README.format(quote(repo, safe="/"))).decode("utf8", "ignore"))
    except Exception:
        found = None
    if found or depth <= 0:
        return found
    try:
        base = (json.loads(_get(API.format(quote(repo, safe="/")))).get("cardData") or {}).get("base_model")
    except Exception:
        return None
    base = base[0] if isinstance(base, list) and base else base
    return active_b(base, depth - 1) if isinstance(base, str) and "/" in base and base != repo else None
