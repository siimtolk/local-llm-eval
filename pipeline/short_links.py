#!/usr/bin/env python3
"""Link lines for the reports: "<link> (Hugging Face)".

With [notify] short_links = true, each URL is shortened once (is.gd, else TinyURL; chat apps that
cannot put a link behind a word show these instead of a 100-character URL) and cached in
paths.short_links, so the same model keeps the same short link. When neither service answers,
the original URL is used: a report is never left without its links. The services allow about one
request a second. With short_links = false (the default) links stay as they are.

Usage: short_links.py <url> ...   (prints the link lines)
"""
import fcntl
import json
import sys
import time
import urllib.parse
import urllib.request

import config

# (name, API, prefix of a good answer), tried in order
SERVICES = [("is.gd", "https://is.gd/create.php?format=simple&url={}", "https://is.gd/"),
            ("TinyURL", "https://tinyurl.com/api-create.php?url={}", "https://tinyurl.com/")]
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) local-llm-eval/1.0"}
MIN_LENGTH = 40  # shorter URLs are left as they are
KEEP = 2000  # cached links
_last_call = 0.0

def _create(url, down):
    """A short link from the first service that works; services that failed are added to down."""
    global _last_call
    for name, api, prefix in SERVICES:
        if name in down:
            continue
        time.sleep(max(0.0, _last_call + 1.1 - time.time()))
        _last_call = time.time()
        try:
            short = urllib.request.urlopen(urllib.request.Request(api.format(urllib.parse.quote(url, safe="")),
                                                                  headers=UA), timeout=15).read().decode().strip()
            if short.startswith(prefix):
                return short
            raise ValueError(short[:100])
        except Exception as exc:
            print(f"short_links: {name} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            down.add(name)
    raise OSError("no shortening service answered")


def shorten(urls):
    """{url: short url} for the given URLs; the original where shortening failed or is turned off."""
    if not config.get("notify", "short_links", False):
        return {url: url for url in urls}
    CACHE = config.path("short_links")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
        out, down = {}, set()
        for url in dict.fromkeys(urls):
            if len(url) < MIN_LENGTH or not url.startswith("http"):
                out[url] = url
            elif url in cache:
                out[url] = cache.pop(url)
                cache[url] = out[url]  # re-inserted: most recently used last
            elif len(down) == len(SERVICES):  # every service is down: do not wait on each link
                out[url] = url
            else:
                try:
                    out[url] = cache[url] = _create(url, down)
                except OSError:
                    out[url] = url
        cache = dict(list(cache.items())[-KEEP:])
        CACHE.write_text(json.dumps(cache, indent=0))
    return out


def short(url):
    return shorten([url])[url]


def expand(url):
    """The original URL of a short link made here (from the cache); other URLs unchanged."""
    if not url.startswith(tuple(prefix for _, _, prefix in SERVICES)):
        return url
    try:
        cache = json.loads(config.path("short_links").read_text())
    except (OSError, ValueError):
        return url
    return next((long for long, s in cache.items() if s == url), url)


OUTLETS = {"huggingface.co": "Hugging Face", "reddit.com": "Reddit", "github.com": "GitHub"}


def outlet(url):
    """Concise site name for a URL: a known name, else the site's own name (www.example.co.uk -> example)."""
    host = urllib.parse.urlsplit(url).hostname or ""
    for site, name in OUTLETS.items():
        if host == site or host.endswith("." + site):
            return name
    parts = [p for p in host.split(".") if p not in ("www", "m", "amp")]
    return (parts[-3] if len(parts) > 2 and parts[-2] in ("co", "com", "org", "ac", "gov") else
            parts[-2] if len(parts) > 1 else host).capitalize()


def link_lines(urls):
    """{url: "<link> (<site>)"} for the given URLs: the finished link line of a report."""
    shorts = shorten(urls)
    return {url: f"{shorts[url]} ({outlet(url)})" for url in shorts}


if __name__ == "__main__":
    for line in link_lines(sys.argv[1:]).values():
        print(line)
