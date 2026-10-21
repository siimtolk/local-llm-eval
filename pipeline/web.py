"""HTTP helpers for discover.py: polite fetching (Reddit's rate limit), text cleanup, dates, and
the top comments of a Reddit thread."""
import html
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

SUMMARY_CHARS = 220
COMMENTS_PER_ITEM = 4
COMMENT_CHARS = 300
ATOM = "{http://www.w3.org/2005/Atom}"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) local-llm-eval/1.0"}


_reddit_lock = threading.Lock()
_reddit_wait_until = 0.0


def reddit_get(url):
    """Reddit allows roughly one anonymous request a minute; wait out its rate-limit window."""
    global _reddit_wait_until
    with _reddit_lock:
        for attempt in range(2):
            time.sleep(max(0.0, _reddit_wait_until - time.time()))
            try:
                resp = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=25)
                headers, body = resp.headers, resp.read()
            except urllib.error.HTTPError as exc:
                if exc.code != 429 or attempt:
                    raise
                headers, body = exc.headers, None
            if body is None or float(headers.get("x-ratelimit-remaining") or 1) < 1:
                _reddit_wait_until = time.time() + float(headers.get("x-ratelimit-reset") or 60) + 2
            if body is not None:
                return body


def get(url):
    if "reddit.com/" in url:
        return reddit_get(url)
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=25).read()


def clean(text, limit=SUMMARY_CHARS):
    text = html.unescape(re.sub(r"<[^>]+>", " ", text or ""))
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text.startswith("submitted by"):  # reddit boilerplate, no content
        return ""
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "..."


def parse_date(value):
    if not value:
        return None
    try:
        when = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        try:
            when = parsedate_to_datetime(value.strip())
        except (TypeError, ValueError):
            return None
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


def community(item):
    """Attach the top comments of the item's Reddit thread."""
    try:
        if "reddit.com/r/" not in item["link"]:
            return
        # Atom feed of the thread: the first entry is the post itself.
        root = ET.fromstring(get(item["link"].rstrip("/") + "/.rss?sort=top"))
        texts = [e.findtext(f"{ATOM}content") for e in list(root.iter(f"{ATOM}entry"))[1:]]
        comments = [c for c in (clean(t, COMMENT_CHARS) for t in texts if t) if len(c) > 40]
        if comments:
            item["community"] = {"where": "Reddit", "count": None, "comments": comments[:COMMENTS_PER_ITEM]}
    except Exception as exc:
        print(f"web: no comments for {item['link']}: {type(exc).__name__}", file=sys.stderr)
