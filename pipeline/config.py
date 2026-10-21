"""This machine's settings for every stage, from local.toml (gitignored) in the repo root.

local.example.toml documents every key; copy it to local.toml and fill in your values. Without a
local.toml the example is used (enough for --dry-run and --help), with a note on stderr.
LLE_CONFIG=<file> picks another file. Paths may start with ~; relative paths are relative to
the repo root, and serving-repo commands relative to paths.serving.
"""
import os
import shlex
import sys
import tomllib
from datetime import datetime, time, timedelta
from functools import cache
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent


@cache
def load():
    path = Path(os.environ.get("LLE_CONFIG") or ROOT / "local.toml")
    if not path.exists() and "LLE_CONFIG" not in os.environ:
        print(f"config: no {path.name}, using local.example.toml", file=sys.stderr)
        path = ROOT / "local.example.toml"
    with open(path, "rb") as f:
        return tomllib.load(f)


def get(section, key, default=None):
    return load().get(section, {}).get(key, default)


def path(key, base=None):
    """paths.<key> as an absolute Path."""
    p = Path(os.path.expanduser(get("paths", key)))
    return p if p.is_absolute() else (base or ROOT) / p


def serving():
    return path("serving")


def models_dir():
    return path("models", serving())


def serving_cmd(section, key):
    """A command from config (string, shell-split) with a relative program resolved in the serving repo."""
    value = get(section, key)
    argv = shlex.split(value) if isinstance(value, str) else list(value)
    if argv and "/" in argv[0] and not os.path.isabs(os.path.expanduser(argv[0])):
        argv[0] = str(serving() / argv[0])
    return [os.path.expanduser(a) for a in argv]


def serve(*args):
    """argv for the serving backend interface: bin/serve {start|stop|health} <model> ..."""
    return serving_cmd("serving", "command") + [str(a) for a in args]


def tz():
    name = get("schedule", "timezone", "")
    return ZoneInfo(name) if name else None


def now():
    """Local wall-clock time as a naive datetime (the schedule's timezone, else the system's)."""
    zone = tz()
    return datetime.now(zone).replace(tzinfo=None) if zone else datetime.now()


def hhmm(key):
    h, m = get("schedule", key).split(":")
    return time(int(h), int(m))


def window_hours():
    start, end = hhmm("window_start"), hhmm("window_end")
    minutes = (end.hour * 60 + end.minute) - (start.hour * 60 + start.minute)
    return (minutes % (24 * 60)) / 60


def minutes(key):
    return timedelta(minutes=get("schedule", key))
