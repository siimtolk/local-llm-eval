#!/usr/bin/env python3
"""Stage 5, report: last night's test results and the new candidates (no LLM).

Runs in the morning (schedule.report_time). For each candidate in the store whose result is not
reported yet: the summary the nightly test wrote with "candidates.py set". A candidate still
"testing" in the morning did not finish; it is marked failed and reported so. Prints nothing
when there is no news.
First it stores the candidates the latest judge brief recommended (candidates.py index), each with
a risk check (risk.py), and lists the new ones with what the check found. Those rated low or medium
go into a multi-select poll ([notify] poll_url): only the ones ticked are downloaded and tested
(candidates.py vote). Without polls, pick one with `candidates.py set <repo> pending`.
High risk or "will not load" ones are not offered: they are marked skipped and reported with the reason.

Usage: report.py [--print]   (--print: write the report to stdout instead of [notify])
"""
import sys
from datetime import datetime, timedelta, timezone

import config
import notify
from candidates import expire, index, now, polls, store
from short_links import link_lines
from texts import t

DONE = ("tested", "rejected", "skipped", "failed")
RISKS = ("low", "medium", "high", "wontload")
OPTION_CHARS = 100
STUCK_AFTER = timedelta(hours=1)  # tests are cut at the deadline minus the restore margin; this runs after it


def risk_label(level):
    return t(f"risk.{level}" if level in RISKS else "risk.none")


def offer():
    """Text about the new proposals, and a poll for the ones that may be downloaded."""
    blocks, options = [], {}
    with store() as data:
        for repo, c in data["candidates"].items():
            if c["status"] != "proposed" or c.get("offered"):
                continue
            c["offered"] = now()
            risk = c.get("risk") or {}
            level = risk.get("level")
            why = "; ".join(risk.get("reasons", [])) or t("risk.clean")
            blocks.append((c, f"*{c['name']}* · {risk_label(level)}\n{c.get('size', '')}\n_{why}_"))
            if level in ("high", "wontload"):
                c.update(status="skipped", updated=now(), reported=True,
                         summary=t("risk.wontload_prefix" if level == "wontload" else "risk.high_prefix") + why)
            else:
                text = f"{c['name']} · {risk_label(level).split(' ')[0] if level in RISKS else '❔'}"
                options[text if len(text) <= OPTION_CHARS else text[:OPTION_CHARS - 1] + "…"] = repo
    if not blocks:
        return ""
    tail = ""
    if options and notify.can_poll():
        none = t("poll.none")
        try:
            message_id = notify.send_poll(t("poll.question", date=f"{config.now():%d.%m}"), [*options, none], len(options) + 1)
        except Exception as err:  # offered again tomorrow
            with store() as data:
                for repo in options.values():
                    data["candidates"][repo].pop("offered", None)
            return t("poll.failed", err=err)
        with polls() as reg:
            reg["polls"][message_id] = {"date": now(), "options": options, "none": none,
                                        "selected": None, "voted_at": None}
            reg["polls"] = dict(list(reg["polls"].items())[-50:])
        tail = t("poll.tail")
    elif options:
        tail = "\n\n" + "\n".join(f"candidates.py set {repo} pending" for repo in options.values())
    links = link_lines([c["link"] for c, _ in blocks])
    return t("report.new") + "\n\n" + "\n\n".join(f"{text}\n{links[c['link']]}" for c, text in blocks) + tail


def build():
    index()  # the judge brief ran earlier this morning: store what it recommended
    expire()
    proposals = offer()
    lines = []
    with store() as data:
        for repo, c in data["candidates"].items():
            if c["status"] == "testing" and datetime.now(timezone.utc) - datetime.fromisoformat(c["updated"]) > STUCK_AFTER:
                c.update(status="failed", updated=now(), reported=False, summary=c["summary"] or t("report.stuck"))
            if c["status"] in DONE and not c["reported"]:
                lines.append((c, f"*{c['name']}* · {t('done.' + c['status'])}\n{c['summary'] or t('report.no_summary')}"))
                c["reported"] = True
        pending = sum(c["status"] == "pending" for c in data["candidates"].values())
    out = []
    if lines:
        links = link_lines([c["link"] for c, _ in lines])
        lines = [f"{text}\n{links[c['link']]}" for c, text in lines]
        tail = t("report.pending", n=pending, word=t("report.one" if pending == 1 else "report.many")) if pending else ""
        out.append(t("report.night") + "\n\n" + "\n\n".join(lines) + tail)
    if proposals:
        out.append(proposals)
    return "\n\n".join(out)


def main():
    text = build()
    if text:
        print(text) if "--print" in sys.argv else notify.send(text)


if __name__ == "__main__":
    main()
