"""Where the loop's messages go ([notify] in local.toml): an HTTP chat bridge, a shell command that
gets the text on stdin, or stdout. The chat ids and the bridge address live only in local.toml.

  http_url: POST {"chatId": chat_id, "message": text}
  poll_url: POST {"chatId", "question", "options", "selectableCount"} -> {"messageId": id}
  command:  run with the text on stdin
"""
import json
import os
import subprocess
import urllib.request
from pathlib import Path

import config


def post_json(url, payload, timeout=60):
    headers = {"Content-Type": "application/json"}
    tf = config.get("notify", "token_file")
    if tf:
        try:
            headers["Authorization"] = "Bearer " + Path(os.path.expanduser(tf)).read_text().strip()
        except OSError:
            raise OSError("notify token_file unreadable") from None
    req = urllib.request.Request(url, json.dumps(payload).encode(), headers)
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read() or b"{}")


def send(text):
    url, command = config.get("notify", "http_url"), config.get("notify", "command")
    if url:
        post_json(url, {"chatId": config.get("notify", "chat_id"), "message": text})
    elif command:
        subprocess.run(command, shell=True, input=text, text=True, check=True)
    else:
        print(text)


def can_poll():
    return bool(config.get("notify", "poll_url"))


def send_poll(question, options, selectable):
    """The poll's message id; raises when polls are not configured or the bridge fails."""
    url = config.get("notify", "poll_url")
    if not url:
        raise OSError("no [notify] poll_url")
    sent = post_json(url, {"chatId": config.get("notify", "chat_id"), "question": question,
                           "options": options, "selectableCount": selectable})
    return sent["messageId"]
