#!/usr/bin/env python3
"""A judge command for a local model: stdin as one user message to an OpenAI-compatible server,
the reply on stdout. Use it as [judge] command when the model you run in production should write
the brief instead of a hosted one, e.g.

    command = "python3 tools/openai_chat.py --url http://127.0.0.1:8080 --api-key-file ~/key.txt"

    tools/openai_chat.py --url URL [--api-key-file FILE] [--model NAME] [--max-tokens N] [--timeout S]

Thinking stays as the server sets it; only the final answer is printed. Exit 1 with the reason on
stderr when the server fails or the reply is empty (judge.py then sends nothing). Stdlib only.
"""
import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="server base URL, without /v1")
    ap.add_argument("--api-key-file")
    ap.add_argument("--model", default="", help="model name; the server's default when empty")
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()
    body = {"messages": [{"role": "user", "content": sys.stdin.read()}], "max_tokens": args.max_tokens}
    if args.model:
        body["model"] = args.model
    headers = {"Content-Type": "application/json"}
    if args.api_key_file:
        headers["Authorization"] = f"Bearer {Path(args.api_key_file).expanduser().read_text().strip()}"
    req = urllib.request.Request(f"{args.url.rstrip('/')}/v1/chat/completions", json.dumps(body).encode(), headers)
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            choice = json.loads(resp.read())["choices"][0]
    except Exception as exc:
        sys.exit(f"openai_chat: {type(exc).__name__}: {exc}")
    reply = re.sub(r"<think>.*?</think>", "", choice["message"].get("content") or "", flags=re.S).strip()
    if not reply:
        sys.exit(f"openai_chat: empty reply (finish_reason {choice.get('finish_reason')})")
    print(reply)


if __name__ == "__main__":
    main()
