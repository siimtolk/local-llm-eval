"""Shared pieces of bench/bench.py: the OpenAI-compatible client and the tool-calling loop.
A cases file (bench/cases.py or one passed with --cases) imports fn and run_tools from here."""
import hashlib
import json
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

MAX_TOKENS = 6000
CHECKPOINT_DAYS = 14  # an older checkpoint is from another run: ignored and removed


class Checkpoint:
    """The finished requests of one section, so a section that was cut off resumes where it stopped.

    Every answered chat request is appended as one JSON line, keyed by the exact request body and how many
    times that body was sent before in the section (reps send the same body). A rerun of the section with
    the same label gets the stored answer for each request it already finished, with its original wall
    time, and asks the server only for the rest; a request whose body differs (a random prompt, a different
    earlier answer) is a miss and goes to the server. Failed requests are not stored. bench.py deletes the
    file once the section is saved in the result file."""

    def __init__(self, path):
        self.path, self.seen, self.cache, self.hits = Path(path), Counter(), {}, 0
        if self.path.exists() and time.time() - self.path.stat().st_mtime > CHECKPOINT_DAYS * 86400:
            self.path.unlink()
        if self.path.exists():
            for line in self.path.read_text(errors="replace").splitlines():
                try:
                    rec = json.loads(line)
                    self.cache.setdefault(rec["key"], []).append(rec["out"])
                except (ValueError, KeyError, TypeError):  # the last line of a killed run may be cut off
                    continue
        self.stored = sum(map(len, self.cache.values()))

    @staticmethod
    def key(body):
        return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def get(self, body):
        """(stored answer or None, key) for this request; counts the request either way."""
        k = self.key(body)
        n, self.seen[k] = self.seen[k], self.seen[k] + 1
        hit = self.cache.get(k, [])
        if n < len(hit):
            self.hits += 1
            return hit[n], k
        return None, k

    def put(self, k, out):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps({"key": k, "out": out}, ensure_ascii=False) + "\n")

    def discard(self):
        self.path.unlink(missing_ok=True)


class Client:
    def __init__(self, base_url, key, temperature=None, extra=None):
        self.url, self.key = base_url.rstrip("/") + "/v1/chat/completions", key
        self.temperature = temperature  # None: the server's default sampling
        self.extra = extra or {}  # sent with every request, e.g. reasoning_effort / chat_template_kwargs
        self.checkpoint = None  # a Checkpoint while bench.py runs a section: finished requests are kept and reused

    def body(self, messages, max_tokens, **extra):
        body = {"messages": messages, "max_tokens": max_tokens, **self.extra, **extra}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        return body

    def chat(self, messages, tools=None, max_tokens=MAX_TOKENS):
        body = self.body(messages, max_tokens)
        if tools:
            body["tools"] = tools
        cp = self.checkpoint
        if cp:
            stored, k = cp.get(body)
            if stored is not None:
                return stored
        req = urllib.request.Request(self.url, json.dumps(body).encode(), {
            "Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        t0 = time.time()
        try:
            out = json.loads(urllib.request.urlopen(req, timeout=900).read())
        except urllib.error.HTTPError as e:  # e.g. a prompt longer than the server's context: the case fails, the run goes on
            out = {"choices": [{"message": {"content": ""}, "finish_reason": "error"}],
                   "error": f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}"}
        except (TimeoutError, json.JSONDecodeError) as e:  # one slow or garbled reply: the case fails, the run goes on
            out = {"choices": [{"message": {"content": ""}, "finish_reason": "error"}],
                   "error": f"{type(e).__name__}: {str(e)[:300]}"}
        out["_wall"] = time.time() - t0
        out["_truncated"] = out["choices"][0].get("finish_reason") == "length"
        if cp and "error" not in out:
            cp.put(k, out)
        return out

    def stream(self, messages, max_tokens):
        """Streamed request timed on this side, so speed works with any OpenAI-compatible server:
        ttft = until the first generated token (reasoning or content), gen = first to last token.
        Token counts come from the standard usage chunk (stream_options.include_usage), else the
        number of token chunks. llama-server's own timings are kept when the server sends them."""
        body = self.body(messages, max_tokens, stream=True, stream_options={"include_usage": True})
        req = urllib.request.Request(self.url, json.dumps(body).encode(), {
            "Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        t0 = time.perf_counter()
        first = last = None
        chunks, usage, timings, text = 0, {}, {}, []
        with urllib.request.urlopen(req, timeout=900) as resp:
            for line in resp:
                line = line.strip()
                if not line.startswith(b"data:") or line == b"data: [DONE]":
                    continue
                try:
                    ev = json.loads(line[5:])
                except ValueError:  # a malformed event: skip it
                    continue
                usage = ev.get("usage") or usage
                timings = ev.get("timings") or timings
                delta = (ev.get("choices") or [{}])[0].get("delta") or {}
                text.append(delta.get("content") or "")
                if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning") or delta.get("tool_calls"):
                    last = time.perf_counter()
                    first = first or last
                    chunks += 1
        n = usage.get("completion_tokens") or chunks
        gen = (last - first) if first and last and last > first else None
        return {"wall": time.perf_counter() - t0, "ttft": first - t0 if first else None,
                "n": n, "prompt_n": usage.get("prompt_tokens"),
                "tok_s": (n - 1) / gen if gen and n > 1 else None, "timings": timings, "text": "".join(text)}


def fn(name, desc, props, required=None):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": required or list(props)}}}


def run_tools(client, messages, tools, handler, max_turns=12):
    """Agent loop: feed tool results back until the model answers in plain text."""
    messages, calls, wall, tokens = list(messages), [], 0.0, 0
    for _ in range(max_turns):
        r = client.chat(messages, tools)
        wall += r["_wall"]
        tokens += r.get("usage", {}).get("completion_tokens", 0)
        msg = r["choices"][0]["message"]
        tcs = msg.get("tool_calls") or []
        if not tcs:
            return calls, msg.get("content") or "", wall, tokens
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tcs})
        for tc in tcs:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
                result = handler(name, args)
            except Exception as e:
                args, result = {"_raw": tc["function"].get("arguments")}, {"error": f"bad call: {e}"}
            calls.append((name, args))
            messages.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": json.dumps(result)})
    return calls, "(no final answer within turn limit)", wall, tokens

