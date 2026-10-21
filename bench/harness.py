"""Shared pieces of bench/bench.py: the OpenAI-compatible client and the tool-calling loop.
A cases file (bench/cases.py or one passed with --cases) imports fn and run_tools from here."""
import json
import time
import urllib.error
import urllib.request

MAX_TOKENS = 6000


class Client:
    def __init__(self, base_url, key, temperature=None, extra=None):
        self.url, self.key = base_url.rstrip("/") + "/v1/chat/completions", key
        self.temperature = temperature  # None: the server's default sampling
        self.extra = extra or {}  # sent with every request, e.g. reasoning_effort / chat_template_kwargs

    def body(self, messages, max_tokens, **extra):
        body = {"messages": messages, "max_tokens": max_tokens, **self.extra, **extra}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        return body

    def chat(self, messages, tools=None, max_tokens=MAX_TOKENS):
        body = self.body(messages, max_tokens)
        if tools:
            body["tools"] = tools
        req = urllib.request.Request(self.url, json.dumps(body).encode(), {
            "Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        t0 = time.time()
        try:
            out = json.loads(urllib.request.urlopen(req, timeout=900).read())
        except urllib.error.HTTPError as e:  # e.g. a prompt longer than the server's context: the case fails, the run goes on
            out = {"choices": [{"message": {"content": ""}, "finish_reason": "error"}],
                   "error": f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}"}
        out["_wall"] = time.time() - t0
        out["_truncated"] = out["choices"][0].get("finish_reason") == "length"
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
                ev = json.loads(line[5:])
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

