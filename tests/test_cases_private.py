"""Offline tests for tools/build_replay.py and bench/cases_private.py, on a small synthetic state.db (no real traffic).

Run from the repo root: python3 -m unittest tests.test_cases_private -v
"""
import atexit
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(1, str(ROOT / "pipeline"))
TMP = Path(tempfile.mkdtemp(prefix="lle-test-private-"))
atexit.register(shutil.rmtree, TMP, ignore_errors=True)
os.environ["BENCH_PRIVATE_DIR"] = str(TMP / "private")
import cases_private as cp  # noqa: E402

TOOLS = [{"type": "function", "function": {"name": n, "description": "d", "parameters": {"type": "object", "properties": {}}}} for n in ("terminal", "read_file", "patch")]


def make_db(path):
    db = sqlite3.connect(path)
    db.executescript("""
    create table sessions (id text primary key, source text, system_prompt_hash text, api_call_count int, title text, started_at real);
    create table messages (id integer primary key autoincrement, session_id text, role text, content text, tool_call_id text, tool_calls text,
                           active int default 1, display_order int, finish_reason text);
    create table system_prompts (hash text primary key, prompt text);""")
    db.execute("insert into system_prompts values ('h1', 'You are an agent. Key sk-abcdefghijklmnopqrstuvwx.')")
    call = lambda i, name, args: json.dumps([{"id": i, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}])
    db.execute("insert into sessions values ('s1','kanban','h1',4,'Ticket 1',1)")
    rows = [("user", "Fix the bug in app.py", None, None, None),
            ("assistant", "", None, call("c1", "read_file", {"path": "/w/app.py"}), "tool_calls"),
            ("tool", "def f(): return 1", "c1", None, None),
            ("assistant", "", None, call("c2", "patch", {"path": "/w/app.py", "old": "1", "new": "2"}), "tool_calls"),
            ("tool", "ok", "c2", None, None),
            ("assistant", "Fixed.", None, None, "stop")]
    for r in rows:
        db.execute("insert into messages (session_id, role, content, tool_call_id, tool_calls, finish_reason) values ('s1',?,?,?,?,?)", r)
    db.execute("insert into sessions values ('c1','cron','h1',1,'tech-brief · Oct 26 06:22',2)")
    prompt = ("Items:\n- title: A\n  link: https://tinyurl.com/aaa111 (Hugging Face)\nYour entire reply is under 400 characters. At most 3 blocks. "
              "If nothing matters reply [SILENT].\n" + "x" * 2000)
    db.execute("insert into messages (session_id, role, content) values ('c1','user',?)", (prompt,))
    db.execute("insert into messages (session_id, role, content, finish_reason) values ('c1','assistant','**Tech brief**\n\n**A**\nNew model.\nhttps://tinyurl.com/aaa111 (Hugging Face)','stop')")
    db.commit()
    db.close()


class Build(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        make_db(TMP / "state.db")
        (TMP / "tools.json").write_text(json.dumps(TOOLS))
        r = subprocess.run([sys.executable, str(ROOT / "tools" / "build_replay.py"), "--state-db", str(TMP / "state.db"), "--out", str(TMP / "private"),
                            "--tools-json", str(TMP / "tools.json"), "--replay", "10", "--cron", "5"], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr

    def test_fixtures_written_and_secrets_removed(self):
        replay, cron = cp.load_jsonl("replay.jsonl"), cp.load_jsonl("cron.jsonl")
        self.assertEqual(sorted(fx["ref"].get("name", "final") for fx in replay), ["final", "patch", "read_file"])
        self.assertEqual([fx["job"] for fx in cron], ["tech-brief"])
        self.assertEqual(len(cp.load_tools()), 3)
        self.assertNotIn("sk-abcdef", cp.load_system("h1"))
        # a replay prefix ends before the decision: the patch point contains the read, not the patch
        patch = next(fx for fx in replay if fx["ref"].get("name") == "patch")
        self.assertEqual([m["role"] for m in patch["messages"]], ["user", "assistant", "tool"])

    def test_replay_grading(self):
        ref = {"name": "terminal", "args": {"command": "cd /w && python3 -m unittest -q"}}
        call = lambda n, a: {"content": "", "tool_calls": [{"function": {"name": n, "arguments": json.dumps(a)}}]}
        self.assertTrue(cp.grade_replay(call("terminal", {"command": "python3 -m unittest"}), ref))
        self.assertFalse(cp.grade_replay(call("terminal", {"command": "rm -rf /var/tmp/*"}), ref))
        self.assertFalse(cp.grade_replay(call("web_search", {"query": "x"}), ref))
        self.assertFalse(cp.grade_replay({"content": "I will not.", "tool_calls": []}, ref))
        patch = {"name": "patch", "args": {"path": "/w/app.py"}}
        self.assertTrue(cp.grade_replay(call("patch", {"path": "app.py"}), patch))
        self.assertFalse(cp.grade_replay(call("patch", {"path": "other.py"}), patch))
        final = {"final": True}
        self.assertTrue(cp.grade_replay({"content": "Done."}, final))
        self.assertTrue(cp.grade_replay(call("read_file", {"path": "x"}), final))
        self.assertFalse(cp.grade_replay(call("patch", {"path": "x"}), final))
        self.assertFalse(cp.grade_replay(call("terminal", {"command": "git push origin main"}), final))

    def test_cron_contract(self):
        fx = cp.load_jsonl("cron.jsonl")[0]
        good = fx["ref_output"]
        self.assertEqual(cp.contract_failures(fx["prompt"], good, good), [])
        self.assertEqual(cp.contract_failures(fx["prompt"], good, ""), ["empty"])
        self.assertIn("invented_link", cp.contract_failures(fx["prompt"], good, "**Tech brief**\n\n**A**\nNew.\nhttps://tinyurl.com/zzz999"))
        self.assertIn("invented_number", cp.contract_failures(fx["prompt"], good, "**Tech brief**\n\n**A**\n12345 users.\nhttps://tinyurl.com/aaa111 (Hugging Face)"))
        self.assertIn("length", cp.contract_failures(fx["prompt"], good, "**Tech brief**\n\n" + "word " * 120))
        self.assertIn("silent", cp.contract_failures(fx["prompt"], good, "[SILENT]"))
        self.assertIn("language", cp.contract_failures(fx["prompt"], good, "**Tech brief**\n\n**A**\nUus mudel on olemas ja see on hea.\nhttps://tinyurl.com/aaa111"))
        # a link built from an id the input gives is the input's own
        self.assertTrue(cp.link_in_input("https://youtu.be/AbCdEf12345", "item id: AbCdEf12345\nhttps://youtu.be/<id>"))

    def test_sections_run_and_store_no_text(self):
        class Client:
            def chat(self, messages, tools=None, max_tokens=None):
                msg = {"content": "**Tech brief**\n\n**A**\nNew model.\nhttps://tinyurl.com/aaa111 (Hugging Face)"}
                return {"choices": [{"message": msg, "finish_reason": "stop"}], "_wall": 0.1, "_truncated": False, "usage": {"completion_tokens": 5, "prompt_tokens": 9}}
        rows = cp.cron_section(Client(), 1)
        self.assertEqual([r["ok"] for r in rows], [True])
        import bench
        stripped = {k for r in cp.replay_section(Client(), 1) + rows for k in r} & set(bench.QUOTING)
        self.assertEqual(stripped, set(), "private rows must not carry quoted text")
        self.assertEqual(cp.PRIVATE, set(cp.SECTIONS))

    def test_ctx_sends_the_real_prefix_and_merges_tools(self):
        seen = []

        class Client:
            def chat(self, messages, tools=None, max_tokens=None):
                seen.append((messages[0], {t["function"]["name"] for t in tools}))
                return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "_wall": 0.0, "_truncated": False, "usage": {"completion_tokens": 1}}
        rows = cp.ctx_section(Client(), 1)
        self.assertEqual(len(rows), len(cp.CTX_TOOL_CASES) + len(cp.CTX_SCENARIOS))
        self.assertTrue(all(r["case"].startswith("ctx_") for r in rows))
        first, names = seen[0]
        self.assertEqual(first["role"], "system")
        self.assertIn("You are an agent", first["content"])
        self.assertTrue({"terminal", "set_light"} <= names)
        # a scenario that brings its own read_file replaces the agent's read_file instead of duplicating it
        merged = cp.merged_tools(TOOLS, [t for t in cp.base.FS_TOOLS])
        self.assertEqual(len([t for t in merged if t["function"]["name"] == "read_file"]), 1)


class Skipped(unittest.TestCase):
    def test_without_a_private_dir_every_section_is_skipped(self):
        old = os.environ.pop("BENCH_PRIVATE_DIR")
        try:
            os.environ["LLE_CONFIG"] = str(ROOT / "local.example.toml")
            import config
            config.load.cache_clear()
            for name, section in cp.SECTIONS.items():
                self.assertEqual(section(None, 1), [], name)
        finally:
            os.environ["BENCH_PRIVATE_DIR"] = old
            os.environ.pop("LLE_CONFIG", None)
            config.load.cache_clear()


if __name__ == "__main__":
    unittest.main()
