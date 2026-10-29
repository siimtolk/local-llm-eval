"""Offline tests for pipeline/experiments.py and the runner's unified queue (stdlib only, no network).

Every test runs on a settings dict made from local.example.toml with the stores, results and wiki
inbox in a temp dir (config.load is patched, so the other test modules' LLE_CONFIG does not matter).
Run from the repo root: python3 -m unittest discover -s tests -v
"""
import atexit
import io
import json
import shutil
import sys
import tempfile
import tomllib
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
TMP = Path(tempfile.mkdtemp(prefix="lle-test-exp-"))
atexit.register(shutil.rmtree, TMP, ignore_errors=True)
sys.path.insert(0, str(ROOT / "pipeline"))

import candidates  # noqa: E402
import config  # noqa: E402
import experiments  # noqa: E402
import runner  # noqa: E402

SPEC = {"type": "config", "slug": "kv-q8", "title": "KV cache q8_0 instead of q4_0",
        "hypothesis": "q8_0 keeps quality and costs little speed",
        "variant": {"server_args_add": ["--cache-type-k", "q8_0"], "server_args_remove": ["--cache-type-k"],
                    "request_extra": {}, "notes": ""},
        "metrics": ["passed", "passed.tools", "tok_s.prose"],
        "checks": ["the server log shows q8_0 for K"], "cost": {"gpu_min": 70, "disk_gb": 0, "usd": 0}}


def result(suite="2026-10-18", tools=(True, True, False), code=(True,), prose=(40.0, 44.0), ttft=(1.0, 2.0)):
    return {"label": "x", "suite": suite,
            "speed": {"prose": [{"tok_s": s, "ttft": f} for s, f in zip(prose, ttft)],
                      "parallel": {"runs": [{"tok_s": 30.0, "ttft": 1.5}, {"tok_s": 32.0, "ttft": 1.5}], "aggregate_tok_s": 60},
                      "alternating": {"turns": [{"ttft": 3.0}, {"ttft": 1.0}], "first_ttft": 3.0},
                      "deep_decode": {"30k": {"tok_s": 20.0, "ttft": 9.0}, "90k": {"error": "too small"}}},
            "tools": [{"case": "a", "ok": ok, "wall": 1} for ok in tools],
            "code": [{"case": "b", "ok": ok, "wall": 1} for ok in code],
            "total_seconds": 10}


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(dir=TMP))
        with open(ROOT / "local.example.toml", "rb") as f:
            conf = tomllib.load(f)
        conf["paths"].update(store=str(self.dir / "state" / "candidates.json"), state=str(self.dir / "state"),
                             experiments=str(self.dir / "state" / "experiments.json"),
                             results=str(self.dir / "results"), serving=str(self.dir / "serving"))
        conf["production"]["baseline"] = "prod-base"
        conf["wiki"]["inbox"] = str(self.dir / "inbox")
        conf.setdefault("experiments", {})["result_command"] = []
        self.conf = conf
        (self.dir / "results").mkdir()
        patch = mock.patch.object(config, "load", return_value=conf)
        patch.start()
        self.addCleanup(patch.stop)

    def cli(self, *args):
        out, err = io.StringIO(), io.StringIO()
        code = 0
        with mock.patch.object(sys, "argv", ["experiments.py", *args]), redirect_stdout(out), redirect_stderr(err):
            try:
                experiments.main()
            except SystemExit as ex:
                code = ex.code if isinstance(ex.code, int) else 1
                err.write(str(ex.code or ""))
        return code, (out.getvalue() + err.getvalue()).strip()

    def spec_file(self, **changes):
        spec = json.loads(json.dumps(SPEC))
        for key, value in changes.items():
            if key.startswith("variant_"):
                spec["variant"][key[8:]] = value
            else:
                spec[key] = value
        path = self.dir / "spec.json"
        path.write_text(json.dumps(spec))
        return str(path)

    def add(self, **changes):
        code, out = self.cli("add", "--spec-file", self.spec_file(**changes))
        self.assertEqual(code, 0, out)
        return out

    def exps(self):
        with experiments.store(readonly=True) as data:
            return data["experiments"]

    def write_result(self, label, res):
        (self.dir / "results" / f"{label}.json").write_text(json.dumps(res))


class TestPure(Base):
    def test_metrics(self):
        m = experiments.metrics(result())
        self.assertEqual((m["passed"], m["passed.tools"], m["passed.code"]), (3, 2, 1))
        self.assertAlmostEqual(m["tok_s.prose"], 42.0)
        self.assertAlmostEqual(m["ttft.prose"], 1.5)
        self.assertAlmostEqual(m["tok_s.parallel"], 31.0)
        self.assertAlmostEqual(m["ttft.alternating"], 2.0)
        self.assertAlmostEqual(m["tok_s.deep_decode"], 20.0)  # the depth with an error has no number
        self.assertNotIn("tok_s.alternating", m)

    def test_extra_sections_are_metrics_but_not_in_passed(self):
        r = result()
        r["extra"] = {"robust": [{"case": "a", "ok": True}, {"case": "b", "ok": False}], "longagent": []}
        m = experiments.metrics(r)
        self.assertEqual(m["extra.robust"], 1)
        self.assertNotIn("extra.longagent", m)
        self.assertEqual(m["passed"], experiments.metrics(result())["passed"])
        only = experiments.metrics({"suite": "s", "extra": {"longagent": [{"case": "a", "ok": True}]}})
        self.assertEqual(only, {"extra.longagent": 1})  # an extras-only run has no core "passed"
        self.assertIn("extra.robust", experiments.KNOWN_METRICS)
        t = experiments.metrics({"suite": "s", "own_traffic": {"replay": [{"case": "a", "ok": True}]}})
        self.assertEqual(t, {"own_traffic.replay": 1})
        self.assertIn("own_traffic.ctx", experiments.KNOWN_METRICS)

    def test_compare_one(self):
        self.assertEqual(experiments.compare_one("x", 110.0, 100.0), {"variant": 110.0, "baseline": 100.0, "delta": 10.0, "ratio": 1.1})
        self.assertEqual(experiments.compare_one("x", 1, 0)["ratio"], None)
        self.assertEqual(experiments.compare_one("x", None, 1)["delta"], None)
        self.assertIn("no value (variant)", experiments.compare_text("x", experiments.compare_one("x", None, 1)))

    def test_suite_mismatch_is_failed(self):
        status, rows, why = experiments.compare(SPEC, result(suite="2026-10-18"), result(suite="2026-10-01"))
        self.assertEqual((status, rows, why), ("failed", {}, "baseline suite differs"))

    def test_compare_has_no_verdict(self):
        worse = result(tools=(False,) * 3, prose=(10.0, 10.0))
        worse["extra"] = {"robust": [{"case": "a", "ok": True}]}
        status, rows, _ = experiments.compare(SPEC, worse, result())
        self.assertEqual(status, "done")  # measured; nothing passes or fails
        self.assertEqual(list(rows), ["passed", "passed.tools", "tok_s.prose", "extra.robust"])
        self.assertEqual((rows["passed"]["variant"], rows["passed"]["baseline"], rows["passed"]["delta"]), (1, 3, -2))
        self.assertEqual(rows["extra.robust"]["baseline"], None)
        self.assertEqual(list(experiments.compare({}, result(), result())[1]), experiments.DEFAULT_METRICS)

    def test_status_mapping_of_every_candidate_status(self):
        want = {"proposed": "planned", "pending": "approved", "downloading": "approved", "downloaded": "queued",
                "testing": "running", "tested": "done", "rejected": "done", "skipped": "dropped",
                "failed": "failed"}
        self.assertEqual(set(want), set(candidates.STATUSES))
        for status, unified in want.items():
            with self.subTest(status=status):
                self.assertEqual(experiments.unified_status({"status": status}), unified)


class TestAdd(Base):
    def test_valid_spec_is_planned(self):
        eid = self.add()
        self.assertRegex(eid, r"^\d{8}-kv-q8$")
        e = self.exps()[eid]
        self.assertEqual((e["status"], e["type"], e["priority"], e["comparison"]), ("planned", "config", 0, {}))
        self.assertNotIn("pass_rule", e)

    def test_refusals(self):
        bad = {"bad slug": {"slug": "KV Q8!"}, "short slug": {"slug": "kv"},
               "unknown metric": {"metrics": ["passed", "speed.magic"]},
               "dotdot build": {"variant_llama_cpp": "../llama.cpp-next"},
               "absolute build": {"variant_llama_cpp": "/opt/llama.cpp"},
               "newline arg": {"variant_server_args_add": ["--ctx-size\n4096"]},
               "backend on config": {"variant_backend": "vllm"},
               "bad type": {"type": "agent"}}
        for name, changes in bad.items():
            with self.subTest(name):
                code, out = self.cli("add", "--spec-file", self.spec_file(**changes))
                self.assertEqual(code, 1, out)
                self.assertIn("Not added", out)
        self.assertEqual(self.exps(), {})

    def test_engine_with_backend_and_duplicate(self):
        self.add(type="engine", variant_backend="vllm", slug="vllm-engine")
        code, out = self.cli("add", "--spec-file", self.spec_file(type="engine", variant_backend="vllm", slug="vllm-engine"))
        self.assertEqual(code, 1)
        self.assertIn("exists already", out)

    def test_add_type_model_calls_candidates_add(self):
        with mock.patch.object(candidates, "add", return_value=(True, "Queued a/B-GGUF")) as add:
            code, out = self.cli("add", "--type", "model", "a/B-GGUF", "--backend", "vllm", "--now")
        self.assertEqual((code, out), (0, "Queued a/B-GGUF"))
        add.assert_called_once_with("a/B-GGUF", "vllm", now_=True)


class TestTransitions(Base):
    def test_approve_and_queue(self):
        eid = self.add()
        other = self.add(slug="still-planned")
        self.assertEqual(self.cli("queue")[0], 0)
        self.assertEqual(self.exps()[eid]["status"], "planned")  # queue takes approved ones only
        self.assertEqual(self.cli("approve", eid)[0], 0)
        self.assertEqual(self.exps()[eid]["status"], "approved")
        code, out = self.cli("approve", eid)
        self.assertEqual(code, 1, out)  # approved already
        self.cli("queue")
        self.assertEqual((self.exps()[eid]["status"], self.exps()[other]["status"]), ("queued", "planned"))

    def test_drop_and_set(self):
        eid = self.add()
        self.assertEqual(self.cli("drop", eid, "not worth it")[0], 0)
        self.assertEqual((self.exps()[eid]["status"], self.exps()[eid]["summary"]), ("dropped", "not worth it"))
        code, out = self.cli("set", eid, "done", "I think it is good")
        self.assertEqual(code, 1, out)  # done comes from record, never set
        self.assertEqual(self.exps()[eid]["status"], "dropped")

    def test_unified_list_shows_both_stores(self):
        self.add()
        with candidates.store() as data:
            data["candidates"]["a/B-GGUF"] = {"name": "B", "status": "downloaded", "brief": "2026-10-01"}
        code, out = self.cli("list", "--json")
        items = json.loads(out)
        self.assertEqual(items["a/B-GGUF"]["type"], "model")
        self.assertEqual(items["a/B-GGUF"]["status"], "queued")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(self.cli("show", "a/B-GGUF", "--json")[1])["candidate"]["status"], "downloaded")


class TestRecord(Base):
    def setUp(self):
        super().setUp()
        self.eid = self.add()
        self.cli("approve", self.eid)
        self.cli("queue")
        self.cli("set", self.eid, "running")
        self.write_result("prod-base", result())

    def test_record_with_wiki_note(self):
        self.write_result(f"exp-{self.eid}", result(prose=(20.0, 20.0)))
        code, out = self.cli("record", self.eid, f"exp-{self.eid}")
        self.assertEqual(code, 0, out)
        self.assertIn("done", out)
        self.assertIn("tok_s.prose 20 vs 42", out)
        e = self.exps()[self.eid]
        self.assertEqual((e["status"], e["result"]), ("done", f"exp-{self.eid}"))
        self.assertEqual(e["comparison"]["passed"]["variant"], 3)
        self.assertNotIn("verdict", e)
        notes = list((self.dir / "inbox").glob("*.md"))
        self.assertEqual(len(notes), 1)
        text = notes[0].read_text()
        self.assertIn("No verdict", text)
        self.assertNotIn("beneficial", text)

    def test_old_verdict_name_records(self):
        self.write_result(f"exp-{self.eid}", result())
        self.assertEqual(self.cli("record", self.eid, f"exp-{self.eid}")[0], 0)
        self.assertEqual(self.exps()[self.eid]["status"], "done")

    def test_suite_mismatch_via_cli(self):
        self.write_result(f"exp-{self.eid}", result(suite="2026-11-01"))
        code, out = self.cli("record", self.eid, f"exp-{self.eid}")
        self.assertIn("baseline suite differs", out)
        self.assertEqual(self.exps()[self.eid]["status"], "failed")

    def test_missing_result_is_failed(self):
        self.cli("record", self.eid, "exp-nothing")
        self.assertEqual(self.exps()[self.eid]["status"], "failed")

    def test_bad_label_is_refused(self):
        code, out = self.cli("record", self.eid, "../prod-base")
        self.assertEqual(code, 1, out)
        self.assertEqual(self.exps()[self.eid]["status"], "running")

    def test_result_command_is_called_with_the_id(self):
        calls = self.dir / "calls.txt"
        stub = self.dir / "stub.py"
        stub.write_text("import sys\nopen(sys.argv[1], 'a').write(' '.join(sys.argv[2:]) + '\\n')\n")
        self.conf["experiments"]["result_command"] = [sys.executable, str(stub), str(calls), "--result"]
        self.write_result(f"exp-{self.eid}", result())
        self.cli("record", self.eid, f"exp-{self.eid}")
        self.assertEqual(calls.read_text().splitlines(), [f"--result {self.eid}"])

    def test_result_command_failure_is_logged_not_raised(self):
        self.conf["experiments"]["result_command"] = [str(self.dir / "no-such-program")]
        self.write_result(f"exp-{self.eid}", result())
        code, out = self.cli("record", self.eid, f"exp-{self.eid}")
        self.assertEqual(code, 0)
        self.assertIn("result_command failed", out)

    def test_candidates_set_calls_the_hook_on_final_status(self):
        with candidates.store() as data:
            data["candidates"]["a/B-GGUF"] = {"name": "B", "status": "testing"}
        with mock.patch.object(experiments, "result_hook") as hook:
            for status in ("testing", "rejected"):
                with mock.patch.object(sys, "argv", ["candidates.py", "set", "a/B-GGUF", status, "x"]):
                    candidates.main()
        hook.assert_called_once_with("a/B-GGUF")


class TestRunner(Base):
    def test_priority_2_experiment_before_priority_1_candidate(self):
        cands = {"candidates": {"a/Model-GGUF": {"status": "downloaded", "model": "model-27b", "priority": 1,
                                                 "brief": "2026-10-01"}}}
        exps = {"experiments": {"20261024-kv-q8": {"status": "queued", "type": "config", "priority": 2,
                                                   "created": "2026-10-24T00:00:00+00:00"},
                                "20261024-not-queued": {"status": "approved", "type": "config", "priority": 9}}}
        ready = runner.ready_queue(cands, 10 ** 6, exps)
        self.assertEqual([(kind, key) for *_, kind, key in ready],
                         [("experiment", "20261024-kv-q8"), ("model", "a/Model-GGUF")])

    def test_slow_engine_waits_for_a_long_slot(self):
        exps = {"experiments": {"20261024-vllm": {"status": "queued", "type": "engine", "priority": 0, "created": "",
                                                  "variant": {"backend": "vllm"}}}}
        self.assertEqual(runner.ready_queue({"candidates": {}}, runner.MIN_TEST_S, exps), [])
        self.assertEqual(len(runner.ready_queue({"candidates": {}}, runner.VLLM_MIN_TEST_S, exps)), 1)

    def test_prompt_names_procedure_and_spec(self):
        eid = self.add()
        e = self.exps()[eid]
        prompt = runner.prompt_for(eid, e, config.now().replace(hour=23, minute=59))
        self.assertIn("EXPERIMENT.md", prompt)
        self.assertIn("<<<SPEC", prompt)
        self.assertIn("$EXPERIMENTS = ", prompt)
        self.assertIn(f"exp-{eid}", prompt)

    def test_unfinished_experiment_is_failed(self):
        eid = self.add()
        self.cli("approve", eid)
        self.cli("queue")
        with mock.patch.object(runner, "agent", return_value=(None, "stopped after 60 min", False)), \
                mock.patch.object(runner.subprocess, "run"), mock.patch.object(runner, "log"):
            runner.test_experiment(eid, config.now().replace(hour=23, minute=59))
        e = self.exps()[eid]
        self.assertEqual(e["status"], "failed")
        self.assertIn("stopped after 60 min", e["summary"])


if __name__ == "__main__":
    unittest.main()
