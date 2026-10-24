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
        "pass_rule": [{"metric": "passed", "min_delta": 0}, {"metric": "tok_s.prose", "min_ratio": 0.95}],
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
        conf.setdefault("experiments", {})["verdict_command"] = []
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

    def test_each_operator(self):
        v, b = {"x": 110.0}, {"x": 100.0}
        cases = [({"min_delta": 10}, True), ({"min_delta": 11}, False), ({"max_delta": 10}, True), ({"max_delta": 9}, False),
                 ({"min_ratio": 1.1}, True), ({"min_ratio": 1.11}, False), ({"max_ratio": 1.1}, True),
                 ({"max_ratio": 1.09}, False), ({"min_delta": 0, "max_ratio": 1.05}, False)]
        for ops, want in cases:
            with self.subTest(ops=ops):
                self.assertIs(experiments.check({"metric": "x", **ops}, v, b)[0], want)

    def test_missing_value_or_zero_baseline_fails(self):
        self.assertFalse(experiments.check({"metric": "x", "min_delta": 0}, {}, {"x": 1})[0])
        self.assertFalse(experiments.check({"metric": "x", "min_ratio": 1}, {"x": 1}, {"x": 0})[0])

    def test_suite_mismatch_is_failed(self):
        status, v, why = experiments.evaluate(SPEC, result(suite="2026-10-18"), result(suite="2026-10-01"))
        self.assertEqual(status, "failed")
        self.assertEqual(why, "baseline suite differs")
        self.assertEqual((v["passed"], v["failed"]), ([], []))

    def test_evaluate_beneficial_and_not(self):
        self.assertEqual(experiments.evaluate(SPEC, result(), result())[0], "beneficial")
        self.assertEqual(experiments.evaluate(SPEC, result(tools=(False,) * 3), result())[0], "not-beneficial")

    def test_status_mapping_of_every_candidate_status(self):
        want = {"proposed": "planned", "pending": "approved", "downloading": "approved", "downloaded": "queued",
                "testing": "running", "tested": "beneficial", "rejected": "not-beneficial", "skipped": "dropped",
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
        self.assertEqual((e["status"], e["type"], e["priority"], e["verdict"]), ("planned", "config", 0, {}))

    def test_refusals(self):
        bad = {"bad slug": {"slug": "KV Q8!"}, "short slug": {"slug": "kv"},
               "unknown metric": {"metrics": ["passed", "speed.magic"]},
               "unknown rule metric": {"pass_rule": [{"metric": "tok_s.nonsense", "min_ratio": 1}]},
               "rule without operator": {"pass_rule": [{"metric": "passed"}]},
               "rule with a string": {"pass_rule": [{"metric": "passed", "min_delta": "0"}]},
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
        code, out = self.cli("set", eid, "beneficial", "I think it is good")
        self.assertEqual(code, 1, out)  # the verdict is computed, never set
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


class TestVerdict(Base):
    def setUp(self):
        super().setUp()
        self.eid = self.add()
        self.cli("approve", self.eid)
        self.cli("queue")
        self.cli("set", self.eid, "running")
        self.write_result("prod-base", result())

    def test_beneficial_with_wiki_note(self):
        self.write_result(f"exp-{self.eid}", result(prose=(44.0, 46.0)))
        code, out = self.cli("verdict", self.eid, f"exp-{self.eid}")
        self.assertEqual(code, 0, out)
        self.assertIn("beneficial", out)
        e = self.exps()[self.eid]
        self.assertEqual((e["status"], e["result"], e["verdict"]["failed"]), ("beneficial", f"exp-{self.eid}", []))
        self.assertEqual(e["verdict"]["metrics"]["passed"], [3, 3])
        self.assertEqual(set(e["verdict"]["rules"]), {"passed", "tok_s.prose"})
        notes = list((self.dir / "inbox").glob("*.md"))
        self.assertEqual(len(notes), 1)
        self.assertIn("beneficial", notes[0].read_text())

    def test_not_beneficial(self):
        self.write_result(f"exp-{self.eid}", result(prose=(20.0, 20.0)))
        self.cli("verdict", self.eid, f"exp-{self.eid}")
        e = self.exps()[self.eid]
        self.assertEqual((e["status"], e["verdict"]["failed"]), ("not-beneficial", ["tok_s.prose"]))

    def test_suite_mismatch_via_cli(self):
        self.write_result(f"exp-{self.eid}", result(suite="2026-11-01"))
        code, out = self.cli("verdict", self.eid, f"exp-{self.eid}")
        self.assertIn("baseline suite differs", out)
        self.assertEqual(self.exps()[self.eid]["status"], "failed")

    def test_missing_result_is_failed(self):
        self.cli("verdict", self.eid, "exp-nothing")
        self.assertEqual(self.exps()[self.eid]["status"], "failed")

    def test_bad_label_is_refused(self):
        code, out = self.cli("verdict", self.eid, "../prod-base")
        self.assertEqual(code, 1, out)
        self.assertEqual(self.exps()[self.eid]["status"], "running")

    def test_verdict_command_is_called_with_the_id(self):
        calls = self.dir / "calls.txt"
        stub = self.dir / "stub.py"
        stub.write_text("import sys\nopen(sys.argv[1], 'a').write(' '.join(sys.argv[2:]) + '\\n')\n")
        self.conf["experiments"]["verdict_command"] = [sys.executable, str(stub), str(calls), "--verdict"]
        self.write_result(f"exp-{self.eid}", result())
        self.cli("verdict", self.eid, f"exp-{self.eid}")
        self.assertEqual(calls.read_text().splitlines(), [f"--verdict {self.eid}"])

    def test_verdict_command_failure_is_logged_not_raised(self):
        self.conf["experiments"]["verdict_command"] = [str(self.dir / "no-such-program")]
        self.write_result(f"exp-{self.eid}", result())
        code, out = self.cli("verdict", self.eid, f"exp-{self.eid}")
        self.assertEqual(code, 0)
        self.assertIn("verdict_command failed", out)

    def test_candidates_set_calls_the_hook_on_final_status(self):
        with candidates.store() as data:
            data["candidates"]["a/B-GGUF"] = {"name": "B", "status": "testing"}
        with mock.patch.object(experiments, "verdict_hook") as hook:
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
