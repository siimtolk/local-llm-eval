import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import live_board  # noqa: E402


def rows(*oks):
    return [{"case": f"c{i}", "ok": ok} for i, ok in enumerate(oks)]


class TestExtraScores(unittest.TestCase):
    def test_extra_and_own_traffic_are_kept_apart(self):
        res = {"tools": rows(True), "extra": {"robust": rows(True, False, True), "vision": rows(True), "other": rows(True)},
               "own_traffic": {"replay": rows(True, True, False), "ctx": rows(False)},
               "section_versions": {"robust": "v1", "replay": "v1", "vision": "v2"}}
        s = live_board.extra_scores(res)
        self.assertEqual(s["extra"], {"vision": [1, 1], "robust": [2, 3]})  # unknown sections are not shown
        self.assertEqual(s["own_traffic"], [2, 4])
        self.assertEqual(s["own_traffic_parts"], {"replay": [2, 3], "ctx": [0, 1]})
        self.assertEqual(s["section_versions"], {"robust": "v1", "vision": "v2", "replay": "v1"})

    def test_a_result_without_them_has_none(self):
        s = live_board.extra_scores({"tools": rows(True)})
        self.assertEqual((s["extra"], s["own_traffic"], s["own_traffic_parts"]), ({}, None, {}))

    def test_they_do_not_change_the_graded_total(self):
        base = {"tools": rows(True, False)}
        with_extra = dict(base, extra={"robust": rows(True)}, own_traffic={"ctx": rows(True)})
        self.assertEqual(live_board.totals(base, set()), live_board.totals(with_extra, set()))


if __name__ == "__main__":
    unittest.main()
