# Development

- **`main` is what the nightly run uses.** The nightly agent commits results and journal entries
  straight to `main`, so `main` must always work: never push half-finished changes there.
- **Develop on branches** (`git switch -c <topic>`), test with the dry runs below, then merge
  in the daytime, outside the test window.
- **Install the leak check once per clone** and never bypass it (no `--no-verify`):
  `ln -sf ../../tools/leak_check.sh .git/hooks/pre-commit`. Machine-specific values (paths,
  chat ids, addresses) belong in `local.toml`, which is gitignored; extra patterns for your
  machine go in `.leak-patterns.local`. CI runs the same check plus gitleaks on every push.
- **Checks without GPUs:**
  ```bash
  python3 -m py_compile bench/*.py pipeline/*.py tools/*.py
  python3 bench/bench.py --help
  python3 pipeline/fetch.py --dry-run
  python3 pipeline/judge.py --dry-run
  python3 pipeline/runner.py model-test --dry-run --at 01:00
  python3 tools/results_table.py
  tools/leak_check.sh --all
  ```
- **Bench cases** (`bench/cases.py`): results are comparable only within one `SUITE` version.
  Changing what a case asks or how it is graded means a new `SUITE` date.
- Python 3.11+ (tomllib), stdlib only.
