# Cutover: move a machine's nightly test onto this repo (phase 2)

Written for the reference machine; ready to execute, **not executed**. About an hour, in the
daytime, outside the test window, with production stopped once (step 3). Every step names the
file and line that changes. `$EVAL`, `$SERVING`, `$CWA` (the chat-bot repo that starts the
runner) and `$UNITS` (`~/.config/systemd/user`) are the paths in this machine's `local.toml` and
unit files; set them in the shell first:

```bash
L=<the directory that holds the repos>          # parent of $SERVING, $CWA and this repo
EVAL=$L/local-llm-eval  SERVING=$L/serve-llm-with-llama  NEW=$L/local-llm-serving
CWA=$L/claude-wa  UNITS=~/.config/systemd/user
STAMP=$(date +%Y%m%dT%H%M%S)
```

What moves: the model test (runner), the downloads (fetch) and the candidate procedures.
What stays for now: the assistant's own morning jobs (the local-models brief and the
model-test-brief) and the chat bridge's poll votes keep running from the assistant's config repo
against the same store
(`paths.store` already points at it), so the morning report does not change. Moving discover,
judge and report onto this repo is phase 3.

## 0. Preconditions

```bash
systemctl --user is-active claude-model-test claude-model-test-early claude-bkg-tasks   # all inactive
curl -s 127.0.0.1:8080/health                                                          # {"status":"ok"}
python3 $EVAL/pipeline/candidates.py list | awk '$1=="testing"'                         # nothing
git -C $SERVING status --short; git -C $CWA status --short                              # clean, or commit first
```

## 1. Pause the two timers that start our code (production and llm-guard stay as they are)

```bash
systemctl --user stop claude-model-test.timer claude-model-fetch.timer
```

## 2. Merge the backend interface into the serving repo

```bash
git -C $SERVING fetch origin
git -C $SERVING merge --no-edit origin/eval-interface        # adds bin/serve, a README section; nothing else changes
git -C $SERVING push
git -C $SERVING worktree remove $L/local-llm-serving-wt      # the branch stays on GitHub
$SERVING/bin/serve health qwen3-8-27b-rvn                    # {"status":"ok"}, exit 0
```

## 3. Rename the serving directory, symlink the old path back (production stopped once)

`/etc/systemd/system/serve-llm.service` keeps `WorkingDirectory=` and `ExecStart=` on the old
path; the symlink makes that work without sudo edits until phase 3. The llama.cpp worktree
(`llama.cpp-prod/.git` → `.../serve-llm-with-llama/llama.cpp/.git/worktrees/...`) and the build's
absolute paths resolve through the symlink as well.

```bash
$SERVING/bin/stop_production.sh                 # the Claude fallback takes 8080 within a second
curl -s 127.0.0.1:8080/health                   # 503 from the fallback: expected
mv $SERVING $NEW && ln -s local-llm-serving $SERVING
sudo -n systemctl start serve-llm || $NEW/bin/ensure_production.sh
for i in $(seq 60); do curl -sf 127.0.0.1:8080/health | grep -q ok && break; sleep 10; done; curl -s 127.0.0.1:8080/health
git -C $NEW remote -v                           # origin is siimtolk/local-llm-serving already
git -C $NEW/llama.cpp worktree repair $NEW/llama.cpp-prod   # optional: record the new path
```

## 4. local.toml (this repo, gitignored)

| Line | Before | After |
|---|---|---|
| `[paths] serving` | `"$L/serve-llm-with-llama"` | `"$L/local-llm-serving"` |
| `[serving] api_key_file` | `"$L/serve-llm-with-llama/secrets/primary/api_key.txt"` | `"$L/local-llm-serving/secrets/primary/api_key.txt"` |

Then regenerate the agent's allowlist and check:

```bash
cd $EVAL && python3 tools/agent_settings.py
python3 pipeline/runner.py model-test --dry-run --at 01:00
```

## 5. Units in `$UNITS` (back up each first: `cp X X.bak-$STAMP`)

| Unit | Line | After |
|---|---|---|
| `claude-model-test.service` | `Description=` | `Nightly local model test (local-llm-eval/pipeline/runner.py model-test)` |
| | `WorkingDirectory=` | `$EVAL/pipeline` |
| | `ExecStart=` | `/usr/bin/python3 runner.py model-test` (unchanged) |
| `claude-model-fetch.service` | `Description=` | `Download local model candidates (local-llm-eval/pipeline/fetch.py)` |
| | `WorkingDirectory=` | `$EVAL/pipeline` |
| | `ExecStart=` | `/usr/bin/python3 fetch.py` |
| `llm-guard.service` | `ExecStart=` | `$NEW/bin/ensure_production.sh` |
| `llm-claude-fallback.service` | `ExecStart=` | `/usr/bin/python3 $NEW/bin/claude_fallback.py` (and the path in `Description=`) |

Unit names stay (`[agent] nightly_units` and `bkg_unit` in local.toml use them). Then:

```bash
systemctl --user daemon-reload
systemctl --user restart llm-claude-fallback    # idle while production runs; picks up the new path
```

## 6. The chat-bot repo (`$CWA`), backups first (`cp X X.bak-$STAMP`)

- `runner.py`, `main()`: hand the model-test commands to this repo; `chat` stays. Add before
  `if sys.argv[1:] == ["chat"]:`
  ```python
  if sys.argv[1:2] in (["model-test"], ["bkg-start"], ["bkg-run"], ["bkg-stop"]):
      os.execv(sys.executable, [sys.executable, "<EVAL>/pipeline/runner.py", *sys.argv[1:]])
  ```
  (`<EVAL>` written out). `listener.mjs` keeps calling `runner.py bkg-start` / `bkg-stop`; the
  transient unit `claude-bkg-tasks` then runs this repo's `runner.py bkg-run`.
- `git_nightly.sh` line 16, `REPOS=(...)`: replace `.../serve-llm-with-llama` with
  `.../local-llm-serving` and add `.../local-llm-eval`. The eval repo's pre-commit hook runs the
  leak check on the nightly commit; a blocked commit is reported like any other failure.
- `vllm_test.sh` line 5 and `prod_build_switch.sh` line 5, `cd .../serve-llm-with-llama`:
  `.../local-llm-serving` (works through the symlink either way; they still use the serving
  repo's own `bench/` until phase 3).
- `fetch_candidates.py`, `candidates.py`: unchanged; nothing calls `fetch_candidates.py` after
  step 5, `candidates.py` writes the same store.

Commit and push `$CWA` (`Cutover: model test runs from local-llm-eval`).

## 7. Resume and check

```bash
systemctl --user start claude-model-test.timer claude-model-fetch.timer
systemctl --user list-timers | grep -E 'claude-model|llm-guard|git-nightly'
systemctl --user cat claude-model-test claude-model-fetch | grep -E 'WorkingDirectory|ExecStart'
```

- [ ] `curl -s 127.0.0.1:8080/health` ok and one chat completion answers (production restarted in step 3)
- [ ] fallback handover seen in step 3 (503 while stopped, port given back after start)
- [ ] `$NEW/bin/serve health qwen3-8-27b-rvn` exit 0; `$NEW/bin/serve stop --all` prints no error
- [ ] `python3 $EVAL/pipeline/fetch.py --dry-run` and `runner.py model-test --dry-run --at 01:00` run
- [ ] `python3 $EVAL/bench/bench.py --label cutover-smoke --only judge --results-dir /tmp` 18/18 in about 40 s
- [ ] `python3 $CWA/runner.py model-test --dry-run` reaches this repo's runner (prints the queue)
- [ ] one manual test with a downloaded candidate: `/runbkgtasks` in the chat, watch
      `journalctl --user -u claude-bkg-tasks -f` until the agent serves on 8085, then `/stopbkgtasks`
      (expect "Tootmismudel vastab." and the candidate back in the queue)
- [ ] first night: `$EVAL/state/runs/` has the run log, the result is in `results/` and committed,
      the morning model-test-brief arrives, `git-nightly` succeeds for all four repos

## Rollback

```bash
systemctl --user stop claude-model-test.timer claude-model-fetch.timer
for f in claude-model-test.service claude-model-fetch.service llm-guard.service llm-claude-fallback.service; do
  cp $UNITS/$f.bak-$STAMP $UNITS/$f; done
systemctl --user daemon-reload && systemctl --user restart llm-claude-fallback
for f in runner.py git_nightly.sh vllm_test.sh prod_build_switch.sh; do cp $CWA/$f.bak-$STAMP $CWA/$f; done
# directory (only if the symlink causes trouble; production stops once more):
$NEW/bin/stop_production.sh && rm $SERVING && mv $NEW $SERVING && (sudo -n systemctl start serve-llm || $SERVING/bin/ensure_production.sh)
systemctl --user start claude-model-test.timer claude-model-fetch.timer
```

The merged `bin/serve` can stay: nothing else uses it. A candidate left `testing` by an
interrupted run goes back with `python3 $CWA/candidates.py set <repo> downloaded`.
