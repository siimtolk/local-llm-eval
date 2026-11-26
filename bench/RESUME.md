# Resume run: the sections an earlier test did not reach

The benchmark has no length cap, so a long one (a slow model, the extra sections) can run into the end
of the night. Nothing is lost: bench.py saves after every section and keeps every answered request in a
checkpoint, and after each test the runner queues the labels that lack sections of the full suite
(`pipeline/resume.py`). Tonight you continue those labels for one model (or one experiment): serve it
again exactly as before, run bench.py with `--only <the missing sections>` and the same label (the results
are merged into the same file; a section cut off in the middle resumes at the request it stopped at), and
record it. The runner's prompt names the labels, what each lacks and the exact bench command for each.

Everything in `bench/NIGHTLY.md` → Rules applies (time, untrusted model files, what you may change). The
`$NAMES` are this machine's settings at the end of the prompt, as in NIGHTLY.md, plus `$RESUME`
(`pipeline/resume.py`), `$BENCH_BG` and `$BENCH_STOP_AT`.

## Steps

1. **Read how it was served:** the label's section in `$EVAL/JOURNAL.md` (the "Setup" lines), the config
   `$SERVING/models.d/<model>.conf.sh` (an experiment: its spec and `bench/EXPERIMENT.md` steps 2–4) and
   `ls -la $MODELS/<model>/`. Serve it the same way: the same config, files and flags. If the journal names
   a llama.cpp build that is no longer there, use `$SERVING/$CANDIDATE_LLAMA_CPP` and say so in the write-up
   (the sections then come from another build than the label's core numbers). Weights missing → restore
   nothing (production is still up), record each label with
   `$RESUME done <label> 'not run: weights missing'` and stop: the runner keeps the entry.
2. **Stop production** (`$STOP_PRODUCTION`) and **serve** the model as in NIGHTLY.md step 5 (for a vLLM / capy
   build its own procedure, NIGHTLY_VLLM.md / NIGHTLY_CAPY.md, with `--backend`). Wait for health; not serving
   within 30 min → step 4, and record `not run: <reason>`.
3. **Benchmark** each label with the command from the prompt, main label first:
   `cd $EVAL`, then `$BENCH_BG start $SERVING/logs/bench-<label>-resume.log <the command's arguments after bench/bench.py>`
   and `$BENCH_BG wait --stop-at "$BENCH_STOP_AT"`, again while it exits 3. No other time limit, no `timeout`,
   no own background command (as NIGHTLY.md step 6). If it stops at `$BENCH_STOP_AT` (exit 4), go on: what is
   still missing runs the next night. Do not change the command's `--reps`, `--temperature` or
   `--request-extra`: they are the label's own, so the new sections compare with its other numbers.
4. **Restore production:** as NIGHTLY.md step 7 (`$SERVE stop <model>` with its `--backend`, then
   `$START_PRODUCTION || $ENSURE_PRODUCTION`; check health and one chat completion).
5. **Write it up** in `$EVAL/JOURNAL.md`, in the label's existing section (do not start a new one): a line
   `**Resumed <date>:**` with the setup if it differed, a row per new section in its per-section table
   (passed, failed cases) and, for the extra sections, production's numbers from `$RESULTS/$BASELINE.json`
   next to them; what is still missing (`$RESUME missing <label>`). No verdict (NIGHTLY.md step 10). Then
   `python3 tools/results_table.py --write`.
6. **Commit and push** in `$EVAL`: `results/<label>.json`, `JOURNAL.md`, `README.md`, message
   `Resume <model>: <sections>`; the pre-commit leak check must pass. `git push`.
7. **Record** each label: `$RESUME done <label> '<summary>'`, plain text in `$SUMMARY_LANGUAGE`, at most
   ~300 characters: the new sections' scores against production's, and what is still missing, e.g.
   `lisad: robust 64/72 (prod 62), longagent 20/21 (20); replay, ctx järgmisel ööl`.
   The morning report adds it to the model's summary.

If anything fails, restore production first (step 4), then record `not run: <one-sentence reason>`.
