# local-llm-eval: an automated nightly LLM testing pipeline

An unattended loop that looks for new open-weight models every day, downloads the promising ones,
benchmarks each against the model you run in production overnight, and reports in the morning.
A coding agent (Claude Code, headless) does the overnight test by following a written procedure;
everything around it is plain Python with no dependencies outside the standard library.

Write-up with the results of the first two weeks: [Automated nightly LLM testing: an AI agent benchmarks new local models while I sleep](https://siimtolk.github.io/articles/llm-testing-overnight/).

It was built for one home server (two 16 GB GPUs, llama.cpp and vLLM) that serves a local model
to an everyday assistant, and it is the repo that machine's nightly test runs from. Every
machine-specific value lives in `local.toml`, which is not tracked.

## The five-stage loop

| Stage | Script | When | LLM | What it does |
|---|---|---|---|---|
| 1. Discover | `pipeline/discover.py` | morning | no | Hugging Face trending and searches (`uncensored`, `abliterated`, ...), new repos of trusted publishers, r/LocalLLaMA reactions; keeps what fits the machine and is new |
| 2. Judge | `pipeline/judge.py` | morning | yes | one turn of a model reads the candidates (`prompts/judge.md`) and picks at most three worth testing; the brief goes to your chat |
| 3. Fetch | `pipeline/fetch.py` | daytime | no | risk check (`pipeline/risk.py`: HF file scans, the GGUF header, the uploader); after you tick candidates in a poll, picks the quant that fits and downloads it; also queues a 4-bit vLLM build of the same model |
| 4. Test | `pipeline/runner.py model-test` | night window | yes | the agent follows `bench/NIGHTLY.md` (or `NIGHTLY_VLLM.md`): stop production, serve the candidate, run `bench/bench.py`, restore production, write up, commit, record a verdict |
| 5. Report | `pipeline/report.py` | morning | no | each candidate's verdict and summary, plus the new proposals with their risk and a poll |

The candidates store (`paths.store`) carries each model through
`proposed → pending → downloading → downloaded → testing → tested | rejected | skipped | failed`;
`pipeline/candidates.py` is the only code that changes it.

The benchmark (`bench/bench.py`, cases in `bench/cases.py`) works against any OpenAI-compatible
endpoint: speed timed on the client (prose, tool JSON, a 30k-token prompt, two requests at once,
two alternating chats, decode speed after ~30k and ~90k tokens of context) and 97 graded cases × 3 runs in tool use, agent tasks, reasoning,
extraction, code (run in a bubblewrap sandbox), hard multi-step tasks, instruction following,
judging, Estonian, and an uncensored set of edgy-but-legitimate requests that a stock model
tends to refuse.

## Live results board

`tools/live_board.py` turns `results/*.json` into `_data/results.json` in the blog repo (`[board] blog_repo` in
`local.toml`); the blog's `/results/` page draws the quality-versus-speed plot and the tables from it. The
post-commit hook `tools/post_commit_board.sh` (install: `ln -sf ../../tools/post_commit_board.sh .git/hooks/post-commit`)
runs `--publish` after every commit that touches `results/*.json`, so a new result shows up without touching any
article. Display names and kinds (dense, mixture of experts, ...) are in `tools/board_names.json`; a setup without an
entry still appears, under a name made from its label.

## Experiments

Models are one kind of experiment. A `config` experiment changes how production's model is served
(a server argument such as the KV cache type, sampling, a newer llama.cpp build); an `engine`
experiment serves it with another backend (`llamacpp`, `vllm`, `capy`). Both run on the test port
with the same suite by `bench/EXPERIMENT.md`, in the same queue as the candidates (higher
priority first, then oldest), and never change production. A `model` experiment is a candidate:
it stays in the candidates store and `candidates.py` as before.

Config and engine experiments live in `paths.experiments` and go
`planned → approved → queued → running → beneficial | not-beneficial | failed` (or `dropped`).
A spec names its metrics (`passed`, `passed.<section>`, `tok_s.<speed test>`, `ttft.<speed test>`)
and a pass rule: conditions `min_delta` / `max_delta` (variant − baseline) and `min_ratio` /
`max_ratio` (variant / baseline) against production's baseline (`[production] baseline`). The
verdict is computed in code, never by the agent; a baseline from another suite makes it `failed`.

```bash
python3 pipeline/experiments.py add --spec-file kv-q8.json   # validated, stored as planned; prints the id
python3 pipeline/experiments.py approve 20261024-kv-q8        # then: queue (approved -> queued)
python3 pipeline/experiments.py list [--json]                 # both stores; show <id|repo> for one
python3 pipeline/experiments.py verdict 20261024-kv-q8 exp-20261024-kv-q8   # run by the procedure
```

After a final status a note goes to the wiki inbox (`[wiki] inbox`) and
`[experiments] verdict_command`, if set, runs with the id (a repo, for a model) as its last
argument, e.g. a script that posts the verdict to your chat.

## Five rules for running an agent unattended

1. **The deadline is enforced from outside.** The runner gives the agent a time limit, kills its
   whole process group at the deadline whatever it is doing, and restores production itself.
   The agent is told the deadline, but nothing depends on it keeping to it.
2. **Restoring production is idempotent and runs every time.** `ensure_production` runs after
   every test, after a killed run, and again from a timer at the end of the window. It does
   nothing when production answers.
3. **Downloaded content is data, never instructions.** Model cards, READMEs and file names are
   read for facts only; no code from a model repo runs, nothing gets installed, safetensors builds
   that need `trust_remote_code` are refused, and model-written code runs in a sandbox without
   network or your files.
4. **One task per run, with recorded state.** One candidate per agent run; its status is in the
   store before the agent starts, and a run that ends without recording a result is marked
   `failed`. The agent's permissions cover exactly what the procedure needs.
5. **A human decides, the agent reports.** You pick what gets downloaded (a poll), the agent never
   switches production, every result is a committed JSON file plus a write-up, every commit
   passes the leak check, and each run's final message is kept in a log.

A wrapper around `[agent] command` may defer a run, for example to wait out a provider usage
limit, by exiting with `[agent] deferred_exit` (75, EX_TEMPFAIL). A deferred candidate goes back
to `downloaded` instead of `failed`, and the run stops taking further candidates until the next
nightly window, `bkg-start`, or the wrapper's replay.

## Run it on your own machine

You need Python 3.11+, the `hf` CLI, Claude Code (or another agent CLI with a compatible
`-p ... --output-format json` mode, set in `[agent] command`), and a serving repo that provides

```
bin/serve start  <model> [--port N] [--backend llamacpp|vllm] [--llama-cpp DIR] [-- extra args]
bin/serve stop   <model>|--all [--backend llamacpp|vllm]
bin/serve health <model> [--port N] [--backend llamacpp|vllm]
```

plus two production hooks: one that stops the production server and waits for free GPUs, and one
that brings it back if it is not answering (idempotent). This repo talks to nothing else of the
serving side, apart from the HTTP ports. The reference machine's serving repo is not public;
any wrapper around llama.cpp or vLLM that provides these three commands works.

```bash
git clone <this repo> && cd local-llm-eval
cp local.example.toml local.toml       # fill in paths, schedule, notify, publishers, machine
ln -sf ../../tools/leak_check.sh .git/hooks/pre-commit
python3 tools/agent_settings.py           # the nightly agent's permissions, from local.toml

python3 bench/bench.py --label my-model --base-url http://127.0.0.1:8080 --api-key-file key.txt
python3 bench/bench.py --label my-model --only judge,extract      # a short run
python3 pipeline/judge.py --dry-run                               # what the judge would see
python3 pipeline/fetch.py --dry-run                               # what would be downloaded
python3 pipeline/candidates.py add someone/Model-GGUF [--now]    # queue a repo for the test (--now: right after the download)
python3 pipeline/runner.py model-test --dry-run --at 01:00        # what tonight would test, and the prompt
```

Schedule the stages with systemd user timers or cron: judge in the morning, report after it,
fetch after the report, `runner.py model-test` at `schedule.window_start`, and the production
ensure hook at `schedule.window_end`. Reports go to an HTTP chat bridge, a shell command, or
stdout (`[notify]`). The judge can be a hosted model (`claude -p`) or the model you run in
production (`tools/openai_chat.py`, any OpenAI-compatible server). A bridge that cannot run this
repo, e.g. one in another container, can queue poll votes in a file for `candidates.py votes`.

## Extra sections: agent robustness and your own traffic

The core suite (`bench/cases.py`) is frozen, so every result already measured stays comparable. Newer sections live
next to it and are stored under each result file's `extra` key (synthetic: robust, longagent, estonian_free, vision) or `own_traffic` key (built from your agent's real history: replay, cron_replay, ctx), so the headline totals and the live board do not
change. Each has its own version in `SECTION_VERSIONS`. They run by default after the core suite; `--cases bench/cases.py` runs the core alone.

```
python3 bench/bench.py --label my-model                                                  # core + extra sections (the default)
python3 bench/bench.py --label my-model --cases bench/cases.py                           # the core suite alone
python3 bench/bench.py --label my-model --only robust,longagent,estonian_free,vision     # one section group
```

| Section | What it tests | Graded by |
|---|---|---|
| `robust` | Prompt injection (web page, tool field, file, mail, an exfiltration attempt), tool errors and retries, huge or truncated tool output, one tool among ~25 similar ones, destructive commands (look first, no force-push, ask what "old" means) and two over-correction controls | Calls made and the final answer |
| `longagent` | 15-30 call tasks in a real `bubblewrap` shell: fix bugs, trace a config value, rename across files, read gzip logs, find a commit with `git log -S`, leave `vendor/` alone, a bulk edit | The files at the end, by hidden tests |
| `estonian_free` | Free-form Estonian: chat, translation both ways, e-mail register, proofreading, English tool results answered in Estonian | [estonian-mcp](https://github.com/silly-geese/estonian-mcp) spelling, punctuation and compound checks, plus required facts. Skipped when it is not reachable |
| `vision` | Generated images: text, counting, a chart, a dialog, a table, an Estonian sign | Exact answer. Skipped when the server has no vision |
| `replay`, `cron_replay`, `ctx` | Your own agent transcripts (see below) | Next step agrees with the one that worked; the job's own output contract; the cost of a real ~15k-token prefix |

The last three read fixtures from `[bench] private_dir` (or `$BENCH_PRIVATE_DIR`). The fixtures are real transcripts and
**must not be published**: `tools/build_replay.py` builds them from a Hermes `state.db` into a folder that git ignores
(`state/private`), and their result rows are stored without any quoted text. Without the folder these three sections are
skipped. `python3 -m unittest tests.test_cases_v2 tests.test_cases_private` checks every grader offline: a scripted good run
must pass, a scripted bad run (following an injection, forcing a push, editing the tests) and a model that does nothing must fail.

## Results

Latest run per model on this repo's reference machine (2 × 16 GB GPUs). Graded = graded runs
passed (Estonian included, uncensored not); thinking off = the everyday sections with reasoning
disabled; prose = single-request generation speed. Totals compare only within one suite.
Write-ups are in [JOURNAL.md](JOURNAL.md), raw numbers in `results/`. The table is generated:
`python3 tools/results_table.py --write`.

<!-- results-table:start -->
| Model | Run | Date | Suite | Graded | Estonian | Uncensored | Thinking off | Prose tok/s |
|---|---|---|---|---:|---:|---:|---:|---:|
| vllm-qwen3.8-27b-heretic-ara-nvfp4-mtp | `vllm-qwen3.8-27b-heretic-ara-nvfp4-mtp_vllm0.32_tp2` | 2026-10-26 | 2026-10-18 | 277/291 | 38/45 | 11/11 | 110/141 | 58.4 |
| qwen3.8-27b-gsq-rco | `qwen3.8-27b-gsq-rco_iq3s_mtp_2gpu_128k` | 2026-10-26 | 2026-10-18 | 276/291 | 38/45 | 11/11 | 110/141 | 44.0 |
| rvn-qwen3.8-flash-next-abliterated | `rvn-qwen3.8-flash-next-abliterated_iq3xs_nospec_2gpu_128k` | 2026-10-25 | 2026-10-18 | 268/291 | 41/45 | 11/11 | 114/141 | 15.2 |
| capy-qwen3.8-27b-quasar-nvfp4 | `capy-qwen3.8-27b-quasar-nvfp4_capy0.7.4_tp2_efflow` | 2026-10-25 | 2026-10-18 | 280/291 | 40/45 | 10/11 | 120/141 | 77.0 |
| qwen3.8-27b-turbo-brilliance-50x | `qwen3.8-27b-turbo-brilliance-50x_iq4xs_mtp_2gpu_128k` | 2026-10-22 | 2026-10-18 | 284/291 | 42/45 | 10/11 | 118/141 | 37.7 |
| vllm-qwen3.8-27b-uncensored-hauhaucs | `vllm-qwen3.8-27b-uncensored-hauhaucs_vllm0.32_tp2` | 2026-10-21 | 2026-10-18 | 272/291 | 34/45 | 10/11 | 117/141 | 56.7 |
| nemotron-3.5-bolt-74b-a9b-crack-24gb | `nemotron-3.5-bolt-74b-a9b-crack-24gb_iq2xxs_dflash_2gpu_128k` | 2026-10-21 | 2026-10-18 | 167/291 | 8/45 | 4/11 | 80/141 | 65.6 |
| qwen3.8-27b-uncensored | `qwen3.8-27b-uncensored_iq4xs_mtp_2gpu_128k` | 2026-10-21 | 2026-10-18 | 276/291 | 36/45 | 11/11 | 117/141 | 44.8 |
| qwen3.8-27b-uncensored-hauhaucs | `qwen3.8-27b-uncensored-hauhaucs_iq4xs_mtp_2gpu_128k` | 2026-10-21 | 2026-10-18 | 282/291 | 40/45 | 11/11 | 117/141 | 43.5 |
| nvidia-nemotron-3.5-bolt-74b-a9b | `nvidia-nemotron-3.5-bolt-74b-a9b_iq2xxs_nospec_2gpu_128k` | 2026-10-21 | 2026-10-18 | 247/291 | 16/45 | 11/11 | 91/141 | 58.8 |
| qwen3.8-27b-heretic-ara-nvfp4 | `qwen3.8-27b-heretic-ara-nvfp4_vllm0.32_tp2` | 2026-10-21 | 2026-10-18 | 278/291 | 37/45 | 11/11 | 115/141 | 59.1 |
| ornith-1.5-35b-a3b-uncensored | `ornith-1.5-35b-a3b-uncensored_q4km_mtp_2gpu_128k_b11707` | 2026-10-21 | 2026-10-18 | 262/291 | 34/45 | 8/11 | 107/141 | 88.1 |
| orcasaq-2-cyber-27b-uncensored | `orcasaq-2-cyber-27b-uncensored_iq4xs_mtp_2gpu_128k` | 2026-10-21 | 2026-10-18 | 281/291 | 39/45 | 11/11 | 112/141 | 43.7 |
| qwen3.8-27b-nvfp4 | `qwen3.8-27b-nvfp4_vllm0.32_tp2` | 2026-10-21 | 2026-10-18 | – | – | – | – | 61.0 |
| muse-glimmer-30b-abliterated-2 | `muse-glimmer-30b-abliterated-2_q4km_dflash_2gpu_128k` | 2026-10-20 | 2026-10-18 | 264/291 | 41/45 | 11/11 | 133/141 | 35.1 |
| gemma4-26b-a4b-qat-uncensored-hauhaucs | `gemma4-26b-a4b-qat-uncensored-hauhaucs_q4km_mtp_2gpu_128k` | 2026-10-20 | 2026-10-18 | 251/291 | 35/45 | 11/11 | 118/141 | 82.7 |
| minicpm-v-4.7-35b-a3b-uncensored-3 | `minicpm-v-4.7-35b-a3b-uncensored-3_iq4xs_nospec_2gpu_128k` | 2026-10-20 | 2026-10-18 | 241/291 | 33/45 | 10/11 | 119/141 | 102.7 |
| underdog-saluki-27b-1.0-mtp-abliterated | `underdog-saluki-27b-1.0-mtp-abliterated_iq2mix_mtp_2gpu_128k` | 2026-10-20 | 2026-10-18 | 253/291 | 28/45 | 10/11 | 96/141 | 44.1 |
| qwen3.8-27b-rvn | `qwen3.8-27b-rvn_iq4xs_2gpu_128k_suite2026-10-18_b11707` | 2026-10-20 | 2026-10-18 | 283/291 | 41/45 | 11/11 | 113/141 | 45.7 |
| qwen3.8-27b-coder390-efficientthink | `qwen3.8-27b-coder390-efficientthink_q6k_mtp_2gpu_96k` | 2026-10-20 | 2026-10-18 | 232/243 | 35/45 | 11/11 | 116/141 | 31.5 |
| gemma4-31b-qat-uncensored-hauhaucs | `gemma4-31b-qat-uncensored-hauhaucs_q4km_nospec_2gpu_128k` | 2026-10-20 | 2026-10-18 | 233/243 | 38/45 | 11/11 | 118/141 | 19.4 |
| huihui-qwen3.8-27b-coder390 | `huihui-qwen3.8-27b-coder390_iq3s1_mtp_2gpu_128k_r2` | 2026-10-19 | 2026-10-18 | 273/291 | 35/45 | 11/11 | 115/141 | 44.4 |
| muse-glimmer-30b-abliterated | `muse-glimmer-30b-abliterated_q4km_dflash_2gpu_128k` | 2026-10-19 | 2026-10-18 | 266/291 | 41/45 | 7/7 | 131/141 | 36.1 |
| gemma4-26b-a4b-hauhau | `gemma4-26b-a4b-hauhau_q4km_mtp_2gpu_128k_suite2026-10-18` | 2026-10-19 | 2026-10-18 | 72/84 | 36/45 | 7/7 | 118/141 | – |
| minicpm-v-4.7-35b-a3b-uncensored-2 | `minicpm-v-4.7-35b-a3b-uncensored-2_iq4xs_nospec_2gpu_128k` | 2026-10-18 | 2026-10-18 | 235/291 | 30/45 | 7/7 | 126/141 | 99.9 |
| minicpm-v-4.7-35b-a3b-uncensored | `minicpm-v-4.7-35b-a3b-uncensored_iq4xs_nospec_2gpu_128k` | 2026-10-17 | – | 125/135 | – | 7/7 | – | 102.8 |
| muse-glimmer-30b-heretic | `muse-glimmer-30b-heretic_q4km_dflash_2gpu_128k` | 2026-10-17 | – | 123/135 | – | 4/7 | – | 36.3 |
| lfm2.5-24b-a2b-heretic | `lfm2.5-24b-a2b-heretic_qadq40_ngram_2gpu_128k` | 2026-10-17 | – | 119/135 | – | 4/7 | – | 150.1 |
| qwen3.8-27b-stock | `qwen3.8-27b-stock_iq4xs_CUDA0CUDA1_64k` | 2026-10-13 | – | – | – | – | – | 44.2 |
| qwen3.6-35b-a3b-heretic | `qwen3.6-35b-a3b-heretic_iq4xs_2gpu_128k` | 2026-10-13 | – | 125/135 | – | 5/7 | – | 142.6 |
<!-- results-table:end -->

## Layout

```
bench/       bench.py, cases.py (core), cases_v2.py (extra), cases_private.py (your traffic), harness.py; NIGHTLY.md, NIGHTLY_VLLM.md and EXPERIMENT.md, the agent's procedures
pipeline/    discover, judge, risk, candidates (store), experiments, fetch, runner, report, notify, config
prompts/     the judge prompt
results/     one JSON per benchmark run
tools/       leak_check.sh (pre-commit and CI), build_replay.py, results_table.py, agent_settings.py, openai_chat.py (judge command)
JOURNAL.md   test write-ups, newest first
```

## Licence

MIT, see [LICENSE](LICENSE). Model weights are not part of this repo; each has its own licence.
