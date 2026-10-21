# local-llm-eval

An unattended loop that looks for new open-weight models every day, downloads the promising ones,
benchmarks each against the model you run in production overnight, and reports in the morning.
A coding agent (Claude Code, headless) does the overnight test by following a written procedure;
everything around it is plain Python with no dependencies outside the standard library.

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
two alternating chats) and 97 graded cases × 3 runs in tool use, agent tasks, reasoning,
extraction, code (run in a bubblewrap sandbox), hard multi-step tasks, instruction following,
judging, Estonian, and an uncensored set of edgy-but-legitimate requests that a stock model
tends to refuse.

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
python3 pipeline/runner.py model-test --dry-run --at 01:00        # what tonight would test, and the prompt
```

Schedule the stages with systemd user timers or cron: judge in the morning, report after it,
fetch after the report, `runner.py model-test` at `schedule.window_start`, and the production
ensure hook at `schedule.window_end`. Reports go to an HTTP chat bridge, a shell command, or
stdout (`[notify]`).

## Results

Latest run per model on this repo's reference machine (2 × 16 GB GPUs). Graded = graded runs
passed (Estonian included, uncensored not); thinking off = the everyday sections with reasoning
disabled; prose = single-request generation speed. Totals compare only within one suite.
Write-ups are in [JOURNAL.md](JOURNAL.md), raw numbers in `results/`. The table is generated:
`python3 tools/results_table.py --write`.

<!-- results-table:start -->
| Model | Run | Date | Suite | Graded | Estonian | Uncensored | Thinking off | Prose tok/s |
|---|---|---|---|---:|---:|---:|---:|---:|
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
bench/       bench.py, cases.py, harness.py; NIGHTLY.md and NIGHTLY_VLLM.md, the agent's procedures
pipeline/    discover, judge, risk, candidates (store), fetch, runner, report, notify, config
prompts/     the judge prompt
results/     one JSON per benchmark run
tools/       leak_check.sh (pre-commit and CI), results_table.py
JOURNAL.md   test write-ups, newest first
```

## Licence

MIT, see [LICENSE](LICENSE). Model weights are not part of this repo; each has its own licence.
