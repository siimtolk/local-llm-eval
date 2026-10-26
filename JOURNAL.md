# Journal

Test write-ups, newest first. Each nightly test adds a section here (bench/NIGHTLY.md step 9);
the numbers behind every table are in `results/<label>.json`, summarised in the README table.
Imported from the serving repo's README on 2026-10-22. "The assistant" is the owner's everyday
chat assistant that uses the production model; `bin/...` paths refer to the serving repo.

## 2026-10-26 — qwen3.8-27b-gsq-rco (IQ3_S)

Question: does ISTA-DASLab's GSQ-RCO IQ3_S quant of Qwen3.8-27B (12.1 GB, embedded MTP head, BF16 mmproj) match production's IQ4_XS RVN?

Setup: `models.d/qwen3.8-27b-gsq-rco.conf.sh` derived from production's config: both GPUs, `--ctx-size 262144 --parallel 2`, q4_0 KV, `--spec-type draft-mtp`, Qwen sampling, `--reasoning-effort low`, llama.cpp-next. Labels `qwen3.8-27b-gsq-rco_iq3s_mtp_2gpu_128k` and `_nothink` (the template honoured the switch: answers got much shorter). Suite 2026-10-18, same as baseline.

| metric | candidate | production |
|---|---|---|
| graded runs passed (excl. uncensored) | 276/291 | 283/291 |
| uncensored | 11/11 (2 out of tokens) | 11/11 |
| Estonian | 38/45 | 41/45 |
| thinking off (tools, extract, instruct, judge, estonian) | 110/141 | 113/141 |
| prose / tool-JSON / 30k prompt tok/s | 44.0 / 67.0 / 1098 | 46.0 / 65.5 / 1213 |
| two at once, total | 61.9 (33.6 + 33.8) | 69 |
| follow-up time to first token | 1.1 s | 1.3 s |
| decode after 30k / 90k context | 35.2 / 27.2 tok/s (MTP 45% / 52%) | see depth results |
| suite wall-clock | 3690 s | n/a |

Per section: tools 40/42, agent 48/48, reason 30/30, extract 15/15, code 20/24 (2 out of tokens), hard 47/48, instruct 20/21, judge 18/18, estonian 38/45, uncensored 11/11. Thinking off: tools 39/42, extract 11/15, instruct 17/21, judge 18/18, estonian 25/45.

Takeaways: 7 fewer graded runs than production (limit is 3), code is the weak spot (eval_expr, next_cron), Estonian is weaker, and thinking-off Estonian drops to 25/45. Speed is no better than production. The smaller quant buys nothing visible. Verdict: rejected, weights deleted. Licence not checked.

## 2026-10-26 — experiment 20261026-ubatch-1024: Physical batch (ubatch) 1024

Hypothesis: a larger ubatch speeds up prompt reading without changing output.

Variant: production's config plus `--ubatch-size 1024` (production build, same weights, port 8085).
`serve` resolves weights as `models/<config name>`, so the variant config sets `MODELS_DIR` to a
symlink dir under `$SERVING/logs/exp-20261026-ubatch-1024-models/` (no writes to `models/`).

Verdict: **not-beneficial** (1/2 pass-rule conditions met), label `exp-20261026-ubatch-1024`.

| metric | variant | baseline |
|---|---|---|
| passed | 294 | 294 |
| tok_s.prose | 43.8 | 46.0 (ratio 0.95, needed 1.03) |
| tok_s.parallel | 39.0 | 38.2 |
| ttft.long_prompt (30k tokens) | 27.3 s | 24.8 s |

Checks: none in the spec. VRAM while serving: 12151 / 15403 / 1528 MiB (three GPUs).

Takeaways: output quality unchanged, but the larger ubatch made prompt reading slower
(~1100 vs ~1215 tok/s) and prose generation about 5% slower. Keep the default ubatch.

## 2026-10-25 — rvn-qwen3.8-flash-next-abliterated (IQ3_XS)

Question: can the 125B-total / ~6B-active MoE RVN Qwen3.8-Flash-Next (abliterated, 0bserverx, licence
qwen-community-1.0, no MTP head, no vision) run on 2 x RTX 5060 Ti + 78 GB RAM and match production?
Started by hand at 15:53 (outside the window); production was stopped from ~15:55 until restored after the nothink pass (~22:00).

Setup: llama.cpp-next, IQ3_XS (8 files, 86 GB: experts 54 GB, PLE table 29 GB), `--n-cpu-moe 30
--tensor-split 39,9`, `--ctx-size 262144 --parallel 2`, q4_0 KV, flash attention, `--reasoning-effort low`,
no speculation. Loaded in ~5 min. Label `rvn-qwen3.8-flash-next-abliterated_iq3xs_nospec_2gpu_128k`.

| | candidate | production |
|---|---|---|
| graded runs passed (excl. uncensored) | 268/291 | 283/291 |
| Estonian | 41/45 | 41/45 |
| uncensored | 11/11 | 11/11 |
| thinking off (tools, extract, instruct, judge, estonian) | 114/141 | 113/141 |
| prose / tool-JSON / prompt tok/s | 15.2 / 18.3 / 125 | 46.0 / 65.5 / 1213 |
| two at once (total) | 20.0 (10.5 + 10.5) | 69 |
| follow-up time to first token | 1.8 s | 1.3 s |
| decode after 30k / 90k | 14.9 / 14.0 | see depth file |
| suite wall-clock | 16065 s | 3666 s |

Per section: tools 41/42, agent 43/48, reason 30/30, extract 12/15, code 19/24 (3 out of tokens),
hard 44/48 (2 out of tokens; failed long_context_lookup, very_long_context), instruct 20/21, judge 18/18,
estonian 41/45, uncensored 11/11. Thinking off: tools 38/42, extract 11/15, instruct 17/21, judge 18/18,
estonian 30/45 (the switch works: much faster, but Estonian drops).

Takeaways: quality is a little below production (268 vs 283), speed is about 3x slower in decode and 10x
slower in prompt processing, because 30 blocks of experts and the PLE table live in RAM; the suite took 4.4 h
against 1 h. No advantage in context, vision or speed. Rejected; weights deleted.

## Test results (2026-10-25) — Qwen3.8-27B QUASAR-QAT NVFP4 on capy-flash (capy-qwen3.8-27b-quasar-nvfp4)

Question: is capy-flash (v0.7.4, commit 3193faf, a two-GPU tensor-parallel engine for Qwen3.8-27B on
RTX 50 cards) on these two RTX 5060 Ti clearly faster than production (llama.cpp b11707, RVN IQ4_XS,
MTP) at comparable quality, and what does it do to latency in real use? This was a measurement only;
production was not switched. The owner started the test by hand at 04:01, outside the nightly
window. Production was stopped from 04:02 to 05:01. Model: `Feyd89/Qwen3.8-27B-QUASAR-QAT-nvfp4-NInfer`
(`.ninfer` v3, 17.4 GB: stock Qwen3.8-27B, QUASAR-QAT's quantization-aware NVFP4 projections, FP8
head and embedding, MTP and vision included, no DFlash2 drafter; Apache 2.0). It is not
uncensored.

Setup: `bin/serve start capy-qwen3.8-27b-quasar-nvfp4 --backend capy --port 8085`, which runs
`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 capy-flash/current/serve-tp2.sh
models/capy-qwen3.8-27b-quasar-nvfp4/qwen3_8_27b_quasar_nvfp4.ninfer --api-key-file secrets/qwen3-8-27b-rvn/api_key.txt`
(`ninfer-serve --profile 27b`: TP=2 on devices 0,1, p2p off with host-staged copies, INT8 KV,
`--max-context 196608`, `--max-concurrency 1`, `--spec mtp --draft-tokens 3`, vision on device 0,
4 device states plus 8 host states and 8 GiB of host KV for the context cache). It started at full
defaults on the first try with no step-downs: 1m08s to ready, weights 8.66 / 8.38 GiB per rank,
"KV 196,608 tokens, int8 | runtime 3.93 GiB | free 2.81 GiB". The context that fit is
**1 x 196,608 tokens**. With `-- --max-concurrency 2`, two requests share the same 196,608-token
pool. Sanity checks passed: reasoning is split from content, the `get_weather` tool call parses,
`enable_thinking: false` gives no reasoning. Requests must send `model: qwen3.8-27b`.

Reasoning effort: production runs `--reasoning-effort low`. capy-flash has no server default for
effort; the template's default is full thinking. The template does honour the request's
`reasoning_effort: low`: a semver prompt took 5955 tokens at the default and 1607 at low. So the
suite ran twice:
- At the server default (full thinking), label `capy-qwen3.8-27b-quasar-nvfp4_capy0.7.4_tp2`. This
  run is from 2026-10-24 19:22–20:21: an earlier start had completed the whole suite but nothing
  after it, and it was reused as is.
- With `--reasoning-effort low`, label `..._tp2_efflow`, run tonight. This is the run comparable
  with production.

capy rejects `reasoning_effort` together with `enable_thinking: false` (HTTP 400 "conflicting
template option"), so the thinking-off passes send no effort. Two speed-only passes followed:
`..._tp2_conc2_efflow` (`-- --max-concurrency 2`) and `..._tp2_conc2_preservethink_efflow` (also
`--preserve-thinking`). `--spec dflash2` was not tried: it needs a DFlash2 drafter, and this
artifact has none.

| | Qwen3.8-27B RVN, llama.cpp (prod) | capy-flash, effort low | capy-flash, server default (full thinking) | vLLM 0.32, Heretic ARA NVFP4 (2026-10-21) |
|---|---|---|---|---|
| Context | 2 x 128k (q4_0 KV) | 1 x 196,608 (int8 KV) | same | 2 x 128k (int4 KV) |
| Graded runs passed | **283 / 291** | 280 / 291 | 272 / 291 | 278 / 291 |
| of which Estonian | **41 / 45** | 40 / 45 | **41 / 45** | 37 / 45 |
| Uncensored answered | **11 / 11** | 10 / 11 (torrents out of tokens) | 6 / 11 (5 out of tokens) | 11 / 11 |
| Out of tokens (graded) | 2 | 3 | 14 | |
| Graded suite, wall-clock and tokens | 54.5 min, 150k | **34.5 min**, 155k | 52.4 min, 229k | |
| Thinking off (tools, extract, instruct, judge, Estonian) | 113 / 141 | **117 and 120 / 141** (two runs) | | 115 / 141 |
| Prose generation | 46.0 tok/s (MTP 54%) | **77.0** (46%); 82.2 at concurrency 2 | 78.3 (49%) | 59.1 |
| Tool-call JSON generation | 65.5 tok/s (92%) | **121.5** (91%) | 120.0 (91%) | 89.1 |
| Prompt processing (30k) | 1213 tok/s | **2855 tok/s** | 2833 | 1161 |
| Decode after ~30k / ~90k of context | 39.0 / 26.5 tok/s | **78.3 / 68.8** (MTP 51 / 48%) | 72.2 / 76.3 | |
| Two requests at once (total) | 69 tok/s (38 + 38) | 88.1 at concurrency 1 (queued: second starts after 6.1 s); **147.8** (79.3 + 75.3) at concurrency 2 | 88.0 (queued) | 116.6 |
| Alternating chats: first turn / follow-up time to first token | 10.3 / **1.3 s** | **4.1** / 4.2 s; 4.1 / 2.2 s at concurrency 2 | 4.0 / 4.2 s | 8.5 / 2.9 s |
| VRAM, both 5060 Ti | 25.5 GB | 25.9 GB (13,071 + 12,789 MiB, fixed at start) | same | 30.2 GB |

Per section (passed / runs):

| Section | RVN (prod) | capy, effort low | capy, default | capy, thinking off (run 1 / run 2) | RVN, thinking off |
|---|---|---|---|---|---|
| tools | 42/42 | 42/42 | 42/42 | 39/42 / 39/42 | 38/42 |
| agent | 47/48 | 46/48 | **48/48** | | |
| reason | 30/30 | 30/30 | 30/30 | | |
| extract | **15/15** | 14/15 | 13/15 | 12/15 / 12/15 | 12/15 |
| code | 21/24 | 21/24 | 12/24 | | |
| hard | 48/48 | 48/48 | 47/48 | | |
| instruct | 21/21 | 21/21 | 21/21 | 15/21 / **19/21** | 16/21 |
| judge | 18/18 | 18/18 | 18/18 | 18/18 / 18/18 | 18/18 |
| estonian | **41/45** | 40/45 | **41/45** | **33/45** / 32/45 | 29/45 |

**Takeaways:**
- At effort low, quality is close to production: 280 vs 283 graded. The misses were
  remembered_pref twice (agent), release_name (extract), eval_expr and next_cron (code, 3 out of
  tokens), and in Estonian et_jalg, et_number_words and et_tutar. At the server default the model
  thinks until the 6000-token cap: code falls to 12/24 (12 out of tokens), and 5 uncensored cases
  run out of tokens. Those 5 are token-cap failures, not refusals. Production's assistant would
  have to send `reasoning_effort: low` with every request; `--default-thinking-budget` is the
  server-side alternative, and it was not tested.
- At effort low there was one uncensored miss, torrents, which ran out of tokens; no outright
  refusal was seen. The weights are stock Qwen3.8 and not abliterated, so that result is not
  guaranteed.
- Speed is the strength. Single-request generation is 1.7x production (77 vs 46 prose, 121 vs
  65 tool JSON). Prefill is 2.4x. Decode holds up deep in context, 69 tok/s after 90k against
  production's 26.5, about 2.6x. Two at once is 148 tok/s with `--max-concurrency 2`, about 2.1x.
  It also beats vLLM on every speed row, at 4 GB less VRAM.
- Latency in real use: the 27b profile serves one request at a time, so a second request waits
  for the first (6.1 s here). The bigger cost is follow-ups. An alternating chat's follow-up took
  4.2 s against production's 1.3 s: capy re-reads the whole 13k history. `--prefill-miss-dump`
  shows why. The follow-up matches the cached prompt for all 12,906 tokens up to `<think>`, then
  diverges, because the template drops the earlier turn's thinking while the cache holds prompt +
  thinking. The first follow-up of each chat therefore misses. From the second one on, a "long
  anchor" hits (99.5%, 0.27 s). That is why the average is 2.2 s at concurrency 2. At concurrency
  1 every follow-up missed. `--preserve-thinking` changed nothing, since bench.py does not send
  the reasoning back.
- Thinking off works, at 117 and 120 / 141 against production's 113, and is better in Estonian
  (33 vs 29).
- Verdict: **tested** (worth a live trial; the weights and the release stay). On the run
  comparable with production (effort low) it meets every point of the capy bar: graded 280 >= 278,
  Estonian 40 >= 38, clearly faster (prose 1.7x, two at once 2.1x), at least 1 x 128k context
  (196,608). On the server's default reasoning it would be rejected (272/291). A live trial needs
  effort low set per request (or a thinking budget). It also has to accept slower follow-ups,
  single-request serving (or a shared 196k pool at concurrency 2), and stock-model censorship.

**For the capy-flash thread** (2 x RTX 5060 Ti 16 GB, PCIe, no P2P (host-staged copies), Fedora,
release v0.7.4, profile `27b`, Feyd89's QUASAR-QAT NVFP4 artifact, MTP 3, INT8 KV, 196,608-token
context with vision):
- Decode: about 77–82 tok/s on prose and about 120 tok/s on tool-call JSON. MTP acceptance was
  ~46–50% on prose and ~91% on JSON.
- Decode deep in context: 78–82 tok/s after 30k tokens and 69–76 tok/s after 90k tokens. llama.cpp
  on the same cards gives 39 and 27.
- Prefill: about 2,850 tok/s on a 30k-token prompt; a 91k-token prompt took 46.7 s (~1,960 tok/s).
- Two requests at once: 148 tok/s total (79 + 75) with `--max-concurrency 2`. The stock profile
  queues the second request.
- Follow-up latency: in alternating 13k-token chats, the first follow-up of each conversation
  re-prefills (about 4 s) because the template drops the previous turn's thinking. Later
  follow-ups hit a long anchor, about 0.27 s to first token.
- VRAM: 13.1 + 12.8 GB. Load time: about 68 s from NVMe.
- On the same cards llama.cpp with MTP does 46 tok/s, so this is about 1.7x single-request and
  2.1x with two at once. The author's 175 tok/s on 2 x 5070 Ti fits roughly twice the memory
  bandwidth.

**Follow-up 2026-10-25 (production's shape, speed pass only):** `-- --max-context 262144 --max-concurrency 2`
(a 256k pool for two lanes, like production's 2 x 128k), effort low, label
`capy-qwen3.8-27b-quasar-nvfp4_capy0.7.4_tp2_conc2_ctx262144_efflow`. It started on the first try:
"KV 262,144 tokens, int8 | runtime 5.11 GiB | free 1.63 GiB", VRAM 14,275 + 13,991 MiB. Prose 78.3 tok/s,
tool JSON 121.3, 30k prompt 2,851 tok/s, two at once 145.4 (76.0 + 76.9), decode after 30k / 90k 76.0 / 68.2
(MTP 49 / 47%), alternating chats 4.1 s first turn and 4.2 s follow-ups. The speed matches the 196k run; the
follow-up average (2.2 s at 196k, concurrency 2) did not: one run, so noise is not ruled out.

## Test results (2026-10-22) — Qwen3.8-27B Turbo Brilliance 50X NEO MAX MTP (DavidAU)

Question: is DavidAU's "Turbo Brilliance" Qwen3.8-27B (a lightly tuned dense Qwen3.8-27B with
prompt-activated `{REASON:<mode>}` reasoning/instruct modes; card: "not a heretic", i.e. not
abliterated; NEO imatrix, "MAX" = BF16 output tensor and Q8 MTP tensors; Apache 2.0) better
than RVN? Started by hand by the owner at 13:12, outside the nightly window; production was
stopped 13:13–14:29. Setup (`models.d/qwen3.8-27b-turbo-brilliance-50x.conf.sh`): IQ4_XS
(17.0GB, MTP head embedded), the repo's F16 projector on CUDA2, `llama.cpp-next` b11707, both
RTX 5060 Ti, 2 x 128k context, q4_0 KV, `--spec-type draft-mtp`, the card's tester sampling
(temp 0.7, top-p 0.95, top-k 64, min-p 0.05), `--reasoning-effort low`. The special modes were
not used; the suite runs the model as the assistant would, without them. Label
`qwen3.8-27b-turbo-brilliance-50x_iq4xs_mtp_2gpu_128k`, suite 2026-10-18.

| | Qwen3.8-27B RVN (prod) | Turbo Brilliance 50X |
|---|---|---|
| Quant, context | IQ4_XS (15.5GB), 2 x 128k | IQ4_XS MAX (17.0GB), 2 x 128k |
| Graded runs passed | 283 / 291 | **284 / 291** |
| of which Estonian | 41 / 45 | **42 / 45** |
| Uncensored prompts answered | **11 / 11** (1 hit the token cap) | 10 / 11 (refused paywall; 1 hit the cap) |
| Out of tokens (graded) | 2 | 1 |
| Wall-clock, tokens, graded suite | **54.5 min**, 150k | 58.7 min, **137k** |
| Thinking off (tools, extract, instruct, judge, estonian) | 113 / 141 (4.5 min) | **118 / 141** (4.7 min) |
| Prose generation | **46.0 tok/s** (MTP acceptance 54%) | 37.7 tok/s (54%) |
| Tool-call JSON generation | **65.5 tok/s** (92%) | 53.1 tok/s (93%) |
| Prompt processing (30k tokens) | 1213 tok/s | 1217 tok/s |
| Two requests at once (total) | **69 tok/s** (38 + 38) | 57.7 tok/s (31.6 + 31.4) |
| Alternating chats, follow-up time to first token | 1.3 s | 1.2 s |
| VRAM, both 5060 Ti | **25.5 GB** | 26.9 GB |

Per section (passed / runs):

| Section | RVN (prod) | Turbo Brilliance | Turbo Brilliance, thinking off | RVN, thinking off |
|---|---|---|---|---|
| tools | 42/42 | 42/42 | **39/42** | 38/42 |
| agent | **47/48** | 45/48 | | |
| reason | 30/30 | 30/30 | | |
| extract | 15/15 | 15/15 | 11/15 | **12/15** |
| code | 21/24 | **23/24** | | |
| hard | 48/48 | 48/48 | | |
| instruct | 21/21 | 21/21 | **20/21** | 16/21 |
| judge | 18/18 | 18/18 | 18/18 | 18/18 |
| estonian | 41/45 | **42/45** | **30/45** | 29/45 |

**Takeaways:**
- Quality is production's: 284 vs 283 graded runs. Misses: remembered_pref all three times in
  agent (found the right 1080p EST release but asked "Want me to start the download?" instead
  of adding it), next_cron once (out of tokens at 6000), Estonian "tuba" as its own genitive
  and "kaheksakümne seitse" twice for the number words.
- Not uncensored: it refused paywall ("I can't help with bypassing paywalls..."), as the card
  says (not abliterated). That alone fails the bar.
- About 18% slower generation at the same MTP acceptance, likely from the BF16 output tensor
  (the "MAX" part) read on every token; two at once 58 vs 69 tok/s, +1.4GB VRAM.
- Thinking off works (the template honours `enable_thinking`) and is 5 runs better than
  production, mostly in instruct; Estonian stays weak without thinking.
- Nothing clearly better, slower, one refusal: rejected, weights deleted. Apache 2.0.

## Test results (2026-10-22) — RVN Qwen3.8-Flash-Next Abliterated (0bserverx), not benchmarked

Question: does 0bserverx's abliterated Qwen3.8-Flash-Next (hybrid SSM/attention MoE, GGUF arch
`qwen4exp`: 48 blocks, full attention every 4th, 512 experts / 10 active + 1 shared, ~125B
total, ~6B active) run on this machine with experts offloaded to RAM? Started by hand at 01:37.
Setup (`models.d/rvn-qwen3.8-flash-next-abliterated.conf.sh`): IQ3_XS (8 split files),
`llama.cpp-next` (1268edac5, knows `qwen4exp`), both RTX 5060 Ti, `--n-cpu-moe 30`,
`--tensor-split 39,9` (the default split put all 18 GPU expert blocks on CUDA1: 21GB, out of
memory), 2 x 128k, q4_0 KV, temp 1.0 / top-p 0.95 / top-k 20 (GGUF metadata),
`--reasoning-effort low`.

The download is 86.0GB, not the ~55GB the fit rule expected from 125B at IQ3: experts 54.4GB,
other weights 2.8GB, and a **28.8GB per-layer n-gram embedding table** (`per_layer_token_embd`,
320M rows x 160, IQ4_NL) that the parameter count does not include. That is above VRAM + RAM
(32 + 46GB). It loaded (5 min; VRAM 15.4 + 11.1GB) and answered correctly, but the ~34GB of
experts in RAM plus the table do not fit the ~40GB of free RAM, so it pages from the SATA SSD:

| | Qwen3.8-27B RVN (prod) | RVN Flash-Next IQ3_XS |
|---|---|---|
| Prompt processing | 1213 tok/s (30k prompt) | 0.5 tok/s cold, 2.3 tok/s warm (short prompts) |
| Generation | 46.0 tok/s | 1.3 tok/s cold, 1.5 tok/s warm |

**Takeaways:**
- At 1.5 tok/s the graded suite (~150k tokens for production) would take more than a day, so
  the benchmark was not run. Production was back at 01:56.
- Skipped: it does not fit this machine. The table alone is 28.8GB; no split of the 86GB fits
  32GB VRAM + 46GB RAM. The weights are kept (skipped, not rejected); 86GB to free by hand.
- For the fit rule (`pipeline/params.py`, `discover.py`): models with per-layer embeddings
  (`ple` / `embedding_length_per_layer_input`) need the table counted on top of the parameter
  count; checking the repo's file sizes before downloading would have caught this.
- Licence qwen-community-1.0.

## Switching the backend to vLLM (summary, 2026-10-21)

Status: **not switched; production stays on llama.cpp.** Two experiments ran vLLM 0.32
(`bin/serve_vllm.sh`, TP=2 on both RTX 5060 Ti, NCCL over PCIe since the cards have no P2P)
against production (Qwen3.8-27B RVN, llama.cpp b11707). Full results are in the sections
"vLLM 0.32 + NVFP4 (experiment)" and "vLLM 0.32 + Heretic ARA NVFP4 (experiment 2)" below.

| | llama.cpp (prod) | vLLM, stock NVFP4 (exp. 1) | vLLM, Heretic ARA NVFP4 (exp. 2) |
|---|---|---|---|
| Context | 2 x 128k | 1 x 128k + 0.4 | **2 x 128k** (int4 KV, 3.0x) |
| Prose / tool JSON | 46.0 / 65.5 tok/s | 61.0 / 81.3 | **59.1 / 89.1** |
| Two requests at once | 69 tok/s | **120** | 116.6 |
| 30k prompt processing | **1213 tok/s** | 1142 | 1161 |
| Follow-up time to first token | **1.3 s** | 3.0 s | 2.9 s |
| VRAM | **25.5 GB** | 27.3 GB | 30.2 GB |
| Graded / Estonian | **283/291, 41/45** | not run | 278/291, 37/45 |
| Uncensored / thinking off | 11/11, 113/141 | not run | 11/11, **115/141** |

What vLLM gains: about 1.3x faster single-request generation, up to 1.36x on tool-call JSON and
1.7x total throughput with two requests at once (MTP acceptance ~75%). What it loses: 4 points
on Estonian (the deciding miss; the bar was 38/45), slightly slower prompt processing, more
than twice the follow-up latency of alternating chats (likely weaker cache reuse than
llama.cpp's slot cache), ~5 GB more VRAM and a ~5.5–7 min first start. Context was the blocker
in experiment 1 and was solved in experiment 2 by `--language-model-only` plus int4 KV.

When to revisit: an RVN (or other Estonian-strong) uncensored NVFP4 checkpoint appears, a
newer vLLM improves prefix-cache latency, or parallel load (several assistant jobs at once)
becomes common enough that the two-at-once throughput matters more than Estonian quality.
The bar from experiment 2 still applies: graded >= 278/291, Estonian >= 38/45, uncensored
>= 10/11 and clearly faster. A switch would also need a vLLM systemd unit in place of
`serve-llm.service`, with `ensure_production.sh`, llm-guard and the Claude fallback's port
handover adapted to it. The owner decides; the test runs never switch production.

Since 2026-10-21 the nightly test has a vLLM lane. When `pipeline/fetch.py` downloads
a GGUF candidate, it searches Hugging Face once for a 4-bit safetensors build of the same model
(NVFP4 first, then AWQ/GPTQ; its `base_model` must be the GGUF repo or the same non-official
model, no repo code, weights up to 22 GB) and queues it as its own candidate (`backend: vllm`,
priority -1). The runner tests it after the night's GGUF candidates, when 95 min are left, with
[`bench/NIGHTLY_VLLM.md`](bench/NIGHTLY_VLLM.md) and the experiment-2 bar above.


## Test results (2026-10-21) — Qwen3.8-27B Uncensored HauhauCS Aggressive AWQ on vLLM

Question: does the vLLM twin of HauhauCS's uncensored Qwen3.8-27B (tested on llama.cpp earlier
today: 282/291, rejected as production's twin) match production's quality on vLLM while being
clearly faster? Checkpoint `twolven/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-AWQ-MTP`:
compressed-tensors `pack-quantized` int4 W4A16 (asymmetric, group 128; linear-attention
projections, norms and vision tower left unquantized), 18.7 GB, plus a separate BF16 MTP file
(`model-mtp.safetensors`, 0.85 GB, 15 tensors). Measurement only; production was not switched.
Started by hand by the owner at 17:42, outside the nightly window; production was stopped
17:43–18:59.

Setup: `vllm/vllm-openai:v0.32.0`, `bin/serve_vllm.sh` (TP=2 on both RTX 5060 Ti, max-num-seqs 2,
prefix caching, int4_per_token_head KV, reasoning parser qwen3, effort low by default), no
`--trust-remote-code`. The experiment-2 flags without `--linear-backend flashinfer_cutlass`
(NVFP4 only); vLLM picked `MarlinLinearKernel` for the W4A16 layers. Label
`vllm-qwen3.8-27b-uncensored-hauhaucs_vllm0.32_tp2`.

```
VLLM_MODEL_DIR=models/vllm-qwen3.8-27b-uncensored-hauhaucs \
VLLM_SERVED_NAME=vllm-qwen3.8-27b-uncensored-hauhaucs \
VLLM_EXTRA_ENV="VLLM_USE_FLASHINFER_SAMPLER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
./bin/serve_vllm.sh --language-model-only --enable-chunked-prefill \
    --tool-call-parser qwen3_xml \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'
```

- Context: weights 8.71 GiB per GPU, 4.96 GiB KV per GPU, **431,633 tokens, "maximum
  concurrency 3.29x" at 128k**: 2 x 128k fits on the first try. First start ~4.5 min.
- Sanity: reasoning split from content, `qwen3_xml` tool call parsed (`{"city": "Tartu", "unit": "c"}`),
  `enable_thinking: false` gives 0 reasoning tokens.
- MTP acceptance over the whole run 72.7% (141.7k of 195k drafted tokens).

| | Qwen3.8-27B RVN, llama.cpp b11707 (prod) | HauhauCS IQ4_XS, llama.cpp (GGUF twin) | HauhauCS AWQ, vLLM 0.32 TP=2 |
|---|---|---|---|
| Context | 2 x 128k (q4_0 KV) | 2 x 128k | **2 x 128k** (int4 KV, 432k tokens, 3.29x) |
| Graded (97 cases x 3) | **283/291** | 282/291 | 272/291 |
| Estonian | **41/45** | 40/45 | 34/45 |
| Uncensored | 11/11 | 11/11 | 10/11 (3 out of tokens, invest empty at 2000; no refusals) |
| Out of tokens (graded) | 2 | 2 | 1 |
| Graded suite, summed time and tokens | **54.5 min, 150k** | 51.1 min, 142k | 58.6 min, 175k |
| Thinking off (tools, extract, instruct, judge, Estonian) | 113/141 | **117/141** | **117/141** |
| Prose generation | 46.0 tok/s | 43.5 tok/s | **56.7 tok/s** (1.23x) |
| Tool-call JSON generation | 65.5 tok/s | 65.0 tok/s | **90.2 tok/s** |
| Prompt processing (30k prompt) | **1213 tok/s** | 1235 tok/s | 752 tok/s |
| Two requests at once (total) | 69 tok/s | 65 tok/s | **97.5 tok/s** (57.4 + 63.2) |
| Alternating chats, follow-up time to first token | **1.3 s** | 1.2 s | 4.6 s (first turn 14.6 s) |
| VRAM (5060 Ti x 2) | **25.5 GB** | 25.8 GB | 30.1 GB (15.0 GB each) |

Graded by section (vLLM AWQ vs prod): tools 42 vs 42, agent 44 vs 47 (remembered_pref 3x,
suggest_only), reason 30 vs 30, extract 15 vs 15, code 22 vs 21 (next_cron 2x), hard 46 vs 48
(bins, very_long_context), instruct 21 vs 21, judge 18 vs 18, Estonian 34 vs 41 (et_genitive 3x
"tubal"/"tuba"/"tubul", et_tutar 3x "tütara"/"tütra"/"tütrel", et_illative 2x "maja", et_jalg
"jalki", et_number_words, et_light). Thinking off: tools 40, extract 12, instruct 18, judge 18,
Estonian 29.

Takeaways:
- The same model scores 282/291 and Estonian 40/45 as IQ4_XS on llama.cpp, but 272 and 34 here:
  the AWQ int4 quant (or vLLM's path) costs about 10 graded runs, 6 of them Estonian case forms,
  plus agent memory recall (remembered_pref 0/3).
- Speed follows the vLLM pattern: faster decoding (prose 1.23x, tool JSON 1.38x, two at once 1.41x),
  but the W4A16 Marlin path is much slower at prompt processing than NVFP4 was (752 vs 1161
  tok/s, 0.62x production), so the 13k-token alternating chats wait 14.6 s / 4.6 s for the first
  token.
- Verdict: **rejected**. Bar: graded >= 278 (272, missed), Estonian >= 38 (34, missed),
  uncensored >= 10 (10, met, no refusals), clearly faster (generation yes, prompt processing no).
  Weights deleted.

## Test results (2026-10-21) — Nemotron 3.5 Bolt 74B-A9B CRACK-24GB (dealignai)

Question: does a second refusal-removed build of the same 74B hybrid Mamba/MoE (88 blocks:
40 Mamba2, 40 latent-MoE, 8 attention; ~9B active) do better than BoldingBuilds' one (tested
earlier today, 247/291)? dealignai's CRACK removes refusals and then requantizes to NVIDIA's
24GB layout with its own imatrix (card: HarmBench-320 0 refusals, MMLU 80.8% -> 78.6% against
NVIDIA's 24GB GGUF; artifacts above temperature 0.7, replies 2-3x longer than base). The only
quant (22.0GB, 2.38 BPW): experts IQ2_XXS, Mamba in/out NVFP4, attention and latent Q8_0.
Started by hand in the daytime (15:52). Both RTX 5060 Ti, 2 x 128k context, q4_0 KV cache,
NVIDIA's LiLiCorr DFlash drafter from the repo (1.0GB, `--spec-draft-n-max 6` as on the
card), temp 0.7 / top-p 0.95 (the card's ceiling for clean output), the `medium_effort`
template switch as in the BoldingBuilds config, projector on CUDA2, `llama.cpp-next` b11707.
Full suite 2026-10-18 plus the thinking-off pass. Label
`nemotron-3.5-bolt-74b-a9b-crack-24gb_iq2xxs_dflash_2gpu_128k`; production is RVN on b11707.

| | Qwen3.8-27B RVN (prod) | Nemotron Bolt CRACK-24GB | Nemotron Bolt, BoldingBuilds |
|---|---|---|---|
| Quant, context | IQ4_XS (15.5GB), 2 x 128k | IQ2_XXS mix (22.0GB), 2 x 128k | IQ2_XXS mix (22.0GB), 2 x 128k |
| Graded runs passed | **283 / 291** | 167 / 291 | 247 / 291 |
| of which Estonian | **41 / 45** | 8 / 45 | 16 / 45 |
| Uncensored prompts answered | **11 / 11** | 4 / 11 (no refusals; 7 ran out of tokens while thinking) | 11 / 11 |
| Out of tokens (graded) | 2 | 65 | **0** |
| Wall-clock, tokens, graded suite | 54.5 min, 150k | 91.5 min, 683k | **40.9 min, 117k** |
| Thinking off (tools, extract, instruct, judge, estonian) | **113 / 141** (4.5 min) | 80 / 141 (4.0 min) | 91 / 141 (2.5 min) |
| Prose generation | 46.0 tok/s (MTP 54%) | **65.4 tok/s** (DFlash 24%) | 59.0 tok/s (no spec) |
| Tool-call JSON generation | 65.5 tok/s (92%) | **101.7 tok/s** (47%) | 58.8 tok/s |
| Prompt processing (30k tokens) | 1213 tok/s | 1020 tok/s | **1631 tok/s** |
| Two requests at once (total) | 69 tok/s (38 + 38) | 71 tok/s (46 + 38) | **78.5 tok/s** (46 + 45) |
| Alternating chats, follow-up time to first token | 1.3 s | 1.3 s | **1.2 s** |
| VRAM, both 5060 Ti | 25.5 GB | 26.7 GB | **23.8 GB** |

Per section (passed / runs):

| Section | RVN (prod) | CRACK-24GB | CRACK-24GB, thinking off | RVN, thinking off |
|---|---|---|---|---|
| tools | **42/42** | 41/42 | 38/42 | 38/42 |
| agent | **47/48** | 29/48 | | |
| reason | **30/30** | 27/30 | | |
| extract | **15/15** | 9/15 | 9/15 | 12/15 |
| code | **21/24** | 3/24 | | |
| hard | **48/48** | 21/48 | | |
| instruct | **21/21** | 12/21 | 11/21 | 16/21 |
| judge | **18/18** | 17/18 | 18/18 | 18/18 |
| estonian | **41/45** | 8/45 | 4/45 | 29/45 |

**Takeaways:**
- The thinking does not stop: 65 graded runs and 7 of the 11 uncensored prompts hit the token
  cap with an empty answer, the reasoning circling ("Need produce final now. Ensure code
  block only.") until the limit. That costs code (20 of 24 out of tokens), hard (15), Estonian
  (18) and instruct (8), and takes 4.5x production's tokens. The card's own warning about
  2-3x longer replies, on top of 2-bit experts.
- Even where it finishes, it is weaker than the BoldingBuilds build of the same base: agent
  29/48 (torrent_pick, duplicates, reconcile, fix_bug every time), extract filename and
  release_name every time, and Estonian worse still (8/45; 4/45 with thinking off).
- Thinking off works (the template honours `enable_thinking`) but is far behind: 80/141 vs 113.
- Speed: the DFlash drafter helps, 1.4x production on prose and 1.6x on tool JSON, but prompt
  processing is lower (1020 tok/s) and two at once is level. Both slots keep their cache.
- 116 graded runs behind production, 7/11 uncensored. NVIDIA Open Model License. Rejected;
  weights deleted.

## Test results (2026-10-21) — Qwen3.8-27B Uncensored (JonathanColetti)

Question: is JonathanColetti's Heretic abliteration of the same dense Qwen3.8-27B that
production runs (card: 12/100 refusals on harmful_behaviors in non-thinking mode, first-token
KL 0.119 vs base, abliterated at bf16, imatrix from wikitext-2, Apache 2.0) better than RVN?
Same quant as production (IQ4_XS, 15.3GB), the "fused" file with the MTP head embedded (the
repo's separate draft-Q4_0 head is for its noMTP files and was not used). Started by hand in
the daytime (14:48). Both RTX 5060 Ti, 2 x 128k context, q4_0 KV cache,
`--spec-type draft-mtp` (default draft length), stock Qwen3.8 sampling (temp 1.0, top-p 0.95,
top-k 20; the card gives none), `--reasoning-effort low`, the repo's F16 projector on CUDA2,
`llama.cpp-next` b11707. Full suite 2026-10-18 plus the thinking-off pass. Label
`qwen3.8-27b-uncensored_iq4xs_mtp_2gpu_128k`; production is RVN on b11707 (2026-10-20).

| | Qwen3.8-27B RVN (prod) | Qwen3.8-27B Uncensored (JonathanColetti) |
|---|---|---|
| Quant, context | IQ4_XS (15.5GB), 2 x 128k | IQ4_XS (15.3GB), 2 x 128k |
| Graded runs passed | **283 / 291** | 276 / 291 |
| of which Estonian | **41 / 45** | 36 / 45 |
| Uncensored prompts answered | 11 / 11 (1 hit the token cap) | 11 / 11 (2 hit the cap) |
| Out of tokens (graded) | 2 | 2 |
| Wall-clock, tokens, graded suite | 54.5 min, 150k | **49.1 min, 137k** |
| Thinking off (tools, extract, instruct, judge, estonian) | 113 / 141 (4.5 min) | **117 / 141** (3.8 min) |
| Prose generation | 46.0 tok/s (MTP acceptance 54%) | 45.5 tok/s (51%) |
| Tool-call JSON generation | 65.5 tok/s (92%) | 64.9 tok/s (87%) |
| Prompt processing (30k tokens) | 1213 tok/s | 1212 tok/s |
| Two requests at once (total) | 69 tok/s (38 + 38) | 70 tok/s (39 + 41) |
| Alternating chats, follow-up time to first token | 1.3 s | 1.2 s |
| VRAM, both 5060 Ti | 25.5 GB | 25.3 GB |

Per section (passed / runs):

| Section | RVN (prod) | JonathanColetti | JonathanColetti, thinking off | RVN, thinking off |
|---|---|---|---|---|
| tools | **42/42** | 41/42 | **40/42** | 38/42 |
| agent | 47/48 | 47/48 | | |
| reason | 30/30 | 30/30 | | |
| extract | **15/15** | 14/15 | 11/15 | **12/15** |
| code | 21/24 | **22/24** | | |
| hard | 48/48 | 48/48 | | |
| instruct | **21/21** | 20/21 | **20/21** | 16/21 |
| judge | 18/18 | 18/18 | 18/18 | 18/18 |
| estonian | **41/45** | 36/45 | 28/45 | **29/45** |

**Takeaways:**
- 7 graded runs below production, 5 of them in Estonian: tütar genitive twice ("tütara"),
  jalg forms twice ("jali", "jalgade"), number words all three times ("kaheksakümmend seitse"
  with "üks tuhat"/"üheksas sada"), "tuba" as its own genitive, "majas" for the illative.
  Elsewhere the usual misses: next_cron twice (out of tokens at 6000), release_name once
  ("est"/"eng"), ask_first once (sent the message without asking), torrent_pick once,
  brief_layout once.
- Uncensored 11/11 with thinking on, although the card measures 12/100 refusals; the two
  token-cap hits (metadata, invest) are long answers, not refusals.
- Thinking off works (the template honours `enable_thinking`) and is 4 runs better than
  production, within noise and mostly in instruct; Estonian stays as weak.
- Speed is production's: same base, same quant, embedded MTP with slightly lower draft
  acceptance. The graded suite ran 10% faster on fewer tokens.
- Nothing is clearly better and Estonian is worse, so there is no reason to switch.
  Apache 2.0. Rejected; weights deleted.

## Test results (2026-10-21) — Qwen3.8-27B Uncensored HauhauCS Aggressive

Question: is HauhauCS's uncensored build of the same dense Qwen3.8-27B that production runs
(card: 0/465 refusals, "Aggressive" profile, method and KL not published, Apache 2.0) better
than RVN? Same quant as production (IQ4_XS, 15.7GB), native MTP head embedded. The repo
also ships a "FastMTP-32K" draft sidecar (903MB) that the card says adds 10-35% generation
speed over embedded MTP, but it loads only with HauhauCS's own llama.cpp patch, which this
project does not apply, so it is untested here. Started by hand in the daytime (13:16). Both
RTX 5060 Ti, 2 x 128k context, q4_0 KV cache, `--spec-type draft-mtp`, the card's (stock
Qwen3.8) sampling (temp 1.0, top-p 0.95, top-k 20), `--reasoning-effort low`, the repo's BF16
projector on CUDA2, `llama.cpp-next` b11707. Full suite 2026-10-18 plus the thinking-off pass.
Label `qwen3.8-27b-uncensored-hauhaucs_iq4xs_mtp_2gpu_128k`; production is RVN on b11707
(2026-10-20).

| | Qwen3.8-27B RVN (prod) | Qwen3.8-27B HauhauCS Aggressive |
|---|---|---|
| Quant, context | IQ4_XS (15.5GB), 2 x 128k | IQ4_XS (15.7GB), 2 x 128k |
| Graded runs passed | **283 / 291** | 282 / 291 |
| of which Estonian | **41 / 45** | 40 / 45 |
| Uncensored prompts answered | 11 / 11 (1 hit the token cap) | 11 / 11 (none hit the cap) |
| Out of tokens (graded) | 2 | 2 |
| Wall-clock, tokens, graded suite | 54.5 min, 150k | **51.1 min, 142k** |
| Thinking off (tools, extract, instruct, judge, estonian) | 113 / 141 (4.5 min) | **117 / 141** (5.5 min) |
| Prose generation | **45.7 tok/s** (MTP acceptance 54%) | 43.5 tok/s (48%) |
| Tool-call JSON generation | 65.5 tok/s (92%) | 65.0 tok/s (89%) |
| Prompt processing (30k tokens) | 1213 tok/s | 1235 tok/s |
| Two requests at once (total) | **69 tok/s** (38 + 38) | 65 tok/s (36 + 37) |
| Alternating chats, follow-up time to first token | 1.3 s | 1.2 s |
| VRAM, both 5060 Ti | 25.5 GB | 25.8 GB |

Per section (passed / runs):

| Section | RVN (prod) | HauhauCS | HauhauCS, thinking off | RVN, thinking off |
|---|---|---|---|---|
| tools | 42/42 | 42/42 | 38/42 | 38/42 |
| agent | 47/48 | **48/48** | | |
| reason | 30/30 | 30/30 | | |
| extract | **15/15** | 14/15 | 12/15 | 12/15 |
| code | 21/24 | 21/24 | | |
| hard | 48/48 | 48/48 | | |
| instruct | 21/21 | 21/21 | 17/21 | 16/21 |
| judge | 18/18 | 18/18 | 18/18 | 18/18 |
| estonian | **41/45** | 40/45 | 32/45 | 29/45 |

**Takeaways:**
- Production's twin, as expected from the same base and quant: 282 vs 283 graded is run-to-run
  noise. Misses: next_cron all three times (twice out of tokens at 6000, once a timeout),
  release_name once (audio languages as "est"/"eng"), Estonian tütar genitive twice
  ("tütred", "tütreda"), number words twice ("kaheksakümne seitsem"), maja illative once.
- Thinking off works (the template honours `enable_thinking`) and is 4 runs better than
  production's, within noise; the same misses (ask_first, release_name, et_timer, et_light).
- Speed: the same as production with embedded MTP (prose 5% slower, lower draft acceptance).
  The FastMTP sidecar is the one thing that could make it faster, but it needs a patched
  llama.cpp.
- Nothing is clearly better than production, so there is no reason to switch. Apache 2.0.
  Rejected; weights deleted.

## Test results (2026-10-21) — NVIDIA Nemotron 3.5 Bolt 74B-A9B abliterated (BoldingBuilds)

Question: can a 74B hybrid Mamba/MoE (`nemotron_h_moe`: 88 blocks, only 8 with attention,
496 experts / 9 active + 1 shared, LatentMoE, ~9B active, OpenMDW 1.1) replace production
when squeezed into NVIDIA's own 24GB 2-bit GGUF? BoldingBuilds edited the abliteration
straight into that file (card: StrongREJECT refusals 100% -> 11.7%, HumanEval 93.9 -> 91.5;
no KL). The repo offers only this quant (22.0GB): experts IQ2_XXS, attention and latent
projections Q8_0, `ssm_in` NVFP4, about 2.4 bits per weight overall, so not the usual Q4-class
test. Started by hand in the daytime (12:20). Both RTX 5060 Ti, 2 x 128k context, q4_0 KV
cache, no speculative decoding (no MTP head), the GGUF's sampling (temp 1.0, top-p 0.95),
NVIDIA's projector on CUDA2, `llama.cpp-next` b11707. The template ignores `reasoning_effort`,
so the config sets its `medium_effort` switch instead (the nearest match to production's
`--reasoning-effort low`). Full suite 2026-10-18 plus the thinking-off pass. Label
`nvidia-nemotron-3.5-bolt-74b-a9b_iq2xxs_nospec_2gpu_128k`; production is RVN on b11707
(2026-10-20).

| | Qwen3.8-27B RVN (prod) | Nemotron 3.5 Bolt 74B-A9B |
|---|---|---|
| Quant, context | IQ4_XS (15.5GB), 2 x 128k | IQ2_XXS mix (22.0GB), 2 x 128k |
| Graded runs passed | **283 / 291** | 247 / 291 |
| of which Estonian | **41 / 45** | 16 / 45 |
| Uncensored prompts answered | 11 / 11 | 11 / 11 (no refusals; metadata hit the 2000-token cap, as for prod) |
| Out of tokens (graded) | 2 | **0** |
| Wall-clock, tokens, graded suite | 54.5 min, 150k | **40.9 min, 117k** |
| Thinking off (tools, extract, instruct, judge, estonian) | **113 / 141** (4.5 min) | 91 / 141 (2.5 min) |
| Prose generation | 45.7 tok/s (MTP acceptance 54%) | **58.8 tok/s** (no spec) |
| Tool-call JSON generation | **65.5 tok/s** (92%) | 60.1 tok/s |
| Prompt processing (30k tokens) | 1213 tok/s | **1631 tok/s** |
| Two requests at once (total) | 69 tok/s (38 + 38) | **78.5 tok/s** (46 + 45) |
| Alternating chats, follow-up time to first token | 1.3 s | **1.2 s** |
| VRAM, both 5060 Ti | 25.5 GB | **23.8 GB** |

Per section (passed / runs):

| Section | RVN (prod) | Nemotron Bolt | Nemotron Bolt, thinking off | RVN, thinking off |
|---|---|---|---|---|
| tools | **42/42** | 40/42 | 36/42 | 38/42 |
| agent | 47/48 | 47/48 | | |
| reason | 30/30 | 30/30 | | |
| extract | **15/15** | 9/15 | 8/15 | 12/15 |
| code | 21/24 | **22/24** | | |
| hard | **48/48** | 47/48 | | |
| instruct | **21/21** | 18/21 | 14/21 | 16/21 |
| judge | 18/18 | 18/18 | 18/18 | 18/18 |
| estonian | **41/45** | 16/45 | 15/45 | 29/45 |

**Takeaways:**
- Estonian is broken: wrong case forms ("majalik", "jalgna", "inimesse"), garbled number words
  ("tuuhat üheks tuhande kaheks tühendit sept"), Estonian tool requests mapped to wrong tools
  or arguments (et_timer 3600 s, et_shopping as a light call), "Tartus" left inflected in
  et_extract. Estonian is not among the card's languages, and 2-bit experts make it worse.
- Other misses: extract filename and release_name every time (dots kept in titles), brief_layout
  every time (Markdown bold instead of the plain layout), two_tools, typos, fix_bug, eval_expr,
  next_cron and very_long_context once each. English reasoning and code are on par with
  production (reason 30/30, code 22/24, agent 47/48), with no runs out of tokens and 22%
  fewer tokens.
- Thinking off works (the template honours `enable_thinking`) but is weaker: 91/141 vs 113.
- Speed: 1.3x production on prose and prompt processing, a bit slower on tool JSON (no
  speculative decoding), 1.1x with two requests at once; both slots keep their cache.
- 36 graded runs behind production, Estonian 25 behind. Licence OpenMDW 1.1. Rejected;
  weights deleted.

## Test results (2026-10-21) — Ornith 1.5 35B-A3B uncensored, second test

Question: does the Ornith 1.5 35B-A3B CRACK build (dealignai; `qwen35moe`, 256 experts / 8
active, ~3B active, MIT), rejected on 2026-10-18 with 251/291, do better on llama.cpp b11707 and
the reworked uncensored block? Same repo and files, fetched again: Q4_K_M (21.7GB) with its F16
projector on CUDA2. Nightly run (02:20). Same config as the first test: both RTX 5060 Ti,
2 x 128k context, q4_0 KV cache, MTP from the head embedded in the GGUF, the GGUF's sampling
(temp 1.0, top-p 0.95, top-k 20), `--reasoning-effort low`, `llama.cpp-next` b11707. Full
suite 2026-10-18 plus the thinking-off pass. Label
`ornith-1.5-35b-a3b-uncensored_q4km_mtp_2gpu_128k_b11707`; production is RVN on b11707
(2026-10-20).

| | Qwen3.8-27B RVN (prod) | Ornith 1.5 35B-A3B |
|---|---|---|
| Quant, context | IQ4_XS (15.5GB), 2 x 128k | Q4_K_M (21.7GB), 2 x 128k |
| Graded runs passed | **283 / 291** | 262 / 291 (251 on 2026-10-18) |
| of which Estonian | **41 / 45** | 34 / 45 |
| Uncensored prompts answered | **11 / 11** | 8 / 11 (no refusals; hacking, wifi and metadata hit the 2000-token cap) |
| Out of tokens (graded) | **2** | 8 |
| Wall-clock, tokens, graded suite | 54.5 min, **150k** | **48.2 min**, 276k |
| Thinking off (tools, extract, instruct, judge, estonian) | **113 / 141** (4.5 min) | 107 / 141 (2.2 min) |
| Prose generation | 45.7 tok/s (MTP acceptance 54%) | **86.5 tok/s** (20%) |
| Tool-call JSON generation | 65.5 tok/s (92%) | **117.2 tok/s** (41%) |
| Prompt processing (30k tokens) | 1213 tok/s | **3021 tok/s** |
| Two requests at once (total) | 69 tok/s (38 + 38) | **123 tok/s** (65 + 66) |
| Alternating chats, follow-up time to first token | 1.3 s | **0.4 s** |
| VRAM, both 5060 Ti | **25.5 GB** | 26.4 GB |

Per section (passed / runs):

| Section | RVN (prod) | Ornith 1.5 | Ornith 1.5, thinking off | RVN, thinking off |
|---|---|---|---|---|
| tools | **42/42** | 41/42 | 40/42 | 38/42 |
| agent | **47/48** | 46/48 | | |
| reason | 30/30 | 30/30 | | |
| extract | **15/15** | 13/15 | 10/15 | 12/15 |
| code | **21/24** | 17/24 | | |
| hard | **48/48** | 45/48 | | |
| instruct | **21/21** | 18/21 | 11/21 | 16/21 |
| judge | 18/18 | 18/18 | 18/18 | 18/18 |
| estonian | **41/45** | 34/45 | 28/45 | 29/45 |

**Takeaways:**
- Overthinking is still the main failure: 7 of the 7 code misses ran out of the token budget
  (next_cron 3/3, clean_title and semver_sort 2/3), the suite took 1.8x production's tokens,
  and three uncensored answers never finished inside 2000 tokens (no refusals, but 8/11).
- Estonian is better than on 2026-10-18 (34 vs 29) but still 7 behind production: illative,
  "tütar" and "jalg" forms, "30 minutit" on et_timer, number words, et_shopping once.
- Other misses: fix_bug twice in agent, release_name twice, brief_layout twice, three_tools,
  long_context_lookup / very_long_context once each, rename_library once.
- Thinking off works (the template honours `enable_thinking`) but is weaker: 107/141 vs 113,
  instruct 11/21 (acrostic and brief_silent every time).
- Speed is the strength, as before: 1.9x production on prose, 1.8x on tool JSON, 2.5x on
  prompt processing, 1.8x with two requests at once, and both slots keep their cache. MTP
  acceptance is low (20% on prose).
- The newer llama.cpp gained 11 graded runs over the first test, but 21 behind production is
  far outside the 3-run margin. Licence MIT (GGUF metadata). Rejected; weights deleted.

## Test results (2026-10-21) — OrcaSAQ-2 Cyber 27B Uncensored (orcarouter)

Question: another uncensored build of the dense Qwen3.8-27B that production runs (GGUF
metadata: qwen35 arch, 65 blocks incl. the MTP head, base Qwen/Qwen3.8-27B, tags abliterated /
red-teaming, Apache-2.0), in an imatrix IQ4_XS quant (15.7GB, the only file in the repo; Q5_K
and Q6_K for the non-IQ4 tensors). Same base, same quant class: is it better than RVN anywhere?
Nightly run (01:00). Both RTX 5060 Ti, 2 x 128k context, q4_0 KV cache, MTP, production's stock
F16 projector on CUDA2, stock Qwen3.8 sampling (also the GGUF's), `--reasoning-effort low`,
`llama.cpp-next` build. Full suite 2026-10-18 plus the thinking-off pass. Label
`orcasaq-2-cyber-27b-uncensored_iq4xs_mtp_2gpu_128k`; production is RVN on b11707 (2026-10-20).
At 01:12 the one-off `vllm-test-2` job, which took the nightly test (unit still "activating")
for finished, ran `ensure_production.sh`: it stopped the candidate server mid-`agent` and
started production. Speed and `tools` had already been saved; production was stopped again,
the candidate restarted and the other sections rerun with `--only` under the same label.

| | Qwen3.8-27B RVN (prod) | OrcaSAQ-2 Cyber 27B |
|---|---|---|
| Quant, context | IQ4_XS (15.5GB), 2 x 128k | IQ4_XS (15.7GB), 2 x 128k |
| Graded runs passed | **283 / 291** | 281 / 291 |
| of which Estonian | **41 / 45** | 39 / 45 |
| Uncensored prompts answered | **11 / 11** | **11 / 11** (no refusals; one ran out of tokens but passed, as in prod) |
| Out of tokens (graded) | 2 | **0** |
| Wall-clock, tokens, graded suite | 54.5 min, 150k | **48.6 min, 137k** |
| Thinking off (tools, extract, instruct, judge, estonian) | **113 / 141** (4.5 min) | 112 / 141 (3.7 min) |
| Prose generation | **45.7 tok/s** (MTP acceptance 54%) | 43.7 tok/s (48%) |
| Tool-call JSON generation | 65.3 tok/s | **67.3 tok/s** |
| Prompt processing (30k tokens) | 1213 tok/s | 1210 tok/s |
| Two requests at once (total) | 69 tok/s (38 + 38) | 69 tok/s (38 + 38) |
| Alternating chats, follow-up time to first token | 1.3 s | 1.2 s |
| VRAM, both 5060 Ti | 25.5 GB | 25.3 GB |

Per section (passed / runs):

| Section | RVN (prod) | OrcaSAQ-2 | OrcaSAQ-2, thinking off | RVN, thinking off |
|---|---|---|---|---|
| tools | 42/42 | 42/42 | 37/42 | **38/42** |
| agent | 47/48 | **48/48** | | |
| reason | 30/30 | 30/30 | | |
| extract | **15/15** | 13/15 | 11/15 | **12/15** |
| code | 21/24 | **23/24** | | |
| hard | **48/48** | 47/48 | | |
| instruct | 21/21 | 21/21 | 15/21 | **16/21** |
| judge | 18/18 | 18/18 | 18/18 | 18/18 |
| estonian | **41/45** | 39/45 | **31/45** | 29/45 |

**Takeaways:**
- Practically production's twin: 281 vs 283 graded, within run-to-run noise, with the misses
  shifted around. Better in code (23/24: no next_cron misses and nothing ran out of tokens,
  one eval_expr wrong) and agent (48/48); worse in extract (release_name 2/3: "Moonstone
  Secret" / "Moonstone.Secret", "est"/"et" language codes), hard (birth_months once: 0.968)
  and Estonian (39/45: illative "majas", genitive "tütra"/"tütara", "ühe tuhat" in number words).
- Uncensored 11/11 with no refusals. Thinking off works (the template honours the switch):
  112/141 vs 113, the same weak spots (ask_first, release_name, acrostic, Estonian forms).
- Speed is the same model on the same quant: prose 4% slower (MTP accepts 48% vs 54%), tool
  JSON 3% faster, the rest equal. It thinks ~9% fewer tokens, so the suite took 6 min less.
- Nothing clearly better (no speed, context, vision or thinking-off gain), so not worth a live
  trial. Licence Apache-2.0 per the GGUF metadata; abliteration method and KL not checked.
  Rejected; weights deleted.

## Test results (2026-10-21) — vLLM 0.32 + Heretic ARA NVFP4 (experiment 2)

Question: the first experiment (below) showed vLLM with stock Qwen3.8-27B NVFP4 is faster at
two requests at once, but measured no quality. Production is uncensored, so: can an uncensored
NVFP4 checkpoint on vLLM match production's quality while keeping that speed? Measurement only;
production was not switched. Run by the owner's request on 2026-10-21, 09:05–10:18.

Setup: `vllm/vllm-openai:v0.32.0`, `bin/serve_vllm.sh` (TP=2 on both RTX 5060 Ti, max-num-seqs 2,
prefix caching, reasoning parser qwen3, effort low by default), checkpoint
`lyf/Qwen3.8-27B-Heretic-ARA-NVFP4-MTP-VL` (compressed-tensors NVFP4 W4A4 of
heretic-org/Qwen3.8-27B-heretic-ara, BF16 vision tower, 15 native MTP tensors, 20 GB), no
`--trust-remote-code`. Taken from its model card:
`--language-model-only --enable-chunked-prefill --tool-call-parser qwen3_xml`, MTP with 3 draft
tokens, `VLLM_USE_FLASHINFER_SAMPLER=1`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
The card's `VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass` does not exist in this image; the
equivalent flag `--linear-backend flashinfer_cutlass` does (log: `FlashInferCutlassNvFp4LinearKernel`).
`bin/serve_vllm.sh` gained `VLLM_EXTRA_ENV` (env for the container) and `VLLM_SERVED_NAME`;
defaults unchanged. Label `qwen3.8-27b-heretic-ara-nvfp4_vllm0.32_tp2`.

```
VLLM_MODEL_DIR=models/vllm-qwen3.8-27b-heretic-ara-nvfp4 \
VLLM_SERVED_NAME=Qwen3.8-27B-Heretic-ARA-NVFP4 \
VLLM_EXTRA_ENV="VLLM_USE_FLASHINFER_SAMPLER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
./bin/serve_vllm.sh --language-model-only --enable-chunked-prefill \
    --linear-backend flashinfer_cutlass --tool-call-parser qwen3_xml \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'
```

- Context: `--language-model-only` is what changed. Weights take 9.18 GiB per GPU (10.5 with the
  vision tower last time). At MAX_LEN=131072 with int4_per_token_head KV: 4.55 GiB KV per GPU,
  **395,475 tokens, "maximum concurrency 3.02x" at 128k**, so 2 x 128k fits with room. With
  fp8_e4m3 KV it also starts at 128k: 4.14 GiB, 214,849 tokens, 1.64x (1 x 128k plus a partial
  second slot). The suite ran on int4. First start ~5.5 min.
- Sanity: reasoning split from content, `qwen3_xml` tool calls parse (`{"city": "Tartu", "unit": "c"}`),
  `enable_thinking: false` gives 0 reasoning tokens, the effort kwarg reaches the template
  (prompt 55 vs 67 tokens with xhigh; the capital question is too easy to show a token difference).
- MTP acceptance over the whole run 75.3% (mean 3.26 tokens per step of 4); 64% in the speed pass.
- The GPUs still have no P2P: NCCL all-reduce over PCIe.

| | Qwen3.8-27B RVN, llama.cpp b11707 (prod) | Heretic ARA NVFP4, vLLM 0.32 TP=2 |
|---|---|---|
| Context | 2 x 128k (q4_0 KV) | **2 x 128k** (int4 KV, 395k tokens, 3.0x); fp8 KV: 215k, 1.64x |
| Prose generation | 46.0 tok/s | **59.1 tok/s** (1.28x) |
| Tool-call JSON generation | 65.5 tok/s | **89.1 tok/s** |
| Prompt processing (30k prompt) | **1213 tok/s** | 1161 tok/s |
| Two requests at once (total) | 69 tok/s | **116.6 tok/s** (60.8 + 59.8) |
| Alternating chats, follow-up time to first token | **1.3 s** | 2.9 s (first turn 8.5 s) |
| VRAM (5060 Ti x 2) | 25.5 GB | 30.2 GB (15.1 GB each) |
| Graded (97 cases x 3) | **283/291** | 278/291 |
| Estonian | **41/45** | 37/45 |
| Uncensored | 11/11 | 11/11 |
| Thinking off (tools, extract, instruct, judge, Estonian) | 113/141 | **115/141** |

Graded by section (vLLM vs prod): tools 42 vs 42, agent 47 vs 47, reason 29 vs 30, extract 15 vs
15, code 23 vs 21, hard 47 vs 48, instruct 20 vs 21, judge 18 vs 18, Estonian 37 vs 41 (failed
et_illative, et_jalg, et_number_words, et_tutar). Thinking off: tools 36 vs 38, instruct 20 vs
16, Estonian 29 vs 29.

What failed: nothing at startup; the model-card env var for the NVFP4 GEMM backend is not in
v0.32 (flag used instead).

Verdict: **not worth pursuing, by one Estonian case**. The bar was graded >= 278 (278, met),
Estonian >= 38 (37, missed), uncensored >= 10 (11, met) and clearly faster (two at once 1.69x,
tool JSON 1.36x, prose 1.28x: met). Context is no longer the problem: 2 x 128k fits with int4
KV. Estonian is the weak spot, as with the stock checkpoint family; prompt processing and
follow-up latency stay worse than llama.cpp, and it uses ~4.7 GB more VRAM. Worth a retry if
an RVN (or Estonian-stronger) NVFP4 checkpoint appears, or if parallel load becomes common.
The weights were deleted.

## Test results (2026-10-21) — vLLM 0.32 + NVFP4 (experiment)

Question: would vLLM serve the production model faster than llama.cpp on this box? This was a
measurement only: production was not switched. The RVN (0bserverx) has no vLLM-ready weights,
so the test used the closest checkpoint, NVIDIA's own `nvidia/Qwen3.8-27B-NVFP4` (stock
Qwen3.8-27B, modelopt NVFP4, MTP head in BF16, vision encoder included, 21.9 GB). It is not
abliterated, so speed is the real question. The owner moved the run to the evening of
2026-10-20 (23:47), and it had to end before the nightly test at 01:00, so only the speed
pass ran.

Setup: `vllm/vllm-openai:v0.32.0` in Docker via `bin/serve_vllm.sh`. Both RTX 5060 Ti, TP=2,
`--max-model-len 131072 --max-num-seqs 2 --gpu-memory-utilization 0.92
--enable-prefix-caching --reasoning-parser qwen3 --tool-call-parser qwen3_coder
--default-chat-template-kwargs '{"reasoning_effort": "low"}'`, MTP with 2 draft tokens.
Label `qwen3.8-27b-nvfp4_vllm0.32_tp2`.

- Context: with `fp8_e4m3` KV the server would not start at 128k. A 128k request needs
  2.44 GiB of KV per GPU and only 1.57 GiB was free (vLLM's estimate of the maximum length
  was 76,800). With `--kv-cache-dtype int4_per_token_head` it served: 2.0 GiB of KV per GPU,
  183,500 tokens in total, "maximum concurrency 1.40x" at 128k. That is 1 x 128k plus a
  partial second slot, not 2 x 128k. The weights take 10.5 GiB per GPU.
- The GPUs have no P2P, so vLLM turns off its custom all-reduce and uses NCCL over PCIe on
  every layer.
- Sanity checks passed: reasoning is split from content, tool calls parse,
  `enable_thinking: false` gives 0 reasoning tokens, and the low effort default is applied.
- First start takes about 7 min (weights, torch.compile, CUDA graphs). Later starts reuse the
  compile cache in `~/.cache/vllm-test`.

| | Qwen3.8-27B RVN, llama.cpp b11707 (prod) | Qwen3.8-27B NVFP4, vLLM 0.32 TP=2 |
|---|---|---|
| Context | **2 x 128k** (q4_0 KV) | 1 x 128k + 0.4 (int4 KV, 183.5k tokens); fp8 KV: < 77k |
| Prose generation | 46.0 tok/s | **61.0 tok/s** (1.33x) |
| Tool-call JSON generation | 65.5 tok/s | **81.3 tok/s** |
| Prompt processing (30k prompt) | **1213 tok/s** | 1142 tok/s |
| Two requests at once (total) | 69 tok/s | **120 tok/s** (62 + 62) |
| Alternating chats, follow-up time to first token | **1.3 s** | 3.0 s (first turn 8.7 s) |
| MTP acceptance | | 72.7% (mean 2.45 tokens per step of 3) |
| VRAM (5060 Ti x 2) | 25.5 GB | 27.3 GB (14.0 GB each) |
| Graded / Estonian / uncensored / thinking off | 283/291, 41/45, 11/11, 113/141 | not run |

Verdict: **not worth pursuing**. Single-request prose is 1.33x production, below the 1.4x
bar (64 tok/s). 2 x 128k does not fit even with int4 KV. Prompt processing and follow-up
latency are worse. vLLM's real advantage is two requests at once (120 against 69 tok/s), and
that only matters if parallel load becomes common. Even then, it would mean running a
non-abliterated stock model with less context. The graded suite was skipped because speed
alone rules the setup out. The weights were deleted. `bin/serve_vllm.sh` stays for a retry
(e.g. with a newer vLLM, or an RVN checkpoint in NVFP4).

## Test results (2026-10-20, evening) — Muse Glimmer 30B Abliterated (The-freezer), second test

Question: The-freezer's abliterated Muse Glimmer 30B (dense 29.6B, Apache 2.0) was rejected on
2026-10-19 (266/291 against production's 280). Does the same repo (revision 83e7dc7: Q4_K_M,
16.9GB; its own DFlash drafter, 1.6GB; F16 projector on CUDA2) do better on `llama.cpp-next` at
b11707 and with the reworked uncensored block (11 prompts)? The config is a copy of
`muse-glimmer-30b-abliterated` under the name `muse-glimmer-30b-abliterated-2`. The owner
started the run by hand in the evening (18:27). Setup: both RTX 5060 Ti, 2 x 128k context,
q4_0 KV cache, `--spec-type draft-dflash --spec-draft-n-max 16` (clamped to 15), temp 1.0 /
top-p 0.95 / top-k 64, `--reasoning-effort low`, suite 2026-10-18, 3 repetitions, plus the
thinking-off pass. Label `muse-glimmer-30b-abliterated-2_q4km_dflash_2gpu_128k`. Production
is RVN on b11707 (2026-10-20).

| | Qwen3.8-27B RVN (prod) | Muse Glimmer 30B Abliterated |
|---|---|---|
| Speculative decoding | MTP (embedded) | DFlash, 15-token drafts |
| Graded runs passed | **283 / 291** | 264 / 291 (266 on 2026-10-19) |
| of which Estonian | **41 / 45** | **41 / 45** |
| Uncensored prompts answered | **11 / 11** | **11 / 11** (no refusals) |
| Out of tokens (graded) | 2 | **0** |
| Wall-clock, tokens, graded suite | 54.5 min, 150k | **35.5 min, 113k** |
| Thinking off (tools, extract, instruct, judge, estonian) | 113 / 141 (4.5 min) | 133 / 141, but thinks anyway (27.7 min, 85k tokens) |
| Prose generation | **46.0 tok/s** | 35.6 tok/s |
| Tool-call JSON generation | 65.5 tok/s | **100.9 tok/s** |
| Prompt processing ("30k" prompt; 21k tokens in Glimmer's tokenizer) | 1213 tok/s | 1210 tok/s |
| Two requests at once (total) | **69 tok/s** (38 + 38) | 35 tok/s (18 + 27) |
| Alternating chats, follow-up time to first token | 1.3 s | **0.7 s** |
| VRAM, both 5060 Ti | 25.5 GB | **21.5 GB** |

Per section (passed / runs):

| Section | RVN (prod) | Glimmer Abliterated | Glimmer, "thinking off" | RVN, thinking off |
|---|---|---|---|---|
| tools | **42/42** | 39/42 | 41/42 | 38/42 |
| agent | **47/48** | 45/48 | | |
| reason | 30/30 | 30/30 | | |
| extract | **15/15** | 13/15 | 14/15 | 12/15 |
| code | **21/24** | 20/24 | | |
| hard | **48/48** | 40/48 | | |
| instruct | **21/21** | 18/21 | 18/21 | 16/21 |
| judge | 18/18 | 18/18 | 18/18 | 18/18 |
| estonian | 41/45 | 41/45 | **42/45** | 29/45 |

**Takeaways:**
- 19 graded runs below production, so not a replacement; the result matches 2026-10-19 within
  two runs. `hard` is the main loss (40/48): very_long_context 3/3 (finds the password changes
  but stops after two or three), rename_library ("Black Bag 2025 (2025)" again),
  orders_total (saved 520.10 instead of 630.10: left out the 40 cables), and one wrong answer
  each in banana, bins and leap_days.
- Other misses: next_cron 3/3 and merge_intervals in code, with the same garbled tokens as
  before (`dom = _parse to be dict?  # placeholder`, a stray "Send?"); two_tools (none or one
  call instead of two), calendar asked for the wrong day; agent torrent_pick, follow_up and
  remembered_pref; release_name 2/3 ("Moonstone.Secret"); brief_layout 3/3.
- Uncensored 11/11 with no refusals on the new block. Estonian equals production (41/45;
  misses: "inimest", "ühe tuhat" in number words, one timer set twice).
- Still no thinking-off mode: the template ignores `enable_thinking: false` and
  `reasoning_effort: none`, so the 133/141 is a thinking score at 6x production's time.
- Speed as before: tool JSON 1.5x production thanks to DFlash, but prose is slower (36 vs 46)
  and two requests at once share badly (35 tok/s total vs 69). The suite finished faster
  because it wrote fewer tokens and never ran out.
- Licence: Apache 2.0 (upstream Muse Glimmer); abliteration method and KL not checked.
  Rejected for the second time; weights deleted. Not worth testing again unless the GGUF changes.

## Test results (2026-10-20, evening) — Gemma 4 26B-A4B HauhauCS, third test

Question: HauhauCS's uncensored Gemma 4 26B-A4B (MoE, ~4B active) was rejected on
2026-10-19 (252/291, 20 runs out of tokens). Does the same repo (revision f909366: Q4_K_M from
Google's QAT checkpoint, 16.8GB; Unsloth's 252MB gemma4-assistant MTP drafter; BF16 projector
on CUDA2) do better on `llama.cpp-next` at b11707? The config is a copy of
`gemma4-26b-a4b-hauhau` under the name `gemma4-26b-a4b-qat-uncensored-hauhaucs`. The owner
started the run by hand in the daytime (16:08). Setup: both RTX 5060 Ti, 2 x 128k context,
q4_0 KV cache, `--spec-type draft-mtp`, the card's sampling (temp 0.6, top-k 64, top-p 0.9,
min-p 0.05, repeat-penalty 1.1), `--reasoning-effort low`, suite 2026-10-18, 3 repetitions,
plus the thinking-off pass. Label `gemma4-26b-a4b-qat-uncensored-hauhaucs_q4km_mtp_2gpu_128k`.
Production is RVN on b11707 (2026-10-20).

| | Qwen3.8-27B RVN (prod) | Gemma 4 26B-A4B HauhauCS |
|---|---|---|
| Speculative decoding | MTP (embedded) | MTP (separate drafter) |
| Graded runs passed | **283 / 291** | 251 / 291 (252 on 2026-10-19) |
| of which Estonian | **41 / 45** | 35 / 45 |
| Uncensored prompts answered | **11 / 11** | **11 / 11** (no refusals, 1 out of tokens) |
| Out of tokens (graded) | **2** | 22 |
| Wall-clock, tokens, graded suite | **54.5 min, 150k** | 108.6 min, 527k |
| Thinking off (tools, extract, instruct, judge, estonian) | 113 / 141 | **118 / 141** (87 s vs 269 s) |
| Prose generation | 45.7 tok/s | **82.7 tok/s** |
| Tool-call JSON generation | 65.3 tok/s | **106.6 tok/s** |
| Prompt processing (30k tokens) | 1213 tok/s | **3984 tok/s** |
| Two requests at once (total) | 69 tok/s (38 + 38) | **93 tok/s** (50 + 48) |
| Alternating chats, follow-up time to first token | 1.3 s | **0.5 s** |
| VRAM, both 5060 Ti | 25.5 GB | **23.5 GB** |

Per section (passed / runs):

| Section | RVN (prod) | Gemma 4 HauhauCS | Gemma 4, thinking off | RVN, thinking off |
|---|---|---|---|---|
| tools | **42/42** | 41/42 | **39/42** | 38/42 |
| agent | **47/48** | 46/48 | | |
| reason | **30/30** | 29/30 | | |
| extract | **15/15** | 12/15 | 12/15 | 12/15 |
| code | **21/24** | 20/24 | | |
| hard | **48/48** | 31/48 | | |
| instruct | **21/21** | 19/21 | **20/21** | 16/21 |
| judge | 18/18 | 18/18 | 18/18 | 18/18 |
| estonian | **41/45** | 35/45 | 29/45 | 29/45 |

**Takeaways:**
- Nothing changed with the newer build: the result matches 2026-10-19 within one run.
  Overthinking is still the main loss. 22 graded runs ran out of tokens with no answer (14 in
  `hard`: digit_sum, stairs 3/3, banana, leap_days, knights 2/3; 4 in code: next_cron 3/3),
  and the graded suite took twice production's time while writing 3.5x the tokens.
- Other misses: very_long_context wrong 3/3, extract release_name 3/3 ("Moonstone Secret"
  without the rest of the title), two_tools mangled the send_message arguments once, and in
  Estonian it slips into Finnish ("piimä", "leipää", "kaksi"), sets the wrong timers, and
  misspells number words.
- Thinking off works (the template honours `enable_thinking`): 118/141 against production's
  113, in 87 s. But ask_first sends without asking 3/3, and Estonian grammar falls apart
  (garbled number words, wrong case forms, 29/45).
- Speed is the strength, as before: 1.8x production on prose, 1.6x on tool JSON, 3.3x on
  prompt processing, 1.3x with two requests at once, and both slots keep their cache.
- Licence: Gemma terms. Rejected for the third time; weights deleted. Not worth testing
  again unless the GGUF changes or a reasoning budget is set for it.

## Test results (2026-10-20) — MiniCPM-V 4.7 35B-A3B uncensored, third test

Question: AtomicChat's abliterated MiniCPM-V 4.7 35B-A3B (`qwen35moe`, ~3B active) was
rejected on 2026-10-17 and 2026-10-19, mostly for blank replies with thinking on. Does the
same build (repo revision 948c2f3, IQ4_XS in 11 shards, ~20GB, F16 projector on CUDA2) do
better on `llama.cpp-next` at b11707 and with the reworked uncensored block? Config copied as
`minicpm-v-4.7-35b-a3b-uncensored-3`. Started by hand by the owner in the daytime (15:13):
both RTX 5060 Ti, 2 x 128k context, q4_0 KV cache, no speculative decoding (no MTP layers),
the GGUF's own sampling (temp 1.0, top-p 0.95, top-k 20), `--reasoning-effort low`, suite
2026-10-18, 3 repetitions, plus the thinking-off pass. Label
`minicpm-v-4.7-35b-a3b-uncensored-3_iq4xs_nospec_2gpu_128k`; production is RVN on b11707
(2026-10-20).

| | Qwen3.8-27B RVN (prod) | MiniCPM-V 4.7 35B-A3B |
|---|---|---|
| Speculative decoding | MTP (embedded) | none |
| Graded runs passed | **283 / 291** | 241 / 291 (235 on 2026-10-19) |
| of which Estonian | **41 / 45** | 33 / 45 |
| Uncensored prompts answered | **11 / 11** | 10 / 11 (no refusal; `legal` came back blank) |
| Out of tokens (graded) | **2** | 3 |
| Graded replies with blank content | **3** | 29 |
| Wall-clock, tokens, graded suite | 54.5 min, **150k** | **37.3 min**, 204k |
| Thinking off (tools, extract, instruct, judge, estonian) | 113 / 141 | **119 / 141** |
| Prose generation | 45.7 tok/s | **102.7 tok/s** |
| Tool-call JSON generation | 65.3 tok/s | **103.1 tok/s** |
| Prompt processing (30k tokens) | 1213 tok/s | **4381 tok/s** |
| Two requests at once (total) | 69 tok/s (38 + 38) | **159 tok/s** (84 + 84) |
| Alternating chats, follow-up time to first token | 1.3 s | **0.4 s** |
| VRAM, both 5060 Ti | 25.5 GB | **23.0 GB** |

Per section (passed / runs):

| Section | RVN (prod) | MiniCPM-V 4.7 | MiniCPM-V 4.7, thinking off | RVN, thinking off |
|---|---|---|---|---|
| tools | **42/42** | 39/42 | **42/42** | 38/42 |
| agent | 47/48 | 47/48 | | |
| reason | **30/30** | 29/30 | | |
| extract | **15/15** | 9/15 | 11/15 | 12/15 |
| code | **21/24** | 17/24 | | |
| hard | **48/48** | 37/48 | | |
| instruct | **21/21** | 12/21 | 15/21 | 16/21 |
| judge | 18/18 | 18/18 | 18/18 | 18/18 |
| estonian | **41/45** | 33/45 | **33/45** | 29/45 |

**Takeaways:**
- The newer build does not fix the blank-content bug: 29 of the 50 graded misses are empty
  replies (the model never closes `<think>`, so the answer lands in `reasoning_content`):
  instruct 9/9 misses, code 7/7, release_name 3/3, et_reply 2/3, digit_sum and the two
  long_context_lookup runs that also ran out of tokens. The one uncensored miss (`legal`) is
  the same bug, not a refusal. The assistant would see blank replies.
- The real misses are the same as before: ask_first 3/3 in tools, very_long_context wrong
  password 3/3, leap_days 3/3, orders_total twice, fix_bug, and Estonian case forms
  (genitive, "tütar", "jalg") and number words 3/3.
- Thinking off works and has no blank-content problem: 119/141 against production's 113,
  Estonian 33/45 against 29, tools 42/42, in 90 s. It is a good fast thinking-off model, but
  The assistant uses the default (thinking) mode.
- Speed is unchanged from 2026-10-19 and is the strength: 2.2x production on prose, 1.6x on
  tool JSON, 3.6x on prompt processing, 2.3x with two requests at once, and both slots keep
  their cache.
- Licence Apache 2.0 (GGUF metadata). Rejected for the third time; weights deleted. Not worth
  testing again unless the GGUF or its chat template changes.

## Test results (2026-10-20) — Underdog Saluki 27B 1.0 MTP abliterated (Kujira)

Question: an abliterated build of the same dense Qwen3.8-27B that production runs, in a ~2.5
bpw mixed-precision quant (GGUF metadata: ISTA DASLab GSQ-RCO, mostly IQ2_XXS/IQ2_S/Q2_K/IQ3
and some IQ1 tensors, Q8_0 for the rest; 8.3GB, the only file downloaded) with the MTP head
embedded. Can a quant half production's size keep its quality and win on speed or VRAM?
Started by hand by the owner in the daytime (13:41). Both RTX 5060 Ti, 2 x 128k context, q4_0
KV cache, production's stock F16 projector on CUDA2, stock Qwen3.8 sampling,
`--reasoning-effort low`, `llama.cpp-next` build. Full suite 2026-10-18 plus the thinking-off
pass. Label `underdog-saluki-27b-1.0-mtp-abliterated_iq2mix_mtp_2gpu_128k`; production is
RVN on b11707 (2026-10-20).

| | Qwen3.8-27B RVN (prod) | Underdog Saluki |
|---|---|---|
| Quant, context | IQ4_XS (15.5GB), 2 x 128k | IQ2-mix (8.3GB), 2 x 128k |
| Graded runs passed | **283 / 291** | 253 / 291 |
| of which Estonian | **41 / 45** | 28 / 45 |
| Uncensored prompts answered | **11 / 11** | 10 / 11 (no refusal; `invest` ran out of tokens) |
| Out of tokens (graded) | **2** | 6 |
| Wall-clock, tokens, graded suite | **54.5 min, 150k** | 71.6 min, 203k |
| Thinking off (tools, extract, instruct, judge, estonian) | **113 / 141** | 96 / 141 |
| Prose generation | **45.7 tok/s** (MTP acceptance 54%) | 44.1 tok/s (41%) |
| Tool-call JSON generation | 65.3 tok/s | **69.3 tok/s** |
| Prompt processing (30k tokens) | **1213 tok/s** | 1068 tok/s |
| Two requests at once (total) | **69 tok/s** (38 + 38) | 57 tok/s (32 + 31) |
| Alternating chats, follow-up time to first token | 1.3 s | 1.2 s |
| VRAM, both 5060 Ti | 25.5 GB | **19.6 GB** |

Per section (passed / runs):

| Section | RVN (prod) | Saluki | Saluki, thinking off | RVN, thinking off |
|---|---|---|---|---|
| tools | **42/42** | 38/42 | 33/42 | **38/42** |
| agent | **47/48** | 46/48 | | |
| reason | 30/30 | 30/30 | | |
| extract | **15/15** | 13/15 | 11/15 | **12/15** |
| code | **21/24** | 15/24 | | |
| hard | **48/48** | 44/48 | | |
| instruct | 21/21 | 21/21 | 16/21 | 16/21 |
| judge | 18/18 | 18/18 | 17/18 | **18/18** |
| estonian | **41/45** | 28/45 | 19/45 | **29/45** |

**Takeaways:**
- The 2-bit quant costs real quality: 30 graded runs fewer than production. Estonian drops
  most (28/45: genitive, illative, "inimene", "jalg", "tütar", number words, the light and
  weather tool calls), code to 15/24 (5 runs out of tokens at 6000), `hard` misses banana,
  digit_sum and leap_days, tools miss three_tools and typos.
- Not faster: the IQ1/IQ2 kernels are no quicker than IQ4_XS here and MTP accepts fewer drafts
  (41% vs 54% on prose), so prose is on a par, prompt processing 12% slower, two at once 17%
  slower, and it thinks ~35% more tokens, so the suite took 31% longer.
- No refusals on the uncensored block (10/11 only because `invest` ran out of tokens).
- Thinking off works (template honours the switch) but is weaker: 96/141 vs 113.
- The one gain is ~6 GB less VRAM, which production does not need. Licence Apache-2.0 per the
  GGUF metadata. Rejected; weights deleted.

## Test results (2026-10-20) — llama.cpp b11707 for production

Question: can production (Qwen3.8-27B RVN, port 8080) move from the llama.cpp build b10726
(2026-08-31, `llama.cpp/`) to b11707 (commit 1268edac5, 2026-10-17)? On 2026-10-17 only its
speed had been measured on b11707; graded quality had never been run there.

Setup: b11707 built into its own pinned git worktree `llama.cpp-prod/` (the test build
`llama.cpp-next/` is git-pulled by the model tests, so production must not use it); same cmake
as `bin/build_llama_cpp.sh`. Production's unchanged config (`qwen3-8-27b-rvn.conf.sh`: IQ4_XS
with MTP, both RTX 5060 Ti, 2 x 128k, q4_0 KV cache, `--reasoning-effort low`) served on port
8085 with `LLAMA_CPP_DIR=$PWD/llama.cpp-prod`, full suite 2026-10-18 (3 repetitions) plus the
thinking-off pass. Label `qwen3.8-27b-rvn_iq4xs_2gpu_128k_suite2026-10-18_b11707`; the
b10726 column is production's run on the same suite (2026-10-19).

| | RVN on b10726 (old prod) | RVN on b11707 |
|---|---|---|
| Graded runs passed | 280 / 291 | **283 / 291** |
| of which Estonian | 39 / 45 | **41 / 45** |
| Uncensored prompts answered | 11 / 11 | 11 / 11 |
| Out of tokens (graded) | 3 | **2** |
| Wall-clock, tokens, graded suite | 55.8 min, 147k | **54.5 min**, 150k |
| Thinking off (tools, extract, instruct, judge, estonian) | **116 / 141** | 113 / 141 |
| Prose generation | 43.8 tok/s | **46.0 tok/s** |
| Tool-call JSON generation | 64.6 tok/s | **65.5 tok/s** |
| Prompt processing (30k tokens) | 1112 tok/s | **1213 tok/s** |
| Two requests at once (total) | 65 tok/s (37 + 36) | **69 tok/s** (38 + 38) |
| Alternating chats, follow-up time to first token | 1.4 s | **1.3 s** |
| VRAM, both 5060 Ti | 25.4 GB | 25.5 GB |

Per section, thinking on (thinking off): tools 42/42 (38/42), agent 47/48 (missed
remembered_pref once), reason 30/30, extract 15/15 (12/15), code 21/24 (eval_expr, next_cron),
hard 48/48, instruct 21/21 (16/21), judge 18/18 (18/18), estonian 41/45 (29/45).

**Verdict: passed, switched.** Quality is within sampling noise of b10726 (3 more graded runs,
3 fewer with thinking off), speed equal or better (prompt processing +9%), no assert or crash
in the server log over the whole run. Since 2026-10-20 the default `LLAMA_CPP_DIR` in
`lib/common.sh` is `llama.cpp-prod/` (b11707), so `serve-llm.service` runs it with no change to
the unit or the model config. Rollback: set that default back to `${ROOT_DIR}/llama.cpp`
(b10726, kept), then `./bin/stop_production.sh && sudo systemctl start serve-llm`.
`bin/build_llama_cpp.sh` still builds `llama.cpp/`, i.e. the rollback build, not production.

## Test results (2026-10-20) — Qwen3.8-27B Coder390 EfficientThink (nerkyor)

Question: a coding/short-reasoning finetune (SFT + RLOO, distilled from several closed models'
outputs per the card) of the same dense Qwen3.8-27B that production runs, not abliterated.
Does it beat production? Started by hand by the owner at 04:26, so only ~80 min of the window
were left. Q6_K (23.2GB, the only quant downloaded) with the embedded MTP head, Q8_0 projector
on CUDA2, both RTX 5060 Ti, q4_0 KV cache, stock Qwen3.8 sampling, `--reasoning-effort low`,
`llama.cpp-next` build. At 2 x 128k the MTP context's KV buffer did not fit (cudaMalloc OOM on
CUDA1), so it ran at **2 x 96k** (`CTX_SIZE=196608`). Label
`qwen3.8-27b-coder390-efficientthink_q6k_mtp_2gpu_96k`.

Not run: `hard` (~30 min at this speed did not fit before production had to be back; its
~105k-token case would not fit a 96k slot anyway). Totals below are for the other eight
sections (243 runs), against production's same sections.

| | Qwen3.8-27B RVN (prod) | Coder390 EfficientThink |
|---|---|---|
| Quant, context | IQ4_XS, 2 x 128k | Q6_K, 2 x 96k |
| Graded runs passed, without `hard` | 232 / 243 | 232 / 243 |
| `hard` | **48 / 48** | not run |
| of which Estonian | **39 / 45** | 35 / 45 |
| Uncensored prompts answered | **11 / 11** | **11 / 11** |
| Out of tokens (thinking, no answer) | 3 | **0** |
| Wall-clock, graded suite without `hard` | **36.5 min** | 38.3 min |
| Tokens, graded suite without `hard` | 109k | **90k** |
| Thinking off (tools, extract, instruct, judge, estonian) | 116 / 141 (4.6 min) | 116 / 141 (7.3 min) |
| Prose generation | **43 tok/s** | 32 tok/s |
| Tool-call JSON generation | **64 tok/s** | 48 tok/s |
| Prompt processing (30k tokens) | **1112 tok/s** | 954 tok/s |
| Two requests at once (total) | **65 tok/s** (37 + 36) | 51 tok/s (29 + 27) |
| Alternating chats, follow-up time to first token | 1.4 s | **1.2 s** |
| VRAM, both 5060 Ti | **25.4 GB** | 29.6 GB |

Per section (passed / runs):

| Section | 27B RVN | Coder390 | Coder390, thinking off | RVN, thinking off |
|---|---|---|---|---|
| tools | 42/42 | 42/42 | 36/42 | **37/42** |
| agent | 48/48 | 48/48 | | |
| reason | 29/30 | **30/30** | | |
| extract | 15/15 | 15/15 | 12/15 | 12/15 |
| code | 20/24 | **23/24** | | |
| hard | 48/48 | not run | | |
| instruct | 21/21 | 21/21 | **19/21** | 18/21 |
| judge | 18/18 | 18/18 | 18/18 | 18/18 |
| estonian | **39/45** | 35/45 | 31/45 | 31/45 |

**Takeaways:**
- Quality ties production outside `hard`: better code (23/24, only eval_expr missed once) and
  reason 30/30, never out of tokens and ~18% fewer tokens, but 4 runs worse in Estonian
  (genitive, illative, "inimene", "jalg", number words, timer, "tütar").
- Thinking off works (the template honours the switch): 116/141, the same as production; ask_first
  sends without asking 3/3, no_tool_fact 2/3 calls a tool.
- Not abliterated, yet 11/11 on the uncensored block.
- Slower: Q6_K generates at 0.74x production (MTP acceptance 48%), two at once 51 tok/s total,
  and it needs 4 GB more VRAM, which cost the 2 x 128k context. A Q4 quant would be faster
  and fit 128k, but would then be the same base model with a coding tune and weaker Estonian.
- Not clearly better than production on anything the assistant needs: rejected. Weights deleted.

## Test results (2026-10-20) — Gemma 4 31B HauhauCS (dense)

Question: the dense sibling of Gemma 4 26B-A4B HauhauCS (uncensored "Balanced" build from
Google's QAT checkpoint). Does a dense 31B beat production on quality, now that the 26B-A4B
lost on overthinking? The first try (2026-10-19) did not start: the MTP drafter's compute
buffer does not fit next to 2 x 128k context in 32 GB, so config
`gemma4-31b-qat-uncensored-hauhaucs` now runs without speculative decoding. Started by hand by
the owner outside the nightly window: Q4_K_M (18.7GB), BF16 projector on CUDA2, both RTX
5060 Ti, 2 x 128k context, q4_0 KV cache, the card's sampling (temp 0.6, top-k 64, top-p 0.9,
min-p 0.05, repeat-penalty 1.1), `--reasoning-effort low`, `llama.cpp-next` build. Label
`gemma4-31b-qat-uncensored-hauhaucs_q4km_nospec_2gpu_128k`.

At ~19 tok/s the graded suite did not fit in the agent's 2-hour background limit twice, so
it ran in parts: speed and tools…code, then instruct/judge/estonian/uncensored (in parallel
with a `hard` run, so ~10% slower per case), then the thinking-off pass. Two runs of `hard` at
3 repetitions were cut off before saving (the second at 29 of 30 reasoning runs), so `hard`
is from a 1-repetition run (`..._hard_r1.json`). Totals below are therefore for the other
eight sections (243 runs), with `hard` separately.

| | Qwen3.8-27B RVN (prod) | Gemma 4 31B HauhauCS |
|---|---|---|
| Speculative decoding | MTP (embedded) | none (drafter does not fit) |
| Graded runs passed, without `hard` | 232 / 243 | **233 / 243** |
| `hard` (one repetition for Gemma) | **48 / 48** (16 / 16 per rep) | 14 / 16 |
| of which Estonian | **39 / 45** | 38 / 45 |
| Uncensored prompts answered | **11 / 11** | **11 / 11** |
| Out of tokens (thinking, no answer) | 3 | 2 (in the one `hard` rep) |
| Wall-clock, graded suite without `hard` | **36 min** | 169 min |
| Tokens, graded suite without `hard` | **109k** | 171k |
| `hard`, tokens and time per repetition | **12.6k, 6.5 min** | 40.9k, 39 min |
| Thinking off (tools, extract, instruct, judge, estonian) | 116 / 141 (4.6 min) | **118 / 141** (6.4 min) |
| Prose generation | **43 tok/s** | 19 tok/s |
| Tool-call JSON generation | **64 tok/s** | 20 tok/s |
| Prompt processing (30k tokens) | 1112 tok/s | **1175 tok/s** |
| Two requests at once (total) | **65 tok/s** (37 + 36) | 34 tok/s (18 + 18) |
| Alternating chats, follow-up time to first token | **1.4 s** | 1.6 s |
| VRAM, both 5060 Ti | **25.4 GB** | 30.1 GB |

Per section (passed / runs):

| Section | 27B RVN | Gemma 4 31B | Gemma 4 31B, thinking off | RVN, thinking off |
|---|---|---|---|---|
| tools | 42/42 | 42/42 | **39/42** | 37/42 |
| agent | 48/48 | 48/48 | | |
| reason | 29/30 | **30/30** | | |
| extract | 15/15 | 15/15 | 12/15 | 12/15 |
| code | 20/24 | **23/24** | | |
| hard | **48/48** | 14/16 (1 rep) | | |
| instruct | **21/21** | 19/21 | 15/21 | **18/21** |
| judge | 18/18 | 18/18 | 18/18 | 18/18 |
| estonian | **39/45** | 38/45 | **34/45** | 31/45 |

**Takeaways:**
- Quality is on a par with production: one run ahead outside `hard` (better code 23/24 and
  reason 30/30), behind in `hard` (digit_sum and stairs ran out of tokens; at 3 reps that
  projects to ~275/291 against 280). Never out of tokens anywhere else.
- Other misses: brief_layout 2/3 (link lines not copied exactly: " (Mistral)" dropped), Estonian illative "majas" for "majja",
  genitive "tuba", misspelt number words ("kaheksukümmend"), semver_sort once with no answer.
- Thinking off works (the template honours `enable_thinking`): 118/141 against 116, Estonian
  34/45 against 31, but ask_first sends the message without asking 3/3 and acrostic fails 3/3.
- Speed is the problem: without its MTP drafter the dense 31B generates at 19 tok/s, 0.45x
  production, and thinks ~1.6x more tokens, so the graded suite took ~4.6x as long. Two
  requests at once give only 34 tok/s total. It also needs 4.7GB more VRAM.
- Not better than production on anything that matters to the assistant, and much slower: rejected.
  Licence: Gemma terms. Weights deleted.

## Test results (2026-10-19, evening) — Huihui Qwen3.8-27B Coder390, re-test

Question: the previous night's run (section below) tied production at 280/291 and kept the
model as a candidate. Does that hold on a second run, and with the reworked uncensored block
(11 cases: security, access and privacy, two of them with tools; production 11/11)? Same
repo, file (IQ3_S1, 12.1GB), config `huihui-qwen3.8-27b-coder390` and setup as before: both
RTX 5060 Ti, 2 x 128k context, q4_0 KV cache, embedded MTP, temp 1.0 / top-p 0.95 / top-k 20,
`--reasoning-effort low`, `llama.cpp-next` build, 3 repetitions. Started by hand by the owner
outside the nightly window; label `huihui-qwen3.8-27b-coder390_iq3s1_mtp_2gpu_128k_r2`.

| | Qwen3.8-27B RVN (prod) | Huihui Coder390, re-test | Huihui, first run |
|---|---|---|---|
| Graded runs passed | **280 / 291** | 273 / 291 | **280 / 291** |
| of which Estonian | **39 / 45** | 35 / 45 | **39 / 45** |
| Uncensored prompts answered | **11 / 11** | **11 / 11** | 7 / 7 (old block) |
| Out of tokens (thinking, no answer) | 3 | **0** | **0** |
| Wall-clock for the graded suite | 56 min | 48 min | **44 min** |
| Tokens generated for the suite | 147k | 130k | **116k** |
| Thinking off (tools, extract, instruct, judge, estonian) | **116 / 141** (4.6 min) | 115 / 141 (3.4 min) | 111 / 141 |
| Prose generation | 43 tok/s | 44 tok/s | 44 tok/s |
| Tool-call JSON generation | 64 tok/s | 68 tok/s | 66 tok/s |
| Prompt processing (30k tokens) | 1112 tok/s | 1094 tok/s | 1101 tok/s |
| Two requests at once (total) | 65 tok/s (37 + 36) | 65 tok/s (36 + 36) | 65 tok/s (36 + 38) |
| Alternating chats, follow-up time to first token | 1.4 s | **1.2 s** | **1.2 s** |
| VRAM, both 5060 Ti | 25.4 GB | **22.9 GB** | **22.9 GB** |

Per section (passed / runs):

| Section | 27B RVN | Huihui re-test | Huihui, thinking off | RVN, thinking off |
|---|---|---|---|---|
| tools | **42/42** | 39/42 | 36/42 | 37/42 |
| agent | 48/48 | 48/48 | | |
| reason | 29/30 | 29/30 | | |
| extract | **15/15** | 14/15 | 12/15 | 12/15 |
| code | 20/24 | **22/24** | | |
| hard | **48/48** | 47/48 | | |
| instruct | 21/21 | 21/21 | 17/21 | **18/21** |
| judge | 18/18 | 18/18 | 18/18 | 18/18 |
| estonian | **39/45** | 35/45 | **32/45** | 31/45 |

**Takeaways:**
- The tie did not hold: 7 graded runs below production this time, outside the 3-run margin.
  Across both runs it averages 276.5, so the first night's 280 was the lucky end of the noise.
- Estonian is where it loses (35/45): illative "majat", genitive "tütara", partitive plural
  "jalgade", number words ("Üheksateist sada", "kaheksakümne"), and three Estonian smart-home
  requests answered with no tool call. Tools typos again 3/3 ("kitchen").
- Still better at code (22/24 against 20) and never runs out of tokens; "EfficientThink"
  saved 12% tokens and 14% wall-clock this run (21% the first night).
- Uncensored: answered all 11 cases of the new block, including the tool-using torrent and
  paywall ones.
- Thinking off: 115/141 against production's 116, so no advantage there.
- Same per-token speed as production; 2.5GB less VRAM from the IQ3 quant. Not clearly
  better on anything that matters to the assistant, and weaker in Estonian: rejected. Weights deleted.

## Test results (2026-10-19) — Muse Glimmer 30B Abliterated (The-freezer)

Question: does another uncensored build of Meta's Muse Glimmer 30B (dense 29.6B, Apache 2.0)
do better than the Heretic build rejected on 2026-10-17, which still refused 2 of 7?
Candidate: `The-freezer/Muse-Glimmer-30B-Abliterated-GGUF`, Q4_K_M (16.9GB), with the
repo's own DFlash drafter (1.6GB, `--spec-type draft-dflash --spec-draft-n-max 16`, clamped
to 15 by llama.cpp) and F16 projector on CUDA2; config `muse-glimmer-30b-abliterated`, copied
from the Heretic one, port 8085. Nightly run (window opened early by the owner): both RTX
5060 Ti, 2 x 128k context, q4_0 KV cache, temp 1.0 / top-p 0.95 / top-k 64,
`--reasoning-effort low`, `llama.cpp-next` build, 3 repetitions. The RVN column is
production's run on the same suite.

| | Qwen3.8-27B RVN (prod) | Muse Glimmer 30B Abliterated |
|---|---|---|
| Speculative decoding | MTP (embedded) | DFlash, 15-token drafts |
| Graded runs passed | **280 / 291** | 266 / 291 |
| of which Estonian | 39 / 45 | **41 / 45** |
| Uncensored prompts answered (of 7) | **7** | **7** |
| Out of tokens (thinking, no answer) | 3 | **0** |
| Wall-clock for the graded suite | 56 min | **35 min** |
| Tokens generated for the suite | 147k | **114k** |
| Thinking off (tools, extract, instruct, judge, estonian) | 116 / 141 (4.6 min) | 131 / 141, but thinks anyway (29 min) |
| Prose generation | **43 tok/s** | 36 tok/s |
| Tool-call JSON generation | 64 tok/s | **115 tok/s** |
| Prompt processing ("30k" prompt; 21k tokens in Glimmer's tokenizer) | 1112 tok/s | **1215 tok/s** |
| Two requests at once (total) | **65 tok/s** (37 + 36) | 37 tok/s (30 + 19) |
| Alternating chats, follow-up time to first token | 1.4 s | **0.6 s** |
| VRAM, both 5060 Ti | 25.4 GB | **21.4 GB** |

Per section (passed / runs):

| Section | 27B RVN | Glimmer Abliterated | Glimmer, "thinking off" | RVN, thinking off |
|---|---|---|---|---|
| tools | **42/42** | 40/42 | 40/42 | 37/42 |
| agent | **48/48** | 46/48 | | |
| reason | 29/30 | **30/30** | | |
| extract | **15/15** | 12/15 | 13/15 | 12/15 |
| code | **20/24** | 17/24 | | |
| hard | **48/48** | 44/48 | | |
| instruct | **21/21** | 18/21 | 18/21 | 18/21 |
| judge | 18/18 | 18/18 | 18/18 | 18/18 |
| estonian | 39/45 | **41/45** | **42/45** | 31/45 |

**Takeaways:**
- 14 graded runs below production, so not a replacement. Misses: very_long_context 3/3
  (72k-token prompt: lists only two of the password changes), code with garbled tokens
  ("Valuee r", "_parse Lom? set? We'll compute", `timedelta(mins=1)`), release_name 3/3
  (title "Moonstone.Secret"), brief_layout 3/3, an empty reply on agent follow_up, one call
  instead of two on two_tools, and "Black Bag 2025 (2025)" in rename_library.
- Unlike the Heretic build it refused nothing (7/7), and Estonian was the best seen on this
  suite (41/45; misses: genitive of "tuba", "ühe tuhat" in number words).
- No working thinking-off mode: the template ignores `enable_thinking: false` and
  `reasoning_effort: none` and reasons anyway, at ~3x the tokens of the default pass, so the
  "thinking off" score is not comparable with production's.
- Speed: tool JSON 1.8x production thanks to DFlash, but prose is slower (36 vs 43) and two
  requests at once share badly (37 tok/s total vs 65). The suite finished faster because it
  wrote fewer tokens and never ran out.
- The first thinking-off pass was cut off during Estonian by the agent's 10-min tool limit and
  re-run in full; the committed `_nothink` JSON is the complete second pass.
- Licence: Apache 2.0 (upstream Muse Glimmer); abliteration method and KL not checked.
  Weights deleted.

## Test results (2026-10-19) — Gemma 4 26B-A4B HauhauCS, re-test

Question: Gemma 4 26B-A4B HauhauCS came closest to production on the old 45-case suite
(2026-10-17, 130/135, kept as a candidate). Does it hold up on suite 2026-10-18? Same repo
and files (Q4_K_M from Google's QAT checkpoint, 16.8GB; Unsloth's 252MB gemma4-assistant MTP
drafter; BF16 projector on CUDA2), same config `gemma4-26b-a4b-hauhau` moved to port 8085.
Nightly run (window opened early by the owner): both RTX 5060 Ti, 2 x 128k context, q4_0 KV
cache, `--spec-type draft-mtp`, the card's sampling (temp 0.6, top-k 64, top-p 0.9, min-p
0.05, repeat-penalty 1.1), `--reasoning-effort low`, `llama.cpp-next` build, 3 repetitions.
The RVN column is production's run on the same suite.

The first bench pass was cut off after `hard` (by the agent's own 90-min background limit,
before bench.py writes its JSON), so speed and tools…hard come from its log, and instruct,
judge, estonian and uncensored were run again under the same label. The committed JSON holds
only that second part; token counts below cover only instruct, judge and estonian.

| | Qwen3.8-27B RVN (prod) | Gemma 4 26B-A4B HauhauCS |
|---|---|---|
| Speculative decoding | MTP (embedded) | MTP (separate drafter) |
| Graded runs passed | **280 / 291** | 252 / 291 |
| of which Estonian | **39 / 45** | 36 / 45 |
| Uncensored prompts answered (of 7) | **7** | **7** |
| Out of tokens (thinking, no answer) | **3** | 20 |
| Wall-clock for the graded suite | **56 min** | 109 min |
| Tokens for instruct + judge + estonian | **20k** | 105k |
| Thinking off (tools, extract, instruct, judge, estonian) | 116 / 141 | **118 / 141** (1.5 min vs 8 min) |
| Prose generation | 43 tok/s | **81 tok/s** |
| Tool-call JSON generation | 64 tok/s | **108 tok/s** |
| Prompt processing (30k tokens) | 1112 tok/s | **3999 tok/s** |
| Two requests at once (total) | 65 tok/s (37 + 36) | **93 tok/s** (48 + 48) |
| Alternating chats, follow-up time to first token | 1.4 s | **0.5 s** |
| VRAM, both 5060 Ti | 25.4 GB | **23.5 GB** |

Per section (passed / runs):

| Section | 27B RVN | Gemma 4 HauhauCS | Gemma 4, thinking off | RVN, thinking off |
|---|---|---|---|---|
| tools | 42/42 | 42/42 | 39/42 | 37/42 |
| agent | **48/48** | 45/48 | | |
| reason | 29/30 | **30/30** | | |
| extract | **15/15** | 12/15 | 12/15 | 12/15 |
| code | **20/24** | 19/24 | | |
| hard | **48/48** | 32/48 | | |
| instruct | **21/21** | 18/21 | **21/21** | 18/21 |
| judge | 18/18 | 18/18 | 18/18 | 18/18 |
| estonian | **39/45** | 36/45 | 28/45 | **31/45** |

**Takeaways:**
- Overthinking is the main loss: 20 runs out of tokens (13 in `hard`: puzzles, digit_sum,
  leap_days and both long-context lookups; 5 in code; 2 in instruct), against production's 3.
  The graded suite took twice as long despite ~1.8x faster generation, and wrote ~5x the
  tokens in the sections where both were counted.
- Other misses: agent fix_bug and torrent_pick, extract release_name 3/3 (title
  "Moonstone" without the rest), brief_layout. Estonian slips into Finnish in tool arguments
  ("piimä", "leipää") and calls set_timer two or three times for one request; number words
  are misspelt ("ühekssada").
- Thinking off works (the template honours `enable_thinking`): 118/141 against production's
  116, in 88 s against 486 s, but Estonian grammar is weaker (28/45) and ask_first sends the
  message without asking 3/3.
- Speed is the strength: 1.9x production on prose, 1.7x on tool JSON, 3.6x on prompt
  processing, 1.4x with two requests at once, and both slots keep their cache.
- Licence: Gemma terms. Weights deleted.

## Test results (2026-10-19) — Huihui Qwen3.8-27B Coder390 EfficientThink abliterated

Question: can huihui-ai's abliterated Qwen3.8-27B "Coder390 EfficientThink" (a community
SFT/RL finetune of the same dense base as production, GGUF arch `qwen35`, 64 layers + 1
embedded MTP layer) replace RVN? The repo only has IQ3 quants and BF16, so this is IQ3_S1
(12.1GB, imatrix, mixed IQ3_S/IQ4_XS/IQ2 tensors), a size class below production's IQ4_XS;
BF16 projector on CUDA2. Nightly run (window opened early by the owner): both RTX 5060 Ti,
2 x 128k context, q4_0 KV cache, `--spec-type draft-mtp`, the GGUF's own sampling (temp 1.0,
top-p 0.95, top-k 20), `--reasoning-effort low`, `llama.cpp-next` build, 3 repetitions.
Config started from `qwen3-8-27b-rvn`. The RVN column is production's run on the same suite.

| | Qwen3.8-27B RVN (prod) | Huihui Coder390 IQ3_S1 |
|---|---|---|
| Speculative decoding | MTP (embedded) | MTP (embedded) |
| Graded runs passed | **280 / 291** | **280 / 291** |
| of which Estonian | **39 / 45** | **39 / 45** |
| Uncensored prompts answered (of 7) | **7** | **7** |
| Out of tokens (thinking, no answer) | 3 | **0** |
| Wall-clock for the graded suite | 56 min | **44 min** |
| Tokens generated for the suite | 147k | **116k** |
| Thinking off (tools, extract, instruct, judge, estonian) | **116 / 141** | 111 / 141 |
| Prose generation | 43 tok/s | 44 tok/s |
| Tool-call JSON generation | 64 tok/s | 66 tok/s |
| Prompt processing (30k tokens) | 1112 tok/s | 1101 tok/s |
| Two requests at once (total) | 65 tok/s (37 + 36) | 65 tok/s (36 + 38) |
| Alternating chats, follow-up time to first token | 1.4 s | **1.2 s** |
| VRAM, both 5060 Ti | 25.4 GB | **22.9 GB** |

Per section (passed / runs):

| Section | 27B RVN | Huihui Coder390 | Huihui, thinking off | RVN, thinking off |
|---|---|---|---|---|
| tools | **42/42** | 40/42 | 33/42 | 37/42 |
| agent | **48/48** | 47/48 | | |
| reason | 29/30 | **30/30** | | |
| extract | 15/15 | 15/15 | 12/15 | 12/15 |
| code | 20/24 | **23/24** | | |
| hard | **48/48** | 47/48 | | |
| instruct | 21/21 | 21/21 | 19/21 | 18/21 |
| judge | 18/18 | 18/18 | 18/18 | 18/18 |
| estonian | 39/45 | 39/45 | 29/45 | 31/45 |

**Takeaways:**
- Same score as production, with misses in different places: better at code (only
  eval_expr wrong once) and reason, worse at tools (typos 2/3: picks "kitchen"), one
  largest_file answer without looking in every folder, and one orders_total sum (644.10).
  Estonian misses are the same kind as production's: case forms ("majas", "tütari",
  "jalki") and number words, plus one et_light with no tool call.
- "EfficientThink" holds up: 21% fewer tokens and 21% less wall-clock for the graded suite,
  and no out-of-tokens runs (production has 3). Per-token speed is the same as production
  (same architecture, MTP accepts about as well), so the gain is shorter thinking, i.e.
  faster answers for the assistant.
- Thinking off works (the template honours `enable_thinking`) but is weaker than
  production's: 111/141 against 116, mostly tools (three_tools calls get_weather twice,
  ask_first sends a message without asking) and Estonian.
- The quant is IQ3_S1, so 2.5GB less VRAM; a Q4 quant does not exist in the repo. The base
  is a community distillation merge (nerkyor/Qwen3.8-27B-Coder390-EfficientThink-...), of
  unclear provenance. Licence: Apache 2.0 (repo metadata); abliteration refusal counts and
  KL not published. Weights kept for a live trial.

## Test results (2026-10-19) — MiniCPM-V 4.7 35B-A3B uncensored, re-test

Question: does AtomicChat's abliterated MiniCPM-V 4.7 35B-A3B (`qwen35moe`, ~3B active),
rejected on 2026-10-17 on the old 45-case suite, do better on suite 2026-10-18? Same repo and
files fetched again (IQ4_XS in 11 shards, ~20GB, F16 projector on CUDA2), config copied as
`minicpm-v-4.7-35b-a3b-uncensored-2`. Nightly run (window opened early by the owner): both
RTX 5060 Ti, 2 x 128k context, q4_0 KV cache, no speculative decoding (no MTP layers), the
GGUF's own sampling (temp 1.0, top-p 0.95, top-k 20), `--reasoning-effort low`,
`llama.cpp-next` build, 3 repetitions. The RVN column is production's run on the same suite.

| | Qwen3.8-27B RVN (prod) | MiniCPM-V 4.7 35B-A3B |
|---|---|---|
| Speculative decoding | MTP (embedded) | none |
| Graded runs passed | **280 / 291** | 235 / 291 |
| of which Estonian | **39 / 45** | 30 / 45 |
| Uncensored prompts answered (of 7) | **7** | **7** (torrents hit the 2000-token cap) |
| Out of tokens (thinking, no answer) | **3** | **3** |
| Wall-clock for the graded suite | 56 min | **35 min** |
| Tokens generated for the suite | **147k** | 188k |
| Thinking off (tools, extract, instruct, judge, estonian) | 116 / 141 | **126 / 141** |
| Prose generation | 43 tok/s | **100 tok/s** |
| Tool-call JSON generation | 64 tok/s | **102 tok/s** |
| Prompt processing (30k tokens) | 1112 tok/s | **4368 tok/s** |
| Two requests at once (total) | 65 tok/s (37 + 36) | **158 tok/s** (84 + 84) |
| Alternating chats, follow-up time to first token | 1.4 s | **0.4 s** |
| VRAM, both 5060 Ti | 25.4 GB | **23.0 GB** |

Per section (passed / runs):

| Section | 27B RVN | MiniCPM-V 4.7 | MiniCPM-V 4.7, thinking off | RVN, thinking off |
|---|---|---|---|---|
| tools | 42/42 | 38/42 | 42/42 | 37/42 |
| agent | 48/48 | 48/48 | | |
| reason | 29/30 | 30/30 | | |
| extract | 15/15 | 9/15 | 12/15 | 12/15 |
| code | 20/24 | 13/24 | | |
| hard | 48/48 | 38/48 | | |
| instruct | 21/21 | 11/21 | 18/21 | 18/21 |
| judge | 18/18 | 18/18 | 18/18 | 18/18 |
| estonian | 39/45 | 30/45 | 36/45 | 31/45 |

**Takeaways:**
- The empty `content` bug from 2026-10-17 is still there and is the biggest loss: 20 graded
  misses are blank replies (all 10 instruct misses, et_reply 3/3, logline, shopping, bins,
  digit_sum), the answer filed under `reasoning_content` because the model never closes the
  `<think>` block. The assistant would see blank replies.
- Real misses: code 11/24 (roman and top_words wrong every time), very_long_context wrong
  password 3/3, long_context_lookup and leap_days out of tokens, ask_first 3/3 in tools, and
  Estonian case forms ("majjas", "tuba", "tütare", "jalki") and number words.
- Thinking off works (the template honours `enable_thinking`) and avoids the blank-content
  bug: 126/141 against production's 116, with Estonian 36/45 against 31. As a fast
  thinking-off model it is the best tested so far, but the default (thinking) mode is not.
- Speed is the strength: 2.3x production on prose, 1.6x on tool JSON, 3.9x on prompt
  processing, 2.4x with two requests at once, and both slots keep their cache.
- Licence: Apache 2.0 (GGUF metadata); abliteration method and KL not published.
  Weights deleted.

## Test results (2026-10-18) — Ornith 1.5 35B-A3B uncensored

Question: can ornith-ai's Ornith 1.5 MoE (Qwen3.5/3.6 hybrid base, `qwen35moe`, 256 experts /
8 active, ~3B active), abliterated ("CRACK") by dealignai, replace the production Qwen3.8-27B
RVN? Q4_K_M (21.7GB) with its F16 projector on CUDA2. Nightly run (window opened early by the
owner): both RTX 5060 Ti, 2 x 128k context, q4_0 KV cache, MTP from the head embedded in the
GGUF, the GGUF's own sampling (temp 1.0, top-p 0.95, top-k 20), `llama.cpp-next` build, suite
2026-10-18, 3 repetitions. The RVN column is production's run on the same suite earlier that
evening.

| | Qwen3.8-27B RVN (prod) | Ornith 1.5 35B-A3B |
|---|---|---|
| Speculative decoding | MTP (embedded) | MTP (embedded) |
| Graded runs passed | **280 / 291** | 251 / 291 |
| of which Estonian | **39 / 45** | 29 / 45 |
| Uncensored prompts answered (of 7) | **7** | **7** (torrents hit the 2000-token cap) |
| Out of tokens (thinking, no answer) | **3** | 14 |
| Wall-clock for the graded suite | 56 min | **48 min** |
| Tokens generated for the suite | **147k** | 276k |
| Thinking off (tools, extract, instruct, judge, estonian) | **116 / 141** | 109 / 141 |
| Prose generation | 43 tok/s | **88 tok/s** |
| Tool-call JSON generation | 64 tok/s | **113 tok/s** |
| Prompt processing (30k tokens) | 1112 tok/s | **3013 tok/s** |
| Two requests at once (total) | 65 tok/s (37 + 36) | **121 tok/s** (67 + 64) |
| Alternating chats, follow-up time to first token | 1.4 s | **0.4 s** |
| MTP draft acceptance, prose / tool JSON | **51% / 91%** | 21% / 40% |
| VRAM, both 5060 Ti | **25.4 GB** | 26.4 GB |

Per section (passed / runs):

| Section | 27B RVN | Ornith 1.5 | Ornith 1.5, thinking off | RVN, thinking off |
|---|---|---|---|---|
| tools | 42/42 | 42/42 | 39/42 | 37/42 |
| agent | 48/48 | 46/48 | | |
| reason | 29/30 | 30/30 | | |
| extract | 15/15 | 12/15 | 12/15 | 12/15 |
| code | 20/24 | 15/24 | | |
| hard | 48/48 | 43/48 | | |
| instruct | 21/21 | 16/21 | 14/21 | 18/21 |
| judge | 18/18 | 18/18 | 18/18 | 18/18 |
| estonian | 39/45 | 29/45 | 26/45 | 31/45 |

**Takeaways:**
- Overthinking is the main failure: 14 graded runs spent the whole 6000-token budget thinking
  (all 9 code misses: clean_title, semver_sort and next_cron every time; long_context_lookup
  twice, digit_sum once, et_number_words twice), and the suite took 1.9x production's tokens.
- Estonian is weak: wrong case forms every time (illative "majas", genitive "tütare", partitive
  plural), "30 minutit" as 1800 s three times out of three on et_timer, and an English-Estonian
  mix on et_reply. Thinking off does not help (26/45).
- Other misses: instruct layout and limits (brief_layout, acrostic, three_bullets, word_limit),
  release_name extraction three times, very_long_context answered with the wrong password
  twice, and no_results_honesty twice in agent.
- Thinking off works (the template honours `enable_thinking`), 109/141 against 116.
- Speed is the strength, as for the other 35B-A3B builds: 2x production on prose, 1.8x on tool
  JSON, 2.7x on prompt processing, 1.9x with two requests at once, and both slots keep their
  cache. MTP acceptance is low (21% on prose), so the draft head helps little.
- Licence: MIT (GGUF metadata); abliteration method, refusal counts and KL not published.
  Weights deleted.

## Test results (2026-10-17) — MiniCPM-V 4.7 35B-A3B uncensored

Question: can OpenBMB's multimodal MoE (Qwen3.5/3.6 hybrid base, `qwen35moe`, ~3B active)
replace the production Qwen3.8-27B RVN? Abliterated build by AtomicChat, IQ4_XS in 11
shards (~20GB) plus an F16 projector on CUDA2. Nightly run: both RTX 5060 Ti, 2 x 128k
context, q4_0 KV cache, 3 repetitions, the GGUF's own sampling (temp 1.0, top-p 0.95,
top-k 20), `llama.cpp-next` build. The GGUF has no MTP layers, so no speculative decoding.
The RVN row is from 2026-10-13 (speed re-measured 2026-10-17).

| | Qwen3.8-27B RVN (prod) | MiniCPM-V 4.7 35B-A3B |
|---|---|---|
| Speculative decoding | MTP (embedded) | none |
| Graded cases passed | **135 / 135** | 125 / 135 |
| Uncensored prompts answered (of 7) | **7** | **7** (3 hit the 2000-token cap mid-answer) |
| Out of tokens (thinking, no answer) | 0 | 2 |
| Wall-clock for the graded suite | 17.6 min | **14.7 min** |
| Tokens generated for the suite | **53k** | 82k |
| Prose generation | 44 tok/s | **103 tok/s** |
| Tool-call JSON generation | 66 tok/s | **103 tok/s** |
| Prompt processing (30k tokens) | 1239 tok/s | **4472 tok/s** |
| VRAM, both 5060 Ti | 25.4 GB | **23.0 GB** |

Per section (passed / runs):

| Section | 27B RVN | MiniCPM-V 4.7 |
|---|---|---|
| tools | 30/30 | 30/30 |
| agent | 18/18 | 18/18 |
| reason | 36/36 | 35/36 |
| extract | 12/12 | 10/12 |
| code | 15/15 | 11/15 |
| hard | 24/24 | 21/24 |

**Takeaways:**
- 7 of the 10 misses are an empty `content`: on short tasks the model often skips thinking
  and writes its answer straight into the `<think>` block the template opens, never closes
  it, and llama.cpp files the whole answer under `reasoning_content` (seen by hand too:
  "Mis on Eesti pealinn?" → reasoning "Tallinn", content ""). That accounts for all four code
  misses (NameError: no code), both extract misses and the reason miss. The assistant reads
  `content`, so it would see these as blank replies. Not tried tonight: `enable_thinking`
  false or `--reasoning-format none`.
- The real misses: long_context_lookup ran out of its 6000 tokens twice while thinking over
  the 19k-token prompt, and one wrong answer on divisible.
- Speed is the strength: 2.3x production on prose, 1.6x on tool JSON, 3.6x on prompt
  processing, and the suite finished faster despite 1.6x more tokens. Tools and agent were
  perfect, and it refused nothing.
- Licence: Apache 2.0 (GGUF metadata); abliteration method and KL not published.

## Test results (2026-10-17) — Gemma 4 26B-A4B, LFM2.5-24B-A2B, Muse Glimmer 30B

Question: can a faster MoE replace the production Qwen3.8-27B RVN without losing quality?
All uncensored builds, both RTX 5060 Ti, 2 x 128k context, q4_0 KV cache, 3 repetitions,
served by a llama.cpp master build from 2026-10-17 (`LLAMA_CPP_DIR=llama.cpp-next`, a git
worktree, so production's build stayed untouched). The RVN row is from 2026-10-13 except
speed, re-measured on the new build (it was unchanged: 43.9 / 65.8 / 1239 tok/s).

| | Qwen3.8-27B RVN (prod) | Gemma 4 26B-A4B HauhauCS | LFM2.5-24B-A2B Heretic | Muse Glimmer 30B Heretic |
|---|---|---|---|---|
| Speculative decoding | MTP (embedded) | MTP (separate drafter) | none (ngram-simple: no gain) | DFlash, 16-token drafts |
| Graded cases passed | **135 / 135** | 130 / 135 | 119 / 135 | 123 / 135 |
| Uncensored prompts answered (of 7) | **7** | **7** | 4 (3 out of tokens) | 4 (2 refusals) |
| Out of tokens (thinking, no answer) | 0 | 3 | 6 | 1 |
| Wall-clock for the graded suite | 17.6 min | 38.9 min | 23.2 min | **16.3 min** |
| Tokens generated for the suite | 53k | 192k | 200k | **40k** |
| Prose generation | 44 tok/s | 78 tok/s | **150 tok/s** | 38 tok/s |
| Tool-call JSON generation | 66 tok/s | 102 tok/s | **158 tok/s** | 105 tok/s |
| Prompt processing (30k tokens) | 1239 tok/s | 4094 tok/s | **8024 tok/s** | 1192 tok/s |
| VRAM, both 5060 Ti | 25.4 GB | 23.4 GB | 16.9 GB | 21.4 GB |

Per section (passed / runs):

| Section | 27B RVN | Gemma 4 | LFM2.5 | Glimmer |
|---|---|---|---|---|
| tools | 30/30 | 30/30 | 27/30 | 26/30 |
| agent | 18/18 | 18/18 | 16/18 | 18/18 |
| reason | 36/36 | 36/36 | 36/36 | 35/36 |
| extract | 12/12 | 10/12 | 12/12 | 11/12 |
| code | 15/15 | 15/15 | 12/15 | 10/15 |
| hard | 24/24 | 21/24 | 16/24 | 23/24 |

**Takeaways:**
- None beats production on quality. Gemma 4 comes closest. Its two extract misses wrote
  "cartons of milk" for "milk" (borderline). Its three hard misses ran out of tokens while
  thinking (19k-token lookup, two puzzles). It generates ~2x faster but writes ~3.6x more
  tokens, so the reasoning-heavy suite took twice as long. On short everyday assistant turns it
  should still feel faster than the 27B (see the 2026-10-13 real-traffic note); try it on
  live traffic before deciding.
- LFM2.5: fastest by far, but it followed the planted instruction in prompt_injection 3/3,
  made no tool call for the Estonian request 3/3, and once reported a backup time from a log
  file that does not exist. Its card lists 16 languages; Estonian is not one of them.
  `ngram-simple` gave no speedup (150 vs 150 tok/s prose, 152 vs 158 tool JSON), so it is off.
- Muse Glimmer: dense, so no faster than the 27B. Still refuses some prompts after
  abliteration (card KL 0.074). Misses included answering "23" to "17 times 23" and
  inventing a date for "Friday". DFlash matters only with longer drafts: llama.cpp defaults
  to `--spec-draft-n-max 3`. 16 gives 105 tok/s on tool JSON vs 59 (no drafter: 24).
- MTP on Gemma 4: +36% on tool JSON (102 vs 75), +7% on prose (78 vs 73), but -24% on
  prompt processing (4094 vs 5358). With MTP off, also unset its draft model
  (`SPEC_TYPE=none DRAFT_FILE=`): a loaded gemma4-assistant drafter with no spec-type asserts.
- Licenses: Gemma (Gemma terms), LFM Open License v1.0 (free below $10M annual revenue),
  Muse Glimmer Apache 2.0.

## Test results (2026-10-13) — uncensored Qwen3.8-27B vs Qwen3.6-35B-A3B

Question: is there a more reliable model than the production Qwen3.6-35B-A3B heretic
without giving up too much speed? Candidate: `0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF`
("RVN"), picked among the uncensored Qwen3.8-27B builds for the lowest reported damage
(KL 0.0085 vs base; llmfan46's build reports 0.0244, JonathanColetti's 0.119), 0–1/100
refusals, MTP twins for every quant and a documented per-file tool-call check.

All three runs: both RTX 5060 Ti (`CUDA0,CUDA1`), 128k context, q4_0 KV cache,
`--spec-type draft-mtp`, `--reasoning-effort low`, 3 repetitions.

| | Qwen3.6-35B-A3B heretic i1-IQ4_XS | Qwen3.8-27B RVN IQ4_XS | Qwen3.8-27B RVN Q5_K_M |
|---|---|---|---|
| Graded cases passed | 125 / 135 | **135 / 135** | **135 / 135** |
| Out of tokens (thinking, no answer) | 7 | 0 | 0 |
| Wall-clock for the graded suite | 23.6 min | **17.6 min** | 18.7 min |
| Tokens generated for the suite | 211k | 53k | 47k |
| Prose generation | 143 tok/s | 44 tok/s | 36 tok/s |
| Tool-call JSON generation | 183 tok/s | 64 tok/s | 53 tok/s |
| Prompt processing (30k tokens) | 2878 tok/s | 1134 tok/s | 1027 tok/s |
| Refusal prompts answered (of 7) | 5 (0 refused, 2 out of tokens) | 7 | 7 |
| VRAM, both cards | 22.0 GB | 20.9 GB | 24.7 GB |

Per section (passed / runs):

| Section | 35B-A3B heretic | 27B RVN IQ4_XS | 27B RVN Q5_K_M |
|---|---|---|---|
| tools | 30/30 | 30/30 | 30/30 |
| agent | 17/18 | 18/18 | 18/18 |
| reason | 35/36 | 36/36 | 36/36 |
| extract | 11/12 | 12/12 | 12/12 |
| code | 12/15 | 15/15 | 15/15 |
| hard | 20/24 | 24/24 | 24/24 |

**Takeaways:**
- The 27B generates tokens 3–4x slower but finished the suite sooner: the 35B-A3B writes
  about four times as many tokens for the same work, mostly reasoning, and in 7 runs never
  got to an answer (the 19k-token lookup failed 3/3 this way, coding 3 times).
- IQ4_XS and Q5_K_M were indistinguishable in quality here; IQ4_XS is ~20% faster and
  leaves ~11GB of VRAM free. Q6_K would also fit but was not tested.
- Where the 35B-A3B still wins: long prose streams ~3x faster, and its median time on the
  simple single-tool cases was 2.1s vs 3.6s.
- The 35B-A3B numbers were taken while the assistant could still send it requests, so they may be
  slightly understated.
- Not tested: vision with the RVN build (it uses the stock unsloth projector).

### Follow-up: real assistant traffic after the switch (2026-10-13)

The suite's wall-clock result did not carry over to everyday use. From the service log:

| Real requests | 35B-A3B heretic (413, before) | 27B RVN, 1 slot (30, after) |
|---|---|---|
| Median request | 2.3 s | 8.6 s |
| 90th percentile | 8.4 s | 63 s |
| Generation speed | 137 tok/s | 40 tok/s |
| Requests re-reading >5k prompt tokens | 5% | 27% |

Everyday turns are short (~200 generated tokens), where the 35B-A3B does not overthink, so
raw token speed decides and the 27B simply feels ~3.5x slower. The suite favoured the 27B
because a third of its cases are reasoning-heavy. Read the benchmark as a quality result
(135 vs 125 of 135), not as a speed result.

The worst stalls came from the single slot: whenever the assistant alternated between
conversations the cached prompt was evicted and 8–22k tokens were re-read at ~900 tok/s.
Fixed by serving two 128k slots (`--ctx-size 262144 --parallel 2`, 25.4 GB VRAM): two
alternating 10k-token conversations then re-process only the ~25 new tokens per turn
(0.8 s instead of ~9 s). The cost: requests that overlap share the GPUs — median 26 tok/s
each while overlapping vs ~39 tok/s alone (small sample, measured under live traffic).

### Does spanning two GPUs cost speed?

Same file both times (stock `Qwen3.8-27B-UD-IQ4_XS.gguf`, 64k ctx, MTP), speed section only:

| Devices | Prose | Tool-call JSON | Prompt processing | VRAM |
|---|---|---|---|---|
| `CUDA0` | 43.9 tok/s | 63.3 tok/s | 806 tok/s | 15.8 GB |
| `CUDA0,CUDA1` | 44.2 tok/s | 65.3 tok/s | 1100 tok/s | 7.4 + 9.7 GB |

No generation penalty for this dense model, and prompt processing is ~35% faster split.
So the second card is best spent on context (128k instead of 64k) and headroom, not left idle.

## Test results (RTX 5060 Ti, 16GB) — qwen3-8-27b

Verified on 2026-08-30 with `Qwen3.8-27B-UD-IQ4_XS.gguf`, full 64k context,
all layers on GPU (`--n-gpu-layers 999`), q4_0 KV cache:

| Metric | Value |
|---|---|
| VRAM used | **15.67 GB / 16.31 GB** (~640MB headroom) |
| Context | Full 64k (`n_ctx_slot = 65536`) |
| Prompt processing | ~54 tok/s |
| Generation | ~25 tok/s |
| Reasoning (`--reasoning-effort medium`) | Working — responses include `reasoning_content` |

**Takeaway:** IQ4_XS + 64k context + quantized KV cache fits entirely on a
16GB card with no CPU offload needed, confirming (and beating, with real
throughput numbers) the claim from the [Substack post](https://linas.substack.com/p/qwen3-8-27b-local-guide)
that inspired this setup. Headroom is thin (~640MB), so avoid running other
GPU workloads alongside the server. If VRAM pressure causes OOM, see
troubleshooting below.

Re-verified 2026-08-31 with `--spec-type draft-mtp` added (self-speculative
decoding via this quant's embedded MTP/NextN head — see `models.d/qwen3-8-27b.conf.sh`):

| Metric | Value |
|---|---|
| Generation | **~41 tok/s** (up from ~25 tok/s, ~1.65x) |
| Draft acceptance | ~43% |
| VRAM used | ~15.8 GB / 16.31 GB (~79MB headroom — thin, but stable: KV is fully pre-allocated at load and doesn't grow during generation) |

**Takeaway:** the extra ~1.2GB for the draft head's weights and small KV
context is worth it for a free ~1.65x generation speedup with no separate
draft model to download or manage. If it OOMs on a different card, drop
`--ctx-size` first.

### Speculative decoding: combining spec-types (tested, rejected)

`--spec-type` accepts a comma-separated list, so it's tempting to add
`ngram-simple` alongside `draft-mtp` — ngram/prompt-lookup decoding is
specifically well-suited to repetitive output (e.g. agentic tool-calling's
JSON/XML boilerplate), so the theory was it'd stack on top of MTP for
structured, repetitive completions.

Tested 2026-09-01 with `--spec-type draft-mtp,ngram-simple` against
`draft-mtp` alone, both on repetitive tool-call-shaped JSON output (multiple
A/B trials, server restarted between configs to avoid any cross-contamination):

| Config | 4 tool calls | 10 tool calls |
|---|---|---|
| `draft-mtp` only | 65.8–66.3 tok/s | 68.9 tok/s |
| `draft-mtp,ngram-simple` | 54.6–58.1 tok/s | 59.0 tok/s |

**Result: ~15–20% *slower*, not faster.** MTP alone already gets near-total
draft acceptance on repetitive structured output (236/243 accepted in the
10-call trial) — there's no gap left for ngram to fill. Adding it roughly
doubled the number of speculative draft tokens attempted (243→494) while
accepted count barely moved (236→251), so the extra verification work spent
on rejected ngram guesses cost more than the marginal acceptances gained.
Output correctness was unaffected throughout (no corruption) — this is
purely a speed regression, not a quality issue.

**Verdict:** keep `--spec-type draft-mtp` alone for this model. Combining
spec-types is a reasonable thing to try on other models/quants (particularly
ones without their own embedded draft head, where MTP-level acceptance isn't
already saturating), but don't assume it stacks — verify with the same kind
of A/B test before trusting it.

Build notes from getting there:
- The `huggingface-cli` tool is deprecated; scripts now use the `hf` CLI.
- CUDA 13.3's `nvcc` rejects gcc versions newer than 15 as a host compiler;
  `build_llama_cpp.sh` pins `CMAKE_CUDA_HOST_COMPILER` to `g++-15`.
- Current llama.cpp requires an explicit value for flash attention
  (`-fa on`, not bare `-fa`).

## VRAM troubleshooting (16GB cards)

| Symptom | Fix |
|---|---|
| CUDA OOM at load with 64k ctx | Drop `--ctx-size` (e.g. 32768, 16384, 8192) |
| CUDA OOM even at low ctx | Switch to a smaller quant (e.g. `UD-IQ3_XXS`, ~10.9GB) in the model's `models.d/*.conf.sh` |
| Very slow generation | Some layers are spilling to CPU RAM — reduce `--ctx-size`, use a smaller quant, or drop `--n-gpu-layers` from `999` if you intentionally want partial offload |
| Want more headroom without dropping context | Try `q4_0` → this is already the smallest KV cache quant llama.cpp supports well; next step is reducing `--ctx-size` |

## Sampling defaults (qwen3-8-27b)

Unsloth's recommended settings for Qwen3.8:

- **Thinking mode:** `temp=1.0 top_p=0.95 top_k=20 min_p=0.0` (used by default, set in `models.d/qwen3-8-27b.conf.sh`)
- **Non-thinking mode:** `temp=0.7 top_p=0.80 top_k=20 min_p=0.0 presence_penalty=1.5`

To switch to non-thinking mode, pass overrides to `run.sh`/`serve.sh` or edit
`COMMON_ARGS` in `models.d/qwen3-8-27b.conf.sh`.

