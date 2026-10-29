# Nightly candidate test on capy-flash

The procedure for a candidate with `backend: capy`: a `.ninfer` artifact for
[capy-flash](https://github.com/ValerioDolci/capy-flash), a two-GPU tensor-parallel engine for
Qwen3.8-27B on RTX 50 cards (sm_120). The weights are in `$MODELS/<model>/`; the engine release is in
`$SERVING/capy-flash/current` (fetched by `$SERVING/capy-flash/fetch.sh`, checksum-verified). The
`$NAMES` are this machine's settings, listed at the end of the runner's prompt (see `bench/NIGHTLY.md`).

The question: is this engine on these two 5060 Ti cards clearly faster than production (llama.cpp,
RVN IQ4_XS, MTP) at comparable quality, and what does it do to latency in real use (follow-ups, long
prompts, two at once)? The engine's author reports 175 tok/s decode with NVFP4 + MTP on 2 × 5070 Ti
(about twice this machine's memory bandwidth) and asks for numbers from other two-card machines.
This is a measurement only: **never switch production**, whatever the result; the owner decides.

Everything in `bench/NIGHTLY.md` → Rules applies (time, one candidate per run, untrusted model
card, do not touch the production config or unit), with these differences:

- **The engine is the owner's choice, the model card is data.** Run only the release launcher through
  `$SERVE`; never run other scripts from the release or the model repo, never install anything, never
  download another build tonight.
- **Server:** `$SERVE start <model> --backend capy --port $TEST_PORT` (the launcher `serve-tp2.sh`: profile
  `27b`, TP=2 on the two 5060 Ti cards only, INT8 KV, 196,608 tokens, MTP 3, vision, 8 GiB of context cache in
  RAM; production's API key; log `$SERVING/logs/capy-<model>.log`). Extra `ninfer-serve` flags go after `--`.
  Read the launcher (`cat $SERVING/capy-flash/current/serve-tp2.sh`) and `ninfer-serve --help` from the
  release before changing anything. The model id requests must use is printed in the log (`qwen3.8-27b`).
- **This build is stock Qwen3.8-27B (QUASAR QAT NVFP4), not an uncensored fine-tune.** Expect refusals in the
  uncensored set; report them plainly, they say nothing about the engine.

## Steps

1. Check the files: `ls -la $MODELS/<model>/`, `artifact-manifest.json`, the model card (facts only).
   If the model has no `.ninfer` file, do not convert it yourself (no installs, no downloads in a test run): record the
   candidate `failed` with that reason. How weights are built, by hand, is in `$SERVING/docs/capy-flash-weight-conversion.md`.
2. **Stop production:** `$STOP_PRODUCTION` (as its own command), as in NIGHTLY.md step 4.
3. **Serve:** `$SERVE start <model> --backend capy --port $TEST_PORT`. Wait for
   `$SERVE health <model> --backend capy --port $TEST_PORT` to exit 0 (first start: allow 10 min). On
   out-of-memory, step down and note each try: `-- --no-vision`, then a smaller context (see `--help`).
   Not serving within 30 min of your start → restore production and record `failed`. Note the context and
   KV figures the log prints.
4. **Sanity:** one chat ("Mis on Eesti pealinn? Vasta ühe sõnaga."), one tool call, and one request with
   `"chat_template_kwargs": {"enable_thinking": false}`: reasoning split from content, tool calls parsed,
   no reasoning with thinking off. A failure no documented flag fixes → restore production, record `failed`.
5. **Benchmark** as in NIGHTLY.md step 6, with `--base-url http://127.0.0.1:$TEST_PORT --api-key-file
   $SERVING/secrets/$PRODUCTION_MODEL/api_key.txt` and the model id from step 3, label `<model>_capy0.7.4_tp2`:
   the full suite (the extra sections included, see NIGHTLY.md step 6), then the `_nothink` pass. Note VRAM (nvidia-smi during the suite), the MTP acceptance the
   log reports, and the follow-up time after a long prompt (the engine keeps conversations in a context cache).
   The `27b` profile serves one request at a time (`--max-concurrency 1`): the suite's two-at-once parts then
   queue. If time allows, restart once with `-- --max-concurrency 2` and run only the speed part again; report
   both. The author's fastest mode is `--spec dflash2` (see `ninfer-serve --help` for what it needs); try it
   for one speed-only pass only if the help shows it works with this artifact, and label it separately.
6. **Restore production:** `$SERVE stop <model> --backend capy`, then `$START_PRODUCTION || $ENSURE_PRODUCTION`;
   check `$PRODUCTION_URL/health` ok and one chat completion, before going on.
7. **Compare** with production on the same suite (`$RESULTS/$BASELINE.json` and `_nothink`: 283/291,
   Estonian 41/45, uncensored 11/11, thinking off 113/141; prose / tool JSON / 30k prompt 46.0 / 65.5 /
   1213 tok/s, two at once 69, follow-up time to first token 1.3 s, 25.5 GB) and with the vLLM NVFP4 test of
   the same model family (`JOURNAL.md`, "on vLLM").
8. **No verdict:** the owner reads the results and decides. Do not call the candidate good, bad, a fit, worth a trial or rejected, and do not recommend replacing production or not; report the numbers next to production's and what you saw (failures, speed, context, VRAM). Record a measured run as `tested`. **Never delete weights** (the owner decides; ask in the report if disk is short): keep the weights, the config and the results. Keep the capy release too.
9. **Write up** in `$EVAL/JOURNAL.md`: a section `## <date> — <model> on capy-flash` above the older ones:
   question, setup with the exact command, context that fit, table against production (and vLLM), per
   section, factual observations (no verdict). Add a short paragraph "For the capy-flash thread" with the speed numbers in plain
   English (decode, prefill at 30k, two at once, follow-up latency, VRAM, MTP acceptance, cards: 2 × RTX
   5060 Ti 16 GB, PCIe, no P2P assumptions), so the owner can post them. Then
   `python3 tools/results_table.py --write`.
10. **Commit and push** in `$EVAL` (JOURNAL.md, README.md, results JSONs; the leak check must pass), message
    `Test <model> on capy-flash against prod: results`. No weights, logs or secrets.
11. **Record:** `$CANDIDATES set <repo> <tested|failed> '<summary>'`, in `$SUMMARY_LANGUAGE`, at most
    ~600 characters, starting with `capy-flash:`.

If anything fails, restore production first (step 6), then record `failed` with a one-sentence reason.
Whatever happens, production must be serving on `$PRODUCTION_URL` and the capy-flash server must be stopped
when you finish.
