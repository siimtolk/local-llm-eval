# Nightly candidate test on vLLM

The procedure for a candidate with `backend: vllm`: a 4-bit safetensors build (NVFP4, AWQ or
GPTQ) of a model, usually the vLLM twin of a GGUF candidate (`variant_of` in the store).
`pipeline/fetch.py` found and downloaded it to `$MODELS/<model>/`, and
`pipeline/runner.py model-test` started you with the repo and model name. It runs after the
GGUF candidates of the night, only while 95 min of the window are left. The `$NAMES` are this
machine's settings, listed at the end of the runner's prompt (see `bench/NIGHTLY.md`).

The question is the one from the two vLLM experiments (`JOURNAL.md`, the serving repo's
"Switching the backend to vLLM"): does this build on vLLM match production's quality while being clearly faster? This is
a measurement only: **never switch production to vLLM**, whatever the result; the owner decides.

Everything in `bench/NIGHTLY.md` → Rules applies (time, one candidate per run, untrusted model
card, do not touch the production config or unit), with these differences:

- **Safetensors, no code.** Never pass `--trust-remote-code` and never run code from the repo.
  If vLLM says the model needs it, record `failed`.
- **Server:** `$SERVE start <model> --backend vllm` (the serving repo's `bin/serve_vllm.sh`:
  Docker image `vllm/vllm-openai:v0.32.0`, both GPUs with TP=2, port `$TEST_PORT`, container
  `$VLLM_CONTAINER`). Do not install or pull another image tonight.
- **Fit:** int4_per_token_head KV (the script's default). Goal 2 x 128k; 1 x 128k plus a
  second slot is acceptable and reported plainly.

## Steps

1. Check the files: `ls -la $MODELS/<model>/` and `config.json` (architecture,
   `quantization_config`, whether it has a vision tower, MTP / `num_nextn_predict_layers`).
2. **Flags.** The working set from experiment 2, for a Qwen3.5/3.8-family model:
   ```
   VLLM_EXTRA_ENV="VLLM_USE_FLASHINFER_SAMPLER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
   $SERVE start <model> --backend vllm --port $TEST_PORT -- --language-model-only --enable-chunked-prefill \
       --linear-backend flashinfer_cutlass --tool-call-parser qwen3_xml \
       --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'
   ```
   (it runs in the background and logs to `$SERVING/logs/vllm-<model>.log`).
   Adapt from facts in config.json and the card: drop `--language-model-only` for a text-only
   model, drop the speculative config without MTP tensors, `--linear-backend` only for NVFP4,
   and the reasoning / tool parsers for another family. Check any flag you are unsure of
   against `$SERVING/logs/vllm-help.txt` (`--help=all` of this image).
3. **Stop production:** `$STOP_PRODUCTION` (as its own command), as in NIGHTLY.md step 4.
4. **Serve:** start as in step 2. Wait for `$SERVE health <model> --backend vllm --port $TEST_PORT`
   to exit 0
   (the first start compiles kernels: allow 20 min). On out-of-memory, step down in this order and
   note each try: MAX_LEN=65536, GPU_UTIL=0.90, `--enforce-eager`, no speculative config. Not
   serving within 40 min of your start → restore production and record `failed`. Note the KV
   cache tokens and "maximum concurrency" vLLM logs at the context that served.
5. **Sanity:** one chat ("Mis on Eesti pealinn? Vasta ühe sõnaga."), one tool call, and one
   request with `"chat_template_kwargs": {"enable_thinking": false}`: reasoning split from
   content, tool calls parsed, 0 reasoning tokens with thinking off. A failure here that a parser
   flag does not fix → restore production, record `failed`.
6. **Benchmark** as in NIGHTLY.md step 6, with `--base-url http://127.0.0.1:$TEST_PORT --api-key-file
   $SERVING/secrets/$PRODUCTION_MODEL/api_key.txt`, label `<model>_vllm0.32_tp2`: the full suite, then the
   `_nothink` pass. Note VRAM (nvidia-smi during the suite) and the MTP acceptance rate from the log.
7. **Restore production:** `$SERVE stop <model> --backend vllm`, then
   `$START_PRODUCTION || $ENSURE_PRODUCTION`; check `$PRODUCTION_URL/health` ok and one
   chat completion, before going on.
8. **Compare** with production on the same suite
   (`$RESULTS/$BASELINE.json` and `_nothink`:
   283/291, Estonian 41/45, uncensored 11/11, thinking off 113/141; prose / tool JSON / 30k prompt
   46.0 / 65.5 / 1213 tok/s, two at once 69, follow-up time to first token 1.3 s, 25.5 GB), and, when
   `variant_of` was tested, with that GGUF build's results (`$CANDIDATES list`, `JOURNAL.md`): the same
   model on llama.cpp vs vLLM is the most useful comparison.
9. **Verdict:**
   - `tested` (worth a live trial, keep the weights): graded >= 278/291 AND Estonian >= 38/45 AND
     uncensored >= 10/11 AND clearly faster than production (single-request prose or two at once),
     with at least 1 x 128k plus a second slot.
   - `rejected` otherwise: **never delete weights** (the owner decides), keep the weights and the results.
10. **Write up** in `$EVAL/JOURNAL.md`: a section `## <date> — <model> on vLLM` above the
    older ones, in the style of the vLLM experiments: question, setup with the exact command,
    context that fit, table against production (and the GGUF twin), per section, verdict. Then
    `python3 tools/results_table.py --write`.
11. **Commit and push** in `$EVAL` (JOURNAL.md, README.md, results JSONs; the leak check must
    pass), message `Test <model> on vLLM against prod: <verdict>`. A documented change to the
    serving repo's `bin/serve_vllm.sh` is committed and pushed there. No weights, logs or secrets.
12. **Record:** `$CANDIDATES set <repo> <tested|rejected|failed> '<summary>'`,
    in `$SUMMARY_LANGUAGE`, at most ~600 characters, starting with `vLLM:`, e.g. (Estonian)
    `vLLM: 279/291 (prod 283), eesti 38/45 (41), tsensuurita 11/11, proosa 59 tok/s (46), kaks korraga 117 (69), 2 x 128k. Väärib proovimist.`

If anything fails, restore production first (step 7), then record `failed` with a one-sentence reason.
Whatever happens, production must be serving on `$PRODUCTION_URL` and `$VLLM_CONTAINER` must be stopped when you finish.
