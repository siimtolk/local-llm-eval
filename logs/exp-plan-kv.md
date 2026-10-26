## Hypothesis
Key-cache quantization hurts about twice as much as value-cache quantization (Reddit measurements on Qwen3.8-27B). Production uses q4_0 for both. Raising only K to q5_0 should keep or improve quality on long-context and Estonian cases.

## Variant
Production config with `--cache-type-k q4_0` replaced by `--cache-type-k q5_0`; V stays q4_0. Same build, weights, context and slots. The post's "2k recent-token window" scheme is not available in llama.cpp and is not tested.

## Spec
```spec
{
  "type": "config",
  "slug": "kv-k-q5-v-q4",
  "title": "Asymmetric KV cache: K q5_0, V q4_0",
  "hypothesis": "K is about 2x as sensitive to quantization as V, so spending one more bit on K only should recover quality (long context, Estonian) at a small VRAM cost.",
  "variant": {
    "server_args_remove": [
      "--cache-type-k",
      "q4_0"
    ],
    "server_args_add": [
      "--cache-type-k",
      "q5_0"
    ],
    "notes": "V stays q4_0. The 2k recent-token window scheme from the post does not exist in llama.cpp and is not tested."
  },
  "metrics": [
    "passed",
    "passed.estonian",
    "passed.hard",
    "tok_s.prose",
    "tok_s.deep_decode",
    "ttft.long_prompt"
  ],
  "pass_rule": [
    {
      "metric": "passed",
      "min_delta": 3
    },
    {
      "metric": "tok_s.deep_decode",
      "min_ratio": 0.93
    },
    {
      "metric": "tok_s.prose",
      "min_ratio": 0.95
    }
  ],
  "checks": [
    "Note VRAM use per GPU while serving and confirm both 128k slots still fit"
  ],
  "cost": {
    "gpu_min": 80,
    "disk_gb": 0,
    "usd": 0
  },
  "source": "experiments/2026-10-26-081911-from-the-local-ai-brief-of-2026-10-26-ha.md"
}
```

## Pass rule
Total passed cases at least 3 above production (above the ±3 noise seen in the ubatch test), and no more than a 7% drop in deep-context decode speed or 5% in prose speed.

## Cost
GPU 80 min · disk 0 GB · USD 0

## Out of scope
Never changes production; adopting the variant is a separate implementation ticket.

Source item: experiments/2026-10-26-081911-from-the-local-ai-brief-of-2026-10-26-ha.md
