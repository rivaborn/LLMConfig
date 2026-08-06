# DGX Spark sparkrun recipes

Every recipe that has **actually launched successfully on this cluster**, plus the
local forks written because the upstream recipe could not be driven from
LLMConfig unchanged.

```
├── local/                         our forks — LLMConfig's catalog points HERE
│   ├── qwen35-122b-int4-ft.yaml
│   ├── deepseek-v4-flash-ft.yaml
│   ├── deepseek-v4-flash-dspark.yaml
│   └── mods/                      mods for the forks (must stay adjacent)
└── upstream/                      verbatim snapshots of registry recipes
    ├── *.yaml
    └── mods/                      mods those snapshots declare
```

**`local/` is what runs; `upstream/` is a record.** The three files in `local/`
are referenced by absolute path from the cluster catalog on `.40`, so editing one
changes what launches. Nothing references `upstream/` — those are snapshots kept
so a known-good config survives a registry refresh, and so a diff can answer "did
upstream change under us?".

## What "successfully launched" means here

`data/load_times.yaml` on `.40` records launch durations **success-only** — a
failed launch never produces a sample, it increments a separate `failures:`
counter. So the presence of a `spark:<alias>` samples key is hard evidence that
the alias launched and served at least once. That is the list below; nothing was
included on the strength of being catalogued or looking plausible.

Load times are medians of the recorded samples, measured on GB10 nodes.

| Alias                   | Recipe                                          | Snapshot                                   | Nodes | n | Median load |
| ----------------------- | ----------------------------------------------- | ------------------------------------------ | ----- | - | ----------- |
| `gemma-4-26b`           | `@eugr/gemma4-26b-a4b`                          | `upstream/gemma4-26b-a4b.yaml`             | 1     | 5 | 491 s       |
| `qwen35-122b-int4`      | `@eugr/qwen3.5-122b-int4-autoround`             | `upstream/qwen3.5-122b-int4-autoround.yaml`| 1     | 5 | 343 s       |
| `qwen35-122b-int4-ft`   | **local fork** of the above                     | `local/qwen35-122b-int4-ft.yaml`           | 1     | 1 | 341 s       |
| `qwen36-27b`            | `@official/qwen3.6-27b-fp8-mtp-vllm`            | `upstream/qwen3.6-27b-fp8-mtp-vllm.yaml`   | 1     | 1 | 324 s       |
| `qwen36-35b-a3b`        | `@official/qwen3.6-35b-a3b-fp8-mtp-vllm`        | `upstream/qwen3.6-35b-a3b-fp8-mtp-vllm.yaml`| 1    | 1 | 264 s       |
| `qwen3-coder-next`      | `@official/qwen3-coder-next-int4-autoround-vllm`| `upstream/qwen3-coder-next-int4-autoround-vllm.yaml` | 1 | 1 | 184 s |
| `qwen3-vl-embedding-8b` | `@official/qwen3-vl-embedding-8b-vllm`          | `upstream/qwen3-vl-embedding-8b-vllm.yaml` | 1     | 3 | 145 s       |
| `qwen3-vl-reranker-8b`  | **local fork** `local/qwen3-vl-reranker-8b-fixed.yaml` (2026-07-31) | `upstream/qwen3-vl-reranker-8b-vllm.yaml`  | 1     | 5 | 116 s       |

**Multi-node (tp=2) launches are proven as of 2026-07-30 evening** — both local
DeepSeek forks below launched, served, and (for `-ft`) engaged MTP across
spark1+spark2 the same day the CX7 fabric came up. `load_times` keys them as
`spark:<alias>:x2`.

### The rest, and their status

| Recipe                                                | Why not                                                                                                      |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `local/qwen35-122b.yaml`                              | ✅ **VERIFIED 2026-08-02**: the STABLE `qwen35-122b` alias, catalogued on all four nodes. Runs the RELEASE image (`container: vllm-node`), so a cold load touches nothing but the engine — **5m43s vs the nightly's 19m40s**, measured back-to-back on spark3 with the weights already staged. Serves `vllm-0.23.1rc1`. |
| `local/qwen35-122b_8_3_26.yaml`                       | ✅ **PINNED 2026-08-04** (was `qwen35-122b-int4-nightly`). Serves as **`qwen35-122b_8_3_26`** on spark4. `container:` is pinned BY DIGEST to `eugr/spark-vllm@sha256:1d335d4f…` — the 2026-08-03 build, same as upstream's `prebuilt-vllm-current` (`0.26.1rc1.dev298+g1ea84d74b.d20260803`) and `prebuilt-flashinfer-current` (`0.6.17-51920591-d20260803`). The old name tracked a moving ghcr `nightly:latest` → `eugr/spark-vllm:latest`, which cost ~14 min of image pull on 2026-08-04 and had left all four nodes on four different digests. With `settings.spark_image` passing `--image`, image prep measured **14 min → 0.4 s**, zero bytes downloaded. **`-current` release tags move too** — only the digest and the version strings are immutable. Bump procedure in the recipe header. |
| `local/deepseek-v4-flash-0731.yaml`                   | ✅ **VERIFIED 2026-08-02** (re-verified after a fix): the OFFICIAL release (FP8, 156 G, spec-decode built in) — tp=2 on spark1+2 in ~11 min, DSpark draft on BOTH workers, "Say OK" 5.2 s (preview: ~15 s). **The stable `deepseek-v4-flash` alias points here since 2026-07-31.** ⚠️ Shipped 07-31 **missing `--no-enable-prefix-caching`** — "deliberately absent" was copied from the -dspark note, but vLLM 0.26 defaults it ON, and prefix caching + dspark spec-decode silently CORRUPTS generation (replies degenerate into token salad; per-position acceptance `0.027, 0.000, …` vs a healthy `0.812, 0.684, …`). Fixed `51b7554`. The 07-31 acceptance passed only because "Say OK" is too short to expose it — **verify a model with a several-hundred-token generation.** |
| `local/deepseek-v4-flash-ft.yaml`                     | ✅ **VERIFIED 2026-07-30**: tp=2 launch on spark1+spark2 (330 s), serving, MTP engaged (`DeepSeekV4MTPModel`) — superseded by `-0731` as the daily driver |
| `local/deepseek-v4-flash-dspark.yaml`                 | ✅ **VERIFIED 2026-07-30**: tp=2 launch on spark1+spark2 (510 s), served inference; image built same day — preview-DSpark build, superseded by `-0731`      |
| `local/surya-ocr-2-spark.yaml`                       | ✅ **VERIFIED 2026-08-06 on spark3**: launched in 139 s (18.1 % VRAM) and OCR'd a rendered 4-line invoice through the /v1 gateway, 6/6 checked fields recovered. Emits HTML `<table>` structure — it infers layout, matching the card's claim that layout/OCR/table_rec share one VLM. Spark sibling of the proven 3070 Ti companion config, retuned for throughput. ~48 s for 121 completion tokens with `--enforce-eager` still on — dropping that is the first throughput lever. |
| `local/chandra-ocr-2-spark.yaml`                     | ✅ **VERIFIED 2026-08-06 on spark3**: launched in 224 s (43.4 % VRAM, **co-resident with surya**) and recovered 6/6 on the same invoice. Emits HTML with `data-bbox` coordinates and `data-label` semantics — richer structure than surya's table output, which is the practical difference between the two. `max_num_seqs 32` remains an untested guess against the card's 96-on-H100. |
| `local/glm-5.2-mxfp4-experts-gptq-mtp.yaml`           | ⏳ **STAGED, NOT LAUNCHABLE (2026-08-06)**: `tp=4` and the fabric is still two isolated pairs. Re-verified by ping matrix the same day — there are TWO 200G subnets (`192.168.0.x`, `192.168.2.x`), both carry all four node addresses, and both are **dual-rail within a pair**; spark1/2 ↔ spark3/4 has no path on either. Weights (383.7 GiB) and the digest-pinned container are being staged now so switch day is cable → `SPARK_FABRIC_LINKS` → launch. Verified before staging: repo HEAD sha == the pinned revision, and `index_topk_pattern` is exactly 78 chars = `num_hidden_layers` (config.json ships it `null`, so the override is load-bearing). Three things deliberately unsettled — port 8210 vs the `--port` LLMConfig passes, prefix caching (enabled here, disabled on the DS4 fork sharing this container family), and `num_speculative_tokens` vs the image minimum. |
| `@eugr/openai-gpt-oss-120b`                           | catalogued, never launched — no samples                                                                      |
| `@sparkrun-transitional/qwen3.5-35b-a3b-fp8-sglang`   | **failed** on spark4; appears in `failures:`, never in samples                                               |
| `@sparkrun-transitional/qwen3.5-122b-a10b-fp8-sglang` | catalogued for 2 nodes, never launched; no SGLang image on any node                                          |

**`local/qwen3-vl-reranker-8b-fixed.yaml`** (2026-07-31): the upstream recipe
serves the reranker as a BARE pooling runner — non-discriminative blur (0.909
for an exact answer vs 0.893 for "bananas are yellow") that buried good ANN
hits in ragstack's golden eval (2/10). The fork adds the model card's
classifier conversion (`Qwen3VLForSequenceClassification` hf-overrides,
score = P("yes")) and the repo's `additional_chat_templates/reranker.jinja`
score template — found at launch under `/cache/huggingface` (sparkrun's mount
point inside the container, NOT `/root/.cache`), with an exit-1 guard rather
than ever serving blur again. Verified discrimination: 0.474 vs 0.005 on the
same probe. The `--hf-overrides` JSON is single-brace/literal-only, per the
render rule below.

`qwen3-vl-reranker-8b` also has one recorded failure (on spark1) alongside its
five successes — its `needs_empty_node` transient is lethal to co-residents, so a
success is conditional on the node being free, not on the recipe alone.

## Snapshots vs. live registry

`upstream/` holds **point-in-time copies** taken 2026-07-30 from
`~/.cache/sparkrun/registries/` on `.40`. They exist so a known-good
configuration survives a registry refresh, and so a diff can answer "did upstream
change under us?". They are **not** what runs: the per-node catalogs still
reference `@registry/name`, so a `sparkrun` registry update is picked up
normally. Only the three local forks are referenced by absolute path.

To repoint anything at a snapshot, use the WSL path from `.40`:

```
/mnt/c/Coding/rivaborn/LLMConfig/dgx_sparks/recipes/{local,upstream}/<file>.yaml
```

⚠️ **`mods/` must stay beside its recipes.** sparkrun resolves `mods:` relative to
the recipe file's own directory (falling back to the registry). Four of the
snapshots declare mods — `fix-qwen3-coder-next`, `fix-qwen3-next-autoround`,
`fix-qwen3.5-chat-template` — so those are copied into `upstream/mods/`. A yaml
moved without them silently loses its fix.

## Why the local forks exist

Two independent sparkrun 0.2.40 defects, both of which fail **silently**.

**1. `EXTRA_ARGS` is a dead argument.** `sparkrun run` declares a variadic
`EXTRA_ARGS` (`cli/_run.py:92`) and never reads it, so anything after `--` is
accepted and dropped. A flag hardcoded in a recipe's `command:` with no
`{placeholder}` is unreachable from a catalog — forking is the only way to change
it. That is why `qwen35-122b-int4-ft` exists.

**2. `{{...}}` in a `command:` template is unrenderable.** sparkrun renders with
vpd's `arg_substitute`, **not** `str.format`, so `{{` has no escape meaning — it
is two literal braces. The regex is `re.compile(r"\{(.*?)\}")`, non-greedy, so in

```
'{{"method":"mtp","num_speculative_tokens":{num_speculative_tokens}}}'
```

the first match runs from the opening `{{` to the first `}`, swallowing the inner
placeholder into one match. That blob misses the lookup and `_replace_match`
returns it verbatim (`vpd/legacy/arguments.py:22`), so the doubled braces survive
**and** the placeholder never substitutes — vLLM gets invalid JSON.

Not a `--dry-run` artifact: `Recipe.render_command()` is called from the runtime
launch modules (`runtimes/vllm_ray.py:71`). `-o <key>=` does not help. `{{` alone
is enough, with or without a placeholder inside. **Ten** recipes in the installed
registries use this form, mostly for `--speculative-config` (MTP).

> **Rule:** in a `command:` template, write JSON arguments with **single braces
> and literal values.** Verified against sparkrun's own renderer:
>
> | Written as                                          | Renders as              |
> | --------------------------------------------------- | ----------------------- |
> | `'{{"method":"mtp","num_speculative_tokens":{n}}}'` | unchanged → **invalid** |
> | `'{"method":"mtp","num_speculative_tokens":2}'`     | unchanged → **valid**   |
> | `{max_model_len}`                                    | `262144`                |
>
> The cost: such a value becomes a literal and is **no longer tunable from the
> catalog**. Each fork says so in its header.

**None of the seven `upstream/` snapshots carry this bug** — checked at copy
time. It bites the MTP-flavoured variants (e.g. `gemma4-26b-a4b-nvfp4`,
`qwen3.6-35b-a3b-nvfp4`), not the ones proven here.

Full trap list: `HomelabDocumentation/drafts/sparkrun-flag-traps.md`.
