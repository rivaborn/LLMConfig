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
| `qwen3-vl-reranker-8b`  | `@official/qwen3-vl-reranker-8b-vllm`           | `upstream/qwen3-vl-reranker-8b-vllm.yaml`  | 1     | 5 | 116 s       |

**Every proven recipe is single-node.** No multi-node recipe has launched on this
cluster yet — the CX7 fabric only came up 2026-07-30.

### Not included, and why

| Recipe                                                | Why not                                                             |
| ----------------------------------------------------- | ------------------------------------------------------------------- |
| `local/deepseek-v4-flash-ft.yaml`                     | present as a **fork**, render-verified, but **never launched**      |
| `local/deepseek-v4-flash-dspark.yaml`                 | same, and its `vllm-node-dspark` image did not exist when written   |
| `@eugr/openai-gpt-oss-120b`                           | catalogued, never launched — no samples                             |
| `@sparkrun-transitional/qwen3.5-35b-a3b-fp8-sglang`   | **failed** on spark4; appears in `failures:`, never in samples      |
| `@sparkrun-transitional/qwen3.5-122b-a10b-fp8-sglang` | catalogued for 2 nodes, never launched; no SGLang image on any node |

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
