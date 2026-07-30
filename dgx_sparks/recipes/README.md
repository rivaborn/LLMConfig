# Local sparkrun recipes

Forked/hand-written sparkrun recipes for the 4-node DGX Spark cluster. Each
exists because the upstream recipe could not be driven from LLMConfig unchanged.

Reference them from a catalog by **absolute path** — `sparkrun run` accepts a
file path as well as an `@registry/name`. From LLMConfig on `.40` that path is
the WSL view:

```
/mnt/c/Coding/rivaborn/LLMConfig/dgx_sparks/recipes/<file>.yaml
```

⚠️ **`mods/` must stay beside the recipes.** sparkrun resolves a recipe's `mods:`
entries relative to the recipe file's own directory, so moving a yaml without
`mods/` silently breaks it (`qwen35-122b-int4-ft` is the one that uses it).

## Verification status — read this before trusting a recipe

"Works" means different things below. Nothing here is verified further than stated.

| Recipe                            | Nodes | Verified                                    | Not yet verified                                   |
| --------------------------------- | ----- | ------------------------------------------- | -------------------------------------------------- |
| `qwen35-122b-int4-ft.yaml`        | 1     | ✅ **Launched and served** — `LoadTimes` has samples under `spark:qwen35-122b-int4-ft` | chat-template quality vs stock was the point of the fork; score it |
| `deepseek-v4-flash-ft.yaml`       | 2–4   | ✅ Renders correctly (`sparkrun run --dry-run`, 2026-07-30) | **never launched** — no load, no serve, no benchmark |
| `deepseek-v4-flash-dspark.yaml`   | 2     | ✅ Renders correctly (`sparkrun run --dry-run`, 2026-07-30) | **never launched**; also needs the `vllm-node-dspark` image, which did not exist when this was written |

## The shared reason these forks exist

Two independent sparkrun 0.2.40 defects, both of which fail *silently*:

**1. `EXTRA_ARGS` is a dead argument.** `sparkrun run` declares a variadic
`EXTRA_ARGS` (`cli/_run.py:92`) and never reads it, so anything after `--` is
accepted and dropped. A flag hardcoded in a recipe's `command:` with no
`{placeholder}` is therefore unreachable from a catalog — the only way to change
it is to fork the recipe. That is why `qwen35-122b-int4-ft` exists.

**2. `{{...}}` in a `command:` template is unrenderable.** sparkrun renders with
vpd's `arg_substitute`, **not** `str.format`, so `{{` has no escape meaning — it
is two literal braces. The regex is `re.compile(r"\{(.*?)\}")`, non-greedy, so
against

```
'{{"method":"mtp","num_speculative_tokens":{num_speculative_tokens}}}'
```

the first match runs from the opening `{{` to the first `}`, swallowing the
inner placeholder into a single match. That blob misses the lookup and
`_replace_match` returns it verbatim ("put back the original if we don't have a
match", `vpd/legacy/arguments.py:22`). Both the doubled braces survive **and**
the placeholder never substitutes, so vLLM receives invalid JSON.

This is not a `--dry-run` artifact: `Recipe.render_command()` is called from the
runtime launch modules (`runtimes/vllm_ray.py:71`). `-o <key>=` does not help —
the placeholder is never consulted. The bug bites even with no placeholder
inside; `{{` alone is enough. **Ten** recipes in the installed registries use
this form, mostly for `--speculative-config` (MTP).

### The rule that follows

> In a `command:` template, write JSON arguments with **single braces and
> literal values**. Verified against sparkrun's own renderer:
>
> | Written as                                             | Renders as             |
> | ------------------------------------------------------ | ---------------------- |
> | `'{{"method":"mtp","num_speculative_tokens":{n}}}'`    | unchanged → **invalid** |
> | `'{"method":"mtp","num_speculative_tokens":2}'`        | unchanged → **valid**   |
> | `{max_model_len}`                                       | `262144`               |

The cost is that such a value becomes a literal and is **no longer tunable from
the catalog** — change it by editing the recipe. Each file says so in its header.

See `HomelabDocumentation/drafts/sparkrun-flag-traps.md` for the full trap list.
