# `aidendle94/GLM-5.2-MXFP4-Experts-GPTQ` — provenance and staging record

Supporting files for the sparkrun recipe at
[`../../recipes/local/glm-5.2-mxfp4-experts-gptq-mtp.yaml`](../../recipes/local/glm-5.2-mxfp4-experts-gptq-mtp.yaml).
Staged on this cluster **2026-08-06**, ahead of the 200G switch, so that the
only work left on switch day is enabling the fabric and launching.

## Attribution

The recipe's own `description:` compresses three different contributions into
one clause; split out, they are:

| Contribution | By |
| ------------ | -- |
| Base GLM-5.2 checkpoint | upstream model authors |
| GPTQ-calibrated MXFP4 expert re-quantization, hybrid assembly, container image | **Aidendle94** |
| 4× DGX Spark deployment integration and benchmarking | **Mike Pfaffenberger** (explicitly *not* the model author) |

Read as written, the description parses as claiming Aidendle94 authored the
base checkpoint. Preserved verbatim in the recipe regardless — this note is the
correction, not an edit of someone else's field.

## Pinned revision

```
repo      aidendle94/GLM-5.2-MXFP4-Experts-GPTQ
revision  46537e0e16fcd156627800139b41b9c497fc7ee2
```

Verified 2026-08-06: public, ungated, and the repo **HEAD sha equals the pinned
revision** — `target_revision`, `model_revision`, and the snapshot path baked
into `mtp_config.model` all agree. Last modified 2026-07-24.

## What the checkpoint actually is

| | |
| -- | -- |
| `model_type` | `glm_moe_dsa` |
| architecture | `GlmMoeDsaForCausalLM` |
| `num_hidden_layers` | **78** |
| `max_position_embeddings` | 1048576 |
| attention heads / KV heads | 64 / 64 |
| vocab | 154880 |
| quantization | `hybrid_mxfp4_ct` — MoE experts MXFP4, linear layers compressed-tensors float-quantized |

Layout is **per-layer**, not sharded-by-size: `L3.safetensors` … `L77.safetensors`
(75 files, no gaps) plus three `hybrid-ct-0000{0,1,2}.safetensors` carrying the
first three layers. `mtp-draft/` (4 files, 5.6 GiB) is the MTP draft model the
recipe's `--speculative-config` points at; `vllm_overlay/` (2 files) and
`run-logs/` (14 files) ride along.

**183 files, 383.72 GiB** — full inventory with per-file sizes in
[`file-inventory.tsv`](file-inventory.tsv), which is what a stage should be
verified against.

## The `index_topk_pattern` check

The recipe injects, via `--hf-overrides`:

```
FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS
```

That string is **78 characters** (21 `F`, 57 `S`) and `num_hidden_layers` is
**78** — they match, and the shape corroborates the file layout: `FFF` for the
three full-attention `hybrid-ct` layers, then the sparse pattern across
`L3..L77`.

This override is **load-bearing**: `config.json` ships
`index_topk_pattern: null`, so without it the sparse indexer has no pattern at
all. Do not treat it as redundant decoration.

## `max_model_len` is not being forced

`max_position_embeddings` is 1048576, so the recipe's `max_model_len: 262144`
sits comfortably inside the trained window. `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`
in the env is inherited from the DS4 recipe lineage and is **not** covering for
an over-length request here — worth stating explicitly, because on the DS4
recipe that same flag *was* the tell that the recipe's own 500k default had to
be capped.

## Staging (2026-08-06)

Performed by [`../../stage-model.sh`](../../stage-model.sh). The route is
dictated by the fabric, which was re-verified by ping matrix the same day:

```
spark1(.50) <-> spark2(.51)   200G, both fabric subnets
spark3(.52) <-> spark4(.53)   200G, both fabric subnets
pair <-> pair                 NO PATH — management 1GbE only
```

Both 200G subnets (`192.168.0.x`, `192.168.2.x`) carry all four node addresses
and look mutually reachable; they are **dual-rail within a pair**, not
cross-pair links.

The route is then decided by measurement, not by the topology diagram:

| Hop | Measured 2026-08-06 |
| --- | ------------------- |
| HuggingFace → a node | ~6 MB/s per node, **~11.7 MB/s aggregate** (a shared ~100 Mbit uplink) |
| spark1 → spark2, 200G intra-pair | **585 MB/s** (ssh cipher-bound — a floor) |
| spark1 → spark3, 1GbE management | **105 MB/s** (line rate) |

`--max-workers 8` genuinely opens 8 files over ~60 connections and still cannot
exceed the uplink, so concurrency buys nothing against the WAN.

The consequence is counter-intuitive and worth stating plainly: **the 1GbE
management hop is ~9× faster than the entire internet uplink.** The design that
avoids it — download once per fabric pair — would pull 2 × 383.7 GiB over a
100 Mbit line (~19.4 h). Downloading **once** and fanning out over LAN costs
~9.8 h of WAN plus ~1.3 h of copying. So:

```
HF ──► spark1 ──200G──► spark2
         └────1GbE────► spark3 ──200G──► spark4
```

Pull from the internet exactly once; move it over any LAN link after that.

The pinned container image is pre-pulled on all four nodes by the same script —
it was absent everywhere at staging time, and an unpulled image has previously
cost this cluster ~14 minutes of surprise pull at launch.

## Files here

| File | What it is |
| ---- | ---------- |
| `config.json` | the checkpoint's config, as of the pinned revision |
| `quantization-recipe.yaml` | the **llm-compressor** recipe published in the repo (`recipe.yaml` there) — how the model was quantized. Not a launch recipe; the sparkrun one lives under `recipes/local/` |
| `file-inventory.tsv` | 183 files with byte sizes — the reference for verifying a stage |
