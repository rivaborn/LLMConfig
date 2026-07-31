# Daily driver — a four-model local AI workspace

**Discussion date:** 2026-07-30 · **Status:** ✅ **FULLY IMPLEMENTED 2026-07-31**
(items 1–11 serving half + item 12, the RAG stack — `rivaborn/ragstack` on
`ubuntuservices`). See the
implementation record at the end of this file; the wiki write-up is queued in
`HomelabDocumentation/drafts/_merge/hosts-ollama-host.md` ("the daily-driver
cutover") and `_merge/runbooks-local-llm-server-dgx-spark.md` (Edit 17).

**Sources for every figure below:** the `docs-a` DR wiki replica (snapshot 2026-07-27,
190 pages — `docs-vm` is down with the Saginaw LAN until ~2026-08-29), plus the unposted
drafts in `HomelabDocumentation/drafts/`, principally `llm-model-load-profiles.md`
(measured 2026-07-29), `_merge/runbooks-local-llm-server-dgx-spark.md` (Edits 1–16) and
`_merge/hosts-ollama-host.md`. No hardware was touched to produce this — every Spark and
both GPUs were in use.

---

## The target workspace

Four independent models, always available, each on hardware suited to it.

| # | Role                       | Model                                          | Where                     |
| - | -------------------------- | ---------------------------------------------- | ------------------------- |
| 1 | Fast coding model          | Qwen3.5-122B-int4                              | 1 Spark node              |
| 2 | Larger reasoning model     | DeepSeek-V4-Flash, `tp=2`                      | 2 Spark nodes             |
| 3 | Vision + OCR               | `vl32` on the 3090, `surya2` on the 3070 Ti    | both GPUs on `.40`        |
| 4 | Embeddings / rerank / RAG  | `qwen3-vl-embedding-8b` + `qwen3-vl-reranker-8b` | 1 Spark node            |

---

## Constraints that bound the design

| Constraint                                                                              | Consequence                                                                    |
| --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| A Spark node holds up to **4 co-resident models**, one per slot port 8000–8003, admitted by declared `mem_fraction` | #4 does not strictly need a whole node — but the chosen model pair fills one   |
| A GPU lane holds **one model at a time**, enforced by the eviction-wait gate             | Being dropped — see decision 2 below                                           |
| vLLM charges **~15 GiB of non-torch overhead per model** on GB10 unified memory          | Every extra co-resident model on a Spark costs 15 GiB before its weights       |
| The CX7 fabric is **two isolated L2 islands**, `{.50,.51}` and `{.52,.53}`               | `tp=2` must stay inside a pair; `tp=4` is impossible until the 200G switch lands |
| A `tp` job **claims every member node whole**                                            | DS4-Flash consumes 2 nodes entirely, no co-residency on either                  |
| Spark cold start is **2–4 minutes** (measured 211–240 s)                                  | A RAG service must be pinned resident, never idle-reaped                       |
| GB10 shares **one unified pool** between CPU and GPU; the 3090/3070 Ti are discrete cards | "Spilling is cheap now" is true of the Sparks only — see decision 2            |

---

## Node and fabric layout

```
pair A   .50 + .51   DeepSeek-V4-Flash, tp=2         (both nodes claimed whole)
pair B   .52         Qwen3.5-122B-int4               (0.80 declared -> 90.4% of pool actual)
         .53         SERVICES: VL embedder + VL reranker   (~88% of pool)
3090                 vl32  (Qwen2.5-VL-32B-AWQ)
3070 Ti              surya2 (surya-ocr-2) + the qwen2.5:1.5b opencode /swap relay
ubuntuservices       vector DB + ingestion + orchestration (regular software)
```

The plan fits the two-island fabric exactly — DS4 inside one pair, the two single-node
workloads on the other. This is fortunate rather than planned: any variant needing `tp=4`,
or `tp=2` spanning pairs, is impossible until the switch is cabled.

⚠️ **The layout commits all four nodes with zero spare.** Nothing is left for
`gemma-4-26b`, `qwen3-coder-next`, or a cold-start buffer — any fifth model must evict
something. Auto-placement's "fewest nodes wins" rule has nothing to fall back on.

---

## Decisions taken

### 1. `surya2` on the 3070 Ti, `vl32` on the 3090

Splits the two jobs that were competing for one card. Surya-OCR-2 extracts text with
per-block confidence; `vl32` answers questions about images. They are not substitutes.

`surya2` is `datalab-to/surya-ocr-2`, a Surya-2 OCR VLM on `Qwen3_5ForConditionalGeneration`,
**bf16 ~3 GB, no offload**, `--max-model-len 18000`. It measures 21,420 MiB on the 3090 only
because `serve.sh` gives it `--gpu-memory-utilization 0.85` — ~18 GB of KV pool it does not
need. On an 8 GB card that utilisation figure must come down.

**What this requires building:**

- **`serve-companion.sh`.** It has never existed — not on `.40`, not in `deploy/`. The
  systemd unit `vllm-companion@` references it; it was specified and never built, and
  `COMPANION_VLLM_ENABLED=false` currently makes that explicit. Surya cannot move to Ollama
  instead: it needs `--mm-processor-kwargs '{"min_pixels":3136,"max_pixels":6291456}'` and
  epubocr consumes **vLLM's logprobs** for confidence scoring.
- **Torch-resolve the 3070 Ti by UUID** inside that script, exactly as `serve.sh` resolves
  the 3090. vLLM's worker ignores `CUDA_DEVICE_ORDER` and uses CUDA's FASTEST_FIRST order.
  Skipping this is how a 30B coder once launched on the 8 GB card and OOM'd.
- **Repoint epubocr.** `SURYA_INFERENCE_URL` targets the primary lane's relay at `:11437`;
  the companion lane's relay is `:11438`.
- **Tune for 8 GB.** `max_pixels: 6291456` generates many vision tokens and multimodal
  profile-runs have OOM'd before. Expect to hand-tune `--gpu-memory-utilization` and
  possibly lower `max_pixels`.

Side benefit: moving daily coding to the 122B retires `coder30-awq`, which the load-profile
sweep measured running the 3090 at **98.9% — 24,296 of 24,576 MiB, ~280 MiB spare** — as the
registry's baseline daily coder.

### 2. Drop the one-model-per-GPU-lane rule

**Rationale (user):** the rule was designed for a time when every model ran on the 3090.
With four DGX Sparks absorbing the large models, spilling from VRAM into system RAM is less
of a concern.

**One correction to the premise, recorded so it isn't lost:** the Sparks make spilling cheap
because GB10 shares one unified pool between CPU and GPU. The 3090 and 3070 Ti are discrete
cards with hard VRAM walls, and the 3090 is an **eGPU over Thunderbolt** — `--cpu-offload-gb`
there is the slowest offload path in the lab. Spilling got cheaper on the Sparks, which
already allow 4 co-residents; it did not get cheaper on these two cards.

**Where the change actually pays off: the 3070 Ti, not the 3090.** `vl32` lands at
21,564 MiB of 24,576 — nothing fits beside it regardless of the rule. The 8 GB card is where
co-residency is needed, so `surya2` (~3 GB) can sit beside the `qwen2.5:1.5b` opencode
`/swap` relay (0.6 GB) rather than evicting it.

**What this requires building:**

- **Port `mem_fraction` to `Lane`.** Budgets are a `SparkUnit` concept; lanes have no budget
  model at all.
- **Decide what replaces the eviction-wait gate** — currently evict-all, poll `nvidia-smi`
  until driver baseline, then load. That gate is the lane's core correctness guarantee, not
  merely a spill guard.
- **vLLM-beside-vLLM is tractable** (both declare `--gpu-memory-utilization`).
  **Ollama-beside-vLLM is the hard case** — Ollama takes what it wants with `keep_alive:-1`
  and declares nothing. That is precisely the 3070 Ti pairing wanted here.

### 3. VL embedder + VL reranker co-resident on one Spark

Measured on spark4, 2026-07-29:

| Model                   | declared | actual MiB | share of pool | median load |
| ----------------------- | -------: | ---------: | ------------: | ----------: |
| `qwen3-vl-embedding-8b` |     0.33 |     46,194 |         37.1% |     144.9 s |
| `qwen3-vl-reranker-8b`  |     0.35 |     61,945 |         49.7% |     115.8 s |
| **both**                | **0.68** | **~108 GB**|      **~88%** |           — |

88% is a proven-workable figure (gemma + embedder run at 84–88% on spark4 today), leaving
~13 GB for the second ceiling the troubleshooting runbook documents — Triton JIT, CUDA-graph
capture and prefill activations, none of which boot-time checks size. The node is full;
`vl32` moving to the 3090 is what makes that acceptable.

⚠️ **Launch order is load-bearing. The reranker must go FIRST, onto the empty node.** It
uses `load_format=fastsafetensors`, the eugr image's GPU-direct loader, which killed a
resident gemma in **2 of 2** attempts with `NVRM NV_ERR_NO_MEMORY` — yet boots clean solo in
~2 min every time. The embedder is plain safetensors and joins a resident safely. The
documented rules:

1. A **quantize-at-load** model must be the FIRST model on its node.
2. A **fastsafetensors** recipe must never launch BESIDE a resident.
3. Plain **safetensors** models join a resident safely.

**There is no primitive to express this.** `lane_defaults.yaml` holds one startup model per
unit, so "these two, in this order, at boot" cannot be stated. Same shape as the open
cookbook gap (multi-node groups skipped in snapshot and apply) — worth solving once rather
than twice.

### 4. Both GPUs pinned, both Spark roles pinned

Cold start on a Spark is 2–4 minutes, so the services node must be exempt from the idle
reaper and its models proven-loaded once to become auto-placement candidates.

---

## Options considered and not taken

### For vision / OCR

| Option                                         | Why not                                                                                          |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `surya2` alone on the 3090                     | Best text extraction, but ~3 GB of weights leaves a 24 GB card ~85% idle, and no visual reasoning |
| `vl7` alone (Qwen2.5-VL-7B **FP16**, 20.2 GB)  | FP16 for a 7B is wasteful; an AWQ 7B is ~6 GB. Superseded by putting `vl32` on the 3090           |
| `surya2` + a VLM co-resident on the 3090       | `vl32` at 21.5 GB leaves 2.5 GB — nothing fits beside it even with the lane rule dropped          |
| General VLM on a Spark, OCR on the 3090        | Was the recommendation until the 3070 Ti was brought into play; the two-card split is better      |

### For embeddings / rerank

| Option                                                       | Why not                                                                                                                                              |
| ------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Right-sized text models (Qwen3-Embedding / Reranker 0.6B–4B) | Two 0.6B models ≈ 35 GB (**mostly the 15 GiB × 2 overhead tax**), ~29% of a node — but loses image and page-level embedding, which cannot be added back without a full re-index |
| Embedder on the 3070 Ti via Ollama (`qwen3-embedding:0.6b`, 0.6 GB, already pulled) | Frees a Spark, but there is no good rerank story on an 8 GB card, and the 3070 Ti is now hosting `surya2`                            |
| Services node running embedder + reranker + a general VLM    | Three co-resident slots at ~55% of pool. Made unnecessary by `vl32` going to the 3090                                                                 |

---

## What RAG entails

Mostly **regular software**, with three model calls in it. The models are the small part.

**Indexing — offline, once per document:**

| Step                                     | Kind                              | Where              |
| ---------------------------------------- | --------------------------------- | ------------------ |
| Parse the document                       | software (+ OCR for scans)        | ingestion host / 3070 Ti |
| Chunk it — ~200–1000 tokens with overlap | software                          | ingestion host     |
| Embed each chunk → a vector              | **model** (embedder)              | Spark `.53`        |
| Store vector + text + metadata           | software (vector DB)              | ingestion host     |

**Querying — online, every question:**

| Step                                        | Kind                    | Where           |
| ------------------------------------------- | ----------------------- | --------------- |
| Embed the question                          | **model** (same embedder) | Spark `.53`   |
| ANN search — find ~50 nearest chunks        | software (the DB)       | ingestion host  |
| Optionally BM25 keyword search, fuse the two | software               | ingestion host  |
| Rerank those 50 → best 5                    | **model** (reranker)    | Spark `.53`     |
| Paste the 5 chunks into a prompt            | software                | ingestion host  |
| Answer the question                         | **model** (the LLM)     | 122B or DS4     |

So the services node hosts only steps 3 and 5 — **the embedder and reranker are the only
genuinely new models**. The generator is already in the plan.

### Three things worth knowing

**An embedder is a different kind of model from a chat LLM.** It does not generate text — it
maps a chunk to a fixed-length vector (typically 1024–4096 dims). Nothing to sample, no
temperature, no chat template. Hence `/v1/embeddings`, not `/v1/chat/completions`.

**The embedder is the most expensive thing in the stack to change.** Vectors are only
comparable to vectors from the *same* model, so swapping embedders invalidates the entire
index and forces a full re-embed. This is the strongest argument for keeping the VL embedder
at 46 GB rather than right-sizing to a text-only model: if the corpus ever grows to include
scanned pages, a text-only embedder means re-indexing everything.

**A reranker is a cross-encoder, not a second embedder.** It reads query and document
*together* and scores relevance directly — far more accurate than vector distance, but one
model call per candidate, so it only ever runs on the ~50 the ANN search returned.

### The software half

| Choice          | Option A                                                                     | Option B                                                                       |
| --------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Vector DB       | **pgvector** on the existing Postgres on `ubuntuservices` — nothing new to run, already backed up, SQL joins against existing metadata | **Qdrant** in a container beside it — better ANN, native hybrid search and payload filtering; one more service to run and back up |
| Orchestration   | **Hand-rolled**, ~200 lines. Keeps chunk size, overlap, top-k and rerank depth visible | **LlamaIndex / Haystack** — faster start, but the abstractions bury exactly those knobs |

Keep the state **off the Sparks** either way. A vector index is state; the Sparks are compute
appliances with no documented backup story.

**Suggested first corpus: the wiki itself.** 190 pages of clean markdown, already the thing
most often searched, and correctness is immediately checkable because the right answers are
known. Other candidates in the lab: paperless-ngx (on `ubuntucloud` and `sagubuntuservices`),
`store_expanse` (26,793 archived threads), the comics library, the code repos.

---

## Work items

Ordered roughly by dependency, not priority.

| #  | Item                                                                                                     | Blocks                        |
| -- | -------------------------------------------------------------------------------------------------------- | ----------------------------- |
| 1  | Build `serve-companion.sh`, torch-resolving the 3070 Ti by UUID; set `COMPANION_VLLM_ENABLED=true`       | `surya2` on the 3070 Ti       |
| 2  | Add a `surya2` case to `serve-companion.sh` and tune `--gpu-memory-utilization` / `max_pixels` for 8 GB  | same                          |
| 3  | Repoint epubocr's `SURYA_INFERENCE_URL` from `:11437` to `:11438`                                        | same                          |
| 4  | Port `mem_fraction`-style budgets to `Lane`; decide the replacement for the eviction-wait gate            | multi-model lanes             |
| 5  | Handle Ollama-beside-vLLM co-residency (Ollama declares no budget)                                        | `surya2` + `/swap` relay      |
| 6  | Add ordered multi-model boot to `lane_defaults.yaml` (reranker before embedder)                           | #4's node surviving a restart |
| 7  | Add `/v1/rerank` + `/v1/score` to the LLMConfig gateway                                                   | reranker as a first-class service |
| 8  | Constrain the Cluster tab to within-pair node sets; pin `NCCL_SOCKET_IFNAME` / Ray interfaces             | `SPARK_FABRIC_ENABLED=true`   |
| 9  | Set `SPARK_FABRIC_ENABLED=true`; first `tp=2` DS4 launch inside one pair                                  | role #2                       |
| 10 | Update `lane_defaults.yaml` so the 122B survives a restart by policy, not by refusal                      | role #1                       |
| 11 | Exempt the services node from the idle reaper; proven-load both models once                               | role #4                       |
| 12 | Stand up the vector DB + ingestion pipeline on `ubuntuservices`                                           | role #4                       |

### Existing gaps this design inherits

- **`lane_defaults.yaml` still names `gemma-4-26b` as spark1's startup model**, so the 122B
  currently survives a restart only by `needs_empty_node` refusal — the node is simply left
  empty and the 122B does not come back on its own.
- **`SPARK_FABRIC_ENABLED` is still false.** The fabric is up (two direct pairs, 196 Gb/s raw
  RDMA, 19.49 GB/s NCCL average) but the flag is held because the Cluster tab would otherwise
  offer cross-pair node sets that cannot communicate.
- **`num_speculative_tokens` never substitutes** in `@eugr/deepseek-v4-flash`, so MTP — the
  model's largest single uplift — is silently lost. Needs an upstream fix or a forked recipe.
- **Restarting LLMConfig reloads the whole fleet** via `autoload_defaults()`, with no flag to
  suppress it. Plan any deploy knowing it will change residency on units not being touched.

---

## Open questions

- **Corpus composition.** Not yet decided. It does not change the model choice now that the
  VL embedder is fixed, but it does drive chunking strategy and whether the OCR stage sits in
  the ingestion path at all.
- **Whether `vl32` stays pinned or remains idle-reapable** on the 3090. Pinned costs the card;
  reapable costs ~99 s on each cold reload.
- **Whether the 3070 Ti's `/swap` relay survives** the move, or whether `surya2` takes the
  card alone. Depends entirely on work item 5.

---

## Implementation record (2026-07-31)

Implemented in one day against the plan above, with these deltas:

| Doc item | Outcome |
| -------- | ------------------------------------------------------------------ |
| 1–3      | ✅ `serve-companion.sh` built; **surya2 lives on the 3070 Ti** at util 0.54, window 12288, max_pixels 2359296 (three profile iterations — WSL charges ~2 GiB non-torch CUDA context per process). epubocr repointed and verified (page OCR, mean_conf 0.90) |
| 4–5      | ✅ superseded by **`SlotLane`** (user decision: relay via vLLM too). Static per-slot budgets instead of a Lane `mem_fraction` port; the eviction gate narrowed per slot, not dropped. Ollama-beside-vLLM never built — OllamaCompanion disabled; relay = `qwen2.5-1.5b` AWQ slot (weights CPU-offloaded, fp8 KV) |
| 6        | ✅ boot autoload sorts by `schemas.boot_order_key` (needs_empty_node first, biggest budget next); cookbook `set_default` persists in that order |
| 7        | ✅ was already shipped (`/v1/rerank` + `/v1/score` existed before this work) |
| 8–9      | ✅ done 2026-07-30 (fabric pairs + DS4 tp=2, MTP fork) |
| 10       | ✅ lane defaults = the layout table below; spark1/2 are tombstones (the DS4 group returns via boot reclaim, not autoload) |
| 11       | ✅ reranker+embedder proven-loaded on spark4 (88% figure confirmed: 84.6%); all Sparks were already reaper-exempt |
| 12       | ✅ **done 2026-07-31** — `rivaborn/ragstack` on `ubuntuservices`: Qdrant (:6333) + FastAPI (:11440), corpus = the wiki DR replica (190 pages → 1,130 chunks, 253 s ingest, alias-flip re-index = the DR story). Golden eval **10/10** with ANN + page-dedupe; the reranker is OFF as served (non-discriminative — upstream recipe lacks the Qwen3 yes/no template; Spark-side follow-up) |

Extras the day demanded: `/lane/<unit>/v1` gateway prefix (surya's header-less
attach), gateway `/health`, `PRIMARY_IDLE_UNLOAD_ENABLED` (vl32 pinned without
the global reaper switch), serve.sh global-pkill cross-kill fix, fleet parity
(identical catalogs/weights/images on all four Sparks), SGLang verified on GB10
(`qwen35-35b-a3b`), `gpt-oss-120b-solo.yaml` (retires the unbuildable
vllm-node-mxfp4 dependency).

Final layout (cookbook state **`DF4_RAG Daily Driver`**):

```
pair A   spark1+spark2   deepseek-v4-flash  tp=2 (MTP fork)
         spark3          qwen35-122b-int4-nightly
         spark4          qwen3-vl-reranker-8b + qwen3-vl-embedding-8b  (84.6%)
3090                     vl32  (pinned — reaper exempt)
3070 Ti                  surya2 + qwen2.5-1.5b relay  (SlotLane, 97%)
```

**2026-07-31 update:** the `deepseek-v4-flash` alias now serves the **official
DeepSeek-V4-Flash-0731 release** (`recipes/local/deepseek-v4-flash-0731.yaml`
— FP8 156 G, spec-decode built in, DSpark draft on both workers, "Say OK"
5.2 s vs the preview's ~15 s; TerminalBench 82.7 vs 61.8 per the model card).
Layout, cookbook state, and every consumer unchanged — only the recipe behind
the alias moved. Weights staged on all four nodes; preview weights cached
until ~2026-08-07 as rollback.
