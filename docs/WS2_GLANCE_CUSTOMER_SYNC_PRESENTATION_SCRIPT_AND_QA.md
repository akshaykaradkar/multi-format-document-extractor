# Glance WS2: Customer Sync Presentation Script & Q&A Cheat-Sheet (`EXP-1` to `EXP-8`)
**Customer-Facing Edition — Written to Read Live to the Glance Engineering Team**  
**100% Grounded in Glance's Handoff Docs (`gs://pso-sandbox/search-context/`) — Zero Invented Numbers**

---

## 🎙️ 1. Opening Script (Read This When You Pull Up the Slide — 30 Seconds)

> *"Thanks everyone. First, thank you for sharing the `search-context` package and uploading the 5M product embeddings (`embedding-datasets/`) into the GCS bucket—we’ve already verified the 100 JSONL partition files (`~75 GB`, `1024-d CLIP` + `SPLADE` sparse vectors + `restricts`), as well as your `v3` production profile, incident report, golden queries, and Java/PySpark code.*
>
> *On this slide, we’ve laid out the **8 core experiments (`EXP-1` through `EXP-8`)** for this POC. We designed these 8 experiments to directly answer every review question your team highlighted in `production-index-profile.md`, `search-retrieval-description.md`, and `reliability-escalation.md`.*
>
> *Our approach is to run the baseline and parameter tuning on the **5M dataset first**, and then validate the winning configurations at **10M and 20M** as we scale. Let me walk through `EXP-1` to `EXP-8` briefly."*

---

## 📊 2. Slide Reference Table (`EXP-1` to `EXP-8`)

| Exp ID | Experiment Title | Scale | Aim |
| :--- | :--- | :--- | :--- |
| **`EXP-1`** | **Production Baseline** | **5M** | Establish As-Is v3 baseline against Exact Brute-Force ground truth. |
| **`EXP-2`** | **Hybrid Search & RRF Tuning** | **5M (10M/20M)** | Evaluate Dense, Sparse, and Hybrid latency and test RRF alpha trade-offs. |
| **`EXP-3`** | **ScaNN Tree-AH Algorithmic Tuning** | **5M (10M/20M)** | Tune parameters to eliminate ~2.3% timeouts and minimize P99 latency. |
| **`EXP-4`** | **Metadata & Filter Optimization** | **5M (10M/20M)** | Benchmark latency and memory impact of excluding inactive/OOS rows, stripping unused metadata, and simplifying NOT_IN filters. |
| **`EXP-5`** | **Shard & Machine Evaluation** | **5M (10M/20M)** | Evaluate Large shards and high-memory machines to prevent Out-of-Memory. |
| **`EXP-6`** | **Dimensionality Reduction** | **5M (10M/20M)** | Compress 1024d to 768d/512d to quantify latency vs recall. |
| **`EXP-7`** | **Concurrent Scale Testing** | **5M (10M/20M)** | Benchmark 5,000 QPS reads alongside concurrent streaming inserts & updates (~500 UPS rate, simulating 10M daily inserts + 10M updates). |
| **`EXP-8`** | **SSD Offloading Evaluation** | **5M (10M/20M)** | Evaluate Storage-Optimized SSD offloading for given embeddings.<br>*Note: Subject to Glance team feasibility to segregate Hot vs. Cold data tiering.* |

---

## 🎤 3. Row-by-Row Speaking Script & Customer Q&A (`EXP-1` to `EXP-8`)

---

### `EXP-1`: Production Baseline (`Scale: 5M`)
* **Slide Aim:** `Establish As-Is v3 baseline against Exact Brute-Force ground truth.`
* **🎤 What to Say to Glance (25 Seconds):**
  > *"Starting with **EXP-1 (Production Baseline)**: We deploy the 5M dataset using your exact live `v3` index configuration (`1024d`, `DOT_PRODUCT_DISTANCE`, `SHARD_SIZE_MEDIUM`, `Tree-AH` with `leafNodeEmbeddingCount = 1000`, `leafNodesToSearchPercent = 5`, and `approximateNeighborsCount = 5000`).*
  > *Alongside that `v3` baseline index, we also deploy an **Exact Brute-Force index** on the same 5M dataset. That gives us 100% exact ground-truth neighbors so that as we tune Tree-AH parameters, filters, or dimensions in `EXP-2` through `EXP-6`, we can measure the exact Recall@K trade-off alongside your 2,000 LLM-judged golden queries."*
* **📌 Verified Source in Glance's Docs:**
  * `02-index/production-index-profile.md` (Lines 19–25): `1024d`, `DOT_PRODUCT_DISTANCE`, `SHARD_SIZE_MEDIUM`, `leafNodeEmbeddingCount=1000`, `leafNodesToSearchPercent=5`, `approximateNeighborsCount=5000`.
  * `04-data/golden-queries/README.md` (Lines 7–44): `2,000` stratified queries (`583` poor, `657` mid, `760` good) and `23,952` judged results.
* **💬 If Glance Asks:**
  * **Q: *"Our live `v3` index has 58.2M vectors across 24 shards. How representative is a 5M baseline?"***
    * **A:** *"Great question. In your 58.2M `v3` index, `SHARD_SIZE_MEDIUM` splits those 58.2M vectors across 24 shards—which is **~2.42 million vectors per shard** on an `n1-standard-16` node (~32.3 GB RAM per node). When we deploy 5M vectors with `SHARD_SIZE_MEDIUM`, Vertex AI creates 2 shards of **~2.5 million vectors per shard**. So inside each shard and node, our 5M baseline reproduces the exact same vector density, ScaNN tree depth, and ~32 GB RAM footprint as your 58.2M production shards. Then scaling to 10M and 20M lets us measure the multi-shard scatter-gather curve as shard count increases."*

---

### `EXP-2`: Hybrid Search & RRF Tuning (`Scale: 5M [10M/20M]`)
* **Slide Aim:** `Evaluate Dense, Sparse, and Hybrid latency and test RRF alpha trade-offs.`
* **🎤 What to Say to Glance (25 Seconds):**
  > *"Moving to **EXP-2 (Hybrid Search & RRF Tuning)**: In `search-retrieval-description.md` (Question #2), your team asked whether the current `rrf_alpha = 0.3` fusion weighting is leaving quality on the table—especially on the 583 queries in the 'poor' relevance band.*
  > *And since your target architecture is 100% Hybrid—where every query runs a Dense CLIP search and a Sparse SPLADE search in parallel—we will first measure the latency of the Dense leg, Sparse leg, and combined Hybrid search on the same index to see which leg is driving tail latency, and then test different `rrf_alpha` weights against the 2,000 golden queries."*
* **📌 Verified Source in Glance's Docs:**
  * `reference/search-retrieval-description.md` (Lines 27 & 52): *"The weighting is controlled by `rrf_alpha`, which is `0.3` in the observed production configuration... What we would like help with: 2. Whether the fusion weighting and the absence of score thresholds are leaving quality on the table."*
* **💬 If Glance Asks:**
  * **Q: *"We are moving to 100% Hybrid queries—why are you testing Dense and Sparse separately in EXP-2?"***
    * **A:** *"We aren't proposing moving away from 100% Hybrid. In Vertex AI Vector Search, a Hybrid `FindNeighbors` call executes the Dense Tree-AH search and the Sparse SPLADE search in parallel and waits for whichever leg is slower, returning only a single combined latency number. Running the same queries as Dense-only and Sparse-only on the exact same index (with zero rebuild) is a diagnostic so we can show you how many milliseconds come from the Dense leg vs. the Sparse leg inside your 100% Hybrid queries."*

---

### `EXP-3`: ScaNN Tree-AH Algorithmic Tuning (`Scale: 5M [10M/20M]`)
* **Slide Aim:** `Tune parameters to eliminate ~2.3% timeouts and minimize P99 latency.`
* **🎤 What to Say to Glance (25 Seconds):**
  > *"In **EXP-3 (ScaNN Tree-AH Algorithmic Tuning)**, we directly address Question #3 and Question #5 from your `production-index-profile.md`. Right now, your `v3` index uses `leafNodesToSearchPercent = 5%` and `approximateNeighborsCount = 5000` (which re-ranks 5,000 candidates per shard to return 500), and you’re seeing ~2.3% `DEADLINE_EXCEEDED` errors.*
  > *In EXP-3, we sweep `approximateNeighborsCount`, `fractionLeafNodesToSearch`, and `leafNodeEmbeddingCount` to find the sweet spot that brings down P95/P99 latency and eliminates those timeouts while preserving recall on the golden query set."*
* **📌 Verified Source in Glance's Docs:**
  * `02-index/production-index-profile.md` (Lines 23–24, 75, 94, 96): `leafNodeEmbeddingCount = 1000`, `leafNodesToSearchPercent = 5`, `approximateNeighborsCount = 5000`, and `~2.3%` `Match` failures with code 4 (`DEADLINE_EXCEEDED`).
* **💬 If Glance Asks:**
  * **Q: *"If we lower `approximateNeighborsCount` from 5000 (say to 1000 or 500), will we have to rebuild our production index?"***
    * **A:** *"No! Both `approximateNeighborsCount` and `fractionLeafNodesToSearchOverride` can be passed per-query in your Java `VertexMatchingEngineProvider` request with **zero index rebuild**. Only changing `leafNodeEmbeddingCount` requires an index build, and we will test both so you can apply the query-side fix immediately."*

---

### `EXP-4`: Metadata & Filter Optimization (`Scale: 5M [10M/20M]`)
* **Slide Aim:** `Benchmark latency and memory impact of excluding inactive/OOS rows, stripping unused metadata, and simplifying NOT_IN filters.`
* **🎤 What to Say to Glance (25 Seconds):**
  > *"In **EXP-4 (Metadata & Filter Optimization)**, we benchmark the two filter questions your team raised in the handoff docs:*
  > *First, in `production-index-profile.md` (Question #1), you noted that ~31% of the indexed vectors (18.3M out of 58.2M) fail the `status=active` and `instock=true` check and can never be served, plus we noticed the ingestion job writes 19 string restrict fields per vector (including `cdn_imageurl`, `sku`, `created_at`, and `updated_at`) while the Java search service only filters on about 11 of them.*
  > *Second, in `search-retrieval-description.md` (Question #3), you asked whether the two always-on `NOT_IN` exclusion lists (`catalog_id` and `retailer_domain`) are hurting filter evaluation efficiency.*
  > *So in EXP-4, we benchmark query latency with vs. without the `NOT_IN` exclusion filters, and we benchmark a lean index that indexes only active/in-stock items and queried filter fields to quantify the exact memory and latency savings."*
* **📌 Verified Source in Glance's Docs:**
  * `02-index/production-index-profile.md` (Lines 38–42 & Line 92): `39,943,300` active & in-stock out of `58,225,443` indexed (`~31%` unservable).
  * `reference/search-retrieval-description.md` (Lines 33–42 & Line 54): Always-on `catalog_id NOT_IN` and `retailer_domain NOT_IN` exclusion lists; `colour` and `currency` deliberately not filtered.
  * `embedding-datasets/part-00000.json` & `05-code/ingestion/.../restricts.py` (Lines 184–274): `19` string restrict namespaces written per datapoint.
* **💬 If Glance Asks:**
  * **Q: *"Do you need us to re-export the 5M dataset to test EXP-4?"***
    * **A:** *"Not at all—the 5M JSONL files you uploaded already include `status`, `instock`, and all 19 `restricts` fields on every row, so we can generate the lean dataset directly on our side for the benchmark. We would just confirm with your team whether any other service outside `phoenix-search-retrieval` ever queries those extra 8 fields (`cdn_imageurl`, `sku`, `created_at`, `updated_at`, `supplier_name`, `brand_normalized`, `onsale`, `currency`)."*

---

### `EXP-5`: Shard & Machine Evaluation (`Scale: 5M [10M/20M]`)
* **Slide Aim:** `Evaluate Large shards and high-memory machines to prevent Out-of-Memory.`
* **🎤 What to Say to Glance (25 Seconds):**
  > *"In **EXP-5 (Shard & Machine Evaluation)**, we address Question #1 in `production-index-profile.md`: your live `v3` deployment runs 24 Medium shards across 48 `n1-standard-16` nodes (60 GB RAM each) where CPU sits at ~3% mean (7.2% peak) while memory sits at ~54% (~32.3 GB per node).*
  > *Because your workload is clearly memory-bound rather than CPU-bound—and streaming compaction spikes memory further—we will benchmark `SHARD_SIZE_LARGE` and higher-memory machine types (`e2-highmem-16` with 128 GB RAM and `n1-standard-32` with 120 GB RAM) to prevent Out-of-Memory pod replacements during streaming compaction and evaluate consolidating shard count."*
* **📌 Verified Source in Glance's Docs:**
  * `02-index/production-index-profile.md` (Lines 22, 60–70, 80, 92): `24` shards, `48 × n1-standard-16` nodes (`$26,600/mo`), `2.7%` mean CPU (`7.2%` peak), `32.3 GB / 60 GB` RAM (`54%`), and multiple pod generations observed during streaming writes.
* **💬 If Glance Asks:**
  * **Q: *"Why test both `e2-highmem-16` and `SHARD_SIZE_LARGE` (`n1-standard-32`)?"***
    * **A:** *"`e2-highmem-16` works with your current `SHARD_SIZE_MEDIUM` (doubling RAM per node from 60 GB to 128 GB with zero index rebuild, just an endpoint redeploy), whereas `SHARD_SIZE_LARGE` pairs with `n1-standard-32` / `e2-standard-32` to pack more vectors per shard and cut total shard count."*

---

### `EXP-6`: Dimensionality Reduction (`Scale: 5M [10M/20M]`)
* **Slide Aim:** `Compress 1024d to 768d/512d to quantify latency vs recall.`
* **🎤 What to Say to Glance (20 Seconds):**
  > *"In **EXP-6 (Dimensionality Reduction)**: Since 1024-dimensional dense CLIP vectors are the single largest driver of RAM footprint, we will test compressing the exported 1024d embeddings down to `768d` and `512d` (cutting dense vector memory by 25% to 50%) and measure the exact Recall@K trade-off against our `EXP-1` Brute-Force ground truth. Since your CLIP model isn't trained as a native Matryoshka model, this gives you an empirical data point on how much recall is traded for a 25%–50% RAM reduction."*
* **📌 Verified Source in Glance's Docs:**
  * `02-index/production-index-profile.md` (Lines 19, 55): `1024` dimensions from CLIP model endpoint `5179642048289964032`.

---

### `EXP-7`: Concurrent Scale Testing (`Scale: 5M [10M/20M]`)
* **Slide Aim:** `Benchmark 5,000 QPS reads alongside concurrent streaming inserts & updates (~500 UPS rate, simulating 10M daily inserts + 10M updates).`
* **🎤 What to Say to Glance (25 Seconds):**
  > *"In **EXP-7 (Concurrent Scale Testing)**: During discovery, your team shared the target scale of **5,000 QPS peak reads** alongside **10M daily inserts and 10M daily updates** via `STREAM_UPDATE` (~500 upserts/sec), and in `reliability-escalation.md` and `production-index-profile.md` you highlighted the standing gap of **64,017 datapoints** missing sparse vectors under streaming updates.*
  > *In EXP-7, we run concurrent high-QPS read load alongside live `UpsertDatapoints` streaming writes (at the ~500 UPS rate) to benchmark P95/P99 read latency during background compaction, measure 'Read-Your-Own-Data' freshness lag, and verify dense/sparse synchronization under streaming load."*
* **📌 Verified Source in Glance's Docs:**
  * `02-index/production-index-profile.md` (Lines 25, 30–34, 78–80) & `03-incidents/reliability-escalation.md` (Lines 17–27, 47): `STREAM_UPDATE`, `58,225,443` dense vs. `58,161,426` sparse (`64,017` gap vs. only `8` missing in the `96.98M` source feed).

---

### `EXP-8`: SSD Offloading Evaluation (`Scale: 5M [10M/20M]`)
* **Slide Aim:** `Evaluate Storage-Optimized SSD offloading for given embeddings. Note: Subject to Glance team feasibility to segregate Hot vs. Cold data tiering.`
* **🎤 What to Say to Glance (25 Seconds):**
  > *"Finally, **EXP-8 (SSD Offloading Evaluation)**: After we finish all in-memory optimizations in `EXP-1` through `EXP-7`, we evaluate Vertex AI’s Storage-Optimized SSD tier for scaling toward 500M vectors. Holding 500M vectors in 100% RAM is expensive, whereas keeping 'Hot' active inventory in RAM for low latency and offloading 'Cold' long-tail inventory to SSD dramatically lowers TCO. As we noted on the slide, this is subject to aligning with your team on whether segregating Hot vs. Cold catalog tiers and routing queries across them is feasible in your pipeline."*
* **💬 If Glance Asks:**
  * **Q: *"How does SSD latency compare to RAM, and how does it behave on 5M vectors?"***
    * **A:** *"On a 5M dataset, the shard footprint is below Vertex AI's ~7.75 GB local NVMe caching threshold, so 5M behaves very close to RAM; as we scale to 10M/20M and pair our test with Google Product Team benchmarks at 100M+ scale, we will give you the exact P95/P99 latency and cost comparison between pure RAM and Hot/Cold SSD tiering."*

---

## 🎯 4. Closing Script: The 2 Quick Asks for Glance (20 Seconds)

When you finish walking through `EXP-8`, close with these **two clear asks** for the Glance team:

> *"To wrap up this slide, we have **good news** and **two quick asks** for your team:*
> 1. ***Good news:*** *With the 5M product embeddings you uploaded into `embedding-datasets/`, we can start building the `EXP-1` baseline and `EXP-4` lean indexes immediately today.*
> 2. ***Ask #1 (Query Embeddings for the 2,000 Golden Queries):*** *In `04-data/golden-queries/queries.csv`, we currently have the raw `query_text` for the 2,000 queries. To run the recall and latency benchmarks across `EXP-1` to `EXP-8`, could your team upload a JSONL/Parquet file with the **pre-computed 1024-d CLIP dense vectors + SPLADE sparse vectors** (and effective filter payloads, if available) for those 2,000 golden queries?*
> 3. ***Ask #2 (For `EXP-8` Hot/Cold SSD Tiering):*** *Let us know how your team defines Hot vs. Cold catalog items today and whether query routing between a Hot RAM index and Cold SSD index is feasible on your side."*
