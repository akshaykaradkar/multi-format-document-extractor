# Glance WS2: Sync Call Speaking Script & Q&A Cheat-Sheet (`EXP-0` to `EXP-7`)

**Prepared For:** Akshay Karadkar  
**Purpose:** Live speaking guide and Q&A reference sheet matching the **exact `EXP-0` through `EXP-7` table** shared with the team. If anyone on the sync call (Madhu, Anita, or Glance) asks *"Why do we need this experiment?"*, *"How are we going to run it?"*, or *"Why is this V1.0 vs. V2.0?"*, you can read these answers directly.

---

## 1. Quick Reference: Your Shared Table (`EXP-0` to `EXP-7`)

| Exp ID | Experiment Title | Track | Scale (`5M` First $\rightarrow$ Repeat) | Aim |
| :--- | :--- | :---: | :--- | :--- |
| **EXP-0** | **Production Baseline ("As-Is" `v3`) & SOW Brute-Force Benchmark** | **V1.0 & V2.0** | `5M` *(repeat on `10M/15M/20M`)* | Establish the Step 0 "As-Is" Tree-AH Control baseline AND benchmark **Exact Brute-Force (`BruteForceConfig` / V2 Exact `kNN` 100% recall) vs. ANN** latency, recall & cost as required by SOW. |
| **EXP-1** | **Query Mode Isolation & Dense/Sparse Embedding Combination Split (`80/20, 70/30, 50/50, 30/70, 20/80`)** | **V1.0 & V2.0** | `5M` *(repeat on `10M/15M/20M`)* | Isolate `Dense-Only` (`<100ms P95`) vs. `Sparse-Only` vs. `Hybrid` (`<150ms P95`), and sweep **Dense/Sparse Embedding Combination Splits (`80/20, 70/30, 50/50, 30/70 [Prod], 20/80` / `rrf_alpha` & V2 `RRFRanker`)**. |
| **EXP-2** | **Tree-AH Hyperparameter & Zero-Rebuild Per-Query Override Sweep** | **V1.0** | `5M` *(repeat on `10M/15M/20M`)* | Eliminate the `2.3%` `DEADLINE_EXCEEDED` timeouts by sweeping zero-rebuild per-query `approximate_neighbor_count` (`5000 -> 200`), `fraction_leaf_nodes_to_search_override` (`2%..10%`), and build `leafNodeEmbeddingCount` (`500..2000`). |
| **EXP-3** | **Filter Hygiene, Deny-List Ablation, Country Sharding (`US/IN/JP`) & Two-Tier Retrieval** | **V1.0 & V2.0** | `5M` *(repeat on `10M/15M/20M`)* | Remove static `NOT_IN` deny-lists, normalize case-variant restricts, fix `gender=na` starvation on `47` non-fashion queries, test Country-Partitioned indexes (`US/IN/JP`), and benchmark Two-Tier post-filtering. |
| **EXP-4** | **Shard Consolidation (`SHARD_SIZE_LARGE`), Whitelisted Machines, `5,000 QPS` Hot-Key Skew & PSC** | **V1.0 & V2.0** | `5M` *(projected to `58M` & `500M`)* | Consolidate from 24 `MEDIUM` shards (`n1-standard-16`) onto whitelisted `SHARD_SIZE_LARGE` (`e2-highmem-16` [128GB RAM] / `n1-standard-32` [120GB RAM]), run Script 15's `5,000 QPS` Hot-Key Skew Stress Test, and test PSC vs. Public Endpoint. |
| **EXP-5** | **Dimensionality Reduction (`1024 -> 768 -> 512 -> 256` PCA/OPQ) & Quantization (`SQ8` vs `TurboQuant`)** | **V1.0 & V2.0** | `5M` *(repeat on `10M/15M/20M`)* | Measure exact Latency, Brute-Force `Recall@K`, Golden Query `NDCG@10`, and RAM savings when stepping down dimensions **`1024 -> 768 -> 512 -> 256`** and enabling native **`SQ8`** (`4x–6x` total RAM reduction). |
| **EXP-6** | **Concurrent `5,000 QPS` Reads + Streaming Updates (`500 UPS`, `10M Inserts + 10M Updates`), "Read-Your-Own-Data"** | **V1.0 & V2.0** | `5M` *(repeat on `10M/15M/20M`)* | Run Script 16's 3-Engine Test: `5,000 QPS` reads + `20M/day` streaming writes (`10M daily inserts + 10M daily updates` at `500 UPS` / `1,626/s` burst) + "Read-Your-Own-Data" (RYOW) freshness lag + reproduce/fix `Crowding attribute mismatch` & `64,017` sparse gap. |
| **EXP-7** | **Final Stage: Hot/Cold Catalog Segregation & SSD Offloading Evaluation (`500M`)** | **V1.0 & V2.0** | `5M -> 20M` + `500M` Model | Evaluate Hot (`active+instock` in RAM) vs. Cold (`OOS/tail` offloaded/SSD) tiering and `~7.75 GB` shard threshold modeling for `500M`. |

---

## 2. General Cross-Cutting Questions (If Asked Before Diving Into Individual Rows)

### Q1: *"Why does every experiment say `5M (repeat on 10M/15M/20M)`? What if Glance only gives us 5M and never uploads 10M, 15M, or 20M?"*
> **Your Answer:**  
> *"Right now, Glance hasn't uploaded the embeddings yet because even 5M takes time for them to export. So we designed our entire playbook so **100% of `EXP-0` through `EXP-7` runs end-to-end on the `5M` dataset first**.  
> Second, even if Glance only ever gives us `5M`, we built a **`5M -> 500M` CPU-Equivalence trick in Script 16**: on a `50M`-vector shard at `500M` scale, searching `3%` of leaves scans **`1.5 million` vectors per query**. On our `5M` dataset, we pass a `30%` leaf-search override (`30%` of `5M` = **`1.5 million` vectors scanned per query**), which forces the exact same single-shard CPU and RAM scan work on real Glance data without needing synthetic filler!  
> And if Glance *does* upload `10M`, `15M`, or `20M` later, we don't change a single script—we simply re-run the same pipeline on those slices to plot the multi-point scaling curve."*

### Q2: *"Why are `EXP-0` through `EXP-6` all In-Memory (`RAM-Only`), and why is Optimized Storage / SSD (`EXP-7`) kept at the very end?"*
> **Your Answer:**  
> *"Two reasons, which we also aligned on in the Sept 28 sync:  
> First, if we mix SSD disk I/O with our tuning experiments, we won't know if a latency spike came from disk I/O or from our Tree-AH / filter / streaming parameters. So `EXP-0` through `EXP-6` are strictly **In-Memory (`RAM-Only`)** to get clean baselines.  
> Second, as discussed on Sept 28, a `5M` sample is smaller than the `~7.75 GB` shard threshold needed to trigger physical SSD offloading anyway. So we keep Hot/Cold segregation and SSD offloading at the very end (`EXP-7`) for our `500M` TCO model."*

---

## 3. Experiment-by-Experiment Speaking Guide & Q&A (`EXP-0` to `EXP-7`)

---

### `EXP-0`: Production Baseline ("As-Is" `v3`) & SOW Brute-Force Benchmark (`V1.0 & V2.0`)

* **1. Why We Need It (30-Second Pitch):**
  > *"Before we change any parameter, we need a 'Step 0 Control Baseline' that replicates Glance's exact current production `v3` setup (`1024-d CLIP` + `SPLADE`, `30/70` split, `leaf=1000`, `search=5%`, `approxNeighbors=5000`). That way, every improvement we make in `EXP-1` to `EXP-7` is measured as a clean delta ($\Delta$ Latency, $\Delta$ Recall, $\Delta$ Cost) against `EXP-0`. At the same time, the SOW explicitly asks us to benchmark **Brute Force (`100%` exact recall) vs. ANN (`Tree-AH`)**."*
* **2. How We Will Do It (`Script 2`, `Script 11`, `Script 12`):**
  - **Step 1 (`Script 11`):** Profile the `5M` dataset (verify dense vectors are unit L2-normalized and all `13,990` judged golden-query products are inside the `5M` sample).
  - **Step 2 (`Script 2` & `Script 12` — Brute Force):** Run **100% Exact Brute-Force Search** (`BruteForceConfig` in V1.0 and Exact `kNN` search in V2.0) on the `2,000` Golden Queries. This gives us the **100% exact top-500 ground-truth neighbors** and Brute-Force latency.
  - **Step 3:** Deploy Glance's exact "As-Is" `v3` ANN index and measure its `Recall@10/50/100/500` (compared against Brute Force), `p50/p95/p99` latency, and `DEADLINE_EXCEEDED %`.
* **3. Likely Questions & Your Ready Answers:**
  - **Q: *"Why do we need Brute Force if Brute Force is too slow for 500M in production?"***  
    **A:** *"Two reasons: (1) It is an explicit deliverable in the SOW to show Glance the latency and cost gap between Brute Force (`100%` exact) and ANN (`Tree-AH`), and (2) you cannot calculate `Recall@K` of an approximate index unless you first run Brute Force to know what the true 100% exact top-K neighbors actually are!"*
  - **Q: *"How do we do `EXP-0` in both V1.0 and V2.0?"***  
    **A:** *"In V1.0, we compare `BruteForceConfig` vs. `TreeAhConfig`. In V2.0, querying a `Collection` before attaching an index runs native Exact `kNN` (Brute Force), and attaching a V2.0 Index runs ANN. So we get the exact same comparison in both."*

---

### `EXP-1`: Query Mode Isolation & Dense/Sparse Embedding Combination Split (`80/20, 70/30, 50/50, 30/70, 20/80`) (`V1.0 & V2.0`)

* **1. Why We Need It (30-Second Pitch):**
  > *"First, the SOW has separate latency SLAs for **Dense-Only (`P95 < 100ms`)** and **Hybrid (`P95 < 150ms`)**, so we must test `Dense-Only`, `Sparse-Only`, and `Hybrid` separately.  
  > Second, in Glance's handover docs (`search-retrieval-description.md`), they asked if their current **`30% Dense / 70% Sparse` (`rrf_alpha = 0.30`)** split is hurting relevance on their `583` `poor` golden queries. When we analyzed their `2,000` golden queries, we found that giving `70%` weight to SPLADE Sparse causes keyword collisions—like searching `'velvet saree party'` and getting velvet shoes, or long VLM `'outfit photo model...'` queries matching background words like `'stadium'`. Sweeping **`80/20, 70/30, 50/50, 30/70, 20/80`** directly solves this."*
* **2. How We Will Do It (`Script 12`):**
  - Run the `2,000` Golden Queries across 3 modes (`100% Dense`, `100% Sparse`, `Hybrid`) to verify the `<100ms` and `<150ms` SOW SLAs.
  - Sweep the Dense (`CLIP`) / Sparse (`SPLADE`) percentage combinations: **`80/20` (`alpha=0.8`), `70/30` (`0.7`), `60/40` (`0.6`), `50/50` (`0.5`), `40/60` (`0.4`), `30/70` (`0.3` - current prod), and `20/80` (`0.2`)** in V1.0 (`rrf_alpha`) and V2.0 (`RRFRanker`).
  - Test **Dynamic Query-Length Routing**: using `30/70` (sparse-heavy) for 1–2 word brand/SKU queries, `50/50` for medium queries, and `80/20` (dense-heavy) for 6+ word semantic/VLM queries.
* **3. Likely Questions & Your Ready Answers:**
  - **Q: *"Does changing the embedding split (`80/20` vs `30/70`) require rebuilding the index?"***  
    **A:** *"No! Zero index rebuild. `rrf_alpha` (in V1.0) and `RRFRanker` weights (in V2.0) are passed per-query at search time."*
  - **Q: *"How will we know if `70/30` or `80/20` actually improved relevance on the `2,000` Golden Queries?"***  
    **A:** *"We compare every split against Glance's Step 0 Golden Query baseline (`Global Score@5 = 2.008`, `NDCG@10 = 0.600`, and `poor`-band `Score@5 = 0.780`) using Brute-Force Recall@K, retention of known good products (`score >= 2`), and LLM-as-a-Judge (`0–3` scale) on newly surfaced top-10 results."*

---

### `EXP-2`: Tree-AH Hyperparameter & Zero-Rebuild Per-Query Override Sweep (`V1.0`)

* **1. Why We Need It (30-Second Pitch):**
  > *"In Glance's `production-index-profile.md`, they asked why **`2.3%` of their search calls fail with `DEADLINE_EXCEEDED` timeouts even though CPU is at `2.7%`**, and whether `leafNodesToSearchPercent = 5` and `approximateNeighborsCount = 5000` are right.  
  > When we inspected their Java code (`VertexMatchingEngineProvider.java`), we found that every query asks for `500` neighbors (`numNeighbors = 500`), but inherits the index's `approximateNeighborsCount = 5000`—which forces each shard to read and dot-product `5,000 × 4 KB = 20 MB` of full `FP32` vectors (`480 MB` across 24 shards per query!). In `EXP-2`, we sweep these knobs to eliminate that 10x over-fetch."*
* **2. How We Will Do It (`Script 14`):**
  - Use Vertex AI V1.0's **runtime per-query override fields** (`Query.approximate_neighbor_count` swept from `5000 -> 2000 -> 1000 -> 500 -> 300 -> 200` and `Query.fraction_leaf_nodes_to_search_override` swept from `2% -> 3% -> 5% -> 7% -> 10%`) with **zero index rebuild**.
  - Also test index-build `leafNodeEmbeddingCount` (`500 vs 1000 vs 2000`).
* **3. Likely Questions & Your Ready Answers:**
  - **Q: *"Why is `EXP-2` marked `V1.0` only instead of `V1.0 & V2.0`?"***  
    **A:** *"Because `leafNodeEmbeddingCount`, `leafNodesToSearchPercent`, and `approximateNeighborsCount = 5000` are manual V1.0 `Tree-AH` parameters. In Vector Search 2.0, the index is **autotuned automatically** by Google, so those manual legacy knobs don't exist in V2.0—instead, we compare our tuned V1.0 `Tree-AH` index against V2.0's autotuned index."*
  - **Q: *"Can Glance apply the `EXP-2` fix to their live 58.2M production index right away without rebuilding it?"***  
    **A:** *"Yes! Because `setApproximateNeighborCount(int)` and `setFractionLeafNodesToSearchOverride(double)` are runtime per-query proto fields in V1.0, Glance can deploy the winning setting (e.g., `500` or `1000` instead of `5000`) with a 2-line code change in `VertexMatchingEngineProvider.java` and zero index rebuild."*

---

### `EXP-3`: Filter Hygiene, Deny-List Ablation, Country Sharding (`US/IN/JP`) & Two-Tier Retrieval (`V1.0 & V2.0`)

* **1. Why We Need It (30-Second Pitch):**
  > *"Glance runs 5 mandatory filters on 100% of queries (including two `NOT_IN` deny-lists for `catalog_id` and `retailer_domain`, plus dual-case `country IN ['IN', 'in']`), plus 4 optional filters (`gender, category, brand [194k values], price`). Furthermore, in their ingestion code we found two major issues: (1) **`31.4%` of their index (`18.3M` vectors) is inactive or out-of-stock** sitting in RAM, plus every row indexes microsecond `created_at`/`updated_at` timestamps that are never queried; and (2) `47` non-fashion/home golden queries fail (`Score@5 < 0.8`) because strict `gender` filters block `gender=na` home decor items. `EXP-3` fixes all of these and tests **Anita's Two-Tier Filtering** (Vertex AI + Bigtable post-filtering)."*
* **2. How We Will Do It (`Script 11` & `Script 14`):**
  - **Step 1 (Corpus & Filter Hygiene):** Purge the `31.4%` inactive/OOS vectors and unused `created_at`/`updated_at` tokens from RAM; remove the static `NOT_IN` deny-lists at ingestion; normalize `["IN", "in"] -> ["in"]`; and add `"na"` to `gender` (`gender IN [g, "na"]`) for non-fashion queries.
  - **Step 2 (Country Sharding `US / IN / JP`):** Since `US` is `67.4%`, `IN` is `17.9%`, and `JP` is `14.7%` of the catalog and every query filters by 1 country, compare a Global Index vs. Country-Partitioned Indexes (`IN` queries scan 5x fewer vectors).
  - **Step 3 (Single-Stage vs. Two-Tier Retrieval):** Compare evaluating all filters inside Vertex AI vs. **Anita's Two-Tier Architecture** (keeping low-cardinality filters in Vertex AI and evaluating `194k` `brand` + dynamic `price` in parallelized Bigtable hydration; and in V2.0, testing V2.0's native SQL/CEL filters + `~150B` projected metadata payload which skips the Bigtable hop).
* **3. Likely Questions & Your Ready Answers:**
  - **Q: *"Why did we include Country Sharding (`US/IN/JP`) here?"***  
    **A:** *"In Glance's `production-index-profile.md`, they showed that `US` makes up `67.4%` of the index while `IN` is `17.9%` and `JP` is `14.7%`, and 100% of queries filter by country. Today, when an India user searches, over 82% of vectors in every leaf node belong to the US and get discarded. Testing Country-Partitioned indexes shows how much faster and higher-recall `IN` and `JP` queries become when they don't scan US items."*

---

### `EXP-4`: Shard Consolidation (`SHARD_SIZE_LARGE`), Whitelisted Machines, `5,000 QPS` Hot-Key Skew & PSC (`V1.0 & V2.0`)

* **1. Why We Need It (30-Second Pitch):**
  > *"Glance's #1 question in `production-index-profile.md` is: **'We run 24 shards × 2 replicas = 48 `n1-standard-16` nodes (`$26,600/mo`) at `2.7%` CPU and `54%` memory. Is 24 shards at 2 replicas the right shape?'**  
  > Because they are memory-bound (`54%` RAM) and not CPU-bound (`2.7%` CPU), running 24 small shards wastes CPU and forces every query to fan out across 24 shards—where a tiny `0.1%` stall on any 1 shard becomes a **$1 - (1 - 0.001)^{24} = 2.37\%$** timeout rate across 24 shards! Plus, they currently use a **Public Endpoint (`*.vdb.vertexai.goog`)** which suffered shared proxy CPU saturation in P1 Incidents #1 and #3."*
* **2. How We Will Do It (`Script 9`, `Script 13`, `Script 15`):**
  - **Shard & Machine Consolidation (V1.0 `Script 13`):** Switch from `SHARD_SIZE_MEDIUM` (`n1-standard-16`: `16 vCPU, 60 GB RAM`) to **`SHARD_SIZE_LARGE` on whitelisted `e2-highmem-16` (`16 vCPU, 128 GB RAM` — +113% RAM per node with the exact same 16 vCPUs!)** and `n1-standard-32` (`32 vCPU, 120 GB RAM`). Combined with purging the `31.4%` dead vectors and `SQ8`, this consolidates 24 shards down to **`4–6` shards**!
  - **Replica Scaling & `5,000 QPS` Hot-Key Flash-Sale Skew (`Script 15` on V1.0 & V2.0):** Ramp load from `67 -> 189 -> 1,000 -> 5,000 QPS` under both uniform traffic and **Zipfian Hot-Key Skew** (thousands of concurrent queries hitting the same viral category/brand).
  - **PSC vs. Public Endpoint (`Script 9`):** Benchmark Private Service Connect (PSC) / Direct VPC Peering vs. Public Endpoint (`*.vdb.vertexai.goog`).
* **3. Likely Questions & Your Ready Answers:**
  - **Q: *"Why specifically `e2-highmem-16` and `n1-standard-32` for `SHARD_SIZE_LARGE`?"***  
    **A:** *"Vertex AI Vector Search V1.0 strictly whitelists only two machine types for `SHARD_SIZE_LARGE`: `e2-highmem-16` (`16 vCPU, 128 GB RAM`) and `n1-standard-32` (`32 vCPU, 120 GB RAM`). Since Glance is only using `2.7%` of their 16 vCPUs today, `e2-highmem-16` more than doubles RAM (`60 GB -> 128 GB`) without paying for 16 extra idle vCPUs."*

---

### `EXP-5`: Dimensionality Reduction (`1024 -> 768 -> 512 -> 256` PCA/OPQ) & Quantization (`SQ8` vs `TurboQuant`) (`V1.0 & V2.0`)

* **1. Why We Need It (30-Second Pitch):**
  > *"At `500M` vectors, uncompressed `1024-d FP32` vectors take `4 KB` per item just for dense vectors (`2 TB` of raw vectors before replication and streaming headroom). The SOW and Sept 28 sync require us to test **stepping down dimensions (`1024 -> 768 -> 512 -> 256`)** and **Scalar Quantization (`SQ8` vs `TurboQuant`)** to achieve a **4x–6x total RAM and cost reduction** while verifying that Recall@K and Golden Query relevance don't drop."*
* **2. How We Will Do It (`Script 4` & `Script 13`):**
  - **Step 1 (Dimensionality Reduction Ladder `1024 -> 768 -> 512 -> 256`):** Because Glance's `1024-d CLIP` model is *not* Matryoshka-trained, we cannot just slice the first 512 numbers. Using `Script 4`, we fit an out-of-sample Orthogonal Procrustes / Whitened PCA projection matrix to project `1024-d -> 768-d (-25%) -> 512-d (-50%) -> 256-d (-75%)` (re-normalizing L2 norm to `1.0`), and measure Latency, Brute-Force `Recall@K`, Golden Query `NDCG@10`, and RAM at each step.
  - **Step 2 (Quantization `SQ8` vs `FP32` vs `TurboQuant`):** Enable Vertex AI's native `SCALAR_QUANTIZATION (SQ8)` (`4 bytes -> 1 byte` per dim = `-75%` dense bytes, `~50%` total hybrid shard RAM reduction alone) at `1024-d`, `768-d`, `512-d`, and `256-d`, and compare against `FP32` and `TurboQuant`.
* **3. Likely Questions & Your Ready Answers:**
  - **Q: *"In the Sept 28 meeting, someone mentioned `TurboQuant` had recall loss and `CLIP` isn't Matryoshka—how are we handling that?"***  
    **A:** *"That's why we separate `EXP-5` into two independent levers: (1) Native Vertex `SQ8` at full `1024-d`, which compresses dense bytes by 4x (`-50%` total hybrid shard RAM) with `< 0.5%` recall loss and requires zero PCA truncation; and (2) Out-of-sample trained PCA/OPQ (`1024 -> 768 -> 512 -> 256`) rather than naive Matryoshka slicing, plus comparing against native `768-d FashionSigLIP` and `Gemini Multimodal Embeddings`."*

---

### `EXP-6`: Concurrent `5,000 QPS` Reads + Streaming Updates (`500 UPS`, `10M Inserts + 10M Updates`), "Read-Your-Own-Data" (`V1.0 & V2.0`)

* **1. Why We Need It (30-Second Pitch):**
  > *"In production, Glance doesn't just run static reads—they have **`20M updates/day` (`10M daily new inserts + 10M daily in-place updates` = `231/s` average, `500 UPS` SOW target, `1,626/s` peak burst)** happening **at the exact same time** as search queries.  
  > Moreover, in `reliability-escalation.md`, Glance escalated two streaming bugs: (1) **`Crowding attribute mismatch` P1 outages (Cases `72751412` & `73225742`)** during streaming updates, and (2) **`64,017` products in their index that have a dense vector but are missing their sparse vector**. `EXP-6` tests `5,000 QPS` reads + `500 UPS` writes simultaneously, measures **'Read-Your-Own-Data' freshness**, and proves the permanent fix for both P1 bugs."*
* **2. How We Will Do It (`Script 16` & `Script 10`):**
  - **Run Script 16's 3 Simultaneous Engines:**
    1. **Engine 1 (`5,000 QPS` Hybrid Reads):** Multi-core worker processes firing pre-serialized Hybrid queries at `5,000 QPS`.
    2. **Engine 2 (`20M/Day` Streaming Writes = `10M Daily Inserts + 10M Daily Updates` at `500 UPS` / `1,626/s` burst):** Streams 50% new product IDs and 50% existing product updates via `STREAM_UPDATE` (`UpsertDatapoints` in V1.0 / `UpsertDataObjects` in V2.0), isolating Ground-Truth IDs so updates don't corrupt the recall check.
    3. **Engine 3 (Live Recall & "Read-Your-Own-Data" Probe):** Continuously checks live `Recall@K` during the write storm, AND every 30 seconds inserts/updates a test product and polls search every `100ms` to measure exact **"Read-Your-Own-Data" visibility lag (in seconds)**.
  - **Reproduce & Fix the `Crowding Attribute Mismatch` Bug:** In Glance's `update_pipeline.py:L192`, `crowding_tag` (`f"{parent_id}-{core_color}"`) can change or become `None` on re-upserts (and `restrict_only=True` omits `crowding_tag`). Because V1.0 flushes Dense and Sparse streaming buffers asynchronously, a mutated `crowding_tag` causes Dense and Sparse sub-indices to hold different crowding values for the same ID! We verify that enforcing an **immutable, non-null `crowding_tag`** (and V2.0's atomic `DataObject` updates) results in **0 errors and 0 missing sparse vectors**.
* **3. Likely Questions & Your Ready Answers:**
  - **Q: *"How do we test 'Read-Your-Own-Data' while 5,000 QPS reads and 500 UPS writes are running?"***  
    **A:** *"Engine 3 in Script 16 upserts a known sentinel vector (or flips `instock: false -> true` on an existing item) with a high-precision timestamp $t_0$, and immediately queries for that exact vector every `100ms` until it appears in the top results at $t_1$. The difference $(t_1 - t_0)$ gives us the exact `p50, p95, p99` Read-Your-Own-Data freshness lag in seconds under full load."*

---

### `EXP-7`: Final Stage — Hot/Cold Catalog Segregation & SSD Offloading Evaluation (`500M`) (`V1.0 & V2.0`)

* **1. Why We Need It (30-Second Pitch):**
  > *"Once all our In-Memory (`RAM-only`) experiments (`EXP-0` to `EXP-6`) are locked in, the final step is optimizing storage cost for `500M` items. As Farhad mentioned in the Sept 28 sync, Glance already has a mechanism to segregate **Hot** vs. **Cold** catalog items (and we already saw in their profile that `31.4%` of their index is inactive/out-of-stock). In `EXP-7`, we evaluate keeping **Hot active/in-stock items in RAM** while offloading **Cold/OOS/long-tail items** or using SSD-backed storage tiering."*
* **2. How We Will Do It (`Script 13`):**
  - Partition the catalog by active/in-stock status and query/impression frequency into a **Hot In-Memory Tier** vs. **Cold / Offloaded Tier**.
  - Because a `5M` sample is near/below the `~7.75 GB` shard threshold needed to trigger physical SSD offloading directly, combine our empirical `5M -> 20M` RAM measurements (`e2-highmem-16` + `SQ8` + `PCA`) with Google Product Engineering's `100M–500M` SSD latency/IOPS curves to deliver the final **`500M` In-Memory vs. Hot/Cold SSD TCO table**.
* **3. Likely Questions & Your Ready Answers:**
  - **Q: *"Why is `EXP-7` at the very end instead of testing SSD from day one?"***  
    **A:** *"Two reasons: First, our primary SLA (`Dense P95 < 100ms`, `Hybrid P95 < 150ms` at `5,000 QPS`) must be certified In-Memory first without disk I/O noise. Second, as noted in the Sept 28 sync, `5M` vectors don't cross the `~7.75 GB` shard threshold for physical SSD offloading, so we run all `5M -> 20M` functional benchmarks in RAM first and evaluate Hot/Cold & SSD tiering as the final `500M` TCO optimization stage."*
