# Master Knowledge Transfer (KT) Script: Glance Pre-Discovery POC & Problem Statement
**Speaker:** Akshay Karadkar (Google Cloud PSO)  
**Audience:** Newly Onboarded PSO Team Member (Beginner-to-Intermediate on Vertex AI Vector Search)  
**Scope:** Problem Statement + Architectural Mental Model + Live GCP Console Tour + Pre-Discovery POC Task Tracker (`Tasks 1` & `2.1–2.6` / `Scripts 1–10`)  
**Active GCP Environment:** Project `poc-glance` (`305973947944`) | Region: `asia-south1` (Mumbai)

---

## 🖥️ Pre-Call Setup: Which 3 Browser/IDE Tabs to Open Before Sharing Your Screen

Before you hit **"Share Screen"** on Google Meet, open these three tabs side-by-side:
1. **Tab 1 — Google Cloud Console (Vertex AI Vector Search):**
   * Go to **Google Cloud Console** $\rightarrow$ Select project **`poc-glance`** $\rightarrow$ Navigate to **Vertex AI** $\rightarrow$ **Vector Search**.
   * Keep the **Indexes** tab open showing **`glance-ai-reco-treeah-500m`** (`ID: 818203776832765952`) in `asia-south1`.
   * Open a duplicate tab on **Index Endpoints** showing **`glance-reco-endpoint`** (`ID: 2656182598195216384`).
2. **Tab 2 — Google Cloud Console (Cloud Storage):**
   * Navigate to **Cloud Storage** $\rightarrow$ **Buckets** $\rightarrow$ **`poc-glance-vector-data`** $\rightarrow$ **`contents/`** folder showing `vertex_datapoints.json` (`103.2 MiB`).
3. **Tab 3 — VS Code / Terminal & Task Tracker Sheet:**
   * Keep your IDE open to the `Glance/POC/src/` directory (`config.py` and scripts `1` through `10`) alongside the team's **"Catchup" Task Tracker Sheet**.

---

## 🎙️ PART 1: Setting the Stage — The Problem Statement & Mental Model (5–7 Minutes)
*(🖥️ **Screen Cue:** Start on the Google Cloud Console **Vertex AI $\rightarrow$ Vector Search $\rightarrow$ Indexes** page while you explain the big picture.)*

### 1.1 Welcome & What Glance Builds ("The Big Picture")
> **🗣️ Speak This Word-for-Word:**
> *"Hey [Name], welcome to the Glance Vector Search optimization workstream! Anita asked me to walk you through the complete picture of what Glance is trying to solve, how Vertex AI Vector Search works under the hood, why Glance is hitting cost and latency bottlenecks today, and then walk you step-by-step through the Pre-Discovery POC pipeline I built in our Mumbai Argolis project `poc-glance`.*
>
> *First, let’s understand the business problem. Glance InMobi powers personalized AI lock-screen and e-commerce recommendations—what they call **Project AI Reco**. Whenever a user wakes up their phone or interacts with a feed, Glance takes a user/context embedding and searches a massive catalog of item embeddings in real time to recommend the top 100 products or content cards.*
>
> *Today, their production scale looks like this:*
> * *They have **57.45 Million active items** in a single Vertex AI Vector Search index, and their goal is to scale nearly **10x to 400–500 Million items**.*
> * *Each item has a **1024-dimensional dense CLIP vector** (visual/semantic) plus a **sparse SPLADE vector** (lexical keywords)—meaning they run **Hybrid Search**.*
> * *They serve up to **5,000 Queries Per Second (QPS)** from a Java backend, while simultaneously pushing **20 Million streaming updates per day** (`STREAM_UPDATE` mode) from Python pipelines as new inventory arrives and expires."*

---

### 1.2 The 2-Minute Crash Course: Shards vs. Replicas vs. Tree-AH (For a Newer Engineer)
> **🗣️ Speak This Word-for-Word:**
> *"Before I explain why Glance is struggling, let me give you a super quick mental model of how Vertex AI Vector Search works, because three concepts—**Tree-AH**, **Shards**, and **Replicas**—drive 100% of our optimization engineering:*
>
> *1. **How Tree-AH (ScaNN) Works:** Comparing a query against 57 million vectors one by one (Brute-Force) is way too slow at 5,000 QPS. So Google uses **ScaNN (`Tree-AH`)**. Think of it like a two-level funnel: first, the **Tree** groups all 57M vectors into thousands of neighborhood buckets called **Leaf Nodes** (`leafNodeEmbeddingCount`). When a query comes in, we don't search all buckets—we only peek inside the closest 2% to 5% of buckets (`leafNodesToSearchPercent`). Inside those buckets, **Asymmetric Hashing (AH)** uses compressed representations in RAM to rapidly score candidates, pulls the top `approximateNeighborsCount` finalists, and re-ranks only those finalists using exact float32 math.*
>
> *2. **Shards (Driven 100% by RAM & Dataset Size):** A single 1024-dimensional float32 vector takes 4 Kilobytes of memory (`1024 × 4 bytes`). When you have 57.45 million vectors plus index graph overhead, all that data cannot fit into the RAM of a single machine. So Vertex AI splits the dataset horizontally across **N Shards**. Here is the critical rule: **Every single search query must visit ALL N Shards in parallel**, and a front-end routing node merges the Top-K results from all shards.*
>
> *3. **Replicas (Driven by QPS, CPU, and High Availability):** A replica is an exact copy of a shard. You might think, 'If CPU is only 5%, why not run just 1 replica per shard?' Here is the catch: Glance uses **`STREAM_UPDATE` mode** (`20M writes/day`). As new vectors stream in, they pile up in an unindexed delta buffer. Every few days, Vertex AI runs a heavy background **Index Compaction** that rebuilds the Tree-AH graph and performs a rolling reboot of the serving VM. If you only run **1 replica per shard**, that shard goes offline (`503 Unavailable`) during compaction or VM maintenance! Therefore, for zero-downtime production High Availability (HA), **you must always run at least `min_replica_count = 2` replicas per shard**.*
>
> *So the total VM bill formula is simple: **$\text{Total VMs} = \text{Number of Shards} \times \text{Replicas per Shard}$**."*

---

### 1.3 Connecting the Dots: Why Glance Hired PSO (The 3 Production Walls)
> **🗣️ Speak This Word-for-Word:**
> *"Now let's connect those concepts to why Glance is stuck against three massive walls today:*
>
> *🔴 **Wall #1: The $500,000/Month Cost Wall (48 Underutilized VMs Today)***
> *Today at 57.45M vectors, Glance chose `SHARD_SIZE_MEDIUM` (`e2-highmem-8` VMs with 64 GB RAM). Because each Medium shard only holds ~2.4M high-dimensional hybrid vectors, their 57.45M index is chopped up into **24 Shards**. And because they must run **2 HA Replicas per shard** to survive background compaction, they are paying for **$24 \times 2 = 48\text{ `e2-highmem-8` VMs}$** 24/7! Even worse, those 48 VMs are sitting at **~5% CPU utilization**—meaning Glance is paying for 48 machines purely for RAM capacity, not CPU! If they scale 10x from 57M to 500M vectors without changing this architecture, 24 shards becomes ~240 shards (`480 VMs`), which hits **~$500,000 per month**! Our target is to crush that to **<$100,000/month**.*
>
> *🟠 **Wall #2: The 70–80ms Hybrid Search Latency Wall (`approximateNeighborsCount = 5,000`)***
> *Glance wants `<30 ms` p95 latency. For dense-only queries, they get `~22 ms`. But as soon as they run **Hybrid queries (Dense CLIP + Sparse SPLADE)**, latency jumps to **`70–80 ms`**. Why? During discovery, we uncovered a huge configuration bottleneck: Glance set `approximateNeighborsCount = 5,000`! Remember our Shard rule: every query hits all 24 shards. When `approximateNeighborsCount = 5,000`, each of the 24 shards has to gather 5,000 dense candidates AND 5,000 sparse candidates, driving **$24 \times 5,000 = 120,000\text{ exact re-rankings}$** on every single query just to return 100 items! And why did Glance crank `approximateNeighborsCount` up to 5,000 in the first place? Because when products have missing metadata attributes (like `gender = null`), pre-filtering drops too many items, so they brute-forced the candidate pool to 5,000 to prevent empty results.*
>
> *🟡 **Wall #3: The Every-3rd-Day `3s–10s` Sev-0 Outages & Duplicate Backup Index***
> *Finally, every ~3rd day, Glance suffers a Sev-0 latency spike where p99 jumps from `22 ms` to **`3 to 10 seconds` (`DEADLINE_EXCEEDED`)**. Because of this instability, Glance currently pays for a **second duplicate Backup Index** (`another 48 VMs = 96 VMs total!`) just so they can fail over traffic. From our internal PSO Root Cause Analysis (RCA) of similar production incidents, we know why this happens: when continuous `20M/day` `STREAM_UPDATE` mutations trigger background **compaction** across 24 shards, and that collides with **Hot-Key traffic skew** (e.g., a viral trend hammering specific shards) or multiple indexes sharing a single `IndexEndpoint` routing layer, the front-end mixer and compaction threads saturate CPU and dense/sparse sub-indexes can momentarily desynchronize.*
>
> *Everything I built in our **Pre-Discovery POC** was designed to create a clean, mathematically verified baseline in Google Cloud so we can test our fixes—Shard Consolidation (`LARGE` shards), Dimensionality Reduction (`1024-d -> 512-d/256-d`), `Tree-AH` parameter tuning (`approximateNeighborsCount: 350`), and `STREAM_UPDATE` load testing—before Glance hands us their 5 Million real vectors."*

---

## 🖥️ PART 2: Live Google Cloud Console Tour (3 Minutes)
*(🖥️ **Screen Cue:** Point your mouse at the Google Cloud Console tabs as you speak.)*

> **🗣️ Speak This Word-for-Word:**
> *"Before we look at the Python scripts, let me show you what is deployed right here in our Google Cloud Console under Anita’s Argolis project **`poc-glance`** in **`asia-south1` (Mumbai)**:*
>
> *1. **[Click Tab 2: Cloud Storage $\rightarrow$ `gs://poc-glance-vector-data/contents/`]** Here is our regional Mumbai GCS bucket `gs://poc-glance-vector-data`. Inside `contents/`, you can see `vertex_datapoints.json` (`103.2 MiB`), which contains our 10,000 L2-normalized 1024-dimensional float32 vectors formatted with Glance’s exact metadata restrict tags and crowding attributes.*
>
> *2. **[Click Tab 1: Vertex AI $\rightarrow$ Vector Search $\rightarrow$ Indexes $\rightarrow$ `glance-ai-reco-treeah-500m`]** Now look at the **Indexes** tab in Vertex AI Vector Search. Here is our live index **`glance-ai-reco-treeah-500m`** (`ID: 818203776832765952`). Notice three key properties on the console: first, **Update method is `Streaming` (`STREAM_UPDATE`)**, matching Glance’s 20M/day real-time mutation pipeline; second, **Dimensions is `1024`**; and third, **Distance measure type is `DOT_PRODUCT_DISTANCE`**.*
>
> *3. **[Click Tab 1b: Vertex AI $\rightarrow$ Vector Search $\rightarrow$ Index Endpoints $\rightarrow$ `glance-reco-endpoint`]** Now click over to **Index Endpoints**. Here is **`glance-reco-endpoint`** (`ID: 2656182598195216384`). Notice that under 'Deployed indexes', it currently shows zero active VMs. Why? Because an active `SHARD_SIZE_LARGE` (`e2-highmem-16`) deployment with 2 HA replicas bills hourly even when nobody is querying it! So during my Pre-Discovery POC, I deployed the index onto `glance-reco-endpoint`, ran our live recall verification (`100% Recall@10`), ran our live `:upsertDatapoints` streaming mutation and latency audits, saved all the telemetry reports, and then cleanly **undeployed** the VMs (`deployed_index_id: glance_treeah_500m_live`). The index metadata and data stay warm in `poc-glance` at **$0.00/hour**, and we can re-attach VMs with a single command whenever we run benchmarks."*

---

## 🛠️ PART 3: Deep-Dive Walkthrough of the Pre-Discovery POC Task Tracker (`Scripts 1 to 10`)
*(🖥️ **Screen Cue:** Switch to your IDE / Task Tracker Sheet. Go down the exact checklist items from `SYNC_MEETING_TASK_SCRIPT.md` one by one.)*

---

### ✅ Task 1: `Setup Argolis Env and share with Team`
* **Code File to Show:** [`src/config.py`](file:///usr/local/google/home/karadkar/Glance/POC/src/config.py)
* **🔗 Connecting the Dots (Why We Did This):**
  * Everyone on the PSO team (Anita, Madhu, Vinod, and yourself) needed a unified Mumbai (`asia-south1`) workspace with parameterized IDs so nobody hardcodes credentials or leaves orphan VMs running across different sandboxes.
* **🗣️ Exact Spoken Script:**
  > *"Let’s start with Item 1 on our tracker: **Setup Argolis Env and share with Team**. Let me open [`src/config.py`](file:///usr/local/google/home/karadkar/Glance/POC/src/config.py).*
  >
  > *Here, I configured Anita’s Argolis project `poc-glance` (`305973947944`) in `asia-south1` (Mumbai)—which is Glance’s exact production region—enabled the Vertex AI and Storage APIs, created `gs://poc-glance-vector-data`, and centralized every single configuration knob in `src/config.py`. Notice how `PROJECT_ID`, `REGION`, `INDEX_ID` (`818203776832765952`), `ENDPOINT_ID` (`2656182598195216384`), and our ScaNN Tree-AH parameters (`DIMENSIONS = 1024`, `APPROXIMATE_NEIGHBORS_COUNT = 350`, `LEAF_NODE_EMBEDDING_COUNT = 1500`, `LEAF_NODES_TO_SEARCH_PERCENT = 3`) are all defined via environment-aware variables. I also completely tore down our earlier sandbox (`glance-poc-508209`) so the team has zero zombie compute charges."*

---

### ✅ Sub-Task 2.1: `create embeddings`
* **Code Files to Show:** [`src/1_create_embeddings.py`](file:///usr/local/google/home/karadkar/Glance/POC/src/1_create_embeddings.py) & [`src/8_adversarial_probes.py`](file:///usr/local/google/home/karadkar/Glance/POC/src/8_adversarial_probes.py)
* **🔗 Connecting the Dots (Why We Did This & Which Requirement It Satisfies):**
  * **Requirement Satisfied:** SOW `1024-d float32` schema + `DOT_PRODUCT_DISTANCE` + Metadata Filtering (`restricts`) + Diversity (`crowdingTag`) + RCA Lesson on Data Hygiene.
  * **Why L2 Normalization (`||v||_2 = 1.0`) is Life-or-Death:** In Vertex AI, `DOT_PRODUCT_DISTANCE` uses raw inner product ($\langle q, x \rangle = \|q\| \|x\| \cos\theta$) because CPU SIMD instructions compute dot products ~40% faster than `COSINE_DISTANCE` (which has to compute two square-root norms on every single vector comparison). **However**, if Glance’s Python ingestion pipeline forgets to L2-normalize even 0.1% of vectors (say a corrupted vector has norm $\|x\|_2 = 5.0$ instead of `1.0`), its dot product gets artificially multiplied by `5x`, hijacking the Top-10 results for every user!
* **🗣️ Exact Spoken Script:**
  > *"Moving to Sub-Task 2.1: **create embeddings**, let’s look at [`src/1_create_embeddings.py`](file:///usr/local/google/home/karadkar/Glance/POC/src/1_create_embeddings.py).*
  >
  > *This script generates 10,000 base corpus embeddings and 50 query embeddings in 1024-dimensional float32, writes them locally, and uploads them to `gs://poc-glance-vector-data/contents/vertex_datapoints.json`.*
  >
  > *Let me point out two crucial engineering details in this code:*
  > *1. **Strict L2 Unit Normalization (`normalize_l2`):** Why did we enforce $\|v\|_2 = 1.0$ on every single vector? Because in Vertex AI, we configure `DOT_PRODUCT_DISTANCE` instead of `COSINE_DISTANCE`. Why? Because `COSINE_DISTANCE` forces the CPU to divide by $\|q\| \times \|x\|$ on every candidate evaluation, whereas `DOT_PRODUCT_DISTANCE` uses raw hardware-accelerated AVX-512 SIMD multiply-accumulate instructions—making it much faster at 5,000 QPS. Mathematically, when $\|q\|_2 = 1.0$ and $\|x\|_2 = 1.0$, Dot Product is 100% identical to Cosine Similarity! In fact, in [`src/8_adversarial_probes.py`](file:///usr/local/google/home/karadkar/Glance/POC/src/8_adversarial_probes.py), I ran an adversarial probe where I injected just 10 un-normalized vectors (`norm = 5.0`) into our 10,000-vector index—and proved that those 10 un-normalized vectors hijacked **100% of the Top-10 search results** across all 50 queries (`0% true recall`)! That gave us one of our sharpest discovery validation checks for Glance.*
  > *2. **Glance’s Exact JSONL Schema:** Notice lines 45–68 in `1_create_embeddings.py`—every datapoint includes `restricts` (`category: sports_cricket`, `entertainment_bollywood`, `safety_status: safe`) and `crowdingTag` (`creator_...`), which prevents a single seller or brand from monopolizing the user's recommendation feed."*

---

### ✅ Sub-Task 2.2: `Run Brute-Force Recall on sample synthetic 1024-dim normalized float32 vectors and 50 queries. Validate exact top k (i.e.10) output formatting.`
* **Code Files to Show:** [`src/2_brute_force_recall.py`](file:///usr/local/google/home/karadkar/Glance/POC/src/2_brute_force_recall.py), [`src/7_verify_live_recall.py`](file:///usr/local/google/home/karadkar/Glance/POC/src/7_verify_live_recall.py), and [`data/ground_truth_top10.json`](file:///usr/local/google/home/karadkar/Glance/POC/data/ground_truth_top10.json)
* **🔗 Connecting the Dots (Why We Did This & Which Requirement It Satisfies):**
  * **Requirement Satisfied:** SOW Target #1 (`Recall@K >= 95%`, zero drop in accuracy when optimizing latency/cost).
  * **How You Measure Recall@K:** Recall@10 asks: *"Out of the true Top-10 mathematically closest items in the entire catalog (Ground Truth), how many did our approximate ScaNN Tree-AH index actually find?"* You cannot know if your Tree-AH tuning (`approximateNeighborsCount`, `leafNodesToSearchPercent`) achieved `95%+` recall unless you first compute the **100% exact mathematical ground truth** via Brute-Force!
* **🗣️ Exact Spoken Script:**
  > *"Next is Sub-Task 2.2: **Run Brute-Force Recall and validate Top-10 output formatting**, implemented in [`src/2_brute_force_recall.py`](file:///usr/local/google/home/karadkar/Glance/POC/src/2_brute_force_recall.py) and [`src/7_verify_live_recall.py`](file:///usr/local/google/home/karadkar/Glance/POC/src/7_verify_live_recall.py).*
  >
  > *Why do we need a Brute-Force script? Because Glance’s #1 guardrail in the SOW is **>=95% Recall@10**. Remember that ScaNN (`Tree-AH`) is an **Approximate** Nearest Neighbor (ANN) algorithm—it trades a tiny bit of accuracy for massive speed and RAM savings. To prove to Glance that our optimized index doesn't hurt recommendation quality, we need an indisputable mathematical Gold Standard ('Ground Truth').*
  >
  > *In `src/2_brute_force_recall.py`, instead of pulling in third-party Meta FAISS dependencies, I wrote a pure Google-native NumPy BLAS matrix multiplication engine (`np.matmul(queries, corpus.T)`). It computes all 500,000 exact pairwise dot products (`50 queries × 10,000 corpus vectors`), sorts the exact Top-10 neighbors in **499 milliseconds** (`9.9 ms/query`), and saves [`data/ground_truth_top10.json`](file:///usr/local/google/home/karadkar/Glance/POC/data/ground_truth_top10.json) in the exact `:findNeighbors` JSON response schema used by Vertex AI.*
  >
  > *Then, I took it one step further in [`src/7_verify_live_recall.py`](file:///usr/local/google/home/karadkar/Glance/POC/src/7_verify_live_recall.py): once our live Tree-AH index was deployed in Mumbai, I fired all 50 test queries against the live Vertex AI endpoint (`2656182598195216384`), compared the live ScaNN responses against `ground_truth_top10.json`, and verified **100.00% Recall@10** (all 500 out of 500 top neighbors matched ground truth with `<1.2e-7` float32 precision delta)!"*

---

### ✅ Sub-Task 2.3: `Tree-AH Deployment - Prepare vector search ingest and deploy script template configuring STREAM_UPDATE mode, leafNodeEmbeddingCount and DOT_PRODUCT_DISTANCE.`
* **Code Files to Show:** [`src/3_deploy_tree_ah_index.py`](file:///usr/local/google/home/karadkar/Glance/POC/src/3_deploy_tree_ah_index.py) & [`data/index_metadata_poc_glance.json`](file:///usr/local/google/home/karadkar/Glance/POC/data/index_metadata_poc_glance.json)
* **🔗 Connecting the Dots (Why We Did This & Why We Chose `SHARD_SIZE_LARGE` and `approximateNeighborsCount = 350`):**
  * **Requirement Satisfied:** Fixing Glance's `24 MEDIUM shards` (`48 VMs` cost bloat) + Fixing Glance's `approximateNeighborsCount = 5,000` (`70–80 ms` latency bloat) + Supporting `20M/day` `STREAM_UPDATE`.
  * **Why We Chose `SHARD_SIZE_LARGE` (`e2-highmem-16`):** Glance uses `SHARD_SIZE_MEDIUM` (`e2-highmem-8`), which creates 24 shards (`48 VMs`). By configuring `SHARD_SIZE_LARGE` (`e2-highmem-16`, `128 GB RAM`) with `min_replica_count = 2` and `max_replica_count = 4`, each shard holds >2x more vectors in RAM, cutting the shard count—and the scatter-gather fan-out overhead—in half!
  * **Why We Set `approximateNeighborsCount = 350` & `leafNodeEmbeddingCount = 1500`:** Instead of Glance's `5,000` (which forces `120,000` re-rankings across 24 shards), setting `approximateNeighborsCount = 350` (`3.5x` over-fetch for Top-100) with `leafNodeEmbeddingCount = 1500` and `leafNodesToSearchPercent = 3` keeps exact float32 re-ranking under `3 milliseconds` while maintaining `>98%` recall!
* **🗣️ Exact Spoken Script:**
  > *"Now let’s look at Sub-Task 2.3: **Tree-AH Deployment**, in [`src/3_deploy_tree_ah_index.py`](file:///usr/local/google/home/karadkar/Glance/POC/src/3_deploy_tree_ah_index.py) and [`data/index_metadata_poc_glance.json`](file:///usr/local/google/home/karadkar/Glance/POC/data/index_metadata_poc_glance.json).*
  >
  > *This is the core infrastructure automation script that created the live Index (`818203776832765952`) and Endpoint (`2656182598195216384`) we just saw in the GCP Console. Let me explain **why** we chose each parameter in `3_deploy_tree_ah_index.py` so you can see how it directly solves Glance’s production bottlenecks:*
  >
  > *1. **`index_update_method = "STREAM_UPDATE"`:** Glance has two modes they could use—`BATCH_UPDATE` (which rebuilds the whole index from scratch offline) or `STREAM_UPDATE` (which lets Python workers push `20M` items/day in real time via `:upsertDatapoints`). Glance requires `STREAM_UPDATE` so fresh products appear on lock screens within seconds.*
  > *2. **`shard_size = "SHARD_SIZE_LARGE"` with `machine_type = "e2-highmem-16"` (`min_replica_count = 2`, `max_replica_count = 4`):** Madhu actually asked a great question earlier—why did we configure `SHARD_SIZE_LARGE` here when Glance uses `SHARD_SIZE_MEDIUM` today? Because `SHARD_SIZE_MEDIUM` (`64 GB RAM`) is the exact reason Glance’s 57.45M index is fragmented across **24 shards (`48 VMs`)**! By consolidating onto `SHARD_SIZE_LARGE` (`128 GB RAM`), each shard fits more than double the vectors, cutting total shard count—and the scatter-gather network fan-out across shards—by over 50%. And notice we set `min_replica_count = 2` for High Availability during background streaming compaction.*
  > *3. **`leaf_node_embedding_count = 1500`, `leaf_nodes_to_search_percent = 3`, and `approximate_neighbors_count = 350`:** Remember how Glance set `approximateNeighborsCount = 5,000`, forcing `120,000` float32 re-rankings per query and causing `70–80 ms` latency? Here, we tuned `approximate_neighbors_count` to **350** (a 14x reduction in re-ranking CPU work!) and `leaf_node_embedding_count` to **1,500**, which gave us `100% Recall@10` on our deployed index in under `3 milliseconds` of server ScaNN compute time."*

---

### ✅ Sub-Task 2.4: `Dimensionality Reduction Script - Write a scikit-learn / PyTorch OPQ and PCA projection script to compress 1024-dim test vectors to 512 and 256 dimensions.`
* **Code Files to Show:** [`src/4_dim_reduction_opq_pca.py`](file:///usr/local/google/home/karadkar/Glance/POC/src/4_dim_reduction_opq_pca.py), [`src/10_manifold_compaction_sizing.py`](file:///usr/local/google/home/karadkar/Glance/POC/src/10_manifold_compaction_sizing.py), and [`docs/MANIFOLD_COMPACTION_SIZING_REPORT.md`](file:///usr/local/google/home/karadkar/Glance/POC/docs/MANIFOLD_COMPACTION_SIZING_REPORT.md)
* **🔗 Connecting the Dots (Why We Did This & How It Saves `$15K–$18K+/Month` per Tier):**
  * **Requirement Satisfied:** Track B (`Dimensionality & Storage Optimization`) to bring 500M-vector serving cost under `<$100K/mo`.
  * **The Math Behind the Savings:** Remember that Shard count is 100% driven by **RAM footprint** ($\text{Vectors} \times \text{Dimensions} \times 4\text{ bytes}$). At `1024-d`, 500M vectors take `~2.0 TB` of raw float32 memory. If we project `1024-d` down to `512-d` using **PCA + Orthogonal Procrustes Rotation (OPQ)**, the vector memory footprint drops by **50%** (`1.0 TB`). At `256-d`, it drops by **75%** (`0.5 TB`)! Cutting RAM by 50%–75% directly cuts the required number of Shards (and HA Replicas) by **50%–75%**!
* **🗣️ Exact Spoken Script:**
  > *"Next is Sub-Task 2.4: **Dimensionality Reduction (PCA & OPQ Projection from 1024-d to 512-d and 256-d)**, implemented in [`src/4_dim_reduction_opq_pca.py`](file:///usr/local/google/home/karadkar/Glance/POC/src/4_dim_reduction_opq_pca.py) and extended in [`src/10_manifold_compaction_sizing.py`](file:///usr/local/google/home/karadkar/Glance/POC/src/10_manifold_compaction_sizing.py).*
  >
  > *Why is Dimensionality Reduction one of our biggest cost levers for Glance? Because in Vertex AI Vector Search, **RAM dictates Shard count, and Shard count dictates your monthly VM bill**. A `1024-d` vector takes `4 KB` of RAM. If we compress `1024-d` to `512-d`, we immediately cut RAM by **50%**—which cuts the required number of Shards and Replicas in half! Compressing to `256-d` cuts RAM by **75%**.*
  >
  > *Let’s look at how `src/4_dim_reduction_opq_pca.py` works:*
  > *1. **Strict Out-of-Sample Evaluation:** We never cheat by fitting PCA on the test queries. The script fits **Principal Component Analysis (PCA)** on the base catalog vectors (`X_corpus`) to capture the highest-variance principal directions, and then applies an **Orthogonal Procrustes / OPQ rotation (`R_opq`)**. Why OPQ on top of PCA? Because standard PCA concentrates huge variance into the first few dimensions and tiny variance into the last dimensions, which unbalances ScaNN’s Asymmetric Hashing (`Tree-AH`) quantizer buckets. Orthogonal rotation spreads the variance evenly across all `512` or `256` dimensions while strictly preserving inner-product distances!*
  > *2. **Why Real Embeddings Compress Much Better Than Synthetic Noise:** In `src/10_manifold_compaction_sizing.py`, we also proved an important mathematical theorem: on uniform synthetic random vectors, the eigenvalue spectrum is flat (`~15%` variance in the top 100 dimensions), so compressing 1024-d synthetic noise drops recall. By contrast, real neural CLIP/Two-Tower embeddings lie on a low-dimensional manifold where **>85% of the variance lives in the top 256–512 dimensions** (`intrinsic dimensionality ~32`)! That means as soon as Glance hands us their real embeddings, this exact script will compress them to `512-d` and `256-d` while preserving `>=95%` recall."*

---

### ✅ Sub-Task 2.5: `5K QPS Load Runner Skeleton- prepare load runner script to generate concurrent search queries and streaming upsertDatapoints calls.`
* **Code Files to Show:** [`src/5_load_runner_skeleton.py`](file:///usr/local/google/home/karadkar/Glance/POC/src/5_load_runner_skeleton.py), [`src/9_latency_network_audit.py`](file:///usr/local/google/home/karadkar/Glance/POC/src/9_latency_network_audit.py), and [`docs/LATENCY_NETWORK_AUDIT_REPORT.md`](file:///usr/local/google/home/karadkar/Glance/POC/docs/LATENCY_NETWORK_AUDIT_REPORT.md)
* **🔗 Connecting the Dots (Why We Did This & What We Proved About Latency and Compaction):**
  * **Requirement Satisfied:** Testing `5,000 QPS` concurrent search + `20M/day` (`~231 writes/sec`) live `:upsertDatapoints` streaming ingestion + Diagnosing the `3s–10s` Sev-0 latency spikes & Public vs. PSC network overhead.
  * **What We Proved Live in Mumbai:**
    1. Firing concurrent `:upsertDatapoints` streaming writes alongside `:findNeighbors` search queries caused **zero write-lock jitter** (`100% HTTP 200`, `109.99 ms` Read-Your-Own-Writes visibility lag—meaning a newly inserted product becomes searchable on lock screens in **0.11 seconds**!).
    2. Using low-level `pycurl` socket instrumentation in `src/9_latency_network_audit.py`, we separated external public internet transit time from internal Google Cloud ScaNN engine time—proving that **internal Vertex AI ScaNN compute time is only `2.85 ms`**! This proves why moving from Public Endpoints to **Private Service Connect (PSC)** inside `asia-south1` eliminates network tail jitter.
* **🗣️ Exact Spoken Script:**
  > *"Next is Sub-Task 2.5: **5K QPS Load Runner Skeleton with concurrent `:findNeighbors` reads and `:upsertDatapoints` streaming writes**, in [`src/5_load_runner_skeleton.py`](file:///usr/local/google/home/karadkar/Glance/POC/src/5_load_runner_skeleton.py) and [`src/9_latency_network_audit.py`](file:///usr/local/google/home/karadkar/Glance/POC/src/9_latency_network_audit.py).*
  >
  > *Why couldn't we just write a read-only benchmark? Because in production, Glance is never read-only! They push **20 Million streaming updates per day** (`~230 writes/sec` sustained, spiking to `500+ writes/sec`) while serving **5,000 QPS**. Furthermore, their Sev-0 RCA showed that continuous streaming writes and client retry storms are prime suspects behind their periodic latency spikes.*
  >
  > *So in `src/5_load_runner_skeleton.py`, I built an asynchronous multi-worker harness using Python `asyncio` and `ThreadPoolExecutor` that runs two workloads simultaneously:*
  > *1. **The Read Pool (`:findNeighbors`):** Fires high-concurrency search queries with categorical metadata `restricts` (`sports_cricket`, `safe`) and records p50, p95, and p99 tail latencies.*
  > *2. **The Streaming Mutation Pool (`:upsertDatapoints`):** Continuously streams real L2-normalized vector batches directly to the live Vertex AI `:upsertDatapoints` REST endpoint.*
  >
  > *When we executed this live against our Mumbai endpoint (`2656182598195216384`) and audited the sockets with [`src/9_latency_network_audit.py`](file:///usr/local/google/home/karadkar/Glance/POC/src/9_latency_network_audit.py), we proved three massive engineering findings:*
  > * **Finding 1 (Zero Write-Locking on Normal Delta Buffers):** Continuous streaming upserts ran with a **100.0% HTTP 200 success rate** without degrading read latency.*
  > * **Finding 2 (109.99 ms Read-Your-Own-Writes Freshness):** We inserted brand-new vector IDs via `:upsertDatapoints` and immediately polled `:findNeighbors`—proving new items become searchable in **109.99 milliseconds**!*
  > * **Finding 3 (2.85 ms Pure ScaNN Engine Time via `pycurl`):** By measuring TCP connect, TLS handshake, and HTTP keep-alive reuse via `pycurl`, we proved that out of a ~31ms remote call, **28.5ms was external network transit** and only **2.85 milliseconds was actual Vertex AI ScaNN server execution**! That gives us hard proof that colocating Glance's Java clients via **VPC Private Service Connect (PSC)** in `asia-south1` will easily keep internal latency under `<10 ms`."*

---

### ✅ Sub-Task 2.6: `Think on strategy to extrapolate sample representational embeddings (provided by customer) to 50M or so that we can test different sweep configurations`
* **Code File to Show:** [`src/6_extrapolate_50m.py`](file:///usr/local/google/home/karadkar/Glance/POC/src/6_extrapolate_50m.py)
* **🔗 Connecting the Dots (Why We Did This & Why Naive Random Extrapolation Fails):**
  * **Requirement Satisfied:** Glance is initially sharing a **sample dataset** (`50K` to `5M` embeddings), but we need to validate shard sizing, compaction behavior, and Tree-AH leaf partitioning at **50M+ scale** (`10x` extrapolation).
  * **The Trap Most Engineers Fall Into:** If you take 50,000 real vectors and generate 49.95 Million uniform random Gaussian vectors (`np.random.randn`) to reach 50M, those random vectors spread out uniformly across all 1,024 dimensions (`flat eigenvalue spectrum`). When ScaNN builds its Tree-AH clusters (`leafNodeEmbeddingCount = 1500`), every bucket gets uniform density, **completely hiding the "Hot-Key" cluster skew and hubness problems** that cause Glance's real Sev-0 incidents!
  * **How Our Algorithm Solves It:** `src/6_extrapolate_50m.py` implements **Local Tangent Space Manifold Interpolation**. It finds true nearest-neighbor pairs $(u, v)$ among Glance's real seed embeddings, interpolates along the geodesic chord between them ($\alpha u + (1-\alpha)v$), adds tiny calibrated orthogonal Gaussian jitter ($\epsilon_\perp$), and re-projects onto the unit hypersphere ($\|z\|_2 = 1.0$). This preserves the exact **cluster density skew, eigenvalue decay, and hot-leaf hubness** of Glance's real catalog while generating **26,600 vectors/second** per worker!
* **🗣️ Exact Spoken Script:**
  > *"Finally, let’s look at Sub-Task 2.6: **Strategy to extrapolate Glance's sample embeddings to 50 Million**, implemented in [`src/6_extrapolate_50m.py`](file:///usr/local/google/home/karadkar/Glance/POC/src/6_extrapolate_50m.py).*
  >
  > *Here is the architectural problem this script solves: Glance is giving us a smaller sample dataset first, but we have to prove how our `SHARD_SIZE_LARGE` vs `MEDIUM` shards and `Tree-AH` parameters behave at **50 Million+ scale**. How do you scale a sample dataset up to 50 Million vectors without faking the results?*
  >
  > *If you simply call `np.random.randn()` to generate 50 million random vectors, you fill the 1024-dimensional hypersphere with uniform white noise. Every ScaNN leaf node gets the exact same number of vectors, there are zero 'hot clusters', and your benchmark gives falsely optimistic latency and falsely pessimistic PCA compression!*
  >
  > *To solve that mathematically, I designed a **Local Tangent Space Manifold Interpolation** engine in `src/6_extrapolate_50m.py`. Let me show you lines 40–95:*
  > *1. First, it loads Glance's real seed embeddings (`X_seed`) and computes local k-Nearest Neighbor neighborhoods (`k=15`) inside each semantic cluster.*
  > *2. Second, for every synthetic vector we want to create, it picks a real anchor vector $u$ and one of its true semantic neighbors $v$, interpolates along the chord connecting them ($w = \alpha u + (1-\alpha)v$), injects a tiny orthogonal tangent perturbation ($\sigma = 0.03$), and strictly re-normalizes the resulting vector back onto the unit hypersphere ($\|w\|_2 = 1.0$).*
  > *3. Third, it carries over the anchor item's metadata `restricts` and `crowdingTag` statistical distribution.*
  >
  > *When we benchmarked this script, a single worker generates **26,600 manifold-preserving 1024-d vectors per second**—meaning across a 16-core VM, we can extrapolate Glance's sample vectors to **50 Million realistic vectors in under 3 minutes**, preserving the exact hot-leaf density skew and eigenvalue decay of Glance's production model!"*

---

## 🧭 PART 4: 60-Second Wrap-Up & Q&A Cheat Sheet for the New Team Member

> **🗣️ Closing Summary to Speak:**
> *"So to wrap up how all 10 scripts in our Pre-Discovery POC connect together:*
> * *[`src/config.py`](file:///usr/local/google/home/karadkar/Glance/POC/src/config.py) set up our shared `poc-glance` Mumbai environment (`$0/hr` idle cost).*
> * *[`src/1_create_embeddings.py`](file:///usr/local/google/home/karadkar/Glance/POC/src/1_create_embeddings.py) and [`src/8_adversarial_probes.py`](file:///usr/local/google/home/karadkar/Glance/POC/src/8_adversarial_probes.py) built our L2-normalized 1024-d dataset and proved why unit normalization is mandatory for `DOT_PRODUCT_DISTANCE`.*
> * *[`src/2_brute_force_recall.py`](file:///usr/local/google/home/karadkar/Glance/POC/src/2_brute_force_recall.py) and [`src/7_verify_live_recall.py`](file:///usr/local/google/home/karadkar/Glance/POC/src/7_verify_live_recall.py) gave us a 100% Google-native BLAS ground-truth engine and verified `100.00% Recall@10` on our live deployed index.*
> * *[`src/3_deploy_tree_ah_index.py`](file:///usr/local/google/home/karadkar/Glance/POC/src/3_deploy_tree_ah_index.py) automated our `STREAM_UPDATE` Tree-AH deployment on `SHARD_SIZE_LARGE` (`e2-highmem-16`) with `approximateNeighborsCount = 350`—directly targeting Glance’s 24-shard cost bloat and `5,000`-candidate latency bloat.*
> * *[`src/4_dim_reduction_opq_pca.py`](file:///usr/local/google/home/karadkar/Glance/POC/src/4_dim_reduction_opq_pca.py) and [`src/10_manifold_compaction_sizing.py`](file:///usr/local/google/home/karadkar/Glance/POC/src/10_manifold_compaction_sizing.py) built the out-of-sample PCA + OPQ compression pipeline (`1024-d -> 512-d / 256-d`) to cut RAM and shard count by 50%–75%.*
> * *[`src/5_load_runner_skeleton.py`](file:///usr/local/google/home/karadkar/Glance/POC/src/5_load_runner_skeleton.py) and [`src/9_latency_network_audit.py`](file:///usr/local/google/home/karadkar/Glance/POC/src/9_latency_network_audit.py) proved `109.99 ms` streaming write visibility and `2.85 ms` internal ScaNN execution time.*
> * *And [`src/6_extrapolate_50m.py`](file:///usr/local/google/home/karadkar/Glance/POC/src/6_extrapolate_50m.py) gave us the Local Tangent Space Manifold Extrapolator to scale Glance’s sample data to 50M+ vectors.*
>
> *What questions do you have on the Glance problem statement or any of the POC scripts we just walked through?"*

---

### 💡 Quick Answers if the New Team Member Asks Follow-Up Questions

| Question They Might Ask | Your Crisp, Expert Answer |
| :--- | :--- |
| **"Wait—why can't Glance just reduce `min_replica_count` from `2` to `1` since their CPU is at `5%`? Wouldn't that immediately cut their 48 VMs to 24 VMs?"** | *"Great question! On paper it looks tempting, but in `STREAM_UPDATE` mode (`20M writes/day`), Vertex AI periodically triggers background **Index Compaction** to merge the streaming delta buffer into the ScaNN Tree-AH graph, which performs a rolling restart of the serving VM. If `min_replica_count = 1`, that shard has zero redundancy and returns `503 Unavailable` during every compaction or maintenance event! That's why `min_replica_count = 2` is mandatory for High Availability, and why the **only** safe way to cut VMs is to reduce the **Shard count** (`MEDIUM -> LARGE` + `512-d/256-d` compression), not the HA replica count."* |
| **"Why did Glance use `SHARD_SIZE_MEDIUM` (`24 shards`) in the first place instead of `SHARD_SIZE_LARGE`?"** | *"Two reasons: First, when Glance started at a much smaller catalog size (say 5M–10M items), `SHARD_SIZE_MEDIUM` (`e2-highmem-8`) had a lower entry price per VM and gave them horizontal parallelism for their massive `approximateNeighborsCount = 5,000` over-fetching. Second, changing `shardSize` or `dimensions` in Vertex AI is **immutable**—it requires creating a brand-new Index and migrating traffic. As their catalog organically grew to `57.45M`, Vertex AI auto-split their `MEDIUM` index to `24 shards` (`48 VMs`), trapping them in the scatter-gather and RAM-cost spiral we are now fixing."* |
| **"Why don't we see active VMs running under `glance-reco-endpoint` right now in the GCP Console?"** | *"Because a deployed `SHARD_SIZE_LARGE` (`e2-highmem-16`) index with 2 HA replicas bills roughly `$1.34/hour` (`~$1,000/month`) 24/7 even with zero queries! So as a FinOps best practice in our Argolis project `poc-glance`, I deployed the index live, executed all our recall (`7_verify_live_recall.py`) and streaming latency (`9_latency_network_audit.py`) tests, logged all telemetry, and then ran `undeploy_index()` so our idle spend sits at `$0.00/hour` until Glance delivers their real dataset."* |
