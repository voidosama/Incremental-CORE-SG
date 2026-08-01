# Experiment suite

Reference for the thesis evaluation: what is measured, on what data, against what
baseline, and **why each of those choices was made**. Several of the decisions here look
arbitrary until you know what went wrong first, so the reasoning is recorded alongside the
setup. When something changes in the routine, change it here too.

Last updated: 2026-07-29.

---

## 1. The claims under test

| | claim | what would falsify it |
|---|---|---|
| **A** | The CORE-SG can be updated with new data without rebuilding, at comparable quality | update cost approaching rebuild cost, or quality degrading over a stream |
| **B** | The design scales across GPUs | per-device memory growing with device count; speed-up far below the predicted Amdahl ceiling |
| **C** | Anti-hub bridges replace the MST half at a fraction of the cost | MSTs materially heavier than exact, or clusterings that differ |

Everything below exists to produce evidence for one of these three, or to rule out an
alternative explanation for that evidence.

---

## 2. Ground truth: the exact CORE-SG, never class labels

**Every quality claim is measured against the exact CORE-SG** — exact k-NN half (brute
force) plus exact `MST_kmax` — not against human class labels.

This is the single most important methodological decision in the suite. The claim being
tested is *"my approximation matches the thing it approximates"*. Scoring clusterings
against ground-truth class labels would instead measure *"is HDBSCAN\* good at this data
set"*, which is a different question with a different answer, and one this thesis is not
making a claim about. On MNIST-style data that substitution produces bad numbers that say
nothing about the contribution.

Class labels are written out by the prep scripts, but only for inspection. No experiment
consumes them.

`MissingEdgeMode::ExactMst` and `IncrementalKnnParams::brute_force_limit` exist precisely
to make this reference available. They are `O(n²)` and cap the comparison around n ≈ 4000.

---

## 3. Metrics

### Used

| metric | where | why |
|---|---|---|
| **k-NN recall** | incremental, drift, sensitivity | fraction of true neighbours recovered; the standard measure for the k-NN half |
| **MST total weight ratio** | bridges, budget | approx / exact. The primary quality number for the bridge estimate |
| **Sorted weight-profile gap** | bridges, budget | largest difference between the two sorted weight sequences. Since the single-linkage dendrogram's merge *heights* are exactly the sorted MST weights, this **is** dendrogram similarity, in a form you can compute |
| **DBSCAN\* ARI** | bridges, budget, incremental | agreement of the flat clusterings cut from both MSTs at a common density threshold. The downstream number that actually matters |
| **Component count before repair** | bridges, repair sweep | whether the bridges alone connect the graph — the thing they exist to do |
| **Wall-clock, split by phase** | all | instance / merge / refine, because only the instance phase parallelises across devices |
| **Peak host RSS, per-device VRAM, largest-instance bytes** | scaling | Claim B is a memory argument as much as a timing one |

### Deliberately *not* used

**Edge overlap / Jaccard / "% of MST edges recovered".** This is the trap that would have
produced a wrong table. Measured on this system:

| mpts | weight ratio | edge Jaccard | ARI |
|---|---|---|---|
| 1 | 1.0008 | 0.99 | 1.0000 |
| 8 | 1.0007 | 0.29 | 1.0000 |
| 32 | 1.0006 | 0.19 | 1.0000 |

Overlap collapses to 0.19 while the clustering is byte-identical. The reason is a theorem,
not a quirk: all MSTs of a graph share the same weight multiset, and for any threshold ε
the components of `{edges ≤ ε}` depend on **the graph, not on which MST you extracted**.
So two different MSTs of the same graph induce *identical* single-linkage hierarchies.
Mutual reachability makes this bite hard, because raising mpts flattens the weight
landscape into plateaus of ties and any tie-break gives a different-but-equally-optimal
tree. Overlap measures tie-breaking, not quality.

It is still emitted, under the column name `edge_jaccard_diagnostic_only`, so it cannot be
lifted into a results table by accident.

**NMI.** Highly correlated with ARI and biased toward many clusters. ARI alone is
defensible.

---

## 4. Data sets

Three families, and **the split is load-bearing** — they test different things and are not
interchangeable.

### 4.1 `real_paper` — the CORE-SG paper's own data

Comparability with the baseline paper. Reproduced from its stated recipes.

| set | n | dim | notes |
|---|---|---|---|
| `fma_chroma` | 106 574 | 84 | chroma_cens, 12 bins × 7 statistics |
| `fma_mfcc` | 106 574 | 140 | 20 coefficients × 7 statistics |
| `20ng_tfidf` | 10 969 | 1000 | top-1000 TF-IDF terms |
| `20ng_pca500` | 10 969 | 500 | the above, PCA to 500 components (77 % variance; the paper reports ~75 %) |

ImageNet was in the paper but is **dropped**: its Google Drive link is dead, and it needs
a licence acceptance. **GIST1M** substitutes for it — high-dimensional real image
descriptors, same role, no licence friction.

> **Trap, and it is a serious one.** The paper's 20NG recipe leaves ~3 % of documents
> with an all-zero vector (they contain none of the top-1000 terms). `TfidfVectorizer`
> L2-normalises, so every real document sits at distance **exactly 1.0** from every empty
> one — *nearer than its genuine neighbours*. The k-NN of a typical document therefore
> becomes k empty documents at identical distance: measured, **r₁ = r₂₀ exactly for
> 86.8 % of points**. Not a float32 artifact (float64 gives bit-identical results).
>
> This is not merely an estimator problem. If `r₁ = … = r₂₀` then `core_dist` is constant
> across every mpts in that range, the mutual reachability distances barely move with
> mpts, and **the whole premise of CORE-SG collapses on that data**. `prepare_datasets.py`
> now drops the empty rows, which takes the tie rate to **0.0 %**.

### 4.2 `real_scale` — scale and high dimensionality

| set | n | dim | metric |
|---|---|---|---|
| `sift1m` | 1 M (capped 200 k) | 128 | L2 |
| `gist1m` | 1 M (capped 100 k) | 960 | L2 |
| `glove100` | 1.18 M (capped 200 k) | 100 | **cosine** |

GloVe is included specifically because it is *not* L2 — it exercises the cosine path
(normalise on load, then squared-L2 ranking).

> **These sets cannot test the bridges.** Measured: **every one of them has a connected
> k-NN graph**. The bridges therefore have no work to do, both the approximate and exact
> arms are trivially identical, and any weight ratio of 1.000000 from these sets is
> vacuous. They measure *cost*, not bridge quality. Say so explicitly in the write-up.

### 4.3 `manifold` — synthetic, where the bridges are load-bearing

The family that exists because of the finding above. Data is generated in a low
dimensionality and **rotated into a high one by a random orthonormal map**, which
preserves every pairwise distance exactly. So the cluster structure and separation are
untouched while the ambient dimensionality rises and the intrinsic one stays put — giving
a data set with *known* ground-truth intrinsic dimensionality.

| set | n | ambient | intrinsic | clusters | separation | components |
|---|---|---|---|---|---|---|
| `manifold_exact` | 4 000 | 128 | 6 | 16 | 30 | **16** |
| `manifold_mid` | 100 000 | 128 | 6 | 150 | 25 | **117** |
| `manifold_wide` | 100 000 | 256 | 8 | 300 | 40 | **273** |

> **Sizing rule.** Fragmentation requires **points-per-cluster ≫ k**. With few points per
> cluster a node's k neighbours are forced to reach outside it, cross-cluster edges appear
> and the graph connects. That is why `manifold_exact` keeps the cluster count at 16: it
> has to fragment *and* stay under the `O(n²)` exact-reference ceiling at n = 4000.

This family also validates the intrinsic-dimensionality estimator against known truth: on
6-in-128 it recovers **5.67**.

### 4.4 `capacity` — larger than one GPU

`capacity10m`: 10 M points, ambient 32, intrinsic 8, 200 clusters, k = 64.

At ~848 B/point a single instance needs **7.90 GiB** resident, which does not fit an 8 GB
card once driver overhead is counted. Four instances need **1.97 GiB each** and complete.
This is a capacity argument — about what becomes *possible*, not about speed — and it is
the cleanest demonstration of why the instance design matters.

---

## 5. Baselines

### Internal (the ones that carry the argument)

| baseline | mode | tests |
|---|---|---|
| **Full recompute** | rebuild from scratch each batch | Claim A. This is the primary baseline |
| **Exact CORE-SG** | `MissingEdgeMode::ExactMst` + exact k-NN | Claim C ground truth |
| **k-NN only** | `MissingEdgeMode::None`, repair off | the no-bridges ablation: do bridges matter at all |
| **Ambient vs intrinsic budget** | `budget_dimension` | which dimensionality term the budget should use |

There is no existing incremental multi-mpts CORE-SG, so full recompute *is* the honest
comparison, and the absence of a competitor strengthens rather than weakens the
contribution.

### External

**FISHDBC** (Dell'Amico, arXiv:1910.07283) — the closest published work: an incremental
HDBSCAN\* approximation that keeps an incremental MST fed by HNSW candidate edges. It is
the *opposite* design decision on the same problem (maintain an MST vs avoid maintaining
one), which makes it worth comparing and mandatory to cite.

`scripts/baseline_fishdbc.py` emits the same CSV schema. FISHDBC is not on PyPI, and its
`setup.py` imports Cython without declaring it as a build dependency, so pip's isolated
build environment fails on it:

```bash
conda activate coresg     # provides cython
pip install --no-build-isolation git+https://github.com/matteodellamico/flexible-clustering
```

Verified working (2026-07-29, `manifold_mid`, 3000 points, 4 batches): ARI against exact
HDBSCAN\* reaches 1.0000 by batch 3, ~0.26 s per 375-point batch.

> **Trap, and it cut against the baseline.** The reference must be **recomputed on the
> points present at each batch**. Computing it once on the whole file and slicing scores
> FISHDBC against structure it has not been shown, producing an artificially low early
> ARI that climbs as the stream catches up (measured: 0.7588 → 0.8527 at batch 1 once
> fixed). Handicapping a competitor through a measurement artifact is worse than the
> equivalent error in your own favour. The residual ramp is legitimate and now visible in
> the `reference_clusters` column: at batch 1 the reference itself finds only 120 of the
> eventual 150 clusters, because clusters below `min_cluster_size` are not yet detectable.

> **Compare quality and cost *slope*, never raw wall-clock.** FISHDBC is single-threaded
> Python; this project is multi-threaded C++/CUDA. A wall-clock race measures the
> languages. Score both against the same reference (exact HDBSCAN\*) via ARI, and compare
> how per-batch cost *grows* with stream length.
>
> Scope difference to state explicitly: FISHDBC serves a **single** mpts. CORE-SG exists
> to serve a whole range from one graph. FISHDBC is a baseline for the incremental claim,
> not for the multiple-mpts claim, where it has no equivalent.

### Deliberately excluded

**FAISS / GNND / RAFT.** This project does not claim a faster k-NN graph — it *uses*
Dynamic NN-Descent. Cite it, don't re-benchmark it. One brute-force recall check is enough
to show the k-NN half is sound. Defending a contribution you did not make is the classic
thesis time-sink.

**I-HASTREAM.** Incremental MST over *micro-clusters*, a different granularity. Related
work, not a fair baseline.

---

## 6. The experiments

Run via `coresg_exp --experiment NAME`, orchestrated by `scripts/run_suite.py`.

### `bridges` — Claim C, the core result
Anti-hub bridges vs exact `MST_kmax`, **both arms sharing an exact k-NN half** so only the
bridge estimate varies. Reports bridge construction time, weight ratio and ARI at several
mpts, the k-NN-only ablation, and an anti-hub count sweep. Capped at n ≈ 4000 by the
`O(n²)` reference. **Only `manifold_exact` genuinely stresses it.**

### `budget` — which budget rule to adopt
Five candidate rules, all on the same exact k-NN half and the same exact-MST reference, so
the only thing that varies is how many anti-hubs the budget asks for:

| name | rule |
|---|---|
| `ambient` | the specified `d·ln(|E|)`, uncapped |
| `ambient_capped` | the same, plus the `max_pair_ops` cost backstop |
| `intrinsic` | ambient `d` replaced by estimated intrinsic dimensionality |
| `intrinsic_x2`, `intrinsic_x4` | intrinsic scaling with headroom |

Also emits `budget_uncapped` (what the rule asked for before any clamp) and
`budget_capped`, which is what distinguishes *"the formula is well scaled"* from *"a
backstop rescued it"*.

### `incremental` — Claim A
Build on a fraction, insert the rest in batches. Per-batch cost split by phase, recall
against an exact reference **built over the points that exist at that moment**, and a
full-rebuild baseline on the same final data.

> **Trap.** Scoring an early batch against a reference built over the *whole file* counts
> not-yet-inserted neighbours as misses. That bug made early batches look like 0.62 recall
> when they were actually 0.9989.

### `arrival` — robustness
The same experiment with data arriving **grouped by region** instead of shuffled, which is
what a real stream often looks like and is the adversarial case for any incremental
method.

### `drift` — long-horizon Claim A
Many small batches with quality sampled at **every** one, plus `recall_delta_vs_first`.
Separate from `incremental` because slow degradation over a long stream is invisible when
you only check the endpoints.

### `mpts` — the CORE-SG payoff
All MSTs for mpts ∈ [1, k_max] from one graph, against the extrapolated cost of computing
each from the complete graph. This is what CORE-SG exists for and it must appear.

### `scaling` — Claim B
CPU threads, then GPU count × instance count. Reports the phase split (hence the parallel
fraction and the Amdahl ceiling), peak host RSS, per-device VRAM, and largest-instance
bytes.

### `sensitivity` — robustness
k_max, instance count, anti-hub multiplier, per-component quota, NN-Descent sample size.

> **Trap.** This experiment does **not** report MST quality ratios. Two separately built
> `CoreSG` objects get two different randomised k-NN halves, and that variation exceeds the
> bridge effect — it shows up as a weight ratio *below* 1.0, which is impossible for a real
> approximation. Bridge quality belongs in `bridges`, where the k-NN half is held exact.

---

## 7. Output

One long-format CSV, appended across runs and machines:

```
experiment,dataset,n,dim,k,config,value,seed,batch,metric,measurement
```

Long format so heterogeneous metrics coexist without reshaping, flushed per row so a
three-hour sweep survives a crash, appendable so laptop and cluster results concatenate.
`batch` is `-1` for run-level rows and the device id for per-device memory rows.

**Every experiment runs multiple seeds** (`--seeds`, default 5). NN-Descent is randomised
and its reverse cache is deliberately race-tolerant, so single-run numbers are not
defensible. Report mean and spread.

---

## 8. Running it

```bash
# one-off preparation
python scripts/prepare_datasets.py newsgroups --out-dir data
python scripts/prepare_datasets.py fma data/fma_metadata.zip --out-dir data
python scripts/prepare_datasets.py hdf5 data/glove-100-angular.hdf5 --key train --out data/glove100.fvecs
python scripts/run_suite.py --generate

# the suite
python scripts/run_suite.py --list
python scripts/run_suite.py --seeds 5 --out results.csv

# subsets
python scripts/run_suite.py --only bridges,budget --groups manifold
python scripts/run_suite.py --only scaling --families capacity10m   # cluster only
```

On the cluster (4× RTX 2080 SUPER, sm_75):

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_BUILD_TYPE=Release
```

> **Measurement hazards on multi-GPU.** CUDA context creation costs 100–300 ms *per
> device* and updates are sub-second — warm every device before timing. And PCIe is
> shared: the design has no cross-device traffic, but four GPUs uploading instance vectors
> at once contend for one bus, which is likely to dominate the deviation from linear
> scaling ahead of Amdahl.

---

## 9a. Cluster run, 2026-08-01 (4x RTX 2080 SUPER)

First full sweep. 14,878 rows, 5 seeds, 37 runs.

**Incremental beats a full rebuild on every data set** — the strongest result in the set:

| data set | recall (incremental) | recall (rebuild) | ARI vs rebuild |
|---|---|---|---|
| gist1m | **0.9771** | 0.8884 | 0.9910 |
| glove100 | **0.9043** | 0.7742 | 0.9732 |
| 20ng_pca500 | **0.9656** | 0.8246 | 0.9443 |
| sift1m | **0.9968** | 0.9817 | 0.9992 |
| manifold_mid | **0.9994** | 0.9960 | 0.9977 |

Eight incremental batches cost 2.5–3x a *single* rebuild, i.e. ~3x cheaper than rebuilding
per batch, at higher quality — the incremental path accumulates more total descent.

**GPU strong scaling** (instance count fixed across device counts), speed-up on 4 GPUs at
16 instances: gist1m **2.68x**, sift1m **2.53x**, manifold_mid **2.50x**, capacity10m
**3.23x**. Mean parallel fraction f = 0.725, so the 4-GPU Amdahl ceiling is 2.19x — the
larger data sets exceed it because f rises with instance size.

**Budget: ambient is not a sublinear estimate.** It selects **100 % of n** on 20ng_pca500
and gist1m, and 29–40 % everywhere else. Intrinsic uses 1.4–10 %, runs 5–30x faster, and
costs at most 0.6 % in MST weight (0.1 % at x4).

### What this run exposed

**Only one data set exercised the bridges.** `components(knn_only)` was 1 on 20ng_pca500,
fma_chroma, fma_mfcc, gist1m and glove100; 2 on sift1m; **16 on manifold_exact**. Five of
seven real sets are connected, so their weight ratio of 1.00000 is vacuous — nothing was
approximated. Claim C rested on a single data set. Fixed by adding `manifold_c8` /
`manifold_c24` / `manifold_c32` (7 / 24 / 32 components), which cost ~2 s each to run.

**The capacity experiment never tested its own premise.** Instance counts were
{4, 8, 16}, all of which fit (2.01 / 1.50 / 1.25 GiB). `instances=1` — the 7.90 GiB
configuration that does not fit an 8 GB card — was unreachable. The sweep cost **~2 days**
and produced a scaling curve the other three data sets already gave. Replaced by a
1-seed probe at instances {1,2,4} with the OOM fallback disabled.

**bridges and budget are deterministic.** They set `brute_force_limit = n`, so every
quality figure came back with a standard deviation of exactly 0.00000 across 5 seeds.
Dropped to 2 seeds; extra seeds buy timing jitter only.

**`refine_nodes_fraction` was exactly 0.25 on every data set and every batch** — the cap
is always binding, so the measurement reflects the ceiling rather than the algorithm's
natural appetite, and `refine_max_fraction` is not in the sensitivity sweep. Open.

## 9. Reference measurements

Recorded so drift in the routine is detectable. Dev box: RTX 4070 Ti SUPER, CUDA 12.6.

| measurement | value |
|---|---|
| Theorem 1 verification (exact CORE-SG vs complete graph) | identical weight, all mpts ∈ [1,16] |
| Bridge weight ratio, `manifold_exact` | **1.00000**, ARI 1.0000 |
| Bridge components, `manifold_exact` | 16 → **1** |
| Bridge speed-up vs exact MST | **38×** (0.0073 s vs 0.2766 s) |
| mpts sweep (32 MSTs from one graph) | 0.060 s vs 8.9 s → **149×** |
| k-NN recall, n=200k d=32 | 0.9236 CPU / 0.9237 CUDA (identical quality) |
| k-NN build, n=200k d=32 | 16.6 s (1 core) → 1.46 s (all) → **0.74 s (GPU)** |
| Incremental final recall vs rebuild | 0.9985 vs 0.9854 |
| Refinement set (capped) | 25 % of graph, recall 0.9996 |
| Drift over 20 batches | **+0.026** (improves, then flat) |
| Parallel fraction f | **0.79** → 4-GPU ceiling **2.45×** |
| Intrinsic dim: SIFT / GIST / GloVe / 20NG | 10.2 / 28.5 / 34.5 / 36.3 |
| Intrinsic budget saving | 5–49× fewer anti-hubs, identical quality on real data |
| Capacity: 1 instance vs 4 | 7.90 GiB (OOM on 8 GB) vs 1.97 GiB |

---

## 10. Open gaps

1. **FISHDBC is installed and verified** but has not been run across the matrix — only a
   3000-point smoke test on `manifold_mid`. It is single-threaded Python, so keep `--limit`
   modest (20 k or below) when running it for real.
2. **Multi-GPU numbers are unmeasured.** One GPU locally. On one device, more instances is
   *slower* (they serialise); the instance count is a scheduling knob for multi-GPU, not a
   single-device speed knob.
3. **FMA is converted and verified to build** (30 k rows, ~0.33 s) but no experiment has
   been run on it yet — it is in the matrix and will be covered by a full suite run. Note
   it joins the other real sets in having a **connected k-NN graph** (components 1 → 1),
   so it tests cost, not bridge quality.
4. **The budget rule is decided on evidence but not yet adopted as the default.** See §12.
   `budget_dimension` still defaults to `Ambient`, i.e. the specified rule.
5. **Anti-hub count sensitivity is data-dependent** — flat on low-d blobs, monotone on
   manifolds, irrelevant on connected real data. Do not generalise from one shape.

---

## 11. The budget rule: measured comparison

All five rules, exact k-NN half on every arm, exact `MST_kmax` as reference, 3 seeds.

**`manifold_exact`** — 16 fragmented components, so the bridges are load-bearing and
quality differences are visible:

| rule | eff dim | asked | selected | % of n | bridge s | ratio @2 | ratio @8 | ARI |
|---|---|---|---|---|---|---|---|---|
| `ambient` | 128 | 1 422 | 1 422 | 35.5 % | 0.0090 | 1.00000 | 1.00000 | 1.0000 |
| `ambient_capped` | 128 | 1 422 | 1 422 | 35.5 % | 0.0084 | 1.00000 | 1.00000 | 1.0000 |
| `intrinsic` | 5.5 | 56 | 56 | 1.4 % | 0.0005 | 1.00736 | 1.00591 | 1.0000 |
| `intrinsic_x2` | 5.5 | 111 | 111 | 2.8 % | 0.0005 | 1.00332 | 1.00267 | 1.0000 |
| **`intrinsic_x4`** | 5.5 | 222 | 222 | 5.5 % | **0.0007** | **1.00133** | **1.00107** | 1.0000 |

**`gist1m`** — connected k-NN graph, so the bridges are idle and only cost differs:

| rule | eff dim | asked | capped | selected | % of n | bridge s |
|---|---|---|---|---|---|---|
| `ambient` | 960 | 10 944 | clamped to n | 4 000 | **100 %** | 0.0756 |
| `ambient_capped` | 960 | 10 944 | yes | 2 041 | 51.0 % | 0.0219 |
| `intrinsic` | 24.1 | 274 | – | 274 | 6.9 % | 0.0015 |
| **`intrinsic_x4`** | 24.1 | 1 094 | – | 1 094 | 27.4 % | 0.0077 |

### Reading it

**The multiplier does exactly what the flat cost curve predicted.** On the manifold,
quality improves monotonically (1.00736 → 1.00332 → 1.00133) while bridge time barely
moves (0.0005 → 0.0007). `intrinsic_x4` lands within **0.13 %** of the exact MST at **13×
less bridge time than ambient**.

**The cost cap alone is not enough.** On GIST it does help — 4 000 → 2 041 anti-hubs, 3.5×
faster — but it still selects **51 % of n**. It treats the symptom; the rule is still
asking for 10 944 anti-hubs on 4 000 points.

**Ambient's degeneracy is visible in the `capped` column.** On GIST it asks for 10 944 and
gets clamped to n = 4 000: it selects *every point*, so the "approximation" is the exact
MST it exists to replace.

**Recommendation: `intrinsic_x4` as the rule, `max_pair_ops` retained as a backstop.** They
compose — the multiplier fixes the scaling, the cap covers the case where the intrinsic
estimator is unreliable (sparse duplicate-heavy text, where it correctly refuses and falls
back to ambient). Adopting it is a one-line default change in `CoreSGParams`; it is left at
`Ambient` so the specified rule remains the baseline until deliberately changed.

Caveat before adopting: this is two data sets. Run `--only budget` across the full matrix
first — `manifold_wide` in particular, which has 273 components and will stress coverage
harder than `manifold_exact`'s 16.

## 12. Changing this suite

If you change the routine, update this file in the same commit. In particular:

- adding a data set → say which family and **what it tests that the others don't**;
- adding a metric → say what would falsify a claim with it, and check it is not another
  edge-overlap (a number that moves for reasons unrelated to quality);
- changing a default → the reference table in §9 becomes stale; re-measure or mark it;
- comparing anything against anything → confirm the k-NN half is held fixed, or you are
  measuring NN-Descent's randomness rather than the thing you meant to measure.
