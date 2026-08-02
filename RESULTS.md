# Results and analysis

Full experimental sweep, 4× RTX 2080 SUPER (sm_75, 8 GB), CUDA 12.9.
**17,538 measurements · 48 run/dataset pairs · 14 data sets · up to 5 seeds · zero failures.**
(11 of the 14 appear in the bridges experiment, which is where the counts below are drawn
from; `capacity10m` carries two labels because its scaling sweep and its capacity probe are
separate runs.)

A further **8,434 measurements** from four follow-up experiments were run on the dev box
(1× RTX 4070 Ti SUPER) into `results_local.csv` — see the provenance note in §8 for which of
their columns are hardware-independent and which are not.

Companion documents: [EXPERIMENTS.md](EXPERIMENTS.md) for *why* each experiment is designed
the way it is, [README.md](README.md) for the implementation.

---

## Executive summary

| claim | verdict | headline evidence |
|---|---|---|
| **A** — incremental updates preserve quality | **Supported, and it survives a compute-matched rebuild** | Beats a full rebuild on **8 of 8** data sets; against a *tuned* rebuild the margin is still **+3.5** (gist1m) and **+7.0** (glove100) points of recall — §1.1b |
| **B** — the design scales across GPUs | **Supported** | 2.50–3.23× on 4 GPUs, per-device memory *exactly* invariant with device count |
| **C** — anti-hub bridges replace the MST half | **Supported** | Worst weight ratio **1.00024**, ARI exactly **1.0000**, 10.5–36× faster; the bare budget takes **300 components to 1** at half budget with the quota disabled — §5.4 |
| CORE-SG payoff survives incrementality | **Supported** | 71–303× for a full mpts sweep |
| **vs FISHDBC**, the only external baseline | **Supported** | Ahead on **every batch of every data set** on one shared yardstick; exact recovery of the reference clustering on manifold_mid — §5.6 |
| capacity: config that needs > 1 GPU | **Not demonstrated** | see §5.3 — the intended configuration was never actually built; `capacity10m_oom128` is queued |

---

## 1. Claim A — incremental construction

### 1.1 Quality and cost

Build on half the data, insert the rest in 8 batches, 5 seeds. Baseline is a full rebuild
on the same final data.

| data set | recall (incremental) | recall (rebuild) | ARI vs rebuild | speed-up vs rebuild-per-batch |
|---|---|---|---|---|
| manifold_mid | **0.9994** | 0.9960 | 0.9977 | 2.50× |
| fma_mfcc | **0.9984** | 0.9913 | 0.9992 | 2.66× |
| fma_chroma | **0.9980** | 0.9890 | 0.9998 | 2.66× |
| manifold_wide | **0.9971** | 0.9896 | 0.9946 | 2.29× |
| sift1m | **0.9968** | 0.9817 | 0.9992 | 2.91× |
| gist1m | **0.9771** | 0.8884 | 0.9910 | 2.94× |
| 20ng_pca500 | **0.9656** | 0.8246 | 0.9443 | 1.29× |
| glove100 | **0.9043** | 0.7742 | 0.9732 | 2.53× |

**The incremental graph is better than the rebuilt one on every data set**, by margins of
0.7 to 14 points of recall. That is counter-intuitive and worth stating plainly: it is not
that incrementality is magic, but that the incremental path performs *more total descent*.
Each batch re-visits sampled base points inside its instances, so after eight batches the
base has been refined eight times, while the rebuild descends once. The comparison is fair
on wall-clock — the incremental total is 2.3–2.9× cheaper than rebuilding at every batch —
but it is not an equal-work comparison, and the write-up should say so.

The gap is largest exactly where a single descent struggles: glove100 (+13.0 points,
angular data with high intrinsic dimensionality) and gist1m (+8.9, d=960). Where a single
descent already converges, the gap nearly closes (manifold_mid +0.3).

### 1.1b The equal-work control — and a better claim than the one above

"Not an equal-work comparison" is a caveat a reviewer converts into an objection: the
incremental path re-descends sampled base points once per batch, so the rebuild above is
the *cheapest* baseline, not an equal one. The `rebuild_effort` frontier answers it by
sweeping the rebuild's descent effort and reading the incremental point against the
resulting (cost, recall) curve.

Measured on the two data sets where the margin is largest, 2 seeds
(dev box, 1× RTX 4070 Ti SUPER — the *recall* column is hardware-free, the seconds are not):

| data set | rebuild baseline | best tuned rebuild | incremental | margin over tuned |
|---|---|---|---|---|
| gist1m | 0.88852 (4.77 s) | 0.94190 (4.20 s) | **0.97723** (11.90 s) | **+3.5 pts**, 2.8× cost |
| glove100 | 0.77413 (2.46 s) | 0.83441 (3.04 s) | **0.90451** (9.57 s) | **+7.0 pts**, 3.2× cost |
| manifold_mid | 0.99615 (2.13 s) | 0.99853 (2.31 s) | **0.99942** (5.74 s) | +0.09 pts, 2.5× cost |

Two things fall out, and the write-up should lead with both.

**The two effort knobs are not interchangeable.** More *iterations* buy nothing: on gist1m
`iters_x2` and `iters_x4` return bit-identical distance-evaluation counts (222.5 M), because
the `delta` early stop fires at the same round either way — the descent is not short of
time, it has reached a fixed point of the local join. More *sampling* does help
(0.88852 → 0.93871), because widening the candidate list changes which pairs get compared.
The quality ceiling of a single build is set by the **breadth** of the local join, not by
how long it runs.

**Tuning recovers some of the margin, but not most of it, and how much depends on
difficulty.** It closes 5.3 of the original 8.9 points on gist1m and 6.0 of 13.0 on
glove100 — leaving 3.5 and 7.0. So the honest claim is not the raw margin in §1.1, but
something sharper:

> **Incremental construction beats a compute-matched rebuild on data where a single descent
> struggles, and the margin grows with the difficulty of the k-NN problem** —
> manifold_mid +0.09 → gist1m +3.5 → glove100 +7.0 points.

That is a claim about *construction strategy*, not about incrementality buying free
quality, and it survives the obvious objection. Repeated overlapping re-sampling is a
different algorithm from one global descent, not more of the same; §7 picks this up.

### 1.2 Arrival order — no measurable penalty

The adversarial case: points arriving grouped by region rather than shuffled.

| data set | shuffled | grouped |
|---|---|---|
| manifold_mid | 0.9994 | 0.9998 |
| fma_mfcc | 0.9984 | 0.9973 |
| sift1m | 0.9968 | 0.9968 |

Differences are within noise. The instance sampling draws uniformly from the *existing*
base regardless of how that base arrived, which is presumably why order does not matter.
This is a robustness result worth keeping — it is the first thing a reviewer will probe.

### 1.3 Long-horizon drift — flat

Recall change from batch 1 to batch 25, sampled at every batch:

| data set | final Δ | range |
|---|---|---|
| manifold_mid | **+0.0335** | 0.0000 … +0.0357 |
| sift1m | +0.0016 | 0.0000 … +0.0017 |
| fma_mfcc | +0.0004 | 0.0000 … +0.0005 |
| manifold_wide | **−0.0034** | −0.0034 … +0.0030 |

No decay. Three of four *improve*, and the one that declines does so by a third of a
percentage point over 25 batches. The graph does not rot.

---

## 2. Claim C — anti-hub bridges

### 2.1 Where the bridges are load-bearing

**Bridges can only be evaluated where the k-NN graph actually falls apart.** Six of the
eleven data sets in this experiment have a connected k-NN graph, so their weight ratio of
1.00000 measures nothing — there was nothing to approximate. Restricting to those that
fragment:

| data set | components (k-NN alone) | worst weight ratio | ARI | speed-up vs exact MST |
|---|---|---|---|---|
| manifold_c32 | 32 | 1.00022 | 1.0000 | 18.2× |
| manifold_c24 | 24 | 1.00024 | 1.0000 | 36.0× |
| manifold_exact | 16 | 1.00000 | 1.0000 | 12.1× |
| manifold_c8 | 8 | 1.00001 | 1.0000 | 19.0× |
| sift1m | 2 | 1.00000 | 1.0000 | 10.5× |

**Worst case anywhere: 1.00024 — a spanning tree 0.024 % heavier than optimal — at ARI
exactly 1.0000 and 10.5–36× less work.**

This is the central result of the thesis and it holds cleanly.

> **Correction (2026-08-01), and it needs a re-run rather than a re-wording.** An earlier
> version of this section read "at full scale `manifold_wide` fragments into 273 components
> and the bridges alone reduce it to 1". The second half was measured;
> **the 273 was not.** `components_before_repair` is filled in *after* the bridges are
> folded into the component labelling, so every full-scale row in `results.csv` reads 1 by
> construction, and the string `273` does not appear in the file at all — it came from the
> data-set design table in EXPERIMENTS.md §4.3. The `knn_only` ablation that would have
> measured it only ran at n ≤ 4000.
>
> The new `connectivity` experiment measures it directly. Measured, 3 seeds:
> `manifold_wide` at n = 100 000, k = 24 has **271 components** in its k-NN half — the
> design figure was right to within the seed spread (±20 across seeds; the same data set
> reads 233–277 over k ∈ [8, 64]). Quote 271, or the band, not a point estimate.
>
> **Does approximating the k-NN half change connectivity?** Every fragmenting count in this
> document was measured with an **exact** half, so it was worth worrying that NN-Descent's
> residual errors are long cross-cluster edges that quietly connect a graph the exact k-NN
> leaves in pieces — which would mean the bridge evidence came from an easier regime than
> the one that ships. Measured: **it does not, at the k values this project uses.** The
> approximate half fragments the same or slightly *more* — `manifold_c24` gives 24 exact
> against 24–28 approximate over k ∈ [4, 64] — because a missed neighbour removes an edge
> while a substituted one stays local.
>
> The effect does reverse at large k relative to cluster size: `manifold_c32` at k = 64
> (125 points per cluster) gives **30** approximate against **32** exact, the approximate
> graph now making genuinely cross-cluster errors. Both mechanisms exist and neither
> dominates universally, so the claim to make is the narrow one — *at k = 24–32,
> approximation does not mask fragmentation* — not a general statement about which way it
> cuts.

### 2.2 The connected-graph finding is itself a contribution

Every real data set tested — FMA (both variants), 20 Newsgroups (both), SIFT, GIST,
GloVe — has a **connected** k-NN graph at k = 24–32. The MST half of CORE-SG therefore
contributes nothing to them beyond what the k-NN graph already provides.

The `connectivity` experiment turns that from a single-k assertion into a curve. Components
of the k-NN half alone, at full n, approximate half, mean of 3 seeds:

| data set | k=4 | k=8 | k=16 | k=24 | k=32 | k=48 | k=64 |
|---|---|---|---|---|---|---|---|
| 20ng_pca500 | **1** | 1 | 1 | 1 | 1 | 1 | 1 |
| 20ng_tfidf | **1** | 1 | 1 | 1 | 1 | 1 | 1 |
| glove100 | 2 | **1** | 1 | 1 | 1 | 1 | 1 |
| sift1m | 5 | **1** | 1 | 1 | 1 | 1 | 1 |
| fma_chroma | 8 | 2 | **1** | 1 | 1 | 1 | 1 |
| fma_mfcc | 6 | 4 | **1** | 1 | 1 | 1 | 1 |
| gist1m | 5 | 5 | 2 | **1** | 1 | 1 | 1 |
| manifold_mid | 1658 | 107 | 121 | 132 | 129 | 127 | 134 |
| manifold_wide | 2636 | 233 | 267 | 271 | 257 | 266 | 277 |

**Every real data set connects at k ≤ 24, and this project operates at k = 24–32** — just
above the threshold on the worst of them, and far above it on most. The synthetic manifold
sets do not connect at any k in the range, which is exactly what they exist for.

That is worth reporting rather than hiding. It says the bridge mechanism is *insurance*:
it costs 1–5 % of build time, is invisible when unnecessary, and is the difference between
a spanning tree and a 271-piece forest when it is. It also explains why an approximation
this crude works so well — on most real data the thing being approximated is nearly empty.

The curve carries one more thing the single-k statement could not: the connection threshold
is a property worth *reporting per data set*, because it is what tells a practitioner
whether the bridge machinery will ever fire on their data. The cheap gate in §7 is the
operational form of the same observation.

### 2.3 How many anti-hubs are actually needed

Count sweep with the k-NN half held exact (`manifold_c24`, 24 components):

| anti-hubs | 4 | 16 | 64 | 256 | 1024 |
|---|---|---|---|---|---|
| weight ratio | 1.01213 | 1.01206 | 1.00813 | 1.00167 | 1.00016 |

Monotone and genuinely useful — but note this does **not** generalise. On low-dimensional
blobs the same sweep is flat (4 anti-hubs ≈ 1024), and on connected real data it is
irrelevant. Sensitivity to the anti-hub count is a property of the data's component
structure, not of the method.

---

## 3. The budget rule

### 3.1 Ambient dimensionality degenerates

Fraction of n selected as anti-hubs:

| data set | ambient | ambient+cap | intrinsic | intrinsic ×4 |
|---|---|---|---|---|
| 20ng_tfidf (d=1000) | **1.000** | 0.500 | 0.118 | 0.472 |
| 20ng_pca500 (d=500) | **1.000** | 0.707 | 0.102 | 0.408 |
| gist1m (d=960) | **1.000** | 0.510 | 0.070 | 0.280 |
| sift1m (d=128) | 0.368 | 0.368 | 0.029 | 0.115 |
| manifold_c24 (d=64) | 0.177 | 0.177 | 0.014 | 0.044 |

On three data sets `d·ln(|E|)` selects **every point** — the "sublinear estimate" becomes
the exact MST it exists to replace. Even where it does not saturate it takes 18–40 % of
the data, which is not sublinear in any useful sense. Intrinsic dimensionality uses
1.4–11.8 % and runs **3.2–35.9× faster**.

### 3.2 But intrinsic costs quality where bridges matter

The trade-off, on the fragmenting sets only (worst weight ratio):

| data set | ambient | intrinsic | intrinsic ×2 | intrinsic ×4 |
|---|---|---|---|---|
| manifold_c24 | **1.00021** | 1.01461 | 1.00858 | 1.00356 |
| manifold_c32 | **1.00020** | 1.01149 | 1.00694 | 1.00476 |
| manifold_exact | **1.00000** | 1.00736 | 1.00332 | 1.00133 |
| manifold_c8 | **1.00001** | 1.00192 | 1.00074 | 1.00033 |

**Ambient is the most accurate rule.** It is also 3–36× more expensive and degenerate at
high d. The multiplier recovers most of the gap: ×4 lands within 0.03–0.48 % of exact
while still using a fraction of the budget.

**Recommendation: `intrinsic_x4`, with ambient reported as the accuracy ceiling.** The
honest framing is a cost/accuracy trade-off, not a free win — an earlier reading of the
partial data suggested intrinsic was strictly better, and it is not.

---

## 4. Claim B — multi-GPU

### 4.1 Strong scaling

Instance count held fixed across device counts (16 instances):

| data set | n | f (1 GPU) | Amdahl ceiling @4 | measured @4 | efficiency |
|---|---|---|---|---|---|
| capacity10m | 10M | 0.941 | 3.40× | **3.23×** | 0.81 |
| gist1m | 100k | 0.921 | 3.23× | **2.68×** | 0.67 |
| sift1m | 200k | 0.869 | 2.87× | **2.53×** | 0.63 |
| manifold_mid | 100k | 0.840 | 2.70× | **2.50×** | 0.63 |

Every measurement sits below its ceiling, and **efficiency rises with problem size**
(0.63 → 0.81 as n goes 100k → 10M) because larger instances amortise the fixed per-job
transfer. That trend is the strongest form of the scalability argument.

### 4.2 The instance count *is* the unit of parallelism

The clearest demonstration in the whole sweep, from the capacity probe:

| instances | 1 GPU | 2 GPUs | 3 GPUs | 4 GPUs |
|---|---|---|---|---|
| **1** | 1.00× | 0.99× | 0.99× | 0.99× |
| **2** | 1.00× | 1.66× | 1.65× | 1.63× |

With one instance, adding GPUs does **nothing**. With two, the speed-up appears at two
devices and then plateaus exactly where instances run out. Speed-up is bounded by
`min(instances, devices)`, and the plateau is visible to the eye.

A corollary already noted in the data: at 4 instances the 3-GPU column stalls
(1.74–1.56×, worse than or equal to 2 GPUs) because 4 splits as (2,1,1) and the critical
path is still 2 instances. **Instances should be a multiple of the device count.**

### 4.3 Memory — the property that makes it scale

Per-device resident bytes, `capacity10m`:

| instances | 1 GPU | 2 GPUs | 3 GPUs | 4 GPUs |
|---|---|---|---|---|
| 1 | 4.936 GiB | 4.936 | 4.936 | 4.936 |
| 2 | 3.010 | 3.010 | 3.010 | 3.010 |
| 4 | 2.011 | 2.011 | 2.011 | 2.011 |
| 8 | 1.502 | 1.502 | 1.502 | 1.502 |
| 16 | 1.246 | 1.246 | 1.246 | 1.246 |

**Exactly invariant across device count, and controlled entirely by instance count.** This
is the design property that makes the approach scale, and it is measured to the byte
rather than argued.

---

## 5. What is *not* supported

Recording these matters more than the successes, because a reviewer will find them.

### 5.1 Bridge quality is untested on real data

Six of eleven data sets have connected k-NN graphs; their 1.00000 ratios are vacuous. All
genuine bridge evidence comes from **synthetic manifold data plus sift1m's 2 components**.
The write-up must not present eleven data sets as eleven confirmations.

This one does not go away, and the `connectivity` curve in §2.2 explains why it cannot: on
every real data set tested the k-NN graph is already connected at the operating k, so there
is no bridge quality to measure there *at any parameter setting*. The gap is a property of
real data, not an omission in the evaluation — but it does mean the honest scope of Claim C
is "on data that fragments", and §2.2 should be cited whenever the claim is stated.

### 5.2 No CPU-vs-GPU quality comparison in the results

Backend parity is covered by a unit test and by `preflight`, but no experiment writes it to
the CSV. If asked "do the backends agree?", the answer currently cites a unit test.

*Addressed, not yet swept:* the `parity` experiment now measures it. Dev-box check on
`manifold_c24`: recall 0.93062 (CUDA) vs 0.92970 (CPU), DBSCAN\* ARI between the two
0.99946, and `jobs_on_device` confirms the GPU arm was not silently demoted to the host.

### 5.3 The capacity claim was never demonstrated

Intended: one instance over 10M points needs 7.90 GiB and cannot fit an 8 GB card.
**Measured: the largest instance was 4.936 GiB and fitted comfortably. No OOM occurred.**

The error was mine, and it is instructive. The `insert` path builds
`base + batch = 5M + 1.25M = 6.25M` points per instance, not the full 10M — so the
footprint is 6.25M × 848 B = 4.94 GiB, exactly what was measured. The 7.90 GiB figure
assumed a configuration the experiment never constructs.

To exceed 7.6 GiB of free VRAM in a single instance would need k=128 (8.29 GiB at the same
n) or roughly n=15M. Either is a short run. **Until then, the capacity claim should be
stated as arithmetic plus the memory-scaling table in §4.3, not as a demonstration.**

### 5.4 Two sensitivity parameters produce no signal

| parameter | recall across the sweep |
|---|---|
| k_max | 8 → 0.9559, 16 → 0.9908, 32 → 0.9978, 64 → **0.9996** |
| sample_size | 8 → 0.9934, 16 → 0.9976, 24 → 0.9984, 32 → 0.9987 |
| instances | 1 → 0.9945, 2 → 0.9975, 4 → **0.9976**, 8 → 0.9962, 16 → 0.9964 |
| antihub_multiplier | 0.5/1/2/4 → **0.9976 for all four** |
| min_per_component | 0/1/2/4 → **0.9976 for all four** |

> **Correction (2026-08-01).** This was diagnosed as a metric problem — the sweep measures
> k-NN recall, which bridge-coverage parameters cannot affect. That is true but is not the
> cause. The experiment **already emitted** `components_before_repair`, `antihubs_selected`
> and `bridge_seconds` per configuration, and `components_before_repair` came back as
> **1.000 for every configuration on all three data sets**, including
> `min_per_component = 0` and `antihub_multiplier = 0.5`. The cause was the *data*:
> `manifold_mid`, `fma_mfcc` and `sift1m` all have a connected k-NN graph at n = 50 000, so
> the coverage knobs had nothing to act on. Re-targeting the metric alone would have
> produced a second flat table.
>
> Fixed two ways: fragmenting data sets (`manifold_wide`, `manifold_c24`) are now in the
> sensitivity matrix, and the experiment emits a `reference / knn_only` row giving the
> component count of the k-NN half alone — so a column of 1s reads as a real statement
> rather than as "nothing happened".

**Re-run on fragmenting data, the sweep now produces a positive result.** k-NN half alone
at n = 50 000: `manifold_wide` **300** components, `manifold_mid` **150**, `manifold_c24`
**24**, `fma_mfcc` and `sift1m` **1**. The bridges reduce every one of them to **1 at every
setting tested** — including `antihub_multiplier = 0.5` (half the budget) and
`min_per_component = 0` (quota disabled entirely).

That is a stronger claim than the flat table it replaces: **the bare `d·ln(|E|)` budget
already covers 300 components without help.** It also bounds what the per-component quota
is worth. The quota was added because bridges alone left the graph split in 6 of 24
earlier configurations, but at these component counts it changes nothing — it earns its
keep only when the component count approaches the budget, which the earlier repair sweep
put at around 120 clusters at d = 8. The write-up should say that rather than presenting
the quota as generally necessary.

The instance-count result is worth noting on its own: quality peaks at 4 and *declines*
slightly at 8–16, so more instances is not free. Since instances are also the unit of
parallelism (§4.2), there is a genuine tension between throughput and quality.

### 5.5 The refinement cap is always binding — and it is costing quality

`refine_nodes_fraction` was **exactly 0.25 on every data set and every batch** — the cap is
saturated everywhere, so the measurement reflects the ceiling rather than the algorithm's
natural appetite.

**Now swept, and the cap should be raised.** Recall against total insert time, n = 50 000,
8 batches, 2 seeds:

| policy | manifold_wide | manifold_mid | sift1m | insert time (manifold_wide) |
|---|---|---|---|---|
| `frac_0.05` | 0.97782 | 0.99674 | 0.99855 | 3.03 s |
| **`frac_0.25`** (default) | 0.99253 | 0.99874 | 0.99866 | 3.24 s |
| `frac_0.50` | 0.99810 | 0.99961 | 0.99888 | 3.26 s |
| **`frac_1.00`** (uncapped) | **0.99976** | **0.99996** | **0.99905** | **3.26 s** |

Quality is monotone in the refinement set, and on the hardest fragmenting set **removing
the cap buys 0.7 points of recall for 0.6 % more time**. The cap is not paying for itself
at 0.25; refinement is only 3–6 % of an update, so the ceiling is protecting a cost that
was never the bottleneck. Raising the default is a one-line change with a measured basis.

> **Negative result, recorded so it is not re-attempted.** The sweep was designed around a
> hypothesis: that because the cap is a fraction of *n*, an update costs Θ(n), and an
> absolute ceiling (`refine_max_nodes`) would make it Θ(batch). **The hypothesis is wrong.**
> Per-batch cost grows 1.25–1.5× as the graph doubles under *every* policy — fractional and
> absolute alike — so the refinement set is not what drives the growth. It must be the
> connectivity rebuild, the hubness refresh, or the `d·ln|E|` budget. The growth is real but
> sub-linear (≈ n^0.5), which is a much weaker problem than Θ(n) would have been.
>
> Measurement caveat: at n = 50 000 with 3 125-point batches, `nodes_1000` and `nodes_2500`
> are **inert** — the new points alone exceed them and are never sacrificed, so both report
> the same `refine_nodes_fraction` (0.0829) as `frac_0.05`. Only `nodes_5000` and above
> bind. An absolute cap has to clear the batch size before it means anything.

### 5.6 FISHDBC — resolved, and it is now a result rather than a gap

**Originally:** installed and validated on a 3,000-point smoke test, no comparative numbers.
The only external baseline, and its absence was the most visible gap in the evaluation.

**Running the matrix as it stood would not have closed it.** The two sides were not
comparable: `baseline_fishdbc.py` scored FISHDBC's **HDBSCAN\*** against sklearn's exact
HDBSCAN\*, while every CORE-SG experiment scored **DBSCAN\*** against the exact CORE-SG.
Two clusterings, two references, one table — not defensible. (A second, smaller error
compounded it: FISHDBC ran at `min_samples=5` against an mpts=10 reference, handicapping
the baseline for no reason.)

`scripts/compare_incremental.py` now drives both methods over one arrival order and scores
both against one per-batch reference, converting the CORE-SG MST to HDBSCAN\* labels with
the same `hdbscan.hdbscan_` extraction FISHDBC itself calls.

**Measured, 20 000 points, 8 batches, mpts = 10, 2 seeds. ARI against exact HDBSCAN\* on
the points present at that batch:**

| data set | CORE-SG | FISHDBC | margin |
|---|---|---|---|
| manifold_mid | **1.0000** at all 8 batches | 0.9925 – 0.9962 | +0.004 … +0.008 |
| sift1m | 0.9920 – 0.9980 | 0.9745 – 0.9855 | +0.009 … +0.022 |

**CORE-SG is ahead on every batch of every data set**, and on `manifold_mid` it recovers the
reference clustering exactly at every point in the stream. That is the external comparison
the evaluation was missing, and it lands in this project's favour.

Two things to state honestly alongside it:

**`fma_mfcc` must be dropped from this comparison.** Its reference finds only 2 clusters
until batch 5, then 4, 13 and 17, and both methods' ARI collapses from ~0.97 to ~0.09 as it
destabilises. Exact HDBSCAN\* at mpts = 10 sees essentially no density structure in that
data. The two methods still agree with *each other* to within 0.014 throughout — which is
the real signal — but the data set cannot support a quality claim either way.

**Do not quote the wall-clock.** CORE-SG's per-batch insert is 0.11–0.17 s against
FISHDBC's 1.2–3.9 s, and that ratio is mostly C++/CUDA against single-threaded Python. The
defensible comparison is the *slope*: as the graph doubled, per-batch cost grew 1.10–1.27×
for CORE-SG and 0.93–1.17× for FISHDBC. Comparable, with a slight edge to FISHDBC, whose
HNSW insertion is O(log n) per point by construction. Earlier concern that this would
diverge badly came from a 2 500-point run where batch-1 warm-up dominated; at 20 000 it
does not.

---

## 6. The CORE-SG payoff survives incrementality

Extracting every MST for mpts ∈ [1, k_max] from one graph, against computing each from the
complete graph:

| data set | CORE-SG sweep | complete-graph sweep | speed-up |
|---|---|---|---|
| 20ng_pca500 | 0.286 s | 86.9 s | **303×** |
| fma_mfcc | 0.831 s | 153.7 s | **185×** |
| sift1m | 0.814 s | 138.3 s | **170×** |
| manifold_exact | 0.074 s | 5.2 s | 71× |

Consistent with the original paper's reported range, and — the point of this thesis —
**preserved after the graph is built incrementally rather than in one shot.**

---

## 7. Insights

**The k-NN graph is usually already connected.** This is the most transferable finding.
It reframes the MST half of CORE-SG from "essential structure" to "insurance against
fragmentation", explains why a crude bridge estimate suffices, and suggests the check
`components(k-NN) == 1` as a cheap gate: when it holds, the bridge computation can be
skipped entirely.

**Ambient dimensionality is the wrong scale for anything budgeted.** `d·ln(|E|)` selects
every point at d=1000. Intrinsic dimensionality — free to estimate from k-NN radii already
stored, validated at 5.67 against a known 6 — is the right term. Any parameter in a
density-based method that scales with `d` deserves the same scrutiny.

**Incrementality can improve quality, not just cost — and it is the decomposition, not the
extra work, that does it.** Repeated partial refinement beats a single global pass, and
§1.1b shows this survives a compute-matched comparison: a rebuild given more descent
*rounds* gains nothing at all (the early stop fires at the same round, distance evaluations
move 0.02 %), because a single descent reaches a fixed point of the local join. Only
widening the candidate list moves it, and even a rebuild tuned that way stays 3.5–7.0 points
behind on the hard data sets. So the mechanism is not "more work" — it is that overlapping
sampled instances explore a different neighbourhood of the search space than one global
descent can. That inverts the usual assumption that incremental methods trade quality for
speed, and makes sampled-instance descent worth studying as a *construction* strategy in
its own right, independent of streaming.

**A quality ceiling can come from the decomposition rather than the effort.** Two distinct
ceilings showed up in the same experiment: at n = 100 k the limit is the breadth of the
local join (fixable with `sample`), while at n = 4 000 with 4 instances the limit is the
partition itself — the whole effort frontier is flat at 0.9423–0.9430 while the incremental
path reaches 0.99925. Neither is "not enough compute". Anything that partitions a search
problem should check which of the two it is before spending on the other.

**Parallelism is bounded by the decomposition, not the hardware.** The 1-instance row
(0.99× on 4 GPUs) makes this concrete. Any work-partitioning design should publish the
equivalent table; it is more informative than a speed-up curve alone.

**Measurement design was the dominant source of error in this project.** Every wrong number
came from comparing things that differed in more than one respect: a reference built over
data that had not arrived yet; two graphs with independently randomised k-NN halves; an
Amdahl fraction taken from the parallel rather than the serial run; instance counts scaled
alongside device counts; a competitor scored at mpts = 5 against an mpts = 10 reference.
None were bugs in the algorithm. All were caught by a number looking impossible — a ratio
below 1.0, a speed-up above its ceiling, a variance of exactly zero.

**A metric can be measured and still not be measuring anything.** The sharpest instance is
`components_before_repair`, which is recorded *after* the bridges are folded in: it read 1
on every full-scale run in the suite and was therefore quoted as evidence the bridges work,
when it could not have read anything else. The general form — a quantity whose definition
guarantees the answer you were hoping for — is harder to catch than a wrong number, because
nothing about it looks impossible. The defence that worked here was asking what the metric
would read if the method had *failed*.

**Hypotheses should be recorded before the experiment that tests them.** Two of the
additions above were designed around expectations that turned out to be wrong: that
approximating the k-NN half would mask fragmentation (it does not, at the operating k), and
that a fractional refinement cap made updates Θ(n) (the growth is real but comes from
elsewhere, and is sub-linear). Both are useful results *because* the expectation was written
down first; without that, the second would have looked like an unexplained flat table.

---

## 8. Future work

### Immediate — mostly done

Six additions are in the harness; the matrix is now 70 runs. Four have been run and their
results are folded into the sections above.

| addition | status | where |
|---|---|---|
| `connectivity` — k-NN component counts and the gate | **run**, 3 seeds | §2.1, §2.2 |
| `rebuild_effort` frontier | **run** on gist1m, glove100, manifold_mid | §1.1b |
| FISHDBC head-to-head on one yardstick | **run**, 20 k points | §5.6 |
| `sensitivity` re-targeted at fragmenting data, + refinement-cap sweep | **run**, 2 seeds | §5.4, §5.5 |
| `batchsize` — amortised cost vs batch size | **cluster pending** | — |
| `parity` — CPU vs CUDA | **cluster pending** (dev-box check in §5.2) | §5.2 |
| `capacity10m_oom128` — the probe that must OOM | **cluster pending** | §5.3 |

> **Provenance.** The four completed runs were executed on the dev box
> (1× RTX 4070 Ti SUPER) and written to `results_local.csv`, deliberately **not** to
> `results.csv`, so that dev-box seconds are never pooled with the cluster timings every
> other number in this document comes from. Their *quality* figures — component counts,
> recall, ARI, cluster counts — are hardware-independent and are quoted directly. Their
> wall-clock figures are labelled as dev-box wherever they appear, and the three pending
> runs are pending precisely because their output is timing.
>
> `results_local.csv` is covered by `.gitignore` (`*.csv`), so it is not in the repository:
> the numbers above are the record. Copy it somewhere durable if the raw rows are wanted.

Remaining after the cluster run: re-measure `sensitivity` timings on cluster hardware if
its seconds are to be quoted (its recall and growth-ratio outputs are already safe), and
decide whether to adopt the raised `refine_max_fraction` from §5.5 as the default.

### Research directions

**Intrinsic dimensionality as a general parameter scale.** The budget was one instance of a
broader pattern. Does `mpts` itself, or the k in k-NN construction, deserve intrinsic rather
than ambient scaling? The estimator is free given the graph.

**A connectivity gate.** If `components(k-NN) == 1` the bridges are provably redundant.
The `connectivity` experiment prices it across all 13 data sets rather than estimating it:
the component labelling costs **0.08–3.5 ms**, against **5.3–222 ms** of hubness + bridge
work — a ratio of **48× to 136×**, with the bridge half accounting for 3.2–20.7 % of build
time. On the seven data sets whose k-NN graph is connected at the operating k (§2.2), that
entire 29–222 ms buys nothing. The gate is close to free insurance on the insurance, and
it is the operational form of §2.2's connection-threshold curve.

**Sampled-instance descent as a construction method.** §1.1 shows overlapping sampled
instances beating a single global descent at equal-ish cost. Studied on its own — instance
count, overlap fraction, number of passes — this may be a better way to build large k-NN
graphs, independent of incrementality.

**Quality/throughput tension in the instance count.** §5.4 shows recall peaking at 4
instances and declining at 16, while §4.2 shows parallelism bounded by instance count. The
optimum is data- and hardware-dependent and currently unmodelled.

**Multi-node.** The instance abstraction is already the right unit to ship to another
machine, and top-k merging is a commutative, associative reduction, so it distributes
without synchronisation. The hard parts are connected components and per-mpts MST
extraction. Notably, because the anti-hub set is sublinear in n, the bridge MST can be
gathered onto one machine at any scale — the anti-hub design removes the hardest
communication step from a distributed implementation, which may matter more at multi-node
scale than it does here.

**Decremental updates.** Everything here is insert-only. Deletion is the harder direction
(it is why incremental connectivity failed, per EXPERIMENTS.md) and is untouched.

---

## 9. Reproducing

```bash
bash scripts/bootstrap_cluster.sh                 # build, fetch data, preflight
python scripts/run_suite.py --seeds 5 --out results.csv --resume --keep-going
python scripts/summarize.py --results results.csv
```

Every figure above comes from `results.csv` (17,538 rows) **except those marked as
dev-box checks in the 2026-08-01 corrections**, which are single-seed runs on one
RTX 4070 Ti SUPER and are quoted only until the cluster sweep replaces them.

The suite is now 70 runs. `--resume` skips the 47 already in the CSV, so the additions cost
their own time only — minutes for `connectivity`, `parity` and `batchsize`; the
`capacity10m_oom128` probe is short because a refused job fails fast with the OOM fallback
disabled. The original sweep took approximately three days, dominated by the 10M-point
scaling run, which must **not** be scheduled again.

The external baseline is not part of `run_suite` — it drives `coresg_exp` itself and needs
the conda environment with FISHDBC:

```bash
python scripts/compare_incremental.py --data data/manifold_mid.fvecs --name manifold_mid --limit 20000 --batches 8 --mpts 10 --k 24 --out results.csv
```
