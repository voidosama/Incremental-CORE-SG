# Results and analysis

Full experimental sweep, 4× RTX 2080 SUPER (sm_75, 8 GB), CUDA 12.9.
**17,538 measurements · 48 run/dataset pairs · 14 data sets · up to 5 seeds · zero failures.**
(11 of the 14 appear in the bridges experiment, which is where the counts below are drawn
from; `capacity10m` carries two labels because its scaling sweep and its capacity probe are
separate runs.)

Companion documents: [EXPERIMENTS.md](EXPERIMENTS.md) for *why* each experiment is designed
the way it is, [README.md](README.md) for the implementation.

---

## Executive summary

| claim | verdict | headline evidence |
|---|---|---|
| **A** — incremental updates preserve quality | **Supported, strongly** | Incremental beats a full rebuild on **8 of 8** data sets while costing 2.3–2.9× less than rebuilding per batch |
| **B** — the design scales across GPUs | **Supported** | 2.50–3.23× on 4 GPUs, per-device memory *exactly* invariant with device count |
| **C** — anti-hub bridges replace the MST half | **Supported** | Worst weight ratio **1.00024**, ARI exactly **1.0000**, 10.5–36× faster, across 5 fragmenting data sets |
| CORE-SG payoff survives incrementality | **Supported** | 71–303× for a full mpts sweep |
| capacity: config that needs > 1 GPU | **Not demonstrated** | see §5.3 — the intended configuration was never actually built |

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

> **The control this table needs (2026-08-01).** "Not an equal-work comparison" is a
> caveat a reviewer will convert into an objection, so the `rebuild_effort` frontier now
> answers it directly: sweep the rebuild's descent effort and read the incremental point
> against the resulting (cost, recall) curve. Dev-box, `manifold_mid`, 1 seed:
>
> | arm | seconds | recall |
> |---|---|---|
> | rebuild, baseline | 2.13 | 0.99615 |
> | rebuild, `iters_x4` | 1.68 | 0.99606 |
> | rebuild, `sample_x2` | 2.10 | 0.99840 |
> | rebuild, `iters_x2_sample_x2` | 2.31 | 0.99853 |
> | **incremental** | 5.74 | **0.99942** |
>
> More *iterations* buy nothing — the descent has converged, the early stop fires at the
> same 4 rounds and distance evaluations move 0.02 %. More *sampling* does help, and is
> nearly free in wall-clock. So the defensible claim is narrower than the table above:
> **a better-tuned rebuild closes most of the gap, and the residue costs 2.5×.** The
> write-up should lead with that rather than with the raw margin.

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
> The new `connectivity` experiment measures it directly. Single-seed dev-box check:
> `manifold_wide` at n = 100 000, k = 24 has **257 components** in its k-NN half, so the
> claim stands and the figure was close. It is quoted here as provisional until the cluster
> sweep writes it to `results.csv`.
>
> One thing that check settled and is worth stating in the write-up: every fragmenting
> count in this document was measured with an **exact** k-NN half, and it was reasonable to
> worry that NN-Descent's residual errors would be long cross-cluster edges that quietly
> connect a graph the exact k-NN leaves in pieces — which would mean the bridge evidence
> was gathered in an easier regime than the one that ships. Measured: the opposite. On
> `manifold_c24` the exact half gives 24 components and the approximate half gives 26–27.
> Approximation fragments *more*, because misses remove edges while errors stay local. The
> evidence is conservative, not flattered.

### 2.2 The connected-graph finding is itself a contribution

Every real data set tested — FMA (both variants), 20 Newsgroups (both), SIFT, GIST,
GloVe — has a **connected** k-NN graph at k = 24–32. The MST half of CORE-SG therefore
contributes nothing to them beyond what the k-NN graph already provides.

That is worth reporting rather than hiding. It says the bridge mechanism is *insurance*:
it costs 1–5 % of build time, is invisible when unnecessary, and is the difference between
a spanning tree and a 273-piece forest when it is. It also explains why an approximation
this crude works so well — on most real data the thing being approximated is nearly empty.

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
> component count of the k-NN half alone — so a column of 1s reads as "the bridges took 24
> components to 1" rather than "nothing happened". Verified on `manifold_c24`: the
> reference row reports 24, every swept configuration reports 1.

The instance-count result is worth noting on its own: quality peaks at 4 and *declines*
slightly at 8–16, so more instances is not free. Since instances are also the unit of
parallelism (§4.2), there is a genuine tension between throughput and quality.

### 5.5 The refinement cap is always binding

`refine_nodes_fraction` was **exactly 0.25 on every data set and every batch** — the cap is
saturated everywhere, so the measurement reflects the ceiling rather than the algorithm's
natural appetite, and `refine_max_fraction` is not swept.

### 5.6 FISHDBC was verified but never run across the matrix

Installed and validated on a 3,000-point smoke test (ARI 1.0000 by batch 3). No comparative
numbers exist. This is the only external baseline and its absence is the most visible gap.

*Addressed, not yet swept — and running the matrix as it stood would not have closed it.*
The two sides were not comparable: `baseline_fishdbc.py` scored FISHDBC's **HDBSCAN\***
against sklearn's exact HDBSCAN\*, while every CORE-SG experiment scored **DBSCAN\***
against the exact CORE-SG. Two clusterings, two references, one table — not defensible.
(A second, smaller error compounded it: FISHDBC ran at `min_samples=5` against an mpts=10
reference, handicapping the baseline for no reason.)

`scripts/compare_incremental.py` now drives both methods over one arrival order and scores
both against one per-batch reference, converting the CORE-SG MST to HDBSCAN\* labels with
the same `hdbscan.hdbscan_` extraction FISHDBC itself calls. Dev-box run
(`manifold_mid`, 3 000 points, 4 batches, mpts = 10):

| batch | n | CORE-SG | FISHDBC | reference clusters |
|---|---|---|---|---|
| 1 | 1 875 | 0.5870 | 0.5985 | 120 |
| 2 | 2 250 | 0.8501 | 0.8608 | 143 |
| 3 | 2 625 | **1.0000** | **1.0000** | 149 |
| 4 | 3 000 | **1.0000** | **1.0000** | 150 |

The two track each other to within 0.012 ARI and both reach 1.0000 by batch 3, on the same
yardstick. The early ramp is the reference's own (it finds 120 of 150 clusters at batch 1),
and both methods find the *same* cluster counts where they differ from it.

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

**Incrementality can improve quality, not just cost.** Repeated partial refinement beats a
single global pass. That inverts the usual assumption that incremental methods trade
quality for speed, and is worth investigating as a *construction* strategy in its own
right: descend a large graph in overlapping sampled instances rather than all at once.

**Parallelism is bounded by the decomposition, not the hardware.** The 1-instance row
(0.99× on 4 GPUs) makes this concrete. Any work-partitioning design should publish the
equivalent table; it is more informative than a speed-up curve alone.

**Measurement design was the dominant source of error in this project.** Every wrong number
came from comparing things that differed in more than one respect: a reference built over
data that had not arrived yet; two graphs with independently randomised k-NN halves; an
Amdahl fraction taken from the parallel rather than the serial run; instance counts scaled
alongside device counts. None were bugs in the algorithm. All were caught by a number
looking impossible — a ratio below 1.0, a speed-up above its ceiling, a variance of exactly
zero.

---

## 8. Future work

### Immediate — all implemented, none swept

Six additions are in the harness and smoke-tested on the dev box; the matrix is now 70
runs. What remains is cluster time, not code.

1. **Close the capacity claim.** `capacity10m_oom128` probes k=128, where one instance
   needs ~8.3 GiB and must fail. Converts §5.3 from arithmetic to demonstration.
2. **Run the FISHDBC head-to-head** via `compare_incremental.py` (§5.6), on
   `manifold_mid`, `fma_mfcc` and `sift1m` at `--limit 20000`.
3. **Re-target the anti-hub sensitivity sweep** — at fragmenting *data*, not just at a
   different metric (§5.4).
4. **Backend parity** (§5.2).
5. **The `connectivity` experiment**, which supplies the k-NN component counts §2.1 and
   §5.1 were both missing, and prices the connectivity gate below.
6. **The `rebuild_effort` frontier**, which is the answer to the obvious objection to §1.1
   — see the correction there.

### Research directions

**Intrinsic dimensionality as a general parameter scale.** The budget was one instance of a
broader pattern. Does `mpts` itself, or the k in k-NN construction, deserve intrinsic rather
than ambient scaling? The estimator is free given the graph.

**A connectivity gate.** If `components(k-NN) == 1` the bridges are provably redundant.
The `connectivity` experiment now prices it rather than estimating it: on `manifold_wide`
the component labelling costs **1.4 ms** against **115 ms** of hubness + bridge work, which
is 9.3 % of build time — an ~80× ratio. On the six data sets with a connected k-NN graph
that 115 ms buys nothing at all, so the gate is close to free insurance on the insurance.

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
