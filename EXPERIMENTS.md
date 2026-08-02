# Experiment suite

Reference for the thesis evaluation: what is measured, on what data, against what
baseline, and **why each of those choices was made**. Several of the decisions here look
arbitrary until you know what went wrong first, so the reasoning is recorded alongside the
setup. When something changes in the routine, change it here too.

Last updated: 2026-08-01.

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

> **Trap, and it made the comparison unjoinable.** `baseline_fishdbc.py` scores FISHDBC's
> **HDBSCAN\*** clustering (EOM extraction at a `min_cluster_size`) against sklearn's exact
> HDBSCAN\*. Every `coresg_exp` experiment scores a **DBSCAN\*** clustering (a flat cut of
> the MST at one threshold) against the exact CORE-SG. Two different clusterings measured
> against two different references do not go in one table, however similar the numbers
> look. Running the FISHDBC matrix as it stood would have produced two columns that could
> not honestly be compared.

> **Second trap, and it cut against the baseline again.** FISHDBC's constructor defaults to
> `min_samples=5`; sklearn's `HDBSCAN` defaults `min_samples` to `min_cluster_size`. Left
> alone, the baseline ran FISHDBC at mpts = 5 and scored it against an mpts = 10 reference.
> All three sides are now pinned to one `--mpts`.

**`scripts/compare_incremental.py`** is the head-to-head. It fixes one arrival order and
hands it to both methods, so at every batch they have been shown exactly the same points;
runs `coresg_exp --experiment stream`, which exports the CORE-SG MST after each batch;
converts that MST to HDBSCAN\* labels with `hdbscan.hdbscan_`'s
`label → condense_tree → compute_stability → get_clusters` — **literally the code FISHDBC's
own `cluster()` calls on its MST**; and scores both against one per-batch reference. The
only thing that differs between the two rows is the MST each method produced, which is the
thing under test. Rows join on `(dataset, seed, batch)`.

Verified end to end (`manifold_mid`, 3 000 points, 4 batches, mpts = 10, dev box):

| batch | n | CORE-SG ARI | FISHDBC ARI | clusters found (both) | reference clusters |
|---|---|---|---|---|---|
| 1 | 1 875 | 0.5870 | 0.5985 | 99 | 120 |
| 2 | 2 250 | 0.8501 | 0.8608 | 134 | 143 |
| 3 | 2 625 | **1.0000** | **1.0000** | 149 | 149 |
| 4 | 3 000 | **1.0000** | **1.0000** | 149 | 150 |

The two methods track each other to within 0.012 ARI and both reach 1.0000 by batch 3. The
early ramp is the legitimate one already documented above — the *reference itself* only
finds 120 of the eventual 150 clusters at batch 1, because clusters below
`min_cluster_size` are not yet detectable — and the `reference_clusters` column carries it.
Note that both methods find the *same* 99 and 134 clusters where they both differ from the
reference, i.e. they agree with each other more closely than either agrees with it.

> **A forest cannot be extracted.** `hdbscan_labels_from_mst` refuses unless the MST has
> exactly n−1 edges. Extraction on a disconnected graph silently produces a malformed
> dendrogram rather than an error, so this is checked rather than trusted; it is also a
> live end-to-end test of `ensure_connected`.

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

**The equal-work control (`config = rebuild_effort`).** The incremental path performs
strictly *more* total descent than one rebuild: each batch re-visits sampled base points
inside its instances, so after eight batches the base has been refined eight times while
the rebuild descends once. A plain rebuild is therefore a cheaper baseline, not an equal
one, and "incremental wins on recall" against it invites the obvious reply that it simply
did more work. That reply has to be answered in the write-up, not left for a reviewer.

Rather than hand-tune one matched configuration, the rebuild's descent effort is swept and
the whole **(cost, recall) frontier** is emitted, so the incremental point can be read
against the rebuild curve at equal cost. `refine_rounds` and `distance_evals` are recorded
alongside, because a flat frontier has two different meanings and only those columns
separate them. Measured on `manifold_mid` (n = 100 k, 1 seed, dev box):

| arm | seconds | recall |
|---|---|---|
| rebuild, baseline | 2.13 | 0.99615 |
| rebuild, `iters_x2` | 1.83 | 0.99600 |
| rebuild, `iters_x4` | 1.68 | 0.99606 |
| rebuild, `sample_x2` | 2.10 | **0.99840** |
| rebuild, `iters_x2_sample_x2` | 2.31 | **0.99853** |
| **incremental** | 5.74 | **0.99942** |

Two things follow, and both belong in the paper. **More iterations buy nothing** — the
descent has already converged, `delta` early-stops at the same 4 rounds and `distance_evals`
moves 0.02 %. **More sampling does help**, and nearly for free in wall-clock. So the
honest framing is not "incremental beats a rebuild" but: *a better-tuned rebuild closes
most of the gap, and the residue (0.99942 vs 0.99853) costs 2.5×.* Reporting the naive
comparison alone would have overstated the result.

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

### `connectivity` — how connected is the k-NN half, and at what k
Component count of the k-NN graph **alone**, at full scale, plus a sweep over k.

> **Trap, and it invalidated a headline number.** `components_before_repair` is filled in
> by `CoreSG::repair_connectivity` *after* the bridges have been folded into the component
> labelling. On every full-scale run in this suite it therefore reads **1 by construction** —
> it says the bridges connected the graph, not how far apart the graph was. The only arm
> that measures the k-NN half alone lived in `bridges`, which caps n at the O(n²)
> reference. So no full-scale run had ever measured the thing the bridges exist to fix, and
> "manifold_wide fragments into 273 components" came from the data-set design table rather
> than from `results.csv` (the string `273` does not appear in it).
>
> Measured here directly, `manifold_wide` at n=100 000, k=24 has **257** components in its
> k-NN half. The claim survives; it is now a measurement.

Three arms, and the second pair is the one that could have bitten:

| arm | n | k-NN half | answers |
|---|---|---|---|
| `approx_full` | full | NN-Descent | what the bridges actually face in production |
| `approx_matched` | `exact_limit` | NN-Descent | the same, at the size the exact arm can reach |
| `exact_matched` | `exact_limit` | brute force | does approximating the half change connectivity? |

Every fragmenting component count quoted elsewhere in this project was measured with an
**exact** k-NN half. It was reasonable to worry that NN-Descent's residual errors would be
long cross-cluster edges and would quietly *connect* a graph the exact k-NN leaves in
pieces, which would mean the bridge evidence was measured in an easier regime than the one
it ships in. **Measured: the opposite.** On `manifold_c24` the exact half gives 24
components and the approximate half gives 26–27 — approximation fragments *more*, because
NN-Descent's misses remove edges while its errors stay local (it finds near neighbours, not
distant ones). The approximate regime is therefore the harder one, and the evidence is
conservative rather than flattered. Worth one sentence in the write-up; it is the first
thing a careful reader will wonder about.

The k sweep turns the project's most transferable observation — real k-NN graphs are
usually already connected — from a single-k assertion into a curve: the k at which each
data set connects.

It also prices the **connectivity gate**. `components_seconds` is the union-find pass that
derives the labelling, i.e. what the check costs; the `gate` arm's `bridge_seconds` plus
`hubness_seconds` is what the check would let you skip. On `manifold_wide`: **1.4 ms
against 115 ms**, the latter being 9.3 % of build time.

> **Second trap, smaller.** Do not quote `repair_seconds` as the cost of the check. It is
> only the fold-in of the bridges on top of an *already valid* labelling — single-digit
> microseconds — and would overstate the gate's value by four orders of magnitude.

### `batchsize` — amortised cost against batch size
Batch size as a fraction of n, against total cost, per-point cost and final recall, with a
single rebuild as the line it is read against. The suite previously contained two batch
sizes in total (6 250 in `incremental`, 900 in `drift`), which is not a curve.

It matters because an update has a fixed cost that does not shrink with the batch — the
hubness refresh, the bridge MST and the connectivity pass all run once per insert
regardless of how many points arrived — so below some batch size the structure is cheaper
to rebuild than to update. The phase totals are emitted separately from wall-clock
precisely so the fixed part can be separated from the per-point part.

### `parity` — do the CPU and CUDA backends agree?
Recall of each backend against one brute-force reference, plus the agreement of what they
produce downstream (MST weight profile, DBSCAN\* ARI between the two). `jobs_on_device` is
emitted so a silent host demotion cannot make the arms agree perfectly and mean nothing.

> **Read the weight ratio as symmetric here.** Everywhere else in this suite a ratio below
> 1.0 signals the measurement error described under `sensitivity`. In this experiment the
> two arms *are* two separately built `CoreSG` objects on purpose — the backend is the
> thing under test and everything else is held fixed — so a value either side of 1.0 is
> legitimate and means only that one arm found a slightly heavier tree.

### `stream` — the export that makes the FISHDBC comparison joinable
Not part of `all`; it only does anything when `scripts/compare_incremental.py` supplies an
arrival order and an output directory. See §5.

### `sensitivity` — robustness
k_max, instance count, anti-hub multiplier, per-component quota, NN-Descent sample size,
`refine_max_fraction`, plus a `reference / knn_only` row.

> **Trap.** This experiment does **not** report MST quality ratios. Two separately built
> `CoreSG` objects get two different randomised k-NN halves, and that variation exceeds the
> bridge effect — it shows up as a weight ratio *below* 1.0, which is impossible for a real
> approximation. Bridge quality belongs in `bridges`, where the k-NN half is held exact.

> **Trap, and it was previously mis-diagnosed.** `antihub_multiplier` and
> `min_per_component` came back flat across the sweep. That was put down to the experiment
> measuring k-NN recall, which those parameters cannot affect. It was not the metric: the
> experiment already emitted `components_before_repair`, and that was **1.000 for every
> configuration** including `min_per_component = 0`. The cause was the *data* — the sweep
> ran on `manifold_mid`, `fma_mfcc` and `sift1m`, none of which fragment at n = 50 000, so
> the coverage knobs had nothing to act on. Re-targeting the metric alone would have
> produced a second flat table. Fixed two ways: fragmenting data sets are in the matrix,
> and the experiment now emits a `reference / knn_only` row so a column of 1s can be read
> as "the bridges took 24 components to 1" rather than "nothing happened".

**The refinement cap policy (`config = refine_cap`)** is the one place this project has a
measured structural disadvantage, so it gets its own sweep with two families:

| family | rule | update cost |
|---|---|---|
| `frac_*` | cap the refinement set at a fraction of **n** (the shipped default, 0.25) | **Θ(n)** — grows as the graph grows, however few points arrived |
| `nodes_*` | `refine_max_nodes`, a fixed ceiling (disabled by default) | **Θ(batch)** |

That distinction is why it matters. In the head-to-head (§5) FISHDBC's per-batch cost is
flat while this project's rises with n, and the fractional cap is the reason: HNSW
insertion is O(log n) per point, whereas a set sized as a fraction of n is not. If an
absolute cap holds quality at a flat slope, adopting it is a one-line default change with
a real result behind it — so per-batch rows are emitted alongside the totals, and
`summarize.py` reports the growth ratio (first batch to last, over a run in which the
graph doubles from n/2 to n). **~1.0 means Θ(batch); ~2.0 means Θ(n).**

> **Trap.** This cannot be swept with the same helper as the other parameters. That helper
> inserts the whole second half in one batch, so the new points alone are 50 % of the graph
> — and since the refinement set is built new points → first ring → second ring, and the cap
> drops the *outermost* nodes first while never sacrificing the new points, every fractional
> setting below 0.5 collapses to exactly 0.5 and the knob appears inert. The cap can only
> bind on the rings when the batch is small, so this sweep runs the realistic shape: many
> small batches.

> **Sizing trap for the absolute arms.** `nodes_*` is absolute, so what it *means* depends
> on the data set. At n = 4 000 a 5 000-node ceiling is simply "no cap", and every arm
> collapses to the same measurement — the dev-box check on `manifold_c24` gave a growth
> ratio of 0.74–0.81 for **every** policy, fractional and absolute alike, because at that
> size the first batch's warm-up dominates and no cap binds. The discriminating measurement
> needs the n = 50 000 data sets, where 2 500 nodes is 5 % of the graph. Do not read the
> small-n run as evidence either way.

Fractional arms on `manifold_c24` behave monotonically once the shape is right — 0.05 →
recall 0.99895, 0.25 → 0.99938, 1.0 → 1.00000 — while total insert time stays flat at
0.44–0.47 s, i.e. **relaxing the cap is nearly free** at that size.

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

# the cheap additions -- minutes, and they close the gaps in §10
python scripts/run_suite.py --only connectivity --out results.csv
python scripts/run_suite.py --only batchsize,parity,sensitivity --out results.csv

# the external baseline, both sides on one yardstick (not part of run_suite:
# it drives coresg_exp itself and needs the conda env with FISHDBC)
python scripts/compare_incremental.py --data data/manifold_mid.fvecs \
    --name manifold_mid --limit 20000 --batches 8 --mpts 10 --k 24 --out results.csv
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

**GPU strong scaling** (instance count fixed across device counts), 16 instances:

| data set | f (1 GPU) | Amdahl ceiling @4 | measured @4 | efficiency |
|---|---|---|---|---|
| capacity10m (10M) | 0.941 | 3.40x | **3.23x** | 0.81 |
| gist1m (100k) | 0.921 | 3.23x | **2.68x** | 0.67 |
| sift1m (200k) | 0.869 | 2.87x | **2.53x** | 0.63 |
| manifold_mid (100k) | 0.840 | 2.70x | **2.50x** | 0.63 |

**Scaling improves with problem size** — f rises 0.840 → 0.941 and efficiency 0.63 → 0.81
as n goes from 100k to 10M, because larger instances amortise the fixed per-job transfer.
That trend is the strongest form of the scalability claim, and it is the reason the
expensive capacity10m sweep was worth its cost after all.

> **`f` must be measured on the serial baseline.** It *falls* as devices are added — the
> instance phase is exactly the part being parallelised, so it shrinks — dropping from
> 0.941 to 0.798 between 1 and 4 GPUs on capacity10m. Averaging f across device counts
> understates the ceiling enough that every measurement appears to exceed it, which is
> what an earlier version of this file reported. Take f from the 1-GPU run.

**Per-device memory is exactly flat as devices are added** — 1275 MiB on 1, 2, 3 and 4
GPUs for capacity10m — and falls with instance count (2.01 / 1.50 / 1.25 GiB at 4 / 8 / 16
instances). That invariance is the property that makes the design scale, and it is now
measured rather than asserted.

**Budget: ambient is not a sublinear estimate.** It selects **100 % of n** on 20ng_pca500
and gist1m, and 29–40 % everywhere else. Intrinsic uses 1.4–10 %, runs 5–30x faster, and
costs at most 0.6 % in MST weight (0.1 % at x4).

### Claim C, after adding the fragmenting configurations

Bridges are load-bearing only where the k-NN graph actually breaks apart. Five data sets
now do:

| data set | components (k-NN alone) | worst weight ratio | ARI | bridge speed-up |
|---|---|---|---|---|
| manifold_c32 | 32 | 1.00022 | 1.0000 | 18.2x |
| manifold_c24 | 24 | 1.00024 | 1.0000 | 36.0x |
| manifold_exact | 16 | 1.00000 | 1.0000 | 12.1x |
| manifold_c8 | 8 | 1.00001 | 1.0000 | 19.0x |
| sift1m | 2 | 1.00000 | 1.0000 | 10.5x |

**Worst case across all five: weight ratio 1.00024, ARI exactly 1.0000, 10.5-36x faster
than the exact MST.** The six remaining data sets have connected k-NN graphs and are
reported separately as cost-only.

At full scale `manifold_wide` fragments into **273** components and the anti-hub bridges
alone reduce that to **1** — the per-component quota doing exactly what it was added for.
Its incremental run also beats a rebuild (recall 0.9971 vs 0.9896, ARI 0.9946) and its
drift over 25 batches is **-0.0034**, i.e. flat.

### The budget on `20ng_tfidf` (d=1000), the extreme case

| rule | effective d | selected | fraction of n | bridge time |
|---|---|---|---|---|
| ambient | 1000 | 4000 | **100 %** | 0.1245 s |
| ambient_capped | 1000 | 2000 | 50 % | 0.0316 s |
| **intrinsic** | **41.6** | 472 | 11.8 % | **0.0054 s** |
| intrinsic_x4 | 41.6 | 1889 | 47 % | 0.0392 s |

Ambient selects *every point*: the "approximation" is the exact MST. Intrinsic is **23x
faster** and uses an eighth of the budget.

### What this run exposed

**Only one data set exercised the bridges.** `components(knn_only)` was 1 on 20ng_pca500,
fma_chroma, fma_mfcc, gist1m and glove100; 2 on sift1m; **16 on manifold_exact**. Five of
seven real sets are connected, so their weight ratio of 1.00000 is vacuous — nothing was
approximated. Claim C rested on a single data set. Fixed by adding `manifold_c8` /
`manifold_c24` / `manifold_c32` (7 / 24 / 32 components), which cost ~2 s each to run.

**The capacity experiment never tested its own premise.** Instance counts were
{4, 8, 16}, all of which fit (2.01 / 1.50 / 1.25 GiB). `instances=1` — the 7.90 GiB
configuration that does not fit an 8 GB card — was unreachable, so the capacity claim has
no supporting measurement. Added a 1-seed probe at instances {1,2,4} with the OOM fallback
disabled, which costs minutes rather than the ~2 days the full sweep took.

(The sweep itself was *not* wasted, contrary to a first reading of the partial data: at
10M points it produced both the best speed-up in the suite and the size trend above.)

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
| Parallel fraction f (1-GPU baseline) | 0.84-0.94 → 4-GPU ceiling 2.70-3.40x |
| Intrinsic dim: SIFT / GIST / GloVe / 20NG | 10.2 / 28.5 / 34.5 / 36.3 |
| Intrinsic budget saving | 5–49× fewer anti-hubs, identical quality on real data |
| Capacity: 1 instance vs 4 | 7.90 GiB (OOM on 8 GB) vs 1.97 GiB |

---

## 10. Open gaps

1. **The head-to-head against FISHDBC is verified end to end but not run across the
   matrix** — only a 3 000-point run on `manifold_mid` (§5). It is single-threaded Python
   *and* the per-batch reference is O(n²), so keep `--limit` modest (20 k or below).
2. **The six new/changed experiments have been smoke-tested on the dev box, not swept.**
   `connectivity`, `batchsize`, `parity`, the `rebuild_effort` frontier, the re-targeted
   `sensitivity` sweep and the `capacity10m_oom128` probe are all in the matrix (70 runs)
   and none of their numbers are in `results.csv` yet. Every figure quoted for them in this
   file is a single-seed dev-box measurement, recorded so a cluster run can be checked
   against it — not a result.
3. **`capacity10m` at k=128 is the run that decides the capacity claim.** At k=64 the
   `insert` path builds base + batch = 6.25 M points per instance, needing 4.94 GiB, which
   fits an 8 GB card comfortably — so the intended configuration was never constructed and
   no OOM occurred. k=128 takes the same configuration to ~8.3 GiB. Until it runs, the
   claim is arithmetic plus the memory-scaling table.
4. **The budget rule is decided on evidence but not yet adopted as the default.** See §12.
   `budget_dimension` still defaults to `Ambient`, i.e. the specified rule.
5. **Anti-hub count sensitivity is data-dependent** — flat on low-d blobs, monotone on
   manifolds, irrelevant on connected real data. Do not generalise from one shape.
6. **Multi-GPU numbers come from the cluster only.** On one device, more instances is
   *slower* (they serialise); the instance count is a scheduling knob for multi-GPU, not a
   single-device speed knob.

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
