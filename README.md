# Incremental, multi-GPU CORE-SG

An implementation of the CORE-SG replacement graph (Neto, Naldi, Campello & Sander,
ICDE 2022) that is **incremental** and runs on **CPU (single- or multi-threaded),
one GPU, or several GPUs**.

The two halves of a CORE-SG are handled separately, as the thesis plan proposes:

| half | original | here |
|---|---|---|
| `k_max`-NNG | one offline pass | **Dynamic NN-Descent** (Wang et al., IEEE TBD 2025), made incremental by sampled instances |
| `MST_kmax` | exact MST over the complete mutual-reachability graph, `O(n² d)` | **anti-hub bridge estimate**, cost independent of `n` |

---

## Building

Requires CMake ≥ 3.21 and a C++17 compiler. CUDA is optional and auto-detected.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release
```

On Windows with multiple Visual Studio installs, point CMake at the one that has the
CUDA MSBuild integration:

```bash
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_GENERATOR_INSTANCE="C:/Program Files/Microsoft Visual Studio/2022/Community"
```

Useful options: `-DCORESG_ENABLE_CUDA=OFF`, `-DCORESG_ENABLE_AVX2=OFF`,
`-DCMAKE_CUDA_ARCHITECTURES="70;80;86;89;90"`.

**`sm_70` is a hard floor.** The local-join kernel holds per-node spin locks inside a
warp, which only makes progress under independent thread scheduling (Volta+). CMake
errors out if you ask for less.

## Running

```bash
coresg_cli devices
coresg_cli build   --data X.fvecs --kmax 32 --out graph.coresg
coresg_cli update  --in graph.coresg --data batch.fvecs --out graph.coresg
coresg_cli extract --in graph.coresg --mpts 2,4,8,16 --out-prefix mst
coresg_cli demo    --n 100000 --dim 16 --batches 4

coresg_bench all     --n 60000 --dim 24 --k 32
coresg_bench norm    --n 6000  --dim 16 --k 24    # reverse-score denominators
coresg_bench repair  --n 20000 --k 24             # when do bridges alone split?
coresg_bench conn    --n 100000 --dim 16 --k 24   # component-finding strategies
```

Data formats: `.fvecs`, `.bvecs`, `.csv`, `.tsv`, raw `.bin` (with `--dim`).

---

## Evaluation

**[RESULTS.md](RESULTS.md)** — the full analysis of the completed sweep: what each claim is
supported by, what is *not* supported, insights, and future work.

**[EXPERIMENTS.md](EXPERIMENTS.md)** — the suite design: data sets, baselines, metrics, and
the reasoning behind each choice, including the traps that produced wrong numbers before
they were caught.

```bash
python scripts/run_suite.py --list
python scripts/run_suite.py --seeds 5 --out results.csv
```

## Architecture

```
Dataset ──► IncrementalKnn ──► KnnGraph ──► HubnessIndex ──► missing edges ──► CoreSG
              │                   │             (observer)         │
              │                   └── core distances ──────────────┤
              ▼                                                    ▼
          Backend (CPU | CUDA)  ◄── JobScheduler            mst(mpts) for any mpts ≤ k_max
```

| file | role |
|---|---|
| `include/coresg/knn_graph.hpp` | sorted top-k lists; `try_insert` **reports the evicted neighbour** |
| `include/coresg/nndescent.hpp` | Dynamic NN-Descent, restricted-active-set variant |
| `include/coresg/incremental_knn.hpp` | sampled instances, merge, restricted refinement |
| `include/coresg/hubness.hpp` | reverse degrees, reverse score, anti-hub selection |
| `include/coresg/connectivity.hpp` | components: parallel rebuild + incremental certificate |
| `include/coresg/mst.hpp` | Kruskal, parallel Borůvka, dense/virtual Prim |
| `include/coresg/coresg.hpp` | assembly, per-`mpts` MST extraction, exact reference |
| `include/coresg/backend.hpp` | device abstraction + job scheduler |
| `cuda/` | device kernels (`.cu`), behind a POD-only API |

### Why the design is multi-GPU-shaped

The unit of work is a **self-contained sub-problem**: a contiguous block of vectors plus
a warm-started local graph, with no reference to global state. A device only ever holds
`|sample| + |new points|` vectors, so instances can be queued onto whatever device is
free. `JobScheduler` owns one worker thread per device and pins slot == thread, so a GPU
stays bound to one thread and keeps its allocations and streams alive across jobs.

Nothing in the k-NN path needs a cross-device reduction. The only global steps are the
merge (host-side, lock-protected top-k insertion) and the anti-hub MST, whose distance
matrix is split row-wise across devices by `JobScheduler::mrd_matrix`.

### Dynamic NN-Descent

Follows Algorithm 1 of the paper: sampling happens on the **live** graph `G` (no static
snapshot `H`), the reverse graph `R` is **incomplete and bounded** to `smpN` entries per
flavour, and there is **no barrier inside a round**. Threads touch disjoint nodes and
synchronise only through the k-NN graph's per-node spin locks.

Two implementation notes:

* Unfilled neighbour slots hold `+inf`, so the "is this candidate worth a lock" test is a
  single float comparison against the last slot and never reads the concurrently-mutated
  degree counter. This removes the only data race from the hot path.
* Candidate rejection happens in **rank space** (squared L2), so the square root is only
  paid for pairs that will actually be inserted. This matters a lot in low dimensions.

The GPU kernel is the whole algorithm in one launch, one block per node, with the local
join accumulated through shared-memory tiles (Fig. 2 of the paper). Both operand tiles are
stored transposed `[d][i]` so the 16 lanes covering a tile row hit 16 distinct banks.
The descent runs entirely in rank space; distances are squared on upload and rooted on
download, so a graph can move between the CPU and GPU backends mid-pipeline.

### Incremental k-NN

Per batch: `instances` random unique samples of the base points, each merged with the
whole batch, each warm-started from the edges its sampled points already have, each an
independent NN-Descent job. Then all neighbourhoods are merged back (repeated insertion
into sorted top-k lists *is* "keep the best k"), then one restricted NN-Descent over the
new points and their neighbourhood.

Sampling modes: `Partition` (one permutation cut into disjoint blocks — full coverage,
default) or `Independent` (each instance draws its own sample).

**The refinement set is capped** (`refine_max_fraction`, default 0.25). With `k=20` and a
second ring, an uncapped set reaches `k²` nodes per new point and saturates a small graph
— measured at 97.8% of the graph before the cap, 25.0% after, with new-point recall
moving only 0.9999 → 0.9996. The set is built new points → ring 1 → ring 2, so the cap
drops the outermost nodes first and never sacrifices a new point.

### Hubness and the missing edges

```
reverse_degree(v) = |{u : v ∈ NN(u)}|
reverse_score(v)  = reverse_degree(v) / denom(rd of NN(v))
```

with the specified degenerate rules (`rd>0, sum=0 → k`; `rd=0, sum>0 → 0`; both 0 → 1).
Because the denominator is a *local* statistic, the score is invariant to the absolute
density of the region — a border point of a dense cluster scores like a border point of a
sparse one.

`denom` is selectable (`HubnessParams::normalization`, `--normalization`):
`arithmetic` (the specified mean, default), `geometric` (`exp(mean(ln(rd+1))) - 1`), or
`median`. See *Choosing the denominator* below for what that changes — and what it
doesn't.

`HubnessIndex` attaches to the k-NN graph as a `GraphObserver`. Every inserted or evicted
directed edge updates a reverse degree in O(1) and marks the affected nodes, so a refresh
only rescores the dirty region: the nodes whose own degree moved, the nodes whose
neighbour list moved, and the **reverse** neighbours of the first set. With
`store_reverse_lists` (default on) that propagation is exact; with it off, the cheaper
forward approximation is used instead. `test_hubness` asserts the incremental path
produces bit-identical scores *and* an identical selection to a full rescore.

The `d · ln(|E|)` lowest-scoring nodes are selected — plus, by default, at least
`min_per_component` from every k-NN component, see *The fix: a per-component quota* below.
An MST is computed over the selected set in mutual-reachability space, and those edges
become the bridges. They are **stored with their base distance**, not their `mrd_kmax`
weight, so `edges(mpts)` reweights them exactly the way it reweights the k-NN half.

---

## Measured results

RTX 4070 Ti SUPER (sm_89), CUDA 12.6, MSVC 19.41. Gaussian blobs with per-cluster
densities spanning an order of magnitude.

### Correctness

`test_coresg` verifies **Theorem 1 of the CORE-SG paper directly**: with an exact k-NN
half and the exact `MST_kmax` as the bridge set, the MST extracted from the CORE-SG has
the same edge count and the same total weight as the MST over the complete
mutual-reachability graph, for every `mpts ∈ [1, k_max]`.

### k-NN construction, n = 200 000, d = 32, k = 32

| configuration | build | recall@32 |
|---|---|---|
| CPU, 1 thread | 16.612 s | 0.9234 |
| CPU, 4 threads | 3.956 s | 0.9235 |
| CPU, all threads | 1.455 s | 0.9236 |
| **CUDA, 1 instance** | **0.742 s** | 0.9237 |
| CUDA, 4 instances | 1.661 s | 0.9212 |
| brute force reference | 39.206 s | 1.0 |

22× over a single core, 2× over the whole CPU. Quality is identical across every backend
and thread count.

> On **one** GPU, more instances is *slower* — they serialise on the device and add merge
> cost. The instance count is a scheduling knob for **multi-GPU and queueing**, not an
> accuracy or single-device speed knob. `test_incremental` pins this down: recall varies
> by under 0.005 across 1, 2 and 6 instances.

### Incremental insertion, n = 30 000 → 60 000, d = 24, k = 32

| batch | insert | k-NN | hubness | graph touched | new-point recall |
|---|---|---|---|---|---|
| 1 | 0.273 s | 0.255 s | 0.004 s | 25.0 % | 0.9989 |
| 2 | 0.291 s | 0.272 s | 0.006 s | 25.0 % | 0.9985 |
| 3 | 0.336 s | 0.312 s | 0.006 s | 25.0 % | 0.9984 |
| 4 | 0.360 s | 0.335 s | 0.007 s | 25.0 % | 0.9982 |

Final recall after four incremental batches: **0.9985** — *higher* than a rebuild from
scratch on the same final data (0.9854), because the incremental path effectively runs
more total descent over the base points. Reverse degrees tracked through five batches of
edge churn match a recount of the final graph exactly (`test_incremental`).

### Anti-hub bridges vs the exact MST, n = 4 000, d = 24

| | anti-hub | exact |
|---|---|---|
| bridge construction | 0.001 s (247 selected, 246 edges) | 0.140 s (3 999 edges) |

**210× faster**, and the resulting spanning trees are near-identical in cost:

| mpts | approx weight | exact weight | ratio |
|---|---|---|---|
| 1 | 14 605.82 | 14 593.86 | 1.0008 |
| 8 | 16 766.08 | 16 754.13 | 1.0007 |
| 32 | 18 534.50 | 18 522.54 | 1.0006 |

DBSCAN\* labels cut from both MSTs at the same density threshold (`mpts=8`): **ARI 1.0000**.

The weight ratio is the number that matters, not the Jaccard index. Edge overlap falls as
`mpts` grows (0.99 → 0.19) because raising `mpts` flattens the mutual-reachability
landscape into large plateaus of tied weights, and any tie-break yields a different but
equally optimal tree. Total weight — and the clustering it induces — stays within 0.1 %.

### Cost of a full mpts sweep, n = 4 000

| | time |
|---|---|
| 32 MSTs from one CORE-SG | 0.060 s |
| the same 32 from the complete graph | 8.932 s |

**149×** — the CORE-SG payoff, reproduced.

---

## Deviations from the written specification

Four places where the implementation does not follow the spec literally. Each is a
deliberate call, and each is isolated behind a flag where that was possible.

**1. `reverse_score` limits — the spec's "approaches" bullets are wrong, its threshold
table is right.** With

```
score(v) = rd(v) / (mean over NN(v) of rd) = deg(v)·rd(v) / Σ rd(u)
```

the limits are: as the neighbours' mean approaches `rd(v)` the score approaches **1**
(not `1/k`, not `k`); as their degrees rise it approaches 0; as their degrees fall it
grows **without bound**. `k` is not a ceiling — with `deg = k`, `rd(v) = 100` and one
neighbour of degree 1, the score is `100k`. So the degenerate constant `k` used when
`sum = 0` is a *convention*, not the limit the formula tends to, and it is discontinuous
with the continuous case. Worth stating as a convention in the thesis text.

**2. The mean uses `deg(v)`, not `k`.** For a node whose list is short (a brand-new point
before refinement), dividing by `k` would not be an arithmetic mean. The degenerate-case
constant is still the nominal `k`, as specified.

**3. Update propagation goes to *reverse* neighbours, not neighbours.** The spec says a
node whose reverse degree changes "will trigger a `reverse_score` update in each of its
neighbours". But `score(v)` depends on `rd(u)` for `u ∈ NN(v)`, so when `rd(u)` moves the
nodes that must be rescored are those that *have `u` as a neighbour* — `u`'s reverse
neighbours. Using forward neighbours misses any reverse neighbour that is not also a
forward one. Both are implemented: exact (via the reverse table the spec suggests
keeping) is the default, forward is the fallback when `store_reverse_lists = false`.

**4. Connectivity repair — added, not in the spec.** An estimated bridge set can leave the
graph disconnected, and a disconnected replacement graph yields a *forest*, not a
spanning tree, which breaks every downstream density-based method. After the anti-hub
bridges are added, remaining components are spanned using representatives drawn from each
component's lowest-scoring nodes. Contracting a connected spanning subgraph stays
connected, so keeping only the inter-component edges of the representative MST is
sufficient. Disable with `ensure_connected = false` to measure how often the estimate
alone suffices — `CoreSGStats::components_before_repair` reports it either way, and the
sweep above quantifies exactly when it is needed.

## Choosing the denominator

`coresg_bench norm` — n = 6 000, d = 16, k = 24, 12 clusters, exact k-NN half so only the
hubness layer varies:

| denominator | tagged | selected | overlap | comps (no repair) | weight ratio @8 | ARI @8 | p10 / p50 / p90 |
|---|---|---|---|---|---|---|---|
| arithmetic | 89.6 % | 186 | — | 1 | 1.00085 | 1.0000 | 0.10 / 0.41 / 1.01 |
| geometric | 82.0 % | 186 | 98.9 % | 1 | 1.00085 | 1.0000 | 0.12 / 0.51 / 1.23 |
| median | 81.7 % | 186 | 96.2 % | 1 | 1.00090 | 1.0000 | 0.12 / 0.50 / 1.25 |

**Verdict: keep the arithmetic mean.** The robust denominators buy almost nothing —
89.6 % → 82 % tagged — and change nothing at all where it counts: identical bridge
quality, identical clustering, 96–99 % of the selected set is the same nodes.

The reason they can't help is more interesting than the result. It is **not** that the
mean is skewed by hubs. It is that *a neighbourhood is not a uniform sample of the graph*:

```
reverse degree, whole population       : mean 24.00   median 18
reverse degree, inside neighbourhoods  : mean 44.86   median 37
```

A node of reverse degree `r` appears in `r` neighbourhoods, so the degrees you see when
you look inside one are size-biased upwards — the friendship paradox. Roughly **2×**
here. Every denominator computed from neighbours inherits that bias, robust or not.

That also predicts where "neutral" actually sits. A node of typical degree scores
`E[rd] / (E[rd²]/E[rd]) = E[rd]²/E[rd²]`, which on this data is **0.535** — and the
observed median score is 0.51 (geometric) and 0.50 (median). The robust denominators land
on the prediction; the arithmetic mean lands lower (0.41) because it adds within-
neighbourhood skew on top of the size bias.

So if the anti-hub *tag* is to be used on its own, the fix is not the type of mean — it is
the threshold. Compare against `E[rd]²/E[rd²]` (one cheap pass over the degrees) instead
of against 1. With that correction the geometric or median denominator becomes genuinely
interpretable, since its neutral point is predictable from a population statistic. The
selection that feeds the bridges is unaffected either way, so this is a presentation
choice, not a correctness one.

## Ambient vs intrinsic dimensionality in the budget

The budget `d·ln(|E|)` uses the *ambient* dimensionality. Since the bridge MST costs
`m²·d/2` with `m ∝ d`, the whole step scales as **d³** — and on real data that breaks:

| dataset | ambient d | **intrinsic** | budget: ambient → intrinsic | bridge cost |
|---|---|---|---|---|
| SIFT1M | 128 | **10.2** | 1 825 → 146 | 156× cheaper |
| GIST1M | 960 | **28.5** | 13 842 → 411 | 92 G → 0.08 G ops, **1133×** |
| GloVe-100 | 100 | **34.5** | 1 443 → 498 | 8× |
| 20NG tfidf | 1000 | **36.3** | 10 969 → 455 | 581× |

`estimate_intrinsic_dimension` is the Levina–Bickel / Hill MLE read off the k-NN radii
already stored — no distance computations, one `O(n·k)` pass. Validated against known
ground truth: on 6-dimensional data rotated into 128 dimensions it recovers **5.67**.
SIFT's ~10 matches the published 10–16.

Two implementation notes that turned out to matter. Aggregate with the **harmonic mean**,
not the median: it is the unbiased pooling (MacKay & Ghahramani's correction) *and* the
only one robust to the exploded local estimates that concentrated data produces. And gate
on a **reliability check** — when most neighbourhoods carry no scale information the
estimate is worthless, so it falls back to ambient rather than guessing.

Ablation with the k-NN half held exact on both arms (`coresg_exp --experiment budget`):

| | selected | % of n | bridge time | weight ratio | ARI |
|---|---|---|---|---|---|
| SIFT / GIST / GloVe / 20NG, ambient | 1 143–4 000 | 28–100 % | — | 1.000000 | 1.0000 |
| same, intrinsic | 112–386 | 2.8–9.7 % | **5–49× faster** | 1.000000 | 1.0000 |

Zero quality cost — but that comparison is weaker than it looks: **every one of those real
data sets has a connected k-NN graph**, so the bridges had no work to do and both arms
were trivially exact. On synthetic manifold data with genuine density separation
(6-in-128, 60 clusters) a real trade-off appears: ambient (1 415 anti-hubs) gives weight
ratio 1.0010, intrinsic (126) gives 1.0218, at 14× less bridge time. DBSCAN\* **ARI is
1.0000 for both**, so the extra weight does not change the clustering.

**Correcting an earlier claim.** An anti-hub count sweep on low-dimensional blobs
suggested the budget was massively over-provisioned (4 anti-hubs ≈ 1024). That does not
generalise. On manifold data with cluster structure the improvement is monotone and
substantial:

| anti-hubs | 4 | 16 | 64 | 256 | 1024 |
|---|---|---|---|---|---|
| weight ratio | 1.0180 | 1.0180 | 1.0163 | 1.0103 | **1.0015** |
| bridge time (s) | 0.0006 | 0.0006 | 0.0008 | 0.0008 | 0.0041 |

Whether extra anti-hubs help is **data dependent**. Cost is nearly flat to ~256, so the
defensible recommendation is intrinsic dimensionality for the *scaling* (it removes the
d³ blowup and the degeneracy) combined with a multiplier or floor, rather than a bare
`ID·ln(|E|)`, which under-provisions on clustered manifolds.

## Does the anti-hub bridge set ever miss a component?

`coresg_bench repair` — n = 20 000, repair disabled, sweeping the shapes that should
stress the estimate. Component count from k-NN + bridges alone:

| clusters | d=2, sep 4 | d=2, sep 12 | d=8, sep 4 | d=8, sep 12 | d=32, sep 4 | d=32, sep 12 |
|---|---|---|---|---|---|---|
| 4 | 1 | 1 | 1 | 1 | 1 | 1 |
| 12 | 1 | 1 | 1 | 1 | 1 | 1 |
| 40 | 1 | **2** | 1 | **3** | 1 | 1 |
| 120 | 1 | **8** | **17** | **41** | **5** | **12** |

Repair was needed in 7 / 24 configurations. **The intuition that the anti-hubs rarely
miss a connection is correct up to about 12 clusters — 0 failures in 12 configurations.**
It stops being correct when the number of density-separated regions approaches the
anti-hub budget.

The mechanism is exact and predictable: the budget is `d·ln(|E|)`, which does **not**
depend on the number of clusters. At d=2 it is 25 nodes; with 120 well-separated clusters
most clusters receive zero selected anti-hubs, and a bridge cannot leave a cluster that
has no representative. Hence d=8 with 120 clusters (102 selected) fragments into 41
pieces, while d=32 (408 selected) only reaches 12. Wider separation always makes it worse,
because the k-NN half is already more fragmented.

Two takeaways: the repair is not optional insurance at high cluster counts, and the budget
needs a term that reacts to the observed component count — `d·ln(|E|)` alone cannot see
how many clusters there are.

### The fix: a per-component quota

`HubnessParams::min_per_component` (default **2**) guarantees that every k-NN component
contributes at least that many of its own lowest-scoring nodes, on top of the global
budget. Components are already computed before selection runs, so this costs one extra
`O(n · quota)` pass with a tiny constant.

Re-running the same sweep, budget rule against quota rule:

| clusters | d | sep | selected (budget) | comps (budget) | selected (quota) | comps (quota) |
|---|---|---|---|---|---|---|
| 12 | 8 | 12 | 102 | 1 | 102 | **1** |
| 40 | 2 | 12 | 25 | 2 | 29 | **1** |
| 40 | 8 | 12 | 102 | 1 | 113 | **1** |
| 120 | 2 | 12 | 25 | 7 | 40 | **1** |
| 120 | 8 | 4 | 102 | 18 | 144 | **1** |
| 120 | 8 | 12 | 102 | 38 | 229 | **1** |
| 120 | 32 | 12 | 408 | 12 | 447 | **1** |

**Bridges alone now connect the graph in 24 / 24 configurations, against 18 / 24 before.**

It is a targeted fix, not a blanket cost increase: wherever the budget already sufficed it
adds *nothing* (25 → 25, 102 → 102, and `+0` on ordinary 4-cluster data in the unit test).
The worst case is 102 → 229 selected, and since the anti-hub MST is `O(m²)` on a step that
takes 0.001 s, a 5× there is invisible. Bridge accuracy is unchanged — weight ratio still
≤ 1.0008, DBSCAN\* ARI still 1.0000.

It also closes a hole a score-ranked budget structurally cannot: a small, uniform,
detached component has **no** node scoring below 1. If every member has the same reverse
degree as its neighbours, everyone is exactly neutral, so no score ranking will ever reach
that region however detached it is. Bucketing by component sidesteps the score entirely.
`test_hubness` pins this down on a graph where nothing at all is tagged an anti-hub.

## Finding the components cheaply

Since the repair needs a component labelling every batch, that labelling had better not
be the most expensive thing in an update. Three approaches, measured on n = 100 000,
d = 16, k = 24 (`coresg_bench conn`):

| batch | destroyed edges | legacy | **parallel rebuild** | incremental |
|---|---|---|---|---|
| 1 000 (1 %) | 51 345 | 14.29 ms | **1.13 ms** | 18.65 ms |
| 5 000 (5 %) | 237 154 | 14.61 ms | **1.34 ms** | 65.41 ms |
| 20 000 (20 %) | 782 922 | 19.18 ms | **2.00 ms** | 209.22 ms |

**First: exact connectivity cannot be sublinear in n.** A single isolated point is
invisible unless something touches it, and sublinear property testers only separate
"connected" from "ε-far from connected" — blind to exactly the small distant cluster that
matters here. So Ω(n) is a floor, and the question is only how tight the constant is and
how well it parallelises.

**What worked (10×):** stop materialising the edge list. The old path built an
`n·k/2`-entry `vector<Edge>` purely to feed union-find. `ConnectivityTracker::rebuild`
runs a lock-free union-find directly over the adjacency arrays, spread across the thread
pool, touching `n·k` ids and allocating nothing. Roots link by index rather than by rank,
which makes concurrent unions acyclic *and* schedule-independent — so the parallel result
is bit-identical to the serial one, which `test_connectivity` asserts. Folding the bridges
in afterwards is `O(n + |bridges|)` on the cached labels rather than a second sweep.

**What didn't (and why):** maintaining components across batches. The scheme is sound —

> if every undirected edge destroyed this batch still has its endpoints connected in the
> resulting graph, the components are unchanged

(replace each destroyed edge in any old path by a replacement path and splice) — and it is
implemented, exact, and tested against a rebuild batch after batch, with the fallback
narrowed to just the components a failed certificate touched. It still loses, decisively,
and for a reason worth writing down: **NN-Descent destroys an enormous number of edges per
batch.** 51 000 undirected edges die at a 1 % batch; 783 000 at 20 %. Each needs a
bidirectional search, and that is simply more work than a 1–2 ms rebuild. The certificate
is not too weak — the diagnostics confirm the fallbacks are genuine (points really do
migrate between clusters as the graph refines, `fail_exhausted` and separate components in
the rebuilt labelling), not budget failures.

Kept behind `incremental_connectivity` (default **off**) because it becomes the right
answer the moment the destroyed set shrinks — which is what freezing converged regions of
the k-NN graph would do.

## Known limits

* `exact_mst()` and `brute_force_graph()` are `O(n² d)` references for evaluation — they
  are not scalable paths. Practical up to ~10⁵ points on GPU.
* Instance staging materialises all instance vector blocks at once (≈ dataset size +
  `instances × |batch|`). Streaming them per device would remove the peak.
* Anti-hub selection re-runs an `O(n)` `nth_element` per batch. Exact and cheap in
  absolute terms, but not sublinear like the rest of the update.
* The reverse-neighbour table is `std::vector<std::vector<idx_t>>` — flexible but with a
  per-node header cost; auto-disabled above `reverse_list_node_limit` nodes.
* L1 and inner product run on CPU only. Inner product is rejected by `CoreSG` outright:
  mutual reachability needs a non-negative dissimilarity.
* Multi-GPU has been exercised through the scheduler and the CPU multi-slot path, but
  this machine has a single GPU — the cross-device numbers are still to be measured.
