# What to re-run after the anchored-insert change (2026-08-06)

`results.csv` (18,485 rows) was produced entirely with the **Resample** strategy, which
is now `--strategy resample` and no longer the default. Every row that involves an
`insert()` is therefore measuring the old algorithm. Rows that only involve `build()` are
unaffected and must **not** be re-run — re-running them costs days and changes nothing.

The change also has a *quality* direction, not only a cost one: an anchored update no
longer re-descends the whole base, so the base's neighbourhoods are no longer refined once
per batch. Recall after a stream is lower than it was (0.91–0.97 against 0.95–0.999), and
that is the trade the whole change buys. Anything reporting post-stream recall has to be
re-measured, not just re-timed.

---

## Do this first — it is the new headline

```bash
python scripts/run_suite.py --only strategy --out results_v2.csv
```

7 GPU families + 3 CPU families, 3 seeds, 8 batches. This is the A/B that decides whether
an update costs `O(batch)` or `O(n_base)`, and it is the experiment the paper's Claim A
should now be built on. Four columns carry the whole argument, all per batch:

| column | anchored | resample |
|---|---|---|
| `instance_nodes` | flat in `n` | linear in `n` |
| `insert_seconds` | flat | climbing |
| `rebuild_speedup` (against a rebuild at *that* `n`) | climbing | roughly flat |
| `largest_instance_bytes` | flat — the capacity claim | climbing |

The CPU rows are not optional. The old strategy **loses to a plain rebuild** on the CPU
backend (measured: 1.23 s per update against a 1.11 s rebuild, sift1m n=100k) and that
inversion is invisible in a GPU-only sweep because the device absorbs the extra work.

### The figure that actually settles the slope

One `strategy` run only varies `n` by 2× (base is half the file), which is not enough to
show a slope convincingly. Run the same experiment at several total sizes with the **batch
size held constant** — `per_batch = 0.5 * limit / batches`, so `batches = limit / 25000`
keeps it at 12,500:

```bash
for L in 100000 200000 400000 1000000; do ./build/bin/coresg_exp --experiment strategy --data data/sift/sift_base.fvecs --name sift1m_n$L --k 32 --limit $L --batches $((L / 25000)) --seeds 3 --out results_v2.csv; done
```

Plot `rebuild_speedup` at the final batch against `n`. Flat update over linear rebuild
predicts a straight line through the origin; measured on the dev box it went 1.29× at
n=200k to 2.59× at n=400k. **This is the plot to lead the paper's Claim A with** — a single
`n` proves nothing, and the absolute number at small `n` is unimpressive on its own.

---

## Must re-run (the insert path changed underneath them)

| experiment | why | rough cost |
|---|---|---|
| `incremental` | per-batch cost and post-stream recall both move | 8 families × 5 seeds |
| `arrival` | same code path; grouped arrival stresses the seeding search differently, and that is a genuinely open question — a beam search on clustered data is the weak case | 3 families |
| `batchsize` | the fixed-vs-amortised split changes completely: the update no longer grows with `n`, so the "fixed share rises at small batches" curve has a different shape | 4 families × 3 seeds |
| `drift` | 25 batches is where a sublinear update either holds quality or rots; the old numbers say nothing about the new path | 4 families |
| `sensitivity` | `refine_max_fraction` is no longer the binding cap (`refine_per_new` is), so the old sweep measures a knob that is now inert. Four anchored knobs are now swept — `search_ef`, `context_per_new`, `owned_warm_fraction`, `search_reverse_cap`. **`search_ef` is the one to plot**: it answers "is the recall difference a fixed loss or a dial?", which is the first thing a reviewer asks | 5 families |
| `parity` | CPU/GPU agreement has to be re-established: the seeding search runs on the host in both configurations, which is a new source of asymmetry | 3 families × 3 seeds |
| `scaling` | **the claim most at risk.** Instance nodes are now `s·|batch| + context` instead of `n_base`, so instances are far smaller and the parallel fraction `f` will drop. The 4-GPU numbers (2.50–3.23×) are stale and probably optimistic for the new path at small batches | 3 families, needs the cluster |
| `compare_incremental.py` (FISHDBC) | easy to miss: it drives `--experiment stream`, which goes through `base_params` and therefore now runs anchored. The published external baseline comparison is measuring the old algorithm | 3 families |

FISHDBC specifically:

```bash
python scripts/compare_incremental.py --data data/manifold_mid.fvecs --name manifold_mid --limit 20000 --batches 8 --mpts 10 --out results_v2.csv
```

Its per-batch cost is flat because HNSW insertion is `O(log n)` per point. That used to be
the one place a streaming competitor looked structurally better than this project; an
anchored update is now flat too, so the comparison is worth re-reading rather than just
re-running.

## Do NOT re-run

| experiment | why it is unaffected |
|---|---|
| `bridges` | sets `brute_force_limit = n`, so the k-NN half is exact and the insert strategy cannot touch it. Deterministic; the 0.00000 standard deviations still hold |
| `budget` | same — the `d·ln|E|` rule and the anti-hub selection are untouched |
| `connectivity` | measures the k-NN graph's component structure as a function of `k`, from `build()` only |
| `mpts` | one graph, many MSTs — extraction only. Its 71–303× rows stay valid and are the **CORE-SG side of the multiple-mpts claim against cuML**: extraction reads the k-NN graph and the bridges, so its cost depends on `n` and `|E|`, not on how the graph was constructed. Join them to `mpts_cuml` from `compare_cuml.py` |
| `scaling` `capacity10m_oom` / `oom128` | those probe `build()` footprints on 10M points and took ~2 days. The insert path is not what they measure |

---

## New experiments to run

### 1. cuML head-to-head (needs Linux/WSL2 — RAPIDS has no native Windows build)

```bash
python scripts/compare_cuml.py --data data/sift/sift_base.fvecs --name sift1m \
    --limit 200000 --k 32 --batches 8 --mpts 10 --mpts-sweep 4,8,16,24,32 \
    --out results.csv
```

Unlike the FISHDBC comparison, **wall-clock is the result here** — both sides are CUDA on
the same card, so the usual "you are measuring the languages" objection does not apply.
Two separable claims come out of it, and they should be reported separately:

1. per-batch update + extraction against a full cuML rebuild, as a function of `n`. The
   *slope* is the claim; a single `n` proves nothing, so run at 100k / 200k / 400k;
2. many mpts from one graph — cuML must refit per value, CORE-SG re-extracts. This is the
   CORE-SG paper's own contribution and cuML has no equivalent, so do not fold it into (1).

Both sides are scored by ARI against exact HDBSCAN* recomputed **per batch**, and against
each other. `min_samples` is pinned to `--mpts` everywhere (cuML and sklearn both default
it to `min_cluster_size`, FISHDBC to 5 — left alone the three sides cluster at three
different densities).

### 2. cuML memory exhaustion

```bash
python scripts/compare_cuml.py --mode capacity --data data/capacity10m.fvecs \
    --name capacity10m --k 32 --mpts 10 \
    --capacity-sizes 1000000,2000000,4000000,8000000,10000000 --out results.csv
```

Walks `n` upward until cuML cannot fit the problem on one card, recording the OOM as a row
rather than a traceback. Then the CORE-SG side at the same sizes, which reports the
per-device footprint that makes the comparison:

```bash
./build/bin/coresg_exp --experiment scaling --data data/capacity10m.fvecs \
    --name capacity10m --k 32 --scaling-instances 1,4,8,16 --no-oom-fallback \
    --skip-cpu-scaling --out results.csv
```

Be honest about what this shows. It is a statement about **what becomes possible**, not
about speed: cuML has to hold the whole problem on one device, while an anchored update is
sized by the batch. Report cuML's peak device bytes next to `largest_instance_bytes`, and
say plainly that a single 8 GB card is the constraint being demonstrated.

---

## Two things the new numbers will expose — plan for them

**The seeding search runs on the CPU and is now 55–65 % of a GPU update.** At n=200k it is
0.62 s of a 0.99 s update. That is why the GPU margin is only 1.29× at 200k while the CPU
margin is 1.77×: on the GPU arm the device finishes its share and waits for the host. This
is the single biggest remaining lever and it is a known, bounded piece of work — batched
beam search is exactly what GPU ANN libraries do. Until it is done, do not claim a large
constant-factor win on GPU; claim the *slope*, which is real and is what widens with `n`
(1.29× at 200k → 2.59× at 400k, same batch size).

**Multi-GPU efficiency will fall.** `f = 0.79–0.94` was measured with instances covering
the whole base. Instances are now bounded by the batch, so there is less work to divide and
the fixed per-job transfer amortises worse. The capacity argument (bounded per-device
memory) is unaffected and arguably stronger, but the strong-scaling table needs re-measuring
before any 4-GPU number is quoted.
