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
python scripts/run_suite.py --only strategy
```

7 GPU families + 3 CPU families, 3 seeds, 8 batches. This is the A/B that decides whether
an update costs `O(batch)` or `O(n_base)`, and it is the experiment the paper's Claim A
should now be built on. The column that carries the argument is `instance_nodes` plotted
against `n`: flat for `anchored`, linear for `resample`.

The CPU rows are not optional. The old strategy **loses to a plain rebuild** on the CPU
backend (measured: 1.23 s per update against a 1.11 s rebuild, sift1m n=100k) and that
inversion is invisible in a GPU-only sweep because the device absorbs the extra work.

---

## Must re-run (the insert path changed underneath them)

| experiment | why | rough cost |
|---|---|---|
| `incremental` | per-batch cost and post-stream recall both move | 8 families × 5 seeds |
| `arrival` | same code path; grouped arrival stresses the seeding search differently, and that is a genuinely open question — a beam search on clustered data is the weak case | 3 families |
| `batchsize` | the fixed-vs-amortised split changes completely: the update no longer grows with `n`, so the "fixed share rises at small batches" curve has a different shape | 4 families × 3 seeds |
| `drift` | 25 batches is where a sublinear update either holds quality or rots; the old numbers say nothing about the new path | 4 families |
| `sensitivity` | `refine_max_fraction` is no longer the binding cap (`refine_per_new` is), so the old sweep measures a knob that is now inert. Add `context_per_new`, `search_ef`, `search_reverse_cap`, `owned_warm_fraction` | 5 families |
| `parity` | CPU/GPU agreement has to be re-established: the seeding search runs on the host in both configurations, which is a new source of asymmetry | 3 families × 3 seeds |
| `scaling` | **the claim most at risk.** Instance nodes are now `s·|batch| + context` instead of `n_base`, so instances are far smaller and the parallel fraction `f` will drop. The 4-GPU numbers (2.50–3.23×) are stale and probably optimistic for the new path at small batches | 3 families, needs the cluster |

## Do NOT re-run

| experiment | why it is unaffected |
|---|---|
| `bridges` | sets `brute_force_limit = n`, so the k-NN half is exact and the insert strategy cannot touch it. Deterministic; the 0.00000 standard deviations still hold |
| `budget` | same — the `d·ln|E|` rule and the anti-hub selection are untouched |
| `connectivity` | measures the k-NN graph's component structure as a function of `k`, from `build()` only |
| `mpts` | one graph, many MSTs — extraction only |
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
