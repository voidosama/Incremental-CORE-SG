# Running the suite on the cluster

Target: 4× RTX 2080 SUPER (sm_75, 8 GB each), 126 GB RAM.

Run the steps in order. **Do not skip step 4** — it is a two-minute check that has already
caught one bug that would have silently corrupted every scaling row.

---

## 1. Environment

```bash
conda env create -f environment.yml
conda activate coresg
```

CUDA comes from the site, not conda, so the toolkit matches the driver:

```bash
module avail cuda            # see what the site offers
module load cuda/12.4        # or whatever is available; 11.8+ is fine
nvcc --version && nvidia-smi
```

If there is no module system, uncomment the `cuda-toolkit` lines in
`environment.yml` and pin at or below the maximum CUDA version `nvidia-smi` reports.

---

## 2. Build

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=75

cmake --build build -j
ctest --test-dir build --output-on-failure
```

`75` is the RTX 2080 SUPER. **sm_70 is a hard floor** — the local-join kernel holds
per-node spin locks inside a warp, which only makes progress under independent thread
scheduling (Volta and newer). CMake refuses anything lower rather than producing a binary
that deadlocks.

Expect 7/7 tests to pass. If `gxx` is too new for nvcc, that surfaces here as a host
compiler error; the pin in `environment.yml` is there to prevent it.

---

## 3. Data

```bash
python scripts/prepare_datasets.py newsgroups --out-dir data
python scripts/prepare_datasets.py fma data/fma_metadata.zip --out-dir data
python scripts/prepare_datasets.py hdf5 data/glove-100-angular.hdf5 --key train \
       --out data/glove100.fvecs
python scripts/run_suite.py --generate          # synthetic families, incl. the 10M set
```

Copy the already-downloaded archives (`sift.tar.gz`, `gist.tar.gz`,
`glove-100-angular.hdf5`, `fma_metadata.zip`) rather than re-downloading — ~3.5 GB, and
the TEXMEX host is slow. Extract with `tar -xzf`.

Disk needed: ~14 GB for data plus ~1.3 GB for the 10M synthetic set.

```bash
python scripts/run_suite.py --list      # should report 37 runs, 0 skipped
```

---

## 4. Preflight — do not skip

```bash
./build/bin/coresg_cli preflight --n 50000 --dim 32 --kmax 24
```

Validates, on the real hardware, the things that fail *silently* rather than loudly:

- every device is sm_70+;
- **each device individually** builds a correct graph (recall against brute force);
- **all devices together** work, and every instance actually ran on a GPU rather than
  being quietly demoted to the host (instances below 512 points are handed to the CPU, so
  a row labelled "4 GPUs" can be pure CPU);
- nothing ran out of device memory;
- per-device memory headroom and the per-point cost.

Exit code 0 means go. Anything else, fix it before starting an eight-hour sweep.

---

## 5. Run

Sequential by construction, appending to one CSV, so an interruption never loses completed
work:

```bash
python scripts/run_suite.py --seeds 5 --out results.csv --keep-going 2>&1 \
  | tee -a run.log
```

`--keep-going` continues past a failing run instead of stopping. Resume after any
interruption — walltime kill, preemption, a bug in run 31 of 37:

```bash
python scripts/run_suite.py --seeds 5 --out results.csv --resume --keep-going
```

`--resume` skips `(experiment, dataset)` pairs already in the CSV, so re-running is safe
and does not duplicate rows.

### Under a scheduler

Do not run this on a login node. With SLURM:

```bash
#!/bin/bash
#SBATCH --job-name=coresg
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=coresg-%j.log

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate coresg
module load cuda/12.4

nvidia-smi
./build/bin/coresg_cli preflight --n 50000 || exit 1
python scripts/run_suite.py --seeds 5 --out results.csv --resume --keep-going
```

Submit with `sbatch run.sh`. If walltime kills it, resubmit the same script — `--resume`
picks up where it stopped.

### Splitting the work

```bash
# fast, no GPU needed
python scripts/run_suite.py --only bridges,budget,mpts --out results.csv --resume

# the long ones
python scripts/run_suite.py --only incremental,drift,sensitivity --out results.csv --resume

# needs all 4 GPUs
python scripts/run_suite.py --only scaling --out results.csv --resume
```

---

## 6. FISHDBC baseline

Separate, because it is single-threaded Python and slow. Keep `--limit` modest.

```bash
python scripts/baseline_fishdbc.py --data data/manifold_mid.fvecs --name manifold_mid \
       --limit 20000 --batches 8 --seeds 3 --out results.csv
python scripts/baseline_fishdbc.py --data data/sift/sift_base.fvecs --name sift1m \
       --limit 20000 --batches 8 --seeds 3 --out results.csv
```

Compare **quality and per-batch cost slope, never raw wall-clock** — a wall-clock race
between Python and CUDA measures the languages. See `EXPERIMENTS.md` §5.

---

## 7. Collect

```bash
wc -l results.csv
python - <<'EOF'
import pandas as pd
d = pd.read_csv("results.csv")
print(d.groupby(["experiment", "dataset"]).size().to_string())
EOF
```

Bring back `results.csv` and `run.log`. Everything else is reproducible from the scripts.

---

## Known hazards

**Warm-up.** CUDA context creation costs 100–300 ms *per device* and updates are
sub-second. The scaling experiment builds once before timing for exactly this reason; if
you add timed code, do the same.

**PCIe contention.** The design has no cross-device traffic, but four GPUs uploading
instance vectors at once share one bus. Expect this — not Amdahl — to dominate the
deviation from linear scaling.

**The Amdahl ceiling is 2.45×, measured.** The parallel fraction is 0.79 (instances 79 %,
merge 16 %, refine 5 %), so 4 GPUs cannot exceed `1/(0.21 + 0.79/4)`. Publish the
prediction alongside the measurement; hitting a predicted ceiling is a far stronger result
than missing an unstated one.

**`capacity10m` is expected to OOM on one instance.** That is the experiment: ~7.9 GiB for
a single instance versus ~2.0 GiB across four. The OOM is caught and recorded as a data
point (`jobs_out_of_memory`) rather than crashing the sweep.

**Instances below 512 points run on the CPU.** If you raise `--instances` on a small data
set, jobs shrink past that threshold and the "GPU" rows become CPU rows. Preflight checks
for this; the CSV records it per run.
