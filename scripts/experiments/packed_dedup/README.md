# Packed mask dedup experiment

This folder tests CPU bit-packed mask suppression before adopting it in released
inference. The helper preserves the dense reference's greedy ordering and strict
`IoU > 0.7` rule, using integer nearest-neighbor pixel weights. It compiles a small
C++17 library into a host-local temporary cache on first use. Python needs NumPy;
the host or container needs a C++ compiler (`c++`, or set `CXX`).

The GBStation experiment compares `owned_off` with `owned_tube07` using the public
checkpoint and whole-clip ownership decoder. Both arms share a model forward.
Dedup uses the release decoder's score and mask thresholds, 0.1 and 0.65, then
suppresses whole tubes before ownership. This differs from the paper's W8
framewise experiment, which used thresholds 0.3 and 0.5. The helper is not enabled
in `infer_segmenter.py`, the demo, or the shared decoder.

## CPU parity and timing

From the repository root, on branch `experiment/packed-dedup`:

```bash
EXPERIMENT_SCRIPTS="$(git rev-parse --show-toplevel)/scripts/experiments/packed_dedup"
python -m unittest discover -s "$EXPERIMENT_SCRIPTS" -p 'test_packed_dedup.py' -v
python "$EXPERIMENT_SCRIPTS/benchmark_packed_dedup.py" \
  --suite extended --repeats 5 --output /tmp/packed-dedup-cpu/benchmark.json
python "$EXPERIMENT_SCRIPTS/benchmark_packed_dedup.py" \
  --case tube400x100 --packed-only --repeats 1 \
  --output /tmp/packed-dedup-cpu/stress.json
```

Each benchmark writes JSON and Markdown. Paired cases require exact keep,
keeper, and IoU equality against `reference_dedup.py` before timing. Timing
includes packing on every call and excludes initial compilation. These inputs
are synthetic moving rectangles, with 25% exact duplicates; the CPU results do
not measure GPU inference or complete video latency.

The Q400, 100-frame stress case skips the dense reference. It reports packed
latency and measured peak process RSS, with no reference speedup or full-size
parity claim. It touches about 896 MB of Boolean input; allow additional space
for packed masks and scratch. Storage estimates and measured RSS are separate
fields in the output. Run the benchmark on GBStation to measure its CPU; timings
from another architecture do not establish its performance.

## GBStation setup

Run these commands on GBStation using the existing benchmark installation. The
wrapper expects Docker image `s4vy-benchmark:20261031`, passwordless Docker access,
and these existing assets under `$HOME/projects/s4vy-demo`:

- `source/vggt-omega` and `weights/upstream-hf/segmenter/s4vy.pt`;
- `benchmarks/paper-eval-20261101/inputs`, including `manifest.json`;
- `analysis/upstream-postprocess-20261111/score.py` and
  `analysis/paper-eval-20261101` for the original scoring environment;
- writable `cache` and output directories, accessible to container UID/GID 2583.

The scorer, benchmark data, checkpoint, and image are local prerequisites, not
included in this experiment folder. The container also needs the C++ compiler.
The wrapper checks the existing demo health service before GPU use, and the
runner checks again between clips. A busy or unreachable service stops the run.
Use a fresh result directory for each inference run; automatic resume is disabled.

```bash
export UPSTREAM_SRC="$(git rev-parse --show-toplevel)"
export EXPERIMENT_SCRIPTS="$UPSTREAM_SRC/scripts/experiments/packed_dedup"
PILOT_RESULTS="$HOME/projects/s4vy-demo/analysis/packed-dedup-pilot-$(date -u +%Y%m%dT%H%M%SZ)"
```

The checkout and experiment scripts are mounted read-only. Results and compiler
caches are written outside the checkout.

## Paired pilot and full evaluation

Start with one clip per benchmark and three warm repeats:

```bash
bash "$EXPERIMENT_SCRIPTS/run_eval_packed_owned.sh" "$PILOT_RESULTS" \
  --limit-per-bench 1 --warm-repeats 3
bash "$EXPERIMENT_SCRIPTS/run_eval_packed_owned.sh" "$PILOT_RESULTS" --score
bash "$EXPERIMENT_SCRIPTS/run_eval_packed_owned.sh" "$PILOT_RESULTS" --summarize
```

After reviewing the pilot output, run the four benchmark cohorts into a new directory:

```bash
FULL_RESULTS="$HOME/projects/s4vy-demo/analysis/packed-dedup-full-$(date -u +%Y%m%dT%H%M%SZ)"
bash "$EXPERIMENT_SCRIPTS/run_eval_packed_owned.sh" "$FULL_RESULTS" --warm-repeats 3
bash "$EXPERIMENT_SCRIPTS/run_eval_packed_owned.sh" "$FULL_RESULTS" --score
bash "$EXPERIMENT_SCRIPTS/run_eval_packed_owned.sh" "$FULL_RESULTS" --summarize
```

The default cohorts are ScanNet-IGGT, DAVIS, LVOS, and VIPSeg. `--score` reuses the
existing scorer after inference; it does not rerun the model. Scoring and
summarization request no GPU. Summarization writes `paired_summary.json`,
`paired_results.csv`, and `paired_results.md`. The JSON retains per-benchmark
warm timing; the CSV and Markdown contain accuracy deltas in percentage points.
Check selected and scored clip counts. Confidence intervals are not computed.
Retain the runtime record, selected-clip manifest, paired predictions, per-clip
receipts, and summaries together.

Review accuracy deltas alongside warm timing before enabling dedup in released
inference. GBStation warm forward-plus-decode timing includes costs absent from
the CPU helper benchmark, especially GPU-to-CPU transfer and mask ownership.
It excludes input loading and output writing, so it is not complete video
latency. The summary averages per-clip repetition medians. Cold graph/compile
costs are recorded separately from warm repetitions. The bounded real-tensor parity
check establishes helper agreement on a subset, not full-size W8 decoder parity.
