#!/usr/bin/env bash
# On GBStation, after staging experiment files; public source mounted read-only.
# UPSTREAM_SRC=/path/to/public-checkout EXPERIMENT_SCRIPTS=/path/to/staged/scripts \
#   bash run_eval_packed_owned.sh /path/to/new-results [--limit-per-bench 1]
# Add --score or --summarize after inference (no GPU requested).
set -euo pipefail
root="$HOME/projects/s4vy-demo"
: "${UPSTREAM_SRC:?Set UPSTREAM_SRC to the unchanged public checkout}"
: "${EXPERIMENT_SCRIPTS:?Set EXPERIMENT_SCRIPTS to staged experiment scripts}"
results="$1"; shift
mkdir -p "$results"
results=$(realpath "$results")
gpu=(--gpus all)
for arg in "$@"; do
  if [[ "$arg" == --score || "$arg" == --summarize ]]; then gpu=(); fi
done
if ((${#gpu[@]})); then
  # Fail if the service cannot confirm no active users.
  curl --fail --silent --max-time 15 "${DEMO_HEALTH_URL:-http://10.88.7.155:7860/healthz}" |
    python3 -c 'import json,sys; h=json.load(sys.stdin); assert h["active_streams"] == h["batch_jobs"] == 0, "Demo is busy"'
fi
sudo -n docker run --rm "${gpu[@]}" --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --user 2583:2583 --cap-drop ALL --security-opt no-new-privileges \
  -e HOME=/tmp -e USER=s4vy -e LOGNAME=s4vy -e PYTHONUNBUFFERED=1 -e OMP_NUM_THREADS=4 \
  -e OPENBLAS_NUM_THREADS=1 -e SEG_NO_FRAMEPOS=1 -e VGGT_OMEGA_ROOT=/workspace/vggt-omega \
  -e UPSTREAM_COMMIT="$(git -C "$UPSTREAM_SRC" rev-parse HEAD)" \
  -e TORCHINDUCTOR_CACHE_DIR=/workspace/cache/torchinductor-upstream \
  -e TRITON_CACHE_DIR=/workspace/cache/triton-upstream \
  -v "$UPSTREAM_SRC:/workspace/upstream:ro" \
  -v "$EXPERIMENT_SCRIPTS:/workspace/experiment:ro" \
  -v "$root/source/vggt-omega:/workspace/vggt-omega:ro" \
  -v "$root/weights:/workspace/weights:ro" \
  -v "$root/benchmarks/paper-eval-20261101/inputs:/workspace/inputs:ro" \
  -v "$root/analysis/upstream-postprocess-20261111:/workspace/original-analysis:ro" \
  -v "$root/analysis/paper-eval-20261101:/workspace/paper-eval:ro" \
  -v "$root/cache:/workspace/cache" -v "$results:/workspace/results" \
  s4vy-benchmark:20261031 python /workspace/experiment/eval_packed_owned.py "$@"
