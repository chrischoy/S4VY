#!/usr/bin/env python3
"""Isolated CPU pilot: paper float64 dedup versus packed dedup, including packing.

Synthetic Boolean rectangles/tubes, not model predictions or end-to-end latency.
Each method runs in its own process. Peak RSS includes interpreter, imports, input
construction, warmup and timed calls; it is not incremental kernel memory.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time

# Set before NumPy import, including in subprocesses. Do not let BLAS oversubscribe
# the login node or silently compare different thread configurations.
for _key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_key] = "1"
import numpy as np

SCRIPTS = Path(__file__).resolve().parent
CASES = {
    "frame64": (64, 1, 112, 200, 450, 800),
    "frame128": (128, 1, 112, 200, 450, 800),
    "tube32x8": (32, 8, 112, 200, 450, 800),
    "frame400": (400, 1, 112, 200, 450, 800),
    "tube64x8": (64, 8, 112, 200, 450, 800),
    "tube400x100": (400, 100, 112, 200, 450, 800),
}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def inputs(case, seed):
    q, frames, h, w, height, width = CASES[case]
    rng = np.random.default_rng(seed)
    masks = np.empty((q, frames, h, w), dtype=bool)
    masks.fill(False)  # Touch input pages so stress RSS includes the full input.
    for i in range(q):
        if i % 4 == 1:
            masks[i] = masks[i - 1]  # Known exact duplicates, not all disjoint.
            continue
        y, x = rng.integers(0, h - 4), rng.integers(0, w - 4)
        dy, dx = rng.integers(4, max(5, h // 2)), rng.integers(4, max(5, w // 2))
        for t in range(frames):
            yy, xx = min(int(y) + t, h - 1), min(int(x) + t, w - 1)
            masks[i, t, yy:min(yy + dy, h), xx:min(xx + dx, w)] = True
    scores = rng.uniform(.31, .99, q).astype(np.float32)
    scores[1::4] = scores[::4][:len(scores[1::4])]  # Exercise index ties.
    wy = np.bincount(np.arange(height) * h // height, minlength=h)
    wx = np.bincount(np.arange(width) * w // width, minlength=w)
    weights = np.broadcast_to(wy[:, None] * wx[None, :], (frames, h, w)).copy()
    return masks, scores, weights


def peak_rss_bytes():
    # Linux VmHWM resets on exec; getrusage can retain a launcher high-water
    # mark inherited across exec and falsely report the parent's allocation.
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def worker(args):
    path = SCRIPTS / ("reference_dedup.py" if args.worker == "reference" else "packed_dedup.py")
    fn = load(path, "bench_" + args.worker).dedup_keep
    masks, scores, weights = inputs(args.case, args.seed)
    # Warmup includes extension loading/compilation, excluded from latency but
    # included in process peak RSS. Timed calls always receive unpacked masks.
    warm_start = time.perf_counter()
    if args.packed_only:
        keep = fn(masks, scores, .7, weights=weights)
        keeper, iou = np.empty(0, int), np.empty(0, float)
    else:
        keep, keeper, iou = fn(masks, scores, .7, weights=weights, return_keeper=True)
    warm_seconds = time.perf_counter() - warm_start
    samples = []
    for _ in range(0 if args.verify_only else args.repeats):
        start = time.perf_counter()
        timed_keep = fn(masks, scores, .7, weights=weights)
        samples.append(time.perf_counter() - start)
        if not np.array_equal(timed_keep, keep):
            raise AssertionError("return_keeper changes keep decisions")
    print(json.dumps(dict(method=args.worker, keep=keep.tolist(), keeper=keeper.tolist(),
                         iou=iou.tolist(), warmup_seconds=warm_seconds,
                         seconds=samples, median_seconds=statistics.median(samples) if samples else None,
                         peak_process_rss_bytes=peak_rss_bytes(),
                         input_mask_bytes=masks.nbytes, weights_bytes=weights.nbytes,
                         unique_weights=np.unique(weights).tolist())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("reference", "packed"), help=argparse.SUPPRESS)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--verify-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--suite", choices=("pilot", "extended"), default="pilot")
    parser.add_argument("--packed-only", action="store_true", help="Skip float64 reference and parity; required for tube400x100")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7301)
    parser.add_argument("--output", type=Path, help="JSON output; matching .md also written")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.case == "tube400x100" and not args.packed_only:
        parser.error("tube400x100 requires --packed-only; no large float64 allocation")
    if args.worker:
        worker(args)
        return
    selected = [args.case] if args.case else list(CASES)[:3 if args.suite == "pilot" else 5]
    rows = []
    for case in selected:
        def run(method, verify=False):
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", method,
                       "--case", case, "--repeats", str(args.repeats), "--seed", str(args.seed)]
            if verify:
                command.append("--verify-only")
            if args.packed_only:
                command.append("--packed-only")
            result = subprocess.run(command, text=True, capture_output=True)
            if result.returncode:
                raise RuntimeError(f"{case}/{method}: {result.stderr}")
            return json.loads(result.stdout)
        if not args.packed_only:
            verified = {method: run(method, verify=True) for method in ("reference", "packed")}
            for field in ("keep", "keeper", "iou"):
                if verified["reference"][field] != verified["packed"][field]:
                    raise AssertionError(f"{case}: exact {field} mismatch")
        # Fresh processes keep correctness/reference allocations out of packed RSS.
        methods = ("packed",) if args.packed_only else ("reference", "packed")
        results = {method: run(method) for method in methods}
        q, frames, h, w, height, width = CASES[case]
        p = frames * h * w
        for result in results.values():
            for field in ("keep", "keeper", "iou"):
                result.pop(field)
        rows.append(dict(case=case, shape=[q, frames, h, w], native_shape=[frames, height, width],
                         exact_keep_keeper_iou=None if args.packed_only else True,
                         reference_float64_two_mask_arrays_bytes=16*q*p,
                         reference_three_pairwise_arrays_bytes=24*q*q,
                         single_bitplane_storage_bytes=q*((p+7)//8),
                         speedup=None if args.packed_only else results["reference"]["median_seconds"]/results["packed"]["median_seconds"],
                         **results))
    report = dict(schema="s4vy_packed_dedup_cpu_benchmark_v1", input_kind="synthetic moving rectangles; 25% exact duplicate masks",
                  environment=dict(python=sys.version.split()[0], numpy=np.__version__, platform=platform.platform(),
                                   cpu=next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), platform.processor()), hostname=platform.node(), blas_threads=1),
                  timing="CPU dedup only; input Boolean arrays and weights ready; packing included each call; warmup excluded",
                  memory="Measured peak RSS is isolated whole-process high-water RSS. Array byte counts are estimates, not measured peaks. Single bitplane excludes weight partitions and scratch.",
                  threshold=.7, seed=args.seed, repeats=args.repeats, rows=rows)
    lines = ["# Packed dedup CPU pilot", "", report["input_kind"] + ". No real predictions, GPU, or end-to-end timing.", "", report["timing"] + ".", "", ("No float64 reference run or speedup for packed-only stress. Parity established only on smaller paired cases, reported separately." if args.packed_only else "Exact keep, keeper, and IoU equality required before reporting each case."), "", "| Case | Float64 ms | Packed ms | Speedup | Float64 peak RSS MiB | Packed peak RSS MiB |", "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        pack = row["packed"]
        if args.packed_only:
            lines.append(f'| {row["case"]} | not run | {pack["median_seconds"]*1000:.3f} | n/a | not measured | {pack["peak_process_rss_bytes"]/2**20:.1f} |')
            continue
        ref = row["reference"]
        lines.append(f'| {row["case"]} | {ref["median_seconds"]*1000:.3f} | {pack["median_seconds"]*1000:.3f} | {row["speedup"]:.2f}x | {ref["peak_process_rss_bytes"]/2**20:.1f} | {pack["peak_process_rss_bytes"]/2**20:.1f} |')
    lines.extend(["", report["memory"], "", f'Python {report["environment"]["python"]}; NumPy {np.__version__}; CPU {report["environment"]["cpu"]}; host {platform.node()}; one BLAS thread; {args.repeats} timed calls per case.'])
    markdown = "\n".join(lines) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        args.output.with_suffix(".md").write_text(markdown)
    print(markdown, end="")


if __name__ == "__main__":
    main()
