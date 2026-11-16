"""Experiment-only CPU binary mask suppression; not wired into inference.

``dedup_keep`` follows dump_streaming's greedy contract. Weighted IoU supports
nonnegative integer pixel multiplicities (the nearest-expansion paper rule).
Contiguous masks use SIMD packing; weighted popcount uses sparse correction words.
Memory is O(N*P/8 + P), including packing, rather than float64 O(N*P) matrices.
A C++ compiler builds a host-local cached library on first use. Call warmup()
before timing to exclude compilation; packing is included in dedup_keep calls.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import platform
import shlex
import subprocess
import tempfile

import numpy as np

_LIB = None


def warmup():
    """Compile/load the CPU helper once; return its cache path."""
    global _LIB
    if _LIB is not None:
        return _LIB._name
    source = Path(__file__).with_suffix(".cpp")
    flags = ["-O3", "-std=c++17", "-shared", "-fPIC"]
    if platform.machine().lower() in ("x86_64", "amd64"):
        flags.append("-mpopcnt")
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    key = hashlib.sha256(source.read_bytes() + repr((compiler, flags)).encode()).hexdigest()[:20]
    cache = Path(tempfile.gettempdir()) / f"s4vy-packed-dedup-{os.getuid()}"
    cache.mkdir(mode=0o700, exist_ok=True)
    if cache.is_symlink() or cache.stat().st_uid != os.getuid() or cache.stat().st_mode & 0o077:
        raise RuntimeError(f"unsafe native library cache: {cache}")
    target = cache / f"packed-{key}.so"
    if not target.exists():
        fd, temporary = tempfile.mkstemp(suffix=".so", dir=cache)
        os.close(fd)
        try:
            subprocess.run([*compiler, *flags, str(source), "-o", temporary], check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
    lib = ctypes.CDLL(str(target))
    ptr = ctypes.c_void_p
    i64 = ctypes.c_int64
    lib.packed_dedup.argtypes = [ptr, i64, i64, ptr, ptr, ptr, i64, ptr,
                                ctypes.c_double, ctypes.c_int, ctypes.c_int, ptr, ptr, ptr]
    lib.packed_dedup.restype = ctypes.c_int
    _LIB = lib
    return str(target)


def dedup_keep(masks, scores, iou_threshold, weights=None, return_keeper=False, priority=None):
    """Source-compatible greedy suppression, for integer-weight binary masks.

    Score descending, lower index first on ties; optional priority sorts first.
    Only already-kept queries suppress; IoU must be strictly above threshold.
    Empty masks survive. Keeper and best_iou describe suppressed queries only.
    Weighted sums are limited to 2**52 so integer arithmetic and source float64
    areas/intersections/unions are exact. Arbitrary fractional weights rejected.
    """
    masks = np.ascontiguousarray(masks, dtype=np.bool_)
    if masks.ndim < 2:
        raise ValueError("masks must have shape (N, ...pixels)")
    n = len(masks)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (n,) or not np.isfinite(scores).all():
        raise ValueError("scores must be finite with shape (N,)")
    if not np.isfinite(iou_threshold) or not 0 <= iou_threshold <= 1:
        raise ValueError("IoU threshold must be in [0, 1]")
    indices = None
    pixels = int(np.prod(masks.shape[1:]))
    if weights is None:
        lengths = np.array([pixels], dtype=np.int64)
        values = np.array([1], dtype=np.int64)
    else:
        weights = np.asarray(weights)
        if weights.shape != masks.shape[1:]:
            raise ValueError("weights must have the trailing shape of masks")
        if (not np.isfinite(weights).all() or (weights < 0).any()
                or (weights > 2**52).any() or (weights != np.floor(weights)).any()):
            raise ValueError("weights must be finite nonnegative integers <= 2**52")
        flat = np.ascontiguousarray(weights, dtype=np.int64).reshape(-1)
        maximum = int(flat.max(initial=0))
        values = (np.flatnonzero(np.bincount(flat, minlength=maximum + 1))
                  if maximum <= 4096 else np.unique(flat))
        values = values[values > 0]
        if len(values) == 1 and np.all(flat == values[0]):
            lengths = np.array([pixels], dtype=np.int64)
        else:
            positions = [np.flatnonzero(flat == value) for value in values]
            lengths = np.array([len(position) for position in positions], dtype=np.int64)
            indices = np.concatenate(positions) if positions else np.empty(0, dtype=np.int64)
        if sum(int(a) * int(b) for a, b in zip(values, lengths)) > 2**52:
            raise ValueError("total weights must be <= 2**52 for exact source parity")
    # NumPy unweighted source uses float32 sums: keep exactness within its range.
    if weights is None and pixels > 2**24:
        raise ValueError("unweighted input exceeds float32 reference's exact integer range")
    comparison_threshold = float(iou_threshold)
    if weights is None:
        # Match NumPy array/scalar promotion (including NumPy 1.x/2.x differences).
        comparison_dtype = np.result_type(np.zeros(1, np.float32), iou_threshold)
        comparison_threshold = np.asarray(iou_threshold, dtype=comparison_dtype).item()
    keys = (np.arange(n), -scores)
    if priority is not None:
        priority = np.asarray(priority, dtype=np.float64)
        if priority.shape != (n,) or not np.isfinite(priority).all():
            raise ValueError("priority must be finite with shape (N,)")
        keys += (-priority,)
    order = np.ascontiguousarray(np.lexsort(keys), dtype=np.int64)
    keep = np.zeros(n, dtype=np.bool_)
    keeper = np.full(n, -1, dtype=np.int64)
    best_iou = np.zeros(n, dtype=np.float64)
    if n:
        warmup()
        status = _LIB.packed_dedup(masks.ctypes.data, n, pixels,
                                  None if indices is None else indices.ctypes.data,
                                  lengths.ctypes.data, values.ctypes.data, len(values), order.ctypes.data,
                                  comparison_threshold, int(return_keeper), int(weights is None), keep.ctypes.data,
                                  keeper.ctypes.data, best_iou.ctypes.data)
        if status:
            raise RuntimeError("packed dedup native allocation/execution failed")
    return (keep, keeper, best_iou) if return_keeper else keep
