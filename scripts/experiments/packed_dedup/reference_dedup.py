"""Dense experiment reference, retained for exact parity checks only.

Extracted unchanged from s4vy-streaming scripts/paper_eval/dump_streaming.py.
"""
import numpy as np


def dedup_keep(masks, scores, iou_threshold, weights=None, return_keeper=False, priority=None):
    """Greedy suppression. masks: bool [N, P]; scores: [N]. Returns bool keep[N].

    Order: score descending, lower index first on ties; with priority[N] (incumbency), priority
    descending first, then score, then index. A query is dropped when its
    mask IoU with any already kept query is > iou_threshold. Empty masks are kept
    (IoU with anything is 0); they never own pixels anyway. weights (optional, same
    trailing shape as one mask): per-cell pixel counts, used by the raw rule whose
    masks are low-res cells upsampled with nearest. return_keeper=True also returns
    keeper[N] (index of the kept query with the highest IoU for a dropped query, -1
    for kept ones) and best_iou[N].
    """
    masks = np.asarray(masks, bool)
    n = len(masks)
    keep = np.zeros(n, bool)
    if n == 0:
        return (keep, np.zeros(0, int), np.zeros(0)) if return_keeper else keep
    if weights is None:
        flat = masks.reshape(n, -1).astype(np.float32)
        area = flat.sum(1)
        inter = flat @ flat.T
    else:
        flat = masks.reshape(n, -1).astype(np.float64)
        weighted = flat * np.asarray(weights, np.float64).reshape(1, -1)
        area = weighted.sum(1)
        inter = weighted @ flat.T
    union = area[:, None] + area[None, :] - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    if priority is None:
        order = np.lexsort((np.arange(n), -np.asarray(scores, np.float64)))
    else:
        order = np.lexsort((np.arange(n), -np.asarray(scores, np.float64), -np.asarray(priority, np.float64)))
    kept = []
    keeper, best_iou = np.full(n, -1), np.zeros(n)
    for q in order:
        if not kept or not (iou[q, kept] > iou_threshold).any():
            kept.append(q)
        elif return_keeper:
            j = int(np.argmax(iou[q, kept]))
            keeper[q], best_iou[q] = kept[j], iou[q, kept[j]]
    keep[kept] = True
    return (keep, keeper, best_iou) if return_keeper else keep
