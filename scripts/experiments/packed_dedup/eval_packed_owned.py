#!/usr/bin/env python3
"""Experiment only: released whole-clip owned decoding versus tube dedup 0.7.

The public source stays unchanged. This is NOT the W8 framewise paper decoder.
Run in the existing GBStation benchmark container; see run_eval_packed_owned.sh.
"""
from __future__ import annotations
import argparse
import ast
import csv
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, '/workspace/upstream')
from packed_dedup import dedup_keep, warmup

ARMS = ['owned_off', 'owned_tube07']


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def require_idle():
    """Abort before starting another clip if demo usage cannot be ruled out."""
    with urllib.request.urlopen('http://10.88.7.155:7860/healthz', timeout=15) as response:
        health = json.load(response)
    if health['active_streams'] != 0 or health['batch_jobs'] != 0:
        raise RuntimeError('Demo is busy; stopping before the next GPU clip')


def weights_for(shape, out_hw):
    """Exact nearest expansion multiplicities, repeated across tube frames."""
    frames, h, w = shape
    H, W = out_hw
    rows = np.bincount(np.arange(H, dtype=np.int64) * h // H, minlength=h)
    cols = np.bincount(np.arange(W, dtype=np.int64) * w // W, minlength=w)
    return np.broadcast_to(rows[:, None] * cols[None, :], (frames, h, w))


def suppress(objectness, mask_logits, out_hw):
    """Time includes thresholding, D2H transfer, weights, packing and suppression."""
    start = now()
    scores = objectness[0].float().softmax(-1)[:, 0]
    active = (scores > .1).nonzero().flatten()
    masks = (mask_logits[0, active].float().sigmoid() > .65).cpu().numpy()
    selected_scores = scores[active].cpu().numpy()
    weights = weights_for(mask_logits.shape[2:], out_hw)
    prep = now()
    keep = dedup_keep(masks, selected_scores, .7, weights=weights)
    packed = now()
    filtered = objectness.clone()
    dropped = active[torch.as_tensor(~keep, device=active.device)]
    # Preserve original query indices and score values for every survivor.
    filtered[0, dropped, 0] = -torch.inf
    filtered[0, dropped, 1:] = 0
    finish = now()
    return filtered, {'dedup_prepare_ms': 1000*(prep-start),
                      'dedup_pack_suppress_ms': 1000*(packed-prep),
                      'dedup_apply_ms': 1000*(finish-packed),
                      'dedup_total_ms': 1000*(finish-start),
                      'eligible_queries': len(active), 'suppressed_queries': int((~keep).sum())}, (masks, selected_scores, weights)


def reference_check(data, source):
    """Bound dense source comparison to <=16 queries and <=65536 cells each."""
    masks, scores, weights = data
    count = min(len(masks), 16)
    flat = masks.reshape(len(masks), int(np.prod(masks.shape[1:])))
    cells = min(flat.shape[1], 65536)
    masks, scores, weights = flat[:count, :cells], scores[:count], weights.reshape(-1)[:cells]
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'dedup_keep')
    scope = {'np': np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), scope)
    actual = dedup_keep(masks, scores, .7, weights=weights, return_keeper=True)
    expected = scope['dedup_keep'](masks, scores, .7, weights=weights, return_keeper=True)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)
    return {'passed': True, 'queries': count, 'cells': cells, 'scope': 'bounded tensor subset; binary helper parity only'}


def score(args):
    # Reuse the original scorer unchanged; only variants and selected clips differ.
    sys.path.insert(0, str(args.scorer.parent))
    spec = importlib.util.spec_from_file_location('paired_original_scorer', args.scorer)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.ARMS = ARMS
    module.ROOT, module.INPUTS = args.results, args.inputs
    manifest = json.loads((args.results/'selection.json').read_text())
    for row in manifest:
        result = module.evaluate((row, False))
        if row['benchmark'] in ['davis', 'lvos']:
            module.evaluate((row, True))
        print(json.dumps({'scored': row['benchmark']+'/'+row['clip']}), flush=True)
    write(args.results/'metrics-complete.json', {'status': 'complete', 'clips': len(manifest)})


def measure_warm(runner, x, out_hw, decode, repeats):
    """Same input and shared forward per pair; alternate decoder order."""
    samples = []
    for rep in range(repeats):
        require_idle()
        start = now()
        prediction = runner(x)
        forwarded = now()
        obj, logits = prediction['objectness'], prediction['mask_logits']
        row = {'forward_ms': 1000*(forwarded-start)}
        for arm in (ARMS if rep % 2 == 0 else ARMS[::-1]):
            start = now()
            if arm == 'owned_tube07':
                filtered, _, binary = suppress(obj, logits, out_hw)
                del binary
                output = decode(filtered, logits, out_hw)
                del filtered
            else:
                output = decode(obj, logits, out_hw)
            row[arm+'_decode_ms'] = 1000*(now()-start)
            del output
            row[arm+'_forward_decode_ms'] = row['forward_ms']+row[arm+'_decode_ms']
        samples.append(row)
        del prediction, obj, logits
    return {'repeats': repeats, 'scope': 'warm forward + decode; excludes input loading and output writing',
            'samples': samples, 'median_ms': {key: float(np.median([r[key] for r in samples]))
                                             for key in samples[0]} if samples else {}}


def summarize(args):
    # Aggregation only: all metric definitions and pooled AP remain in the proven scorer.
    sys.path.insert(0, '/workspace/paper-eval')
    sys.path.insert(0, '/workspace/paper-eval/vendor')
    from score_clip import merge_ap_states
    selection = json.loads((args.results/'selection.json').read_text())
    cohorts = {}
    for row in selection:
        cohorts.setdefault(row['benchmark'], []).append(row)
    output, table = {}, []
    for bench, rows in cohorts.items():
        values = {arm: [] for arm in ARMS}
        ap = {arm: {} for arm in ARMS}
        warmed = []
        for row in rows:
            dst = args.results/'predictions'/bench/row['clip']
            receipt = json.loads((dst/'metrics.json').read_text())
            variants = receipt['variants']
            geometry = json.loads((dst/'geometry.json').read_text()) if bench in ['davis', 'lvos'] else None
            valid = [variants[arm]['metrics'] is not None for arm in ARMS]
            if valid[0] != valid[1]:
                raise ValueError(f'Unpaired scoreability: {bench}/{row["clip"]}')
            if valid[0]:
                for arm in ARMS:
                    metrics = dict(variants[arm]['metrics'])
                    if geometry:
                        metrics.update(geometry['variants'][arm]['metrics'])
                    values[arm].append(metrics)
                    if variants[arm].get('ap_state') is not None:
                        ap[arm][row['clip']] = variants[arm]['ap_state']
            warm = json.loads((dst/'inference.json').read_text()).get('warm_timing')
            if warm and warm['repeats']:
                warmed.append(warm['median_ms'])
        metrics = {}
        if values[ARMS[0]]:
            keys = set(values[ARMS[0]][0])
            for arm in ARMS:
                if any(set(v) != keys for v in values[arm]):
                    raise ValueError(f'Inconsistent metric keys: {bench}')
            for key in sorted(keys):
                metrics[key] = {arm: float(np.mean([v[key] for v in values[arm]])) for arm in ARMS}
        if bench == 'vipseg':
            if any(len(ap[arm]) != len(values[arm]) for arm in ARMS):
                raise ValueError('Missing VIPSeg AP state')
            pooled = {arm: merge_ap_states(ap[arm]) for arm in ARMS}
            for key in pooled[ARMS[0]]:
                metrics[key] = {arm: pooled[arm][key] for arm in ARMS}
        for key, value in metrics.items():
            value['delta'] = value[ARMS[1]]-value[ARMS[0]]
            table.append({'benchmark': bench, 'metric': key, 'clips': len(values[ARMS[0]]),
                          'off': 100*value[ARMS[0]], 'tube07': 100*value[ARMS[1]], 'delta_pp': 100*value['delta']})
        output[bench] = {'clips_selected': len(rows), 'clips_scored': len(values[ARMS[0]]),
                         'metrics': metrics, 'warm_clips': len(warmed),
                         'mean_clip_median_warm_ms': {key: float(np.mean([r[key] for r in warmed]))
                                                     for key in warmed[0]} if warmed else {}}
    write(args.results/'paired_summary.json', {'benchmarks': output, 'intervals': 'not computed',
          'metric_units': 'fraction; table CSV uses percentage points',
          'timing_scope': 'mean of per-clip warm repetition medians, forward+decode only; excludes IO'})
    with (args.results/'paired_results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['benchmark', 'metric', 'clips', 'off', 'tube07', 'delta_pp'])
        writer.writeheader()
        writer.writerows(table)
    lines = ['| Cohort | Metric | Clips | Off | Tube 0.7 | Delta (pp) |',
             '|---|---|---:|---:|---:|---:|']
    lines.extend(f'| {r["benchmark"]} | {r["metric"]} | {r["clips"]} | {r["off"]:.3f} | {r["tube07"]:.3f} | {r["delta_pp"]:+.3f} |' for r in table)
    lines += ['', 'Warm forward + decode (mean of per-clip repetition medians):', '',
              '| Cohort | Warm clips | Off (ms) | Tube 0.7 (ms) | Tube/off |',
              '|---|---:|---:|---:|---:|']
    for bench, cohort in output.items():
        timing = cohort['mean_clip_median_warm_ms']
        if timing:
            off = timing['owned_off_forward_decode_ms']
            on = timing['owned_tube07_forward_decode_ms']
            ratio = f'{on/off:.3f}' if off > 0 else 'n/a'
            lines.append(f'| {bench} | {cohort["warm_clips"]} | {off:.3f} | {on:.3f} | {ratio} |')
    if not any(cohort['warm_clips'] for cohort in output.values()):
        lines += ['', 'No warm repetitions were requested.']
    lines += ['', 'Confidence intervals not computed. Timing excludes input/output IO; '
              'not comparable to the published end-to-end speed table.', '']
    (args.results/'paired_results.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, default=Path('/workspace/results'))
    p.add_argument('--inputs', type=Path, default=Path('/workspace/inputs'))
    p.add_argument('--checkpoint', default='/workspace/weights/upstream-hf/segmenter/s4vy.pt')
    p.add_argument('--benches', default='iggt_scannet,davis,lvos,vipseg', help='comma separated; all for all manifest cohorts')
    p.add_argument('--limit-per-bench', type=int, default=0)
    p.add_argument('--long-side', default='native')
    p.add_argument('--reference', type=Path, default=Path('/workspace/experiment/reference_dedup.py'))
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--score', action='store_true')
    mode.add_argument('--summarize', action='store_true')
    p.add_argument('--warm-repeats', type=int, default=0, help='paired warm forward+decode repetitions; excludes IO')
    p.add_argument('--scorer', type=Path, default=Path('/workspace/original-analysis/score.py'))
    args = p.parse_args()
    if args.warm_repeats < 0 or args.limit_per_bench < 0:
        p.error('repeat and clip limits must be nonnegative')
    if args.score:
        score(args)
        return
    if args.summarize:
        summarize(args)
        return
    if (args.results/'runtime.json').exists():
        raise FileExistsError('Use a fresh result directory; automatic resume is disabled')
    os.environ['SEG_NO_FRAMEPOS'] = '1'
    from segomega.models import SegOmega
    from segomega.engine import setup_trainable
    from segomega.engine.decode import masks_to_idmap_owned, model_input_hw
    from segomega.engine.fast_inference import GraphRunner, optimize_for_inference
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    warmup()  # Compilation excluded from per-clip times.
    model = SegOmega(ckpt_path=None, num_queries=400, num_dec_layers=9, mask_dim=256, dec_dim=256,
                     supervise_depth=False, supervise_camera=False, use_teacher=False,
                     grad_checkpointing=True, load_pretrained=False, mask_feat_ratio=4)
    model = setup_trainable(model, mode='lora', lora_rank=256, lora_alpha=256.)
    sd = torch.load(args.checkpoint, map_location='cpu')
    missing, unexpected = model.load_state_dict(sd.get('model', sd), strict=False)
    load = {'missing_keys': list(missing), 'unexpected_keys': list(unexpected)}
    print(json.dumps({'event': 'loaded', **load}), flush=True)
    model = model.cuda().eval()
    optimization = optimize_for_inference(model)
    runner = GraphRunner(model, capacity=2)
    manifest = json.loads((args.inputs/'manifest.json').read_text())
    selected = []
    for b, d in manifest['benchmarks'].items():
        if args.benches != 'all' and b not in args.benches.split(','):
            continue
        rows = d['clips'][:args.limit_per_bench or None]
        selected += [dict(row, benchmark=b) for row in rows]
    if not selected:
        raise ValueError('No selected clips')
    selected.sort(key=lambda r: (r['frames'], r['shape'], r['benchmark'], r['clip']))
    write(args.results/'selection.json', selected)
    write(args.results/'runtime.json', {'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
          'checkpoint': args.checkpoint, 'source_commit': os.getenv('UPSTREAM_COMMIT'),
          'long_side': args.long_side, 'optimization': optimization, 'load': load, 'arms': ARMS,
          'dedup': 'whole tube, lowres mask>.65, score>.1, nearest pixel weights, IoU>.7',
          'timing': 'CUDA synchronized wall time; dedup includes threshold/D2H/weights/packing; excludes helper compilation and bounded parity check; forward includes cold compile/graph costs',
          'warm_repeats': args.warm_repeats,
          'timing_limit': 'initial pass may be cold; optional paired warm forward+decode excludes IO and is not published end-to-end timing',
          'claim': 'public whole-clip experiment, not W8 parity'})
    long_side = args.long_side if args.long_side == 'native' else int(args.long_side)
    with torch.no_grad():
        for index, row in enumerate(selected):
            require_idle()
            b, c = row['benchmark'], row['clip']
            dst = args.results/'predictions'/b/c
            start = now()
            paths = sorted((args.inputs/b/c/'frames').glob('*.png'))
            assert len(paths) == row['frames']
            origs = np.stack([np.asarray(Image.open(path).convert('RGB'), np.uint8) for path in paths])
            S, H, W = origs.shape[:3]
            Hr, Wr = model_input_hw(H, W, long_side)
            imgs = np.stack([np.asarray(Image.fromarray(f).resize((Wr, Hr), Image.BILINEAR), np.float32)/255 for f in origs])
            x = torch.from_numpy(imgs).permute(0, 3, 1, 2).cuda().unsqueeze(0)
            loaded = now()
            prediction = runner(x)
            forwarded = now()
            obj, logits = prediction['objectness'], prediction['mask_logits']
            baseline = masks_to_idmap_owned(obj, logits, (H, W))
            decoded = now()
            filtered, timing, binary = suppress(obj, logits, (H, W))
            parity = reference_check(binary, args.reference) if index == 0 else None
            del binary
            before_owned = now()
            dedup = masks_to_idmap_owned(filtered, logits, (H, W))
            finished = now()
            timing.update(load_ms=1000*(loaded-start), forward_ms=1000*(forwarded-loaded),
                          owned_off_ms=1000*(decoded-forwarded),
                          owned_after_dedup_ms=1000*(finished-before_owned), frames=S)
            timing['owned_tube07_total_ms'] = timing['dedup_total_ms']+timing['owned_after_dedup_ms']
            dst.mkdir(parents=True, exist_ok=False)
            for arm, (ids, scores) in zip(ARMS, (baseline, dedup)):
                np.savez_compressed(dst/(arm+'.npz'), idmap=ids, scores=scores)
            # Release predictions before the same CUDA graph reuses its output buffers.
            del prediction, obj, logits, filtered, baseline, dedup
            warm_timing = measure_warm(runner, x, (H, W), masks_to_idmap_owned, args.warm_repeats)
            write(dst/'inference.json', {'status': 'complete', **row, 'model_hw': [Hr, Wr],
                  'timing': timing, 'warm_timing': warm_timing, 'reference_check': parity,
                  'prediction_sha256': {arm: sha(dst/(arm+'.npz')) for arm in ARMS}})
            print(json.dumps({'complete': index+1, 'total': len(selected), 'clip': b+'/'+c, **timing}), flush=True)
            del x
            gc.collect()
            torch.cuda.empty_cache()
    write(args.results/'inference-complete.json', {'status': 'complete', 'clips': len(selected)})


if __name__ == '__main__':
    main()
