"""Checkpointed two-model public benchmark, with native word geometry and fixed scoring."""
import argparse
from collections import Counter
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time

from .common import ROOT, reading_order, write_json
from .intake import sha256
from .metrics import aggregate, counts
from .prepare_benchmark import MANIFEST, SOURCES, VENDOR, verify as verify_data

NAMES = ('v6-tiny', 'v6-w8-rec')
RUNS = Path('runs/benchmarks-v1')
REPORT = Path('results/benchmark-comparison.json')
ADAPTER = Path('results/benchmark-adapter-check.json')
PROTOCOL = Path('BENCHMARK_VERIFICATION.md')


def text_hash(path):
    return hashlib.sha256(path.read_text(encoding='utf-8').encode('utf-8')).hexdigest()


def fingerprint():
    files = ['tinyocr/' + name for name in (
        'baseline.py', 'common.py', 'metrics.py', 'intake.py', 'prepare_benchmark.py', 'benchmark.py')]
    source = json.loads((ROOT / SOURCES).read_text(encoding='utf-8'))
    for name in VENDOR:
        relative = 'third_party/hiertext/evaluator/' + name
        if sha256(ROOT / relative) != source['files'][relative]['sha256']:
            raise ValueError('Pinned official evaluator source changed')
    rights = json.loads((ROOT / 'results/benchmark-rights.json').read_text(encoding='utf-8'))['textocr_publisher_bundle']
    if sha256(ROOT / rights['path']) != rights['sha256']:
        raise ValueError('Publisher rights snapshot changed')
    return {'manifest_sha256': sha256(ROOT / MANIFEST), 'sources_sha256': sha256(ROOT / SOURCES),
            'rights_sha256': sha256(ROOT / 'results/benchmark-rights.json'),
            'protocol_sha256': text_hash(ROOT / PROTOCOL), 'uv_lock_sha256': sha256(ROOT / 'uv.lock'),
            'candidate_lock_sha256': sha256(ROOT / 'results/candidate-lock.json'),
            'overlay_sha256': text_hash(ROOT / 'benchmark-requirements.txt'),
            'code_sha256_lf': {name: text_hash(ROOT / name) for name in files},
            'vendor_sha256': {name: sha256(ROOT / 'third_party/hiertext/evaluator' / name) for name in VENDOR}}


def validate_polygons(regions):
    for item in regions:
        poly = item['polygon']
        if (not isinstance(item['text'], str) or len(poly) < 3 or
                any(len(p) != 2 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in p) for p in poly)):
            raise ValueError('Invalid prediction polygon/text')


def predict(model, path, words=True):
    from rapidocr.main import flatten_word_results

    result = model.engine(str(path), use_det=True, use_cls=False, use_rec=True,
                          return_word_box=words, return_single_char_box=False)
    lines = []
    if result.boxes is not None:
        for text, points in zip(result.txts, result.boxes):
            poly = [[float(x), float(y)] for x, y in points]
            xs, ys = zip(*poly)
            lines.append({'text': text, 'polygon': poly, 'box': [min(xs), min(ys), max(xs), max(ys)]})
    lines = reading_order(lines)
    word_regions = [{'text': text, 'polygon': [[float(x), float(y)] for x, y in points]}
                    for text, _, points in flatten_word_results(getattr(result, 'word_results', None))] if words else []
    validate_polygons(lines)
    validate_polygons(word_regions)
    return {'text': '\n'.join(r['text'] for r in lines), 'lines': lines, 'words': word_regions}


def frozen_metadata(model, name):
    meta = model.metadata()
    validate_frozen_metadata(meta, name)
    return meta


def validate_frozen_metadata(meta, name):
    expected = (json.loads((ROOT / 'results/expanded-v6-tiny.json').read_text(encoding='utf-8'))['model_details'] if name == NAMES[0]
                else json.loads((ROOT / 'results/candidate-lock.json').read_text(encoding='utf-8')))
    config = expected.get('pipeline_config', expected.get('config'))
    if meta['config'] != config or meta['artifacts'] != expected['artifacts']:
        raise ValueError('Model/configuration differs from frozen candidate comparison')
    if meta['opencv_threads'] != 1 or any(s['providers'] != ['CPUExecutionProvider'] or
                                         s['intra_threads'] != 1 or s['inter_threads'] != 1 for s in meta['sessions']):
        raise ValueError('Benchmark requires one-thread CPU inference')


def environment():
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'cpu': platform.processor(), 'packages': {p: importlib.metadata.version(p) for p in
                ('rapidocr', 'onnxruntime', 'numpy', 'pillow', 'shapely', 'absl-py')}}


def check_adapter():
    from .baseline import Baseline

    pilot = json.loads((ROOT / 'eval/manifest.json').read_text(encoding='utf-8'))
    expansion = json.loads((ROOT / 'eval/expansion-region-v2.json').read_text(encoding='utf-8'))
    fixtures = [next(i for i in pilot['items'] if i['domain'] == 'receipt-page'),
                next(i for i in pilot['items'] if i['domain'] == 'synthetic-screen'),
                next(i for i in expansion['items'] if i['domain'] == 'full-scene' and i['role'] == 'dev')]
    results = []
    for name in NAMES:
        model = Baseline(name)
        frozen_metadata(model, name)
        for item in fixtures:
            ordinary = predict(model, ROOT / item['image'], words=False)
            adapted = predict(model, ROOT / item['image'], words=True)
            if adapted['text'] != ordinary['text'] or adapted['lines'] != ordinary['lines']:
                raise ValueError('Word adapter changed line predictions')
            if ordinary['lines'] and not adapted['words']:
                raise ValueError('Word adapter returned no words on development fixture')
            results.append({'model': name, 'id': item['id'], 'image_sha256': sha256(ROOT / item['image']),
                            'unchanged_lines': True, 'line_count': len(adapted['lines']),
                            'word_count': len(adapted['words'])})
        del model
    write_json(ROOT / ADAPTER, {'schema': 1, 'adapter_code_sha256_lf': text_hash(Path(__file__)),
                               'fixtures': results, 'new_benchmark_images_used': 0})
    print('Word output adapter verified on six model/development-fixture pairs', flush=True)


def peak_memory():
    if os.name == 'nt':
        class MemoryCounters(ctypes.Structure):
            _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong),
                        *[(name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
                          'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                          'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]]
        values = MemoryCounters()
        values.cb = ctypes.sizeof(values)
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(values), values.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(values.PeakWorkingSetSize)
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if platform.system() == 'Darwin' else peak * 1024)


def journal_rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if not line.endswith('\n') or not line.strip():
                raise ValueError('Torn or blank prediction journal line')
            yield json.loads(line)


def validate_row(row, references, name):
    ref = references.get(row['id'])
    if ref is None or row['model'] != name or row['source'] != ref['source']:
        raise ValueError('Unknown prediction identity')
    if not isinstance(row['latency_ms'], (int, float)) or not math.isfinite(row['latency_ms']) or row['latency_ms'] <= 0:
        raise ValueError('Invalid benchmark latency')
    validate_polygons(row['lines'])
    validate_polygons(row['words'])
    if row['text'] != '\n'.join(r['text'] for r in row['lines']):
        raise ValueError('Journal text differs from line outputs')


def check_journal(path, references, name, complete=False):
    seen = set()
    if Path(path).exists():
        for row in journal_rows(path):
            validate_row(row, references, name)
            if row['id'] in seen:
                raise ValueError('Duplicate prediction journal ID')
            seen.add(row['id'])
    if complete and seen != set(references):
        raise ValueError('Incomplete benchmark prediction coverage')
    return seen


def run_model(name):
    from .baseline import Baseline

    manifest = verify_data()
    refs = {i['id']: i for i in manifest['items']}
    current = fingerprint()
    start_path, finish_path = ROOT / RUNS / (name + '-start.json'), ROOT / RUNS / (name + '-finish.json')
    journal = ROOT / RUNS / (name + '.jsonl')
    if finish_path.exists():
        verify_model(name, manifest, current)
        print(name + ': completed journal verified; no inference', flush=True)
        return
    adapter = json.loads((ROOT / ADAPTER).read_text(encoding='utf-8'))
    if adapter['adapter_code_sha256_lf'] != text_hash(Path(__file__)) or len(adapter['fixtures']) != 6:
        raise ValueError('Run the current adapter check before benchmarking')
    if start_path.exists():
        started = json.loads(start_path.read_text(encoding='utf-8'))
        if started['fingerprint'] != current or started['environment'] != environment():
            raise ValueError('Cannot resume changed benchmark configuration')
    elif journal.exists():
        raise ValueError('Prediction journal exists without start evidence')
    seen = check_journal(journal, refs, name)
    begin = time.perf_counter()
    model = Baseline(name)
    init_ms = (time.perf_counter() - begin) * 1000
    metadata = frozen_metadata(model, name)
    if not start_path.exists():
        started = {'schema': 1, 'model': name, 'fingerprint': current,
                   'adapter_check_sha256': sha256(ROOT / ADAPTER), 'model_details': metadata,
                   'started_at_utc': datetime.now(timezone.utc).isoformat(),
                   'environment': environment(),
                   'benchmark_output_options': {'return_word_box': True, 'return_single_char_box': False},
                   'timing': {'repeats': 1, 'warmups': 2, 'includes_word_box_postprocessing': True}}
        write_json(start_path, started)
    elif metadata != started['model_details']:
        raise ValueError('Resumed model metadata changed')
    pilot = json.loads((ROOT / 'eval/manifest.json').read_text(encoding='utf-8'))
    warmup = next(i for i in pilot['items'] if i['domain'] == 'receipt-page')
    for _ in range(2):
        predict(model, ROOT / warmup['image'])
    segment = {'pid': os.getpid(), 'initialization_ms': init_ms, 'previous_items': len(seen),
               'started_at_utc': datetime.now(timezone.utc).isoformat()}
    journal.parent.mkdir(parents=True, exist_ok=True)
    with journal.open('a', encoding='utf-8', newline='\n') as stream:
        for index, item in enumerate(manifest['items']):
            if item['id'] in seen:
                continue
            tick = time.perf_counter()
            result = predict(model, ROOT / item['image'])
            elapsed = (time.perf_counter() - tick) * 1000
            row = {'id': item['id'], 'model': name, 'source': item['source'], 'latency_ms': elapsed, **result}
            validate_row(row, refs, name)
            stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
            if (index + 1) % 100 == 0:
                print(f'{name}: {index + 1}/{len(refs)} images', flush=True)
    segment['peak_working_set_bytes'] = peak_memory()
    segment['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
    check_journal(journal, refs, name, complete=True)
    write_json(finish_path, {'schema': 1, 'model': name, 'fingerprint': current,
                            'journal_sha256': sha256(journal), 'start_sha256': sha256(start_path),
                            'items': len(refs), 'completion_process': segment,
                            'memory_complete_for_run': segment['previous_items'] == 0,
                            'memory_scope': 'Peak process working set, including imports/manifest/model and output adaptation; excludes offline scoring.'})
    print(name + ': full prediction journal complete', flush=True)


def verify_model(name, manifest, current):
    start_path, finish_path = ROOT / RUNS / (name + '-start.json'), ROOT / RUNS / (name + '-finish.json')
    started, finished = json.loads(start_path.read_text(encoding='utf-8')), json.loads(finish_path.read_text(encoding='utf-8'))
    if started['fingerprint'] != current or finished['fingerprint'] != current:
        raise ValueError('Benchmark run fingerprint changed')
    if finished['start_sha256'] != sha256(start_path) or started['adapter_check_sha256'] != sha256(ROOT / ADAPTER):
        raise ValueError('Start/adapter evidence changed')
    journal = ROOT / RUNS / (name + '.jsonl')
    if finished['journal_sha256'] != sha256(journal):
        raise ValueError('Prediction journal checksum mismatch')
    check_journal(journal, {i['id']: i for i in manifest['items']}, name, complete=True)
    meta = started['model_details']
    validate_frozen_metadata(meta, name)
    if meta['opencv_threads'] != 1 or len(meta['sessions']) != 2:
        raise ValueError('Invalid inference runtime')
    for session in meta['sessions']:
        if session['providers'] != ['CPUExecutionProvider'] or session['intra_threads'] != 1 or session['inter_threads'] != 1:
            raise ValueError('Not single-thread CPU')
    for item in meta['artifacts']:
        if sha256(ROOT / item['path']) != item['sha256'] or (ROOT / item['path']).stat().st_size != item['bytes']:
            raise ValueError('Model artifact changed')
    process = finished['completion_process']
    if (finished['items'] != len(manifest['items']) or process['peak_working_set_bytes'] <= 0 or
            not math.isfinite(process['initialization_ms']) or process['initialization_ms'] <= 0):
        raise ValueError('Invalid process/coverage evidence')
    return started, finished


def make_scorer():
    from third_party.hiertext.evaluator.evaluator import HierTextEvaluator, TextBoxRep
    return HierTextEvaluator(text_box_type=TextBoxRep.POLY, evaluate_text=True)


def score_regions(scorer, reference, predicted):
    import numpy as np
    data = {'gt_boxes': [np.asarray(r['polygon']) for r in reference],
            'gt_weights': np.asarray([1.0 if r['legible'] else 0.0 for r in reference]),
            'gt_texts': np.asarray([r['text'] for r in reference]),
            'detection_boxes': [np.asarray(r['polygon']) for r in predicted],
            'pred_texts': np.asarray([r['text'] for r in predicted])}
    return {k: float(v) for k, v in scorer.evaluate_one_image(data).items()}


def summarize_source(scorer, entries):
    totals = {k: sum(e['official_counts'][k] for e in entries) for k in entries[0]['official_counts']}
    latency = sorted(e['latency_ms'] for e in entries)
    complete = [e for e in entries if e['complete_reference']]
    return {'images': len(entries), 'counts': totals, 'detection_and_e2e': scorer.evaluate(totals),
            'latency_ms': {'p50': statistics.median(latency), 'p95': latency[math.ceil(.95 * len(latency)) - 1]},
            'row_order_diagnostic': {'complete_reference_pages': len(complete),
                'pages_excluded_for_illegible_regions': len(entries) - len(complete),
                'normalized': aggregate([e['normalized'] for e in complete]) if complete else None,
                'case_sensitive': aggregate([e['case_sensitive'] for e in complete]) if complete else None}}


def score_all():
    manifest = verify_data()
    current = fingerprint()
    refs = {i['id']: i for i in manifest['items']}
    scorer = make_scorer()
    result = {'schema': 1, 'fingerprint': current, 'selected_counts': manifest['selected_counts'],
              'excluded_image_count': len(manifest['excluded']),
              'cross_source_shared_ids': manifest['shared_source_ids'],
              'protocols': {'sroie': 'Shared HierText polygon protocol on text regions; not official SROIE leaderboard',
                            'textocr': 'Shared HierText polygon protocol on words; not official TextOCR leaderboard',
                            'hiertext': 'Official HierText word polygon evaluator on overlap-filtered validation subset'},
              'models': {}}
    for name in NAMES:
        started, finished = verify_model(name, manifest, current)
        by_source = {s: [] for s in manifest['selected_counts']}
        for index, row in enumerate(journal_rows(ROOT / RUNS / (name + '.jsonl'))):
            item = refs[row['id']]
            ref = json.loads((ROOT / item['annotations']).read_text(encoding='utf-8'))
            selected = row['lines'] if item['source'] == 'sroie' else row['words']
            entry = {'latency_ms': row['latency_ms'], 'complete_reference': ref['complete_reference'],
                     'official_counts': score_regions(scorer, ref['regions'], selected)}
            if ref['complete_reference']:
                entry['normalized'] = counts(ref['row_order_text'], row['text'])
                entry['case_sensitive'] = counts(ref['row_order_text'], row['text'], False)
            by_source[item['source']].append(entry)
            if (index + 1) % 250 == 0:
                print(f'Scored {name}: {index + 1}/{len(refs)}', flush=True)
        result['models'][name] = {'sources': {s: summarize_source(scorer, rows) for s, rows in by_source.items()},
                                  'model_artifact_bytes': started['model_details']['artifact_bytes'],
                                  'process': finished['completion_process'],
                                  'memory_complete_for_run': finished['memory_complete_for_run'],
                                  'journal_sha256': finished['journal_sha256'],
                                  'environment': started['environment']}
    baseline, candidate = [result['models'][name] for name in NAMES]
    if baseline['environment'] != candidate['environment'] or baseline['process']['pid'] == candidate['process']['pid']:
        raise ValueError('Models must use matched environments and distinct fresh processes')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--model', choices=NAMES)
    mode.add_argument('--check-adapter', action='store_true')
    mode.add_argument('--score', action='store_true')
    mode.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.check_adapter:
        check_adapter()
    elif args.model:
        run_model(args.model)
    else:
        result = score_all()
        if args.verify_only or (ROOT / REPORT).exists():
            if json.loads((ROOT / REPORT).read_text(encoding='utf-8')) != result:
                raise ValueError('Benchmark comparison report mismatch')
            print('Full benchmark evidence and scoring verified; no inference')
        else:
            write_json(ROOT / REPORT, result)
            print('Full benchmark comparison saved')


if __name__ == '__main__':
    main()
