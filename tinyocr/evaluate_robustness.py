"""Measure fixed development diagnostics, preserving a completed comparison."""
import argparse
import hashlib
import json
import math
import time

from .baseline import Baseline
from .common import ROOT, sha256, write_json
from .evaluate_final import NAMES, check_frozen, score, summarize
from .robustness import PATH, prepare

REPORT=ROOT/'results/robustness-v1.json'
REPEATS=3


def provenance():
    files=['tinyocr/robustness.py','tinyocr/evaluate_robustness.py','ROBUSTNESS_VERIFICATION.md']
    return {'manifest_sha256':sha256(PATH),'candidate_lock_sha256':sha256(ROOT/'results/candidate-lock.json'),
            'dependency_lock_sha256':sha256(ROOT/'uv.lock'),
            'lf_text_sha256':{p:hashlib.sha256((ROOT/p).read_text(encoding='utf-8').encode()).hexdigest() for p in files}}


def summaries(rows):
    result={name:summarize([r for r in rows if r['model']==name]) for name in NAMES}
    for name in NAMES:
        blanks=[r for r in rows if r['model']==name and r['domain']=='procedural-negative']
        result[name]['procedural-negative']['images_with_text']=sum(bool(r['prediction'].strip()) for r in blanks)
    return result


def verify_rows(items,rows):
    refs={i['id']:i for i in items}
    expected={(name,i) for name in NAMES for i in refs}
    if len(rows)!=len(expected) or {(r['model'],r['id']) for r in rows}!=expected:
        raise ValueError('Missing, duplicate, or unknown prediction')
    for row in rows:
        ref=refs[row['id']]
        if row['role']!='dev' or ref['role']!='dev' or row['domain']!=ref['domain'] or row['task']!='page':
            raise ValueError('Wrong role/domain/task')
        for key,value in score(ref,row['prediction'],row['regions']).items():
            if row.get(key)!=value:raise ValueError('Wrong reference or error counter')
        if len(row['latency_ms'])!=REPEATS or not all(math.isfinite(t) and t>0 for t in row['latency_ms']):
            raise ValueError('Invalid latency')


def verify():
    manifest=prepare(restore=False)
    check_frozen()
    report=json.loads(REPORT.read_text(encoding='utf-8'))
    if report['provenance']!=provenance() or report['partition']!='dev' or report['repeats']!=REPEATS:
        raise ValueError('Changed diagnostic provenance')
    verify_rows(manifest['items'],report['predictions'])
    if report['domains']!=summaries(report['predictions']):
        raise ValueError('Wrong diagnostic summary')
    for metadata in report['model_details'].values():
        for artifact in metadata['artifacts']:
            if sha256(ROOT/artifact['path'])!=artifact['sha256']:raise ValueError('Changed model')
    print('Verified robustness evidence: 80 items/model; all failures retained')
    return report


def evaluate():
    if REPORT.exists():
        print('Completed diagnostic exists; verifying without inference')
        return verify()
    manifest=prepare()
    models={name:Baseline(name) for name in NAMES}
    check_frozen(models)
    metadata={name:model.metadata() for name,model in models.items()}
    for model in models.values():
        for _ in range(2):model.predict(ROOT/manifest['items'][0]['image'],'page')
    rows=[]
    for index,item in enumerate(manifest['items']):
        for name in (NAMES if index%2==0 else reversed(NAMES)):
            latency=[];previous=None
            for _ in range(REPEATS):
                start=time.perf_counter()
                text,regions=models[name].predict(ROOT/item['image'],'page')
                latency.append((time.perf_counter()-start)*1000)
                if previous is not None and previous!=text:raise ValueError('Nondeterministic output')
                previous=text
            rows.append({'model':name,'id':item['id'],'role':'dev','task':'page','domain':item['domain'],
                         'prediction':text,'regions':regions,'latency_ms':latency,
                         **score(item,text,regions)})
        if (index+1)%10==0:print(f'Robustness {index+1}/{len(manifest["items"])} items, both models',flush=True)
    verify_rows(manifest['items'],rows)
    report={'partition':'dev','provenance':provenance(),'repeats':REPEATS,
            'model_details':metadata,'domains':summaries(rows),'predictions':rows,
            'limitations':manifest['limitations']+['Single-event desktop timing is descriptive, not a speed gate']}
    write_json(REPORT,report)
    return verify()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    report=verify() if args.verify_only else evaluate()
    print(json.dumps(report['domains'],indent=2))
