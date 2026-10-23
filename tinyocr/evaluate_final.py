"""One frozen reserved-final evaluation; completed invocations only verify evidence."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import time
from datetime import datetime,timezone

from .baseline import Baseline
from .check_expansion import check,select_dev
from .common import ROOT,sha256,write_json
from .expand import EXPANSION
from .metrics import aggregate,counts
from .screen_labels import annotated_recall

NAMES=('v6-tiny','v6-w8-rec')
LOCK=ROOT/'results/candidate-lock.json'
START=ROOT/'results/final-start.json'
JOURNAL=ROOT/'results/final-predictions.jsonl'
REPORT=ROOT/'results/final-comparison.json'
REPEATS=3


def select_final(manifest):
    selected=[i for i in manifest['items'] if i['role']=='final']
    if not selected or len({i['id'] for i in selected})!=len(selected):
        raise ValueError('Empty/duplicate final partition')
    return selected


def score(item,text,regions):
    result={}
    if item['score_mode']=='transcription':
        result.update(reference=item['text'],normalized=counts(item['text'],text),
                      case_sensitive=counts(item['text'],text,False))
    if item.get('regions'):
        result['annotation_recovery']=annotated_recall(item['regions'],regions)
    return result


def summarize(rows):
    domains={}
    for domain in sorted({r['domain'] for r in rows}):
        selected=[r for r in rows if r['domain']==domain]
        latencies=sorted(statistics.median(r['latency_ms']) for r in selected)
        summary={'n':len(selected),'p50_ms':statistics.median(latencies),
                 'p95_ms':latencies[math.ceil(len(latencies)*.95)-1]}
        for metric in ('normalized','case_sensitive'):
            if metric in selected[0]:summary[metric]=aggregate([r[metric] for r in selected])
        if 'annotation_recovery' in selected[0]:
            total=sum(r['annotation_recovery']['annotated'] for r in selected)
            hits=sum(r['annotation_recovery']['recovered'] for r in selected)
            summary['annotation_recovery']={'annotated':total,'recovered':hits,'recall':hits/total}
        domains[domain]=summary
    return domains


def compare(items,by_model):
    summaries={name:summarize(rows) for name,rows in by_model.items()}
    checks=[]
    for domain,base in summaries[NAMES[0]].items():
        candidate=summaries[NAMES[1]][domain]
        if 'normalized' in base:
            b,c=base['normalized'],candidate['normalized']
            if b['chars']!=c['chars']:raise ValueError('Inconsistent reference denominators')
            checks.append({'domain':domain,'metric':'char_errors','baseline':b['char_errors'],
                           'candidate':c['char_errors'],'pass':c['char_errors']<=b['char_errors']})
    indexed={name:{r['id']:r for r in rows} for name,rows in by_model.items()}
    lost=[]
    for item in items:
        if item['domain']=='real-screen':
            for index,region in enumerate(item['regions']):
                b=annotated_recall([region],indexed[NAMES[0]][item['id']]['regions'])['recovered']
                c=annotated_recall([region],indexed[NAMES[1]][item['id']]['regions'])['recovered']
                if b and not c:lost.append({'id':item['id'],'region':index,'reference':region['text']})
    checks.append({'domain':'real-screen','metric':'baseline_labels_lost','count':len(lost),'lost':lost,'pass':not lost})
    return {'domains':summaries,'transfer_checks':checks,'transfer_pass':all(c['pass'] for c in checks)}


def fingerprint():
    def text_hash(path):
        return hashlib.sha256(path.read_text(encoding='utf-8').encode('utf-8')).hexdigest()
    return {'candidate_lock_sha256':sha256(LOCK),'manifest_sha256':sha256(EXPANSION),
            'dependency_lock_sha256':sha256(ROOT/'uv.lock'),
            'text_hash_encoding':'UTF-8 with normalized LF newlines',
            'protocol_sha256':text_hash(ROOT/'FINAL_VERIFICATION.md'),
            'code_sha256':{name:text_hash(ROOT/'tinyocr'/name) for name in
                          ('baseline.py','common.py','metrics.py','screen_labels.py','evaluate_final.py')},
            'models':list(NAMES),'repeats':REPEATS}


def check_frozen(models=None):
    manifest=check()
    lock=json.loads(LOCK.read_text(encoding='utf-8'))
    if lock['model']!=NAMES[1] or lock['expanded_manifest_sha256']!=sha256(EXPANSION) or lock['lock_sha256']!=sha256(ROOT/'uv.lock'):
        raise ValueError('Candidate lock/data/dependencies changed')
    if lock['acceptance_sha256']!=sha256(ROOT/'results/acceptance-v6-w8-rec.json'):
        raise ValueError('Prior acceptance report changed')
    baseline=json.loads((ROOT/'results/expanded-v6-tiny.json').read_text(encoding='utf-8'))
    expected={NAMES[0]:baseline['model_details'],NAMES[1]:{'artifacts':lock['artifacts'],'config':lock['pipeline_config']}}
    for name,spec in expected.items():
        for artifact in spec['artifacts']:
            path=ROOT/artifact['path']
            if sha256(path)!=artifact['sha256'] or path.stat().st_size!=artifact['bytes']:
                raise ValueError('Frozen model bytes changed')
        if models is not None:
            metadata=models[name].metadata()
            if metadata['config']!=spec['config']:
                raise ValueError('Pipeline differs from frozen configuration')
            if metadata['opencv_threads']!=1 or any(s['providers']!=['CPUExecutionProvider'] or s['intra_threads']!=1 or s['inter_threads']!=1 for s in metadata['sessions']):
                raise ValueError('Not one-thread CPU')
    return manifest


def read_journal(path=JOURNAL):
    rows=[]
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            rows.append(json.loads(line))  # A torn journal fails closed; never silently drop it.
    if len({(r['model'],r['id']) for r in rows})!=len(rows):
        raise ValueError('Duplicate journal entry')
    return rows


def verify_rows(items,rows):
    refs={i['id']:i for i in items}
    expected={(name,identifier) for name in NAMES for identifier in refs}
    actual={(r['model'],r['id']) for r in rows}
    if actual!=expected or len(rows)!=len(expected):
        raise ValueError('Missing, duplicate, or development prediction')
    for row in rows:
        item=refs[row['id']]
        if row['role']!='final' or any(row[k]!=item[k] for k in ('domain','task','score_mode')):
            raise ValueError('Partition/descriptor mismatch')
        scored=score(item,row['prediction'],row['regions'])
        for key in ('reference','normalized','case_sensitive','annotation_recovery'):
            if row.get(key)!=scored.get(key):raise ValueError('Reference/metric mismatch')
        if len(row['latency_ms'])!=REPEATS or not all(math.isfinite(t) and t>0 for t in row['latency_ms']):
            raise ValueError('Invalid timing evidence')


def verify():
    manifest=check_frozen()
    start=json.loads(START.read_text(encoding='utf-8'))
    if start['fingerprint']!=fingerprint():raise ValueError('Evaluation protocol or artifacts changed')
    report=json.loads(REPORT.read_text(encoding='utf-8'))
    rows=read_journal();items=select_final(manifest)
    if report['partition']!='final' or report['items_per_model']!=len(items):
        raise ValueError('Incorrect final coverage summary')
    verify_rows(items,rows)
    by_model={name:[r for r in rows if r['model']==name] for name in NAMES}
    if report['comparison']!=compare(items,by_model):raise ValueError('Incorrect final summary')
    if report['journal_sha256']!=sha256(JOURNAL) or report['start_sha256']!=sha256(START):
        raise ValueError('Evidence hash mismatch')
    print(f'Final evidence verified: {len(items)} items per model, unchanged frozen candidate; transfer_pass={report["comparison"]["transfer_pass"]}')
    return report


def evaluate(resume=False):
    if REPORT.exists():
        print('Completed final evaluation exists; verification only, no inference')
        return verify()
    manifest=check_frozen()
    items=select_final(manifest)
    if START.exists():
        if not resume:raise ValueError('Incomplete final run exists; explicit --resume required')
        if json.loads(START.read_text(encoding='utf-8'))['fingerprint']!=fingerprint():
            raise ValueError('Cannot resume a changed protocol')
    elif JOURNAL.exists():
        raise ValueError('Unowned final journal exists')
    models={name:Baseline(name) for name in NAMES}
    check_frozen(models)
    if not START.exists():
        start={'started_utc':datetime.now(timezone.utc).isoformat(),'fingerprint':fingerprint(),
               'environment':{'python':platform.python_version(),'platform':platform.platform(),
                              'packages':{p:importlib.metadata.version(p) for p in ('rapidocr','onnxruntime','numpy','pillow')}},
               'model_details':{name:model.metadata() for name,model in models.items()},
               'rule':'No model, configuration, label or eligibility changes driven by this final evaluation'}
        # Exclusive create protects an existing first-run marker.
        with START.open('x',encoding='utf-8',newline='\n') as stream:
            json.dump(start,stream,ensure_ascii=False,indent=2);stream.write('\n')
    rows=read_journal()
    allowed={(name,item['id']) for name in NAMES for item in items}
    if any((r['model'],r['id']) not in allowed for r in rows):raise ValueError('Journal contains non-final input')
    done={(r['model'],r['id']) for r in rows}
    # Warmup only on already-exposed development inputs.
    for model in models.values():
        for domain in sorted({i['domain'] for i in items}):
            item=next(i for i in select_dev(manifest) if i['domain']==domain)
            for _ in range(2):model.predict(ROOT/item['image'],item['task'])
    for index,item in enumerate(items):
        order=NAMES if index%2==0 else tuple(reversed(NAMES))
        for name in order:
            if (name,item['id']) in done:continue
            latency=[];previous=None
            for _ in range(REPEATS):
                begin=time.perf_counter()
                text,regions=models[name].predict(ROOT/item['image'],item['task'])
                latency.append((time.perf_counter()-begin)*1000)
                if previous is not None and text!=previous:raise ValueError('Nondeterministic final output')
                previous=text
            row={'model':name,'id':item['id'],'role':'final','domain':item['domain'],'task':item['task'],
                 'score_mode':item['score_mode'],'prediction':text,'regions':regions,'latency_ms':latency,
                 **score(item,text,regions)}
            with JOURNAL.open('a',encoding='utf-8',newline='\n') as stream:
                stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
            rows.append(row)
        if (index+1)%8==0:print(f'Final comparison {index+1}/{len(items)} items, both models',flush=True)
    verify_rows(items,rows)
    report={'completed_utc':datetime.now(timezone.utc).isoformat(),'partition':'final','items_per_model':len(items),
            'start_sha256':sha256(START),'journal_sha256':sha256(JOURNAL),
            'comparison':compare(items,{name:[r for r in rows if r['model']==name] for name in NAMES}),
            'limitations':['Only 8 independent screens and 8 full scenes; regions share parents',
                           'Partial-screen recovery measures recall only, not hallucinated extra text',
                           'Full-scene CER includes layout order and upstream label noise',
                           'Public-pretrained model contamination and app/template overlap remain unknown',
                           'Latency is descriptive from one comparison event, not a new speed certification',
                           'This partition is now consumed; future tuning requires a new untouched test set']}
    write_json(REPORT,report)
    return verify()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    result=verify() if args.verify_only else evaluate(args.resume)
    print(json.dumps(result['comparison'],indent=2,ensure_ascii=True))
