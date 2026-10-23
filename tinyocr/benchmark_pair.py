"""Interleaved CPU timing, with alternating model order, on development only."""
import argparse
import json
import math
import random
import statistics
import time

from .baseline import Baseline
from .check_expansion import check,select_dev
from .common import MANIFEST,ROOT,sha256,write_json
from .expand import EXPANSION
from .validate import validate


def benchmark(name,passes=9,reverse_load=False):
    if passes != 9:
        raise ValueError('Acceptance uses exactly nine alternating passes')
    pilot=validate();expanded=check()
    items=[{**i,'suite':'pilot'} for i in pilot['items']]+[{**i,'suite':'expanded'} for i in select_dev(expanded)]
    specifications=[('baseline','v6-tiny'),('candidate',name)]
    if reverse_load:specifications.reverse()
    models={label:Baseline(model_name) for label,model_name in specifications}
    # Turn off repeated empty-detection warnings during measured runs, equally for both.
    from rapidocr.utils.log import logger
    logger.setLevel('ERROR')
    for model in models.values():
        for domain in sorted({i['domain'] for i in items}):
            item=next(i for i in items if i['domain']==domain)
            for _ in range(2):model.predict(ROOT/item['image'],item['task'])
    expected={}
    for label,model_name in [('baseline','v6-tiny'),('candidate',name)]:
        for suite,prefix in [('pilot',''),('expanded','expanded-')]:
            data=json.loads((ROOT/f'runs/{prefix}{model_name}.json').read_text(encoding='utf-8'))
            for p in data['predictions']:expected[(label,suite,p['id'])]=p['prediction']
    timings=[]
    for round_index in range(passes):
        ordered=list(items);random.Random(8765+round_index).shuffle(ordered)
        for index,item in enumerate(ordered):
            order=('baseline','candidate') if (index+round_index)%2==0 else ('candidate','baseline')
            record={'pass':round_index,'id':item['id'],'suite':item['suite'],'domain':item['domain'],'first':order[0]}
            for label in order:
                start=time.perf_counter()
                text,_=models[label].predict(ROOT/item['image'],item['task'])
                record[label+'_ms']=(time.perf_counter()-start)*1000
                if text!=expected[(label,item['suite'],item['id'])]:
                    raise ValueError(f'Prediction changed during timing: {label} {item["id"]}')
            timings.append(record)
        print(f'Paired pass {round_index+1}/{passes}: {len(items)} inputs, final set excluded',flush=True)
    domains={}
    for domain in sorted({i['domain'] for i in items}):
        rows=[r for r in timings if r['domain']==domain]
        rounds=[]
        for round_index in range(passes):
            selected=[r for r in rows if r['pass']==round_index]
            result={}
            for label in models:
                values=sorted(r[label+'_ms'] for r in selected)
                result[label+'_p50_ms']=statistics.median(values)
                result[label+'_p95_ms']=values[math.ceil(len(values)*.95)-1]
            for metric in ('p50','p95'):
                result[metric+'_ratio']=result[f'candidate_{metric}_ms']/result[f'baseline_{metric}_ms']
            rounds.append(result)
        domains[domain]={'rounds':rounds,
            'median_p50_ratio':statistics.median(r['p50_ratio'] for r in rounds),
            'median_p95_ratio':statistics.median(r['p95_ratio'] for r in rounds),
            'consistent_p50_slowdown':all(r['p50_ratio']>1 for r in rounds),
            'consistent_p95_slowdown':all(r['p95_ratio']>1 for r in rounds)}
    report={'model':name,'passes':passes,'inputs_per_pass':len(items),'domains':domains,'reverse_load':reverse_load,
            'method':'One warm baseline/candidate pair per input, shuffled each pass; alternating order; 2 warmups per domain/model',
            'screening_rule':'Flag slowdown if any domain p50 or p95 is higher on every pass; all ratios retained; no phone/general-runtime claim',
            'no_consistent_slowdown':not any(r['consistent_p50_slowdown'] or r['consistent_p95_slowdown'] for r in domains.values()),
            'manifest_sha256':{'pilot':sha256(MANIFEST),'expanded':sha256(EXPANSION)},
            'lock_sha256':sha256(ROOT/'uv.lock'),
            'models':{key:model.metadata() for key,model in models.items()},'timings':timings}
    suffix='-reverse' if reverse_load else ''
    path=ROOT/f'runs/paired-{name}{suffix}.json';write_json(path,report)
    summary={k:v for k,v in report.items() if k!='timings'};summary['full_report_sha256']=sha256(path)
    write_json(ROOT/f'results/paired-{name}{suffix}.json',summary)
    verify_pair(name,reverse_load)
    print(json.dumps({d:{k:v for k,v in r.items() if k!='rounds'} for d,r in domains.items()},indent=2))


def verify_pair(name,reverse_load=False):
    suffix='-reverse' if reverse_load else ''
    report=json.loads((ROOT/f'runs/paired-{name}{suffix}.json').read_text(encoding='utf-8'))
    if report['passes']!=9:
        raise ValueError('Acceptance requires nine paired passes')
    if report['manifest_sha256']!={'pilot':sha256(MANIFEST),'expanded':sha256(EXPANSION)} or report['lock_sha256']!=sha256(ROOT/'uv.lock'):
        raise ValueError('Timing inputs or dependencies changed')
    pilot=json.loads(MANIFEST.read_text(encoding='utf-8'))
    expansion=json.loads(EXPANSION.read_text(encoding='utf-8'))
    items={('pilot',i['id']):i for i in pilot['items']}
    items.update({('expanded',i['id']):i for i in select_dev(expansion)})
    rows=report['timings']
    expected={(n,suite,identifier) for n in range(9) for suite,identifier in items}
    keys={(r['pass'],r['suite'],r['id']) for r in rows}
    if len(rows)!=len(expected) or keys!=expected:
        raise ValueError('Missing, duplicate, or final timing samples')
    for row in rows:
        if row['domain']!=items[row['suite'],row['id']]['domain']:
            raise ValueError('Timing domain mismatch')
        if not all(math.isfinite(row[k]) and row[k]>0 for k in ('baseline_ms','candidate_ms')):
            raise ValueError('Invalid paired duration')
    for model in report['models'].values():
        if model['opencv_threads']!=1:raise ValueError('OpenCV thread mismatch')
        for session in model['sessions']:
            if session['providers']!=['CPUExecutionProvider'] or session['intra_threads']!=1 or session['inter_threads']!=1:
                raise ValueError('CPU/session thread mismatch')
        for artifact in model['artifacts']:
            if sha256(ROOT/artifact['path'])!=artifact['sha256']:raise ValueError('Timing model changed')
    if set(report['domains'])!={i['domain'] for i in items.values()}:
        raise ValueError('Missing/extra timing domains')
    for domain,summary in report['domains'].items():
        if len(summary['rounds'])!=9:raise ValueError('Missing timing rounds')
        for n,result in enumerate(summary['rounds']):
            selected=[r for r in rows if r['domain']==domain and r['pass']==n]
            for label in ('baseline','candidate'):
                times=sorted(r[label+'_ms'] for r in selected)
                if result[label+'_p50_ms']!=statistics.median(times) or result[label+'_p95_ms']!=times[math.ceil(len(times)*.95)-1]:
                    raise ValueError('Incorrect latency percentiles')
            for metric in ('p50','p95'):
                if result[metric+'_ratio']!=result['candidate_'+metric+'_ms']/result['baseline_'+metric+'_ms']:
                    raise ValueError('Incorrect latency ratio')
        for metric in ('p50','p95'):
            ratios=[r[metric+'_ratio'] for r in summary['rounds']]
            if summary['median_'+metric+'_ratio']!=statistics.median(ratios) or summary['consistent_'+metric+'_slowdown']!=all(r>1 for r in ratios):
                raise ValueError('Incorrect slowdown summary')
    expected_pass=not any(r['consistent_p50_slowdown'] or r['consistent_p95_slowdown'] for r in report['domains'].values())
    if report['no_consistent_slowdown']!=expected_pass:raise ValueError('Incorrect speed verdict')
    print(f'Paired evidence verified: {len(rows)} pairs, nine passes, no final inputs')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=['v6-w8','v6-w16','v6-w8-rec'],required=True)
    parser.add_argument('--passes',type=int,default=9)
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--reverse-load',action='store_true')
    args=parser.parse_args()
    verify_pair(args.model,args.reverse_load) if args.verify_only else benchmark(args.model,args.passes,args.reverse_load)
