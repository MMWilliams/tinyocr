"""Strict development-only accuracy comparison; never evaluates final data."""
import argparse
import json

from .common import ROOT, sha256, write_json
from .expand import EXPANSION
from .screen_labels import annotated_recall
from .verify_run import verify
from .evaluate_expansion import verify as verify_expanded


def compare(name):
    for model in ('v6-tiny',name):
        verify(model);verify_expanded(model)
    checks=[]
    for prefix in ('','expanded-'):
        base=json.loads((ROOT/f'runs/{prefix}v6-tiny.json').read_text(encoding='utf-8'))
        candidate=json.loads((ROOT/f'runs/{prefix}{name}.json').read_text(encoding='utf-8'))
        if base['manifest_sha256']!=candidate['manifest_sha256']:
            raise ValueError('Different data manifests')
        for domain,b in base['domains'].items():
            c=candidate['domains'][domain]
            if 'normalized' in b:
                left,right=b['normalized'],c['normalized']
                if left['chars']!=right['chars'] or left['n']!=right['n']:
                    raise ValueError('Different references/counts')
                checks.append({'suite':prefix or 'pilot','domain':domain,'metric':'char_errors',
                               'baseline':left['char_errors'],'candidate':right['char_errors'],
                               'pass':right['char_errors']<=left['char_errors']})
        if prefix:
            refs=json.loads(EXPANSION.read_text(encoding='utf-8'))
            bp={p['id']:p for p in base['predictions']};cp={p['id']:p for p in candidate['predictions']}
            lost=[]
            for item in refs['items']:
                if item['role']=='dev' and item['domain']=='real-screen':
                    for index,region in enumerate(item['regions']):
                        before=annotated_recall([region],bp[item['id']]['regions'])['recovered']
                        after=annotated_recall([region],cp[item['id']]['regions'])['recovered']
                        if before and not after:lost.append({'id':item['id'],'region':index,'text':region['text']})
            checks.append({'suite':prefix,'domain':'real-screen','metric':'previously_recovered_labels_lost',
                           'baseline':0,'candidate':len(lost),'pass':not lost,'lost':lost})
    export=json.loads((ROOT/f'results/export-{name}.json').read_text(encoding='utf-8'))
    actual=sum((ROOT/a['path']).stat().st_size for a in export['artifacts'])
    for a in export['artifacts']:
        if sha256(ROOT/a['path'])!=a['sha256']:raise ValueError('Export artifact changed')
    checks.append({'metric':'artifact_bytes','maximum':3500000,'candidate':actual,'pass':actual<=3500000})
    accuracy_pass=all(c['pass'] for c in checks)
    speed_pass=None
    if (ROOT/f'runs/paired-{name}.json').exists():
        from .benchmark_pair import verify_pair
        speed_pass=verify_pair(name)['no_consistent_slowdown']
        if (ROOT/f'runs/paired-{name}-reverse.json').exists():
            speed_pass=speed_pass and verify_pair(name,True)['no_consistent_slowdown']
    report={'model':name,'accuracy_and_size_pass':accuracy_pass,'speed_pass':speed_pass,
            'pass':accuracy_pass and speed_pass is True,'checks':checks,
            'speed_status':'nine paired passes verified' if speed_pass is not None else 'separate paired benchmark required',
            'final_evaluation':'not run'}
    write_json(ROOT/f'results/acceptance-{name}.json',report)
    print(json.dumps(report,indent=2,ensure_ascii=True))
    return report['pass']


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=['v6-w8','v6-w16','v6-w8-rec'],required=True)
    raise SystemExit(0 if compare(parser.parse_args().model) else 1)
