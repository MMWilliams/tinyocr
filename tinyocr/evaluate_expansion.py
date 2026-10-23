"""Evaluate development data only. Reserved final samples have no CLI path here."""
import argparse
import json
import math
import platform
import statistics
import time

from .baseline import Baseline
from .check_expansion import check, select_dev
from .common import ROOT, sha256, write_json
from .expand import EXPANSION
from .metrics import aggregate, counts
from .screen_labels import annotated_recall


def evaluate(name, repeats=3):
    if repeats < 1:
        raise ValueError("repeats must be positive")
    manifest = check()
    items = select_dev(manifest)
    model = Baseline(name)
    for task in sorted({i["task"] for i in items}):
        item = next(i for i in items if i["task"] == task)
        for _ in range(2):
            model.predict(ROOT/item["image"],task)
    predictions = []
    for number,item in enumerate(items):
        latencies, previous = [], None
        for _ in range(repeats):
            start=time.perf_counter()
            text,regions=model.predict(ROOT/item["image"],item["task"])
            latencies.append((time.perf_counter()-start)*1000)
            if previous is not None and text != previous:
                raise ValueError("Prediction changed across repetitions")
            previous=text
        entry={"id":item["id"], "role":"dev", "domain":item["domain"], "task":item["task"],
               "prediction":text, "regions":regions, "latency_ms":latencies, "score_mode":item["score_mode"]}
        if item["score_mode"] == "transcription":
            entry["reference"]=item["text"]
            entry["normalized"]=counts(item["text"],text)
            entry["case_sensitive"]=counts(item["text"],text,casefold=False)
        if item.get("regions"):
            entry["annotation_recovery"]=annotated_recall(item["regions"],regions)
        predictions.append(entry)
        if (number+1)%10 == 0:
            print(f"{name} expanded dev {number+1}/{len(items)}",flush=True)
    domains={}
    for domain in sorted({i["domain"] for i in predictions}):
        rows=[i for i in predictions if i["domain"]==domain]
        latency=sorted(statistics.median(i["latency_ms"]) for i in rows)
        result={"n":len(rows),"p50_ms":statistics.median(latency),"p95_ms":latency[math.ceil(len(latency)*.95)-1]}
        if "normalized" in rows[0]:
            result["normalized"]=aggregate([r["normalized"] for r in rows])
            result["case_sensitive"]=aggregate([r["case_sensitive"] for r in rows])
        if "annotation_recovery" in rows[0]:
            recovered=sum(r["annotation_recovery"]["recovered"] for r in rows)
            annotated=sum(r["annotation_recovery"]["annotated"] for r in rows)
            result["annotation_recovery"]={"recovered":recovered,"annotated":annotated,"recall":recovered/annotated}
        domains[domain]=result
    report={"model":name,"partition":"dev","manifest_sha256":sha256(EXPANSION),
            "lock_sha256":sha256(ROOT/"uv.lock"), "environment":{"platform":platform.platform(),"cpu":platform.processor()},
            "code_sha256":{p.name:sha256(p) for p in sorted((ROOT/"tinyocr").glob("*.py"))},
            "model_details":model.metadata(),"timing":{"repeats":repeats,"warmups_per_task":2,
                "includes":"image decode, preprocessing, inference, postprocessing, reading order",
                "excludes":"model initialization, downloads, scoring", "percentiles":"per-item medians; nearest-rank p95"},
            "domains":domains,"predictions":predictions,"limitations":manifest["limitations"]}
    path=ROOT/f"runs/expanded-{name}.json"
    write_json(path,report)
    summary={k:v for k,v in report.items() if k!='predictions'}
    summary['full_report_sha256']=sha256(path)
    write_json(ROOT/f"results/expanded-{name}.json",summary)
    verify(name)
    print(json.dumps(domains,indent=2))


def verify(name):
    manifest=check()
    references={i['id']:i for i in select_dev(manifest)}
    report=json.loads((ROOT/f'runs/expanded-{name}.json').read_text(encoding='utf-8'))
    if report['manifest_sha256']!=sha256(EXPANSION) or report['lock_sha256']!=sha256(ROOT/'uv.lock'):
        raise ValueError('Changed manifest or dependencies')
    rows=report['predictions']
    if len(rows)!=len(references) or {r['id'] for r in rows}!=set(references):
        raise ValueError('Missing, duplicate, or final IDs in development results')
    for row in rows:
        ref=references[row['id']]
        if row['role']!='dev' or row['domain']!=ref['domain'] or row['task']!=ref['task'] or row['score_mode']!=ref['score_mode']:
            raise ValueError('Partition/domain changed')
        if len(row['latency_ms'])!=report['timing']['repeats'] or not all(math.isfinite(t) and t>0 for t in row['latency_ms']):
            raise ValueError('Invalid timing evidence')
        if ref['score_mode']=='annotated-recall':
            if 'normalized' in row or 'reference' in row:
                raise ValueError('Partial screen scored as full transcription')
        else:
            if row['reference']!=ref['text'] or row['normalized']!=counts(ref['text'],row['prediction']):
                raise ValueError('Reference/metric mismatch')
            if row['case_sensitive']!=counts(ref['text'],row['prediction'],False):
                raise ValueError('Case-sensitive metric mismatch')
        if ref.get('regions') and row['annotation_recovery']!=annotated_recall(ref['regions'],row['regions']):
            raise ValueError('Annotation recovery mismatch')
    if set(report['domains']) != {r['domain'] for r in rows}:
        raise ValueError('Missing/extra domain summary')
    for domain,result in report['domains'].items():
        selected=[r for r in rows if r['domain']==domain]
        if result['n']!=len(selected): raise ValueError('Wrong domain count')
        if 'normalized' in result and result['normalized']!=aggregate([r['normalized'] for r in selected]):
            raise ValueError('Incorrect domain aggregate')
        if 'case_sensitive' in result and result['case_sensitive']!=aggregate([r['case_sensitive'] for r in selected]):
            raise ValueError('Incorrect case-sensitive aggregate')
        if 'annotation_recovery' in result:
            a=sum(r['annotation_recovery']['annotated'] for r in selected)
            h=sum(r['annotation_recovery']['recovered'] for r in selected)
            if result['annotation_recovery']!={'annotated':a,'recovered':h,'recall':h/a}:
                raise ValueError('Incorrect recovery aggregate')
    details=report['model_details']
    if details['opencv_threads']!=1 or len(details['sessions'])!=2:
        raise ValueError('Wrong session configuration')
    for session in details['sessions']:
        if session['providers']!=['CPUExecutionProvider'] or session['intra_threads']!=1 or session['inter_threads']!=1:
            raise ValueError('Not one-thread CPU')
    for artifact in details['artifacts']:
        if sha256(ROOT/artifact['path'])!=artifact['sha256']:raise ValueError('Model changed')
    print(f'Verified expanded {name}: {len(rows)} dev predictions, zero final predictions')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=['v6-tiny','v5-english'],required=True)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    verify(args.model) if args.verify_only else evaluate(args.model)
