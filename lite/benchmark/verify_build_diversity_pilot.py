#!/usr/bin/env python3
# Copyright 2024-present the vsag project
# SPDX-License-Identifier: Apache-2.0
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import struct
from run_mixed_cpu_acceptance import sha, percentile, write_json

def read(path):
    with path.open() as stream:return list(csv.DictReader(stream))

def verify(out,datasets):
    manifest=json.loads((out/'manifest.json').read_text())
    for name,digest in manifest.items():assert sha(out/name)==digest,name
    identity=json.loads((out/'identity.json').read_text())
    assert sha(out/'candidate.patch')==identity['patch_sha256']
    assert identity['source_head']=='dbb2fe0bec685ae7a343b1d2060d0553a98e20d2'
    assert identity['control_sha256']=='3171d0d7e86866f02096db130f1937eb13c5115cbe993d4a6feb2e9987774330'
    rows=json.loads((out/'rows.json').read_text())
    sides=['control','nearest','candidate'] if (out/'nearest-control.json').exists() else ['control','candidate']
    expected={(d,t,s) for d in ['sift10k','gist10k'] for t in range(3) for s in sides}
    assert len(rows)==len(expected) and {(r['dataset'],r['trial'],r['side']) for r in rows}==expected
    commands=json.loads((out/'commands.json').read_text())
    expected_commands = 25 if len(sides)==3 else (17 if any(c['tag']=='build-nearest' for c in commands) else 16)
    assert len(commands)==expected_commands and all(c['exit']==0 for c in commands)
    assert set(c['tag'] for c in commands)>={f'{d}-{t}-{s}' for d,t,s in expected}
    truths={}
    for label,directory in datasets.items():
        for name,info in json.loads((out/(label+'.inputs.json')).read_text()).items():
            assert sha(directory/name)==info['sha256'] and (directory/name).stat().st_size==info['bytes']
        data=(directory/'groundtruth.ivecs').read_bytes()
        truths[label]=[]
        for offset in range(0,len(data),44):
            record=struct.unpack_from('<11i',data,offset)
            assert record[0]==10 and len(set(record[1:]))==10
            truths[label].append(set(record[1:]))
    raw_hits={}
    for row in rows:
        tag=f"{row['dataset']}-{row['trial']}-{row['side']}"
        metrics=row['metrics'];truth=truths[row['dataset']]
        neighbors=read(out/(tag+'.queries.csv.neighbors.csv'))
        timings=read(out/(tag+'.queries.csv.latencies.csv'))
        samples=read(out/(tag+'.queries.csv'))
        assert len(neighbors)==10*len(truth) and len(timings)==len(samples)==len(truth)
        hits=[]
        for query,expected_ids in enumerate(truth):
            selected=neighbors[10*query:10*(query+1)]
            assert [(int(r['query']),int(r['rank'])) for r in selected]==[(query,i) for i in range(10)]
            ids=[int(r['id']) for r in selected];distances=[float.fromhex(r['distance']) for r in selected]
            assert len(set(ids))==10 and all(0<=i<10000 for i in ids)
            assert all(math.isfinite(d) for d in distances) and distances==sorted(distances)
            hits.append(len(set(ids)&expected_ids))
            assert int(samples[query]['hits'])==hits[-1]
        assert math.isclose(sum(hits)/(10*len(truth)),float(metrics['recall_at_k']),abs_tol=5e-7)
        values=[float(r['latency_us']) for r in timings]
        for fraction,key in [(.5,'search_p50_us'),(.99,'search_p99_us')]:
            assert math.isclose(percentile(values,fraction),float(metrics[key]),abs_tol=2e-6)
        raw_hits[(row['dataset'],row['trial'],row['side'])]=hits
    summary=[]
    for dataset in ['sift10k','gist10k']:
        item=dict(dataset=dataset)
        for side in sides:
            selected=[r['metrics'] for r in rows if r['dataset']==dataset and r['side']==side]
            item[side]={key:statistics.median(float(r[key]) for r in selected) for key in ['recall_at_k','build_ms','search_p50_us','search_p99_us','save_ms','snapshot_bytes']}
        assert len({tuple(raw_hits[(dataset,t,'control')]) for t in range(3)})==1
        assert len({tuple(raw_hits[(dataset,t,'candidate')]) for t in range(3)})==1
        a=raw_hits[(dataset,0,'control')];b=raw_hits[(dataset,0,'candidate')]
        item['query_win_loss_tie']=[sum(y>x for x,y in zip(a,b)),sum(y<x for x,y in zip(a,b)),sum(y==x for x,y in zip(a,b))]
        summary.append(item)
    return dict(audit='PASS',runs=len(rows),recorded_queries=sum(len(truths[r['dataset']]) for r in rows),summary=summary)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path)
    parser.add_argument('--gist',type=Path,required=True);parser.add_argument('--sift',type=Path,required=True)
    args=parser.parse_args();result=verify(args.output,{'gist10k':args.gist,'sift10k':args.sift})
    write_json(args.output/'verification.json',result);print(json.dumps(result,indent=2))
