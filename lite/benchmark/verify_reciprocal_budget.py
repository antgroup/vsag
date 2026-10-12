#!/usr/bin/env python3
# Copyright 2024-present the vsag project
# SPDX-License-Identifier: Apache-2.0
import argparse
import hashlib
import csv
import json
import math
from pathlib import Path
import statistics
import struct
import tarfile
from run_mixed_cpu_acceptance import sha, percentile, write_json

def verify(out,datasets):
    manifest=json.loads((out/'manifest.json').read_text())
    for name,digest in manifest.items():assert sha(out/name)==digest,name
    receipt=json.loads((out/'archive.json').read_text())
    assert sha(out/'raw.tar.gz')==receipt['sha256']
    assert (out/'raw.tar.gz').stat().st_size==receipt['bytes']
    with tarfile.open(out/'raw.tar.gz') as archive:
        assert set(archive.getnames())==set(manifest)|{'manifest.json'}
        assert len(archive.getmembers())==receipt['members']
        for name,digest in manifest.items():
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==digest,name
        assert json.loads(archive.extractfile('manifest.json').read())==manifest
    identity=json.loads((out/'identity.json').read_text())
    assert identity['source_head']=='dbb2fe0bec685ae7a343b1d2060d0553a98e20d2'
    assert identity['control_sha256']=='3171d0d7e86866f02096db130f1937eb13c5115cbe993d4a6feb2e9987774330'
    assert identity['patch_sha256']==sha(out/'candidate.patch')=='e377e6918491d8966c76602ef9fcee6fafd502e5400d0918e7fc82c4f11c33b9'
    rows=json.loads((out/'rows.json').read_text())
    expected={(d,t,s,b) for d in datasets for t in range(3) for s,b in [('control',512)]+[('candidate',b) for b in [128,192,256,384]]}
    paired=(out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text())['paired_low']
    if paired:expected={(d,t,s,b) for d in datasets for t in range(3) for s in ['control','candidate'] for b in [128,192,256,384]}
    assert len(rows)==len(expected) and {(r['dataset'],r['trial'],r['side'],r['budget']) for r in rows}==expected
    commands=json.loads((out/'commands.json').read_text())
    assert len(commands)==len(expected)+4 and all(c['exit']==0 for c in commands)
    truths={};hits_by_cell={};queries=0
    for dataset,directory in datasets.items():
        for name,info in json.loads((out/(dataset+'.inputs.json')).read_text()).items():
            assert sha(directory/name)==info['sha256'] and (directory/name).stat().st_size==info['bytes']
        data=(directory/'groundtruth.ivecs').read_bytes();truths[dataset]=[]
        for offset in range(0,len(data),44):
            record=struct.unpack_from('<11i',data,offset)
            assert record[0]==10 and len(set(record[1:]))==10
            truths[dataset].append(set(record[1:]))
    def read(name):
        with (out/name).open() as stream:return list(csv.DictReader(stream))
    for row in rows:
        dataset,trial,side,budget=[row[k] for k in ['dataset','trial','side','budget']]
        tag=f'{dataset}-{trial}-{side}-{budget}';truth=truths[dataset];metrics=row['metrics']
        command=[c for c in commands if c['tag']==tag]
        assert len(command)==1 and command[0]['query_budget']==str(budget)
        assert int(metrics['max_degree'])==16 and int(metrics['ef_search'])==128
        assert int(metrics['base_count'])==10000 and int(metrics['query_count'])==len(truth)
        assert int(metrics['query_ef_search'])==budget
        neighbors=read(tag+'.queries.csv.neighbors.csv');timings=read(tag+'.queries.csv.latencies.csv');samples=read(tag+'.queries.csv')
        assert len(neighbors)==10*len(truth) and len(timings)==len(samples)==len(truth)
        hits=[]
        for query,expected_ids in enumerate(truth):
            selected=neighbors[query*10:(query+1)*10]
            assert [(int(r['query']),int(r['rank'])) for r in selected]==[(query,i) for i in range(10)]
            ids=[int(r['id']) for r in selected];distances=[float.fromhex(r['distance']) for r in selected]
            assert len(set(ids))==10 and all(0<=i<10000 for i in ids)
            assert all(math.isfinite(d) for d in distances) and distances==sorted(distances)
            hits.append(len(set(ids)&expected_ids));assert int(samples[query]['hits'])==hits[-1]
        assert math.isclose(sum(hits)/(10*len(truth)),float(metrics['recall_at_k']),abs_tol=5e-7)
        values=[float(r['latency_us']) for r in timings]
        for fraction,key in [(.5,'search_p50_us'),(.99,'search_p99_us')]:
            assert math.isclose(percentile(values,fraction),float(metrics[key]),abs_tol=2e-6)
        hits_by_cell[(dataset,trial,side,budget)]=hits;queries+=len(truth)
    summary=[]
    for dataset in datasets:
        cells=[(s,b) for b in [128,192,256,384] for s in ['control','candidate']] if paired else [('control',512)]+[('candidate',b) for b in [128,192,256,384]]
        for side,budget in cells:
            selected=[r['metrics'] for r in rows if (r['dataset'],r['side'],r['budget'])==(dataset,side,budget)]
            assert len({tuple(hits_by_cell[(dataset,t,side,budget)]) for t in range(3)})==1
            summary.append(dict(dataset=dataset,side=side,budget=budget,**{key:statistics.median(float(r[key]) for r in selected) for key in ['recall_at_k','search_p50_us','search_p99_us','build_ms','snapshot_bytes']}))
    return dict(audit='PASS',runs=len(rows),recorded_queries=queries,summary=summary)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path)
    parser.add_argument('--gist',type=Path,required=True);parser.add_argument('--sift',type=Path,required=True)
    args=parser.parse_args();result=verify(args.output,{'gist10k':args.gist,'sift10k':args.sift})
    write_json(args.output/'verification.json',result);print(json.dumps(result,indent=2))
