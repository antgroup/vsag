#!/usr/bin/env python3
# Copyright 2024-present the vsag project
# SPDX-License-Identifier: Apache-2.0
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import tarfile
from run_mixed_cpu_acceptance import sha, write_json, percentile
from run_quality_target import audit_query

def verify(out,sift):
    manifest=json.loads((out/'manifest.json').read_text())
    for name,digest in manifest.items():assert sha(out/name)==digest,name
    archive_receipt=json.loads((out/'archive.json').read_text())
    assert sha(out/'raw.tar.gz')==archive_receipt['sha256']
    with tarfile.open(out/'raw.tar.gz') as archive:
        assert len(archive.getmembers())==archive_receipt['members']
        assert set(archive.getnames())==set(manifest)|{'manifest.json'}
        for name,digest in manifest.items():assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==digest
    rows=json.loads((out/'rows.json').read_text());protocol=json.loads((out/'protocol.json').read_text())
    expected={(d,t,s) for d in ['gist10k','sift10k'] for t in range(3) for s in ['control','candidate','full']}
    assert len(rows)==18 and {(r['dataset'],r['trial'],r['side']) for r in rows}==expected
    commands=json.loads((out/'commands.json').read_text())
    assert len(commands)==58 and all(c['exit']==0 for c in commands)
    for side in ['control','candidate']:
        tests=(out/(side+'-tests.stdout')).read_text()
        assert '100% tests passed, 0 tests failed out of 5' in tests
        records=json.loads((out/(side+'-compile_commands.json')).read_text())
        core=[r for r in records if '/src/lite/' in r['file'] and not r['file'].endswith('_test.cpp')]
        assert core and all('-O3' in r['command'] and '-Ofast' not in r['command'] and '-ffast-math' not in r['command'] for r in core)
    truth={};queries=0
    for dataset,directory in [('gist10k',out/'dataset/scale-10000'),('sift10k',sift)]:
        for name,info in json.loads((out/(dataset+'.inputs.json')).read_text()).items():assert sha(directory/name)==info['sha256'] and (directory/name).stat().st_size==info['bytes']
        data=(directory/'groundtruth.ivecs').read_bytes();truth[dataset]=[]
        for offset in range(0,len(data),44):
            record=struct.unpack_from('<11i',data,offset);assert record[0]==10 and len(set(record[1:]))==10
            truth[dataset].append(set(record[1:]))
    for row in rows:
        dataset,trial,side=[row[k] for k in ['dataset','trial','side']]
        tag=f'{dataset}-{trial}-{side}'
        actual=audit_query(out/tag,truth[dataset],0,len(truth[dataset]))
        for name,value in actual.items():assert math.isclose(value,row[name],abs_tol=1e-9),tag
        budgets=protocol['frozen_gist_budgets' if dataset=='gist10k' else 'frozen_sift_budgets']
        assert row['budget']==budgets[side] and int(row['resources']['ef'])==row['budget']
        queries+=actual['queries']
    summary=json.loads((out/'summary.json').read_text())
    assert len(summary)==6
    for cell in summary:
        selected=[r for r in rows if r['dataset']==cell['dataset'] and r['side']==cell['side']]
        for key in ['recall','p50_us','p99_us','snapshot_bytes']:assert math.isclose(cell[key],statistics.median(r[key] for r in selected),abs_tol=1e-9)
        assert cell['all_target_met']==all(r['recall']>=.990 for r in selected)
    replay=json.loads((out/'replay-summary.json').read_text())
    assert len(replay)==2 and {r['dataset'] for r in replay}=={'gist10k','sift10k'}
    for cell in replay:
        dataset=cell['dataset'];stem=dataset+'-candidate-replay.csv'
        with (out/(stem+'.operations.csv')).open() as stream:operations=list(csv.DictReader(stream))
        assert len(operations)==cell['operations']==90000
        cell['per_api']={}
        for kind,name in [(0,'update'),(1,'remove'),(2,'add')]:
            values=[float(r['latency_us']) for r in operations if int(r['operation'])==kind];cell['per_api'][name]=dict(calls=len(values),p50_us=percentile(values,.5),p99_us=percentile(values,.99),calls_per_timed_second=len(values)*1e6/sum(values))
        for cycle in range(30000):
            selected=operations[cycle*3:(cycle+1)*3]
            assert [int(r['operation']) for r in selected]==[0,1,2]
            assert all(int(r['cycle'])==cycle and int(r['id'])==((cycle//10000)*65537+(cycle%10000)*8191)%10000 and math.isfinite(float(r['latency_us'])) and float(r['latency_us'])>=0 for r in selected)
        with (out/(stem+'.neighbors.csv')).open() as stream:neighbors=list(csv.DictReader(stream))
        assert len(neighbors)==1000
        truth_data=(out/(dataset+'-replay-truth.ivecs')).read_bytes();hits=0
        for query in range(100):
            selected=neighbors[query*10:(query+1)*10]
            assert [(int(r['query']),int(r['rank'])) for r in selected]==[(query,i) for i in range(10)]
            ids=[int(r['id']) for r in selected];assert len(set(ids))==10 and all(0<=i<10000 for i in ids)
            truth_ids=struct.unpack_from('<11i',truth_data,query*44)[1:]
            hits+=len(set(ids)&set(truth_ids))
        assert math.isclose(hits/1000,float(cell['metrics']['recall_at_k']),abs_tol=5e-7)
    assert '100% tests passed, 0 tests failed out of 5' in (out/'candidate-sanitizer-tests.stdout').read_text()
    return dict(audit='PASS',runs=len(rows),recorded_queries=queries,replay_operations=180000,candidate_sanitizer='5/5 PASS',replay_quality_target_met=all(float(c['metrics']['recall_at_k'])>=.990 for c in replay),quality_target_not_contractual=True,summary=summary,replay=replay)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);parser.add_argument('--sift',type=Path,required=True)
    args=parser.parse_args();result=verify(args.output,args.sift)
    write_json(args.output/'verification.json',result);print(json.dumps(result,indent=2))
