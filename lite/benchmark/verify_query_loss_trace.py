#!/usr/bin/env python3
# Copyright 2024-present the vsag project
# SPDX-License-Identifier: Apache-2.0
"""Recompute disjoint loss accounting and previous-result parity from raw records."""
import argparse
import csv
import json
from pathlib import Path
import struct
from run_mixed_cpu_acceptance import sha, write_json


def classify(rows):
    counts=dict(hits=0,unvisited=0,shortlist_loss=0,visited_rank_loss=0)
    for row in rows:
        visited,formal,rerank=[int(row[key]) for key in ['visited','formal','visited_rerank']]
        assert visited in [0,1] and formal in [0,1] and rerank in [0,1]
        assert not formal or visited
        assert not rerank or visited
        if formal: counts['hits']+=1
        elif not visited: counts['unvisited']+=1
        elif rerank: counts['shortlist_loss']+=1
        else: counts['visited_rank_loss']+=1
    assert sum(counts.values())==len(rows)
    return counts


def read(path):
    with path.open() as stream: return list(csv.DictReader(stream))


def verify(output, reference=None):
    manifest=json.loads((output/'manifest.json').read_text())
    for name,digest in manifest.items(): assert sha(output/name)==digest,name
    identity=json.loads((output/'identity.json').read_text())
    assert identity['source_head']=='dbb2fe0bec685ae7a343b1d2060d0553a98e20d2' and identity['ef']==2048
    assert identity['library_sha256']=='3171d0d7e86866f02096db130f1937eb13c5115cbe993d4a6feb2e9987774330'
    assert sha(output/'source-trace_query_losses.cpp')==identity['cpp_sha256']
    assert sha(output/'source-run_query_loss_trace.py')==identity['runner_sha256']
    reference=reference or Path(identity['reference'])
    dataset=reference/'dataset/scale-10000'
    for name,digest in identity['inputs'].items(): assert sha(dataset/name)==digest,name
    truths=(dataset/'groundtruth.ivecs').read_bytes()
    prior=json.loads((reference/'rows.json').read_text())
    commands=json.loads((output/'commands.json').read_text())
    assert len(commands)==8 and all(c['exit']==0 for c in commands)
    summaries=[]
    for storage in ['fp32','fp16','rabitq8']:
        assert {r['snapshot_sha256'] for r in prior if r['side']=='lite' and r['storage']==storage}=={identity['storage'][storage]['snapshot_sha256']}
        evidence=read(output/(storage+'.truth.csv'))
        queries=read(output/(storage+'.queries.csv'))
        actual=read(output/(storage+'.neighbors.csv'))
        assert len(evidence)==len(actual)==10000 and len(queries)==1000
        expected=[]
        for suffix in ['ef2048-calibration','heldout']: expected.extend(read(reference/(storage+'-0-lite-'+suffix+'.neighbors.csv')))
        for a,b in zip(actual,expected):
            assert [a[k] for k in ['query','rank','id']]==[b[k] for k in ['query','rank','id']]
            assert struct.pack('<f',float.fromhex(a['distance']))==struct.pack('<f',float.fromhex(b['distance']))
        for query in range(1000):
            record=struct.unpack_from('<11i',truths,44*query)
            selected=evidence[query*10:(query+1)*10]
            assert [(int(r['query']),int(r['id'])) for r in selected]==[(query,i) for i in record[1:]]
            native_ids={int(r['id']) for r in actual[query*10:(query+1)*10]}
            assert all(int(r['formal'])==int(int(r['id']) in native_ids) for r in selected)
            summary=queries[query]
            assert int(summary['query'])==query
            for key,flag in [('formal_hits','formal'),('visited_truth','visited'),('visited_rerank_hits','visited_rerank')]:
                assert int(summary[key])==sum(int(r[flag]) for r in selected)
            assert int(summary['visited_truth'])<=int(summary['visited_nodes'])<=10000
            if storage=='rabitq8':
                assert all(int(r['all_codes']) in [0,1] for r in selected)
                assert int(summary['all_codes_hits'])==sum(int(r['all_codes']) for r in selected)
            else:
                assert int(summary['all_codes_hits'])==-1 and all(int(r['all_codes'])==-1 for r in selected)
        selected=evidence[1000:]
        counts=classify(selected)
        summaries.append(dict(storage=storage,truth_records=len(selected),**counts,
            global_code_hits=sum(int(r['all_codes']) for r in selected) if storage=='rabitq8' else None,
            mean_visited=sum(int(r['visited_nodes']) for r in queries[100:])/900,
            min_visited=min(int(r['visited_nodes']) for r in queries[100:]),max_visited=max(int(r['visited_nodes']) for r in queries[100:])))
    unseen={s:{(int(r['query']),int(r['id'])) for r in read(output/(s+'.truth.csv'))
        if int(r['query'])>=100 and r['visited']=='0'} for s in ['fp32','fp16','rabitq8']}
    overlaps={a+'-'+b:len(unseen[a]&unseen[b]) for a,b in [('fp32','fp16'),('fp32','rabitq8'),('fp16','rabitq8')]}
    return dict(audit='PASS',diagnostic_queries=3000,truth_records=30000,
        ordered_native_neighbors=30000,shared_unvisited_truth_pairs=overlaps,summary=summaries)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);parser.add_argument('--reference',type=Path)
    args=parser.parse_args();result=verify(args.output,args.reference)
    write_json(args.output/'verification.json',result)
    print(json.dumps(result,indent=2))
