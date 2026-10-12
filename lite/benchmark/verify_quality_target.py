#!/usr/bin/env python3
# Copyright 2024-present the vsag project
# SPDX-License-Identifier: Apache-2.0
"""Recompute coverage, raw quantiles/truth hits and frozen-parameter decisions."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import struct

from run_mixed_cpu_acceptance import sha, write_json


def quantile(values, fraction):
    ordered = sorted(values)
    return ordered[max(1, math.ceil(len(values)*fraction))-1]


def verify(output, dataset=None):
    protocol = json.loads((output/'protocol.json').read_text())
    manifest = json.loads((output/'manifest.json').read_text())
    if (output/'measured-runner.py').exists():
        assert sha(output/'measured-runner.py') == protocol['runner_sha256']
    if (output/'source-query_fixed_snapshot.cpp').exists():
        assert sha(output/'source-query_fixed_snapshot.cpp') == protocol['cpp_sha256']

    for name, digest in manifest.items(): assert sha(output/name) == digest, name
    dataset = dataset or output/'dataset/scale-10000'
    for name, record in json.loads((output/'dataset-receipt.json').read_text()).items():
        assert sha(dataset/name) == record['sha256'] and (dataset/name).stat().st_size == record['bytes']
    grid = protocol['grid']
    assert grid == [128,256,512,1024,2048,4096] and protocol['target'] == .990
    assert protocol['calibration'] == [0,100] and protocol['heldout'] == [100,1000]
    assert protocol['source_head'] == protocol['full_receipt']['full_source_head']
    receipt = protocol['full_receipt']
    assert receipt['fresh_core'] is True and receipt['ccache'] is False and receipt['prior_object_count'] == 0
    truth_data = (dataset/'groundtruth.ivecs').read_bytes()
    assert len(truth_data) == 44000
    truths = []
    for query in range(1000):
        record = struct.unpack_from('<11i',truth_data,query*44)
        assert record[0] == 10 and len(set(record[1:])) == 10 and all(0<=i<10000 for i in record[1:])
        truths.append(set(record[1:]))
    rows = json.loads((output/'rows.json').read_text())
    selections = json.loads((output/'frozen-selections.json').read_text())
    assert len(rows) == 126
    calibration = {(storage,trial,side,ef) for storage in ['fp32','fp16','rabitq8']
        for trial in range(3) for side in ['lite','full'] for ef in grid}
    actual = [(r['storage'],r['trial'],r['side'],r['ef']) for r in rows if r['cohort']=='calibration']
    assert len(actual)==108 and set(actual)==calibration
    heldout = [(r['storage'],r['trial'],r['side']) for r in rows if r['cohort']=='heldout']
    assert len(heldout)==18 and set(heldout)=={(s,t,b) for s in ['fp32','fp16','rabitq8'] for t in range(3) for b in ['lite','full']}
    commands=json.loads((output/'commands.json').read_text())
    assert len(commands)==150 and all(c['exit']==0 for c in commands)
    resources=[]
    for row in rows:
        tag=f"{row['storage']}-{row['trial']}-{row['side']}"
        cohort=row['cohort']
        begin,end=protocol[cohort]
        stem=tag+'-heldout' if cohort=='heldout' else tag+f"-ef{row['ef']}-calibration"
        command=next(c for c in commands if c['tag']==stem)
        argv=command['argv']
        assert argv[:3]==['taskset','-c','0']
        assert [int(argv[i]) for i in [6,7,8,9,14]] == [960,10000,row['ef'],begin,0]
        assert int(argv[10])==end
        with (output/(stem+'.queries.csv')).open() as stream: timings=list(csv.DictReader(stream))
        with (output/(stem+'.neighbors.csv')).open() as stream: neighbors=list(csv.DictReader(stream))
        assert [int(r['query']) for r in timings]==list(range(begin,end))
        assert len(neighbors)==(end-begin)*10
        hits=0
        for offset,query in enumerate(range(begin,end)):
            selected=neighbors[offset*10:(offset+1)*10]
            assert [(int(r['query']),int(r['rank'])) for r in selected]==[(query,i) for i in range(10)]
            ids=[int(r['id']) for r in selected]
            distances=[float.fromhex(r['distance']) for r in selected]
            assert len(set(ids))==10 and all(0<=i<10000 for i in ids)
            assert all(math.isfinite(d) for d in distances) and distances==sorted(distances)
            hits += sum(i in truths[query] for i in ids)
        values=[float(r['latency_us']) for r in timings]
        assert all(math.isfinite(v) and v>=0 for v in values)
        assert row['queries']==end-begin and row['hits']==hits and row['recall']==hits/(10*(end-begin))
        assert row['p50_us']==quantile(values,.5) and row['p99_us']==quantile(values,.99)
        hashes={r['snapshot_sha256'] for r in rows if (r['storage'],r['trial'],r['side'])==(row['storage'],row['trial'],row['side'])}
        assert len(hashes)==1
        resource_lines=(output/(stem+'.stdout')).read_text().splitlines()
        header=next(i for i,line in enumerate(resource_lines) if line.startswith('queries,ef,phase_cpu_us,'))
        resource=next(csv.DictReader(resource_lines[header:header+2]))
        assert int(resource['queries'])==end-begin and int(resource['ef'])==row['ef']
        assert all(float(v)>=0 and math.isfinite(float(v)) for v in resource.values())
        resources.append(dict(stem=stem,cohort=cohort,side=row['side'],storage=row['storage'],**resource))
    for key,selected in selections.items():
        storage,side=key.rsplit('-',1)
        eligible=[ef for ef in grid if min(r['recall'] for r in rows if r['cohort']=='calibration' and r['storage']==storage and r['side']==side and r['ef']==ef)>=.990]
        assert selected==dict(ef=min(eligible) if eligible else 4096,calibration_target_met=bool(eligible))
        assert all(r['ef']==selected['ef'] for r in rows if r['cohort']=='heldout' and r['storage']==storage and r['side']==side)
    summary=[]
    for storage in ['fp32','fp16','rabitq8']:
        for side in ['lite','full']:
            selected=[r for r in rows if r['cohort']=='heldout' and r['storage']==storage and r['side']==side]
            phase=[r for r in resources if r['cohort']=='heldout' and r['storage']==storage and r['side']==side]
            summary.append(dict(storage=storage,side=side,**selections[storage+'-'+side],heldout_target_all_trials=all(r['recall']>=.990 for r in selected),
                recall_median=statistics.median(r['recall'] for r in selected),recall_min=min(r['recall'] for r in selected),
                p50_us=statistics.median(r['p50_us'] for r in selected),p99_us=statistics.median(r['p99_us'] for r in selected),
                phase_cpu_us=statistics.median(float(r['phase_cpu_us']) for r in phase),process_peak_rss_kib=statistics.median(int(r['process_peak_rss_kib']) for r in phase)))
    return dict(audit='PASS',calibration_processes=108,heldout_processes=18,recorded_queries=27000,distinct_queries=1000,summary=summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    parser.add_argument('--dataset',type=Path)
    args=parser.parse_args()
    result=verify(args.output,args.dataset)
    write_json(args.output/'verification.json',result)
    print(json.dumps(result,indent=2))
