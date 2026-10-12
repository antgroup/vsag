#!/usr/bin/env python3
# Copyright 2024-present the vsag project
# SPDX-License-Identifier: Apache-2.0
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import tarfile
from run_mixed_cpu_acceptance import sha, write_json
from run_quality_target import audit_query

def main():
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);args=parser.parse_args();out=args.output
    manifest=json.loads((out/'manifest.json').read_text())
    for name,digest in manifest.items():assert sha(out/name)==digest
    receipt=json.loads((out/'archive.json').read_text());assert sha(out/'raw.tar.gz')==receipt['sha256']
    with tarfile.open(out/'raw.tar.gz') as archive:
        assert len(archive.getmembers())==receipt['members'] and set(archive.getnames())==set(manifest)|{'manifest.json'}
        for name,digest in manifest.items():assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==digest
    prior=out.parent/'aligned-fp32-full-r2-20261012';saved=Path('/home/ubuntu/project/aligned-fp32-followup-20261012')
    rows=json.loads((out/'rows.json').read_text());commands=json.loads((out/'commands.json').read_text())
    expected={(l,t,b) for l in ['sift-full','gist-replay'] for t in range(3) for b in [256,384,512,1024]}
    assert len(rows)==len(commands)==24 and {(r['label'],r['trial'],r['budget']) for r in rows}==expected
    assert all(c['exit']==0 for c in commands)
    for label in ['sift-full','gist-replay']:
        inputs=json.loads((out/(label+'.inputs.json')).read_text());assert sha(saved/(label+'.snapshot'))==inputs['snapshot_sha256']
        queries=Path('/home/ubuntu/project/vsag-lite-datasets/sift/prepared-10k-100k/scale-10000/queries.fvecs') if label=='sift-full' else saved/'gist-replay-queries.fvecs'
        truth_path=queries.with_name('groundtruth.ivecs') if label=='sift-full' else prior/'gist10k-replay-truth.ivecs'
        assert sha(queries)==inputs['query_sha256'] and sha(truth_path)==inputs['truth_sha256']
        data=truth_path.read_bytes();truth=[set(struct.unpack_from('<11i',data,i*44)[1:]) for i in range(100)]
        for row in [r for r in rows if r['label']==label]:
            tag=f"{label}-{row['trial']}-{row['budget']}";actual=audit_query(out/tag,truth,0,100)
            for key,value in actual.items():assert math.isclose(value,row[key],abs_tol=1e-9)
    summary=json.loads((out/'summary.json').read_text());assert len(summary)==8
    for cell in summary:
        selected=[r for r in rows if r['label']==cell['label'] and r['budget']==cell['budget']]
        for key in ['recall','p50_us','p99_us']:assert math.isclose(cell[key],statistics.median(r[key] for r in selected),abs_tol=1e-9)
        assert cell['all_target_met']==all(r['recall']>=.990 for r in selected)
    write_json(out/'verification.json',dict(audit='PASS',runs=24,queries=2400,scope='Development/post-mutation calibration, not fresh independent validation',summary=summary))
    print('PASS24runs/2400queries; diagnostic calibration only')
if __name__=='__main__':main()
