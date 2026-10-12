#!/usr/bin/env python3
# Copyright 2024-present the vsag project
# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import os
from pathlib import Path
import statistics
import struct
import subprocess
import tarfile
from run_mixed_cpu_acceptance import sha, write_json
from run_quality_target import audit_query

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(exist_ok=False)
    root=Path(__file__).resolve().parent;prior=root/'results/aligned-fp32-full-r2-20261012'
    saved=Path('/home/ubuntu/project/aligned-fp32-followup-20261012')
    libraries=json.loads((prior/'libraries.json').read_text())
    assert sha(saved/'libvsag-lite.so')==libraries['candidate']['sha256']
    full=Path('/home/ubuntu/project/vsag-full-strict-install-20261012/lib')
    assert sha(full/'libvsag.so')==libraries['full']['sha256']
    env={k:v for k,v in os.environ.items() if not k.startswith('VSAG_')}
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    commands=[];rows=[]
    for label in ['sift-full','gist-replay']:
        snapshot=saved/(label+'.snapshot');digest=sha(snapshot)
        queries=Path('/home/ubuntu/project/vsag-lite-datasets/sift/prepared-10k-100k/scale-10000/queries.fvecs') if label=='sift-full' else saved/'gist-replay-queries.fvecs'
        truth_path=queries.with_name('groundtruth.ivecs') if label=='sift-full' else prior/'gist10k-replay-truth.ivecs'
        data=truth_path.read_bytes();truth=[set(struct.unpack_from('<11i',data,i*44)[1:]) for i in range(100)]
        write_json(out/(label+'.inputs.json'),dict(snapshot_sha256=digest,query_sha256=sha(queries),truth_sha256=sha(truth_path),library_sha256=libraries['full' if label=='sift-full' else 'candidate']['sha256'],query_binary_sha256=sha(saved/('full-query' if label=='sift-full' else 'candidate-query'))))
        for trial in range(3):
            for budget in ([256,384,512,1024] if trial%2==0 else [1024,512,384,256]):
                tag=f'{label}-{trial}-{budget}';stem=out/tag
                env['LD_LIBRARY_PATH']=str(full if label=='sift-full' else saved)
                argv=['taskset','-c','0',str(saved/('full-query' if label=='sift-full' else 'candidate-query')),str(snapshot),str(queries),'128' if label=='sift-full' else '960','10000',str(budget),'0','100',str(stem),str(saved/'sift-full.config.json'),'fp32','0']
                result=subprocess.run(argv,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                (out/(tag+'.stdout')).write_text(result.stdout);(out/(tag+'.stderr')).write_text(result.stderr)
                commands.append(dict(tag=tag,argv=argv,exit=result.returncode));write_json(out/'commands.json',commands)
                assert result.returncode==0
                row=dict(label=label,trial=trial,budget=budget,**audit_query(stem,truth,0,100))
                rows.append(row);write_json(out/'rows.json',rows);print(row,flush=True)
        assert sha(snapshot)==digest
    summary=[]
    for label in ['sift-full','gist-replay']:
        for budget in [256,384,512,1024]:
            group=[r for r in rows if r['label']==label and r['budget']==budget]
            assert len(group)==3
            summary.append(dict(label=label,budget=budget,all_target_met=all(r['recall']>=.990 for r in group),**{k:statistics.median(r[k] for r in group) for k in ['recall','p50_us','p99_us']}))
    write_json(out/'summary.json',summary)
    manifest={p.name:sha(p) for p in out.iterdir() if p.is_file()};write_json(out/'manifest.json',manifest)
    with tarfile.open(out/'raw.tar.gz','w:gz') as archive:
        for name in sorted(list(manifest)+['manifest.json']):archive.add(out/name,arcname=name,recursive=False)
    write_json(out/'archive.json',dict(sha256=sha(out/'raw.tar.gz'),bytes=(out/'raw.tar.gz').stat().st_size,members=len(manifest)+1))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
