#!/usr/bin/env python3
# Copyright 2024-present the vsag project
# SPDX-License-Identifier: Apache-2.0
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
from run_mixed_cpu_acceptance import sha, write_json

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--paired-low',action='store_true',help='Compare both sides at identical low budgets')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    feature=Path('/home/ubuntu/project/vsag-pr2904-promotion-20261012')
    library=Path('/home/ubuntu/project/pr2904-promotion-audit-20261012/release-final/libvsag-lite.so')
    patch=root/'results/reciprocal-diversity-r1-20261012/candidate.patch'
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=feature,text=True).strip()=='dbb2fe0bec685ae7a343b1d2060d0553a98e20d2'
    assert not subprocess.check_output(['git','status','--porcelain','-uno'],cwd=feature)
    assert sha(library)=='3171d0d7e86866f02096db130f1937eb13c5115cbe993d4a6feb2e9987774330'
    assert sha(patch)=='e377e6918491d8966c76602ef9fcee6fafd502e5400d0918e7fc82c4f11c33b9'
    env={k:v for k,v in os.environ.items() if not k.startswith('VSAG_')}
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',VSAG_GRAPH_STORAGE='fp32',VSAG_GRAPH_DEGREE='16',VSAG_GRAPH_EF='128')
    commands=[];rows=[]
    def run(tag,argv,cwd=None):
        with (out/(tag+'.stdout')).open('w') as stdout,(out/(tag+'.stderr')).open('w') as stderr:
            done=subprocess.run([str(x) for x in argv],cwd=cwd,env=env,stdout=stdout,stderr=stderr)
        commands.append(dict(tag=tag,argv=[str(x) for x in argv],exit=done.returncode,query_budget=env.get('VSAG_GRAPH_QUERY_EF'),library=env.get('LD_LIBRARY_PATH')))
        write_json(out/'commands.json',commands)
        assert done.returncode==0,tag
    with tempfile.TemporaryDirectory(prefix='vsag-budget-',dir='/dev/shm') as temporary:
        scratch=Path(temporary);source=scratch/'source'
        for name in ['src','include','lite']:shutil.copytree(feature/name,source/name)
        run('apply',['git','apply',patch],cwd=source)
        run('configure',['cmake','-S',source/'lite','-B',scratch/'build','-DCMAKE_BUILD_TYPE=Release','-DENABLE_RABITQ_LITE_BACKEND=ON','-DENABLE_TESTS=OFF'])
        run('build',['cmake','--build',scratch/'build','--target','vsag_lite','--parallel','2'])
        candidate=scratch/'build/libvsag-lite.so'
        shutil.copyfile(patch,out/'candidate.patch')
        write_json(out/'identity.json',dict(source_head='dbb2fe0bec685ae7a343b1d2060d0553a98e20d2',control_sha256=sha(library),candidate_sha256=sha(candidate),patch_sha256=sha(patch),candidate_bytes=candidate.stat().st_size,scope='FP32 reciprocal candidate query-budget calibration; development data only'))
        write_json(out/'protocol.json',dict(paired_low=args.paired_low,budgets=[128,192,256,384],trials=3,cpu=0,threads=1,development_data=True))
        builder=root/'results/quality-target-r2-20261012/source-graph_crud_quality.cpp'
        shutil.copyfile(builder,out/'source-builder.cpp')
        run('compile',['c++','-std=c++17','-O3','-DNDEBUG','-I'+str(feature/'include'),builder,'-L'+str(library.parent),'-Wl,-rpath,'+str(library.parent),'-lvsag-lite','-o',scratch/'builder'])
        datasets={'gist10k':root/'results/quality-target-r2-20261012/dataset/scale-10000','sift10k':Path('/home/ubuntu/project/vsag-lite-datasets/sift/prepared-10k-100k/scale-10000')}
        for dataset,inputs in datasets.items():
            write_json(out/(dataset+'.inputs.json'),{name:dict(sha256=sha(inputs/name),bytes=(inputs/name).stat().st_size) for name in ['base.fvecs','queries.fvecs','groundtruth.ivecs']})
            for trial in range(3):
                order=[('control',512)]+[('candidate',budget) for budget in [128,192,256,384]]
                order=order[trial:]+order[:trial]
                if args.paired_low:
                    order=[]
                    budgets=[128,192,256,384]
                    budgets=budgets[trial:]+budgets[:trial]
                    for budget in budgets:
                        sides=['control','candidate'] if trial%2==0 else ['candidate','control']
                        order.extend((side,budget) for side in sides)
                for side,budget in order:
                    tag=f'{dataset}-{trial}-{side}-{budget}'
                    env['VSAG_GRAPH_QUERY_EF']=str(budget)
                    env['LD_LIBRARY_PATH']=str(candidate.parent if side=='candidate' else library.parent)
                    run(tag,['taskset','-c','0',scratch/'builder',inputs,scratch/(tag+'.snapshot'),0,1,str(out/tag)+'.queries.csv'])
                    lines=(out/(tag+'.stdout')).read_text().splitlines()
                    pos=next(i for i,line in enumerate(lines) if line.startswith('base_count,query_count,'))
                    metrics=next(csv.DictReader(lines[pos:pos+2]))
                    assert int(metrics['query_ef_search'])==budget
                    rows.append(dict(dataset=dataset,trial=trial,side=side,budget=budget,metrics=metrics))
                    write_json(out/'rows.json',rows)
                    print(tag,metrics['recall_at_k'],metrics['search_p50_us'],flush=True)
    assert sha(library)=='3171d0d7e86866f02096db130f1937eb13c5115cbe993d4a6feb2e9987774330'
    assert not subprocess.check_output(['git','status','--porcelain','-uno'],cwd=feature)
    manifest={p.name:sha(p) for p in out.iterdir() if p.is_file()}
    write_json(out/'manifest.json',manifest)
    with tarfile.open(out/'raw.tar.gz','w:gz') as archive:
        for name in sorted(list(manifest)+['manifest.json']):archive.add(out/name,arcname=name,recursive=False)
    write_json(out/'archive.json',dict(bytes=(out/'raw.tar.gz').stat().st_size,sha256=sha(out/'raw.tar.gz'),members=len(manifest)+1))
if __name__=='__main__':main()
