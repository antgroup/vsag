#!/usr/bin/env python3
# Copyright 2024-present the vsag project
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic only: trace identical fixed Lite indices, not instrumented timings."""
import argparse
import csv
import json
import os
from pathlib import Path
import struct
import subprocess
import tarfile
import tempfile
from run_mixed_cpu_acceptance import sha, write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--feature',type=Path,default=Path('/home/ubuntu/project/vsag-pr2904-promotion-20261012'))
    parser.add_argument('--lite-build',type=Path,default=Path('/home/ubuntu/project/pr2904-promotion-audit-20261012/release-final'))
    parser.add_argument('--reference',type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    reference=(args.reference or root/'results/quality-target-r2-20261012').resolve()
    dataset=reference/'dataset/scale-10000'
    output=args.output.resolve(); output.mkdir(parents=True,exist_ok=False)
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.feature,text=True).strip()=='dbb2fe0bec685ae7a343b1d2060d0553a98e20d2'
    assert not subprocess.check_output(['git','status','--porcelain','-uno'],cwd=args.feature)
    library=args.lite_build/'libvsag-lite.so'
    assert sha(library)=='3171d0d7e86866f02096db130f1937eb13c5115cbe993d4a6feb2e9987774330'
    commands=[]
    env={k:v for k,v in os.environ.items() if not k.startswith('VSAG_')}
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',VSAG_GRAPH_DEGREE='16',VSAG_GRAPH_EF='128',VSAG_GRAPH_QUERY_EF='512')
    def run(tag,argv):
        with (output/(tag+'.stdout')).open('w') as stdout,(output/(tag+'.stderr')).open('w') as stderr:
            completed=subprocess.run([str(x) for x in argv],env=env,stdout=stdout,stderr=stderr)
        commands.append(dict(tag=tag,argv=[str(x) for x in argv],exit=completed.returncode))
        write_json(output/'commands.json',commands)
        assert completed.returncode==0,tag
    prior=json.loads((reference/'rows.json').read_text())
    metadata=dict(library_sha256=sha(library),source_head='dbb2fe0bec685ae7a343b1d2060d0553a98e20d2',
        cpp_sha256=sha(root/'trace_query_losses.cpp'),runner_sha256=sha(Path(__file__)),reference=str(reference),
        inputs={name:sha(dataset/name) for name in ['base.fvecs','queries.fvecs','groundtruth.ivecs']},ef=2048,storage={})
    (output/'source-trace_query_losses.cpp').write_bytes((root/'trace_query_losses.cpp').read_bytes())
    (output/'source-run_query_loss_trace.py').write_bytes(Path(__file__).read_bytes())
    with tempfile.TemporaryDirectory(prefix='vsag-loss-trace-',dir='/dev/shm') as directory:
        scratch=Path(directory)
        builder_source=scratch/'builder.cpp'
        builder_source.write_bytes((reference/'source-graph_crud_quality.cpp').read_bytes())
        (output/'source-builder.cpp').write_bytes(builder_source.read_bytes())
        flags=['c++','-std=c++17','-O3','-DNDEBUG','-Wall','-Wextra','-Wpedantic','-I'+str(args.feature/'include'),'-I'+str(args.feature/'src')]
        link=['-L'+str(args.lite_build),'-Wl,-rpath,'+str(args.lite_build),'-lvsag-lite']
        run('compile-builder',flags+[builder_source]+link+['-o',scratch/'builder'])
        run('compile-trace',flags+[root/'trace_query_losses.cpp']+link+['-o',scratch/'trace'])
        for storage in ['fp32','fp16','rabitq8']:
            env['VSAG_GRAPH_STORAGE']=storage
            snapshot=scratch/(storage+'.snapshot')
            run(storage+'-build',['taskset','-c','0',scratch/'builder',dataset,snapshot,0,1,output/(storage+'.build.csv')])
            expected={r['snapshot_sha256'] for r in prior if r['side']=='lite' and r['storage']==storage}
            assert len(expected)==1 and sha(snapshot) in expected
            metadata['storage'][storage]=dict(snapshot_sha256=sha(snapshot),bytes=snapshot.stat().st_size)
            run(storage+'-trace',['taskset','-c','0',scratch/'trace',snapshot,dataset,storage,output/storage,2048])
            assert sha(snapshot) in expected
            with (output/(storage+'.neighbors.csv')).open() as stream: actual=list(csv.DictReader(stream))
            expected_rows=[]
            for suffix in ['ef2048-calibration','heldout']:
                with (reference/(storage+'-0-lite-'+suffix+'.neighbors.csv')).open() as stream: expected_rows.extend(csv.DictReader(stream))
            assert len(actual)==len(expected_rows)==10000
            for left,right in zip(actual,expected_rows):
                assert [left[k] for k in ['query','rank','id']]==[right[k] for k in ['query','rank','id']]
                assert struct.pack('<f',float.fromhex(left['distance']))==struct.pack('<f',float.fromhex(right['distance']))
            print(storage,'fixed-snapshot/native-ordered-result parity PASS',flush=True)
        assert sha(library)==metadata['library_sha256']
    write_json(output/'identity.json',metadata)
    manifest={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()}
    write_json(output/'manifest.json',manifest)
    with tarfile.open(output/'raw.tar.gz','w:gz') as archive:
        for name in sorted(list(manifest)+['manifest.json']): archive.add(output/name,arcname=name,recursive=False)
    write_json(output/'archive.json',dict(bytes=(output/'raw.tar.gz').stat().st_size,sha256=sha(output/'raw.tar.gz'),members=len(manifest)+1))
    print('Diagnostic complete, no performance improvement claimed',flush=True)
if __name__=='__main__': main()
