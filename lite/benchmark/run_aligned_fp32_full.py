#!/usr/bin/env python3
# Copyright 2024-present the vsag project
# SPDX-License-Identifier: Apache-2.0
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import statistics
import struct
import subprocess
import tarfile
import tempfile
from run_mixed_cpu_acceptance import sha, write_json
from run_quality_target import audit_query

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parent
    feature=Path('/home/ubuntu/project/vsag-pr2904-promotion-20261012')
    full=Path('/home/ubuntu/project/vsag-full-strict-install-20261012')
    raw=Path('/home/ubuntu/project/vsag-lite-datasets/gist/hf-100k/base-100k.fbin')
    full_receipt=json.loads((root/'results/full-strict-build-r1-20261012/receipt.json').read_text())
    assert sha(full/'lib/libvsag.so')==full_receipt['library_sha256']
    strict=json.loads((root/'results/full-strict-build-r1-20261012/strict-fp-receipt.json').read_text())
    assert strict['audited_core_commands']>50 and strict['production_source_unchanged']
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=feature,text=True).strip()=='dbb2fe0bec685ae7a343b1d2060d0553a98e20d2'
    assert not subprocess.check_output(['git','status','--porcelain','-uno'],cwd=feature)
    env={k:v for k,v in os.environ.items() if not k.startswith('VSAG_')}
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',VSAG_FULL_PROFILE='small',VSAG_GRAPH_STORAGE='fp32',VSAG_GRAPH_DEGREE='16',VSAG_GRAPH_EF='128',VSAG_GRAPH_QUERY_EF='512')
    commands=[];rows=[]
    def run(tag,argv,cwd=None):
        result=subprocess.run([str(x) for x in argv],cwd=cwd,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        (out/(tag+'.stdout')).write_text(result.stdout);(out/(tag+'.stderr')).write_text(result.stderr)
        commands.append(dict(tag=tag,argv=[str(x) for x in argv],exit=result.returncode,library=env.get('LD_LIBRARY_PATH')))
        write_json(out/'commands.json',commands);assert result.returncode==0,tag
        return result.stdout
    protocol=dict(storage='fp32',degree=16,construction=128,cpu=0,threads=1,trials=3,
        gist_query_source='GIST base source rows10000:11000, disjoint from index rows0:10000 and earlier official query cohort',
        frozen_gist_budgets={'control':512,'candidate':256,'full':512},
        frozen_sift_budgets={'control':384,'candidate':128,'full':128},
        target=.990,target_is_diagnostic=True,sift_is_development=True,
        compiler='Lite and isolated native Full core -O3/-g without fast-math. Not identical complete flags or floating evaluation order.',
        crud='Existing candidate CTest including allocation rollback; not long-term CRUD acceptance')
    write_json(out/'protocol.json',protocol)
    with tempfile.TemporaryDirectory(prefix='vsag-aligned-',dir='/dev/shm') as temporary:
        scratch=Path(temporary);libs={};identities={}
        patch=root/'results/reciprocal-diversity-r1-20261012/candidate.patch'
        for side in ['control','candidate']:
            source=scratch/side/'source'
            for name in ['src','include','lite','cmake','extern']:shutil.copytree(feature/name,source/name)
            if side=='candidate':run(side+'-apply',['git','apply',patch],source)
            build=scratch/side/'build'
            run(side+'-configure',['cmake','-S',source/'lite','-B',build,'-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CXX_FLAGS_RELEASE=-O3 -g -DNDEBUG','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON','-DENABLE_RABITQ_LITE_BACKEND=ON','-DENABLE_TESTS=ON','-DFETCHCONTENT_SOURCE_DIR_CATCH2=/home/ubuntu/vsag-deps/extracted-20260903/catch2-src'])
            run(side+'-build',['cmake','--build',build,'--parallel','2'])
            run(side+'-tests',['ctest','--test-dir',build,'--output-on-failure'])
            shutil.copyfile(build/'compile_commands.json',out/(side+'-compile_commands.json'))
            compile_records=json.loads((build/'compile_commands.json').read_text())
            core=[r for r in compile_records if '/src/lite/' in r['file'] and not r['file'].endswith('_test.cpp')]
            assert core and all('-O3' in r['command'] and '-Ofast' not in r['command'] and '-ffast-math' not in r['command'] for r in core)
            libs[side]=build;identities[side]=dict(sha256=sha(build/'libvsag-lite.so'),bytes=(build/'libvsag-lite.so').stat().st_size)
        identities['full']=dict(sha256=sha(full/'lib/libvsag.so'),bytes=(full/'lib/libvsag.so').stat().st_size)
        write_json(out/'libraries.json',identities)
        flags=['c++','-std=c++17','-O3','-DNDEBUG']
        for side in ['control','candidate','full']:
            include=full/'include' if side=='full' else feature/'include'
            lib=full/'lib' if side=='full' else libs[side]
            common=flags+['-I'+str(include),'-L'+str(lib),'-Wl,-rpath,'+str(lib)]
            link='-lvsag' if side=='full' else '-lvsag-lite'
            builder=root/'results/quality-target-r2-20261012'/('source-full_rabitq_dataset_main.cpp' if side=='full' else 'source-graph_crud_quality.cpp')
            run(side+'-builder-compile',common+[builder,link,'-o',scratch/(side+'-builder')])
            run(side+'-query-compile',common+(['-DVSAG_BENCH_FULL'] if side=='full' else [])+[root/'query_fixed_snapshot.cpp',link,'-o',scratch/(side+'-query')])
        run('prepare-compile',flags+['-I'+str(feature/'src'),root/'prepare_gist.cpp','-L'+str(libs['control']),'-Wl,-rpath,'+str(libs['control']),'-lvsag-lite','-o',scratch/'prepare'])
        with raw.open('rb') as stream:
            count,dim=struct.unpack('<II',stream.read(8));assert count>=11000 and dim==960
            stream.seek(8+10000*dim*4);payload=stream.read(1000*dim*4);assert len(payload)==1000*dim*4
        (scratch/'fresh-queries.fbin').write_bytes(struct.pack('<II',1000,dim)+payload)
        write_json(out/'fresh-query-source.json',dict(source=str(raw),source_sha256=sha(raw),begin=10000,end=11000,payload_sha256=sha(scratch/'fresh-queries.fbin')))
        run('prepare',[scratch/'prepare',raw,scratch/'fresh-queries.fbin',out/'dataset',10000,1000])
        assert sha(out/'dataset/scale-10000/base.fvecs')==sha(root/'results/quality-target-r2-20261012/dataset/scale-10000/base.fvecs')
        datasets={'gist10k':out/'dataset/scale-10000','sift10k':Path('/home/ubuntu/project/vsag-lite-datasets/sift/prepared-10k-100k/scale-10000')}
        for dataset,inputs in datasets.items():
            truths=[];data=(inputs/'groundtruth.ivecs').read_bytes()
            for offset in range(0,len(data),44):truths.append(set(struct.unpack_from('<11i',data,offset)[1:]))
            write_json(out/(dataset+'.inputs.json'),{n:dict(sha256=sha(inputs/n),bytes=(inputs/n).stat().st_size) for n in ['base.fvecs','queries.fvecs','groundtruth.ivecs']})
            budgets=protocol['frozen_gist_budgets' if dataset=='gist10k' else 'frozen_sift_budgets']
            for trial in range(3):
                sides=['control','candidate','full'];sides=sides[trial:]+sides[:trial]
                for side in sides:
                    tag=f'{dataset}-{trial}-{side}';snapshot=scratch/(tag+'.snapshot');stem=out/tag;budget=budgets[side]
                    env['LD_LIBRARY_PATH']=str(full/'lib' if side=='full' else libs[side])
                    env['VSAG_GRAPH_QUERY_EF']=str(budget)
                    argv=[scratch/(side+'-builder'),inputs,snapshot,'fp32',budget,0] if side=='full' else [scratch/(side+'-builder'),inputs,snapshot,0,1,str(stem)+'.build.csv']
                    run(tag+'-build',['taskset','-c','0']+argv)
                    config=scratch/(tag+'.config.json')
                    if side=='full':
                        parameters=json.loads(Path(str(snapshot)+'.parameters.json').read_text());config.write_text(json.dumps(parameters['build']));write_json(out/(tag+'.parameters.json'),parameters)
                    digest=sha(snapshot)
                    output=run(tag+'-query',['taskset','-c','0',scratch/(side+'-query'),snapshot,inputs/'queries.fvecs',960 if dataset=='gist10k' else 128,10000,budget,0,len(truths),stem,config,'fp32',0])
                    assert sha(snapshot)==digest
                    audit=audit_query(stem,truths,0,len(truths))
                    lines=output.splitlines();pos=next(i for i,line in enumerate(lines) if line.startswith('queries,ef,'))
                    resources=next(csv.DictReader(lines[pos:pos+2]))
                    rows.append(dict(dataset=dataset,trial=trial,side=side,budget=budget,snapshot_sha256=digest,snapshot_bytes=snapshot.stat().st_size,resources=resources,**audit))
                    write_json(out/'rows.json',rows);print(tag,audit,flush=True)
        summary=[]
        for dataset in datasets:
            for side in ['control','candidate','full']:
                group=[r for r in rows if r['dataset']==dataset and r['side']==side]
                summary.append(dict(dataset=dataset,side=side,budget=group[0]['budget'],all_target_met=all(r['recall']>=.990 for r in group),**{k:statistics.median(r[k] for r in group) for k in ['recall','p50_us','p99_us','snapshot_bytes']},phase_cpu_us=statistics.median(float(r['resources']['phase_cpu_us']) for r in group),process_peak_rss_kib=statistics.median(float(r['resources']['process_peak_rss_kib']) for r in group)))
        write_json(out/'summary.json',summary)
        replay_rows=[]
        for dataset,inputs in datasets.items():
            fixture=scratch/(dataset+'-replay')
            run(dataset+'-replay-prepare',['python3',root/'prepare_candidate_replay.py',inputs,fixture,out/(dataset+'-replay-inputs.json')])
            shutil.copyfile(fixture/'changed-groundtruth.ivecs',out/(dataset+'-replay-truth.ivecs'))
            budget=protocol['frozen_gist_budgets' if dataset=='gist10k' else 'frozen_sift_budgets']['candidate']
            env.update(LD_LIBRARY_PATH=str(libs['candidate']),VSAG_GRAPH_QUERY_EF=str(budget),VSAG_GRAPH_REPLACEMENTS=str(fixture/'replacement.fvecs'),VSAG_GRAPH_CRUD_MODE='all')
            tag=dataset+'-candidate-replay'
            output=run(tag,['taskset','-c','0',scratch/'candidate-builder',fixture,scratch/(tag+'.snapshot'),3,10000,str(out/tag)+'.csv'])
            lines=output.splitlines();pos=next(i for i,line in enumerate(lines) if line.startswith('base_count,query_count,'))
            metrics=next(csv.DictReader(lines[pos:pos+2]))
            with (out/(tag+'.csv.operations.csv')).open() as stream:operations=list(csv.DictReader(stream))
            assert len(operations)==90000
            for cycle in range(30000):
                selected=operations[cycle*3:(cycle+1)*3]
                assert [int(r['operation']) for r in selected]==[0,1,2]
                assert all(int(r['cycle'])==cycle and int(r['id'])==((cycle//10000)*65537+(cycle%10000)*8191)%10000 for r in selected)
            replay_rows.append(dict(dataset=dataset,operations=len(operations),metrics=metrics,scope='Three whole-index passes; real-value Update/Remove/Add; live count and Save/Load result identity verified by consumer. Not concurrent/endurance duration/periodic snapshots.'))
            write_json(out/'replay-summary.json',replay_rows)
            env.pop('VSAG_GRAPH_REPLACEMENTS');env.pop('VSAG_GRAPH_CRUD_MODE')
            print(tag,metrics,flush=True)
        sanitized=scratch/'sanitized'
        run('candidate-sanitizer-configure',['cmake','-S',scratch/'candidate/source/lite','-B',sanitized,'-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CXX_FLAGS_RELEASE=-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer','-DCMAKE_SHARED_LINKER_FLAGS=-fsanitize=address,undefined','-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address,undefined','-DENABLE_RABITQ_LITE_BACKEND=ON','-DENABLE_TESTS=ON','-DFETCHCONTENT_SOURCE_DIR_CATCH2=/home/ubuntu/vsag-deps/extracted-20260903/catch2-src'])
        run('candidate-sanitizer-build',['cmake','--build',sanitized,'--parallel','2'])
        env.update(ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1',LD_LIBRARY_PATH=str(sanitized))
        run('candidate-sanitizer-tests',['ctest','--test-dir',sanitized,'--output-on-failure'])
    manifest={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}
    write_json(out/'manifest.json',manifest)
    with tarfile.open(out/'raw.tar.gz','w:gz') as archive:
        for name in sorted(list(manifest)+['manifest.json']):archive.add(out/name,arcname=name,recursive=False)
    write_json(out/'archive.json',dict(sha256=sha(out/'raw.tar.gz'),bytes=(out/'raw.tar.gz').stat().st_size,members=len(manifest)+1))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
