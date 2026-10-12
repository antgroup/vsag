#!/usr/bin/env python3
# Copyright 2024-present the vsag project
# SPDX-License-Identifier: Apache-2.0
"""Fixed-snapshot calibration, frozen common-target choices, then held-out queries."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import struct
import subprocess
import tarfile
import tempfile

from run_mixed_cpu_acceptance import sha, write_json, percentile
from run_pinned_full_acceptance import validate_fresh_receipt


def audit_query(prefix, truths, begin, end, count=10000):
    with Path(str(prefix)+'.queries.csv').open() as stream:
        times = list(csv.DictReader(stream))
    with Path(str(prefix)+'.neighbors.csv').open() as stream:
        neighbors = list(csv.DictReader(stream))
    assert [int(row['query']) for row in times] == list(range(begin, end))
    assert len(neighbors) == 10 * (end-begin)
    hits = 0
    for offset, query in enumerate(range(begin, end)):
        selected = neighbors[offset*10:(offset+1)*10]
        assert [(int(r['query']), int(r['rank'])) for r in selected] == [(query, rank) for rank in range(10)]
        ids = [int(r['id']) for r in selected]
        distances = [float.fromhex(r['distance']) for r in selected]
        assert len(set(ids)) == 10 and all(0 <= i < count for i in ids)
        assert all(math.isfinite(d) for d in distances)
        assert distances == sorted(distances)
        hits += len(set(ids) & truths[query])
    values = [float(row['latency_us']) for row in times]
    assert all(math.isfinite(v) and v >= 0 for v in values)
    return dict(queries=end-begin, hits=hits, recall=hits/(10*(end-begin)),
        p50_us=percentile(values, .5), p99_us=percentile(values, .99))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature', type=Path, default=Path('/home/ubuntu/project/vsag-pr2904-promotion-20261012'))
    parser.add_argument('--lite-build', type=Path, default=Path('/home/ubuntu/project/pr2904-promotion-audit-20261012/release-final'))
    parser.add_argument('--full-install', type=Path, default=Path('/home/ubuntu/project/vsag-full-fresh-install-20261012'))
    parser.add_argument('--source-repo', type=Path, default=Path('/home/ubuntu/project/vsag-lite-baseline-v01'))
    parser.add_argument('--source-ref', default='a62e62e08db3f0d7da2e4439785feed6fcf90257')
    parser.add_argument('--full-receipt', type=Path)
    parser.add_argument('--raw-gist', type=Path, default=Path('/home/ubuntu/project/vsag-lite-datasets/gist/hf-100k'))
    parser.add_argument('--prior-dataset', type=Path, default=Path('/home/ubuntu/project/vsag-lite-datasets/gist/prepared-10k-100k/scale-10000'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    feature = args.feature
    lite_build = args.lite_build
    full = args.full_install
    source = args.source_repo
    ref = args.source_ref
    expected = 'dbb2fe0bec685ae7a343b1d2060d0553a98e20d2'
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    receipt_path = args.full_receipt or root/'results/full-fresh-build-r4-20261012/receipt.json'
    receipt = json.loads(receipt_path.read_text())
    validate_fresh_receipt(receipt, expected, full)
    assert sha(full/'lib/libvsag.so') == receipt['library_sha256']
    assert sha(lite_build/'libvsag-lite.so') == '3171d0d7e86866f02096db130f1937eb13c5115cbe993d4a6feb2e9987774330'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=feature,text=True).strip() == expected
    assert not subprocess.check_output(['git','status','--porcelain','-uno'],cwd=feature)
    env = {k:v for k,v in os.environ.items() if not k.startswith('VSAG_')}
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',VSAG_FULL_PROFILE='small')
    commands = []
    def run(tag, argv):
        with (output/(tag+'.stdout')).open('w') as stdout, (output/(tag+'.stderr')).open('w') as stderr:
            result = subprocess.run([str(x) for x in argv],env=env,stdout=stdout,stderr=stderr)
        commands.append(dict(tag=tag,argv=[str(x) for x in argv],exit=result.returncode))
        write_json(output/'commands.json',commands)
        assert result.returncode == 0, tag
    grid = [128,256,512,1024,2048,4096]
    target = .990
    write_json(output/'protocol.json',dict(grid=grid,target=target,calibration=[0,100],heldout=[100,1000],
        degree=16,build_ef=128,trials=3,source_head=expected,full_receipt=receipt,
        cpp_sha256=sha(root/'query_fixed_snapshot.cpp'),runner_sha256=sha(Path(__file__)),
        warmup='one entire cohort before measured pass; no output IO inside measured phase',
        compiler_boundary='Full native -Ofast/-g versus Lite -O3; consumers both -O3',
        cpu_boundary='phase CPU includes result bookkeeping; API wall latency excludes dataset construction'))
    (output/'measured-runner.py').write_bytes(Path(__file__).read_bytes())
    (output/'source-query_fixed_snapshot.cpp').write_bytes((root/'query_fixed_snapshot.cpp').read_bytes())
    write_json(output/'raw-input-receipt.json',{name:dict(bytes=(args.raw_gist/name).stat().st_size,sha256=sha(args.raw_gist/name))
        for name in ['base-100k.fbin','queries.fbin']})
    with tempfile.TemporaryDirectory(prefix='vsag-quality-target-',dir='/dev/shm') as directory:
        scratch = Path(directory)
        for name in ['graph_crud_quality.cpp','full_rabitq_dataset_main.cpp']:
            (scratch/name).write_bytes(subprocess.check_output(['git','show',ref+':lite/benchmark/'+name],cwd=source))
            (output/('source-'+name)).write_bytes((scratch/name).read_bytes())
        (output/'source-prepare_gist.cpp').write_bytes((root/'prepare_gist.cpp').read_bytes())
        compile_flags = ['c++','-std=c++17','-O3','-DNDEBUG','-Wall','-Wextra','-Wpedantic']
        run('compile-prepare',compile_flags+['-I'+str(feature/'src'),root/'prepare_gist.cpp','-L'+str(lite_build),'-Wl,-rpath,'+str(lite_build),'-lvsag-lite','-o',scratch/'prepare'])
        dataset = output/'dataset/scale-10000'
        raw = args.raw_gist
        run('prepare',[scratch/'prepare',raw/'base-100k.fbin',raw/'queries.fbin',output/'dataset',10000,1000])
        old = args.prior_dataset
        assert sha(dataset/'base.fvecs') == sha(old/'base.fvecs')
        assert (dataset/'queries.fvecs').read_bytes()[:100*3844] == (old/'queries.fvecs').read_bytes()
        assert (dataset/'groundtruth.ivecs').read_bytes()[:100*44] == (old/'groundtruth.ivecs').read_bytes()
        write_json(output/'dataset-receipt.json',{name:dict(bytes=(dataset/name).stat().st_size,sha256=sha(dataset/name))
            for name in ['base.fvecs','queries.fvecs','groundtruth.ivecs','manifest.json']})
        truth_data = (dataset/'groundtruth.ivecs').read_bytes()
        truths = []
        for row in range(1000):
            values = struct.unpack_from('<11i',truth_data,row*44)
            assert values[0] == 10 and len(set(values[1:])) == 10
            truths.append(set(values[1:]))
        for side in ['lite','full']:
            prefix = full if side == 'full' else feature
            lib = full/'lib' if side == 'full' else lite_build
            flags = compile_flags + ['-I'+str(prefix/'include'),'-L'+str(lib),'-Wl,-rpath,'+str(lib)]
            link = '-lvsag' if side == 'full' else '-lvsag-lite'
            run('compile-'+side+'-builder',flags+[scratch/('full_rabitq_dataset_main.cpp' if side=='full' else 'graph_crud_quality.cpp'),link,'-o',scratch/(side+'-builder')])
            run('compile-'+side+'-query',flags+(['-DVSAG_BENCH_FULL'] if side=='full' else [])+[root/'query_fixed_snapshot.cpp',link,'-o',scratch/(side+'-query')])
        rows = []
        snapshots = {}
        for storage in ['fp32','fp16','rabitq8']:
            for trial in range(3):
                for side in (['full','lite'] if trial%2==0 else ['lite','full']):
                    tag = f'{storage}-{trial}-{side}'
                    snapshot = scratch/(tag+'.snapshot')
                    snapshots[tag] = snapshot
                    env.update(VSAG_GRAPH_STORAGE=storage,VSAG_GRAPH_DEGREE='16',VSAG_GRAPH_EF='128',VSAG_GRAPH_QUERY_EF='512')
                    argv = ([scratch/'full-builder',dataset,snapshot,'rabitq3x5' if storage=='rabitq8' else storage,512,0]
                            if side=='full' else [scratch/'lite-builder',dataset,snapshot,0,1,output/(tag+'.build.queries.csv')])
                    run(tag+'-build',['taskset','-c','0']+argv)
                    config = scratch/(tag+'.config.json')
                    if side == 'full':
                        parameters = json.loads(Path(str(snapshot)+'.parameters.json').read_text())
                        config.write_text(json.dumps(parameters['build']))
                        write_json(output/(tag+'.parameters.json'),parameters)
                    digest = sha(snapshot)
                    for ef in (grid if trial%2==0 else list(reversed(grid))):
                        stem = output/f'{tag}-ef{ef}-calibration'
                        run(stem.name,['taskset','-c','0',scratch/(side+'-query'),snapshot,dataset/'queries.fvecs',960,10000,ef,0,100,stem,config,storage,0])
                        audit = audit_query(stem,truths,0,100)
                        rows.append(dict(storage=storage,trial=trial,side=side,ef=ef,cohort='calibration',snapshot_sha256=digest,**audit))
                    assert sha(snapshot) == digest
                    write_json(output/'rows.json',rows)
                    print(tag,'calibration complete',flush=True)
        selections = {}
        for storage in ['fp32','fp16','rabitq8']:
            for side in ['lite','full']:
                qualified = [ef for ef in grid if all(r['recall']>=target for r in rows if r['storage']==storage and r['side']==side and r['ef']==ef)]
                selections[storage+'-'+side] = dict(ef=min(qualified) if qualified else max(grid),calibration_target_met=bool(qualified))
        write_json(output/'frozen-selections.json',selections)
        for storage in ['fp32','fp16','rabitq8']:
            for trial in range(3):
                for side in (['lite','full'] if trial%2==0 else ['full','lite']):
                    tag=f'{storage}-{trial}-{side}'
                    selected=selections[storage+'-'+side]
                    stem=output/(tag+'-heldout')
                    run(stem.name,['taskset','-c','0',scratch/(side+'-query'),snapshots[tag],dataset/'queries.fvecs',960,10000,selected['ef'],100,1000,stem,scratch/(tag+'.config.json'),storage,0])
                    audit=audit_query(stem,truths,100,1000)
                    digest=sha(snapshots[tag])
                    assert digest == next(r['snapshot_sha256'] for r in rows if r['storage']==storage and r['trial']==trial and r['side']==side)
                    rows.append(dict(storage=storage,trial=trial,side=side,ef=selected['ef'],cohort='heldout',snapshot_sha256=digest,**audit))
                    write_json(output/'rows.json',rows)
                    print(tag,'heldout',audit,flush=True)
    manifest={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file() and 'dataset' not in p.relative_to(output).parts}
    write_json(output/'manifest.json',manifest)
    with tarfile.open(output/'raw.tar.gz','w:gz') as archive:
        for name in sorted(list(manifest)+['manifest.json']): archive.add(output/name,arcname=name,recursive=False)
    write_json(output/'archive.json',dict(sha256=sha(output/'raw.tar.gz'),bytes=(output/'raw.tar.gz').stat().st_size,members=len(manifest)+1))
    print('COMPLETE calibration and frozen heldout; inspect target failures, not automatic acceptance',flush=True)


if __name__ == '__main__':
    main()
