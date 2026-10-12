#!/usr/bin/env python3
# Copyright 2024-present the vsag project
# SPDX-License-Identifier: Apache-2.0
"""Isolated FP32 nearest-selection candidate; no production checkout changes."""
import argparse
import difflib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import tarfile
from run_mixed_cpu_acceptance import sha, write_json

INSERT = '''        if (not fp16_) {
            std::vector<uint64_t> selected;
            selected.reserve(count);
            for (const auto& neighbor : *found) {
                const uint64_t slot = slots_.at(neighbor.id);
                if (slot == excluded) {
                    continue;
                }
                bool diverse = true;
                for (const auto accepted : selected) {
                    if (distance(accepted, slot) < neighbor.distance) {
                        diverse = false;
                        break;
                    }
                }
                if (diverse) {
                    selected.push_back(slot);
                    if (selected.size() == count) {
                        return selected;
                    }
                }
            }
            // Preserve Lite's degree-fill contract after heuristic selection.
            for (const auto& neighbor : *found) {
                const uint64_t slot = slots_.at(neighbor.id);
                if (slot != excluded and
                    std::find(selected.begin(), selected.end(), slot) == selected.end()) {
                    selected.push_back(slot);
                    if (selected.size() == count) {
                        break;
                    }
                }
            }
            return selected;
        }
'''
RECIPROCAL = '''            bool diverse_drop = false;
            if (not fp16_) {
                for (uint64_t i = neighbors.size(); i > 0; --i) {
                    const uint64_t candidate = ranked[i - 1].slot;
                    const bool safe = candidate == target ? not incoming_[candidate].empty()
                                                         : incoming_[candidate].size() > 1;
                    if (not safe) {
                        continue;
                    }
                    for (uint64_t j = 0; j + 1 < i; ++j) {
                        if (distance(ranked[j].slot, candidate) < ranked[i - 1].distance) {
                            dropped = i - 1;
                            diverse_drop = true;
                            break;
                        }
                    }
                    if (diverse_drop) {
                        break;
                    }
                }
            }
'''


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reciprocal',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    feature=Path('/home/ubuntu/project/vsag-pr2904-promotion-20261012')
    library=Path('/home/ubuntu/project/pr2904-promotion-audit-20261012/release-final/libvsag-lite.so')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=feature,text=True).strip()
    assert head=='dbb2fe0bec685ae7a343b1d2060d0553a98e20d2'
    assert not subprocess.check_output(['git','status','--porcelain','-uno'],cwd=feature)
    assert sha(library)=='3171d0d7e86866f02096db130f1937eb13c5115cbe993d4a6feb2e9987774330'
    env={k:v for k,v in os.environ.items() if not k.startswith('VSAG_')}
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        VSAG_GRAPH_STORAGE='fp32',VSAG_GRAPH_DEGREE='16',VSAG_GRAPH_EF='128',VSAG_GRAPH_QUERY_EF='512')
    commands=[]
    def run(tag,argv):
        with (out/(tag+'.stdout')).open('w') as stdout,(out/(tag+'.stderr')).open('w') as stderr:
            done=subprocess.run([str(v) for v in argv],env=env,stdout=stdout,stderr=stderr)
        commands.append(dict(tag=tag,argv=[str(v) for v in argv],exit=done.returncode))
        write_json(out/'commands.json',commands)
        assert done.returncode==0,tag
    with tempfile.TemporaryDirectory(prefix='vsag-build-diversity-',dir='/dev/shm') as directory:
        scratch=Path(directory);source=scratch/'source'
        for name in ['src','include','lite']:
            shutil.copytree(feature/name,source/name)
        shutil.copyfile(feature/'.clang-format',source/'.clang-format')
        path=source/'src/lite/graph_backend.cpp';old=path.read_text()
        anchor='''        std::vector<uint64_t> result;
        result.reserve(count);
        for (const auto& neighbor : *found) {'''
        assert old.count(anchor)==1
        path.write_text(old.replace(anchor,INSERT+anchor))
        run('format',['clang-format-15','-i',path])
        new=path.read_text()
        (out/'candidate.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/src/lite/graph_backend.cpp',tofile='b/src/lite/graph_backend.cpp')))
        run('configure',['cmake','-S',source/'lite','-B',scratch/'build','-DCMAKE_BUILD_TYPE=Release','-DENABLE_RABITQ_LITE_BACKEND=ON','-DENABLE_TESTS=OFF'])
        run('build-nearest',['cmake','--build',scratch/'build','--target','vsag_lite','--parallel','2'])
        if args.reciprocal:
            nearest_dir=scratch/'nearest';nearest_dir.mkdir()
            shutil.copyfile(scratch/'build/libvsag-lite.so',nearest_dir/'libvsag-lite.so')
            anchor_drop='''            uint64_t dropped = max_degree_;
            for (uint64_t i = neighbors.size(); i > 0; --i) {'''
            assert path.read_text().count(anchor_drop)==1
            path.write_text(path.read_text().replace(anchor_drop,
                '            uint64_t dropped = max_degree_;\n'+RECIPROCAL+
                '            for (uint64_t i = neighbors.size(); i > 0 and not diverse_drop; --i) {'))
            run('format-reciprocal',['clang-format-15','-i',path])
            (out/'candidate.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),path.read_text().splitlines(True),fromfile='a/src/lite/graph_backend.cpp',tofile='b/src/lite/graph_backend.cpp')))
            run('build-reciprocal',['cmake','--build',scratch/'build','--target','vsag_lite','--parallel','2'])
            write_json(out/'nearest-control.json',dict(sha256=sha(nearest_dir/'libvsag-lite.so'),bytes=(nearest_dir/'libvsag-lite.so').stat().st_size))
            env['VSAG_PILOT_RECIPROCAL']='1'

        run('build',['cmake','--build',scratch/'build','--target','vsag_lite','--parallel','2'])
        candidate=scratch/'build/libvsag-lite.so'
        write_json(out/'identity.json',dict(source_head=head,control_sha256=sha(library),candidate_sha256=sha(candidate),candidate_bytes=candidate.stat().st_size,patch_sha256=sha(out/'candidate.patch'),scope='FP32 nearest + reciprocal safe occlusion' if args.reciprocal else 'FP32 nearest only'))
        builder=root/'results/quality-target-r2-20261012/source-graph_crud_quality.cpp'
        (out/'source-builder.cpp').write_bytes(builder.read_bytes())
        run('compile-builder',['c++','-std=c++17','-O3','-DNDEBUG','-I'+str(feature/'include'),builder,'-L'+str(library.parent),'-Wl,-rpath,'+str(library.parent),'-lvsag-lite','-o',scratch/'builder'])
        rows=[]
        datasets={'gist10k':root/'results/quality-target-r2-20261012/dataset/scale-10000',
            'sift10k':Path('/home/ubuntu/project/vsag-lite-datasets/sift/prepared-10k-100k/scale-10000')}
        for dataset,inputs in datasets.items():
            write_json(out/(dataset+'.inputs.json'),{name:dict(sha256=sha(inputs/name),bytes=(inputs/name).stat().st_size) for name in ['base.fvecs','queries.fvecs','groundtruth.ivecs']})
            for trial in range(3):
                order=['control','nearest','candidate'] if args.reciprocal else ['control','candidate']
                if args.reciprocal: order=order[trial:]+order[:trial]
                elif trial%2: order.reverse()
                for side in order:
                    tag=f'{dataset}-{trial}-{side}';stem=out/tag
                    env['LD_LIBRARY_PATH']=str(nearest_dir if side=='nearest' else candidate.parent if side=='candidate' else library.parent)
                    run(tag,['taskset','-c','0',scratch/'builder',inputs,scratch/(tag+'.snapshot'),0,1,str(stem)+'.queries.csv'])
                    lines=(out/(tag+'.stdout')).read_text().splitlines()
                    import csv
                    pos=next(i for i,line in enumerate(lines) if line.startswith('base_count,query_count,'))
                    metrics=next(csv.DictReader(lines[pos:pos+2]))
                    rows.append(dict(dataset=dataset,trial=trial,side=side,metrics=metrics))
                    write_json(out/'rows.json',rows)
                    print(tag,metrics['recall_at_k'],metrics['search_p50_us'],metrics['build_ms'],flush=True)
        assert sha(library)=='3171d0d7e86866f02096db130f1937eb13c5115cbe993d4a6feb2e9987774330'
    assert not subprocess.check_output(['git','status','--porcelain','-uno'],cwd=feature)
    manifest={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}
    write_json(out/'manifest.json',manifest)
    with tarfile.open(out/'raw.tar.gz','w:gz') as archive:
        for name in sorted(list(manifest)+['manifest.json']):archive.add(out/name,arcname=name,recursive=False)
    write_json(out/'archive.json',dict(bytes=(out/'raw.tar.gz').stat().st_size,sha256=sha(out/'raw.tar.gz'),members=len(manifest)+1))
if __name__=='__main__':main()
