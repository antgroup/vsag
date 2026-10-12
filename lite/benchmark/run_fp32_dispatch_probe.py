#!/usr/bin/env python3
# Copyright 2024-present the vsag project
# SPDX-License-Identifier: Apache-2.0
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
from run_mixed_cpu_acceptance import sha, write_json

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--aligned-candidate',action='store_true')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parent
    lite_source=Path('/home/ubuntu/project/vsag-pr2904-promotion-20261012')
    full_source=Path('/home/ubuntu/project/vsag-full-fresh-source-20261012')
    lite_lib=Path('/home/ubuntu/project/pr2904-promotion-audit-20261012/release-final/libvsag-lite.so')
    full_lib=Path('/home/ubuntu/project/vsag-full-fresh-install-20261012/lib/libvsag.so')
    if args.aligned_candidate:
        lite_lib=Path('/home/ubuntu/project/aligned-fp32-followup-20261012/libvsag-lite.so')
        full_lib=Path('/home/ubuntu/project/vsag-full-strict-install-20261012/lib/libvsag.so')
        identities=json.loads((root/'results/aligned-fp32-full-r2-20261012/libraries.json').read_text())
        assert sha(lite_lib)==identities['candidate']['sha256'] and sha(full_lib)==identities['full']['sha256']
    else:
        assert sha(lite_lib)=='3171d0d7e86866f02096db130f1937eb13c5115cbe993d4a6feb2e9987774330'
        assert sha(full_lib)=='8740d2199518703883021d0d49b427e0e6a01be32a6866f561cfd86012f9c888'
    for source in [lite_source,full_source]:
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()=='dbb2fe0bec685ae7a343b1d2060d0553a98e20d2'
    commands=[];results=[]
    def run(tag,argv):
        result=subprocess.run([str(x) for x in argv],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        (out/(tag+'.stdout')).write_text(result.stdout);(out/(tag+'.stderr')).write_text(result.stderr)
        commands.append(dict(tag=tag,argv=[str(x) for x in argv],exit=result.returncode))
        write_json(out/'commands.json',commands)
        assert result.returncode==0,tag
        return result.stdout
    with tempfile.TemporaryDirectory(prefix='vsag-dispatch-',dir='/dev/shm') as temporary:
        scratch=Path(temporary)
        for side,source,library in [('lite',lite_source,lite_lib),('full',full_source,full_lib)]:
            run(side+'-compile',['c++','-std=c++17','-O3','-I'+str(source/'src')]+(['-DVSAG_BENCH_FULL'] if side=='full' else ['-DVSAG_LITE_HAS_X86_SIMD'])+[root/'fp32_dispatch_probe.cpp','-L'+str(library.parent),'-Wl,-rpath,'+str(library.parent),'-lvsag' if side=='full' else '-lvsag-lite','-o',scratch/side])
            lines=run(side+'-runtime',['taskset','-c','0',scratch/side]).splitlines()
            records=[line for line in lines if line.startswith('{"side":')]
            assert len(records)==1;result=json.loads(records[0])
            assert result['scalar_avx512']==1
            if side=='full':assert result['batch4_avx512']==1
            results.append(result)
    run('cpu',['lscpu'])
    write_json(out/'receipt.json',dict(audit='PASS',results=results,lite_sha256=sha(lite_lib),full_sha256=sha(full_lib),cpp_sha256=sha(root/'fp32_dispatch_probe.cpp'),boundary='Aligned candidate and strict native Full O3/g' if args.aligned_candidate else 'Full -Ofast/-g vs Lite -O3; runtime dispatch only; no timing measured'))
    print(json.dumps(results,indent=2))
if __name__=='__main__':main()
