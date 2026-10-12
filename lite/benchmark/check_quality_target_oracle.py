#!/usr/bin/env python3
# Copyright 2024-present the vsag project
# SPDX-License-Identifier: Apache-2.0
"""Independent direct FP64 L2 oracle cross-check, outside all measured runs."""
import argparse
import csv
from pathlib import Path
import struct
import subprocess
import tempfile
from run_mixed_cpu_acceptance import sha, write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    parser.add_argument('--dataset',type=Path)
    args=parser.parse_args()
    dataset=args.dataset or args.output/'dataset/scale-10000'
    source=Path(__file__).with_name('quality_target_fp64.cpp')
    with tempfile.TemporaryDirectory(prefix='vsag-fp64-oracle-',dir='/dev/shm') as directory:
        binary=Path(directory)/'oracle'
        subprocess.run(['c++','-std=c++17','-O3','-DNDEBUG',str(source),'-o',str(binary)],check=True)
        with (args.output/'fp64-neighbors.csv').open('w') as stream:
            subprocess.run([str(binary),str(dataset/'base.fvecs'),str(dataset/'queries.fvecs')],stdout=stream,check=True)
    with (args.output/'fp64-neighbors.csv').open() as stream: rows=list(csv.DictReader(stream))
    assert len(rows)==10000
    truth=(dataset/'groundtruth.ivecs').read_bytes()
    mismatches=[]
    for query in range(1000):
        selected=rows[query*10:(query+1)*10]
        assert [(int(r['query']),int(r['rank'])) for r in selected]==[(query,rank) for rank in range(10)]
        actual=[int(r['id']) for r in selected]
        record=struct.unpack_from('<11i',truth,query*44)
        assert record[0]==10
        if actual!=list(record[1:]): mismatches.append(dict(query=query,fp64=actual,recorded=list(record[1:])))
    receipt=dict(oracle='Direct C++ FP64 coordinate-difference squared-L2, ties by ascending ID; no Lite distance kernel or fast-math',
        source_sha256=sha(source),queries=1000,base_count=10000,mismatches=mismatches,
        inputs={name:sha(dataset/name) for name in ['base.fvecs','queries.fvecs','groundtruth.ivecs']})
    write_json(args.output/'fp64-oracle.json',receipt)
    assert not mismatches, 'FP64 oracle differs; inspect receipt, do not rewrite old results'
    print('FP64 oracle PASS: 1000 complete ordered top-10 records')
if __name__=='__main__': main()
