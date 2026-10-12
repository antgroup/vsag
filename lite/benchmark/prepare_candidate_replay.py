#!/usr/bin/env python3
# Copyright 2024-present the vsag project
# SPDX-License-Identifier: Apache-2.0
import argparse
from pathlib import Path
import struct
import sys
sys.path.insert(0,'/home/ubuntu/project/vsag-lite-datasets/sift/python-deps')
import numpy as np
from run_mixed_cpu_acceptance import sha, write_json

def decode(path):
    data=path.read_bytes();dim=struct.unpack_from('<i',data)[0]
    records=np.frombuffer(data,dtype='<f4').reshape(-1,dim+1)
    assert np.all(records[:,0].view('<i4')==dim)
    return records[:,1:].copy()
def encode(rows):
    result=np.empty((len(rows),rows.shape[1]+1),dtype='<f4')
    result[:,0].view('<i4')[:]=rows.shape[1];result[:,1:]=rows
    return result.tobytes()
def truth(base,queries):
    vectors=base.astype(np.float64);output=[]
    for query in queries:
        delta=vectors-query.astype(np.float64);distances=np.einsum('ij,ij->i',delta,delta)
        ids=np.lexsort((np.arange(len(base)),distances))[:10]
        output.append(struct.pack('<11i',10,*ids))
    return b''.join(output)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('dataset',type=Path);parser.add_argument('output',type=Path);parser.add_argument('receipt',type=Path)
    args=parser.parse_args();args.output.mkdir(exist_ok=False)
    base=decode(args.dataset/'base.fvecs');queries=decode(args.dataset/'queries.fvecs')[:100]
    assert len(base)==10000 and len(queries)==100
    initial=truth(base,queries)
    assert initial==(args.dataset/'groundtruth.ivecs').read_bytes()[:4400]
    current=base.copy();replacements=[]
    for round in range(3):
        ids=(round*65537+np.arange(10000)*8191)%10000;assert len(set(ids))==10000
        donors=(ids+7919*(round+1))%10000
        changed=(np.float32(.875)*current[ids]+np.float32(.125)*base[donors]).astype('<f4')
        assert np.all(np.isfinite(changed)) and np.all(np.any(changed!=current[ids],axis=1))
        replacements.append(changed);current[ids]=changed
    (args.output/'base.fvecs').symlink_to((args.dataset/'base.fvecs').resolve())
    (args.output/'queries.fvecs').write_bytes(encode(queries))
    (args.output/'groundtruth.ivecs').write_bytes(initial)
    (args.output/'changed-groundtruth.ivecs').write_bytes(truth(current,queries))
    (args.output/'replacement.fvecs').write_bytes(encode(np.concatenate(replacements)))
    write_json(args.receipt,dict(rounds=3,crud_ops=10000,queries=100,recipe='FP32 .875*current+.125*original donor; no post-normalization',numpy_version=np.__version__,schedule='(round*65537+operation*8191)%10000',truth='Direct FP64 differences, ties by ascending ID',source={n:sha(args.dataset/n) for n in ['base.fvecs','queries.fvecs','groundtruth.ivecs']},files={p.name:sha(p) for p in args.output.iterdir()}))
if __name__=='__main__':main()
