# Copyright 2024-present the vsag project
# Licensed under the Apache License, Version 2.0.
"""Independent small-subset exhaustive oracle (numpy/h5py required)."""
import sys, struct
import numpy as np
import h5py
path,out,n,nq=sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4])
def parse(raw,count):
    rows=[]; offset=0
    for _ in range(count):
        length=struct.unpack_from('<I',raw,offset)[0]; offset+=4
        ids=np.frombuffer(raw,dtype='<u4',count=length,offset=offset); offset+=4*length
        vals=np.frombuffer(raw,dtype='<f4',count=length,offset=offset); offset+=4*length
        rows.append(dict(zip(map(int,ids),map(float,vals))))
    return rows
with h5py.File(path,'r') as f:
    base=f['train'][:n]; queries=f['test'][:nq]
    docs=parse(f['train_sparse'][:],n); sparseq=parse(f['test_sparse'][:],nq)
dense=queries@base.T
sparse=np.zeros((nq,n),np.float32)
for q,terms in enumerate(sparseq):
    for i,doc in enumerate(docs):
        score=np.float32(0)
        for t,v in terms.items():
            score=np.float32(score + np.float32(np.float32(v)*np.float32(doc.get(t,0.0))))
        sparse[q,i]=score
actual=np.fromfile(out+'/ground_truth.bin',dtype='<i8').reshape(3,nq,100)
for s,a in enumerate([.3,.5,.7]):
    a=float(np.float32(a)); scores=a*dense.astype(np.float64)+(1-a)*sparse.astype(np.float64)
    order=np.argsort(-scores,axis=1,kind='stable')[:,:100]
    mismatches=np.count_nonzero(order!=actual[s])
    print('alpha',a,'ordered mismatches',mismatches)
    if mismatches:
        for q in range(nq):
            if np.any(order[q]!=actual[s,q]):
                print('query',q,'membership difference',set(order[q])^set(actual[s,q]))
                for j in np.flatnonzero(order[q]!=actual[s,q]):
                    x,y=order[q,j],actual[s,q,j]; print(j,x,y,scores[q,x],scores[q,y])
    for q in range(nq):
        assert set(order[q])==set(actual[s,q])
        for j in np.flatnonzero(order[q]!=actual[s,q]):
            assert abs(scores[q,order[q,j]]-scores[q,actual[s,q,j]])<1e-5
print('Independent exhaustive dense + sparse gold verification passed')
