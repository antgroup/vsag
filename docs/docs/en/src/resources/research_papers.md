# Research Papers

## 1. Effective and General Distance Computation for Approximate Nearest Neighbor Search [[ICDE'25]](http://arxiv.org/abs/2404.16322)

Approximate K-nearest-neighbor (AKNN) search in high-dimensional spaces is a key and challenging
problem. Distance computation dominates AKNN runtime, and existing approaches rely on approximate
distances to gain efficiency, usually at the cost of accuracy. The state-of-the-art ADSampling
uses random projection to estimate distances and a correction step to mitigate accuracy loss, but
is limited in both effectiveness and generality because both steps depend on random projection.
This work improves distance computation by using data-aware orthogonal projections and a
data-driven correction procedure decoupled from the approximation step. Extensive experiments
show 1.6×–2.1× speedups over ADSampling on real-world datasets with higher accuracy.

> Integrated into VSAG under the name BSA; used to reduce the amount of high-precision re-ranking
> data inside disk-based indexes.

## 2. VSAG: An Optimized Search Framework for Graph-based Approximate Nearest Neighbor Search [[VLDB'25]](http://arxiv.org/abs/2503.17911)

Approximate nearest-neighbor search (ANNS) is foundational to vector databases and AI
infrastructure. Recent graph-based ANNS algorithms deliver both high accuracy and practical
efficiency, but production performance is still limited by random memory access patterns and
expensive distance computations. Moreover, graph-based ANNS is highly parameter-sensitive, and
finding optimal parameters traditionally requires repeatedly rebuilding the index. This paper
introduces VSAG, an open-source framework that targets these issues in production. VSAG is widely
deployed across Ant Group services and combines three key optimizations: (i) efficient memory
access via prefetching and cache-friendly vector layout to reduce L3 misses; (ii) automatic
parameter tuning without rebuilding the index; and (iii) efficient distance computation leveraging
modern hardware, scalar quantization, and low-precision fallbacks. On real-world datasets VSAG
matches or exceeds state-of-the-art accuracy while achieving up to 4× higher throughput than
HNSWlib.

> Integrated into VSAG; enabled through the [`Tune`](../advanced/optimizer.md) API (historically
> called the "ELP Optimizer" and implemented behind the `use_elp_optimizer` key).

## 3. EnhanceGraph: A Continuously Enhanced Graph-based Index for High-dimensional Approximate Nearest Neighbor Search [[arxiv]](https://arxiv.org/abs/2506.13144)

Driven by rapid progress in deep learning, high-dimensional ANNS has received growing attention.
We observe that graph-based indexes generate large amounts of search and construction logs over
their lifetime, but static indexes fail to exploit these valuable signals. This paper proposes
EnhanceGraph, a framework that folds both log types into a novel structure called a **conjugate
graph** to improve search quality. Guided by theoretical analysis and observations of the
limitations of graph-based indexes, we propose several optimisations: for search logs, the
conjugate graph stores edges from local optima to the global optimum to strengthen routing;
for construction logs it stores edges pruned from the proximity graph to improve k-NN recall.
Experiments on public and real industrial datasets show EnhanceGraph significantly improves
accuracy without sacrificing search efficiency, with recall gains reaching from 41.74% to 93.42%.
EnhanceGraph has been integrated into VSAG.

> Integrated into VSAG on HNSW-like indexes; enable via the
> [`use_conjugate_graph`](../advanced/enhance_graph.md) parameter.

## 4. SINDI: an Efficient Index for Approximate Maximum Inner Product Search on Sparse Vectors [[arxiv]](https://arxiv.org/abs/2509.08395)

Maximum inner product search (MIPS) on sparse vectors is critical for multi-way retrieval used in
retrieval-augmented generation (RAG). Recent inverted-index and graph-based algorithms combine
high accuracy with practical efficiency, but production performance is often limited by redundant
distance computations and frequent random memory accesses. Furthermore, the compressed storage
format of sparse vectors makes it hard to take advantage of SIMD acceleration. This paper presents
the Sparse Inverted Non-redundant Distance Index (SINDI), which combines three key optimisations:
(i) efficient inner-product computation that uses SIMD acceleration and eliminates redundant
identifier lookups for batched computations; (ii) memory-friendly design that replaces random
access on raw vectors with sequential access on inverted lists, greatly reducing memory-access
latency; and (iii) vector pruning that keeps only the non-zero entries with larger magnitude, so
query throughput improves while accuracy is preserved. On real-world datasets SINDI is
state-of-the-art across scales, languages, and models. On MsMarco, for Recall@50 above 99%, SINDI
delivers 4.2×–26.4× higher single-thread QPS than SEISMIC and PyANNs. SINDI has been integrated
into VSAG.

> SINDI is an index type inside VSAG.
