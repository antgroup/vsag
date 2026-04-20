# Roadmap

As AI capabilities keep advancing and strong open-source LLMs become widespread, demand for
unstructured-data retrieval has exploded. Vector algorithms are the cornerstone of unstructured
retrieval, and the VSAG community will keep investing in algorithmic research to help partners
improve retrieval performance, reduce latency, and cut costs.

In 2025 we plan to ship the first major release:

- VSAG 1.0 provides comprehensive support for both graph-based and inverted-index structures, as
  well as in-memory and memory-plus-disk hybrid retrieval modes, delivering low memory cost and
  outstanding search performance.

Planned algorithms and features:

- Support for common data types to cover diverse unstructured retrieval scenarios
  - FP32 vectors: mainstream retrieval scenarios
  - INT8, BF16, FP16 vectors: adapt to quantized embedding models without extra storage overhead
  - Sparse vectors: extending text-retrieval workloads
- Fully optimized core index types covering the majority of retrieval scenarios
  - Graph index HGraph: high precision and low latency
  - Inverted index IVF: large K and batch query workloads
- Rich quantization options for the memory/recall trade-off
  - RabitQ (BQ): ultra-high compression with minimal memory
  - PQ: flexible compression ratios for accuracy-tolerant scenarios
  - SQ4, SQ8: standard quantization with minor recall loss and large memory/perf gains
- Multi-platform instruction support to simplify distribution
  - x86_64: SSE, AVX, AVX2, AVX-512
  - ARM: NEON, SVE
  - Optional matrix-multiplication libraries: Intel MKL, OpenBLAS
- Resource isolation and fine-grained runtime configurability
  - Memory: per-index allocators, enabling tenant-level memory management
  - CPU: injectable thread pools to boost write and search throughput

Beyond these, there is much more we want to discuss, design, and build in the open-source
community — follow the VSAG project to stay up to date!
