<div align="center">
  <h1><img alt="vsag-pages" src="docs/banner.svg" width=500/></h1>

![CircleCI](https://img.shields.io/circleci/build/github/antgroup/vsag?logo=circleci&label=CircleCI)
[![codecov](https://codecov.io/gh/antgroup/vsag/graph/badge.svg?token=KDT3SpPMYS)](https://codecov.io/gh/antgroup/vsag)
![GitHub License](https://img.shields.io/github/license/antgroup/vsag)
![GitHub Release](https://img.shields.io/github/v/release/antgroup/vsag?label=last%20release)
![GitHub Contributors](https://img.shields.io/github/contributors/antgroup/vsag)
[![arXiv](https://badgen.net/static/arXiv/2404.16322/red)](http://arxiv.org/abs/2404.16322)
[![arXiv](https://badgen.net/static/arXiv/2503.17911/red)](http://arxiv.org/abs/2503.17911)
[![arXiv](https://badgen.net/static/arXiv/2509.08395/red)](https://arxiv.org/abs/2509.08395)

![PyPI - Version](https://img.shields.io/pypi/v/pyvsag)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyvsag)
[![PyPI Downloads](https://static.pepy.tech/badge/pyvsag)](https://pepy.tech/projects/pyvsag)
[![PyPI Downloads](https://static.pepy.tech/badge/pyvsag/month)](https://pepy.tech/projects/pyvsag)
[![PyPI Downloads](https://static.pepy.tech/badge/pyvsag/week)](https://pepy.tech/projects/pyvsag)
</div>


## What is VSAG

VSAG is a vector indexing library used for similarity search. The indexing algorithm allows users to search through various sizes of vector sets, especially those that cannot fit in memory. The library also provides methods for generating parameters based on vector dimensions and data scale, allowing developers to use it without understanding the algorithm’s principles. VSAG is written in C++ and provides a Python wrapper package called [pyvsag](https://pypi.org/project/pyvsag/).

## Performance
The VSAG algorithm SINDI for sparse vector search achieves a breakthrough in performance, substantially outperforming previous **state-of-the-art (SOTA)** solutions. In our internal tests on a 40-core Intel(R) Xeon(R) Silver 4210R CPU, VSAG's QPS exceeds that of the previous SOTA algorithm, Zilliz, by 166% on the sparse-full(8M) at 98% recall. While the official ann-benchmarks on sparse track runs on an Azure Standard D8lds v5 VM, we plan to submit our results under the official benchmark environment soon to formally validate this performance leap.

### sparse-full-inner-product
![](./docs/sparse-full_10_ip.png)

The VSAG algorithm achieves a significant boost of efficiency and outperforms the previous **state-of-the-art (SOTA)** by a clear margin. Specifically, VSAG's QPS exceeds that of the previous SOTA algorithm, Glass, by over 100%, and the baseline algorithm, HNSWLIB, by over 300% according to the ann-benchmark result on the GIST dataset at 90% recall.
The test in [ann-benchmarks](https://ann-benchmarks.com/) is running on an r6i.16xlarge machine on AWS with `--parallelism 31`, single-CPU, and hyperthreading disabled.
The result is as follows:

### gist-960-euclidean
![](./docs/gist-960-euclidean_10_euclidean.png)

## Getting Started
### Integrate with CMake
```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.11)

project (myproject)

set (CMAKE_CXX_STANDARD 11)

# download and compile vsag
include (FetchContent)
FetchContent_Declare (
  vsag
  GIT_REPOSITORY https://github.com/antgroup/vsag
  GIT_TAG main
)
FetchContent_MakeAvailable (vsag)
include_directories (vsag-cmake-example PRIVATE ${vsag_SOURCE_DIR}/include)

# compile executable and link to vsag
add_executable (vsag-cmake-example src/main.cpp)
target_link_libraries (vsag-cmake-example PRIVATE vsag)

# add dependency
add_dependencies (vsag-cmake-example vsag)
```
### Examples

Currently Python and C++ examples are provided, please explore [examples](./examples/) directory for details.

We suggest you start with [101_index_hnsw.cpp](./examples/cpp/101_index_hnsw.cpp) and [example_hnsw.py](./examples/python/example_hnsw.py).

## Building from Source
Please read the [DEVELOPMENT](./DEVELOPMENT.md) guide for instructions on how to build.

## Who's Using VSAG
- [OceanBase](https://github.com/oceanbase/oceanbase)
- [TuGraph](https://github.com/TuGraph-family/tugraph-db)
- [GreptimeDB](https://github.com/GreptimeTeam/greptimedb)
- [Hologres](https://www.aliyun.com/product/bigdata/hologram)
- [PolarDB](https://www.aliyun.com/product/polardb/mysql)

![vsag_users](./docs/vsag_users_20251106.svg)

If your system uses VSAG, then feel free to make a pull request to add it to the list.

## How to Contribute
Although VSAG is initially developed by the Vector Database Team at Ant Group, it's the work of
the [community](https://github.com/antgroup/vsag/graphs/contributors), and contributions are always welcome!
See [CONTRIBUTING](./CONTRIBUTING.md) for ways to get started.

## Community
![Discord](https://img.shields.io/discord/1298249687836393523?logo=discord&label=Discord)

Thrive together in VSAG community with users and developers from all around the world.

- Discuss at [discord](https://discord.com/invite/JyDmUzuhrp).
- Follow us on [Weixin Official Accounts](./docs/weixin-qr.jpg)（微信公众平台）to get the latest news.

## Roadmap v1.0 (ETA Mar. 2026)
- **Versatile Data Type Support**
  - **FP32 Vectors**: For standard, high-precision retrieval scenarios.
  - **INT8, BF16, FP16 Vectors**: Natively support quantized embedding models to reduce memory footprint.
  - **Sparse Vectors**: To enhance text retrieval capabilities.

- **Optimized Index Types**
  - **HGraph (Graph Index)**: For scenarios demanding high recall and low latency.
  - **IVF (Inverted File Index)**: Optimized for large-scale search (high `k`) and batch queries.
  - **SINDI (Sparse Inverted Non-redundant Distance Index)**: Optimized sparse vector index.

- **Advanced Quantization Methods**
  - **RaBitQ (BQ)**: Extreme compression for minimal memory usage.
  - **PQ (Product Quantization)**: Flexible compression for tuning the memory-recall trade-off.
  - **SQ4 & SQ8 (Scalar Quantization)**: Balanced performance with minimal recall loss for memory and speed gains.

- **Cross-Platform CPU Optimizations**
  - **x86_64**: `SSE`, `AVX`, `AVX2`, `AVX512`, `AMX`.
  - **ARM**: `Neon`, `SVE`.
  - **Optional Libraries**: Supports `Intel-MKL` and `OpenBLAS` for accelerated matrix multiplication.

- **Granular Resource Management**
  - **Memory Isolation**: Per-index memory allocators for tenant-level memory management.
  - **CPU Control**: Supports custom thread pools to scale ingestion and search throughput.

## Our Publications

1. VSAG: An Optimized Search Framework for Graph-based Approximate Nearest Neighbor Search [_VLDB (industry)_, 2025]  
   **Xiaoyao Zhong, Haotian Li, Jiabao Jin, Mingyu Yang, Deming Chu, Xiangyu Wang, Zhitao Shen, Wei Jia**, George Gu, Yi Xie, Xuemin Lin, Heng Tao Shen, Jingkuan Song, Peng Cheng  
   [PDF](https://www.vldb.org/pvldb/vol18/p5017-cheng.pdf) | [DOI](https://doi.org/10.14778/3750601.3750624)

2. Effective and General Distance Computation for Approximate Nearest Neighbor Search [_ICDE_, 2025]  
   **Mingyu Yang**, Wentao Li, **Jiabao Jin, Xiaoyao Zhong, Xiangyu Wang, Zhitao Shen**, **Wei Jia,** Wei Wang \
   [PDF](https://arxiv.org/pdf/2404.16322) | [DOI](https://doi.org/10.1109/ICDE65448.2025.00087)  

3. SINDI: an Efficient Index for Approximate Maximum Inner Product Search on Sparse Vectors [_arxiv_, 2025]  
   **Ruoxuan Li, Xiaoyao Zhong, Jiabao Jin**, Peng Cheng, Wangze Ni, Lei Chen, **Zhitao Shen, Wei Jia, Xiangyu Wang**, Xuemin Lin, Heng Tao Shen, Jingkuan Song  
   [PDF](https://arxiv.org/pdf/2509.08395)

4. EnhanceGraph: A Continuously Enhanced Graph-based Index for High-dimensional Approximate Nearest Neighbor Search [_arxiv_, 2025]  
   **Xiaoyao Zhong, Jiabao Jin**, Peng Cheng, **Mingyu Yang**, Lei Chen, Haoyang Li, **Zhitao Shen**, Xuemin Lin, Heng Tao Shen, Jingkuan Song  
   [PDF](https://arxiv.org/pdf/2506.13144)

5. Fast High-dimensional Approximate Nearest Neighbor Search with Efficient Index Time and Space [_arxiv_, 2025]  
   **Mingyu Yang**, Wentao Li, Wei Wang  
   [PDF](https://arxiv.org/pdf/2411.06158)

## Reference
VSAG referenced the following works during its implementation:
1. RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search [_SIGMOD_, 2024]  
  Jianyang Gao, Cheng Long  
   [PDF](https://dl.acm.org/doi/pdf/10.1145/3654970) | [DOI](https://doi.org/10.1145/3654970) | [CODE](https://github.com/VectorDB-NTU/RaBitQ-Library)


2. Quasi-succinct Indices [_WSDM_, 2013]  
  Sebastiano Vigna  
   [PDF](https://dl.acm.org/doi/pdf/10.1145/2433396.2433409) | [DOI](https://doi.org/10.1145/2433396.2433409)

## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=antgroup/vsag&type=Date)](https://star-history.com/#antgroup/vsag&Date)

## License
[Apache License 2.0](./LICENSE)

