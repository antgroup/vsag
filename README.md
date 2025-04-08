<div align="center">
  <h1><img alt="vsag-pages" src="docs/banner.svg" width=500/></h1>

![CircleCI](https://img.shields.io/circleci/build/github/antgroup/vsag?logo=circleci&label=CircleCI)
[![codecov](https://codecov.io/gh/antgroup/vsag/graph/badge.svg?token=KDT3SpPMYS)](https://codecov.io/gh/antgroup/vsag)
![GitHub License](https://img.shields.io/github/license/antgroup/vsag)
![GitHub Release](https://img.shields.io/github/v/release/antgroup/vsag?label=last%20release)
![GitHub Contributors](https://img.shields.io/github/contributors/antgroup/vsag)
[![arXiv](https://badgen.net/static/arXiv/2404.16322/red)](http://arxiv.org/abs/2404.16322)
[![arXiv](https://badgen.net/static/arXiv/2503.17911/red)](http://arxiv.org/abs/2503.17911)

![PyPI - Version](https://img.shields.io/pypi/v/pyvsag)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyvsag)
[![PyPI Downloads](https://static.pepy.tech/badge/pyvsag)](https://pepy.tech/projects/pyvsag)
[![PyPI Downloads](https://static.pepy.tech/badge/pyvsag/month)](https://pepy.tech/projects/pyvsag)
[![PyPI Downloads](https://static.pepy.tech/badge/pyvsag/week)](https://pepy.tech/projects/pyvsag)
</div>


## What is VSAG

VSAG is a vector indexing library used for similarity search. The indexing algorithm allows users to search through various sizes of vector sets, especially those that cannot fit in memory. The library also provides methods for generating parameters based on vector dimensions and data scale, allowing developers to use it without understanding the algorithm’s principles. VSAG is written in C++ and provides a Python wrapper package called [pyvsag](https://pypi.org/project/pyvsag/).

## Performance
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

![vsag_users](./docs/vsag_users.svg)

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

## Roadmap
- v0.14 (ETA: Mar. 2025)
  - support inverted index(be like IVFFlat) based on datacell
  - support extrainfo storage within vector
  - implement a new MultiIndex that supports efficient pre-filtering on enumerable tags
- v0.15 (ETA: Apr. 2025)
  - support sparse vector searching
  - introduce pluggable product quantization(known as PQ) in datacell
- v0.16 (ETA: May 2025)
  - support neon instruction acceleration on ARM platform
  - support using GPU to accelerate index building
  - provide an optimizer that supports optimizing search parameters by recall or latency

## Reference
Reference to cite when you use VSAG in a research paper:
```
@article{Yang2024EffectiveAG,
  title={Effective and General Distance Computation for Approximate Nearest Neighbor Search},
  author={Mingyu Yang and Wentao Li and Jiabao Jin and Xiaoyao Zhong and Xiangyu Wang and Zhitao Shen and Wei Jia and Wei Wang},
  year={2024},
  url={https://arxiv.org/abs/2404.16322}
}

@article{zhong2025vsag,
  title={VSAG: An Optimized Search Framework for Graph-based Approximate Nearest Neighbor Search},
  author={Zhong, Xiaoyao and Li, Haotian and Jin, Jiabao and Yang, Mingyu and Chu, Deming and Wang, Xiangyu and Shen, Zhitao and Jia, Wei and Gu, George and Xie, Yi and others},
  journal={arXiv preprint arXiv:2503.17911},
  year={2025}
}
```

For the implementation of RaBitQ quantization in VSAG, please cite:
```
@article{10.1145/3654970,
author = {Gao, Jianyang and Long, Cheng},
title = {RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search},
year = {2024},
issue_date = {June 2024},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {2},
number = {3},
url = {https://doi.org/10.1145/3654970},
doi = {10.1145/3654970},
abstract = {Searching for approximate nearest neighbors (ANN) in the high-dimensional Euclidean space is a pivotal problem. Recently, with the help of fast SIMD-based implementations, Product Quantization (PQ) and its variants can often efficiently and accurately estimate the distances between the vectors and have achieved great success in the in-memory ANN search. Despite their empirical success, we note that these methods do not have a theoretical error bound and are observed to fail disastrously on some real-world datasets. Motivated by this, we propose a new randomized quantization method named RaBitQ, which quantizes D-dimensional vectors into D-bit strings. RaBitQ guarantees a sharp theoretical error bound and provides good empirical accuracy at the same time. In addition, we introduce efficient implementations of RaBitQ, supporting to estimate the distances with bitwise operations or SIMD-based operations. Extensive experiments on real-world datasets confirm that (1) our method outperforms PQ and its variants in terms of accuracy-efficiency trade-off by a clear margin and (2) its empirical performance is well-aligned with our theoretical analysis.},
journal = {Proc. ACM Manag. Data},
month = may,
articleno = {167},
numpages = {27},
keywords = {Johnson-Lindenstrauss transformation, approximate nearest neighbor search, quantization}
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=antgroup/vsag&type=Date)](https://star-history.com/#antgroup/vsag&Date)

## License
[Apache License 2.0](./LICENSE)

