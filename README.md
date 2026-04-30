<div align="center">
  <h1><img alt="vsag-pages" src="docs/banner.svg" width=500/></h1>

![CircleCI](https://img.shields.io/circleci/build/github/antgroup/vsag?logo=circleci&label=CircleCI)
[![codecov](https://codecov.io/gh/antgroup/vsag/graph/badge.svg?token=KDT3SpPMYS)](https://codecov.io/gh/antgroup/vsag)
![GitHub License](https://img.shields.io/github/license/antgroup/vsag)
![GitHub Release](https://img.shields.io/github/v/release/antgroup/vsag?label=last%20release)
![GitHub Contributors](https://img.shields.io/github/contributors/antgroup/vsag)
[![arXiv](https://badgen.net/static/arXiv/2505.03212/red)](https://arxiv.org/abs/2505.03212)
[![arXiv](https://badgen.net/static/arXiv/2603.21710/red)](https://arxiv.org/abs/2603.21710)
[![arXiv](https://badgen.net/static/arXiv/2509.08395/red)](https://arxiv.org/abs/2509.08395)
[![arXiv](https://badgen.net/static/arXiv/2411.06158/red)](https://arxiv.org/abs/2411.06158)
[![arXiv](https://badgen.net/static/arXiv/2503.17911/red)](https://arxiv.org/abs/2503.17911)
[![arXiv](https://badgen.net/static/arXiv/2404.16322/red)](https://arxiv.org/abs/2404.16322)
[![arXiv](https://badgen.net/static/arXiv/2506.13144/red)](https://arxiv.org/abs/2506.13144)

![PyPI - Version](https://img.shields.io/pypi/v/pyvsag)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyvsag)
[![PyPI Downloads](https://static.pepy.tech/badge/pyvsag)](https://pepy.tech/projects/pyvsag)
[![PyPI Downloads](https://static.pepy.tech/badge/pyvsag/month)](https://pepy.tech/projects/pyvsag)
[![PyPI Downloads](https://static.pepy.tech/badge/pyvsag/week)](https://pepy.tech/projects/pyvsag)
</div>


## What is VSAG

VSAG is a vector indexing library used for similarity search. The indexing algorithm allows users to search through various sizes of vector sets, especially those that cannot fit in memory. The library also provides methods for generating parameters based on vector dimensions and data scale, allowing developers to use it without understanding the algorithm’s principles. VSAG is written in C++ and provides a Python wrapper package called [pyvsag](https://pypi.org/project/pyvsag/) and a Node.js/TypeScript binding package called `vsag`.

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

### Quickstart

Below is a minimal example of creating an HNSW index, building it with random vectors, and performing a k-NN search — shown in C++, Python, and TypeScript.

<details>
<summary><b>C++</b></summary>

```cpp
#include <vsag/vsag.h>
#include <iostream>
#include <random>

int main() {
    // Prepare data
    int64_t num_vectors = 1000, dim = 128;
    auto ids = new int64_t[num_vectors];
    auto vectors = new float[dim * num_vectors];
    std::mt19937 rng(47);
    std::uniform_real_distribution<float> dist;
    for (int64_t i = 0; i < num_vectors; ++i) ids[i] = i;
    for (int64_t i = 0; i < dim * num_vectors; ++i) vectors[i] = dist(rng);

    auto base = vsag::Dataset::Make();
    base->NumElements(num_vectors)->Dim(dim)->Ids(ids)->Float32Vectors(vectors);

    // Create and build index
    auto index = vsag::Factory::CreateIndex("hnsw", R"(
        {"dtype":"float32","metric_type":"l2","dim":128,
         "hnsw":{"max_degree":16,"ef_construction":100}})").value();
    index->Build(base);

    // Search
    auto query_vector = new float[dim];
    for (int64_t i = 0; i < dim; ++i) query_vector[i] = dist(rng);
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(query_vector)->Owner(true);

    auto result = index->KnnSearch(query, 10, R"({"hnsw":{"ef_search":100}})").value();
    for (int64_t i = 0; i < result->GetDim(); ++i)
        std::cout << result->GetIds()[i] << ": " << result->GetDistances()[i] << std::endl;
    return 0;
}
```

</details>

<details>
<summary><b>Python</b></summary>

```python
import pyvsag
import numpy as np
import json

dim = 128
num_elements = 1000

ids = list(range(num_elements))
data = np.float32(np.random.random((num_elements, dim)))

# Create and build index
index_params = json.dumps({
    "dtype": "float32", "metric_type": "l2", "dim": dim,
    "hnsw": {"max_degree": 16, "ef_construction": 100},
})
index = pyvsag.Index("hnsw", index_params)
index.build(vectors=data, ids=ids, num_elements=num_elements, dim=dim)

# Search
query = np.float32(np.random.random(dim))
search_params = json.dumps({"hnsw": {"ef_search": 100}})
result_ids, result_dists = index.knn_search(vector=query, k=10, parameters=search_params)
for rid, rdist in zip(result_ids, result_dists):
    print(f"{rid}: {rdist}")
```

</details>

<details>
<summary><b>TypeScript (Node.js)</b></summary>

```typescript
import { Index } from "vsag";

const dim = 128;
const numVectors = 1000;

const ids = new BigInt64Array(numVectors);
const vectors = new Float32Array(dim * numVectors);
for (let i = 0; i < numVectors; i++) ids[i] = BigInt(i);
for (let i = 0; i < dim * numVectors; i++) vectors[i] = Math.random();

// Create and build index
const indexParams = JSON.stringify({
    dtype: "float32", metric_type: "l2", dim,
    hnsw: { max_degree: 16, ef_construction: 100 },
});
const index = new Index("hnsw", indexParams);
index.build(vectors, ids, numVectors, dim);

// Search
const query = new Float32Array(dim);
for (let i = 0; i < dim; i++) query[i] = Math.random();

const searchParams = JSON.stringify({ hnsw: { ef_search: 100 } });
const { ids: resultIds, distances } = index.knnSearch(query, 10, searchParams);
for (let i = 0; i < 10; i++) {
    console.log(`${resultIds[i]}: ${distances[i]}`);
}
```

</details>

### Integrate with CMake
```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.18)

project (myproject)

set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

# download and compile vsag
include (FetchContent)
FetchContent_Declare (
  vsag
  GIT_REPOSITORY https://github.com/antgroup/vsag
  GIT_TAG main
)
FetchContent_MakeAvailable (vsag)

# compile executable and link to vsag
add_executable (vsag-cmake-example src/main.cpp)
target_include_directories (vsag-cmake-example SYSTEM PRIVATE ${vsag_SOURCE_DIR}/include)
target_link_libraries (vsag-cmake-example PRIVATE vsag)

# add dependency
add_dependencies (vsag-cmake-example vsag)
```

### Examples

C++, Python, and TypeScript examples are provided. Please explore the [examples](./examples/) directory for details.

We suggest you start with:
- **C++**: [101_index_hnsw.cpp](./examples/cpp/101_index_hnsw.cpp)
- **Python**: [example_hnsw.py](./examples/python/example_hnsw.py)
- **TypeScript**: [101_index_hnsw.ts](./examples/typescript/101_index_hnsw.ts)

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

1. Elastic Index Selection for Label-Hybrid AKNN Search [_VLDB_, 2026]  
   **Mingyu Yang**, Wenxuan Xia, Wentao Li, Raymond Chi-Wing Wong, Wei Wang  
   [PDF](https://arxiv.org/pdf/2505.03212) | [DOI](https://doi.org/10.14778/3785297.3785304)

2. FGIM: a Fast Graph-based Indexes Merging Framework for Approximate Nearest Neighbor Search [_SIGMOD_, 2026]  
   **Zekai Wu, Jiabao Jin**, Peng Cheng, **Xiaoyao Zhong**, Lei Chen, Yongxin Tong, **Zhitao Shen**, Jingkuan Song, Heng Tao Shen, Xuemin Lin  
   [PDF](https://arxiv.org/pdf/2603.21710) | [DOI](https://doi.org/10.1145/3786651)

3. SINDI: an Efficient Index for Approximate Maximum Inner Product Search on Sparse Vectors [_ICDE_, 2026]  
   **Ruoxuan Li, Xiaoyao Zhong, Jiabao Jin**, Peng Cheng, Wangze Ni, Lei Chen, **Zhitao Shen, Wei Jia, Xiangyu Wang**, Xuemin Lin, Heng Tao Shen, Jingkuan Song  
   [PDF](https://arxiv.org/pdf/2509.08395)

4. Quantization Meets Projection: A Happy Marriage for Approximate k-Nearest Neighbor Search [_VLDB_, 2026]  
   **Mingyu Yang**, Liuchang Jing, Wentao Li, Wei Wang  
   [PDF](https://arxiv.org/pdf/2411.06158)

5. VSAG: An Optimized Search Framework for Graph-based Approximate Nearest Neighbor Search [_VLDB (industry)_, 2025]  
   **Xiaoyao Zhong, Haotian Li, Jiabao Jin, Mingyu Yang, Deming Chu, Xiangyu Wang, Zhitao Shen, Wei Jia**, George Gu, Yi Xie, Xuemin Lin, Heng Tao Shen, Jingkuan Song, Peng Cheng  
   [PDF](https://www.vldb.org/pvldb/vol18/p5017-cheng.pdf) | [DOI](https://doi.org/10.14778/3750601.3750624)

6. Effective and General Distance Computation for Approximate Nearest Neighbor Search [_ICDE_, 2025]  
   **Mingyu Yang**, Wentao Li, **Jiabao Jin, Xiaoyao Zhong, Xiangyu Wang, Zhitao Shen**, **Wei Jia,** Wei Wang \
   [PDF](https://arxiv.org/pdf/2404.16322) | [DOI](https://doi.org/10.1109/ICDE65448.2025.00087)  

7. EnhanceGraph: A Continuously Enhanced Graph-based Index for High-dimensional Approximate Nearest Neighbor Search [_arxiv_, 2025]  
   **Xiaoyao Zhong, Jiabao Jin**, Peng Cheng, **Mingyu Yang**, Lei Chen, Haoyang Li, **Zhitao Shen**, Xuemin Lin, Heng Tao Shen, Jingkuan Song  
   [PDF](https://arxiv.org/pdf/2506.13144)

## Reference
VSAG referenced the following works during its implementation:
1. RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search [_SIGMOD_, 2024]  
  Jianyang Gao, Cheng Long  
   [PDF](https://dl.acm.org/doi/pdf/10.1145/3654970) | [DOI](https://doi.org/10.1145/3654970) | [CODE](https://github.com/VectorDB-NTU/RaBitQ-Library)


2. Quasi-succinct Indices [_WSDM_, 2013]  
  Sebastiano Vigna  
   [PDF](https://dl.acm.org/doi/pdf/10.1145/2433396.2433409) | [DOI](https://doi.org/10.1145/2433396.2433409)

## Contributors

<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<!-- readme: contributors -start -->
<table>
    <tbody>
        <tr>
            <td align="center">
                <a href="https://github.com/LHT129">
                    <img src="https://avatars.githubusercontent.com/u/176897537?v=4" width="100" alt="LHT129"/>
                    <br />
                    <sub><b>LHT129</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/inabao">
                    <img src="https://avatars.githubusercontent.com/u/37021995?v=4" width="100" alt="inabao"/>
                    <br />
                    <sub><b>inabao</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/wxyucs">
                    <img src="https://avatars.githubusercontent.com/u/12595343?v=4" width="100" alt="wxyucs"/>
                    <br />
                    <sub><b>Xiangyu Wang</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/ShawnShawnYou">
                    <img src="https://avatars.githubusercontent.com/u/58975154?v=4" width="100" alt="ShawnShawnYou"/>
                    <br />
                    <sub><b>ShawnShawnYou</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/jac0626">
                    <img src="https://avatars.githubusercontent.com/u/118544282?v=4" width="100" alt="jac0626"/>
                    <br />
                    <sub><b>jac</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/Coien-rr">
                    <img src="https://avatars.githubusercontent.com/u/83146518?v=4" width="100" alt="Coien-rr"/>
                    <br />
                    <sub><b>Cooper</b></sub>
                </a>
            </td>
        </tr>
        <tr>
            <td align="center">
                <a href="https://github.com/Carrot-77">
                    <img src="https://avatars.githubusercontent.com/u/61344086?v=4" width="100" alt="Carrot-77"/>
                    <br />
                    <sub><b>Carrot-77</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/nedchu">
                    <img src="https://avatars.githubusercontent.com/u/11944144?v=4" width="100" alt="nedchu"/>
                    <br />
                    <sub><b>Deming Chu</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/shadowao">
                    <img src="https://avatars.githubusercontent.com/u/13804928?v=4" width="100" alt="shadowao"/>
                    <br />
                    <sub><b>azl</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/yulijunzj">
                    <img src="https://avatars.githubusercontent.com/u/22726506?v=4" width="100" alt="yulijunzj"/>
                    <br />
                    <sub><b>L J. Yun</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/Roxanne0321">
                    <img src="https://avatars.githubusercontent.com/u/188438858?v=4" width="100" alt="Roxanne0321"/>
                    <br />
                    <sub><b>Roxanne</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/jingyueob">
                    <img src="https://avatars.githubusercontent.com/u/212298588?v=4" width="100" alt="jingyueob"/>
                    <br />
                    <sub><b>jingyueob</b></sub>
                </a>
            </td>
        </tr>
        <tr>
            <td align="center">
                <a href="https://github.com/antfin-oss">
                    <img src="https://avatars.githubusercontent.com/u/48939886?v=4" width="100" alt="antfin-oss"/>
                    <br />
                    <sub><b>Ant OSS</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/misaka0714">
                    <img src="https://avatars.githubusercontent.com/u/129934985?v=4" width="100" alt="misaka0714"/>
                    <br />
                    <sub><b>baoyuan</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/Danbaiwq">
                    <img src="https://avatars.githubusercontent.com/u/212493818?v=4" width="100" alt="Danbaiwq"/>
                    <br />
                    <sub><b>Danbaiwq</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/HeHuMing">
                    <img src="https://avatars.githubusercontent.com/u/223064905?v=4" width="100" alt="HeHuMing"/>
                    <br />
                    <sub><b>HuMing He</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/jiacai2050">
                    <img src="https://avatars.githubusercontent.com/u/3848910?v=4" width="100" alt="jiacai2050"/>
                    <br />
                    <sub><b>Jiacai Liu</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/lyxiong0">
                    <img src="https://avatars.githubusercontent.com/u/29161506?v=4" width="100" alt="lyxiong0"/>
                    <br />
                    <sub><b>Liyao Xiong</b></sub>
                </a>
            </td>
        </tr>
        <tr>
            <td align="center">
                <a href="https://github.com/mly5269">
                    <img src="https://avatars.githubusercontent.com/u/130448862?v=4" width="100" alt="mly5269"/>
                    <br />
                    <sub><b>mly</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/Ningsir">
                    <img src="https://avatars.githubusercontent.com/u/34963409?v=4" width="100" alt="Ningsir"/>
                    <br />
                    <sub><b>Xinger</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/pkusunjy">
                    <img src="https://avatars.githubusercontent.com/u/11880269?v=4" width="100" alt="pkusunjy"/>
                    <br />
                    <sub><b>Sun Jiayu</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/cubicc">
                    <img src="https://avatars.githubusercontent.com/u/51120671?v=4" width="100" alt="cubicc"/>
                    <br />
                    <sub><b>cubicc</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/dasurax">
                    <img src="https://avatars.githubusercontent.com/u/9841872?v=4" width="100" alt="dasurax"/>
                    <br />
                    <sub><b>dasurax</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/hhy3">
                    <img src="https://avatars.githubusercontent.com/u/44047980?v=4" width="100" alt="hhy3"/>
                    <br />
                    <sub><b>Zihao Wang</b></sub>
                </a>
            </td>
        </tr>
        <tr>
            <td align="center">
                <a href="https://github.com/jiaweizone">
                    <img src="https://avatars.githubusercontent.com/u/251354?v=4" width="100" alt="jiaweizone"/>
                    <br />
                    <sub><b>wei</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/mingyu-hkustgz">
                    <img src="https://avatars.githubusercontent.com/u/151442761?v=4" width="100" alt="mingyu-hkustgz"/>
                    <br />
                    <sub><b>Mingyu Yang</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/skylhd">
                    <img src="https://avatars.githubusercontent.com/u/13144296?v=4" width="100" alt="skylhd"/>
                    <br />
                    <sub><b>lhd</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/stuBirdFly">
                    <img src="https://avatars.githubusercontent.com/u/84010733?v=4" width="100" alt="stuBirdFly"/>
                    <br />
                    <sub><b>stuBirdFly</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/xfmeng17">
                    <img src="https://avatars.githubusercontent.com/u/32661584?v=4" width="100" alt="xfmeng17"/>
                    <br />
                    <sub><b>XFMENG17</b></sub>
                </a>
            </td>
        </tr>
    </tbody>
</table>
<!-- readme: contributors -end -->
<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=antgroup/vsag&type=Date)](https://star-history.com/#antgroup/vsag&Date)

## License
[Apache License 2.0](./LICENSE)
