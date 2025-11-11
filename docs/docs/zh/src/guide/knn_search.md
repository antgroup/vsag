# k-近邻搜索

以下内容，我们假设你已经安装 VSAG。我们提供了一些 C++ 和 Python 的代码示例，可以在 [`examples/`](https://github.com/antgroup/vsag/tree/main/examples) 目录找到。

## 获取一些向量

VSAG 主要用于处理固定维度 d 的向量集合，通常维度为几百到几千维。这些向量需要按照按行的方式组织才能写入 VSAG，类似于 `vector[num_vectors][dim]` 这样的 C++ 数组。从接口上说，VSAG 只依赖传入的向量集合首地址指针 （`const float*` 类型），所以应用可以自由选择使用 C++ 数组、`std::vector` 或者手动分配的内存来存储原始向量。

> 当前 VSAG 只支持 32-bit 的浮点数向量。

一次 k-近邻搜索需要两个向量集合。

- `base` 集合代表数据库中的所有向量，我们将会在其中进行搜索，它的大小是 **向量数** \* **向量维度**；
- `query` 集合代表查询向量，我们要为其查找最近的邻居，它的大小是 **向量数** \* **向量维度**。当前，VSAG 只支持 **向量数** = 1的查询，即不支持批量查询；

现在，我们生成一些 d=128 维的随机向量，以及它们对应的 ID（搜索结果会以 ID 形式返回）。

```cpp
    int64_t num_vectors = 10000;
    int64_t dim = 128
    int64_t *ids = new int64_t[num_vectors];
    float *datas = new float[num_vectors * dim];
    std::uniform_real_distribution<float> distrib_real;
    for (int64_t i = 0; i < num_vectors; ++i) {
        ids[i] = i;
    }
    for (int64_t i = 0; i < dim * num_vectors; ++i) {
        datas[i] = distrib_real(rng);
    }

    float* query_vector = new float[dim];
    for (int64_t i = 0; i < dim; ++i) {
        query_vector[i] = distrib_real(rng);
    }
```

这里使用的是 C++ 原生数组。当然，你也可以使用 `std::vector<float>` 来实现，并且通过 `data()` 方法得到数组首地址。

## 构建索引并写入向量

VSAG 库的使用主要围绕着 Index 接口，它封装了向量集合，并且提供了一系列能力。在 VSAG 中，Index 有多种实现，每种实现具备的能力和适用的场景不尽相同。在这个示例中，我们将使用最简单的版本，基于暴力搜索的索引 `BruteForce`。

所有索引都需要显式地创建，从而声明向量的维度和相似度计算方法。在这个示例中，向量的维度 dim=128，相似度使用欧几里得距离（L2）计算。

```cpp
    std::string brute_force_build_parameters = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 128
    }
    )";
    auto index = vsag::Factory::CreateIndex("brute_force", brute_force_build_parameters).value();
```

向量索引的数据写入涉及到两个方法：`Build` 和 `Add`。Build 方法带初始化性质，一些依赖训练过程来分析数据分布的索引，需要通过调用 Build 方法来启用。Add 是一般性的向量数据写入方法，大部分的索引都实现了这个方法，除了部分完全静态的索引类型。

BruteForce 索引支持用 Build 和 Add 方法写入数据，这里我们用 Add 方法来演示。

```cpp
    auto base = vsag::Dataset::Make();
    base->NumElements(num_vectors)
        ->Dim(dim)
        ->Ids(ids)
        ->Float32Vectors(datas)
        ->Owner(false);
    index->Add(base);
```

## 搜索

向量索引的一个核心作用是 **k-近邻** 搜索，即对于每个查询向量，在数据库中查找 `k` 个最相近的邻居。

搜索方法需要传入查询向量、k 值以及搜索参数。BruteForce 索引没有搜索参数，所以这里传入一个空的 Json 字符串。返回的结果中包含两个信息：最相近邻居的 ID ，最相近邻居与查询向量的距离。这两个信息可以分别通过结果集的 GetIds() 和 GetDistances() 方法获得。

```cpp
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(query_vector)->Owner(false);

    auto brute_force_search_parameters = R"({})";
    int64_t topk = 10;
    auto result = index->KnnSearch(query, topk, brute_force_search_parameters).value();

    std::cout << "results: " << std::endl;
    for (int64_t i = 0; i < result->GetDim(); ++i) {
        std::cout << result->GetIds()[i] << ": " << result->GetDistances()[i] << std::endl;
    }
```

搜索请求至多返回 k 个结果，这些结果按照最近邻和查询向量的距离升序排序。输出的结果类似于：

```bash
results:
6519: 13.855
2332: 15.2735
2126: 15.5844
7388: 15.6583
795: 15.5958
3979: 15.815
4756: 15.9983
510: 16.1128
8703: 16.1161
5583: 16.1256
```

