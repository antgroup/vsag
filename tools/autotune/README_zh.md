# VSAG AutoTune 工具

`autotune` 的规范文档位于网站源码目录。
typed C++ API 是实验性接口，只在构建目录中提供，不会随 VSAG 安装。

- 本地英文文档：
  [docs/docs/en/src/resources/autotune.md](../../docs/docs/en/src/resources/autotune.md)
- 本地中文文档：
  [docs/docs/zh/src/resources/autotune.md](../../docs/docs/zh/src/resources/autotune.md)
- V1 CLI JSON 输入输出契约：
  [AutoTune V1 API](../../docs/docs/zh/src/resources/autotune_api_v1.md)
- 网站：<https://vsag.io/docs/resources/autotune.html>
- 中文网站：<https://vsag.io/docs/zh/resources/autotune.html>
- 示例请求：
  [examples/sift_hgraph_autotune_request.json](examples/sift_hgraph_autotune_request.json)
- 构建和查询联合调优示例：
  [examples/cpp/326_feature_create_index_with_constraints.cpp][factory-example]
- 已有索引 search 调优示例：
  [examples/cpp/327_feature_autotune_existing_index.cpp][existing-index-example]
- 已有 Pyramid path 调优示例：
  [examples/cpp/328_feature_autotune_existing_pyramid.cpp][pyramid-example]

[factory-example]: ../../examples/cpp/326_feature_create_index_with_constraints.cpp
[existing-index-example]: ../../examples/cpp/327_feature_autotune_existing_index.cpp
[pyramid-example]: ../../examples/cpp/328_feature_autotune_existing_pyramid.cpp
