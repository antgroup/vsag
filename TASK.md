---
title: "Work-Stealing ThreadPool 替代 progschj::ThreadPool"
status: in_progress
priority: high
created_by: "deepseek-harness"
assigned_to: "deepseek-harness"
created_at: "2026-11-05"
updated_at: "2026-11-05"
target_repo: "antgroup/vsag"
related_files:
  - include/vsag/thread_pool.h
  - src/impl/thread_pool/default_thread_pool.h
  - src/impl/thread_pool/default_thread_pool.cpp
  - src/impl/thread_pool/safe_thread_pool.h
  - src/impl/thread_pool/safe_thread_pool_test.cpp
  - src/impl/thread_pool/default_thread_pool_test.cpp
  - extern/thread_pool/thread_pool.cmake
  - src/algorithm/hgraph/hgraph.h
  - src/index_common_param.h
  - src/algorithm/inner_index_interface.h
pr_url: ""
---

# Work-Stealing ThreadPool 替代 progschj::ThreadPool

## 1. 背景

VSAG 当前的线程池栈为三层结构：

```
vsag::ThreadPool (include/vsag/thread_pool.h)      ← 公共抽象接口
    ├── DefaultThreadPool (src/impl/thread_pool/)    ← 包装 progschj::ThreadPool
    └── SafeThreadPool   (src/impl/thread_pool/)    ← 异常安全 + GeneralEnqueue + 工厂
```

底层 `progschj::ThreadPool` 来自 [log4cplus/ThreadPool](https://github.com/log4cplus/ThreadPool)，commit `3507796e`，是一个约 300 行的 header-only 线程池。其核心问题：

- **单一全局队列 + 单一 `std::mutex`**：所有 worker 线程和生产者共享一把锁，高并发下 contention 明显。
- **无 work-stealing**：worker 空转时只能空等，无法协助其他忙碌 worker。
- **线程扩缩用 `detach()`**：有资源泄漏和悬挂风险。
- **`wait_until_nothing_in_flight` 额外一把 mutex + cv**：唤醒路径与任务队列完全独立，增加开销。

对 VSAG 的实际场景（通常 4-16 线程、任务粒度为毫秒到秒级），单锁争用不是最严重的瓶颈，但仍有可观的优化空间——尤其在大量细粒度并行操作（如 KMeans 多 query 分批、Pyramid 层级构建）中，锁争用会拖累整体吞吐。

## 2. VSAG 线程池使用模式分析

2026-11-05 对全部 45+ 处调用点的代码审阅结论：

### 2.1 典型模式

**模式 1：分块并行（最常见）**

```cpp
// 将 N 个元素分成 num_threads 块，每块一个任务
size_t block_size = (total_jobs + num_threads - 1) / num_threads;
for (size_t i = 0; i < total_jobs; i += block_size) {
    futures.emplace_back(thread_pool_->GeneralEnqueue(
        [&, lo = i, hi = min(i + block_size, total_jobs)]() {
            for (size_t cur = lo; cur < hi; ++cur) { /* heavy work */ }
        }));
}
wait_all_futures(futures);
```

出现于：HGraph 构建/refine/warm_start (`hgraph_build.cpp:566,1558,1702,1881`)、`hgraph_mci.cpp:1239`、Pyramid (`pyramid.cpp:598`)、IVF build (`ivf_build.cpp:240,385`)、KMeans/KMedoids、BruteForce、inner_index_interface 等。

**模式 2：每线程固定角色**

```cpp
for (uint64_t i = 0; i < num_threads; i++) {
    futures.emplace_back(pool->GeneralEnqueue(task, i));
}
```

出现于：parallel_searcher (`parallel_searcher.cpp:250`)、IVF search (`ivf_search.cpp:275`)、nearest_centroid_assign 等。

**模式 3：按 bucket/query 细粒度分区**

```cpp
for (each_bucket_or_query) {
    futures.emplace_back(thread_pool_->GeneralEnqueue(process_one, b));
}
```

出现于：IVF nearest partition (`ivf_nearest_partition.cpp:213,372`)、simq (`simq.cpp:370,575,789,1031`)、rabitq split datacell 等。

### 2.2 关键特征

| 特征 | 值 |
|---|---|
| 单次批量提交任务数 | 通常 = 线程数（4-16），最多几十 |
| 单个任务粒度 | 重（加边/搜索/聚类/编码，毫秒到秒级） |
| 任务间依赖 | 无依赖 —— BSP 模式：全提交 → 全等待 |
| 嵌套提交 | 无 |
| 线程数范围 | 默认 4（`Options::num_threads_building_`），实际 1-16 |
| 异常处理 | SafeThreadPool 吞异常记日志；调用侧也做 try-catch + exception_ptr 收集 |

### 2.3 设计启示

- 不是 folly 式「海量微任务」，work-stealing 的收益主要来自**负载均衡**而非减少锁争用。
- 任务粒度足够大，steal 开销可以摊还。
- BSP 模式意味着 `WaitUntilEmpty()` 是关键路径，需要高效实现。
- 需要保留 `GeneralEnqueue`（模板版本）以避免 `std::function` 包装开销。

## 3. 设计方案

### 3.0 C++ 版本约束

VSAG 的 C++ 标准由 `cmake/VSAGCompilerConfig.cmake:86-91` 控制：

- **默认 C++17**（`CMAKE_CXX_STANDARD 17`）
- 仅在 **非 macOS + Clang ≥ 17** 时升到 **C++20**

这意味着 macOS 开发环境和部分 Linux CI 上都只有 C++17。本方案不使用任何 C++20 专有特性（`std::jthread`、`std::atomic::wait` 等），全部用 C++17 可用的原语实现：
- `std::thread` + `std::atomic<bool>` stop 标志 + 手动 `join()`
- `std::condition_variable` 做等待/唤醒
- `std::invoke_result`（C++17）用于 `GeneralEnqueue` 返回类型推导

### 3.1 架构总览

```
                       ┌──────────────────────────┐
                       │   Global MPMC Lock-Free   │
                       │        Task Queue          │
                       │   (bounded, moodycamel-    │
                       │    style design)           │
                       └──────┬───────┬─────────────┘
                              │       │
              ┌───────────────┘       └───────────────┐
              ▼                                       ▼
    ┌───────────────────┐                   ┌───────────────────┐
    │    Worker 0        │                   │    Worker N-1      │
    │  ┌───────────────┐ │                   │  ┌───────────────┐ │
    │  │ Local SPSC    │ │                   │  │ Local SPSC    │ │
    │  │ Task Queue    │ │                   │  │ Task Queue    │ │
    │  └───────────────┘ │                   │  └───────────────┘ │
    │  ┌───────────────┐ │                   │  ┌───────────────┐ │
    │  │ std::thread   │ │                   │  │ std::thread   │ │
    │  │ + atomic stop │ │                   │  │ + atomic stop │ │
    │  └───────────────┘ │                   │  └───────────────┘ │
    └───────────────────┘                   └───────────────────┘
              │                                       │
              └─────────────── Steal ─────────────────┘
              (random victim, take from tail of victim's local queue)
```

### 3.2 文件布局

```
src/impl/thread_pool/
├── work_stealing_thread_pool.h     # 新线程池声明
├── work_stealing_thread_pool.cpp   # 实现
├── work_stealing_thread_pool_test.cpp  # 单元测试
├── safe_thread_pool.h              # 不变（或微调引用新类）
├── default_thread_pool.h           # 保留，标记 deprecated
└── default_thread_pool.cpp         # 保留

include/vsag/
└── thread_pool.h                   # 完全不动
```

### 3.3 公共接口

`include/vsag/thread_pool.h` **完全不动**。新类实现同一套虚函数：

```cpp
class WorkStealingThreadPool : public ThreadPool {
public:
    // --- 构造 / 析构 ---
    explicit WorkStealingThreadPool(uint64_t num_threads);
    ~WorkStealingThreadPool() override;

    // 禁止拷贝/移动
    WorkStealingThreadPool(const WorkStealingThreadPool&) = delete;
    WorkStealingThreadPool& operator=(const WorkStealingThreadPool&) = delete;

    // --- ThreadPool 接口（保持不变） ---
    std::future<void> Enqueue(std::function<void(void)> task) override;
    void WaitUntilEmpty() override;
    void SetPoolSize(uint64_t limit) override;
    void SetQueueSizeLimit(uint64_t limit) override;

    // --- 新增：模板 enqueue（避免 std::function 开销） ---
    template <class F, class... Args>
    auto GeneralEnqueue(F&& f, Args&&... args)
        -> std::future<typename std::invoke_result<F, Args...>::type>;

    // --- 新增：线程亲和性（可选，第一阶段） ---
    void SetAffinity(const std::vector<int>& cpu_ids);
};
```

### 3.4 核心组件设计

#### 3.4.1 全局任务队列（MPMC Lock-Free）

- **设计参考**：`moodycamel::ConcurrentQueue` 的 token-based MPMC 设计，但自实现为项目内代码（零外部依赖）。
- **实现方案**：固定大小 ring buffer + 原子 head/tail 索引 + CAS 操作。
  - 使用 `std::atomic<uint64_t>` 做 enqueue/dequeue 竞争。
  - Bounded 设计：`SetQueueSizeLimit()` 即设置 ring buffer 容量。
  - 满时 enqueue 阻塞（短期用 spin + yield，长期 fallback 到 condition_variable）。
- **复杂度**：预计约 200-300 行。核心是 `enqueue(Task*)` / `dequeue(Task*)` 两个方法。

#### 3.4.2 本地任务队列（SPSC）

- 每个 worker 持有一个 SPSC 无锁队列（单生产者多消费者，但 steal 只从尾部取）。
- **实现**：参考 Chase-Lev work-stealing deque 的简化版。
  - `push()`（owner 线程）和 `pop()`（owner 线程）在头部操作。
  - `steal()`（其他线程）从尾部取。
  - 使用 `std::atomic` + memory order 保证正确性。
- **大小**：固定容量（如 256），满了就 fallback 到全局队列。

#### 3.4.3 Worker 线程

```cpp
struct Worker {
    std::thread thread;               // std::thread + 手动 join（C++17 兼容）
    SPSCQueue local_queue;            // 本地任务队列
    std::atomic<bool> idle{true};     // steal 目标选择用
    uint64_t tasks_completed{0};      // 统计
};
```

- **主循环**：
  1. `local_queue.pop()` → 有任务则执行
  2. `global_queue.dequeue()` → 有任务则执行
  3. 随机选一个 victim worker，`victim.local_queue.steal()` → 有则执行
  4. 全部失败 → 短暂 spin（自适应等待），然后阻塞在 `condition_variable` 上等新任务或 stop 信号

- **stop 机制**：使用 `std::atomic<bool> stop_` 标志。析构时设 `stop_ = true` + `notify_all()`，所有 worker 在下次循环检测到 stop 后退出，然后主线程逐个 `thread.join()`。与 progschj 的 `detach()` 不同，这里保证 join，无资源泄漏风险。

#### 3.4.4 线程扩缩（`SetPoolSize`）

- **扩容**：创建新 `Worker`，启动 `std::thread`。
- **缩容**：设 `pool_size_` 为目标值，`notify_all()` 消费者。多余的 worker 检测到自己的 `worker_index >= pool_size_` 后退出主循环。主线程在 `join()` 时回收。
- 与 progschj 的 `detach()` 不同，这里保证 join，避免资源泄漏。

#### 3.4.5 `WaitUntilEmpty`

- 不再需要 progschj 的额外 mutex + cv。
- 使用 **eventcount** 模式（futex-like）：
  - 一个 `std::atomic<uint64_t> in_flight` 计数器。
  - 每个 worker 取到任务时 `in_flight++`，完成时 `in_flight--`。
  - `WaitUntilEmpty` 自旋 + yield 等待 `in_flight == 0`，最后 fallback 用 condition_variable 阻塞等待。
- 或者更简单的方案：当全局和所有本地队列为空且 `in_flight == 0` 时，唤醒等待者。

#### 3.4.6 异常处理

- `Enqueue` 中不吞异常（由调用方处理）。
- 替代 `SafeThreadPool` 的 try-catch 包装放在 `SafeThreadPool::Enqueue` 中保持不变（因为 SafeThreadPool 不关心底层是 DefaultThreadPool 还是 WorkStealingThreadPool——它只依赖 `ThreadPool*` 指针）。

#### 3.4.7 线程亲和性（可选）

```cpp
void SetAffinity(const std::vector<int>& cpu_ids) {
    // 在 Linux 上用 pthread_setaffinity_np
    // 在 macOS 上用 thread_policy_set (mach)
    // 在其他平台上为 no-op
}
```

第一阶段实现 Linux 和 macOS。

### 3.5 与 SafeThreadPool 的关系

`SafeThreadPool` 持有的是 `ThreadPool* pool_`（基类指针），因此任何实现了 `ThreadPool` 接口的类都可以直接挂进去。唯一改动：

```cpp
// safe_thread_pool.h 的 FactoryDefaultThreadPool
static std::shared_ptr<SafeThreadPool>
FactoryDefaultThreadPool() {
    return std::make_shared<SafeThreadPool>(
        new WorkStealingThreadPool(Options::Instance().num_threads_building()), true);
    //                                 ↑ 原来：new DefaultThreadPool(...)
}
```

### 3.6 与现有测试的关系

- 新增 `work_stealing_thread_pool_test.cpp` 覆盖：基本 enqueue/execute、WaitUntilEmpty 正确性、SetPoolSize 扩缩、SetQueueSizeLimit 背压、多线程并发提交、steal 正确性（用不均匀任务分布验证负载均衡）。
- 现有 `safe_thread_pool_test.cpp` 和 `default_thread_pool_test.cpp` 保持不变。
- `SafeThreadPool` 的测试间接覆盖新线程池（因为 FactoryDefaultThreadPool 会切换到新实现）。

## 4. 实现计划

### 阶段 1：核心实现 + 单测

| 步骤 | 内容 | 预计文件 |
|---|---|---|
| 1.1 | MPMC lock-free 全局队列 | `src/impl/thread_pool/mpmc_queue.h` |
| 1.2 | SPSC work-stealing 本地队列 | `src/impl/thread_pool/spsc_queue.h` |
| 1.3 | WorkStealingThreadPool 主类 | `src/impl/thread_pool/work_stealing_thread_pool.h/.cpp` |
| 1.4 | 单元测试 | `src/impl/thread_pool/work_stealing_thread_pool_test.cpp` |
| 1.5 | SafeThreadPool 工厂切换到新实现 | `src/impl/thread_pool/safe_thread_pool.h` (1行改动) |

### 阶段 2：集成 + Benchmark

| 步骤 | 内容 |
|---|---|
| 2.1 | 全量 C++ 单测通过（确保 SafeThreadPool 行为不变） |
| 2.2 | 编写 micro-benchmark：对比新旧线程池在不同线程数/任务数下的吞吐和延迟 |
| 2.3 | VSAG 构建/检索 end-to-end benchmark（HGraph 1M 构建耗时、检索 QPS） |

### 阶段 3：全面顶替（取决于阶段 2 结果）

| 步骤 | 内容 |
|---|---|
| 3.1 | 删除 `default_thread_pool.h/.cpp` |
| 3.2 | 移除 `extern/thread_pool/thread_pool.cmake` 中的 progschj 依赖 |
| 3.3 | 重命名：`WorkStealingThreadPool` → `DefaultThreadPool`（可选，降低认知负担） |

### 阶段 4（可选/未来）：NUMA 感知

- 为每个 NUMA node 创建独立的全局队列组
- Worker 优先从本 node 队列取任务，跨 node steal 作为最后手段
- 这取决于 VSAG 部署环境的 NUMA 特征，当前不做。

## 5. 性能预期

| 场景 | 预期改进 | 原因 |
|---|---|---|
| 单线程提交少量重任务（当前主路径） | 持平或略微改善 | 单锁争用本来就不严重 |
| 多线程提交大量中粒度任务（KMeans multi-query） | 5-15% 吞吐提升 | 减少 `queue_mutex` 争用 |
| 任务负载不均（部分 worker 先完成） | 明显改善尾部延迟 | work-stealing 让空闲 worker 协助 |
| `WaitUntilEmpty` | 更快 | 无独立 mutex，用 in_flight 原子计数 |

**验收标准**：
- 所有现有测试通过，行为不变。
- HGraph 1M 构建耗时不退化。
- HGraph 1M 检索 QPS 不退化。
- micro-benchmark 在高并发场景下有可测量的吞吐改善。

## 6. 风险与缓解

| 风险 | 缓解 |
|---|---|
| Lock-free 队列实现有 bug（ABA、memory order 错误） | 用 TSan + 大量并发测试覆盖；参考成熟实现（moodycamel）的 memory order |
| 性能不如预期 | 保留 progschj 旧实现，通过编译选项或运行时开关回退 |
| 线程生命周期管理 | 使用 `std::thread` + `std::atomic<bool>` stop 标志 + 手动 `join()`，C++17 兼容，macOS/Linux 均可编译 |
| macOS 与 Linux 行为差异 | 两平台都做单测和 benchmark |

## 7. 参考资料

- [progschj::ThreadPool 源码](https://github.com/log4cplus/ThreadPool/blob/3507796e/ThreadPool.h)
- [moodycamel::ConcurrentQueue](https://github.com/cameron314/concurrentqueue)
- [Chase-Lev work-stealing deque (原论文)](https://www.di.ens.fr/~zappa/readings/ppopp13.pdf)
- VSAG 现有线程池实现：
  - `include/vsag/thread_pool.h` — 公共接口
  - `src/impl/thread_pool/default_thread_pool.h/.cpp` — progschj 包装
  - `src/impl/thread_pool/safe_thread_pool.h` — 异常安全包装
- VSAG 线程池使用点：见 §2 分析，覆盖 `src/algorithm/` 下所有调用。
