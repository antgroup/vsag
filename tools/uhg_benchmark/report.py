# Copyright 2024-present the vsag project
# Licensed under the Apache License, Version 2.0.
"""Generate CSV, Chinese report and recall/QPS figures from completed benchmark JSON."""
import csv
import json
from pathlib import Path
import sys

out = Path(sys.argv[1])
rows = json.loads((out / 'results.json').read_text())
metadata = json.loads((out / 'metadata.json').read_text())
for row in rows:
    times = row['total_seconds_repeats']
    row['qps_min'] = row['queries'] / max(times)
    row['qps_max'] = row['queries'] / min(times)
    row['time_spread_ratio'] = max(times) / min(times)
fields = ['scenario', 'method', 'budget', 'recall_at_100', 'qps', 'qps_min', 'qps_max',
          'latency_p50_ms', 'latency_p95_ms', 'average_union_size']
with (out / 'results.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
scenarios = list(dict.fromkeys(r['scenario'] for r in rows))
text = ['# MS MARCO UHG 对比实验', '', f'已完成配置：{len(rows)}/40。', '',
        '## 数据集', '',
        f"使用 `{metadata['dataset']}`，包含 {metadata['base_count']:,} 条文档和 "
        f"{metadata['query_count']:,} 个查询。每条数据均有 BGE-M3 生成的 "
        f"{metadata['dense_dim']} 维稠密向量及稀疏向量。", '',
        '## 实验设置', '',
        '检索 top-100，完整混合距离为 '
        '`α × (1 − dense_dot) + (1 − α) × (1 − sparse_dot)`。'
        '固定权重为 α=0.3、0.5、0.7；混合场景将这三种权重近似均分给查询，用种子 42 打乱。', '',
        'Recall@100 以全库穷举得到的精确混合 top-100 为基准，不使用 MS MARCO 相关性标注。'
        f"查询单线程执行，每档重复 {rows[0]['repeats']} 次，曲线展示中位数 QPS。", '',
        '## Baseline：two-route', '',
        'HGraph 和 SINDI 分别召回相同数量候选，每路取 **100、150、200、300、500** 条。'
        '取并集并去重后，为每条候选计算稠密、稀疏两项完整距离，按混合距离重排返回 top-100。'
        '计时包含两路召回、去重、距离计算和排序。', '',
        '## UHG', '',
        '复用相同的 HGraph、SINDI 引导索引，构建最大出度为 64 的混合图。'
        '查询使用本次实验对应的 VSAG UHG 实现，束宽为 **100、200、300、500、1000**，'
        '入口数与束宽相同，剪枝系数为 0.3。两种方法在线稀疏查询均不剪枝。', '',
        '离线图按 11 个采样权重选择邻居，并按入选频次合并，采用并行构建。'
        '离线候选每路 100，稠密 ef=200，稀疏 query_prune_ratio=0.9；'
        '该构建时间独立于公共串行 `UHG::Build` 接口的耗时。', '',
        '## 计时说明', '',
        '表格中 QPS 的方括号给出三次计时中的最小值和最大值；曲线图仅展示中位数 QPS。'
        '查询线程未绑定 CPU，部分档位耗时波动明显，接近的 QPS 差异不作为稳定性能优势。', '']
if metadata.get('query_implementation'):
    text += ['## 查询实现', '',
             '本次运行使用 UHG 批量正排距离计算和池化访问标记，复用计算器及邻居批次缓冲区。'
             'HGraph 和 SINDI 继续提供引导召回，混合图复用原实验拓扑。新增正排向量会增加内存和索引文件大小。', '']
if metadata.get('optimization_report'):
    text += ['## UHG 查询优化复测', '',
             '本报告后续表格与两路对比图保留初版实验结果；优化版复测使用相同的混合图与引导索引，'
             '新增正排向量、批量距离计算和访问标记复用。', '',
             f"[查看优化复测报告]({metadata['optimization_report']})", '',
             f"![初版、优化版与 two-route 对比]({metadata['optimization_plot']})", '']
if len(rows) == 40:
    index = {(r['scenario'], r['method'], r['budget']): r for r in rows}
    a = index['alpha_0.500000', 'two_route', 300]
    b = index['alpha_0.500000', 'uhg', 300]
    c = index['alpha_0.700000', 'two_route', 200]
    d = index['alpha_0.700000', 'uhg', 200]
    e = index['mixed_discrete_seed42', 'two_route', 500]
    f = index['mixed_discrete_seed42', 'uhg', 500]
    g = index['mixed_discrete_seed42', 'uhg', 1000]
    def point(r):
        return f"Recall@100 {r['recall_at_100'] * 100:.3f}%，{r['qps']:.2f} QPS"
    text += ['## 关键档位对照', '',
             f"α=0.5：两路每路 300 个候选为 {point(a)}；UHG 束宽 300 为 {point(b)}。", '',
             f"α=0.7：两路每路 200 个候选为 {point(c)}；UHG 束宽 200 为 {point(d)}。", '',
             f"混合场景：两路每路 500 个候选为 {point(e)}；UHG 束宽 500 为 {point(f)}；"
             f"UHG 束宽 1000 为 {point(g)}。", '']
text += ['## Recall–QPS 性能曲线', '',
         f'![two-route 与 UHG 的 Recall–QPS 性能曲线]({(out / "recall_qps.png").resolve()})', '',
         '横轴为 Recall@100，纵轴为单线程 QPS（对数刻度），越靠右上越好。'
         '蓝色为 two-route，橙色为 UHG；点旁数字分别表示每路候选数和 UHG 束宽。'
         '四个子图分别对应 α=0.3、0.5、0.7 和混合权重场景；各点仅展示三次测量的中位数 QPS。', '',
         f'[下载 SVG 矢量图]({(out / "recall_qps.svg").resolve()})', '']
for scenario in scenarios:
    text += [f'## {scenario}', '',
             '| 方法 | 每路候选 / UHG 束宽 | Recall@100 | QPS 中位数 [范围] | P50 ms | P95 ms | 平均并集 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    group = [r for r in rows if r['scenario'] == scenario]
    for r in group:
        union = f"{r['average_union_size']:.1f}" if r['method'] == 'two_route' else '—'
        text.append(f"| {r['method']} | {r['budget']} | {r['recall_at_100']:.4f} | "
                    f"{r['qps']:.2f} [{r['qps_min']:.2f}, {r['qps_max']:.2f}] | {r['latency_p50_ms']:.3f} | {r['latency_p95_ms']:.3f} | {union} |")
    text += ['']
text += ['## 离线耗时', '', '| 项目 | 秒 |', '| --- | ---: |']
for key in ['ground_truth_build_or_load_seconds', 'hgraph_build_seconds',
            'sindi_build_seconds', 'uhg_graph_build_seconds', 'forward_vector_prepare_seconds']:
    if key in metadata:
        text.append(f'| {key} | {metadata[key]:.1f} |')
text += ['', '详细配置见 metadata.json，逐档原始结果见 results.json / results.csv。', '']
(out / 'report.md').write_text('\n'.join(text))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
ncols = min(2, len(scenarios))
nrows = math.ceil(len(scenarios) / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows),
                         constrained_layout=True, squeeze=False)
for ax, scenario in zip(axes.flat, scenarios):
    for method, marker in [('two_route', 'o'), ('uhg', 's')]:
        group = [r for r in rows if r['scenario'] == scenario and r['method'] == method]
        ax.plot([r['recall_at_100'] for r in group], [r['qps'] for r in group],
                marker=marker, label=method)
        for r in group:
            ax.annotate(str(r['budget']), (r['recall_at_100'], r['qps']),
                        xytext=(4, 4), textcoords='offset points', fontsize=8)
    ax.set(title=scenario.replace('alpha_0.', 'alpha=0.').replace('00000', ''),
           xlabel='Recall@100', ylabel='QPS (single thread, log scale)', yscale='log')
    ax.margins(x=.12, y=.2)
    ax.grid(alpha=.3)
    ax.legend()
for ax in list(axes.flat)[len(scenarios):]:
    ax.axis('off')
fig.suptitle('MS MARCO: exact hybrid top-100 retrieval')
fig.savefig(out / 'recall_qps.png', dpi=180)
fig.savefig(out / 'recall_qps.svg')
print(out / 'report.md')
