# Copyright 2024-present the vsag project
# Licensed under the Apache License, Version 2.0.
"""Compare original and optimized UHG using unchanged guide indexes and graph."""
import csv
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

old_dir, new_dir = map(Path, sys.argv[1:3])
old = json.loads((old_dir / 'results.json').read_text())
new = json.loads((new_dir / 'results.json').read_text())
a = json.loads((old_dir / 'metadata.json').read_text())
b = json.loads((new_dir / 'metadata.json').read_text())
assert a['base_count'] == b['base_count'] and a['query_count'] == b['query_count']
old_index = {(r['scenario'], r['method'], r['budget']): r for r in old}
scenarios = list(dict.fromkeys(r['scenario'] for r in new))
comparisons = []
for r in new:
    if r['method'] != 'uhg':
        continue
    before = old_index[r['scenario'], 'uhg', r['budget']]
    comparisons.append(dict(scenario=r['scenario'], beam=r['budget'],
                            old_recall=before['recall_at_100'], new_recall=r['recall_at_100'],
                            old_qps=before['qps'], new_qps=r['qps'],
                            speedup=r['qps'] / before['qps']))
with (new_dir / 'optimization-comparison.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(comparisons[0]))
    writer.writeheader()
    writer.writerows(comparisons)
fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
for ax, scenario in zip(axes.flat, scenarios):
    for data, method, label, marker in [(new, 'two_route', 'two-route', 'o'),
                                        (old, 'uhg', 'UHG original', 's'),
                                        (new, 'uhg', 'UHG optimized', '^')]:
        group = sorted([r for r in data if r['scenario'] == scenario and r['method'] == method],
                       key=lambda r: r['budget'])
        ax.plot([r['recall_at_100'] for r in group], [r['qps'] for r in group],
                marker=marker, label=label)
        for r in group:
            ax.annotate(str(r['budget']), (r['recall_at_100'], r['qps']),
                        xytext=(4, 4), textcoords='offset points', fontsize=8)
    ax.set(title=scenario, xlabel='Recall@100', ylabel='Median QPS (single thread)', yscale='log')
    ax.margins(x=.12, y=.2)
    ax.grid(alpha=.3)
    ax.legend()
for ax in list(axes.flat)[len(scenarios):]:
    ax.axis('off')
fig.suptitle('MS MARCO: same HGraph/SINDI guides and alpha-cover graph')
fig.savefig(new_dir / 'optimization-comparison.png', dpi=180)
fig.savefig(new_dir / 'optimization-comparison.svg')
text = ['# UHG 查询实现优化复测', '',
        f"复用原实验的 HGraph、SINDI、混合图和精确答案；基库 {b['base_count']:,} 条，查询 {b['query_count']:,} 条。"
        'top-100、权重、引导候选数和剪枝系数保持一致，每档单线程测三次，曲线只展示中位数 QPS。', '',
        '优化均在 UHG 层：新增稠密/稀疏正排向量，批量计算邻居距离，复用计算器、批次缓冲区和池化 visited；'
        '无过滤查询省去独立输出堆。HGraph/SINDI 的索引代码与两路基线逻辑均未修改。', '',
        '批次剪枝使用固定阈值，稠密 Batch4 内核与旧单点计算也有浮点归约差异；'
        '因此表中分别记录两版 Recall，同束宽速度比不等于严格同召回率速度比。'
        '线程未绑定 CPU，速度比基于两次运行的中位数，不能替代交错、固定 CPU 的消融测试。', '',
        f'![优化前后性能曲线]({(new_dir / "optimization-comparison.png").resolve()})', '',
        '| 权重场景 | 束宽 | 原 Recall | 新 Recall | 原 QPS | 新 QPS | 同束宽速度比 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
for r in comparisons:
    text.append(f"| {r['scenario']} | {r['beam']} | {r['old_recall']:.5f} | {r['new_recall']:.5f} | "
                f"{r['old_qps']:.2f} | {r['new_qps']:.2f} | {r['speedup']:.2f}× |")
a = json.loads((old_dir / 'metadata.json').read_text())
b = json.loads((new_dir / 'metadata.json').read_text())
text += ['', '## 存储开销', '',
         '| 项目 | GiB |', '| --- | ---: |',
         f"| 原 UHG 索引内存 | {a['uhg_index_bytes'] / 2**30:.2f} |",
         f"| 优化后 UHG 索引内存 | {b['uhg_index_bytes'] / 2**30:.2f} |", '',
         '正排向量使用额外存储，避免倒排点距扫描及逐节点公共接口开销。'
         'HGraph 引导搜索内部的中间距离缓存尚未接入；旧 v1 索引保持可读，使用公共距离回退路径。', '',
         f'[优化版详细实验报告]({(new_dir / "report.md").resolve()})', '']
(new_dir / 'optimization-comparison.md').write_text('\n'.join(text))
print(new_dir / 'optimization-comparison.md')
