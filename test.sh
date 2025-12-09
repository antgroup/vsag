#!/bin/bash

# 配置参数
CONCURRENCY=4  # 同时并发的进程数

# 模式1：批次式并行（每次启动N个进程，全部完成后开始下一批）
batch_parallel() {
    while true; do
        echo "启动 $CONCURRENCY 个并行测试进程..."
        for ((i=1; i<=$CONCURRENCY; i++)); do
            ./build-release/tests/functests "Pyramid Serialize File" &
        done
        wait  # 等待本批次所有进程完成
    done
}

# 模式2：持续并行（始终保持N个进程在运行）
persistent_parallel() {
    while true; do
        jobs_count=$(jobs -p | wc -l)
        if [ $jobs_count -lt $CONCURRENCY ]; then
            ./build-release/tests/functests "Pyramid Serialize File" &
        else
            sleep 0.1  # 避免CPU占用过高
        fi
    done
}

# 选择执行模式（取消注释其中一个）
batch_parallel   # 批次式并行
# persistent_parallel  # 持续并行
