#!/bin/bash

# 配置参数
TEST_BINARY="./build-release/tests/functests"
TEST_CASE="Pyramid Build & ContinueAdd Test"
LOG_FILE="test_run.log"

# 检查测试二进制文件是否存在
if [ ! -f "${TEST_BINARY}" ]; then
    echo "错误：测试程序未找到 [${TEST_BINARY}]"
    exit 1
fi

# 初始化计数器
RUN_COUNT=0

# 创建新鲜日志文件
echo "测试日志 - $(date)" > "${LOG_FILE}"

# 设置退出陷阱（Ctrl+C时显示统计信息）
trap 'echo -e "\n已执行 ${RUN_COUNT} 次测试"; exit 0' SIGINT

echo "开始压力测试，使用 Ctrl+C 停止..."
echo "实时日志：tail -f ${LOG_FILE}"

# 主测试循环
while true; do
    ((RUN_COUNT++))

    # 带时间戳运行测试
    TIMESTAMP=$(date +"%Y-%m-%d %T")
    echo "=== 运行 #${RUN_COUNT} [${TIMESTAMP}] ===" >> "${LOG_FILE}"

    # 执行测试并捕获结果
    if ! "${TEST_BINARY}" "${TEST_CASE}" >> "${LOG_FILE}" 2>&1; then
        echo -e "\n错误：第 ${RUN_COUNT} 次运行失败！"
        echo "最后10行日志输出："
        tail -n 10 "${LOG_FILE}"
        exit 1
    fi

    # 每10次输出进度
    if (( RUN_COUNT % 10 == 0 )); then
        echo -n "."
    fi
done
