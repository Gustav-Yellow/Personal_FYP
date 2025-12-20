#!/bin/bash

# 获取当前脚本所在目录的上级目录（即 My_FYP 根目录）
PROJECT_ROOT=$(cd "$(dirname "$0")/.."; pwd)
cd "$PROJECT_ROOT/LLaMA-Factory" || exit

echo "当前工作目录: $(pwd)"
echo "开始训练..."

# 启动 LLaMA Factory 训练
# 关键参数说明：
# --config_file: 指向你的 yaml 配置文件
# 环境变量 CUDA_VISIBLE_DEVICES=0 指定使用第一张显卡

CUDA_VISIBLE_DEVICES=0 llamafactory-cli train "$PROJECT_ROOT/configs/qlora_config.yaml"

echo "训练结束！模型权重已保存到 configs/qlora_config.yaml 中指定的 output_dir。"