import json
import os
import re
import sys
import torch
import random
from llamafactory.chat import ChatModel

"""
留出法
在留出法训练的mixed本地模型上运行，对通过 0-split_data.py 分割过后的文件；
先通过使用 1-process_data_internal_test.py 对用于训练的 80% 的数据集进行 CoT 数据蒸馏，用于模型微调；
然后再用 剩下的 20% 的测试集中各抽取 50 道 CR 和 MU 问题，然后进行回答。
输入文件：
split_data/dev_CRMUS_CR_internal_test.json
split_data/dev_CRMUS_MU_internal_test.json

输出文件：
prediction_results/internal_test_CRMUS_CR_pred.json
prediction_results/internal_test_CRMUS_MU_pred.json
"""

# ================= 配置区域 (Configuration) =================

# 1. 基础路径
# 假设脚本位于 My_FYP/scripts/ 目录下，BASE_DIR 即为 My_FYP/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 输入数据路径 (方案二：读取划分出的内部测试集)
SPLIT_DIR = os.path.join(BASE_DIR, 'split_data')

# 输出结果路径
PREDICTION_DIR = os.path.join(BASE_DIR, 'prediction_results')
os.makedirs(PREDICTION_DIR, exist_ok=True)

# 2. 模型路径设置
# 基座模型 (请根据您实际使用的模型名称修改，如 "Qwen/Qwen1.5-7B-Chat")
MODEL_NAME_OR_PATH = "Qwen/Qwen1.5-7B-Chat"

# Adapter 路径 (指向您训练好的混合模型权重)
# 假设训练输出在 My_FYP/LLaMA-Factory/output/qwen_crmus_mixed_lora
ADAPTER_NAME_OR_PATH = os.path.join(BASE_DIR, "LLaMA-Factory", "output", "qwen_crmus_mixed_lora_internal_test")

# 3. 提示词模板 (必须与 1_process_data.py 中的训练指令完全一致)
# 只有指令一致，才能激活模型在训练中学到的"先分析后回答"的能力
CR_INSTRUCTION = "请仔细阅读故事，先进行常识推理，解释选项的合理性，最后给出答案。"
MU_INSTRUCTION = "请仔细阅读故事，先进行寓意分析，解释选项的合理性，最后给出答案。"

# ================= 核心功能函数 (Helper Functions) =================

def extract_answer(response_text):
    """
    从模型生成的长文本中提取选项字母 (A/B/C/D)
    模型输出通常为："【推理分析】... \n【答案】\nD"
    """
    # 策略1: 严格匹配 "【答案】" 或 "答案：" 后面的字母
    match = re.search(r"(?:【答案】|答案)[:：]?\s*([A-D])", response_text)
    if match:
        return match.group(1)

    # 策略2: 如果没有标准格式，寻找文本中最后出现的选项字母 (兜底策略)
    match = re.findall(r"([A-D])", response_text)
    if match:
        return match[-1]

    # 策略3: 实在找不到，返回 'C' 防止评测代码报错
    return "C"

def format_options(options):
    """将选项列表格式化为字符串"""
    return "\n".join(options)

# ================= 推理逻辑 (Inference Logic) =================

def process_dataset(task_type, input_filename, output_filename, chat_model):
    """
    读取测试集 -> 模型推理 -> 提取答案 -> 保存结果
    """
    input_file = os.path.join(SPLIT_DIR, input_filename)
    output_file = os.path.join(PREDICTION_DIR, output_filename)

    print(f"\n🚀 开始推理任务: {task_type}")
    print(f"   读取文件: {input_file}")

    if not os.path.exists(input_file):
        print(f"❌ 错误：找不到输入文件 {input_file}")
        print("   请先运行 scripts/0_split_data.py 生成数据集。")
        return

    # 读取数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # 随机抽取50个问题
        sample_size = min(50, len(data))
        data = random.sample(data, sample_size)

    print(f"   数据量: {len(data)} 条")
    results = []

    # 根据任务类型选择对应的指令
    instruction = CR_INSTRUCTION if task_type == "CR" else MU_INSTRUCTION

    # 开始逐条推理
    for i, item in enumerate(data):
        # 1. 构建 Input (与训练时保持格式一致)
        story = item.get('story', '')
        question = item.get('question', '')
        options_str = format_options(item.get('options', []))

        input_text = f"故事：\n{story}\n\n问题：\n{question}\n\n选项：\n{options_str}"

        # 2. 构造消息
        messages = [
            {"role": "user", "content": instruction + "\n" + input_text}
        ]

        # 3. 模型生成
        # temperature=0.1: 降低随机性，让模型更稳定地做题
        response = chat_model.chat(messages, temperature=0.1, top_p=0.9)
        response_text = response[0].response_text

        # 4. 解析答案
        pred_answer = extract_answer(response_text)

        # 5. 保存结果
        # 我们保留原始数据的所有字段，并填入预测结果
        result_item = item.copy()
        # 这里可以考虑跟 3-inference_generalist统一一下，同时把原本保留的正确答案，和模型预测的答案都放在一起
        result_item['answer'] = pred_answer         # 预测的选项 (A/B/C/D)
        result_item['model_reasoning'] = response_text # 保存完整的思维链，方便论文分析
        print(f"    问题ID: {result_item['id']} --- 答案: {result_item['answer']}")

        results.append(result_item)

        # 打印进度 (每10条打印一次)
        if (i + 1) % 10 == 0:
            print(f"   [{i+1}/{len(data)}] 预测: {pred_answer}")

    # 6. 保存最终文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ {task_type} 推理完成！结果已保存至: {output_file}")

# ================= 主执行入口 (Main Execution) =================

def main():
    print("=== 初始化模型 ===")
    # 检查 Adapter 路径是否存在
    if not os.path.exists(ADAPTER_NAME_OR_PATH):
        print(f"❌ 警告：找不到 Adapter 路径: {ADAPTER_NAME_OR_PATH}")
        print("   请确认训练是否成功，或者修改代码中的 ADAPTER_NAME_OR_PATH。")
        return

    # 配置模型参数
    args = {
        "model_name_or_path": MODEL_NAME_OR_PATH,
        "adapter_name_or_path": ADAPTER_NAME_OR_PATH,
        "template": "qwen",       # 确保与训练时的 template 一致
        "finetuning_type": "lora",
        "quantization_bit": 4,    # 保持 4-bit 推理以节省显存
    }

    try:
        chat_model = ChatModel(args)
        print("✅ 模型加载成功！")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    print("\n=== 开始批量推理 ===")

    # 1. 执行常识推理 (CR) 任务
    # 输入: split_data/dev_CRMUS_CR_internal_test.json
    # 输出: prediction_results/dev_CRMUS_CR_internal_test.json
    process_dataset(
        task_type="CR",
        input_filename="dev_CRMUS_CR_internal_test.json",
        output_filename="internal_test_CRMUS_CR_pred.json",
        chat_model=chat_model
    )

    # 2. 执行寓意理解 (MU) 任务
    # 输入: split_data/dev_CRMUS_MU_internal_test.json
    # 输出: prediction_results/dev_CRMUS_MU_internal_test.json
    process_dataset(
        task_type="MU",
        input_filename="dev_CRMUS_MU_internal_test.json",
        output_filename="internal_test_CRMUS_MU_pred.json",
        chat_model=chat_model
    )

    print("\n🎉 所有推理任务结束。请运行 scripts/4_eval.py 查看准确率报告。")

if __name__ == "__main__":
    main()