import json
import os
import random
import re
import faiss
from llamafactory.chat import ChatModel
from sentence_transformers import SentenceTransformer

"""
全量法
在全量训练的mixed本地模型上，对在0-generate_silver_label.py中收取的问题进行回答，并且比较与商用大模型答案之间的准确率
输入文件：
data/test_CRMUS_CR_manual.json
data/test_CRMUS_MU_manual.json

输出文件：
prediction_results/test_CRMUS_CR_pred_generalist.json
prediction_results/test_CRMUS_MU_pred_generalist.json
"""

# ================= 配置区域 =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PRED_DIR = os.path.join(BASE_DIR, 'prediction_results')
os.makedirs(PRED_DIR, exist_ok=True)

# 方案一使用的模型：全量混合训练模型
MODEL_PATH = "Qwen/Qwen1.5-7B-Chat"
ADAPTER_PATH = os.path.join(BASE_DIR, "LLaMA-Factory", "output", "qwen_crmus_mixed_lora")

# 提示词 (需与混合模型训练时保持一致)
# 假设您训练混合模型时也用了 CoT 风格的指令
CR_INSTRUCTION = "请仔细阅读故事，先进行常识推理，解释选项的合理性，最后给出答案。"
MU_INSTRUCTION = "请仔细阅读故事，先进行寓意分析，解释选项的合理性，最后给出答案。"

def extract_answer(response_text):
    match = re.search(r"(?:【答案】|答案)[:：]?\s*([A-D])", response_text)
    if match: return match.group(1)
    match = re.findall(r"([A-D])", response_text)
    if match: return match[-1]
    return "C"

def process_inference_generalist(task_type, input_filename, output_filename, model):
    input_file = os.path.join(DATA_DIR, input_filename)
    output_file = os.path.join(PRED_DIR, output_filename)

    print(f"\n🚀 开始推理任务: {task_type}")
    print(f"   读取文件: {input_file}")

    if not os.path.exists(input_file):
        print(f"❌ 错误：找不到文件 {input_file}。请先运行 0_generate_silver_labels.py")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # 随机抽取50个问题
        sample_size = min(50, len(data))
        data = random.sample(data, sample_size)

    results = []
    instruction = CR_INSTRUCTION if task_type == "CR" else MU_INSTRUCTION

    for i, item in enumerate(data):
        options_str = "\n".join(item['options'])
        input_text = f"故事：\n{item['story']}\n\n问题：\n{item['question']}\n\n选项：\n{options_str}"

        messages = [{
            "role": "user",
            "content": instruction + "\n" + input_text
        }]

        # 本地模型推理
        response = model.chat(messages, temperature=0.1)
        response_text = response[0].response_text

        pred_answer = extract_answer(response_text)

        # 保存结果
        # 注意：这里我们保留了 'answer' (即大模型生成的银标准答案) 作为 truth_label 供参考
        # 并将本地模型的预测存入 'pred_answer'
        result_item = item.copy()
        result_item['answer'] = item['answer'] # 保留银标准作为"正确答案"用于eval
        result_item['pred_answer'] = pred_answer  # 本地模型的预测
        result_item['model_reasoning'] = response_text
        results.append(result_item)

        print(f"   [{i+1}/{len(data)}] 银标准:{item['answer']} vs 本地预测:{pred_answer}")

        if (i + 1) % 10 == 0:
            print(f"   [{i+1}/{len(data)}] 预测: {pred_answer}")

    # 保存
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ 结果已保存至: {output_file}\n")

if __name__ == "__main__":
    # 加载模型
    args = {
        "model_name_or_path": MODEL_PATH,
        "adapter_name_or_path": ADAPTER_PATH,
        "template": "qwen",
        "finetuning_type": "lora",
        "quantization_bit": 4,
    }

    try:
        print("正在加载 Mixed LoRA 模型...")
        chat_model = ChatModel(args)
    except Exception as e:
        print(f"模型加载失败，请检查路径: {ADAPTER_PATH}")
        exit()

    # 1. 对 CR 手动抽样集进行推理
    process_inference_generalist(
        "CR",
        "test_CRMUS_CR_manual.json",          # 输入：含有大模型答案的抽样集
        "test_CRMUS_CR_pred_generalist.json", # 输出：本地模型的预测结果
        chat_model
    )

    # 2. 对 MU 手动抽样集进行推理
    process_inference_generalist(
        "MU",
        "test_CRMUS_MU_manual.json",
        "test_CRMUS_MU_pred_generalist.json",
        chat_model
    )