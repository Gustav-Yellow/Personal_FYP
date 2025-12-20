import json
import os
import random
import time
import re
from openai import OpenAI
from sklearn.metrics import accuracy_score

# ================= 配置区域 =================
API_KEY = "f5079e4a-654c-4137-96f6-1fec94f60629"  # 您的 API Key
BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"  # API 基础地址
MODEL_NAME = "deepseek-r1-250528"  # 模型名称

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
REPORT_DIR = os.path.join(BASE_DIR, 'evaluation_reports')
os.makedirs(REPORT_DIR, exist_ok=True)

# 验证样本数
SAMPLE_SIZE = 50
RANDOM_SEED = 123 # 使用不同的种子，避免和之前的数据重叠

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 提示词：只要求输出答案，不需要详细分析，为了快速验证
VERIFY_PROMPT = """
请阅读以下题目并给出正确选项。

[故事]
{story}

[问题]
{question}

[选项]
{options_str}

请直接输出：【答案】选项字母
"""

def get_teacher_answer(story, question, options):
    prompt = VERIFY_PROMPT.format(
        story=story,
        question=question,
        options_str="\n".join(options)
    )
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        content = response.choices[0].message.content
        match = re.search(r"【答案】\s*([A-D])", content)
        if match: return match.group(1)
        match = re.findall(r"([A-D])", content)
        if match: return match[-1]
        return "C"
    except:
        return "C"

def verify_task(filename, task_name):
    print(f"\n 正在验证 Teacher 模型在 {task_name} 上的能力...")
    filepath = os.path.join(DATA_DIR, filename)

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 随机抽取 50 条有标准答案的数据
    random.seed(RANDOM_SEED)
    sample_data = random.sample(data, min(len(data), SAMPLE_SIZE))

    y_true = []    # 官方标准答案 (Gold Label)
    y_teacher = [] # Teacher 模型预测

    print(f"   抽取样本: {len(sample_data)} 条")

    for i, item in enumerate(sample_data):
        print(f"   [{i+1}/{len(sample_data)}] ...", end="", flush=True)

        gold = item['answer']
        pred = get_teacher_answer(item['story'], item['question'], item['options'])

        y_true.append(gold)
        y_teacher.append(pred)

        if gold == pred:
            print(" ✅")
        else:
            print(f" ❌ (真:{gold} vs 师:{pred})")

        time.sleep(0.3)

    acc = accuracy_score(y_true, y_teacher)
    print(f"\n 产生银标准的 Teacher 模型在 {task_name} 真值集上的准确率: {acc:.2%}")

    # 保存验证结果
    report_path = os.path.join(REPORT_DIR, f'teacher_verification_{task_name}.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Teacher Model Verification - {task_name}\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Sample Size: {len(sample_data)}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"\nGold Labels: {y_true}\n")
        f.write(f"Teacher Preds: {y_teacher}\n")

    return acc

if __name__ == "__main__":
    # 验证 CR
    acc_cr = verify_task("dev_CRMUS_CR.json", "CR")

    # 验证 MU
    acc_mu = verify_task("dev_CRMUS_MU.json", "MU")

    print("\n" + "="*40)
    print("教师模型可信度结论")
    print("="*40)
    print(f"CR 准确率: {acc_cr:.2%}")
    print(f"MU 准确率: {acc_mu:.2%}")

    if acc_cr > 0.8 and acc_mu > 0.8:
        print("结论：Teacher 模型表现优秀，生成的银标准高度可信，可以作为 Baseline 参考。")
    elif acc_cr > 0.6 and acc_mu > 0.6:
        print("结论：Teacher 模型表现良好，银标准具有参考价值，但需在论文中注明可能存在的噪声。")
    else:
        print("警告：Teacher 模型在某些任务上表现不佳，银标准的参考价值有限。")