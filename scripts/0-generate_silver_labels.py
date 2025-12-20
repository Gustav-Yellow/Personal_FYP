import json
import os
import random
import time
import re
from openai import OpenAI

"""
全量法
从不提供正确答案的CR和MU的测试集中各抽取50条数据，交给DeepSeek-R1模型来生成给mixed模型推理结果比较的“银标准”答案。
输入文件：（从中随机抽取50个）
data/test_CRMUS_CR_public.json
data/test_CRMUS_MU_public.json

输出文件：
data/test_CRMUS_CR_manual.json
data/test_CRMUS_MU_manual.json
"""

# ================= 配置区域 =================
# 建议使用推理能力强的模型，如 deepseek-reasoner (R1) 或 gpt-4o
API_KEY = "f5079e4a-654c-4137-96f6-1fec94f60629"  # 您的 API Key
BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"  # API 基础地址
MODEL_NAME = "deepseek-r1-250528"  # 模型名称

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# 抽样数量设置 (为了对齐方案二的验证集比例)
SAMPLE_SIZE_CR = 50
SAMPLE_SIZE_MU = 50
RANDOM_SEED = 42  # 固定种子，保证每次抽取的题目一样

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ================= 提示词模板 =================
# 这里的目标是让大模型做题，获取"参考答案"
GENERATION_PROMPT = """
你是一位逻辑严密的评估专家。请阅读以下儿童故事题，分析逻辑，并给出正确选项。

[故事]
{story}

[问题]
{question}

[选项]
{options_str}

请严格按照以下格式输出：
【分析】简单一句话解释原因
【答案】选项字母（A/B/C/D）
"""

def get_answer_from_api(story, question, options):
    options_str = "\n".join(options)
    prompt = GENERATION_PROMPT.format(
        story=story, question=question, options_str=options_str
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, # 零温度，追求最确定的答案
        )
        content = response.choices[0].message.content

        # 提取答案字母
        match = re.search(r"【答案】\s*([A-D])", content)
        if match:
            return match.group(1), content

        # 兜底匹配
        match = re.findall(r"([A-D])", content)
        if match:
            return match[-1], content

        return "C", content # 实在失败则返回C
    except Exception as e:
        print(f"API Error: {e}")
        return None, None

def generate_silver_dataset(input_file, output_file, sample_size):
    input_path = os.path.join(DATA_DIR, input_file)
    output_path = os.path.join(DATA_DIR, output_file)

    print(f"🚀 正在处理: {input_file} -> 抽取 {sample_size} 条")

    with open(input_path, 'r', encoding='utf-8') as f:
        full_data = json.load(f)

    # 随机抽样
    random.seed(RANDOM_SEED)
    if len(full_data) > sample_size:
        sampled_data = random.sample(full_data, sample_size)
    else:
        sampled_data = full_data

    processed_data = []

    for i, item in enumerate(sampled_data):
        print(f"   [{i+1}/{len(sampled_data)}] 正在请求 API 生成答案 (ID: {item.get('id')})...", end="", flush=True)

        silver_answer, reasoning = get_answer_from_api(
            item['story'], item['question'], item['options']
        )

        if silver_answer:
            # 将大模型的答案填入 'answer' 字段，作为"银标准"
            item['answer'] = silver_answer
            item['teacher_reasoning'] = reasoning # 保存大模型的推理供参考
            processed_data.append(item)
            print(f" ✅ 答案: {silver_answer}")
        else:
            print(f" ❌ 失败")

        time.sleep(0.5) # 避免限流

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    print(f"🎉 已保存至: {output_path}\n")

if __name__ == "__main__":
    # 生成 CR 银标准数据
    generate_silver_dataset(
        "test_CRMUS_CR_public.json",
        "test_CRMUS_CR_manual.json",
        SAMPLE_SIZE_CR
    )

    # 生成 MU 银标准数据
    generate_silver_dataset(
        "test_CRMUS_MU_public.json",
        "test_CRMUS_MU_manual.json",
        SAMPLE_SIZE_MU
    )