import json
import os
import time
from openai import OpenAI  # 请确保已安装: pip install openai

"""
全量法
对使用全量训练的模型所使用的训练集，生成对应的CoT蒸馏之后的解释
输入文件:
data/dev_CRMUS_CR.json
data/dev_CRMUS_MU.json

生成文件：
processed_data/CRMUS_CR_train.json
processed_data/CRMUS_MU_train.json
"""

# ================= 配置区域 =================
# 请替换为您的 API Key 和 Base URL
# 示例：使用 DeepSeek 或 智谱 GLM 或 阿里 Qwen
API_KEY = "f5079e4a-654c-4137-96f6-1fec94f60629"  # 您的 API Key
BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"  # API 基础地址
MODEL_NAME = "deepseek-v3-1-terminus"  # 模型名称

# 输入输出路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed_data')
os.makedirs(PROCESSED_DIR, exist_ok=True)

# 初始化客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ================= Prompt 模板设计 =================

# CR (常识推理) 专用 Prompt
# 利用 'type' 字段增强提示，利用 'answer' 字段进行“答案已知”的逆向推理
CR_SYSTEM_PROMPT = "你是一位精通儿童认知发展和常识推理的教育专家。你的任务是解释为什么某个选项是正确的。"
CR_USER_TEMPLATE = """
任务：针对以下儿童故事题目，撰写一段详细的【推理分析】。
这是一道关于“{knowledge_type}”的测试题。

[故事内容]
{story}

[问题]
{question}

[选项]
{options_str}

[正确答案]
{correct_answer}

要求：
1. 结合故事中的线索和{knowledge_type}，逐步分析为什么正确答案是合理的。
2. 简要说明为什么其他干扰项是不合理的。
3. 输出必须包含明确的思维过程。

请严格按照以下格式输出（不要包含其他多余的寒暄）：
【推理分析】
(在此处生成你的详细分析...)
【答案】
{correct_answer_letter}
"""

# MU (寓意理解) 专用 Prompt
# 侧重于道德寓意和情节映射
MU_SYSTEM_PROMPT = "你是一位精通寓言故事和道德教育的专家。你的任务是剖析故事背后的深层寓意。"
MU_USER_TEMPLATE = """
任务：针对以下寓言故事，撰写一段【寓意解析】。

[故事内容]
{story}

[问题]
{question}

[选项]
{options_str}

[正确答案]
{correct_answer}

要求：
1. 首先概括故事的核心冲突或转折点。
2. 分析故事想要传达的教育意义或现实隐喻。
3. 对比各选项，解释为什么正确答案最贴切，而其他选项偏离了主旨。

请严格按照以下格式输出（不要包含其他多余的寒暄）：
【寓意解析】
(在此处生成你的详细分析...)
【答案】
{correct_answer_letter}
"""

# ================= 核心处理逻辑 =================

def call_llm_api(system_prompt, user_prompt, retries=3):
    """调用大模型 API，包含重试机制"""
    for i in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # 较低温度保证输出稳定
                max_tokens=1024
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  [API Error] 尝试 {i+1}/{retries}: {e}")
            time.sleep(2)  # 等待后重试
    return None

def format_options_text(options_list):
    """将选项列表格式化为文本"""
    return "\n".join(options_list)

def process_dataset(input_filename, output_filename, task_type="CR"):
    input_path = os.path.join(DATA_DIR, input_filename)
    output_path = os.path.join(PROCESSED_DIR, output_filename)

    print(f"🚀 开始处理文件: {input_filename}")
    print(f"   -> 目标路径: {output_filename}")

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 {input_path}")
        return

    processed_data = []
    total_count = len(raw_data)

    # 开始逐条处理
    for idx, item in enumerate(raw_data):
        item_id = item.get('id', f'idx_{idx}')
        print(f"[{idx+1}/{total_count}] 正在生成 ID: {item_id} ...", end="", flush=True)

        # 1. 提取字段
        story = item['story']
        question = item['question']
        options = item['options']
        options_str = format_options_text(options)
        correct_answer_letter = item['answer'] # 例如 "D"

        # 获取完整答案文本（方便大模型理解）
        # 假设选项格式是 "A. xxx", 我们简单匹配一下
        correct_answer_text = f"选项{correct_answer_letter}"
        for opt in options:
            if opt.strip().startswith(correct_answer_letter):
                correct_answer_text = opt
                break

        # 2. 构造 Prompt
        if task_type == "CR":
            # CR 数据通常有 'type' 字段 (如 "生物常识")
            k_type = item.get('type', '常识')
            sys_prompt = CR_SYSTEM_PROMPT
            usr_prompt = CR_USER_TEMPLATE.format(
                knowledge_type=k_type,
                story=story,
                question=question,
                options_str=options_str,
                correct_answer=correct_answer_text,
                correct_answer_letter=correct_answer_letter
            )
        else:
            # MU 数据
            sys_prompt = MU_SYSTEM_PROMPT
            usr_prompt = MU_USER_TEMPLATE.format(
                story=story,
                question=question,
                options_str=options_str,
                correct_answer=correct_answer_text,
                correct_answer_letter=correct_answer_letter
            )

        # 3. 调用 API 获取"蒸馏"后的思维链
        cot_output = call_llm_api(sys_prompt, usr_prompt)

        if cot_output:
            print(" ✅ 成功")

            # 4. 构造 LLaMA-Factory 训练数据格式
            # Instruction: 提示模型需要输出分析
            # Input: 原始故事和问题
            # Output: 大模型生成的"分析+答案"

            train_instruction = f"请仔细阅读故事，先进行{'常识推理' if task_type=='CR' else '寓意分析'}，解释选项的合理性，最后给出答案。"
            train_input = f"故事：\n{story}\n\n问题：\n{question}\n\n选项：\n{options_str}"

            alpaca_entry = {
                "instruction": train_instruction,
                "input": train_input,
                "output": cot_output  # 这里是高质量的 Teacher 生成内容
            }

            processed_data.append(alpaca_entry)
        else:
            print(" ❌ 失败 (API无响应)")

        # 避免触发限流，每条暂停一下
        time.sleep(0.5)

        # 5. 每处理10条保存一次，防止程序崩溃白跑
        if (idx + 1) % 10 == 0:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)

    # 最终保存
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 处理完成！共生成 {len(processed_data)} 条训练数据。")
    print(f"📁 保存至: {output_path}")

if __name__ == "__main__":
    # 处理 CR (常识) 数据
    # 建议先运行 CR，确认没问题后再取消 MU 的注释
    print("=== 开始处理常识推理 (CR) 数据 ===")
    process_dataset('dev_CRMUS_CR.json', 'CRMUS_CR_train.json', task_type="CR")

    print("\n=== 开始处理寓意理解 (MU) 数据 ===")
    process_dataset('dev_CRMUS_MU.json', 'CRMUS_MU_train.json', task_type="MU")