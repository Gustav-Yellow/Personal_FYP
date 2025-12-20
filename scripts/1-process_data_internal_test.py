import json
import os
import time
from openai import OpenAI  # 需要安装: pip install openai

"""
留出法
对使用划分之后的80%的训练集进行训练的internal_test模型指定CoT数据蒸馏
输入文件：
split_data/dev_CRMUS_CR_internal_train.json
split_data/dev_CRMUS_MU_internal_train.json

输出文件：
processed_data/CRMUS_CR_internal_train.json
processed_data/CRMUS_MU_internal_train.json
"""

# ================= 配置区域 (Configuration) =================

# 1. API 设置 (请替换为您实际的 Key 和 URL)
# 推荐使用 智谱GLM-4, 阿里Qwen-Max 或 DeepSeek-V2.5
API_KEY = "f5079e4a-654c-4137-96f6-1fec94f60629"  # 您的 API Key
BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"  # API 基础地址
MODEL_NAME = "deepseek-v3-250324"  # 模型名称

# 2. 路径配置 (适配方案二：Split Data -> Processed Data)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_DIR = os.path.join(BASE_DIR, 'split_data')         # 输入：从划分好的内部训练集读取
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed_data') # 输出：保存为 LLaMA-Factory 训练格式
os.makedirs(PROCESSED_DIR, exist_ok=True)

# 初始化 API 客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ================= Prompt 模板设计 (Prompt Templates) =================

# --- 1. 常识推理 (CR) 专用模板 ---
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

# --- 2. 寓意理解 (MU) 专用模板 ---
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

# ================= 辅助功能函数 (Helper Functions) =================

def format_options_text(options_list):
    """将选项列表格式化为字符串"""
    return "\n".join(options_list)

def get_correct_option_text(options, letter):
    """根据选项字母(如'A')提取完整的选项文本"""
    for opt in options:
        # 兼容 "A.", "A．", "A " 等多种格式
        clean_opt = opt.strip()
        if clean_opt.startswith(letter):
            return clean_opt
    return f"选项{letter}" # 兜底返回

def call_llm_api(system_prompt, user_prompt, retries=3):
    """
    调用大模型 API 生成思维链 (CoT)，包含错误重试机制
    """
    for i in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # 较低温度保证输出逻辑稳定
                max_tokens=1024
            )
            content = response.choices[0].message.content.strip()
            return content
        except Exception as e:
            print(f"  ⚠️ [API Error] 尝试 {i+1}/{retries}: {e}")
            if i < retries - 1:
                time.sleep(2)  # 等待 2 秒后重试
            else:
                print("  ❌ API 调用失败，跳过此条数据。")
                return None

# ================= 主处理逻辑 (Main Logic) =================

def process_dataset(input_filename, output_filename, task_type="CR"):
    """
    读取 Split 数据 -> 调用 API 蒸馏 -> 保存为 Alpaca 格式
    """
    # 拼接完整路径
    input_path = os.path.join(SPLIT_DIR, input_filename)
    output_path = os.path.join(PROCESSED_DIR, output_filename)

    print(f"\n🚀 开始处理任务: {task_type}")
    print(f"   输入文件: {input_path}")
    print(f"   输出目标: {output_path}")

    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"❌ 错误：找不到输入文件 {input_path}")
        print("   请先运行 scripts/0_split_data.py 生成数据。")
        return

    # 读取数据
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    processed_data = []
    total_count = len(raw_data)

    print(f"   数据总量: {total_count} 条")

    # 逐条处理循环
    for idx, item in enumerate(raw_data):
        print(f"   [{idx+1}/{total_count}] 正在蒸馏 ID: {item.get('id', idx)} ... ", end="", flush=True)

        # 1. 提取字段
        story = item.get('story', '')
        question = item.get('question', '')
        options = item.get('options', [])
        correct_letter = item.get('answer', '').strip()

        # 格式化
        options_str = format_options_text(options)
        correct_text = get_correct_option_text(options, correct_letter)

        # 2. 根据任务类型选择 Prompt
        if task_type == "CR":
            k_type = item.get('type', '常识知识')
            sys_prompt = CR_SYSTEM_PROMPT
            usr_prompt = CR_USER_TEMPLATE.format(
                knowledge_type=k_type,
                story=story,
                question=question,
                options_str=options_str,
                correct_answer=correct_text,
                correct_answer_letter=correct_letter
            )
            # 训练时的指令 (Instruction)
            train_instruction = "请仔细阅读故事，先进行常识推理，解释选项的合理性，最后给出答案。"

        else: # MU
            sys_prompt = MU_SYSTEM_PROMPT
            usr_prompt = MU_USER_TEMPLATE.format(
                story=story,
                question=question,
                options_str=options_str,
                correct_answer=correct_text,
                correct_answer_letter=correct_letter
            )
            # 训练时的指令 (Instruction)
            train_instruction = "请仔细阅读故事，先进行寓意分析，解释选项的合理性，最后给出答案。"

        # 3. 调用 API 获取 CoT
        # 注意：这里是真实调用。如果您想先测试跑几条，可以在上面的 for 循环加 [:5] 切片
        cot_output = call_llm_api(sys_prompt, usr_prompt)

        if cot_output:
            print("✅ 成功")

            # 4. 构建 LLaMA-Factory 训练数据格式 (Alpaca Format)
            entry = {
                "instruction": train_instruction,
                "input": f"故事：\n{story}\n\n问题：\n{question}\n\n选项：\n{options_str}",
                "output": cot_output  # 这里包含了大模型生成的“推理分析 + 答案”
            }
            processed_data.append(entry)
        else:
            print("❌ 失败 (API无响应或出错)")

        # 5. 频率限制保护 (Rate Limiting)
        time.sleep(0.5)

        # 6. 定期保存 (每处理 20 条自动保存一次，防止程序意外中断丢失进度)
        if (idx + 1) % 20 == 0:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)

    # 循环结束，最终保存所有数据
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

    print(f"🎉 处理完成！共生成 {len(processed_data)} 条高质量训练数据。")
    print(f"📁 已保存至: {output_path}")

# ================= 执行入口 (Execution Entry) =================

if __name__ == "__main__":
    print("=== 开始执行数据蒸馏 (Data Distillation) ===")

    # 1. 处理常识推理 (CR) 内部训练集
    # 输入: split_data/dev_CRMUS_CR_internal_train.json
    # 输出: processed_data/CRMUS_CR_internal_train.json
    process_dataset(
        input_filename="dev_CRMUS_CR_internal_train.json",
        output_filename="CRMUS_CR_internal_train.json",
        task_type="CR"
    )

    print("\n" + "-"*50 + "\n")

    # 2. 处理寓意理解 (MU) 内部训练集
    # 输入: split_data/dev_CRMUS_MU_internal_train.json
    # 输出: processed_data/CRMUS_MU_internal_train.json
    process_dataset(
        input_filename="dev_CRMUS_MU_internal_train.json",
        output_filename="CRMUS_MU_internal_train.json",
        task_type="MU"
    )

    print("\n✅ 所有任务执行完毕。请检查 processed_data 文件夹。")