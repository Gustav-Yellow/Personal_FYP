import sys
import os
import json
import re
from openai import OpenAI
from tqdm import tqdm

# ================= 路径配置 =================
current_script_path = os.path.abspath(__file__)
utils_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(utils_dir)

print(f"当前项目根目录: {project_root}")

# ================= 配置区域 =================

INPUT_FILES = [
    # 文件名中必须包含 CR 或 MU 以便脚本自动识别类型
    os.path.join(project_root, "processed_data", "CRMUS_CR_train.json"),
    os.path.join(project_root, "processed_data", "CRMUS_MU_train.json")
]
OUTPUT_FILE = os.path.join(project_root, "processed_data", "commonsense_knowledge_base.json")

# DeepSeek 配置 (或你使用的其他模型)
PROVIDER = "deepseek"
API_KEY = "sk-270d89cece2e4045b5ec3d2e83a7f2e9"
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-reasoner"  # DeepSeek-V3

print(f"🚀 当前使用的提取模型: 【{MODEL_NAME}】")
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# ================= 核心逻辑 =================

def extract_knowledge(story, analysis, task_type):
    """
    提取常识，并要求模型进行分类
    task_type: "CR" (常识推理) 或 "MU" (寓意理解)
    """

    # 针对 CR 和 MU 设计略有不同的 Prompt 侧重点
    if task_type == "CR":
        category_instruction = "类别必须具体，例如：'物理常识'、'生物习性'、'植物特性'、'生活常识'、'时间概念'等。"
        focus_instruction = "重点关注因果逻辑、物理规律和生物本能。"
    else: # MU
        category_instruction = "类别主要为：'社会心理'、'伦理道德'、'人性弱点'、'处世哲学'、'价值判断'等。"
        focus_instruction = "重点关注人际互动、道德教训和抽象寓意。"

    prompt = f"""任务：作为一位百科全书编撰者，请阅读下面的故事和解析，提炼 1-2 条**高度抽象的**通用法则，并对其进行分类。

【输入素材】：
故事：{story[:600]}
解析：{analysis[:600]}

【严格要求】：
1. **内容抽象**：去语境化，不要包含故事里的具体角色（如狐狸、农夫），使用通用概念。
2. **准确分类**：{category_instruction}, {focus_instruction}
3. **格式**：仅返回一个 JSON 对象列表，包含 'knowledge' 和 'category' 两个字段。

【正确示范】：
[
  {{"knowledge": "猫科动物通常在夜间视力更好，适合夜间捕食", "category": "生物习性"}},
  {{"knowledge": "由于惯性，高速运动的物体很难立刻停止", "category": "物理常识"}},
  {{"knowledge": "贪婪往往会导致人失去理智从而忽视潜在风险", "category": "人性弱点"}}
]

请返回 JSON 列表："""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            stream=False
        )
        content = response.choices[0].message.content

        # 清洗与正则提取
        clean_content = content.replace("```json", "").replace("```", "").strip()
        match = re.search(r"\[.*\]", clean_content, re.DOTALL)

        if match:
            parsed = json.loads(match.group(0))
        else:
            parsed = json.loads(clean_content)

        # 容错处理：确保是列表
        if isinstance(parsed, dict):
            return [parsed]
        return parsed

    except Exception as e:
        # print(f"❌ 解析出错: {e}")
        return []

def save_checkpoint(data, filepath):
    """
    将当前内存中的所有数据写入文件
    (覆盖写入，保证 JSON 格式的完整性)
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        # print(f"💾 进度已保存 ({len(data)} 条)")
    except Exception as e:
        print(f"⚠️ 保存失败: {e}")

def main():
    # 内存中的主数据列表
    all_knowledge = []

    # 简单的去重集合 (防止同一条常识重复添加)
    seen_knowledge_hashes = set()

    print(f"📂 准备开始提取，结果将保存至: {OUTPUT_FILE}")
    print("💡 策略：每处理 10 条数据自动保存一次。")

    for file_path in INPUT_FILES:
        if not os.path.exists(file_path):
            print(f"⚠️ 跳过不存在的文件: {file_path}")
            continue

        filename = os.path.basename(file_path)
        print(f"\n📄 正在处理文件: {filename}")

        # 自动判断当前任务类型
        current_task_type = "CR" if "CR" in filename and "MU" not in filename else "MU"
        # 如果文件名类似 CRMUS_MU_train，上面逻辑可能误判，修正如下：
        if "CRMUS_MU" in filename: current_task_type = "MU"
        if "CRMUS_CR" in filename: current_task_type = "CR"

        print(f"   --> 识别任务类型为: 【{current_task_type}】")

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 计数器用于每10条保存
        batch_counter = 0

        for item in tqdm(data, desc=f"提取 {current_task_type}"):
            story = item.get('input', item.get('story', ''))
            analysis = item.get('output', item.get('answer', ''))

            if not story: continue

            # 调用 LLM
            facts = extract_knowledge(story, analysis, current_task_type)

            if isinstance(facts, list):
                for fact_item in facts:
                    # 容错：确保 fact_item 是字典且有内容
                    if not isinstance(fact_item, dict): continue

                    knowledge_text = fact_item.get('knowledge', '').strip()
                    category_text = fact_item.get('category', '通用常识').strip()

                    if len(knowledge_text) < 5: continue

                    # 简单的去重指纹
                    fact_hash = f"{current_task_type}_{knowledge_text}"

                    if fact_hash not in seen_knowledge_hashes:
                        seen_knowledge_hashes.add(fact_hash)

                        # 构建最终数据结构
                        record = {
                            "id": len(all_knowledge),
                            "task_type": current_task_type, # CR 或 MU
                            "category": category_text,      # 具体的常识分类
                            "knowledge": knowledge_text,    # 常识内容
                            "source_preview": story    # 来源预览
                        }
                        all_knowledge.append(record)
                        batch_counter += 1

            # --- 增量保存逻辑 ---
            # 每成功提取 10 个单位（不是 10 条常识，是处理了 10 次输入触发的保存）
            # 或者为了更安全，只要 all_knowledge 长度增加 10 就保存
            if batch_counter >= 10:
                save_checkpoint(all_knowledge, OUTPUT_FILE)
                batch_counter = 0 # 重置计数器

    # 最后再保存一次，确保收尾
    if all_knowledge:
        save_checkpoint(all_knowledge, OUTPUT_FILE)
        print(f"\n✅ 全部完成！共提取 {len(all_knowledge)} 条分类常识。")
    else:
        print("\n❌ 未提取到任何数据。")

if __name__ == "__main__":
    main()