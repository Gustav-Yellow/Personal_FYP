import json
import os
import random
import re
import sys
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from llamafactory.chat import ChatModel

"""
全量法：额外添加向量数据库索引
在全量训练的mixed本地模型上，对在0-generate_silver_label.py中收取的问题进行回答，并且比较与商用大模型答案之间的准确率
输入文件：
data/test_CRMUS_CR_manual.json
data/test_CRMUS_MU_manual.json

输出文件：
prediction_results/test_CRMUS_CR_pred_generalist.json
prediction_results/test_CRMUS_MU_pred_generalist.json
"""
# ================= 配置区域 =================
# 1. 基础路径 (自动定位到 My_FYP 根目录)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 输入数据路径 (读取 split_data 中的原始 json)
SPLIT_DIR = os.path.join(BASE_DIR, 'split_data')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# 输出结果路径
PREDICTION_DIR = os.path.join(BASE_DIR, 'prediction_results') # 改为保存到 processed_data 或者 prediction_results
os.makedirs(PREDICTION_DIR, exist_ok=True)

# 2. 模型路径
MODEL_NAME_OR_PATH = "Qwen/Qwen1.5-7B-Chat"
# 确保这里指向你训练好的 Internal Test LoRA
ADAPTER_NAME_OR_PATH = os.path.join(BASE_DIR, "LLaMA-Factory", "output", "qwen_crmus_mixed_lora_internal_test")

# 3. RAG 知识库路径
COT_INDEX_PATH = os.path.join(BASE_DIR, "processed_data", "cot_knowledge_base_full.index")
COT_META_PATH = os.path.join(BASE_DIR, "processed_data", "cot_knowledge_base_full_meta.json")
COMMONSENSE_INDEX_PATH = os.path.join(BASE_DIR, "processed_data", "commonsense.index")
COMMONSENSE_META_PATH = os.path.join(BASE_DIR, "processed_data", "commonsense_meta.json")

# 4. Embedding 模型路径
EMBEDDING_MODEL_PATH = os.path.join(BASE_DIR, "models", "bge-large-zh-v1.5", "AI-ModelScope", "bge-large-zh-v1___5")

# 5. 指令 (Instruction)
CR_INSTRUCTION = "请仔细阅读故事，先进行常识推理，解释选项的合理性，最后给出答案。"
MU_INSTRUCTION = "请仔细阅读故事，先进行寓意分析，解释选项的合理性，最后给出答案。"

# ================= 资源加载 =================
print("1. 正在加载 Embedding 模型 (CPU模式)...")
embed_model = SentenceTransformer(EMBEDDING_MODEL_PATH, device='cpu')

print("2. 正在加载向量库...")
cot_index = faiss.read_index(COT_INDEX_PATH)
with open(COT_META_PATH, 'r', encoding='utf-8') as f:
    cot_meta = json.load(f)

cs_index = faiss.read_index(COMMONSENSE_INDEX_PATH)
with open(COMMONSENSE_META_PATH, 'r', encoding='utf-8') as f:
    cs_meta = json.load(f)


# ================= 功能函数 =================
def format_input_for_model(item):
    """
    将原始 JSON 的 story, question, options 拼装成模型训练时见过的 input 格式
    """
    options_str = "\n".join(item['options'])
    formatted_text = f"故事：\n{item['story']}\n\n问题：\n{item['question']}\n\n选项：\n{options_str}"
    return formatted_text


def retrieve_rag_content(query_text, task_type):
    """
    RAG 检索函数
    :param query_text: 检索用的文本
    :param task_type: 当前任务类型 "CR" 或 "MU"
    """
    # 向量化
    query_vec = embed_model.encode([query_text], normalize_embeddings=True)
    query_vec = np.array(query_vec).astype('float32')

    rag_info = ""

    # ==========================================
    # 1. 检索常识 (Commonsense)
    # ==========================================
    """ 迭代三：k值在下面可能会被过滤，添加判断，如果是CR则k=3,如果是Mu则k=1 """
    if task_type == "CR":
        # print(f"当前的k值为：{3}")
        D, I = cs_index.search(query_vec, k=3)
    elif task_type == "MU":
        # print(f"当前的k值为：{1}")
        D, I = cs_index.search(query_vec, k=1)

    """ 迭代三：增加相似度门槛，避免错误信息填入"""
    commonsense_list = []
    for i, idx in enumerate(I[0]):
        distance = D[0][i]

        if idx == -1: continue;

        meta = cs_meta[idx]

        """ 迭代三：类型过滤 """
        # 如果当前是做 CR 题，但检索出来的是 MU 的道理，直接丢弃！
        # 反之亦然。这能彻底消除跨领域的干扰
        if task_type == "CR" and meta['task_type'] != "CR":
            continue
        if task_type == "MU" and meta['task_type'] != "MU":
            continue

        # 设置相似度门槛
        # L2 距离：越小越相似。通常 < 0.8 或 < 1.0 (取决于向量分布) 是比较相关的。
        # 如果距离太大，说明这是一个生搬硬套的常识，我们直接丢弃。
        # 你可以先打印 print(distance) 观察一下正常的相关距离是多少
        # 这里的 1.2 是一个经验值，可以手动调整
        threshold = 1.2
        if distance > threshold:
            continue

        # 把 Category 加进去，让大模型知道这是哪方面的常识
        # 格式示例： [生物习性] 狮子是肉食动物...
        formatted_knowledge = f"[{meta['category']}] {meta['knowledge']}"
        commonsense_list.append(formatted_knowledge)

        # 只取前 2-3 条最相关的即可，不用太多
        if len(commonsense_list) >= 3:
            break

    if commonsense_list:
        rag_info += "【相关背景知识】\n"
        for i, cs in enumerate(commonsense_list, 1):
            rag_info += f"{i}. {cs}\n"

    """ 迭代二: 将 K 从2个变成3个，此时还没有相似度门槛 """
    # D, I = cs_index.search(query_vec, k=3)
    # commonsense_list = []
    # for idx in I[0]:
    #     if idx != -1:
    #         commonsense_list.append(cs_meta[idx]['knowledge'])
    # if commonsense_list:
    #     rag_info += "【相关常识参考】\n"
    #     for i, cs in enumerate(commonsense_list, 1):
    #         rag_info += f"{i}. {cs}\n"

    """ 迭代一：当前版本的rag信息中会包含之前的答案，有可能成为干扰信息 """
    # 2. 找相似题 (CoT)
    # D, I = cot_index.search(query_vec, k=1)
    # idx = I[0][0]
    # if idx != -1:
    #     match = cot_meta[idx]
    #     rag_info += "\n【相似题目解析参考】\n"
    #     # 注意：CoT 库里的 input 已经是拼装好的格式，可以直接用
    #     rag_info += f"相似故事：{match['input'][:100]}...\n"
    #     rag_info += f"参考分析逻辑：{match['output']}\n"

    """ 迭代二: 如果是CR，就不带CoT """
    # 2. 找相似题 (CoT)
    if task_type == "MU":
        # 执行搜索
        D, I = cot_index.search(query_vec, k=1)

        idx = I[0][0]
        distance = D[0][0] # 获取相似度距离 (L2距离)

        """ 迭代三：增加相似度门槛，避免错误信息填入 """
        # 只有距离小于 1.1 时才认为足够相似，否则视为噪声，不予采纳
        COT_THRESHOLD = 1.1

        if idx != -1 and distance < COT_THRESHOLD:
            match = cot_meta[idx]

            # 移除标准答案，只保留推理 (防止答案泄露)
            raw_output = match['output']
            clean_reasoning = raw_output.split("【答案】")[0].strip()
            clean_reasoning = clean_reasoning.split("答案：")[0].strip()

            rag_info += "\n【相似题目解析参考】（仅供参考逻辑）\n"
            # 加上距离显示，方便你在 json 结果里调试观察
            rag_info += f"[Sim: {distance:.2f}] 相似故事：{match['input'][:50]}...\n"
            rag_info += f"参考分析逻辑：{clean_reasoning}\n"
        else:
            # 如果距离太远，什么都不加，相当于 RAG 对此部分保持沉默
            pass

    else:
        # CR 任务：显式跳过，不添加任何 CoT 信息
        pass

    return rag_info



def extract_answer(response_text):
    """
    从模型输出中提取 A/B/C/D
    """
    # 优先匹配 "【答案】C" 这种明确格式
    match = re.search(r"(?:【答案】|答案)[:：]?\s*([A-D])", response_text)
    if match: return match.group(1)

    # 如果没找到，尝试找最后出现的选项字母，防止提取到推理过程中的字母
    # 比如 "排除A选项..."
    matches = re.findall(r"([A-D])", response_text)
    if matches:
        # 这里需要谨慎，因为模型可能会说 "A不对"，简单的取最后一个有风险
        # 但在你的 CoT 训练数据中，答案通常在最后。
        return matches[-1]

    return "C" # 兜底策略


def process_dataset(task_type, input_filename, output_filename, chat_model, instruction):
    input_path = os.path.join(DATA_DIR, input_filename)
    output_path = os.path.join(PREDICTION_DIR, output_filename)

    if not os.path.exists(input_path):
        print(f"⚠️ 跳过: {input_path} (文件不存在)")
        return

    print(f"\n正在处理: {input_filename} ...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # 只提取50问题来回答
        sample_size = min(50, len(data))
        data = random.sample(data, sample_size)

    results = []

    for item in tqdm(data):
        # 1. 拼装输入文本 (Story + Question + Options)
        # 这是为了给 RAG 做检索，也是给模型看的内容
        model_input_text = format_input_for_model(item)

        # 2. RAG 检索 （老版本）
        # rag_context = retrieve_rag_content(model_input_text, instruction)

        """ 迭代三：优化用于检索的query长度 """
        # 优化检索用的 Query
        # 策略：故事太长会有噪声，我们只取故事的前 150 字 + 问题 进行检索
        # 这样 BGE 更容易抓住重点，而不是被故事细节带偏
        search_query = f"{item['story'][:150]}... {item['question']}"
        # 使用优化后的 query 进行检索
        rag_context = retrieve_rag_content(search_query, task_type)

        # 3. 构造最终 Prompt
        final_prompt = f"""你是一个逻辑推理专家。请参考以下资料回答问题。

{rag_context}
----------------
【当前任务】
{instruction}

{model_input_text}

请基于当前故事进行独立分析，最后给出答案。
"""
        # 4. 调用模型
        messages = [{"role": "user", "content": final_prompt}]
        try:
            responses = chat_model.chat(
                messages,
                max_new_tokens=512,
                temperature=0.3
            )
            response_text = responses[0].response_text
        except Exception as e:
            print(f"Error: {e}")
            response_text = "Error"

        # 5. 提取答案
        pred_answer = extract_answer(response_text)

        # 6. 保存结果 (完全保留原始字段 + 新增字段)
        result_item = item.copy() # 这一步保留了 id, title, story, question, options, answer, type, domain

        # 添加新字段
        result_item['pred_answer'] = pred_answer     # 预测的选项字母 (如 "C")
        result_item['model_reasoning'] = response_text  # 完整的推理过程
        # 如果你想保存 RAG 检索到的内容用于调试，可以取消下面注释
        result_item['rag_debug_info'] = rag_context

        print(f"\n答案:{result_item['answer']}, 预测:{result_item['pred_answer']}\n")

        results.append(result_item)

    # 保存文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ 结果已保存至: {output_path}")


# ================= 主程序入口 =================
if __name__ == "__main__":

    print("3. 正在加载 Qwen LLM...")
    if not os.path.exists(ADAPTER_NAME_OR_PATH):
        print(f"❌ 错误：找不到 Adapter: {ADAPTER_NAME_OR_PATH}")
        sys.exit(1)

    args = {
        "model_name_or_path": MODEL_NAME_OR_PATH,
        "adapter_name_or_path": ADAPTER_NAME_OR_PATH,
        "template": "qwen",
        "finetuning_type": "lora",
        "quantization_bit": 4,
    }

    try:
        chat_model = ChatModel(args)
        print("✅ Qwen 加载成功！")
    except Exception as e:
        print(f"❌ Qwen 加载失败: {e}")
        sys.exit(1)

    print("\n=== 开始推理 (Local RAG) ===")

    # 1. 处理 CR (常识推理)
    # process_dataset(
    #     task_type="CR",
    #     input_filename="test_CRMUS_CR_manual.json",
    #     output_filename="test_CRMUS_CR_pred_generalist_vector.json",
    #     chat_model=chat_model,
    #     instruction=CR_INSTRUCTION
    # )

    # 2. 处理 MU (寓意理解)
    process_dataset(
        task_type="MU",
        input_filename="test_CRMUS_MU_manual.json",
        output_filename="test_CRMUS_MU_pred_generalist_vector.json",
        chat_model=chat_model,
        instruction=MU_INSTRUCTION
    )