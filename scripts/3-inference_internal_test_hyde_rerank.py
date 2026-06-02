import json
import os
import random
import re
import sys
import numpy as np
import faiss
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from llamafactory.chat import ChatModel

# ================= 配置区域 (针对 Internal Test 的修改) =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_DIR = os.path.join(BASE_DIR, 'split_data') # 从 split_data 读取数据
PREDICTION_DIR = os.path.join(BASE_DIR, 'prediction_results')
os.makedirs(PREDICTION_DIR, exist_ok=True)

MODEL_NAME_OR_PATH = "Qwen/Qwen1.5-7B-Chat"
# 【关键】加载 Internal Test 专用的 LoRA 权重
ADAPTER_NAME_OR_PATH = os.path.join(BASE_DIR, "LLaMA-Factory", "output", "qwen_crmus_mixed_lora_internal_test")

# 【关键】加载 Internal 版本的 CoT 知识库
COT_INDEX_PATH = os.path.join(BASE_DIR, "processed_data", "cot_knowledge_base_internal.index")
COT_META_PATH = os.path.join(BASE_DIR, "processed_data", "cot_knowledge_base_internal_meta.json")

# 常识库保持全局通用
COMMONSENSE_INDEX_PATH = os.path.join(BASE_DIR, "processed_data", "commonsense.index")
COMMONSENSE_META_PATH = os.path.join(BASE_DIR, "processed_data", "commonsense_meta.json")

# 模型路径
EMBEDDING_MODEL_PATH = os.path.join(BASE_DIR, "models", "bge-large-zh-v1.5", "AI-ModelScope", "bge-large-zh-v1___5")
RERANKER_MODEL_PATH = os.path.join(BASE_DIR, "models", "Xorbits", "bge-reranker-large")

CR_INSTRUCTION = "请仔细阅读故事，先进行常识推理，解释选项的合理性，最后给出答案。"
MU_INSTRUCTION = "请仔细阅读故事，先进行寓意分析，解释选项的合理性，最后给出答案。"

# ================= 资源加载 =================
print("1. 正在加载 Embedding 模型 (用于 HyDE 向量召回)...")
if not os.path.exists(EMBEDDING_MODEL_PATH):
    raise FileNotFoundError(f"❌ 找不到 Embedding 模型: {EMBEDDING_MODEL_PATH}")
embed_model = SentenceTransformer(EMBEDDING_MODEL_PATH, device='cpu')

print("2. 正在加载 Reranker 模型 (用于精准打分过滤)...")
if not os.path.exists(RERANKER_MODEL_PATH):
    raise FileNotFoundError(f"❌ 找不到 Reranker 模型: {RERANKER_MODEL_PATH}")
rerank_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_PATH)
rerank_model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_PATH)
rerank_model.eval()
if torch.cuda.is_available():
    rerank_model.to('cuda')

print("3. 正在加载 FAISS 向量库 (Internal 版本)...")
cot_index = faiss.read_index(COT_INDEX_PATH)
with open(COT_META_PATH, 'r', encoding='utf-8') as f: cot_meta = json.load(f)
cs_index = faiss.read_index(COMMONSENSE_INDEX_PATH)
with open(COMMONSENSE_META_PATH, 'r', encoding='utf-8') as f: cs_meta = json.load(f)


# ================= 功能函数 =================

def generate_hyde_doc(item, chat_model):
    """【HyDE 阶段】让大模型生成一个假设性解答方向"""
    # 优化 Prompt：强硬要求直接输出，禁止寒暄和前摇分析
    prompt = f"""你是一个阅读理解专家。请简要提取解答以下问题所需的底层核心规律（物理常识或道德哲理）。
要求：限100字以内。直接输出核心规律，**绝对不要**包含任何如“好的”、“根据故事分析”等前言后语。

故事：{item['story']}
问题：{item['question']}
核心规律："""

    try:
        # 修改点：把 max_new_tokens 从 64 放宽到 256，给模型足够的空间把话说完
        responses = chat_model.chat([{"role": "user", "content": prompt}], max_new_tokens=256, temperature=0.5)
        hyde_result = responses[0].response_text.strip()

        # 调试用：你可以取消下面这行的注释，看看大模型到底生成了什么鬼东西
        # print(f"\n[HyDE Debug] 题目: {item['question']} \n生成规律: {hyde_result}\n")

        return hyde_result
    except Exception as e:
        return f"{item['story'][:100]} {item['question']}"

def get_rerank_score(query, document):
    """【Rerank 阶段】计算 Query 和 Document 的交叉注意力相关性得分"""
    pairs = [[query, document]]
    with torch.no_grad():
        inputs = rerank_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        scores = rerank_model(**inputs, return_dict=True).logits.view(-1, ).float()
    return scores.item()

def retrieve_rag_content_hybrid(embed_query, rerank_query, task_type):
    """
    组合检索策略：
    1. 使用 embed_query (HyDE增强) 去 FAISS 中进行高召回 (粗排 k=10)
    2. 使用 rerank_query (原始故事和问题) 去进行高精度验证 (精排 top-3)
    """
    query_vec = embed_model.encode([embed_query], normalize_embeddings=True).astype('float32')
    rag_info = ""

    # ================= 1. 常识库混合检索 =================
    D, I = cs_index.search(query_vec, k=10)
    candidates = []

    for i, idx in enumerate(I[0]):
        if idx == -1: continue
        meta = cs_meta[idx]
        if task_type == "CR" and meta['task_type'] != "CR": continue
        if task_type == "MU" and meta['task_type'] != "MU": continue
        candidates.append(f"[{meta['category']}] {meta['knowledge']}")

    reranked_results = [(get_rerank_score(rerank_query, doc), doc) for doc in candidates]
    reranked_results.sort(key=lambda x: x[0], reverse=True)
    final_cs = [doc for score, doc in reranked_results if score > 0][:3]

    if final_cs:
        rag_info += "【相关背景知识】\n"
        for i, cs in enumerate(final_cs, 1):
            rag_info += f"{i}. {cs}\n"

    # ================= 2. CoT库混合检索 (仅MU任务) =================
    if task_type == "MU":
        D, I = cot_index.search(query_vec, k=5)
        cot_candidates = []
        for i, idx in enumerate(I[0]):
            if idx == -1: continue
            raw_output = cot_meta[idx]['output']
            clean_reasoning = raw_output.split("【答案】")[0].split("答案：")[0].strip()
            cot_candidates.append(f"相似故事：{cot_meta[idx]['input'][:50]}...\n参考分析逻辑：{clean_reasoning}")

        cot_reranked = [(get_rerank_score(rerank_query, doc), doc) for doc in cot_candidates]
        cot_reranked.sort(key=lambda x: x[0], reverse=True)

        if cot_reranked and cot_reranked[0][0] > 0:
            best_score = cot_reranked[0][0]
            best_doc = cot_reranked[0][1]
            rag_info += f"\n【相似题目解析参考】\n[Rerank Score: {best_score:.2f}] {best_doc}\n"

    return rag_info

def extract_answer(response_text):
    match = re.search(r"(?:【答案】|答案)[:：]?\s*([A-D])", response_text)
    if match: return match.group(1)
    match = re.findall(r"([A-D])", response_text)
    if match: return match[-1]
    return "C"

def process_dataset_internal(task_type, input_filename, output_filename, chat_model, instruction):
    """处理 Internal Test，保持与 Generalist 一致的 50 题抽样对比标准"""
    input_path = os.path.join(SPLIT_DIR, input_filename)
    output_path = os.path.join(PREDICTION_DIR, output_filename)

    with open(input_path, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
        random.seed(42) # 【关键】固定随机种子，确保每次抽取的 50 题完全相同
        sample_size = min(50, len(full_data))
        data = random.sample(full_data, sample_size)

    results = []
    for item in tqdm(data, desc=f"处理 {task_type}"):
        model_input_text = f"故事：\n{item['story']}\n\n问题：\n{item['question']}\n\n选项：\n" + "\n".join(item['options'])

        # 步骤 1: 生成 HyDE 假设性解答
        hyde_doc = generate_hyde_doc(item, chat_model)

        # 步骤 2: 准备双路 Query
        embed_query = f"问题：{item['question']}。底层逻辑：{hyde_doc}"
        rerank_query = f"{item['story'][:150]}... {item['question']}"

        # 步骤 3: 组合 RAG 检索
        rag_context = retrieve_rag_content_hybrid(embed_query, rerank_query, task_type)

        # 步骤 4: 最终生成
        final_prompt = f"""你是一个逻辑推理专家。请参考以下资料回答问题。

{rag_context}
----------------
【当前任务】
{instruction}

{model_input_text}

请基于当前故事进行独立分析，最后给出答案。
"""
        try:
            responses = chat_model.chat([{"role": "user", "content": final_prompt}], max_new_tokens=512, temperature=0.3)
            response_text = responses[0].response_text
        except Exception as e:
            response_text = "Error"

        result_item = item.copy()
        result_item['pred_answer'] = extract_answer(response_text)
        result_item['hyde_doc'] = hyde_doc
        result_item['model_reasoning'] = response_text
        result_item['rag_debug_info'] = rag_context
        results.append(result_item)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ Internal Test 混合模式结果已保存至: {output_path}")


if __name__ == "__main__":
    print("4. 正在加载 Qwen LLM (挂载 Internal LoRA) ...")
    if not os.path.exists(ADAPTER_NAME_OR_PATH):
        print(f"❌ 错误：找不到 Adapter: {ADAPTER_NAME_OR_PATH}")
        sys.exit(1)

    chat_model = ChatModel({
        "model_name_or_path": MODEL_NAME_OR_PATH,
        "adapter_name_or_path": ADAPTER_NAME_OR_PATH,
        "template": "qwen",
        "finetuning_type": "lora",
        "quantization_bit": 4
    })

    print("\n=== 开始推理 (Internal Test HyDE + Rerank 终极模式) ===")
    process_dataset_internal(
        "CR",
        "dev_CRMUS_CR_internal_test.json",
        "internal_test_CRMUS_CR_pred_hyde_rerank.json",
        chat_model,
        CR_INSTRUCTION
    )

    process_dataset_internal(
        "MU",
        "dev_CRMUS_MU_internal_test.json",
        "internal_test_CRMUS_MU_pred_hyde_rerank.json",
        chat_model,
        MU_INSTRUCTION
    )