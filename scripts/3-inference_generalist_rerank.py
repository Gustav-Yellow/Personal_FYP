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

"""
全量法：额外添加 RAG 向量查询和 ReRanker 重排机制
在全量训练的mixed本地模型上，对在0-generate_silver_label.py中收取的问题进行回答，并且比较与商用大模型答案之间的准确率
输入文件：
split_data/test_CRMUS_CR_manual.json
split_data/test_CRMUS_MU_manual.json

输出文件：
prediction_results/test_CRMUS_CR_pred_generalist_rerank.json
prediction_results/test_CRMUS_MU_pred_generalist_rerank.json
"""
# ================= 配置区域 =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_DIR = os.path.join(BASE_DIR, 'split_data')
DATA_DIR = os.path.join(BASE_DIR, 'data')
PREDICTION_DIR = os.path.join(BASE_DIR, 'prediction_results')
os.makedirs(PREDICTION_DIR, exist_ok=True)

MODEL_NAME_OR_PATH = "Qwen/Qwen1.5-7B-Chat"
# 全量法 LoRA 权重
ADAPTER_NAME_OR_PATH = os.path.join(BASE_DIR, "LLaMA-Factory", "output", "qwen_crmus_mixed_lora")

# RAG 知识库路径
COT_INDEX_PATH = os.path.join(BASE_DIR, "processed_data", "cot_knowledge_base_full.index")
COT_META_PATH = os.path.join(BASE_DIR, "processed_data", "cot_knowledge_base_full_meta.json")
COMMONSENSE_INDEX_PATH = os.path.join(BASE_DIR, "processed_data", "commonsense.index")
COMMONSENSE_META_PATH = os.path.join(BASE_DIR, "processed_data", "commonsense_meta.json")

# 向量模型与重排模型路径
# EMBEDDING_MODEL_PATH = os.path.join(BASE_DIR, "models", "bge-large-zh-v1.5", "AI-ModelScope", "bge-large-zh-v1___5")
base_embed_dir = os.path.join(BASE_DIR, "models", "bge-large-zh-v1.5", "AI-ModelScope")
path_option_1 = os.path.join(base_embed_dir, "bge-large-zh-v1___5")  # 下划线版本
path_option_2 = os.path.join(base_embed_dir, "bge-large-zh-v1.5")    # 带点版本

if os.path.exists(path_option_1):
    EMBEDDING_MODEL_PATH = path_option_1
elif os.path.exists(path_option_2):
    EMBEDDING_MODEL_PATH = path_option_2
else:
    # 打印出当前目录下的真实内容，帮你诊断
    actual_files = os.listdir(base_embed_dir) if os.path.exists(base_embed_dir) else "AI-ModelScope 目录不存在!"
    raise FileNotFoundError(f"❌ 找不到 Embedding 模型！\n期望的路径: {path_option_1} 或 {path_option_2}\n目前 {base_embed_dir} 目录下的实际内容是: {actual_files}")

RERANKER_MODEL_PATH = os.path.join(BASE_DIR, "models", "Xorbits", "bge-reranker-large")

CR_INSTRUCTION = "请仔细阅读故事，先进行常识推理，解释选项的合理性，最后给出答案。"
MU_INSTRUCTION = "请仔细阅读故事，先进行寓意分析，解释选项的合理性，最后给出答案。"

# ================= 资源加载 =================
print("1. 正在加载 Embedding 模型...")
embed_model = SentenceTransformer(EMBEDDING_MODEL_PATH, device='cpu')

print("2. 正在加载 Reranker 模型...")
rerank_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_PATH)
rerank_model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_PATH)
rerank_model.eval()
if torch.cuda.is_available():
    rerank_model.to('cuda')

print("3. 正在加载向量库...")
cot_index = faiss.read_index(COT_INDEX_PATH)
with open(COT_META_PATH, 'r', encoding='utf-8') as f: cot_meta = json.load(f)
cs_index = faiss.read_index(COMMONSENSE_INDEX_PATH)
with open(COMMONSENSE_META_PATH, 'r', encoding='utf-8') as f: cs_meta = json.load(f)

# ================= 功能函数 =================
def get_rerank_score(query, document):
    """使用 Reranker 模型计算 Query 和 Document 的相关性得分"""
    pairs = [[query, document]]
    with torch.no_grad():
        inputs = rerank_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        if torch.cuda.is_available(): inputs = {k: v.to('cuda') for k, v in inputs.items()}
        scores = rerank_model(**inputs, return_dict=True).logits.view(-1, ).float()
    return scores.item()

def retrieve_rag_content_with_rerank(query_text, task_type):
    query_vec = embed_model.encode([query_text], normalize_embeddings=True).astype('float32')
    rag_info = ""

    # 1. 检索常识库 (CR/MU 都会用到)
    D, I = cs_index.search(query_vec, k=10) # 粗排放大 k 值
    candidates = []
    for i, idx in enumerate(I[0]):
        if idx == -1: continue
        meta = cs_meta[idx]
        if task_type == "CR" and meta['task_type'] != "CR": continue
        if task_type == "MU" and meta['task_type'] != "MU": continue
        candidates.append(f"[{meta['category']}] {meta['knowledge']}")

    # 精排
    reranked_results = [(get_rerank_score(query_text, doc), doc) for doc in candidates]
    reranked_results.sort(key=lambda x: x[0], reverse=True)
    # 取得分大于 0 的前 3 条
    final_cs = [doc for score, doc in reranked_results if score > 0][:3]

    if final_cs:
        rag_info += "【相关背景知识】\n"
        for i, cs in enumerate(final_cs, 1):
            rag_info += f"{i}. {cs}\n"

    # 2. 检索 CoT 库 (仅限 MU)
    if task_type == "MU":
        D, I = cot_index.search(query_vec, k=5)
        cot_candidates = []
        for i, idx in enumerate(I[0]):
            if idx == -1: continue
            raw_output = cot_meta[idx]['output']
            clean_reasoning = raw_output.split("【答案】")[0].split("答案：")[0].strip()
            cot_candidates.append(f"相似故事：{cot_meta[idx]['input'][:50]}...\n参考分析逻辑：{clean_reasoning}")

        cot_reranked = [(get_rerank_score(query_text, doc), doc) for doc in cot_candidates]
        cot_reranked.sort(key=lambda x: x[0], reverse=True)
        if cot_reranked and cot_reranked[0][0] > 0:
            rag_info += f"\n【相似题目解析参考】\n[Rerank Score: {cot_reranked[0][0]:.2f}] {cot_reranked[0][1]}\n"

    return rag_info

def extract_answer(response_text):
    match = re.search(r"(?:【答案】|答案)[:：]?\s*([A-D])", response_text)
    if match: return match.group(1)
    match = re.findall(r"([A-D])", response_text)
    if match: return match[-1]
    return "C"

def process_dataset(task_type, input_filename, output_filename, chat_model, instruction):
    input_path = os.path.join(DATA_DIR, input_filename)
    output_path = os.path.join(PREDICTION_DIR, output_filename)

    with open(input_path, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
        sample_size = min(50, len(full_data))
        data = random.sample(full_data, sample_size)

    results = []
    for item in tqdm(data):
        model_input_text = f"故事：\n{item['story']}\n\n问题：\n{item['question']}\n\n选项：\n" + "\n".join(item['options'])
        search_query = f"{item['story'][:150]}... {item['question']}"
        rag_context = retrieve_rag_content_with_rerank(search_query, task_type)

        final_prompt = f"你是一个逻辑推理专家。请参考以下资料回答问题。\n\n{rag_context}\n----------------\n【当前任务】\n{instruction}\n\n{model_input_text}\n请基于当前故事进行独立分析，最后给出答案。"

        try:
            responses = chat_model.chat([{"role": "user", "content": final_prompt}], max_new_tokens=512, temperature=0.3)
            response_text = responses[0].response_text
        except: response_text = "Error"

        result_item = item.copy()
        result_item['pred_answer'] = extract_answer(response_text)
        result_item['model_reasoning'] = response_text
        result_item['rag_debug_info'] = rag_context
        results.append(result_item)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ Rerank 结果已保存至: {output_path}")

if __name__ == "__main__":
    print("4. 正在加载 Qwen LLM...")
    chat_model = ChatModel({"model_name_or_path": MODEL_NAME_OR_PATH, "adapter_name_or_path": ADAPTER_NAME_OR_PATH, "template": "qwen", "finetuning_type": "lora", "quantization_bit": 4})

    print("\n=== 开始推理 (Generalist Rerank) ===")
    process_dataset("CR", "test_CRMUS_CR_manual.json", "test_CRMUS_CR_pred_generalist_rerank.json", chat_model, CR_INSTRUCTION)
    process_dataset("MU", "test_CRMUS_MU_manual.json", "test_CRMUS_MU_pred_generalist_rerank.json", chat_model, MU_INSTRUCTION)