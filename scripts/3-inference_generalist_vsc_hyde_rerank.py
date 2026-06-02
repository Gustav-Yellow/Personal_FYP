import json
import os
import random
import re
import sys
import numpy as np
import faiss
import torch
from collections import Counter
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from llamafactory.chat import ChatModel

# ================= 配置区域 =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PREDICTION_DIR = os.path.join(BASE_DIR, 'prediction_results')
os.makedirs(PREDICTION_DIR, exist_ok=True)

MODEL_NAME_OR_PATH = "Qwen/Qwen1.5-7B-Chat"
ADAPTER_NAME_OR_PATH = os.path.join(BASE_DIR, "LLaMA-Factory", "output", "qwen_crmus_mixed_lora")

COT_INDEX_PATH = os.path.join(BASE_DIR, "processed_data", "cot_knowledge_base_full.index")
COT_META_PATH = os.path.join(BASE_DIR, "processed_data", "cot_knowledge_base_full_meta.json")
COMMONSENSE_INDEX_PATH = os.path.join(BASE_DIR, "processed_data", "commonsense.index")
COMMONSENSE_META_PATH = os.path.join(BASE_DIR, "processed_data", "commonsense_meta.json")

EMBEDDING_MODEL_PATH = os.path.join(BASE_DIR, "models", "bge-large-zh-v1.5", "AI-ModelScope", "bge-large-zh-v1___5")
RERANKER_MODEL_PATH = os.path.join(BASE_DIR, "models", "Xorbits", "bge-reranker-large")

CR_INSTRUCTION = "请仔细阅读故事，先进行常识推理，解释选项的合理性，最后给出答案。"
MU_INSTRUCTION = "请仔细阅读故事，先进行寓意分析，解释选项的合理性，最后给出答案。"

# 超参数设置
NUM_CANDIDATES = 5  # 阶段二：生成候选推理的数量

# ================= 资源加载 =================
print("1. 正在加载 Embedding 模型...")
embed_model = SentenceTransformer(EMBEDDING_MODEL_PATH, device='cpu')

print("2. 正在加载 Reranker 模型...")
rerank_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_PATH)
rerank_model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_PATH)
rerank_model.eval()
if torch.cuda.is_available(): rerank_model.to('cuda')

print("3. 正在加载 FAISS 向量库...")
cot_index = faiss.read_index(COT_INDEX_PATH)
with open(COT_META_PATH, 'r', encoding='utf-8') as f: cot_meta = json.load(f)
cs_index = faiss.read_index(COMMONSENSE_INDEX_PATH)
with open(COMMONSENSE_META_PATH, 'r', encoding='utf-8') as f: cs_meta = json.load(f)

# ================= 模块一：检索端 (Retriever) =================

def generate_hyde_doc(item, chat_model):
    prompt = f"""你是一个阅读理解专家。请简要提取解答以下问题所需的底层核心规律（物理常识或道德哲理）。
要求：限100字以内。直接输出核心规律，绝对不要包含任何如“好的”、“根据分析”等前言后语。

故事：{item['story']}
问题：{item['question']}
核心规律："""
    try:
        responses = chat_model.chat([{"role": "user", "content": prompt}], max_new_tokens=256, temperature=0.5)
        return responses[0].response_text.strip()
    except Exception:
        return f"{item['story'][:100]} {item['question']}"

def get_rerank_score(query, document):
    pairs = [[query, document]]
    with torch.no_grad():
        inputs = rerank_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        if torch.cuda.is_available(): inputs = {k: v.to('cuda') for k, v in inputs.items()}
        scores = rerank_model(**inputs, return_dict=True).logits.view(-1, ).float()
    return scores.item()

def retrieve_rag_content_hybrid(embed_query, rerank_query, task_type):
    query_vec = embed_model.encode([embed_query], normalize_embeddings=True).astype('float32')
    rag_info = ""

    # 常识检索
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
        rag_info += "【相关背景知识】\n" + "".join([f"{i}. {cs}\n" for i, cs in enumerate(final_cs, 1)])

    # CoT检索
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
            rag_info += f"\n【相似题目解析参考】\n[Rerank Score: {cot_reranked[0][0]:.2f}] {cot_reranked[0][1]}\n"

    return rag_info

def extract_answer(response_text):
    match = re.search(r"(?:【答案】|答案)[:：]?\s*([A-D])", response_text)
    if match: return match.group(1)
    match = re.findall(r"([A-D])", response_text)
    if match: return match[-1]
    return "C"

# ================= 模块二：生成与反思端 (Generator & Critic) =================

def verify_candidate(candidate_text, story, question, rag_context, chat_model):
    """阶段三：CoVe 逻辑质检裁判"""
    critic_prompt = f"""你是一个严厉的逻辑审查员。请阅读以下素材并对【候选推理】进行独立质检。

【事实参考】
{rag_context}

【故事与问题】
{story}
{question}

【候选推理】
{candidate_text}

任务：请检查【候选推理】是否存在以下致命错误：
1. 违背了【事实参考】中的常识或规律。
2. 歪曲了故事原文。
3. 推理过程与最终选出的答案矛盾。

如果你认为推理逻辑严密、无事实错误，请回复 "PASS"。
如果你发现了任何一点错误或幻觉，请回复 "REJECT"。
请严格只回复 PASS 或 REJECT 这一个词，不要做任何解释！
"""
    try:
        # 裁判必须冷静，temperature 设为极低
        responses = chat_model.chat([{"role": "user", "content": critic_prompt}], max_new_tokens=10, temperature=0.1)
        judgment = responses[0].response_text.strip().upper()
        return "PASS" in judgment
    except Exception:
        # 如果裁判出错，保守起见放行
        return True

def process_dataset(task_type, input_filename, output_filename, chat_model, instruction):
    input_path = os.path.join(DATA_DIR, input_filename)
    output_path = os.path.join(PREDICTION_DIR, output_filename)

    with open(input_path, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
        random.seed(42)
        sample_size = min(50, len(full_data))
        data = random.sample(full_data, sample_size)

    results = []

    # 增加总体进度条
    for item in tqdm(data, desc=f"【VSC流水线】处理 {task_type}"):
        model_input_text = f"故事：\n{item['story']}\n\n问题：\n{item['question']}\n\n选项：\n" + "\n".join(item['options'])

        # 阶段一：检索
        hyde_doc = generate_hyde_doc(item, chat_model)
        embed_query = f"问题：{item['question']}。底层逻辑：{hyde_doc}"
        rerank_query = f"{item['story'][:150]}... {item['question']}"
        rag_context = retrieve_rag_content_hybrid(embed_query, rerank_query, task_type)

        final_prompt = f"""你是一个逻辑推理专家。请参考以下资料回答问题。

{rag_context}
----------------
【当前任务】
{instruction}

{model_input_text}

请基于当前故事进行独立分析，最后给出答案。
"""

        # 阶段二：发散生成 (Actor)
        candidates = []
        for i in range(NUM_CANDIDATES):
            try:
                # temperature=0.7 鼓励发散思维
                responses = chat_model.chat([{"role": "user", "content": final_prompt}], max_new_tokens=512, temperature=0.7)
                response_text = responses[0].response_text
                ans = extract_answer(response_text)
                candidates.append({"text": response_text, "ans": ans})
            except Exception:
                continue

        # 阶段三：逻辑质检 (Critic)
        survivors = []
        for idx, cand in enumerate(candidates):
            is_valid = verify_candidate(cand["text"], item['story'], item['question'], rag_context, chat_model)
            if is_valid:
                survivors.append(cand)

        # 阶段四：多数表决 (Majority Voting)
        if len(survivors) > 0:
            # 在合法幸存者中投票
            votes = [cand["ans"] for cand in survivors]
            final_ans = Counter(votes).most_common(1)[0][0]
            # 选一个代表性的 reasoning 作为最终记录
            best_reasoning = [c["text"] for c in survivors if c["ans"] == final_ans][0]
            vote_status = f"Filtered: {len(survivors)}/{NUM_CANDIDATES} survived. Votes: {votes}"
        else:
            # 兜底：如果被全军覆没，退回到在原始 5 个中盲选
            votes = [cand["ans"] for cand in candidates]
            final_ans = Counter(votes).most_common(1)[0][0] if votes else "C"
            best_reasoning = candidates[0]["text"] if candidates else "Error"
            vote_status = f"ALL REJECTED! Fallback to raw votes: {votes}"

        # 打印一下当前的投票结果，让你在跑脚本时不会觉得无聊
        print(f"\n[题目 {item['id']}] {vote_status} -> 最终预测: {final_ans}")

        result_item = item.copy()
        result_item['pred_answer'] = final_ans
        result_item['hyde_doc'] = hyde_doc
        result_item['model_reasoning'] = best_reasoning
        result_item['vsc_debug_info'] = vote_status # 把投票详情存进JSON，写论文时绝对用得上！
        result_item['rag_debug_info'] = rag_context
        results.append(result_item)

    # 保存
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ VSC (HyDE+Rerank) 结果已保存至: {output_path}")

if __name__ == "__main__":
    print("4. 正在加载 Qwen LLM...")
    chat_model = ChatModel({
        "model_name_or_path": MODEL_NAME_OR_PATH,
        "adapter_name_or_path": ADAPTER_NAME_OR_PATH,
        "template": "qwen",
        "finetuning_type": "lora",
        "quantization_bit": 4
    })

    print("\n=== 开始推理 (Verified Self-Consistency 终极形态) ===")

    process_dataset(
        "CR",
        "test_CRMUS_CR_manual.json",
        "test_CRMUS_CR_pred_generalist_vsc.json",
        chat_model,
        CR_INSTRUCTION
    )

    process_dataset(
        "MU",
        "test_CRMUS_MU_manual.json",
        "test_CRMUS_MU_pred_generalist_vsc.json",
        chat_model,
        MU_INSTRUCTION
    )