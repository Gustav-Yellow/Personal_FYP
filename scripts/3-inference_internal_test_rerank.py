import json
import os
import re
import sys
import random
import numpy as np
import faiss
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from llamafactory.chat import ChatModel

"""
留出法：额外添加数据库向量索引，也就是 RAG
在留出法训练的mixed本地模型上运行，对通过 0-split_data.py 分割过后的文件；
先通过使用 1-process_data_internal_test.py 对用于训练的 80% 的数据集进行 CoT 数据蒸馏，用于模型微调；
然后再用 剩下的 20% 的测试集中各抽取 50 道 CR 和 MU 问题，然后进行回答。
输入文件：
split_data/dev_CRMUS_CR_internal_test.json
split_data/dev_CRMUS_MU_internal_test.json

输出文件：
prediction_results/internal_test_CRMUS_CR_pred_rerank.json
prediction_results/internal_test_CRMUS_MU_pred_rerank.json
"""
# ================= 配置区域 =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_DIR = os.path.join(BASE_DIR, 'split_data') # 数据源改为 split_data
PREDICTION_DIR = os.path.join(BASE_DIR, 'prediction_results')
os.makedirs(PREDICTION_DIR, exist_ok=True)

MODEL_NAME_OR_PATH = "Qwen/Qwen1.5-7B-Chat"
# 加载专门为 Internal Test 训练的 LoRA 权重
ADAPTER_NAME_OR_PATH = os.path.join(BASE_DIR, "LLaMA-Factory", "output", "qwen_crmus_mixed_lora_internal_test")

# 加载使用 80% 内部训练集生成的 CoT 知识库
COT_INDEX_PATH = os.path.join(BASE_DIR, "processed_data", "cot_knowledge_base_internal.index")
COT_META_PATH = os.path.join(BASE_DIR, "processed_data", "cot_knowledge_base_internal_meta.json")

# 常识知识库 (全局通用)
COMMONSENSE_INDEX_PATH = os.path.join(BASE_DIR, "processed_data", "commonsense.index")
COMMONSENSE_META_PATH = os.path.join(BASE_DIR, "processed_data", "commonsense_meta.json")

# 向量模型路径
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
if not os.path.exists(EMBEDDING_MODEL_PATH):
    raise FileNotFoundError(f"❌ 找不到 Embedding 模型，请确保文件存在于: {EMBEDDING_MODEL_PATH}")
embed_model = SentenceTransformer(EMBEDDING_MODEL_PATH, device='cpu')

print("2. 正在加载 Reranker 模型...")
if not os.path.exists(RERANKER_MODEL_PATH):
    raise FileNotFoundError(f"❌ 找不到 Reranker 模型，请确保文件存在于: {RERANKER_MODEL_PATH}")
rerank_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_PATH)
rerank_model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_PATH)
rerank_model.eval() # 切换到推理模式
if torch.cuda.is_available():
    rerank_model.to('cuda')

print("3. 正在加载 FAISS 向量知识库...")
cot_index = faiss.read_index(COT_INDEX_PATH)
with open(COT_META_PATH, 'r', encoding='utf-8') as f:
    cot_meta = json.load(f)

cs_index = faiss.read_index(COMMONSENSE_INDEX_PATH)
with open(COMMONSENSE_META_PATH, 'r', encoding='utf-8') as f:
    cs_meta = json.load(f)


# ================= 功能函数 =================
def get_rerank_score(query, document):
    """计算 Query 和 Document 的交叉注意力相关性得分"""
    pairs = [[query, document]]
    with torch.no_grad():
        inputs = rerank_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        scores = rerank_model(**inputs, return_dict=True).logits.view(-1, ).float()
    return scores.item()

def retrieve_rag_content_with_rerank(query_text, task_type):
    """先粗排 (k=10) 再精排 (Reranker) 的检索机制"""
    query_vec = embed_model.encode([query_text], normalize_embeddings=True).astype('float32')
    rag_info = ""

    # ================= 1. 常识库检索 =================
    D, I = cs_index.search(query_vec, k=10) # 粗排一次捞 10 条
    candidates = []

    for i, idx in enumerate(I[0]):
        if idx == -1: continue
        meta = cs_meta[idx]

        # 跨领域过滤
        if task_type == "CR" and meta['task_type'] != "CR": continue
        if task_type == "MU" and meta['task_type'] != "MU": continue

        candidates.append(f"[{meta['category']}] {meta['knowledge']}")

    # 针对候选项进行 Rerank 精排
    reranked_results = [(get_rerank_score(query_text, doc), doc) for doc in candidates]
    reranked_results.sort(key=lambda x: x[0], reverse=True)

    # 过滤掉得分 <= 0 的负向匹配，最多保留 Top-3
    final_cs = [doc for score, doc in reranked_results if score > 0][:3]

    if final_cs:
        rag_info += "【相关背景知识】\n"
        for i, cs in enumerate(final_cs, 1):
            rag_info += f"{i}. {cs}\n"

    # ================= 2. CoT库检索 (仅MU任务) =================
    if task_type == "MU":
        D, I = cot_index.search(query_vec, k=5) # 粗排捞 5 条
        cot_candidates = []

        for i, idx in enumerate(I[0]):
            if idx == -1: continue
            raw_output = cot_meta[idx]['output']
            clean_reasoning = raw_output.split("【答案】")[0].split("答案：")[0].strip()
            # 拼装包含原始故事上下文和纯净分析的字符串，供 Reranker 打分
            cot_candidates.append(f"相似故事：{cot_meta[idx]['input'][:50]}...\n参考分析逻辑：{clean_reasoning}")

        cot_reranked = [(get_rerank_score(query_text, doc), doc) for doc in cot_candidates]
        cot_reranked.sort(key=lambda x: x[0], reverse=True)

        # 只选取相关性得分最高且为正的一条
        if cot_reranked and cot_reranked[0][0] > 0:
            best_score = cot_reranked[0][0]
            best_doc = cot_reranked[0][1]
            rag_info += f"\n【相似题目解析参考】\n[Rerank Score: {best_score:.2f}] {best_doc}\n"

    return rag_info

def extract_answer(response_text):
    """从大模型生成的回复中精准提取 A/B/C/D"""
    match = re.search(r"(?:【答案】|答案)[:：]?\s*([A-D])", response_text)
    if match: return match.group(1)
    match = re.findall(r"([A-D])", response_text)
    if match: return match[-1]
    return "C"

def process_dataset_internal(task_type, input_filename, output_filename, chat_model, instruction):
    """处理 Internal Test 专属的推理逻辑"""
    input_path = os.path.join(SPLIT_DIR, input_filename)
    output_path = os.path.join(PREDICTION_DIR, output_filename)

    if not os.path.exists(input_path):
        print(f"⚠️ 跳过: {input_path} (文件不存在)")
        return

    print(f"\n🚀 正在处理: {input_filename} ...")
    with open(input_path, 'r', encoding='utf-8') as f:
        full_data = json.load(f)

        # 固定随机种子，确保每次抽取的 50 道题都是一样的
        random.seed(42)

        sample_size = min(50, len(full_data))
        data = random.sample(full_data, sample_size)

    print(f"   共发现 {len(data)} 条测试数据。")

    results = []
    for item in tqdm(data, desc=f"推断 {task_type}"):
        model_input_text = f"故事：\n{item['story']}\n\n问题：\n{item['question']}\n\n选项：\n" + "\n".join(item['options'])

        # 截断故事前 150 字避免噪声，用于检索
        search_query = f"{item['story'][:150]}... {item['question']}"
        rag_context = retrieve_rag_content_with_rerank(search_query, task_type)

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
            print(f"推理错误: {e}")
            response_text = "Error"

        result_item = item.copy()
        result_item['pred_answer'] = extract_answer(response_text)
        result_item['model_reasoning'] = response_text
        result_item['rag_debug_info'] = rag_context
        results.append(result_item)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ Rerank 结果已保存至: {output_path}")


# ================= 主程序入口 =================
if __name__ == "__main__":
    print("4. 正在加载 Qwen 7B 模型 (挂载 Internal LoRA) ...")
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
    except Exception as e:
        print(f"模型加载失败，请检查显存或路径: {e}")
        sys.exit(1)

    print("\n=== 开始推理 (Internal Test Rerank 模式) ===")

    # 1. 运行常识推理 (CR) 内部测试集
    process_dataset_internal(
        "CR",
        "dev_CRMUS_CR_internal_test.json",
        "internal_test_CRMUS_CR_pred_rerank.json",
        chat_model,
        CR_INSTRUCTION
    )

    # 2. 运行寓意理解 (MU) 内部测试集
    process_dataset_internal(
        "MU",
        "dev_CRMUS_MU_internal_test.json",
        "internal_test_CRMUS_MU_pred_rerank.json",
        chat_model,
        MU_INSTRUCTION
    )

    print("\n🎉 全部推理完成！你可以使用 4-eval_internal_test_rerank.py 查看最终准确率了。")