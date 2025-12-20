import sys
import os
import json
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

"""
将原本使用大模型生成好的针对训练集问题的分析进行总结，尝试将其中原本的思考和分析构建成CoT向量数据库
输入文件：
processed_data/CRMUS_CR_train.json
processed_data/CRMUS_MU_train.json

输出文件：
processed_data/cot_knowledge_base_full.index
processed_data/cot_knowledge_base_internal.index
processed_data/cot_knowledge_base_full_meta.json
processed_data/cot_knowledge_base_internal_meta.json
"""
# ================= 路径配置 =================
current_script_path = os.path.abspath(__file__)
utils_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(utils_dir)

print(f"当前项目根目录: {project_root}")

# ================= 核心配置区域 =================

# 模式选择: 'internal' 或 'full'
# 'internal': 仅使用 internal_train (80%) 构建库 -> 用于内部测试 (Internal Test)
# 'full':     使用全量 train (100%) 构建库     -> 用于生成提交结果 (Public Submission)
MODE = 'full'
# MODE = 'internal'

# 【关键修改】指向本地模型的最深层路径
# 这是为了绕过 API，直接读取权重
MODEL_PATH = os.path.join(
    project_root,
    "models", "bge-large-zh-v1.5", "AI-ModelScope", "bge-large-zh-v1___5"
)

print(f"\n当前构建模式: 【{MODE.upper()}】")

if MODE == 'internal':
    INPUT_FILES = [
        os.path.join(project_root, "processed_data", "CRMUS_CR_internal_train.json"),
        os.path.join(project_root, "processed_data", "CRMUS_MU_internal_train.json")
    ]
    INDEX_FILE = os.path.join(project_root, "processed_data", "cot_knowledge_base_internal.index")
    META_FILE = os.path.join(project_root, "processed_data", "cot_knowledge_base_internal_meta.json")

elif MODE == 'full':
    INPUT_FILES = [
        os.path.join(project_root, "processed_data", "CRMUS_CR_train.json"),
        os.path.join(project_root, "processed_data", "CRMUS_MU_train.json")
    ]
    INDEX_FILE = os.path.join(project_root, "processed_data", "cot_knowledge_base_full.index")
    META_FILE = os.path.join(project_root, "processed_data", "cot_knowledge_base_full_meta.json")
else:
    raise ValueError("MODE 必须是 'internal' 或 'full'")

def main():
    # 1. 检查模型是否存在
    print(f"正在加载本地 Embedding 模型...\n路径: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print("❌ 错误：找不到模型路径，请检查代码中的 MODEL_PATH！")
        return

    try:
        # 加载 BGE 模型 (device='cuda' 自动使用显卡)
        model = SentenceTransformer(MODEL_PATH, device='cuda')
        print("✅ 模型加载成功！")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 2. 加载数据
    all_data = []
    print(f"准备读取数据文件:")
    for file_path in INPUT_FILES:
        if not os.path.exists(file_path):
            print(f"⚠️ 跳过不存在的文件: {file_path}")
            continue
        print(f" - {os.path.basename(file_path)}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                item['_source_file'] = os.path.basename(file_path)
                all_data.append(item)

    print(f"共加载 {len(all_data)} 条数据，准备向量化...")

    # 3. 批量向量化 (Batch Processing)
    # 相比之前一条条调 API，这种方式极快
    texts_to_encode = []
    valid_data_indices = [] # 记录有效数据的索引

    for idx, item in enumerate(all_data):
        # 策略：用 'input' (题目) 做索引 Key
        key_text = item.get('input', '').strip()
        if key_text:
            texts_to_encode.append(key_text.replace("\n", " "))
            valid_data_indices.append(idx)

    if not texts_to_encode:
        print("没有找到有效的 input 文本。")
        return

    print("正在计算向量 (这也是最耗时的一步)...")
    # show_progress_bar=True 会显示进度条
    embeddings = model.encode(texts_to_encode, normalize_embeddings=True, show_progress_bar=True)

    # 4. 组装元数据 (Metadata)
    print("正在组装元数据...")
    metadata = []
    for i, original_idx in enumerate(valid_data_indices):
        item = all_data[original_idx]
        metadata.append({
            "id": i, # 重置 ID 为当前向量库的顺序
            "input": item.get('input'),
            "output": item.get('output'),
            "source": item.get('_source_file')
        })

    # 5. 保存 FAISS
    vectors = np.array(embeddings).astype('float32')
    dimension = vectors.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)

    faiss.write_index(index, INDEX_FILE)

    with open(META_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\n🎉 CoT 知识库构建完成！")
    print(f"模式: {MODE}")
    print(f"索引保存至: {os.path.basename(INDEX_FILE)}")
    print(f"元数据保存至: {os.path.basename(META_FILE)}")


if __name__ == "__main__":
    main()