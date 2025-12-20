import sys
import os
import json
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ================= 路径配置 (自动定位) =================
current_script_path = os.path.abspath(__file__)
utils_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(utils_dir)

print(f"当前项目根目录: {project_root}")

# ================= 参数配置 =================

# 1. 模型路径 (指向你本地的 BGE 模型)
# 请确保这个路径和你之前成功运行时的路径一致
MODEL_PATH = os.path.join(
    project_root,
    "models", "bge-large-zh-v1.5", "AI-ModelScope", "bge-large-zh-v1___5"
)

# 2. 输入输出文件
INPUT_FILE = os.path.join(project_root, "processed_data", "commonsense_knowledge_base.json")
INDEX_FILE = os.path.join(project_root, "processed_data", "commonsense.index")
META_FILE = os.path.join(project_root, "processed_data", "commonsense_meta.json")

def main():
    print("【阶段二：构建增强型常识向量库】")

    # 1. 检查输入文件
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 错误：找不到输入文件 {INPUT_FILE}")
        print("请先运行 extract_commonsense.py 生成数据！")
        return

    # 2. 加载本地 Embedding 模型
    print(f"正在加载本地 Embedding 模型...\n路径: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print("❌ 错误：找不到模型路径，请检查 MODEL_PATH 设置！")
        return

    try:
        # device='cuda' 自动使用显卡，如果没有显卡会自动切到 cpu
        model = SentenceTransformer(MODEL_PATH, device='cuda')
        print("✅ 模型加载成功！")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("提示：如果显存不足，尝试将 device='cuda' 改为 device='cpu'")
        return

    # 3. 读取并预处理数据
    print(f"读取数据文件: {INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"原始数据量: {len(data)} 条，准备清洗与向量化...")

    texts_to_encode = [] # 待向量化的纯文本列表
    valid_data = []      # 对应的完整元数据列表

    for item in data:
        # 获取核心知识内容
        knowledge_text = item.get('knowledge', '').strip()

        # 简单清洗：移除多余空白符
        if knowledge_text:
            # 向量化时我们只关心常识内容本身
            # (task_type 和 category 是用来过滤的，不需要变成向量的一部分)
            clean_text = knowledge_text.replace("\n", " ")
            texts_to_encode.append(clean_text)

            # 【关键】保存完整的元数据，供推理时使用
            valid_data.append({
                "id": len(valid_data),          # 重置索引 ID
                "original_id": item.get('id'),  # 保留原始 ID
                "task_type": item.get('task_type', 'UNKNOWN'), # CR 或 MU
                "category": item.get('category', '通用'),      # 具体分类
                "knowledge": knowledge_text,    # 知识文本
                # 可选：如果你觉得 source_preview 太占空间可以注释掉下面这行
                "source_preview": item.get('source_preview', '')
            })

    if not texts_to_encode:
        print("❌ 错误：没有找到有效的常识文本数据。")
        return

    # 4. 批量计算向量
    print(f"正在计算 {len(texts_to_encode)} 条向量 (Batch Processing)...")
    # show_progress_bar=True 会显示进度条
    embeddings = model.encode(texts_to_encode, normalize_embeddings=True, show_progress_bar=True)

    # 5. 构建 FAISS 索引
    vectors = np.array(embeddings).astype('float32')
    dimension = vectors.shape[1]
    print(f"向量计算完成。维度: {dimension}")

    print("正在构建并保存索引...")
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)

    # 保存 .index 文件
    faiss.write_index(index, INDEX_FILE)

    # 保存 _meta.json 文件
    with open(META_FILE, 'w', encoding='utf-8') as f:
        json.dump(valid_data, f, ensure_ascii=False, indent=2)

    print("\n🎉 构建成功！")
    print(f"索引文件 (.index): {INDEX_FILE}")
    print(f"元数据文件 (.json): {META_FILE}")
    print("元数据结构示例 (第一条):")
    if valid_data:
        print(json.dumps(valid_data[0], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()