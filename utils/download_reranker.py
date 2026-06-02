# My_FYP/utils/download_reranker.py
import os
from modelscope.hub.snapshot_download import snapshot_download

"""
Re-Ranking 方法，需要额外下载一个 Cross-Encoder 模型用来单塔交叉架构，用于精细打分
业界与 bge-large-zh-v1.5 最匹配的是 bge-reranker-large。
"""
# 自动定位项目根目录下的 models 文件夹
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
models_dir = os.path.join(project_root, 'models')
os.makedirs(models_dir, exist_ok=True)

print(f"准备下载 bge-reranker-large 到目录: {models_dir}")

# 使用 ModelScope 下载，速度国内较快。你之前的 BGE embedding 似乎也是从这里下的
model_dir = snapshot_download(
    'Xorbits/bge-reranker-large', # 也可以用 'AI-ModelScope/bge-reranker-large'
    cache_dir=models_dir,
    revision='master'
)

print(f"✅ Reranker 模型下载完成！路径: {model_dir}")
print("⚠️ 请将上述路径复制，更新到 Rerank 推理脚本的 RERANKER_MODEL_PATH 中。")