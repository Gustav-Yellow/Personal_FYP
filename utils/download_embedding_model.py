from modelscope.hub.snapshot_download import snapshot_download
import os

# 下载 Embedding 向量模型
# 指定下载目录为 models/bge-large-zh-v1.5
save_dir = os.path.join(os.getcwd(), "models", "bge-large-zh-v1.5")

print(f"开始下载模型到: {save_dir} ...")

# 调用 modelscope 下载 BAAI/bge-large-zh-v1.5
model_dir = snapshot_download(
    'AI-ModelScope/bge-large-zh-v1.5',
    cache_dir=save_dir,
    revision='master'
)

print("下载完成！")