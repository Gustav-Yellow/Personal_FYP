import json
import os
from transformers import AutoConfig

# 把你命令行里写的那个路径填到这里
model_path = r"D:\Applications\BaiduNetdisk\BaiduSyncdisk\FYP\Codes\My_FYP\models\bge-large-zh-v1.5"
config_file = os.path.join(model_path, "config.json")

print(f"正在检查文件: {config_file}")

# 1. 检查文件是否存在
if not os.path.exists(config_file):
    print("❌ 错误：找不到 config.json 文件！请检查路径。")
else:
    # 2. 打印文件内容的前几行
    print("✅ 文件存在。内容前 5 行如下：")
    with open(config_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[:5]:
            print(line.strip())

    # 3. 尝试用 Transformers 加载
    print("\n正在尝试加载配置...")
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        print(f"✅ 加载成功！识别到的 model_type 是: {config.model_type}")
    except Exception as e:
        print(f"❌ 加载失败！报错信息:\n{e}")