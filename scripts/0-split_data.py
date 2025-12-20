import json
import os
from sklearn.model_selection import train_test_split

"""
留出法
将原本包含正确答案的训练集进行拆分，提取出来80%的内容依然用来训练，将剩下的20%的内容用来当做带有准确答案的测试集。
输入文件：
data/dev_CRMUS_CR.json
data/dev_CRMUS_MU.json

输出文件：
split_data/dev_CRMUS_CR_internal_test.json
split_data/dev_CRMUS_MU_internal_test.json
"""

# === 配置 ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
SPLIT_DIR = os.path.join(BASE_DIR, 'split_data')
os.makedirs(SPLIT_DIR, exist_ok=True)

TEST_SIZE = 0.2
RANDOM_STATE = 42

def split_dataset(original_filename, task_prefix):
    """
    original_filename: 如 'dev_CRMUS_CR.json'
    task_prefix: 如 'dev_CRMUS_CR' (用于构建新文件名)
    """
    filepath = os.path.join(DATA_DIR, original_filename)
    if not os.path.exists(filepath):
        print(f"❌ 找不到文件: {filepath}")
        return

    print(f"🔪 正在划分 {original_filename} ...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 执行划分
    train_data, test_data = train_test_split(
        data,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    # === 命名修改 ===
    # 训练集: split_data/dev_CRMUS_CR_internal_train.json
    train_file = os.path.join(SPLIT_DIR, f"{task_prefix}_internal_train.json")
    # 测试集: split_data/dev_CRMUS_CR_internal_test.json
    test_file = os.path.join(SPLIT_DIR, f"{task_prefix}_internal_test.json")

    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"   - 训练集 ({len(train_data)} 条) -> {os.path.basename(train_file)}")
    print(f"   - 测试集 ({len(test_data)} 条) -> {os.path.basename(test_file)}")

if __name__ == "__main__":
    split_dataset("dev_CRMUS_CR.json", "dev_CRMUS_CR")
    split_dataset("dev_CRMUS_MU.json", "dev_CRMUS_MU")