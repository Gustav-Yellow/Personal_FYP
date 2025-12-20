import json
import os
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_DIR = os.path.join(BASE_DIR, 'split_data')
PRED_DIR = os.path.join(BASE_DIR, 'prediction_results')


def evaluate_file(pred_source, pred_result, task_name):
    # 真实标签在 split_data 文件夹
    gold_path = os.path.join(SPLIT_DIR, pred_source)
    # 预测结果在 prediction_results 文件夹
    pred_path = os.path.join(PRED_DIR, pred_result)

    print(f"\n📊 评估任务: {task_name}")
    print(f"   基准文件: split_data/{pred_source}")
    print(f"   预测文件: prediction_results/{pred_result}")

    if not os.path.exists(pred_path):
        print("   ⚠️ 预测文件不存在")
        return

    with open(gold_path, 'r', encoding='utf-8') as f:
        golds = json.load(f)
    with open(pred_path, 'r', encoding='utf-8') as f:
        preds = json.load(f)

    # 构建 ID 映射以防顺序错乱
    gold_map = {item['id']: item['answer'] for item in golds}

    y_true = []
    y_pred = []

    for item in preds:
        pid = item['id']
        if pid in gold_map:
            y_true.append(gold_map[pid])
            y_pred.append(item.get('answer', 'C')) # 默认C

    acc = accuracy_score(y_true, y_pred)
    print(f"   ✅ Accuracy: {acc:.2%}")
    print(classification_report(y_true, y_pred, labels=['A','B','C','D'], zero_division=0))

if __name__ == "__main__":
    evaluate_file("dev_CRMUS_CR_internal_test.json", "internal_test_CRMUS_CR_pred.json", "CR 常识推理")
    evaluate_file("dev_CRMUS_MU_internal_test.json", "internal_test_CRMUS_MU_pred.json", "MU 寓意理解")