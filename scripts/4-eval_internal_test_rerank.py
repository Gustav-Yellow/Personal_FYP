import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ================= 配置区域 =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_DIR = os.path.join(BASE_DIR, 'split_data')
PRED_DIR = os.path.join(BASE_DIR, 'prediction_results')
REPORT_DIR = os.path.join(BASE_DIR, 'evaluation_reports')
os.makedirs(REPORT_DIR, exist_ok=True)

def plot_confusion_matrix(y_true, y_pred, title, filename):
    labels = ['A', 'B', 'C', 'D']
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=labels, yticklabels=labels) # 使用橙色主题
    plt.title(title)
    plt.xlabel('Predicted Label (Rerank)')
    plt.ylabel('True Label (Gold Std)')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def evaluate_file(pred_source, pred_result, task_name):
    # 真实标签在 split_data 文件夹
    gold_path = os.path.join(SPLIT_DIR, pred_source)
    # 预测结果在 prediction_results 文件夹
    pred_path = os.path.join(PRED_DIR, pred_result)

    print(f"\n📊 评估任务: {task_name}")
    print(f"   基准文件: {pred_source}")
    print(f"   预测文件: {pred_result}")

    if not os.path.exists(pred_path):
        print(f"   ❌ 预测文件 {pred_path} 不存在，请确保已经运行过 3-inference_internal_test_rerank.py")
        return

    with open(gold_path, 'r', encoding='utf-8') as f:
        golds = json.load(f)
    with open(pred_path, 'r', encoding='utf-8') as f:
        preds = json.load(f)

    # 构建 ID 映射以防顺序错乱
    gold_map = {item['id']: item['answer'].strip().upper() for item in golds}

    y_true = []
    y_pred = []

    for item in preds:
        pid = item['id']
        if pid in gold_map:
            y_true.append(gold_map[pid])
            # 提取大模型预测答案，兜底为 C
            y_pred.append(item.get('pred_answer', 'C').strip().upper())

    acc = accuracy_score(y_true, y_pred)
    print(f"   ✅ Accuracy: {acc:.2%}")

    report = classification_report(y_true, y_pred, labels=['A','B','C','D'], zero_division=0)
    print(report)

    # 保存文本报告
    report_name = f"{task_name.replace(' ', '_')}_Rerank"
    with open(os.path.join(REPORT_DIR, f'{report_name}_report.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Task: {task_name} (Internal Test - Rerank Evaluation)\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)

    # 保存混淆矩阵图片
    plot_path = os.path.join(REPORT_DIR, f'{report_name}_matrix.png')
    plot_confusion_matrix(y_true, y_pred, f'{task_name} Rerank Matrix', plot_path)
    print(f"   -> 报告和图表已保存至 {REPORT_DIR}")

if __name__ == "__main__":
    print("=== 开始评估 Internal Test (Rerank 模式) ===")

    evaluate_file(
        "dev_CRMUS_CR_internal_test.json",
        "internal_test_CRMUS_CR_pred_rerank.json",
        "CR Internal Test"
    )

    evaluate_file(
        "dev_CRMUS_MU_internal_test.json",
        "internal_test_CRMUS_MU_pred_rerank.json",
        "MU Internal Test"
    )