import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ================= 配置区域 =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PRED_DIR = os.path.join(BASE_DIR, 'prediction_results')
REPORT_DIR = os.path.join(BASE_DIR, 'evaluation_reports')
os.makedirs(REPORT_DIR, exist_ok=True)

def plot_confusion_matrix(y_true, y_pred, title, filename):
    labels = ['A', 'B', 'C', 'D']
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels) # 换成蓝色主题区分原版
    plt.title(title)
    plt.xlabel('Local Model (Rerank)')
    plt.ylabel('Teacher Model (Silver Std)')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def evaluate_silver_agreement(task_name, silver_file, pred_file):
    silver_path = os.path.join(DATA_DIR, silver_file)
    pred_path = os.path.join(PRED_DIR, pred_file)

    print(f"\n📊 评估: {task_name}")
    print(f"   银标准 (Teacher): {silver_file}")
    print(f"   预测值 (Student Rerank): {pred_file}")

    if not os.path.exists(pred_path):
        print(f"   ❌ 预测文件 {pred_path} 不存在，请确保已经运行过 3-inference_generalist_rerank.py")
        return

    with open(silver_path, 'r', encoding='utf-8') as f:
        silvers = json.load(f)
    with open(pred_path, 'r', encoding='utf-8') as f:
        preds = json.load(f)

    silver_map = {item['id']: item['answer'].strip().upper() for item in silvers}

    y_silver = []
    y_local = []

    for item in preds:
        pid = item['id']
        if pid in silver_map:
            teacher_ans = silver_map[pid]
            # 提取大模型预测答案，兜底为 C
            local_ans = item.get('pred_answer', 'C').strip().upper()

            y_silver.append(teacher_ans)
            y_local.append(local_ans)

    # 计算准确率 (一致率)
    acc = accuracy_score(y_silver, y_local)
    print(f"   ✅ 相对准确率 (Agreement Rate): {acc:.2%}")

    report = classification_report(y_silver, y_local, labels=['A','B','C','D'], zero_division=0)
    print(report)

    # 保存文本报告
    with open(os.path.join(REPORT_DIR, f'{task_name}_silver_report.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Task: {task_name} (Rerank vs Silver Standard Evaluation)\n")
        f.write(f"Agreement Rate: {acc:.4f}\n\n")
        f.write(report)

    # 保存混淆矩阵图片
    plot_path = os.path.join(REPORT_DIR, f'{task_name}_matrix.png')
    plot_confusion_matrix(y_silver, y_local, f'{task_name} Agreement Matrix', plot_path)
    print(f"   -> 报告和图表已保存至 {REPORT_DIR}")


if __name__ == "__main__":
    print("=== 开始评估 Generalist (Rerank 模式) ===")

    # 评估 CR 任务
    evaluate_silver_agreement(
        "CR_Generalist_Rerank",
        "test_CRMUS_CR_manual.json",
        "test_CRMUS_CR_pred_generalist_rerank.json"
    )

    # 评估 MU 任务
    evaluate_silver_agreement(
        "MU_Generalist_Rerank",
        "test_CRMUS_MU_manual.json",
        "test_CRMUS_MU_pred_generalist_rerank.json"
    )