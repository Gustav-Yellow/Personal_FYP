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
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Local Model (Generalist)')
    plt.ylabel('Teacher Model (Silver Std)')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def evaluate_silver_agreement(task_name, silver_file, pred_file):
    silver_path = os.path.join(DATA_DIR, silver_file)
    pred_path = os.path.join(PRED_DIR, pred_file)

    print(f"\n📊 评估: {task_name}")
    print(f"   银标准 (Teacher): {silver_path}")
    print(f"   预测值 (Student): {pred_path}")

    if not os.path.exists(pred_path):
        print("   ❌ 预测文件不存在，请先运行 3-inference_generalist.py")
        return

    with open(silver_path, 'r', encoding='utf-8') as f:
        silvers = json.load(f)
    with open(pred_path, 'r', encoding='utf-8') as f:
        preds = json.load(f)

    # 构建映射
    # 在 0_generate_silver_labels.py 中，大模型的答案存在 'answer' 字段
    silver_map = {item['id']: item['answer'].strip().upper() for item in silvers}

    y_silver = []
    y_local = []

    for item in preds:
        pid = item['id']
        if pid in silver_map:
            # 银标准答案 (Teacher's Answer)
            teacher_ans = silver_map[pid]

            # 本地模型预测 (Student's Answer)
            # 在 3-inference_generalist.py 中，我把预测结果存为了 'pred_answer'
            local_ans = item.get('pred_answer', 'C').strip().upper()

            y_silver.append(teacher_ans)
            y_local.append(local_ans)

    # 计算指标
    acc = accuracy_score(y_silver, y_local)
    print(f"   ✅ 相对准确率 (Agreement Rate): {acc:.2%}")
    print("   (注：这是本地模型与大模型答案的一致程度)")

    report = classification_report(y_silver, y_local, labels=['A','B','C','D'], zero_division=0)
    print(report)

    # 保存报告
    with open(os.path.join(REPORT_DIR, f'{task_name}_silver_report.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Task: {task_name} (Silver Standard Evaluation)\n")
        f.write(f"Agreement Rate: {acc:.4f}\n\n")
        f.write(report)

    # 绘制混淆矩阵
    plot_path = os.path.join(REPORT_DIR, f'{task_name}_silver_matrix.png')
    plot_confusion_matrix(y_silver, y_local, f'{task_name} Agreement Matrix', plot_path)
    print(f"   -> 图表已保存至 {REPORT_DIR}")

if __name__ == "__main__":
    # 评估 CR
    evaluate_silver_agreement(
        "CR_Generalist_vs_Silver",
        "test_CRMUS_CR_manual.json",
        "test_CRMUS_CR_pred_generalist.json"
    )

    # 评估 MU
    evaluate_silver_agreement(
        "MU_Generalist_vs_Silver",
        "test_CRMUS_MU_manual.json",
        "test_CRMUS_MU_pred_generalist.json"
    )