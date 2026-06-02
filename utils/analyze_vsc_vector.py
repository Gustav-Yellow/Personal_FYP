import json
import os
import re

# ================= 配置区域 =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDICTION_DIR = os.path.join(BASE_DIR, 'prediction_results')

# 需要分析的 VSC 结果文件列表（你可以根据实际情况增删）
FILES_TO_ANALYZE = {
    "Generalist CR": "test_CRMUS_CR_pred_generalist_vsc.json",
    "Generalist MU": "test_CRMUS_MU_pred_generalist_vsc.json",
    "Internal Test CR": "internal_test_CRMUS_CR_pred_vsc.json",
    "Internal Test MU": "internal_test_CRMUS_MU_pred_vsc.json"
}

def analyze_votes(task_name, filename):
    filepath = os.path.join(PREDICTION_DIR, filename)

    if not os.path.exists(filepath):
        print(f"⚠️ 找不到文件: {filepath}，请确认是否已运行对应的推理脚本。")
        return

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_samples = len(data)
    if total_samples == 0:
        return

    correct_final = 0           # 最终预测正确的题数
    correct_in_votes_total = 0  # 最终不论对错，选票中包含正确答案的题数

    wrong_predictions = 0       # 最终预测错误的题数
    correct_in_wrong_preds = 0  # 【核心指标】最终预测错误，但选票中包含正确答案的题数

    for item in data:
        true_ans = item.get('answer', '').strip().upper()
        pred_ans = item.get('pred_answer', '').strip().upper()
        vsc_info = item.get('vsc_debug_info', '')

        # 使用正则提取 Votes 列表中的字母
        # 匹配如 Votes: ['A', 'A', 'B'] 中的 A, B
        match = re.search(r"Votes:\s*(\[.*?\])", vsc_info)
        votes = []
        if match:
            # 提取括号里的字母 A-D
            votes = re.findall(r"['\"]([A-D])['\"]", match.group(1))

        # 统计指标
        is_final_correct = (true_ans == pred_ans)
        is_in_votes = (true_ans in votes)

        if is_final_correct:
            correct_final += 1

        if is_in_votes:
            correct_in_votes_total += 1

        if not is_final_correct:
            wrong_predictions += 1
            if is_in_votes:
                correct_in_wrong_preds += 1

    # ================= 打印分析报告 =================
    acc = correct_final / total_samples
    presence_rate = correct_in_votes_total / total_samples

    # 避免除以 0
    wrong_presence_rate = (correct_in_wrong_preds / wrong_predictions) if wrong_predictions > 0 else 0

    print(f"\n" + "="*45)
    print(f"📊 任务分析: {task_name}")
    print(f"   文件: {filename}")
    print(f"-" * 45)
    print(f"1. 基础信息:")
    print(f"   - 总题数: {total_samples}")
    print(f"   - 最终准确率 (Accuracy): {acc:.2%} ({correct_final}/{total_samples})")

    print(f"\n2. 总体召回能力:")
    print(f"   - 选票中包含正确答案的比例: {presence_rate:.2%} ({correct_in_votes_total}/{total_samples})")
    print(f"   （注：这代表模型在发散思维时，有多大概率能‘想’到正确答案）")

    print(f"\n3. 🎯 错题深度剖析 (验证你的假设):")
    print(f"   - 做错的总题数: {wrong_predictions}")
    print(f"   - 错题中，选票其实包含了正确答案的题数: {correct_in_wrong_preds}")
    print(f"   - 错题的'真理遗漏率': {wrong_presence_rate:.2%} ({correct_in_wrong_preds}/{wrong_predictions})")
    print(f"   （注：这代表模型明明生成了正确选项，却没能投出多数票的遗憾比例）")
    print("="*45)

if __name__ == "__main__":
    print("🚀 开始分析 VSC 投票记录...")
    for task_name, filename in FILES_TO_ANALYZE.items():
        analyze_votes(task_name, filename)