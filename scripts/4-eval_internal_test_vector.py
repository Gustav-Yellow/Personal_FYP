import json
import os
import sys

# ================= 配置区域 =================

# 1. 基础路径 (自动定位到 My_FYP 根目录)
current_script_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(scripts_dir)

# 2. 预测结果所在目录 (请确保这里和 3-inference 脚本中的输出目录一致)
# 根据上一轮代码，我们保存到了 processed_data
PRED_DIR = os.path.join(project_root, 'prediction_results')

# 3. 待评估的文件名
FILES_TO_EVAL = [
    "internal_test_CRMUS_CR_pred_vector.json",
    "internal_test_CRMUS_MU_pred_vector.json"
]

# 4. 错误分析报告输出路径
ERROR_REPORT_DIR = os.path.join(project_root, 'evaluation_report')
os.makedirs(ERROR_REPORT_DIR, exist_ok=True)

# ================= 核心逻辑 =================

def evaluate_file(filename):
    file_path = os.path.join(PRED_DIR, filename)

    if not os.path.exists(file_path):
        print(f"❌ 找不到文件: {file_path}")
        return None

    print(f"\n正在评估: {filename} ...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total = len(data)
    correct = 0
    wrong_cases = []

    for item in data:
        # 获取标准答案 (answer) 和 预测答案 (pred_answer)
        gold = item.get('answer', '').strip().upper()
        pred = item.get('pred_answer', '').strip().upper()

        # 简单清洗，防止空值
        if not gold: gold = "UNKNOWN"
        if not pred: pred = "NULL"

        if gold == pred:
            correct += 1
        else:
            # 记录错误案例用于分析
            wrong_cases.append({
                "id": item.get('id'),
                "question": item.get('question'),
                "gold": gold,
                "pred": pred,
                "model_reasoning": item.get('model_reasoning', '')[:200] + "...", # 只截取前200字
                # 关键：保存 RAG 检索到的内容，方便查看是否是检索误导了模型
                "rag_context_snippet": item.get('rag_debug_info', '')[:300] if 'rag_debug_info' in item else "Not Saved"
            })

    accuracy = (correct / total) * 100 if total > 0 else 0

    return {
        "filename": filename,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "wrong_cases": wrong_cases
    }

def main():
    print(f"=== RAG 模型准确率评估 (Internal Test) ===")
    print(f"读取目录: {PRED_DIR}\n")

    overall_total = 0
    overall_correct = 0
    results = []

    for fname in FILES_TO_EVAL:
        res = evaluate_file(fname)
        if res:
            results.append(res)
            overall_total += res['total']
            overall_correct += res['correct']

            # 打印单个文件的结果
            print(f"  📄 文件: {fname}")
            print(f"  ✅ 正确: {res['correct']} / {res['total']}")
            print(f"  📊 准确率: {res['accuracy']:.2f}%")

            # 保存错题集
            if res['wrong_cases']:
                error_file = os.path.join(ERROR_REPORT_DIR, f"error_{fname}")
                with open(error_file, 'w', encoding='utf-8') as f:
                    json.dump(res['wrong_cases'], f, ensure_ascii=False, indent=2)
                print(f"  📝 错题已保存至: evaluation_report/error_{fname}")

    print("-" * 30)

    # 打印总结果
    if overall_total > 0:
        overall_acc = (overall_correct / overall_total) * 100
        print(f"\n🏆 综合评估结果:")
        print(f"  总样本数: {overall_total}")
        print(f"  总正确数: {overall_correct}")
        print(f"  🔥 总体准确率: {overall_acc:.2f}%")
    else:
        print("\n⚠️ 未找到有效数据进行评估。")

if __name__ == "__main__":
    main()