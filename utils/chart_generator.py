import matplotlib.pyplot as plt
import numpy as np

# ================= 数据准备区域 =================
# 实验组标签
# 您可以在这里修改换行符 \n 来控制标签的显示格式
groups = [
    'A: Internal\n(Baseline)',
    'B: Internal\n(RAG Iter 1)',
    'C: Internal\n(RAG Iter 2)',
    'D: Internal\n(RAG Iter 3)',
    'E: Generalist\n(Baseline)',
    'F: Generalist\n(RAG Iter 3)'
]

# CR (常识推理) 准确率数据
cr_scores = [60.00, 44.00, 52.00, 54.00, 54.00, 60.00]
# MU (寓意理解) 准确率数据
mu_scores = [58.00, 38.00, 54.00, 52.00, 52.00, 54.00]

# ================= 图表绘制配置 =================

# 设置画布大小 (宽, 高) - 建议海报用可以设大一点，例如 (12, 6) 或 (14, 7)
plt.figure(figsize=(12, 6), dpi=300) # dpi=300 保证海报打印清晰度

# 设置柱子的位置和宽度
x = np.arange(len(groups))
width = 0.35  # 柱状图的宽度，0.35 比较适中

# 绘制柱状图
# color 参数推荐：
# 学术蓝: '#4e79a7' (CR), 橙色: '#f28e2b' (MU)
# 或者深蓝: '#1f77b4', 浅蓝: '#aec7e8'
rects1 = plt.bar(x - width/2, cr_scores, width, label='CR Accuracy (Commonsense)', color='#4e79a7', edgecolor='black', linewidth=0.5)
rects2 = plt.bar(x + width/2, mu_scores, width, label='MU Accuracy (Moral)', color='#f28e2b', edgecolor='black', linewidth=0.5)

# ================= 标签与美化 =================

# 设置标题和轴标签
plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
plt.title('Comparison of Accuracy Across RAG Iterations & Evaluation Scenarios', fontsize=14, fontweight='bold', pad=20)

# 设置 x 轴刻度标签
plt.xticks(x, groups, fontsize=10, rotation=0) # rotation 控制标签旋转角度，如果挤在一起可以设为 15 或 30
plt.yticks(np.arange(0, 81, 10), fontsize=10)  # y轴刻度从0到80，步长10
plt.ylim(0, 75) # 设置y轴显示范围，留出顶部空间显示数字

# 添加图例 (Legend)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=11)
# bbox_to_anchor 用于把图例放到图表外面，防止遮挡柱子

# 开启水平网格线，增加易读性 (zorder=0 保证网格在柱子后面)
plt.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

# ================= 数值标注函数 =================
def autolabel(rects):
    """在每个柱子上方显示具体数值"""
    for rect in rects:
        height = rect.get_height()
        plt.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

# 调用函数显示数值
autolabel(rects1)
autolabel(rects2)

# ================= 保存与显示 =================
plt.tight_layout() # 自动调整布局，防止标签被切掉

# 保存图片 (支持 png, pdf, svg 等格式)
# 建议海报使用 .pdf 或 .svg 矢量格式，放大不失真
plt.savefig('rag_accuracy_comparison.png', dpi=300, bbox_inches='tight')
# plt.savefig('rag_accuracy_comparison.svg', format='svg', bbox_inches='tight')

plt.show()