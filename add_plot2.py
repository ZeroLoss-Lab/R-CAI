import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ==========================================
# 1. 填入你用 Llama3-70B 跑出的 1-4 轮打分数据
# ==========================================
rounds = ['Round 1', 'Round 2', 'Round 3', 'Round 4']

# 模拟数据：毒性稳步上升，连贯性在后期明显下降
toxicity_scores = [1.62, 2.46, 2.87, 3.25] 
coherence_scores = [2.10, 2.65, 3.05, 2.40]

# ==========================================
# 2. 全局样式设置 (ACL 标准)
# ==========================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12
})
sns.set_style("whitegrid") # 使用带微弱网格线的白底，折线图更好看

# ==========================================
# 3. 绘制双折线图
# ==========================================
fig, ax = plt.subplots(figsize=(7, 5)) # 单栏图尺寸

# 绘制 Toxicity 折线 (使用警示性的橙红色)
ax.plot(rounds, toxicity_scores, marker='o', markersize=8, linewidth=2.5, 
        color='#d95f02', label='Toxicity Score')

# 绘制 Coherence 折线 (使用学术蓝色)
ax.plot(rounds, coherence_scores, marker='s', markersize=8, linewidth=2.5, 
        color='#1f78b4', label='Coherence Score')

# ==========================================
# 4. 图表修饰
# ==========================================
ax.set_xlabel('Critique-Revision Iteration')
ax.set_ylabel('Evaluation Score (1-5)')
ax.set_ylim(1.5, 4.5) # 根据你的真实数据调整，留出适当的上下空间

# 突出显示第三轮 (通常是最佳平衡点)
ax.axvline(x=2, color='gray', linestyle='--', alpha=0.5)
ax.text(2.1, 4.2, 'Optimal Balance\n(Filtering Threshold)', color='gray', fontsize=11, style='italic')

ax.legend(loc='best', frameon=True, shadow=False)

plt.tight_layout()

# 保存为 PDF
save_path = "iteration_dynamics.pdf"
plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
print(f"✅ 迭代动态图已保存为: {save_path}")

plt.show()