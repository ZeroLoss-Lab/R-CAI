import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ==========================================
# 1. 在这里填入你刚刚跑完的真实数据！
# ==========================================
# X轴的标签（4个配置）
configs = ['No Clamping', '[0.2, 0.8]', '[0.3, 0.7]', '[0.4, 0.6]\n(Ours)']

# Y轴的数据（请替换为你自己的得分）
toxicity_scores = [2.81, 2.89, 2.96, 3.00]  # 整体毒性得分
coherence_scores = [2.82, 2.98, 3.14, 3.24] # 整体逻辑性/连贯性得分
diversity_scores = [3.83, 4.35, 4.98, 5.46] # 整体多样性得分

# ==========================================
# 2. 图表全局样式设置 (学术/ACL 风格)
# ==========================================
# 使用 serif 字体以匹配 LaTeX
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
})
sns.set_style("ticks") # 简洁的坐标轴刻度风格

# 颜色设计：从灰度渐变到主推的蓝色，突出 "Ours"
colors = ['#d9d9d9', '#bdbdbd', '#737373', '#2b8cbe']
edge_color = '#252525'

# 创建 1行3列 的画布，宽长比适合 ACL 双栏占据一整行 (figure*)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

# ==========================================
# 3. 绘图函数定义
# ==========================================
def plot_bar(ax, data, title, ylabel, ylim_bottom, ylim_top):
    bars = ax.bar(configs, data, color=colors, edgecolor=edge_color, linewidth=1.2, width=0.6)
    
    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim_bottom, ylim_top)
    
    # 隐藏上方和右方的边框，看起来更干净
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 在柱子顶部添加具体数值
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),  # 向上偏移 4 个像素
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12)

# ==========================================
# 4. 绘制三个子图
# ==========================================
# 图1: Toxicity (假设满分是 5)
# 注意：把 ylim_top 设置得比最大值稍微高一点，给顶部数值留空间
plot_bar(axes[0], toxicity_scores, title='Overall Toxicity', ylabel='Toxicity Score', ylim_bottom=1.0, ylim_top=5.0)

# 图2: Coherence (假设满分是 5)
plot_bar(axes[1], coherence_scores, title='Overall Coherence', ylabel='Coherence Score', ylim_bottom=1.0, ylim_top=5.0)

# 图3: Diversity (请根据你的多样性指标量纲修改 ylim_top)
plot_bar(axes[2], diversity_scores, title='Overall Diversity', ylabel='Diversity Score', ylim_bottom=2.0, ylim_top=6.5)

# ==========================================
# 5. 调整布局并保存
# ==========================================
plt.tight_layout()

# 保存为 PDF 格式（ACL 强烈建议格式，不失真）
save_path = "ablation_clamping.pdf"
plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
print(f"图表已成功保存为: {save_path}")

plt.show()