import numpy as np
import matplotlib
# 自动选择可用的显示后端
try:
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
except:
    matplotlib.use('Agg')  # 无图形界面时使用非交互式后端
    import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.integrate import quad

# 设置中文字体（优先使用系统已安装的中文字体）
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial Unicode MS"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# ---------------------- 定义高斯混合分布 ----------------------
def gaussian_2d(x, mu, Sigma):
    """二维高斯概率密度函数"""
    dim = len(mu)
    inv_Sigma = np.linalg.inv(Sigma)
    det_Sigma = np.linalg.det(Sigma)
    x_mu = x - mu
    exponent = -0.5 * np.dot(np.dot(x_mu, inv_Sigma), x_mu.T)
    return (1 / (2 * np.pi ** (dim/2) * det_Sigma ** 0.5)) * np.exp(exponent)

def gmm_2d(x, pi1, mu1, Sigma1, mu2, Sigma2):
    """二维高斯混合概率密度函数"""
    return pi1 * gaussian_2d(x, mu1, Sigma1) + (1 - pi1) * gaussian_2d(x, mu2, Sigma2)

def gmm_1d(x, pi1, mu1, sigma1, mu2, sigma2):
    """一维高斯混合概率密度函数"""
    return pi1 * (1 / (sigma1 * np.sqrt(2 * np.pi))) * np.exp(-(x - mu1)**2 / (2 * sigma1**2)) + \
           (1 - pi1) * (1 / (sigma2 * np.sqrt(2 * np.pi))) * np.exp(-(x - mu2)**2 / (2 * sigma2**2))

# 模型参数
pi1 = 0.4
mu1 = np.array([10, 2])
Sigma1 = np.array([[1, 0], [0, 1]])
mu2 = np.array([0, 0])
Sigma2 = np.array([[8.4, 2.0], [2.0, 1.7]])

# ---------------------- 部分(a)(b)：一维边缘分布的均值、众数、中位数 ----------------------
# 第1维边缘分布参数
mu1_1, sigma1_1 = 10, 1
mu2_1, sigma2_1 = 0, np.sqrt(8.4)
mean_x1 = pi1 * mu1_1 + (1 - pi1) * mu2_1
print(f"第1维均值：{mean_x1:.2f}")

# 第2维边缘分布参数
mu1_2, sigma1_2 = 2, 1
mu2_2, sigma2_2 = 0, np.sqrt(1.7)
mean_x2 = pi1 * mu1_2 + (1 - pi1) * mu2_2
print(f"第2维均值：{mean_x2:.2f}")

# 数值求解第1维众数
def neg_gmm1d_x1(x):
    return -gmm_1d(x, pi1, mu1_1, sigma1_1, mu2_1, sigma2_1)
res_x1_mode = minimize(neg_gmm1d_x1, x0=5, method='BFGS')
mode_x1 = res_x1_mode.x[0]
print(f"第1维众数：{mode_x1:.2f}")

# 数值求解第2维众数
def neg_gmm1d_x2(x):
    return -gmm_1d(x, pi1, mu1_2, sigma1_2, mu2_2, sigma2_2)
res_x2_mode = minimize(neg_gmm1d_x2, x0=1, method='BFGS')
mode_x2 = res_x2_mode.x[0]
print(f"第2维众数：{mode_x2:.2f}")

# 数值求解第1维中位数
def cdf_gmm1d_x1(x):
    return quad(lambda t: gmm_1d(t, pi1, mu1_1, sigma1_1, mu2_1, sigma2_1), -np.inf, x)[0]
def find_median_x1(x):
    return abs(cdf_gmm1d_x1(x) - 0.5)
res_x1_median = minimize(find_median_x1, x0=4, method='BFGS')
median_x1 = res_x1_median.x[0]
print(f"第1维中位数：{median_x1:.2f}")

# 数值求解第2维中位数
def cdf_gmm1d_x2(x):
    return quad(lambda t: gmm_1d(t, pi1, mu1_2, sigma1_2, mu2_2, sigma2_2), -np.inf, x)[0]
def find_median_x2(x):
    return abs(cdf_gmm1d_x2(x) - 0.5)
res_x2_median = minimize(find_median_x2, x0=0.8, method='BFGS')
median_x2 = res_x2_median.x[0]
print(f"第2维中位数：{median_x2:.2f}")

# ---------------------- 部分(c)：二维分布的均值、众数 ----------------------
mean_2d = np.array([mean_x1, mean_x2])
print(f"二维均值：{mean_2d}")

# 数值求解二维众数
def neg_gmm2d(x):
    return -gmm_2d(np.array(x), pi1, mu1, Sigma1, mu2, Sigma2)
res_2d_mode = minimize(neg_gmm2d, x0=[5, 1], method='BFGS')
mode_2d = res_2d_mode.x
print(f"二维众数：{mode_2d}")

# ---------------------- 可视化：等高线图、边缘分布图 ----------------------
# 调整网格范围，确保覆盖高概率区域
x = np.linspace(-5, 15, 200)  # 增加采样点使图像更平滑
y = np.linspace(-3, 4, 200)
X, Y = np.meshgrid(x, y)

# 向量化计算Z值，提高效率
Z = np.vectorize(lambda x1, x2: gmm_2d(np.array([x1, x2]), pi1, mu1, Sigma1, mu2, Sigma2))(X, Y)

# 创建图形
plt.figure(figsize=(15, 5))

# 1. 二维等高线图
plt.subplot(1, 3, 1)
max_density = Z.max()
if max_density > 0:  # 确保有有效数据
    levels = np.logspace(np.log10(max_density * 0.001), np.log10(max_density), 10)  # 对数刻度更适合概率分布
    contour = plt.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.7)  # 填充等高线
    plt.contour(X, Y, Z, levels=levels, colors='black', linewidths=0.5)  # 轮廓线
    plt.colorbar(contour, label='概率密度')
    plt.scatter(mean_2d[0], mean_2d[1], color='red', marker='*', s=100, label='均值')
    plt.scatter(mode_2d[0], mode_2d[1], color='blue', marker='o', s=50, label='众数')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('二维高斯混合分布')
    plt.legend()
else:
    plt.text(0.5, 0.5, '无法生成等高线', ha='center', va='center', transform=plt.gca().transAxes)

# 2. 第1维边缘分布图
plt.subplot(1, 3, 2)
x1_plot = np.linspace(-5, 15, 200)
pdf_x1 = [gmm_1d(t, pi1, mu1_1, sigma1_1, mu2_1, sigma2_1) for t in x1_plot]
plt.plot(x1_plot, pdf_x1, color='black')
plt.axvline(mean_x1, color='red', linestyle='--', label=f'均值: {mean_x1:.2f}')
plt.axvline(mode_x1, color='blue', linestyle='--', label=f'众数: {mode_x1:.2f}')
plt.axvline(median_x1, color='green', linestyle='--', label=f'中位数: {median_x1:.2f}')
plt.xlabel('$x_1$')
plt.ylabel('概率密度')
plt.title('第1维边缘分布')
plt.legend()

# 3. 第2维边缘分布图
plt.subplot(1, 3, 3)
x2_plot = np.linspace(-3, 4, 200)
pdf_x2 = [gmm_1d(t, pi1, mu1_2, sigma1_2, mu2_2, sigma2_2) for t in x2_plot]
plt.plot(x2_plot, pdf_x2, color='black')
plt.axvline(mean_x2, color='red', linestyle='--', label=f'均值: {mean_x2:.2f}')
plt.axvline(mode_x2, color='blue', linestyle='--', label=f'众数: {mode_x2:.2f}')
plt.axvline(median_x2, color='green', linestyle='--', label=f'中位数: {median_x2:.2f}')
plt.xlabel('$x_2$')
plt.ylabel('概率密度')
plt.title('第2维边缘分布')
plt.legend()

plt.tight_layout()

# 尝试显示图像，失败则保存
try:
    plt.show()
except:
    plt.savefig('gmm_distribution.png', dpi=300, bbox_inches='tight')
    print("图像已保存为 gmm_distribution.png")