import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# config
INPUT_DATA_PATH = 'data/station_407204_3months.csv'


def triangular_fd(k, vf, kc, kj):
    """
    Newell 三角形基本图 (Piecewise Linear)
    k: 密度
    vf: 自由流速度
    kc: 临界密度 (物理相变点)
    kj: 阻塞密度
    """
    # 自由流分支: q = vf * k
    q_free = vf * k

    # 拥堵流分支: 基于几何相似性算出的斜率
    # q_max = vf * kc
    q_congest = (vf * kc) / (kj - kc) * (kj - k)

    # 实际流量是两者的较小值
    return np.minimum(q_free, q_congest)


def calibrate_triangular_fd():
    print(f"Loading data from {INPUT_DATA_PATH}...")
    df = pd.read_csv(INPUT_DATA_PATH)

    # 清洗并计算密度
    df = df[(df['Speed'] > 0) & (df['Flow'] >= 0)].copy()
    df['Flow_Hour'] = df['Flow'] * 12
    df['Density'] = df['Flow_Hour'] / df['Speed']

    # 使用 SciPy 拟合三角形基本图
    # 边界条件 bounds=([下限], [上限])：限制 vf在50-90, kc在30-150, kj在200-600
    bounds = ([50, 30, 200], [90, 150, 600])
    p0 = [70, 80, 400]  # 初始猜测

    print("Fitting Triangular Fundamental Diagram...")
    popt, pcov = curve_fit(triangular_fd, df['Density'], df['Flow_Hour'], p0=p0, bounds=bounds)
    vf_fit, kc_fit, kj_fit = popt
    capacity = vf_fit * kc_fit

    print("\n=== 🚦 三角形基本图 (Triangular FD) 严谨校准结果 ===")
    print(f"自由流速度 (vf): {vf_fit:.2f}")
    print(f"👉 严格计算出的临界密度 (kc): {kc_fit:.2f} (这才是真实的拥堵分水岭！)")
    print(f"阻塞密度 (kj): {kj_fit:.2f}")
    print(f"最大通行能力 (Capacity): {capacity:.2f}")

    # 可视化
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Density'], df['Flow_Hour'], s=2, alpha=0.3, color='gray', label='Observed Data')

    k_range = np.linspace(0, kj_fit, 500)
    q_fit = triangular_fd(k_range, vf_fit, kc_fit, kj_fit)

    plt.plot(k_range, q_fit, color='red', linewidth=3,
             label=f'Triangular Fit\n$k_c$={kc_fit:.1f}, $q_{{max}}$={capacity:.0f}')
    plt.axvline(x=kc_fit, color='blue', linestyle='--', linewidth=2, label=f'Critical Density ($k_c$={kc_fit:.1f})')

    plt.xlabel('Density (veh/distance)')
    plt.ylabel('Hourly Flow (veh/hr)')
    plt.title('Triangular Fundamental Diagram (Data-driven Regime Split)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    output_path = 'results/triangular_fd.png'
    os.makedirs('results', exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"\n✅ 图片已保存至: {output_path}")


if __name__ == "__main__":
    calibrate_triangular_fd()