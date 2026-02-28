import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ================= 🔧 Config =================
INPUT_DATA_PATH = 'data/station_407204_3months.csv'
OUTPUT_RESULT_DIR = 'results'
OUTPUT_PLOT_NAME = 'hysteresis_loop.png'


# ============================================

def visualize_hysteresis():
    print(f"Loading data from {INPUT_DATA_PATH}...")
    df = pd.read_csv(INPUT_DATA_PATH)

    # 1. 基础数据清洗与计算
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df[(df['Speed'] > 0) & (df['Flow'] >= 0)].copy()

    df['Flow_Hour'] = df['Flow'] * 12
    df['Density'] = df['Flow_Hour'] / df['Speed']

    # 2. 提取连续的时间浮点数 (0.0 - 24.0)，用于渐变色
    df['TimeOfDay'] = df['Timestamp'].dt.hour + df['Timestamp'].dt.minute / 60.0

    # 动态确定坐标轴上限，剔除极端的异常高密度点，让图表更舒展
    max_density_display = min(df['Density'].quantile(0.995) * 1.1, 300)

    # ================= 画图 =================
    os.makedirs(OUTPUT_RESULT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- 子图 1：全局散点图  ---
    ax1 = axes[0]
    scatter = ax1.scatter(df['Density'], df['Flow_Hour'],
                          c=df['TimeOfDay'], cmap='viridis',
                          s=10, alpha=0.6, edgecolors='none')

    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Time of Day (0:00 - 24:00)', fontsize=12)

    ax1.set_xlim(0, max_density_display)
    ax1.set_xlabel('Density (veh/distance)', fontsize=12)
    ax1.set_ylabel('Hourly Flow (veh/hr)', fontsize=12)
    ax1.set_title('Global FD Scatter (Color-coded by Time of Day)', fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- 子图 2：工作日平均宏观轨迹  ---
    ax2 = axes[1]

    # 背景画上所有的灰点作为参照
    ax2.scatter(df['Density'], df['Flow_Hour'], s=2, alpha=0.1, color='gray')

    # 提取所有工作日（周一到周五），过滤掉周末噪音
    df_weekday = df[df['Timestamp'].dt.dayofweek < 5].copy()

    # 按每天的具体时刻（如 08:00, 08:05）分组求平均
    df_weekday['Time'] = df_weekday['Timestamp'].dt.time
    df_avg = df_weekday.groupby('Time')[['Density', 'Flow_Hour']].mean().reset_index()

    # 转换数值型时间列用于着色和排序
    df_avg['TimeFloat'] = df_avg['Time'].apply(lambda x: x.hour + x.minute / 60.0)
    df_avg = df_avg.sort_values('TimeFloat').reset_index(drop=True)

    points_k = df_avg['Density'].values
    points_q = df_avg['Flow_Hour'].values
    times = df_avg['TimeFloat'].values

    # 连线并用连续的 Viridis 颜色渲染
    for i in range(len(points_k) - 1):
        ax2.plot(points_k[i:i + 2], points_q[i:i + 2],
                 color=plt.cm.viridis(times[i] / 24.0), linewidth=2.5)

        # 每隔几个点画一个箭头，指示时间的方向
        if i % 8 == 0 and points_k[i] > 20:
            ax2.annotate('', xy=(points_k[i + 1], points_q[i + 1]), xytext=(points_k[i], points_q[i]),
                         arrowprops=dict(arrowstyle="->", color='black', lw=1.5))

    # 闭合最后一条线（深夜 23:55 到 00:00）
    ax2.plot([points_k[-1], points_k[0]], [points_q[-1], points_q[0]],
             color=plt.cm.viridis(times[-1] / 24.0), linewidth=2.5)

    ax2.set_xlim(0, max_density_display)
    ax2.set_xlabel('Density (veh/distance)', fontsize=12)
    ax2.set_ylabel('Hourly Flow (veh/hr)', fontsize=12)
    ax2.set_title('Average Weekday Macroscopic Trajectory\n(Smoothed Hysteresis Loop)', fontsize=14, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_RESULT_DIR, OUTPUT_PLOT_NAME)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"\n✅ 迟滞环图片已保存至: {output_path}")


if __name__ == "__main__":
    visualize_hysteresis()