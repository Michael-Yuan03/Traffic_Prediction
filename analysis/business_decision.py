import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

# =================  Config =================
CSV_FILE_PATH = 'data/station_407204_3months.csv'
MODEL_PATH = 'checkpoint/champion_model.pth'
NUM_LANES = 4
INTERVAL_MINUTES = 5
HORIZON = 6

# 物理参数 (fd图读图)
VF = 62.06
KC = 87.65
KJ = 600.0
CAPACITY = 5439.34
# ============================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 1. 模型基建 (same as before) ---
class MultiBranchLSTM(nn.Module):
    def __init__(self, hidden_size=64, output_size=1):
        super(MultiBranchLSTM, self).__init__()
        self.lstm_recent = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.lstm_day = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.lstm_week = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fusion_net = nn.Sequential(
            nn.Linear(hidden_size * 3, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_size)
        )

    def forward(self, x_rec, x_day, x_wk):
        _, (h_rec, _) = self.lstm_recent(x_rec)
        _, (h_day, _) = self.lstm_day(x_day)
        _, (h_wk, _) = self.lstm_week(x_wk)
        combined = torch.cat((h_rec.squeeze(0), h_day.squeeze(0), h_wk.squeeze(0)), dim=1)
        return self.fusion_net(combined)


def create_multi_branch_dataset_simple(data_flow, timestamps, len_recent=12, len_period=24, horizon=6):
    X_rec, X_day, X_wk, Y_times = [], [], [], []
    LAG_DAY, LAG_WEEK = 288, 2016
    half_period = len_period // 2
    start_idx = LAG_WEEK + half_period
    end_idx = len(data_flow) - horizon - 3

    for i in range(start_idx, end_idx):
        rec_seq = data_flow[i - len_recent + 1: i + 1, 0]
        day_center = i - LAG_DAY
        day_seq = data_flow[day_center - half_period: day_center + half_period, 0]
        wk_center = i - LAG_WEEK
        wk_seq = data_flow[wk_center - half_period: wk_center + half_period, 0]

        if len(rec_seq) == len_recent and len(day_seq) == len_period and len(wk_seq) == len_period:
            X_rec.append(rec_seq)
            X_day.append(day_seq)
            X_wk.append(wk_seq)
            Y_times.append(timestamps[i])

    return np.array(X_rec), np.array(X_day), np.array(X_wk), np.array(Y_times)


def expected_speed_from_fd(predicted_hourly_flow):
    q = min(predicted_hourly_flow, CAPACITY - 0.1)
    w = CAPACITY / (KJ - KC)
    k_congested = KJ - q / w
    v_congested = q / k_congested

    if q > CAPACITY * 0.80:  # 拥堵阈值设定为 80% 容量
        return v_congested
    return VF


# --- 2. 核心业务与全场景采样引擎 ---
def generate_business_dashboard():
    print("🚦 启动智能导航调度引擎 (寻找全业务场景样本)... 🚦\n")

    df = pd.read_csv(CSV_FILE_PATH)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.sort_values('Timestamp', inplace=True)

    raw_flow = df['Flow'].values.reshape(-1, 1)
    timestamps = df['Timestamp'].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_flow = scaler.fit_transform(raw_flow)

    X_rec, X_day, X_wk, Y_times = create_multi_branch_dataset_simple(scaled_flow, timestamps, horizon=HORIZON)

    model = MultiBranchLSTM(hidden_size=64).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))  # 修复了安全警告
    model.eval()

    # 建立三个业务分支的存储字典
    scenarios = {
        'A_Shift_Success': None,  # 错峰极佳
        'B_Persistent_Jam': None,  # 拥堵死锁
        'C_Free_Flow': None  # 畅通无阻
    }

    # 遍历数据集，直到找齐三个场景
    for i in range(len(X_rec)):
        if all(v is not None for v in scenarios.values()):
            break  # 找齐了就提前结束

        current_time = pd.Timestamp(Y_times[i])

        # 1. 预测当下出发
        t_rec = torch.from_numpy(X_rec[i:i + 1]).float().unsqueeze(2).to(device)
        t_day = torch.from_numpy(X_day[i:i + 1]).float().unsqueeze(2).to(device)
        t_wk = torch.from_numpy(X_wk[i:i + 1]).float().unsqueeze(2).to(device)

        with torch.no_grad():
            pred_delta = model(t_rec, t_day, t_wk).cpu().numpy()
        pred_section_flow = (scaler.inverse_transform(t_rec[:, -1, 0].cpu().numpy().reshape(-1, 1) + pred_delta)[0][
                                 0] / NUM_LANES) * NUM_LANES * 12
        speed_now = expected_speed_from_fd(pred_section_flow)

        # 2. 预测推迟 15 分钟
        t_rec_d = torch.from_numpy(X_rec[i + 3:i + 4]).float().unsqueeze(2).to(device)
        t_day_d = torch.from_numpy(X_day[i + 3:i + 4]).float().unsqueeze(2).to(device)
        t_wk_d = torch.from_numpy(X_wk[i + 3:i + 4]).float().unsqueeze(2).to(device)

        with torch.no_grad():
            pred_delta_d = model(t_rec_d, t_day_d, t_wk_d).cpu().numpy()
        pred_section_flow_d = (scaler.inverse_transform(t_rec_d[:, -1, 0].cpu().numpy().reshape(-1, 1) + pred_delta_d)[
                                   0][0] / NUM_LANES) * NUM_LANES * 12
        speed_delayed = expected_speed_from_fd(pred_section_flow_d)

        # 3. 业务分支归类逻辑
        data_pack = (current_time, pred_section_flow, speed_now, pred_section_flow_d, speed_delayed)

        if pred_section_flow > CAPACITY * 0.80:
            if speed_delayed > speed_now + 5 and scenarios['A_Shift_Success'] is None:
                scenarios['A_Shift_Success'] = data_pack
            elif speed_delayed <= speed_now + 5 and scenarios['B_Persistent_Jam'] is None:
                scenarios['B_Persistent_Jam'] = data_pack
        else:
            if scenarios['C_Free_Flow'] is None:
                scenarios['C_Free_Flow'] = data_pack

    # --- 结果报告 ---
    print("### 🚦 LSTM 结合物理规律的智能出行调度预测 (Predictive Departure Scheduling)\n")
    print(
        "通过将 LSTM 的时序预测结果输入 **Newell 三角形基本图 (Triangular FD)**，系统能够在未来 30 分钟预判道路是否遭遇拥堵（容量下降），并针对不同情况自动给出错峰调度建议。\n")

    titles = [
        ("✅ 场景 A：错峰效益显著 (Shift Recommended)", 'A_Shift_Success'),
        ("❌ 场景 B：拥堵持续死锁 (Persistent Congestion)", 'B_Persistent_Jam'),
        ("🟢 场景 C：全路段畅通无阻 (Free Flow)", 'C_Free_Flow')
    ]

    for title, key in titles:
        data = scenarios[key]
        if data:
            curr_t, flow1, spd1, flow2, spd2 = data
            print(f"**{title}**")
            print(f"- 🕒 查询时刻：`{curr_t.strftime('%Y-%m-%d %H:%M')}`")
            print(f"- 🚗 立即出发 (预测 30 分钟后到达)：流量 `{flow1:.0f}` veh/hr，预期车速 `{spd1:.1f}` km/h")
            print(f"- ⏳ 推迟 15 分钟出发：流量 `{flow2:.0f}` veh/hr，预期车速 `{spd2:.1f}` km/h")
            if 'Success' in key:
                print("- 💡 系统决策：强烈建议晚点出门，避开单向潮汐波峰，体验更顺畅！\n")
            elif 'Jam' in key:
                print("- 💡 系统决策：道路陷入迟滞环死锁，推迟无用，建议立即出发或换乘公共交通。\n")
            else:
                print("- 💡 系统决策：当前路网状态极佳，随时可以出发。\n")

    # --- 绘制 Dashboard 对比图 ---
    print("\n🎨 正在生成业务 Dashboard 可视化图表...")
    os.makedirs('results', exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    fig.suptitle('Predictive Departure Scheduling Dashboard\n(Expected Speed Comparison)', fontsize=16,
                 fontweight='bold')

    plot_data = [
        ('A_Shift_Success', 'Scenario A: Shift Recommended', '#2ecc71', 'Shift 15m improves speed'),
        ('B_Persistent_Jam', 'Scenario B: Persistent Jam', '#e74c3c', 'Jam remains severe'),
        ('C_Free_Flow', 'Scenario C: Free Flow', '#3498db', 'Smooth traffic anytime')
    ]

    for i, (key, title, color, desc) in enumerate(plot_data):
        ax = axes[i]
        if scenarios[key]:
            _, _, spd1, _, spd2 = scenarios[key]
            bars = ax.bar(['Depart Now', 'Depart +15 min'], [spd1, spd2], color=['#95a5a6', color], width=0.5)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_ylabel('Expected Speed (km/h)' if i == 0 else '')
            ax.set_ylim(0, 80)

            # 标注数值
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                        f'{height:.1f} km/h', ha='center', va='bottom', fontweight='bold')
            ax.text(0.5, -0.15, desc, transform=ax.transAxes, ha='center', fontsize=10, style='italic')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    out_path = 'results/business_dashboard.png'
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✅ 可视化图表已保存至: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    generate_business_dashboard()