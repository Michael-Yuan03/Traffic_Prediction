import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import copy
import random
import os

# ================= 🔧 配置  =================
CSV_FILE_PATH = r'D:\Traffic_Prediction\data\station_407204_3months.csv'
MODEL_PATH = 'champion_model.pth'

# --- 物理场景 ---
NUM_LANES = 4
HORIZON = 6

# --- 窗口定义 ---
LEN_RECENT = 12
LEN_PERIOD = 24

# --- 训练参数 (定位测试集) ---
TRAIN_WEEKS = 8
VAL_WEEKS = 1

# --- 模型参数 ---
HIDDEN_SIZE = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 使用设备: {device}")


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


seed_everything(42)


# ================= 1. 数据准备 (与 Script 7 一致) =================
def load_data_simple():
    print(f"🚀 [Step 1] 读取数据...")
    df = pd.read_csv(CSV_FILE_PATH)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.sort_values('Timestamp', inplace=True)
    return df


def create_multi_branch_dataset(data, timestamps, len_recent=12, len_period=24, horizon=6):
    X_rec, X_day, X_wk, Y = [], [], [], []
    valid_timestamps = []

    LAG_DAY = 288
    LAG_WEEK = 2016
    half_period = len_period // 2

    start_idx = LAG_WEEK + half_period
    end_idx = len(data) - horizon

    for i in range(start_idx, end_idx):
        current_flow = data[i, 0]
        future_flow = data[i + horizon - 1, 0]
        delta = future_flow - current_flow

        rec_seq = data[i - len_recent + 1: i + 1, 0]

        day_center = i - LAG_DAY
        day_seq = data[day_center - half_period: day_center + half_period, 0]

        wk_center = i - LAG_WEEK
        wk_seq = data[wk_center - half_period: wk_center + half_period, 0]

        if len(rec_seq) == len_recent and len(day_seq) == len_period and len(wk_seq) == len_period:
            X_rec.append(rec_seq)
            X_day.append(day_seq)
            X_wk.append(wk_seq)
            Y.append(delta)
            valid_timestamps.append(timestamps[i + horizon - 1])

    return np.array(X_rec), np.array(X_day), np.array(X_wk), np.array(Y), np.array(valid_timestamps)


# ================= 2. 模型定义 (必须与 Script 7 一模一样) =================
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
        out = self.fusion_net(combined)
        return out


# ================= 3. 核心：三分支重要性分析 =================
def evaluate_model(model, t_rec, t_day, t_wk, t_y, scaler, shuffle_branch=None):
    """
    运行评估。
    shuffle_branch: 'recent', 'day', 'week' 或 None
    """
    model.eval()

    # 复制数据以免污染原始 Tensor
    in_rec = t_rec.clone()
    in_day = t_day.clone()
    in_wk = t_wk.clone()

    # --- 关键：打乱指定分支 ---
    # 我们打乱的是 Batch 顺序，即：把张三的“昨天”配给李四的“今天”
    # 这样就破坏了该分支与 Target 的关联

    if shuffle_branch == 'recent':
        idx = torch.randperm(in_rec.size(0))
        in_rec = in_rec[idx]

    elif shuffle_branch == 'day':
        idx = torch.randperm(in_day.size(0))
        in_day = in_day[idx]

    elif shuffle_branch == 'week':
        idx = torch.randperm(in_wk.size(0))
        in_wk = in_wk[idx]

    with torch.no_grad():
        pred_delta_norm = model(in_rec, in_day, in_wk).cpu().numpy()

    # 还原逻辑
    # 注意：Base Value 必须始终使用【真实的】Recent 数据的最后一个点
    # 哪怕我们 shuffle 了 recent 分支作为输入给模型看，
    # 我们在还原真实流量时，依然要基于“真实的当前时刻流量”，
    # 否则误差来源就变成了“基准值错了”，而不是“模型预测错了”。

    # t_rec 是原始未打乱的 tensor
    base_flow_norm = t_rec[:, -1, 0].cpu().numpy().reshape(-1, 1)

    pred_flow_norm = base_flow_norm + pred_delta_norm
    true_delta_norm = t_y.cpu().numpy()
    true_flow_norm = base_flow_norm + true_delta_norm

    # 反归一化
    pred_lane = scaler.inverse_transform(pred_flow_norm) / NUM_LANES
    true_lane = scaler.inverse_transform(true_flow_norm) / NUM_LANES

    rmse = math.sqrt(mean_squared_error(true_lane, pred_lane))
    return rmse


# ================= 主程序 =================
def run_analysis():
    # 1. 准备数据
    print("⏳ 正在准备测试数据...")
    df = load_data_simple()
    raw_flow = df['Flow'].values.reshape(-1, 1)
    timestamps = df['Timestamp'].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_flow = scaler.fit_transform(raw_flow)

    X_rec, X_day, X_wk, Y, Y_times = create_multi_branch_dataset(
        scaled_flow, timestamps,
        len_recent=LEN_RECENT,
        len_period=LEN_PERIOD,
        horizon=HORIZON
    )

    # 定位测试集
    POINTS_PER_WEEK = 288 * 7
    train_pts = TRAIN_WEEKS * POINTS_PER_WEEK
    val_pts = VAL_WEEKS * POINTS_PER_WEEK

    # 只取测试集
    test_rec = X_rec[train_pts + val_pts:]
    test_day = X_day[train_pts + val_pts:]
    test_wk = X_wk[train_pts + val_pts:]
    test_y = Y[train_pts + val_pts:]

    print(f"📊 分析样本数: {len(test_y)}")

    # 转 Tensor
    t_rec = torch.from_numpy(test_rec).float().unsqueeze(2).to(device)
    t_day = torch.from_numpy(test_day).float().unsqueeze(2).to(device)
    t_wk = torch.from_numpy(test_wk).float().unsqueeze(2).to(device)
    t_y = torch.from_numpy(test_y).float().unsqueeze(1).to(device)

    # 2. 加载模型
    print(f"🚀 加载模型: {MODEL_PATH}")
    model = MultiBranchLSTM(hidden_size=HIDDEN_SIZE).to(device)

    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH))
            print("✅ 权重加载成功！")
        except RuntimeError as e:
            print(f"❌ 权重加载失败！可能原因：模型结构定义与训练时不一致。\n详细错误: {e}")
            return
    else:
        print("❌ 找不到模型文件！请先运行 Script 7。")
        return

    # 3. 运行分析
    print("\n🔍 开始特征重要性测试 (Permutation Importance)...")

    # 基准
    baseline_rmse = evaluate_model(model, t_rec, t_day, t_wk, t_y, scaler, shuffle_branch=None)
    print(f"✅ Baseline RMSE: {baseline_rmse:.4f}")

    branches = [
        ('Branch 1: Recent (Real-time)', 'recent'),
        ('Branch 2: Daily (Yesterday)', 'day'),
        ('Branch 3: Weekly (Last Week)', 'week')
    ]

    results = []

    for name, code in branches:
        print(f"   👉 打乱 [{name}] ...")
        shuffled_rmse = evaluate_model(model, t_rec, t_day, t_wk, t_y, scaler, shuffle_branch=code)

        diff = shuffled_rmse - baseline_rmse
        pct = (diff / baseline_rmse) * 100
        results.append((name, pct))
        print(f"      -> New RMSE: {shuffled_rmse:.4f} (+{pct:.2f}%)")

    # 4. 画图
    print("\n🏆 最终排名:")
    results.sort(key=lambda x: x[1], reverse=True)
    names = [x[0] for x in results]
    values = [x[1] for x in results]

    for n, v in results:
        print(f"   {n}: +{v:.2f}% Impact")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.Oranges(np.linspace(0.4, 0.9, len(names)))
    bars = ax.barh(names, values, color=colors)
    ax.invert_yaxis()

    ax.set_xlabel('% Increase in RMSE (Importance)', fontsize=12)
    ax.set_title('Feature Importance: Which Branch Matters Most?', fontsize=14, fontweight='bold')

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.1, bar.get_y() + bar.get_height() / 2, f'+{width:.2f}%', va='center', fontweight='bold')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run_analysis()