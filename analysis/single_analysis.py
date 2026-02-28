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

# ================= 🔧 配置 (必须与 Script 8 完全一致) =================
CSV_FILE_PATH = 'data/station_407204_3months.csv'
MODEL_PATH = 'checkpoint/champion_model.pth'  # 👈 加载单流模型的权重

# --- 物理场景 ---
NUM_LANES = 4
HORIZON = 6

# --- 窗口定义 (串联逻辑) ---
LEN_RECENT = 12
LEN_PERIOD = 24
# 总长度 = 24 (Week) + 24 (Day) + 12 (Recent) = 60

# --- 训练参数 ---
TRAIN_WEEKS = 8
VAL_WEEKS = 1

# --- 模型参数 ---
HIDDEN_SIZE = 128  # 必须与 Script 8 一致
DROPOUT = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 使用设备: {device}")


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


seed_everything(42)


# ================= 1. 数据准备  =================
def load_data_simple():
    print(f"🚀 [Step 1] 读取数据...")
    df = pd.read_csv(CSV_FILE_PATH)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.sort_values('Timestamp', inplace=True)
    return df


def create_concatenated_dataset(data, timestamps, len_recent=12, len_period=24, horizon=6):
    X, Y = [], []
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

        # 1. Weekly (Oldest)
        wk_center = i - LAG_WEEK
        wk_seq = data[wk_center - half_period: wk_center + half_period, 0]

        # 2. Daily (Middle)
        day_center = i - LAG_DAY
        day_seq = data[day_center - half_period: day_center + half_period, 0]

        # 3. Recent (Newest)
        rec_seq = data[i - len_recent + 1: i + 1, 0]

        if len(rec_seq) == len_recent and len(day_seq) == len_period and len(wk_seq) == len_period:
            # 拼接: [Week(24), Day(24), Recent(12)]
            combined_seq = np.concatenate((wk_seq, day_seq, rec_seq))
            X.append(combined_seq)
            Y.append(delta)
            valid_timestamps.append(timestamps[i + horizon - 1])

    return np.array(X), np.array(Y), np.array(valid_timestamps)


# ================= 2. 模型定义  =================
class SingleStreamLSTMAttention(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, output_size=1, dropout=0.2):
        super(SingleStreamLSTMAttention, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=0)
        self.dropout_layer = nn.Dropout(dropout)

        self.attention_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_output, _ = self.lstm(x)
        attn_weights = F.softmax(self.attention_net(h_output), dim=1)
        context = self.dropout_layer(torch.sum(attn_weights * h_output, dim=1))
        out = self.fc(context)
        return out


# ================= 3. 核心：单流切片重要性分析 =================
def evaluate_model(model, t_X, t_Y, scaler, shuffle_part=None):
    """
    通过切片索引来打乱特定部分
    Structure: [Week (0-24) | Day (24-48) | Recent (48-60)]
    """
    model.eval()

    # 复制输入
    X_in = t_X.clone()  # Shape: [Batch, 60, 1]

    # --- 关键：根据索引打乱 ---
    batch_size = X_in.size(0)
    idx = torch.randperm(batch_size)  # 生成随机索引

    if shuffle_part == 'week':
        # 打乱前 24 个时间步 (Index 0-24)
        # 注意：我们要保持时间步内部顺序不变，只在 Batch 之间交换
        # X_in[idx, :24, :] 把乱序的 Batch 赋给原位置，但这样写在 PyTorch 里比较复杂
        # 更简单的方法：取出该段 -> 打乱 -> 放回
        part = X_in[:, :24, :]
        X_in[:, :24, :] = part[idx]

    elif shuffle_part == 'day':
        # 打乱中间 24 个时间步 (Index 24-48)
        part = X_in[:, 24:48, :]
        X_in[:, 24:48, :] = part[idx]

    elif shuffle_part == 'recent':
        # 打乱最后 12 个时间步 (Index 48-60)
        part = X_in[:, 48:, :]
        X_in[:, 48:, :] = part[idx]

    with torch.no_grad():
        pred_delta_norm = model(X_in).cpu().numpy()

    # 还原逻辑
    # Base Value 必须是【真实的】Current Flow
    # 在串联序列中，Current Flow 是最后一个点 (Index -1)
    # 注意：必须用原始 t_X 取值，不能用打乱后的 X_in
    base_flow_norm = t_X[:, -1, 0].cpu().numpy().reshape(-1, 1)

    pred_flow_norm = base_flow_norm + pred_delta_norm
    true_delta_norm = t_Y.cpu().numpy()
    true_flow_norm = base_flow_norm + true_delta_norm

    pred_lane = scaler.inverse_transform(pred_flow_norm) / NUM_LANES
    true_lane = scaler.inverse_transform(true_flow_norm) / NUM_LANES

    rmse = math.sqrt(mean_squared_error(true_lane, pred_lane))
    return rmse


# ================= 主程序 =================
def run_single_stream_analysis():
    # 1. 准备数据
    print("⏳ 正在准备测试数据...")
    df = load_data_simple()
    raw_flow = df['Flow'].values.reshape(-1, 1)
    timestamps = df['Timestamp'].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_flow = scaler.fit_transform(raw_flow)

    X, Y, _ = create_concatenated_dataset(
        scaled_flow, timestamps,
        len_recent=LEN_RECENT,
        len_period=LEN_PERIOD,
        horizon=HORIZON
    )

    # 定位测试集
    POINTS_PER_WEEK = 288 * 7
    train_pts = TRAIN_WEEKS * POINTS_PER_WEEK
    val_pts = VAL_WEEKS * POINTS_PER_WEEK

    test_X = X[train_pts + val_pts:]
    test_Y = Y[train_pts + val_pts:]

    print(f"📊 分析样本数: {len(test_Y)}")

    t_X = torch.from_numpy(test_X).float().unsqueeze(2).to(device)
    t_Y = torch.from_numpy(test_Y).float().unsqueeze(1).to(device)

    # 2. 加载模型
    print(f"🚀 加载模型: {MODEL_PATH}")
    model = SingleStreamLSTMAttention(input_size=1, hidden_size=HIDDEN_SIZE, dropout=DROPOUT).to(device)

    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH))
            print("✅ 权重加载成功！")
        except RuntimeError as e:
            print(f"❌ 权重加载失败！结构不匹配。\n{e}")
            return
    else:
        print("❌ 找不到模型文件！请先运行 Script 8。")
        return

    # 3. 运行分析
    print("\n🔍 开始单流模型特征重要性测试...")

    baseline_rmse = evaluate_model(model, t_X, t_Y, scaler, shuffle_part=None)
    print(f"✅ Baseline RMSE: {baseline_rmse:.4f}")

    parts = [
        ('Recent Part (Last 12 steps)', 'recent'),
        ('Daily Part (Middle 24 steps)', 'day'),
        ('Weekly Part (First 24 steps)', 'week')
    ]

    results = []

    for name, code in parts:
        print(f"   👉 打乱 [{name}] ...")
        shuffled_rmse = evaluate_model(model, t_X, t_Y, scaler, shuffle_part=code)

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

    colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(names)))  # 用绿色区分单流
    bars = ax.barh(names, values, color=colors)
    ax.invert_yaxis()

    ax.set_xlabel('% Increase in RMSE (Importance)', fontsize=12)
    ax.set_title('Single-Stream Model Feature Analysis\n(Impact of Shuffling Time Segments)', fontsize=14,
                 fontweight='bold')

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.1, bar.get_y() + bar.get_height() / 2, f'+{width:.2f}%', va='center', fontweight='bold')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run_single_stream_analysis()