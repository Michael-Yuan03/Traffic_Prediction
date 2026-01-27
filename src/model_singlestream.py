import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import math
import copy
import random
import os

# ================= 🔧 配置区域 =================
CSV_FILE_PATH = r'D:\Traffic_Prediction\data\station_407204_3months.csv'
SAVE_MODEL_NAME = 'single_stream_model.pth'

# --- 物理场景 ---
NUM_LANES = 4
HORIZON = 6  # 预测 30分钟后

# --- 窗口定义 (串联逻辑) ---
LEN_RECENT = 12  # 1小时
LEN_PERIOD = 24  # 2小时 (前后各1)
# 总序列长度 = 24 + 24 + 12 = 60

# --- 严格切分策略 ---
TRAIN_WEEKS = 8
VAL_WEEKS = 1

# --- 模型参数 ---
BATCH_SIZE = 256  # 单流模型显存占用小，Batch可以大点
EPOCHS = 200
LEARNING_RATE = 0.001
PATIENCE = 30
HIDDEN_SIZE = 128  # 序列长了，用大一点的 Hidden Size
DROPOUT = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 使用设备: {device}")


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)


# ================= 1. 数据准备 =================
def load_data_simple():
    print(f"🚀 [Step 1] 读取数据...")
    df = pd.read_csv(CSV_FILE_PATH)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.sort_values('Timestamp', inplace=True)
    return df


def create_concatenated_dataset(data, timestamps, len_recent=12, len_period=24, horizon=6):
    """
    构建 [Week_Seq + Day_Seq + Recent_Seq] 的长序列
    """
    X, Y = [], []
    valid_timestamps = []

    LAG_DAY = 288
    LAG_WEEK = 2016
    half_period = len_period // 2

    start_idx = LAG_WEEK + half_period
    end_idx = len(data) - horizon

    print(f"⏳ 正在构建串联数据集 (Concatenated Input)...")

    for i in range(start_idx, end_idx):
        # Y: Residual Target
        current_flow = data[i, 0]
        future_flow = data[i + horizon - 1, 0]
        delta = future_flow - current_flow

        # 1. Weekly Part (Oldest)
        wk_center = i - LAG_WEEK
        wk_seq = data[wk_center - half_period: wk_center + half_period, 0]

        # 2. Daily Part (Middle)
        day_center = i - LAG_DAY
        day_seq = data[day_center - half_period: day_center + half_period, 0]

        # 3. Recent Part (Newest)
        rec_seq = data[i - len_recent + 1: i + 1, 0]

        if len(rec_seq) == len_recent and len(day_seq) == len_period and len(wk_seq) == len_period:
            # --- 核心操作：拼接 ---
            # 顺序: [上周(24) -> 昨天(24) -> 今天(12)]
            combined_seq = np.concatenate((wk_seq, day_seq, rec_seq))

            X.append(combined_seq)
            Y.append(delta)
            valid_timestamps.append(timestamps[i + horizon - 1])

    return np.array(X), np.array(Y), np.array(valid_timestamps)


# ================= 2. 模型定义 (单流 LSTM-Attention) =================
class SingleStreamLSTMAttention(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, output_size=1, dropout=0.2):
        super(SingleStreamLSTMAttention, self).__init__()

        # LSTM 处理长序列 (Seq_Len=60)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=0)
        self.dropout_layer = nn.Dropout(dropout)

        # Attention
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: [Batch, 60, 1]

        # 1. LSTM 提取特征
        h_output, _ = self.lstm(x)  # [Batch, 60, Hidden]

        # 2. Attention 计算权重
        # 这一步就是在 60 个时间步里找重点
        attn_weights = self.attention_net(h_output)
        attn_weights = F.softmax(attn_weights, dim=1)

        # 3. 加权求和
        context = torch.sum(attn_weights * h_output, dim=1)
        context = self.dropout_layer(context)

        out = self.fc(context)
        return out


# ================= 3. 主程序 =================
def run_concatenated_final():
    # 1. 准备数据
    df = load_data_simple()
    raw_flow = df['Flow'].values.reshape(-1, 1)
    timestamps = df['Timestamp'].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_flow = scaler.fit_transform(raw_flow)

    # 构建串联数据集
    X, Y, Y_times = create_concatenated_dataset(
        scaled_flow, timestamps,
        len_recent=LEN_RECENT,
        len_period=LEN_PERIOD,
        horizon=HORIZON
    )

    # 2. 严格按周切分
    POINTS_PER_WEEK = 288 * 7
    train_pts = TRAIN_WEEKS * POINTS_PER_WEEK
    val_pts = VAL_WEEKS * POINTS_PER_WEEK

    total_samples = len(Y)
    if train_pts + val_pts > total_samples:
        raise ValueError(f"数据不足！需要 {train_pts + val_pts}, 只有 {total_samples}")

    print(f"📊 数据集划分 (Strict Weekly): Train={train_pts}, Val={val_pts}, Test={total_samples - train_pts - val_pts}")
    print(f"   输入序列长度: {X.shape[1]} (24+24+12)")

    t_X = torch.from_numpy(X).float().unsqueeze(2).to(device)  # [N, 60, 1]
    t_Y = torch.from_numpy(Y).float().unsqueeze(1).to(device)

    # DataLoader
    train_data = torch.utils.data.TensorDataset(t_X[:train_pts], t_Y[:train_pts])
    val_data = (t_X[train_pts:train_pts + val_pts], t_Y[train_pts:train_pts + val_pts])
    test_data = (t_X[train_pts + val_pts:], t_Y[train_pts + val_pts:])
    test_timestamps = Y_times[train_pts + val_pts:]

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    # 3. 训练
    model = SingleStreamLSTMAttention(input_size=1, hidden_size=HIDDEN_SIZE, dropout=DROPOUT).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')
    best_weights = copy.deepcopy(model.state_dict())
    patience_cnt = 0

    print(f"🚀 开始训练单流串联模型... (保存为 {SAVE_MODEL_NAME})")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            vx, vy = val_data
            val_out = model(vx)
            val_loss = criterion(val_out, vy).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_cnt = 0
            torch.save(model.state_dict(), SAVE_MODEL_NAME)  # 自动保存
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"🛑 Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch + 1} | Train: {avg_train_loss:.5f} | Val: {val_loss:.5f}")

    print(f"💾 模型已保存至: {SAVE_MODEL_NAME}")

    # 4. 评估
    model.load_state_dict(best_weights)
    model.eval()

    tx, ty = test_data
    with torch.no_grad():
        pred_delta_norm = model(tx).cpu().numpy()
        true_delta_norm = ty.cpu().numpy()

    # 还原 (Base Value 是 Recent 部分的最后一个点，即序列的最后一个点 -1)
    # X shape: [N, 60], Last point is current flow
    base_flow_norm = X[train_pts + val_pts:, -1].reshape(-1, 1)

    pred_flow_norm = base_flow_norm + pred_delta_norm
    true_flow_norm = base_flow_norm + true_delta_norm

    pred_total = scaler.inverse_transform(pred_flow_norm)
    true_total = scaler.inverse_transform(true_flow_norm)
    pred_lane = pred_total / NUM_LANES
    true_lane = true_total / NUM_LANES

    rmse = math.sqrt(mean_squared_error(true_lane, pred_lane))
    r2 = r2_score(true_lane, pred_lane)

    print("-" * 50)
    print(f"🔥 Single-Stream Concatenated Result:")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   R²:   {r2:.4f}")
    print("-" * 50)

    # 5. 可视化 (带平滑)
    plt.style.use('seaborn-v0_8-whitegrid')
    df_res = pd.DataFrame({
        'Time': pd.to_datetime(test_timestamps),
        'True': true_lane.flatten(),
        'Pred': pred_lane.flatten()
    })

    # 计算平滑趋势 (Rolling Mean)
    df_res['True_Smooth'] = df_res['True'].rolling(window=3, center=True, min_periods=1).mean()

    thursdays = df_res[df_res['Time'].dt.dayofweek == 3]['Time'].dt.date.unique()
    if len(thursdays) > 0:
        target_date = thursdays[-1]
        plot_data = df_res[df_res['Time'].dt.date == target_date]

        fig, ax = plt.subplots(figsize=(12, 6))

        # 原始数据 (浅色)
        ax.plot(plot_data['Time'], plot_data['True'], label='Observed (Raw)', color='lightgray', alpha=0.5, linewidth=1)
        # 平滑趋势 (深色)
        ax.plot(plot_data['Time'], plot_data['True_Smooth'], label='Observed (Smoothed)', color='gray', alpha=0.8,
                linewidth=2)
        # 预测值 (绿色，以便区分之前的橙色)
        ax.plot(plot_data['Time'], plot_data['Pred'], label='Single-Stream Prediction', color='#2ecc71', linestyle='--',
                linewidth=2)

        ax.set_title(f'Single-Stream Concatenated (Week+Day+Recent)\nRMSE: {rmse:.2f}, R²: {r2:.3f}', fontsize=14,
                     fontweight='bold')
        ax.set_ylabel('Flow Rate (veh/5min/lane)', fontsize=12)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.legend()
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    run_concatenated_final()