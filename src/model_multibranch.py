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
SAVE_MODEL_NAME = 'champion_model.pth'

# --- 物理场景 ---
NUM_LANES = 4
HORIZON = 6  # 预测 30分钟后

# --- 窗口定义 (三分支逻辑) ---
LEN_RECENT = 12  # Branch 1: 过去 1 小时
LEN_PERIOD = 24  # Branch 2/3: 历史同期前后各 1 小时

# --- 严格切分策略 ---
TRAIN_WEEKS = 8
VAL_WEEKS = 1

# --- 模型参数 ---
BATCH_SIZE = 128
EPOCHS = 200
LEARNING_RATE = 0.001
PATIENCE = 30
HIDDEN_SIZE = 64

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


def create_multi_branch_dataset(data, timestamps, len_recent=12, len_period=24, horizon=6):
    X_rec, X_day, X_wk, Y = [], [], [], []
    valid_timestamps = []

    LAG_DAY = 288
    LAG_WEEK = 2016
    half_period = len_period // 2

    start_idx = LAG_WEEK + half_period
    end_idx = len(data) - horizon

    print(f"⏳ 正在构建三分支数据集 (Start Idx: {start_idx}, End Idx: {end_idx})...")

    for i in range(start_idx, end_idx):
        # Target: Residual (Future - Current)
        current_flow = data[i, 0]
        future_flow = data[i + horizon - 1, 0]
        delta = future_flow - current_flow

        # Branch 1: Recent
        rec_seq = data[i - len_recent + 1: i + 1, 0]

        # Branch 2: Daily (Centered)
        day_center = i - LAG_DAY
        day_seq = data[day_center - half_period: day_center + half_period, 0]

        # Branch 3: Weekly (Centered)
        wk_center = i - LAG_WEEK
        wk_seq = data[wk_center - half_period: wk_center + half_period, 0]

        if len(rec_seq) == len_recent and len(day_seq) == len_period and len(wk_seq) == len_period:
            X_rec.append(rec_seq)
            X_day.append(day_seq)
            X_wk.append(wk_seq)
            Y.append(delta)
            valid_timestamps.append(timestamps[i + horizon - 1])

    return np.array(X_rec), np.array(X_day), np.array(X_wk), np.array(Y), np.array(valid_timestamps)


# ================= 2. 模型定义 (三分支) =================
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


# ================= 3. 主程序 =================
def run_final_fusion():
    # 1. 准备数据
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

    # 2. 严格按周切分
    POINTS_PER_WEEK = 288 * 7
    train_pts = TRAIN_WEEKS * POINTS_PER_WEEK
    val_pts = VAL_WEEKS * POINTS_PER_WEEK

    total_samples = len(Y)
    if train_pts + val_pts > total_samples:
        raise ValueError(f"数据不足！需要 {train_pts + val_pts}, 只有 {total_samples}")

    print(f"📊 数据集划分 (Strict Weekly): Train={train_pts}, Val={val_pts}, Test={total_samples - train_pts - val_pts}")

    t_X_rec = torch.from_numpy(X_rec).float().unsqueeze(2).to(device)
    t_X_day = torch.from_numpy(X_day).float().unsqueeze(2).to(device)
    t_X_wk = torch.from_numpy(X_wk).float().unsqueeze(2).to(device)
    t_Y = torch.from_numpy(Y).float().unsqueeze(1).to(device)

    # DataLoader
    train_data = torch.utils.data.TensorDataset(
        t_X_rec[:train_pts], t_X_day[:train_pts], t_X_wk[:train_pts], t_Y[:train_pts]
    )
    val_data = (
        t_X_rec[train_pts:train_pts + val_pts],
        t_X_day[train_pts:train_pts + val_pts],
        t_X_wk[train_pts:train_pts + val_pts],
        t_Y[train_pts:train_pts + val_pts]
    )
    test_data = (
        t_X_rec[train_pts + val_pts:],
        t_X_day[train_pts + val_pts:],
        t_X_wk[train_pts + val_pts:],
        t_Y[train_pts + val_pts:]
    )
    test_timestamps = Y_times[train_pts + val_pts:]

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    # 3. 训练
    model = MultiBranchLSTM(hidden_size=HIDDEN_SIZE).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')
    best_weights = copy.deepcopy(model.state_dict())
    patience_cnt = 0

    print(f"🚀 开始训练... (最佳模型将保存为 {SAVE_MODEL_NAME})")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for bx_r, bx_d, bx_w, by in train_loader:
            optimizer.zero_grad()
            out = model(bx_r, bx_d, bx_w)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            v_r, v_d, v_w, v_y = val_data
            val_out = model(v_r, v_d, v_w)
            val_loss = criterion(val_out, v_y).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_cnt = 0

            # 【关键】自动保存最佳模型到硬盘
            torch.save(model.state_dict(), SAVE_MODEL_NAME)

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

    t_r, t_d, t_w, t_y = test_data
    with torch.no_grad():
        pred_delta_norm = model(t_r, t_d, t_w).cpu().numpy()
        true_delta_norm = t_y.cpu().numpy()

    # 还原 (Base Value = Recent 序列最后一个点)
    base_flow_norm = X_rec[train_pts + val_pts:, -1].reshape(-1, 1)

    pred_flow_norm = base_flow_norm + pred_delta_norm
    true_flow_norm = base_flow_norm + true_delta_norm

    pred_total = scaler.inverse_transform(pred_flow_norm)
    true_total = scaler.inverse_transform(true_flow_norm)
    pred_lane = pred_total / NUM_LANES
    true_lane = true_total / NUM_LANES

    rmse = math.sqrt(mean_squared_error(true_lane, pred_lane))
    r2 = r2_score(true_lane, pred_lane)

    print("-" * 50)
    print(f"🔥 Multi-Branch Fusion (Smoothed) 结果:")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   R²:   {r2:.4f}")
    print("-" * 50)

    # 5. 可视化 (带滑动平均)
    plt.style.use('seaborn-v0_8-whitegrid')
    df_res = pd.DataFrame({
        'Time': pd.to_datetime(test_timestamps),
        'True': true_lane.flatten(),
        'Pred': pred_lane.flatten()
    })

    # 【关键修改】计算滑动平均 (Rolling Mean)
    # window=3 代表当前点+前后点，起到温和的平滑去噪作用
    df_res['True_Smooth'] = df_res['True'].rolling(window=3, center=True, min_periods=1).mean()

    thursdays = df_res[df_res['Time'].dt.dayofweek == 3]['Time'].dt.date.unique()
    if len(thursdays) > 0:
        target_date = thursdays[-1]
        plot_data = df_res[df_res['Time'].dt.date == target_date]

        fig, ax = plt.subplots(figsize=(12, 6))

        # 画原始数据 (淡色)
        ax.plot(plot_data['Time'], plot_data['True'], label='Observed (Raw)', color='lightgray', alpha=0.5, linewidth=1)

        # 画平滑后的真实数据 (深灰色)
        ax.plot(plot_data['Time'], plot_data['True_Smooth'], label='Observed (Smoothed Trend)', color='gray', alpha=0.8,
                linewidth=2)

        # 画预测数据 (橙色)
        ax.plot(plot_data['Time'], plot_data['Pred'], label='Prediction', color='#e67e22', linestyle='--', linewidth=2)

        ax.set_title(f'Multi-Branch Prediction vs Smoothed Trend\nRMSE: {rmse:.2f}, R²: {r2:.3f}', fontsize=14,
                     fontweight='bold')
        ax.set_ylabel('Flow Rate (veh/5min/lane)', fontsize=12)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.legend()
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    run_final_fusion()