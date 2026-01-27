import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import math
import copy
import random
import os

# ================= 🔧 配置区域 =================
CSV_FILE_PATH = r'D:\Traffic_Prediction\data\station_407204_3months.csv'

# --- 物理场景 ---
NUM_LANES = 4  # 4车道
LOOK_BACK = 12  # 1小时
HORIZON = 6  # 30分钟

# --- 切分策略 (Weekly Split) ---
TRAIN_WEEKS = 8  # 前8周训练
VAL_WEEKS = 1  # 第9周验证
# 剩余为测试集

# --- 模型参数 ---
HIDDEN_SIZE = 64
BATCH_SIZE = 64
EPOCHS = 150
LEARNING_RATE = 0.001
PATIENCE = 15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)


# ==========================================================

def load_data_baseline():
    print(f"🚀 [Step 1] 读取数据...")
    df = pd.read_csv(CSV_FILE_PATH)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.sort_values('Timestamp', inplace=True)

    # 单变量特征 (Univariate)
    scaler = MinMaxScaler(feature_range=(0, 1))
    flow_data = scaler.fit_transform(df['Flow'].values.reshape(-1, 1))

    return flow_data, df['Timestamp'].values, scaler


def create_dataset(dataset, look_back=12, horizon=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - horizon + 1):
        a = dataset[i:(i + look_back), 0]
        target = dataset[i + look_back + horizon - 1, 0]
        dataX.append(a)
        dataY.append(target)
    return np.array(dataX), np.array(dataY)


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


def run_weekly_split_experiment():
    # 1. 准备数据
    data_scaled, timestamps, scaler = load_data_baseline()
    X, Y = create_dataset(data_scaled, LOOK_BACK, HORIZON)

    Y_timestamps = timestamps[LOOK_BACK + HORIZON - 1:]
    Y_timestamps = Y_timestamps[:len(Y)]

    # 2. 按周划分数据集
    POINTS_PER_WEEK = 288 * 7
    train_points = TRAIN_WEEKS * POINTS_PER_WEEK
    val_points = VAL_WEEKS * POINTS_PER_WEEK

    if len(X) < (train_points + val_points):
        raise ValueError("数据不足以支持设定的周数分配！")

    print(f"📊 数据集划分 (Weekly Split):")
    print(f"   Train: {TRAIN_WEEKS} Weeks ({train_points} pts)")
    print(f"   Val:   {VAL_WEEKS} Weeks ({val_points} pts)")
    print(f"   Test:  Remaining ({len(X) - train_points - val_points} pts)")

    X_tensor = torch.from_numpy(X).float().unsqueeze(2).to(device)
    Y_tensor = torch.from_numpy(Y).float().unsqueeze(1).to(device)

    trainX, trainY = X_tensor[:train_points], Y_tensor[:train_points]
    valX, valY = X_tensor[train_points:train_points + val_points], Y_tensor[train_points:train_points + val_points]
    testX, testY = X_tensor[train_points + val_points:], Y_tensor[train_points + val_points:]
    test_timestamps = Y_timestamps[train_points + val_points:]

    # 3. 训练
    model = LSTMModel(input_size=1, hidden_size=HIDDEN_SIZE).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"🚀 开始训练 Standard LSTM (Weekly Split)...")

    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_cnt = 0

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(trainX, trainY), batch_size=BATCH_SIZE,
                                               shuffle=True)

    for epoch in range(EPOCHS):
        model.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(valX)
            val_loss = criterion(val_out, valY).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"🛑 Early Stopping at Epoch {epoch + 1}")
                break

        if (epoch + 1) % 20 == 0:
            print(f"   Epoch {epoch + 1} | Val Loss: {val_loss:.6f}")

    # 4. 评估
    model.load_state_dict(best_model_wts)
    model.eval()
    with torch.no_grad():
        pred_norm = model(testX).cpu().numpy()
        true_norm = testY.cpu().numpy()

    pred_total = scaler.inverse_transform(pred_norm)
    true_total = scaler.inverse_transform(true_norm)

    # 统一除以车道数
    pred_lane = pred_total / NUM_LANES
    true_lane = true_total / NUM_LANES

    rmse = math.sqrt(mean_squared_error(true_lane, pred_lane))
    r2 = r2_score(true_lane, pred_lane)

    print("-" * 50)
    print(f"🔥 Standard LSTM (Weekly Split) 结果:")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   R²:   {r2:.4f}")
    print("-" * 50)

    # ================= 🎨 统一可视化 =================
    plt.style.use('seaborn-v0_8-whitegrid')

    df_res = pd.DataFrame({
        'Time': pd.to_datetime(test_timestamps),
        'True': true_lane.flatten(),
        'Pred': pred_lane.flatten()
    })

    # 寻找测试集中【最后】一个周四
    thursdays = df_res[df_res['Time'].dt.dayofweek == 3]['Time'].dt.date.unique()

    if len(thursdays) > 0:
        target_date = thursdays[-1]
        print(f"📅 正在绘制日期: {target_date} (Test Set Last Thursday)")

        plot_data = df_res[df_res['Time'].dt.date == target_date]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(plot_data['Time'], plot_data['True'], label='Observed (Truth)', color='gray', alpha=0.6, linewidth=2)
        ax.plot(plot_data['Time'], plot_data['Pred'], label='Prediction', color='blue', linestyle='--', linewidth=2)

        ax.set_title(f'Standard LSTM (Weekly Split) Prediction\nRMSE: {rmse:.2f}, R²: {r2:.3f}', fontsize=14,
                     fontweight='bold')

        ax.set_ylabel('Flow Rate (veh/5min/lane)', fontsize=12)
        ax.set_xlabel('Time of Day', fontsize=12)

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))

        plt.legend(loc='upper left', frameon=True)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()
    else:
        print("⚠️ 测试集中没有找到周四的数据！")


if __name__ == '__main__':
    run_weekly_split_experiment()