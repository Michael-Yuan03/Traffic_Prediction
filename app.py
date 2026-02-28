import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)

# ================= 核心配置 =================
CSV_FILE_PATH = 'data/station_407204_3months.csv'
MODEL_PATH = 'checkpoint/champion_model.pth'
HORIZON = 6

# 物理参数
VF, KC, KJ, CAPACITY = 62.06, 87.65, 600.0, 5439.34
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiBranchLSTM(nn.Module):
    def __init__(self, hidden_size=64, output_size=1):
        super(MultiBranchLSTM, self).__init__()
        self.lstm_recent = nn.LSTM(1, hidden_size, batch_first=True)
        self.lstm_day = nn.LSTM(1, hidden_size, batch_first=True)
        self.lstm_week = nn.LSTM(1, hidden_size, batch_first=True)
        self.fusion_net = nn.Sequential(
            nn.Linear(hidden_size * 3, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, output_size)
        )

    def forward(self, x_rec, x_day, x_wk):
        _, (h_rec, _) = self.lstm_recent(x_rec)
        _, (h_day, _) = self.lstm_day(x_day)
        _, (h_wk, _) = self.lstm_week(x_wk)
        return self.fusion_net(torch.cat((h_rec.squeeze(0), h_day.squeeze(0), h_wk.squeeze(0)), dim=1))


app_data = {}


def load_system():
    if not os.path.exists(CSV_FILE_PATH):
        print(f"Error: {CSV_FILE_PATH} not found.")
        return

    df = pd.read_csv(CSV_FILE_PATH)
    df['Timestamp'] = pd.to_datetime(df['Timestamp']).dt.tz_localize(None)
    scaler = MinMaxScaler()
    scaled_flow = scaler.fit_transform(df['Flow'].values.reshape(-1, 1))

    LAG_DAY, LAG_WEEK = 288, 2016
    X_rec, X_day, X_wk = [], [], []
    valid_indices, date_time_map, time_key_to_idx = [], {}, {}

    start_idx = LAG_WEEK + 12
    end_idx = len(scaled_flow) - HORIZON - 10

    list_idx = 0
    for i in range(start_idx, end_idx):
        t = df['Timestamp'].iloc[i]
        if 6 <= t.hour <= 10:
            date_str = t.strftime('%Y-%m-%d')
            time_str = t.strftime('%H:%M')
            if date_str not in date_time_map: date_time_map[date_str] = []
            date_time_map[date_str].append(time_str)
            X_rec.append(scaled_flow[i - 11:i + 1, 0])
            X_day.append(scaled_flow[i - LAG_DAY - 12:i - LAG_DAY + 12, 0])
            X_wk.append(scaled_flow[i - LAG_WEEK - 12:i - LAG_WEEK + 12, 0])
            valid_indices.append(i)
            time_key_to_idx[f"{date_str} {time_str}"] = list_idx
            list_idx += 1

    model = MultiBranchLSTM().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location=device))
    model.eval()
    app_data.update({'X_rec': np.array(X_rec), 'X_day': np.array(X_day), 'X_wk': np.array(X_wk),
                     'df': df, 'valid_indices': valid_indices, 'date_time_map': date_time_map,
                     'time_key_to_idx': time_key_to_idx, 'scaler': scaler, 'model': model})


@app.route('/')
def index():
    return render_template('index.html', date_time_map=app_data.get('date_time_map', {}))


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    target_dt = pd.to_datetime(f"{data['date']} {data['time']}")
    delay_minutes = int(data['delay_minutes'])
    time_key = target_dt.strftime('%Y-%m-%d %H:%M')

    if time_key not in app_data['time_key_to_idx']: return jsonify({'error': '未找到数据'})

    idx = app_data['time_key_to_idx'][time_key]
    global_idx = app_data['valid_indices'][idx]
    df = app_data['df']

    hist_times = df['Timestamp'].iloc[global_idx - 11:global_idx + 1].dt.strftime('%H:%M').tolist()
    hist_flows = (df['Flow'].iloc[global_idx - 11:global_idx + 1] * 12).round().tolist()
    hist_speeds = df['Speed'].iloc[global_idx - 11:global_idx + 1].tolist()

    current_flow = hist_flows[-1]
    current_speed = hist_speeds[-1]
    current_density = (current_flow / current_speed) if current_speed > 0 else KJ

    def get_prediction(query_idx):
        t_rec = torch.from_numpy(app_data['X_rec'][query_idx:query_idx + 1]).float().unsqueeze(2).to(device)
        t_day = torch.from_numpy(app_data['X_day'][query_idx:query_idx + 1]).float().unsqueeze(2).to(device)
        t_wk = torch.from_numpy(app_data['X_wk'][query_idx:query_idx + 1]).float().unsqueeze(2).to(device)
        with torch.no_grad(): pred_delta = app_data['model'](t_rec, t_day, t_wk).cpu().numpy()
        base_flow = t_rec[:, -1, 0].cpu().numpy().reshape(-1, 1)
        return app_data['scaler'].inverse_transform(base_flow + pred_delta)[0][0] * 12

    now_pred_flow = get_prediction(idx)
    delayed_pred_flow = get_prediction(idx + delay_minutes // 5)

    status, advice = "运行顺畅", "当前流量未达瓶颈，建议正常出行。"
    if now_pred_flow > CAPACITY * 0.85:
        if delayed_pred_flow < now_pred_flow - 350:
            status, advice = "预期拥堵", f"流量逼近极限，建议推迟 {delay_minutes} 分钟。"
        else:
            status, advice = "持续拥堵", "大流量时段较段，建议避开该路段。"

    return jsonify({
        'history': {'times': hist_times, 'flows': hist_flows},
        'current_status': {'speed': round(current_speed, 1), 'density': round(current_density, 1)},
        'now_pred': {'time': (target_dt + pd.Timedelta(minutes=30)).strftime('%H:%M'), 'flow': round(now_pred_flow)},
        'delayed_pred': {'time': (target_dt + pd.Timedelta(minutes=30 + delay_minutes)).strftime('%H:%M'),
                         'flow': round(delayed_pred_flow)},
        'status': status, 'advice': advice, 'capacity': round(CAPACITY), 'warn_limit': round(CAPACITY * 0.8)
    })


if __name__ == '__main__':
    load_system()
    # 适配 Render 等云平台的端口绑定
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)