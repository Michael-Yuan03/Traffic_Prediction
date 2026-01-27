# Short-Term Traffic Flow Prediction with History-Enhanced LSTM
### 基于历史增强 LSTM 的短时交通流残差预测

## 📖 Project Overview (项目概述)
This project aims to predict short-term traffic flow changes (residuals) for the next 30 minutes. 

Instead of relying solely on recent traffic data (which often leads to "lagging" predictions), this project introduces a **History-Enhanced Architecture**. By incorporating "Daily" (Yesterday) and "Weekly" (Last Week) contexts, the model captures underlying periodic trends, significantly improving prediction accuracy and robustness against noise.

**Key Achievement:** Reduced RMSE from **7.71** (Baseline) to **6.46** (SOTA Performance).

## 🏗️ Model Architectures (模型架构)

We explored three different architectures to validate the hypothesis:

1.  **Baseline LSTM:** Standard LSTM using only recent data. (High reliance on inertia).
2.  **Multi-Branch Fusion Network:** * Three independent LSTM branches processing Recent, Daily, and Weekly patterns separately.
    * Uses a fusion layer to weight the contributions.
    * *Insight:* Highly effective at capturing Weekly seasonality.
3.  **Single-Stream Concatenated LSTM :**
    * Concatenates [Weekly + Daily + Recent] sequences into a single time-series input.
    * Uses Attention mechanism to identify relevant historical context.
    * *Result:* Best performance due to efficient fusion of Daily trends and recent fluctuations.

## 📊 Results & Performance (实验结果)

The models were evaluated on a strict **Weekly Split** (8 weeks train, 1 week val, remaining test).

| Model Architecture      | RMSE | R² Score | Key Characteristic |
|:------------------------| :--- | :--- | :--- |
| **Baseline LSTM**       | 7.71 | 0.929 | High lag, sensitive to noise |
| **Multi-Branch Fusion** | 6.61 | 0.949 | Strong noise robustness, captures Weekly trend |
| **Single-Stream**       | **6.46** | **0.951** | **Lowest error, best trend-following capability** |

### Visualization (预测效果对比)

**1. Champion Model (Single-Stream) vs Ground Truth:**
The model effectively filters high-frequency noise and follows the true trend.
![Single Stream Prediction](results/singlestream.png)

**2. Multi-Branch Fusion Prediction:**
![Multi Branch Prediction](results/multibranch.png)

## 🔍 Feature Importance Analysis (核心发现)

Why did the models improve? We used Permutation Feature Importance to look inside the "Black Box".

**Discovery 1: The Shift from Inertia to History**
* The Baseline model relied almost 100% on recent flow (Inertia).
* The improved models learned to utilize historical patterns significantly.

**Discovery 2: Daily vs. Weekly**
* **Multi-Branch Model:** Prioritized **Weekly** patterns (+23% Importance), treating weekends/weekdays differently.
* **Single-Stream Model:** Prioritized **Daily** patterns (+21% Importance), finding that "Yesterday" is often the best predictor for "Today" in this specific dataset.

![Feature Importance](results/multi_analysis.png)
![Feature Importance](results/single_analysis.png)

## 📂 Project Structure (项目结构)

```text
Traffic_Prediction/
│
├── data/
│   └── station_407204_3months.csv  # Dataset
│
├── src/
│   ├── model_baseline_lstm.py      # Baseline Model
│   ├── model_multibranch.py        # 3-Branch Architecture
│   └── model_singlestream.py       # Best Performing Model
│
├── analysis/
│   ├── multi_analysis.py           # Feature Importance for Multi-Branch
│   └── single_analysis.py          # Feature Importance for Single-Stream
│
├── checkpoint/
│   └── (Saved .pth models)
│
└── results/
    └── (Visualization plots)