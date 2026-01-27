import pandas as pd
import os
import glob
from tqdm import tqdm

# ================= 🔧 配置区域  =================
# 1. txt file path
RAW_DATA_FOLDER = r'D:\Traffic_Prediction\data'

# 2. 想要提取的目标检测器 ID
TARGET_STATION_ID = 407204

# 3. 输出文件的保存路径和名称
OUTPUT_FILE = r'D:\Traffic_Prediction\data\station_407204_3months.csv'

# ==============================================================

def run_etl_process():
    print(f"🚀 [ETL Start] 开始处理数据...")
    print(f"   📂 数据源路径: {RAW_DATA_FOLDER}")
    print(f"   🎯 目标站点ID: {TARGET_STATION_ID}")

    # 1. 获取所有 txt 文件
    #    glob 会自动匹配符合规则的文件路径
    search_pattern = os.path.join(RAW_DATA_FOLDER, "d04_text_station_5min_*.txt")
    file_list = sorted(glob.glob(search_pattern))

    if not file_list:
        print("❌ 错误：在指定文件夹没找到任何 .txt 文件！请检查路径。")
        return

    print(f"   📄 发现文件数量: {len(file_list)} 个")

    # 2. 循环读取并筛选
    print(f"   🔄 正在逐个读取并提取数据 (请稍候)...")

    extracted_data_list = []

    # 使用 tqdm 显示进度条
    for file_path in tqdm(file_list, desc="Processing Files", unit="file"):
        try:
            # PeMS 原始数据没有表头 (header=None)
            # 我们只需要读取关键列以节省内存：
            # Col 0: Timestamp (时间)
            # Col 1: Station ID (站点ID)
            # Col 9: Total Flow (流量)
            # Col 11: Avg Speed (速度)
            # (注：列号基于 PeMS 标准格式)
            df_temp = pd.read_csv(
                file_path,
                header=None,
                usecols=[0, 1, 9, 11],
                names=['Timestamp', 'Station', 'Flow', 'Speed']
            )

            # 筛选目标站点
            df_target = df_temp[df_temp['Station'] == TARGET_STATION_ID].copy()

            # 如果这一天有数据，就存起来
            if not df_target.empty:
                extracted_data_list.append(df_target)

        except Exception as e:
            print(f"\n⚠️ 读取文件出错: {os.path.basename(file_path)} -> {e}")

    # 3. 合并数据
    if extracted_data_list:
        print(f"   🧩 正在合并 {len(extracted_data_list)} 天的数据...")
        all_data = pd.concat(extracted_data_list, ignore_index=True)

        # 4. 数据清洗与排序
        # 转换时间格式
        all_data['Timestamp'] = pd.to_datetime(all_data['Timestamp'], format='%m/%d/%Y %H:%M:%S')
        # 按时间排序
        all_data.sort_values('Timestamp', inplace=True)

        # 简单预览
        print("-" * 30)
        print(f"   📊 数据概览:")
        print(f"   起始时间: {all_data['Timestamp'].min()}")
        print(f"   结束时间: {all_data['Timestamp'].max()}")
        print(f"   总记录数: {len(all_data)} 条 (预期约为 17,000+)")
        print("-" * 30)

        # 5. 保存到 CSV
        # index=False 表示不保存最左边的序号列
        all_data.to_csv(OUTPUT_FILE, index=False)
        print(f"✅ [Success] 数据已成功保存到: {OUTPUT_FILE}")
        print(f"   现在你可以直接用 LSTM 代码读取这个 CSV 文件了！")

    else:
        print("❌ 警告：所有文件中都没有找到目标站点的数据！请检查 Station ID 是否正确。")


if __name__ == '__main__':
    run_etl_process()