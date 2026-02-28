import pandas as pd

# config
file_path = 'data/station_407204_3months.csv'

def explore_data_boundary(file_path):
    try:
        # 读取数据
        df = pd.read_csv(file_path)

        print("=== 数据集基本信息 ===")
        print(f"总行数: {len(df)}")
        print(f"包含的列名 (Columns): {list(df.columns)}\n")

        # 检查关键物理字段
        target_keywords = ['speed', 'occupancy', 'density', 'flow', 'volume', 'count']
        print("=== 核心物理字段检查 ===")
        found_fields = []
        for col in df.columns:
            for kw in target_keywords:
                if kw.lower() in col.lower() and col not in found_fields:
                    print(f"发现潜在物理字段: '{col}'")
                    found_fields.append(col)

        print("\n=== 前五行数据预览 ===")
        print(df.head())

    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}，请检查路径。")

explore_data_boundary(file_path)