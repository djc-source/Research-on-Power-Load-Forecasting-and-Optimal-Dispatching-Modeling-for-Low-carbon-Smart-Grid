import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

def parse_date_correct(date_val):
    if pd.isna(date_val):
        return None

    if isinstance(date_val, (float, np.floating)):
        date_val = int(date_val)  

    if isinstance(date_val, (int, np.integer)):
        date_str = str(date_val)
        if len(date_str) == 8:
            try:
                year = int(date_str[:4])
                month = int(date_str[4:6])
                day = int(date_str[6:8])
                return datetime(year, month, day)
            except:
                return None

    if isinstance(date_val, str):
        try:

            if len(date_val) == 8 and date_val.isdigit():
                year = int(date_val[:4])
                month = int(date_val[4:6])
                day = int(date_val[6:8])
                return datetime(year, month, day)
            else:
                return pd.to_datetime(date_val)
        except:
            return None

    return None

def preprocess_load_weather_data():

    print("开始处理电力负荷和天气数据...")

    excel_file = './data/raw/Load_ALL.xlsx'

    print("读取Excel文件...")
    area1_load = pd.read_excel(excel_file, sheet_name='Area1_Load')
    area1_weather = pd.read_excel(excel_file, sheet_name='Area1_Weather')
    area2_load = pd.read_excel(excel_file, sheet_name='Area2_Load')
    area2_weather = pd.read_excel(excel_file, sheet_name='Area2_Weather')

    print(f"Area1 Load数据形状: {area1_load.shape}")
    print(f"Area1 Weather数据形状: {area1_weather.shape}")
    print(f"Area2 Load数据形状: {area2_load.shape}")
    print(f"Area2 Weather数据形状: {area2_weather.shape}")

    def process_area_data(load_data, weather_data, area_name):

        print(f"\n处理 {area_name} 数据...")

        load_df = load_data.copy()

        date_col = 'YMD'  
        print(f"使用日期列: {date_col}")

        time_cols = [col for col in load_df.columns if str(col).startswith('T') and len(str(col)) == 5]
        print(f"找到 {len(time_cols)} 个时间列")

        load_long = []

        for idx, row in load_df.iterrows():
            date_val = row[date_col]

            date = parse_date_correct(date_val)
            if date is None:
                print(f"无法解析日期: {date_val}")
                continue

            for i, time_col in enumerate(time_cols):
                if pd.isna(row[time_col]):
                    continue

                minutes = i * 15
                hours = minutes // 60
                mins = minutes % 60

                timestamp = date + timedelta(hours=hours, minutes=mins)

                load_long.append({
                    'datetime': timestamp,
                    'load': row[time_col]
                })

        load_df_long = pd.DataFrame(load_long)
        print(f"负荷数据重塑后形状: {load_df_long.shape}")

        weather_df = weather_data.copy()

        weather_date_col = 'Unnamed: 0'  
        print(f"天气数据日期列: {weather_date_col}")

        weather_df_clean = weather_df.copy()
        weather_df_clean.rename(columns={weather_date_col: 'date'}, inplace=True)

        temp_col = '平均温度℃'
        humidity_col = '相对湿度(平均)'
        rain_col = '降雨量（mm）'

        print(f"温度列: {temp_col}")
        print(f"湿度列: {humidity_col}")
        print(f"降雨列: {rain_col}")

        weather_processed = []
        for idx, row in weather_df_clean.iterrows():
            date_val = row['date']

            date = parse_date_correct(date_val)
            if date is None:
                print(f"无法解析天气日期: {date_val}")
                continue

            for i in range(96):
                minutes = i * 15
                hours = minutes // 60
                mins = minutes % 60

                timestamp = date + timedelta(hours=hours, minutes=mins)

                weather_record = {
                    'datetime': timestamp,
                    'temp_avg': row[temp_col] if not pd.isna(row[temp_col]) else np.nan,
                    'humidity': row[humidity_col] if not pd.isna(row[humidity_col]) else np.nan,
                    'rain': row[rain_col] if not pd.isna(row[rain_col]) else np.nan
                }

                weather_processed.append(weather_record)

        weather_df_long = pd.DataFrame(weather_processed)
        print(f"天气数据扩展后形状: {weather_df_long.shape}")

        merged_df = pd.merge(load_df_long, weather_df_long, on='datetime', how='inner')

        merged_df['area'] = area_name

        merged_df = merged_df.dropna()

        merged_df = merged_df.sort_values('datetime')

        merged_df = merged_df.reset_index(drop=True)

        print(f"{area_name} 最终数据形状: {merged_df.shape}")
        if len(merged_df) > 0:
            print(f"日期范围: {merged_df['datetime'].min()} 到 {merged_df['datetime'].max()}")

            print(f"\n{area_name} 前5行数据:")
            print(merged_df.head())

            print(f"\n验证日期格式:")
            print(f"第一个日期: {merged_df['datetime'].iloc[0]} (类型: {type(merged_df['datetime'].iloc[0])})")
            print(f"最后一个日期: {merged_df['datetime'].iloc[-1]} (类型: {type(merged_df['datetime'].iloc[-1])})")

        return merged_df

    area1_processed = process_area_data(area1_load, area1_weather, 'Area1')
    area2_processed = process_area_data(area2_load, area2_weather, 'Area2')

    os.makedirs('data/processed', exist_ok=True)

    area1_processed.to_csv('data/processed/Area1_Preprocessed.csv', index=False, encoding='utf-8', date_format='%Y-%m-%d %H:%M:%S')
    area2_processed.to_csv('data/processed/Area2_Preprocessed.csv', index=False, encoding='utf-8', date_format='%Y-%m-%d %H:%M:%S')

    print(f"\n数据预处理完成！")
    print(f"Area1 数据已保存到: data/processed/Area1_Preprocessed.csv")
    print(f"Area2 数据已保存到: data/processed/Area2_Preprocessed.csv")

    return area1_processed, area2_processed

def split_train_val_test(df, area_name, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):

    print(f"\n为 {area_name} 划分训练集、验证集和测试集...")

    df_sorted = df.sort_values('datetime').reset_index(drop=True)

    n_total = len(df_sorted)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_data = df_sorted.iloc[:n_train]
    val_data = df_sorted.iloc[n_train:n_train + n_val]
    test_data = df_sorted.iloc[n_train + n_val:]

    print(f"训练集大小: {len(train_data)} ({len(train_data)/n_total*100:.1f}%)")
    print(f"验证集大小: {len(val_data)} ({len(val_data)/n_total*100:.1f}%)")
    print(f"测试集大小: {len(test_data)} ({len(test_data)/n_total*100:.1f}%)")

    print(f"训练集时间范围: {train_data['datetime'].min()} 到 {train_data['datetime'].max()}")
    print(f"验证集时间范围: {val_data['datetime'].min()} 到 {val_data['datetime'].max()}")
    print(f"测试集时间范围: {test_data['datetime'].min()} 到 {test_data['datetime'].max()}")

    train_data.to_csv(f'data/processed/{area_name}_train.csv', index=False, encoding='utf-8', date_format='%Y-%m-%d %H:%M:%S')
    val_data.to_csv(f'data/processed/{area_name}_val.csv', index=False, encoding='utf-8', date_format='%Y-%m-%d %H:%M:%S')
    test_data.to_csv(f'data/processed/{area_name}_test.csv', index=False, encoding='utf-8', date_format='%Y-%m-%d %H:%M:%S')

    print(f"{area_name} 数据集划分完成！")

    return train_data, val_data, test_data

if __name__ == "__main__":

    area1_data, area2_data = preprocess_load_weather_data()

    area1_train, area1_val, area1_test = split_train_val_test(area1_data, 'Area1')
    area2_train, area2_val, area2_test = split_train_val_test(area2_data, 'Area2')

    print("\n=== 数据预处理和划分完成 ===")
    print("生成的文件:")
    print("- data/processed/Area1_Preprocessed.csv (完整数据)")
    print("- data/processed/Area2_Preprocessed.csv (完整数据)")
    print("- data/processed/Area1_train.csv (训练集)")
    print("- data/processed/Area1_val.csv (验证集)")
    print("- data/processed/Area1_test.csv (测试集)")
    print("- data/processed/Area2_train.csv (训练集)")
    print("- data/processed/Area2_val.csv (验证集)")
    print("- data/processed/Area2_test.csv (测试集)")

    print("\n=== 数据集信息摘要 ===")
    print(f"Area1 数据总量: {len(area1_data)} 条记录")
    print(f"Area2 数据总量: {len(area2_data)} 条记录")
    print(f"数据时间范围: {area1_data['datetime'].min()} 到 {area1_data['datetime'].max()}")
    print(f"数据频率: 每15分钟一个数据点")
    print(f"每天数据点数: 96个")
    print(f"特征列: datetime, load, temp_avg, humidity, rain, area")

    with open('data/processed/README.txt', 'w', encoding='utf-8') as f:
        f.write("电力负荷预测数据集说明\n")
        f.write("=" * 50 + "\n\n")
        f.write("数据来源: Load_ALL.xlsx\n")
        f.write("处理日期: {}\n\n".format(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')))
        f.write("数据集结构:\n")
        f.write("- Area1_Preprocessed.csv: Area1完整数据集\n")
        f.write("- Area2_Preprocessed.csv: Area2完整数据集\n")
        f.write("- Area1_train.csv: Area1训练集 (70%)\n")
        f.write("- Area1_val.csv: Area1验证集 (15%)\n")
        f.write("- Area1_test.csv: Area1测试集 (15%)\n")
        f.write("- Area2_train.csv: Area2训练集 (70%)\n")
        f.write("- Area2_val.csv: Area2验证集 (15%)\n")
        f.write("- Area2_test.csv: Area2测试集 (15%)\n\n")
        f.write("字段说明:\n")
        f.write("- datetime: 时间戳 (YYYY-MM-DD HH:MM:SS)\n")
        f.write("- load: 电力负荷 (MW)\n")
        f.write("- temp_avg: 平均温度 (℃)\n")
        f.write("- humidity: 相对湿度 (%)\n")
        f.write("- rain: 降雨量 (mm)\n")
        f.write("- area: 区域标识 (Area1/Area2)\n\n")
        f.write("数据特点:\n")
        f.write("- 时间频率: 每15分钟一个数据点\n")
        f.write("- 每天数据点数: 96个\n")
        f.write(f"- 数据时间范围: {area1_data['datetime'].min()} 到 {area1_data['datetime'].max()}\n")
        f.write(f"- Area1数据量: {len(area1_data)} 条记录\n")
        f.write(f"- Area2数据量: {len(area2_data)} 条记录\n")

    print("\n数据集说明文件已保存到: data/processed/README.txt")