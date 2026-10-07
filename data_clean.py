import pandas as pd
import os

def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(current_dir, '..', 'data', 'telecom_churn.csv')
    print(f"尝试读取文件：{filepath}")
    df = pd.read_csv(filepath)
    print("✅ 数据读取成功！")
    print("数据形状：", df.shape)
    print(df.head())
    return df  # <--- 加上这一行！！！