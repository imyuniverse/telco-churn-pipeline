import pandas as pd
import os

def clean_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(current_dir, 'data', 'telecom_churn.csv')
    
    print("1. 正在加载并清洗数据...")
    df = pd.read_csv(filepath)
    print(f"   清洗前数据量: {df.shape}")

    # 1. 去除重复值
    df = df.drop_duplicates()
    
    # 2. 处理数值型缺失值（以 total_charges 为例，兜底处理其他数值列）
    if 'total_charges' in df.columns:
        df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')
        df['total_charges'] = df['total_charges'].fillna(df['total_charges'].median())
    
    # 处理其他数值列
    for col in df.select_dtypes(include=['number']).columns:
        df[col] = df[col].fillna(df[col].median())
    
    # 3. 处理类别型缺失值（修复 Pandas 警告：兼容 Pandas 2.0+）
    for col in df.select_dtypes(include=['object', 'string']).columns:
        df[col] = df[col].fillna('Unknown')
    
    # 4. 打印清洗后对比
    print(f"   清洗后数据量: {df.shape}")
    
    # 打印剩余缺失值
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print(f"   ⚠️ 仍有缺失值的列:\n{missing}")
    else:
        print("   ✅ 所有缺失值已处理完毕！")
    
    # 保存清洗后的数据
    df.to_csv(os.path.join(current_dir, 'data', 'cleaned_churn.csv'), index=False)
    
    return df

if __name__ == "__main__":
    clean_data()