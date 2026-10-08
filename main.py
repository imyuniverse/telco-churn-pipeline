import sys
import os

# 强制把当前文件所在的目录加入系统路径，解决 Mac 路径导致找不到模块的问题
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from data_clean import clean_data
from model_train import train_model

if __name__ == "__main__":
    print("🚀 启动数据管道 ===")
    
    # 步骤1：加载和清洗数据
    data = clean_data()
    
    # 步骤2：训练机器学习模型
    model = train_model(data)
    
    print("✅ 管道全流程执行完毕！")