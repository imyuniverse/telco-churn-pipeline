import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def train_model(df):
    print("2. 正在准备数据与训练模型...")
    
    # 删掉没有预测价值的 ID 和日期列
    df_ml = df.drop(['customer_id', 'signup_date'], axis=1, errors='ignore')
    
    # 把分类变量（如 plan, region）自动转换成数字（One-Hot Encoding）
    df_ml = pd.get_dummies(df_ml, drop_first=True)
    
    # 填充可能存在的缺失值
    df_ml = df_ml.fillna(0)
    
    # 分离特征和目标
    X = df_ml.drop('churn', axis=1)
    y = df_ml['churn']
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练模型
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # 评估
    accuracy = model.score(X_test, y_test)
    print(f"   模型训练完成！测试集准确率: {accuracy * 100:.2f}%")
    return model