import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, confusion_matrix

def train_model(df):
    print("2. 正在准备数据与训练模型...")
    
    # 1. 基础清洗：删掉 ID 和 日期，并且处理缺失值
    df_ml = df.drop(['customer_id', 'signup_date'], axis=1, errors='ignore')
    df_ml = pd.get_dummies(df_ml, drop_first=True)
    
    # 不要用 fillna(0)，数值型用中位数填充，类别型用众数（这里简单用中位数代替演示）
    df_ml = df_ml.fillna(df_ml.median(numeric_only=True))
    df_ml = df_ml.fillna(0) # 剩余类别缺失兜底
    
    X = df_ml.drop('churn', axis=1)
    y = df_ml['churn']
    
    # 2. 分层划分 (Stratify) —— 解决数据不平衡导致测试集分布不均的问题
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. 建立 Baseline (DummyClassifier) —— 证明模型不是“瞎猜”
    print("   -> 正在评估 Baseline 基准模型...")
    base_model = DummyClassifier(strategy="most_frequent")
    base_model.fit(X_train, y_train)
    base_proba = base_model.predict_proba(X_test)[:, 1]
    base_pr_auc = average_precision_score(y_test, base_proba)
    print(f"   Baseline PR-AUC: {base_pr_auc:.4f}")

    # 4. 建立真正的 Pipeline —— 解决收敛警告 + 类别不平衡
    # StandardScaler 做特征缩放，class_weight='balanced' 处理流失客户少的问题
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ])
    
    # 5. 交叉验证 (StratifiedKFold) —— 证明模型稳定性
    print("   -> 正在进行 5折交叉验证...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='average_precision')
    print(f"   CV PR-AUC (Mean ± Std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # 6. 正式训练与评估
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = pipe.predict(X_test)
    
    print("\n   ====== 最终测试集评估 ======")
    print(f"   ROC-AUC : {roc_auc_score(y_test, proba):.4f}")
    print(f"   PR-AUC  : {average_precision_score(y_test, proba):.4f} (最重要！)")
    print(f"   Confusion Matrix:\n{confusion_matrix(y_test, pred)}")
    print(f"   Classification Report:\n{classification_report(y_test, pred, target_names=['Active', 'Churn'])}")
    
    return pipe