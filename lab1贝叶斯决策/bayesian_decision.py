import pandas as pd
import numpy as np
import os
from collections import Counter

# 1. 数据加载
def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 {file_path} 不存在，请检查路径！")
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"加载文件 {file_path} 时出错：{e}")
    return data

try:
    train_data = load_data('train.csv')
    test_data = load_data('test.csv')
except Exception as e:
    print(f"数据加载失败：{e}")
    exit()

# 2. 数据预处理
def preprocess_data(data):
    try:
        # 填充缺失值
        data.fillna(data.median(numeric_only=True), inplace=True)
        
        # 转换类别型数据为数值型
        categorical_columns = ['employment_type', 'industry', 'grade', 'region']
        for col in categorical_columns:
            if col in data.columns:
                data[col] = data[col].astype('category').cat.codes
        
        # 日期处理
        if 'issue_date' in data.columns:
            data['issue_date'] = pd.to_datetime(data['issue_date'], errors='coerce')
            data['issue_year'] = data['issue_date'].dt.year
            data['issue_month'] = data['issue_date'].dt.month
            data['issue_day'] = data['issue_date'].dt.day
            data.drop('issue_date', axis=1, inplace=True)
        
        # 处理工作年限字段
        if 'work_year' in data.columns:
            def parse_work_year(value):
                if pd.isna(value):
                    return np.nan
                if isinstance(value, str):
                    value = value.strip().lower()
                    if '10+' in value:
                        return 10
                    elif '< 1' in value:
                        return 0
                    elif 'year' in value:
                        try:
                            return int(value.split()[0])
                        except ValueError:
                            return np.nan
                return np.nan
            data['work_year'] = data['work_year'].apply(parse_work_year)
            data['work_year'] = data['work_year'].fillna(data['work_year'].median())
        
        # 处理 earlies_credit_mon 字段
        if 'earlies_credit_mon' in data.columns:
            def parse_earlies_credit_mon(value):
                try:
                    for fmt in ['%b-%y', '%d-%b', '%b-%Y']:
                        parsed_date = pd.to_datetime(value, format=fmt, errors='coerce')
                        if not pd.isna(parsed_date):
                            return parsed_date
                except Exception:
                    return pd.NaT
                return pd.NaT
            data['earlies_credit_mon'] = data['earlies_credit_mon'].apply(parse_earlies_credit_mon)
            data['earlies_credit_mon_diff'] = (pd.Timestamp.now() - data['earlies_credit_mon']).dt.days // 30
            data.drop('earlies_credit_mon', axis=1, inplace=True)
        
        # 删除无关属性
        drop_columns = ['loan_id', 'user_id', 'post_code', 'title']
        for col in drop_columns:
            if col in data.columns:
                data.drop(col, axis=1, inplace=True)
    except Exception as e:
        raise ValueError(f"数据预处理时出错：{e}")
    print("数据预处理成功！")
    return data

try:
    train_data = preprocess_data(train_data)
    test_data = preprocess_data(test_data)
except Exception as e:
    print(f"数据预处理失败：{e}")
    exit()

# 3. 贝叶斯分类器实现
class NaiveBayesClassifier:
    def __init__(self):
        self.prior = {}
        self.conditional = {}
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        for cls in self.classes:
            X_cls = X[y == cls]
            self.prior[cls] = len(X_cls) / len(X)
            self.conditional[cls] = {}
            for col in range(X.shape[1]):
                values, counts = np.unique(X_cls[:, col], return_counts=True)
                self.conditional[cls][col] = dict(zip(values, counts / len(X_cls)))

    def predict(self, X):
        predictions = []
        for row in X:
            posteriors = {}
            for cls in self.classes:
                posterior = self.prior[cls]
                for col in range(len(row)):
                    posterior *= self.conditional[cls][col].get(row[col], 1e-6)  # 拉普拉斯平滑
                posteriors[cls] = posterior
            predictions.append(max(posteriors, key=posteriors.get))
        return np.array(predictions)

# 4. 模型训练与评估
X_train = train_data.drop('isDefault', axis=1).values
y_train = train_data['isDefault'].values

X_test = test_data.drop('isDefault', axis=1).values
y_test = test_data['isDefault'].values

print(f"训练数据集大小: {X_train.shape[0]}")
print(f"测试数据集大小: {X_test.shape[0]}")

print("开始训练贝叶斯分类器...")

nb = NaiveBayesClassifier()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

accuracy = np.mean(y_pred == y_test)
print(f'分类准确率: {accuracy:.2%}')