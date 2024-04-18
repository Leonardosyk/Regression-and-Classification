from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd

file_path = './penguins.csv'
df = pd.read_csv(file_path)

# 处理缺失值
df = df.dropna(subset=['sex'])
df['bill_length_mm'].fillna(df['bill_length_mm'].mean(), inplace=True)
df['bill_depth_mm'].fillna(df['bill_depth_mm'].mean(), inplace=True)
df['flipper_length_mm'].fillna(df['flipper_length_mm'].mean(), inplace=True)
df['body_mass_g'].fillna(df['body_mass_g'].mean(), inplace=True)

# 对分类变量进行编码
le_species = LabelEncoder()
df['species'] = le_species.fit_transform(df['species'])
df['island'] = LabelEncoder().fit_transform(df['island'])
df['sex'] = LabelEncoder().fit_transform(df['sex'])

# 分割数据集
X = df.drop(['species', 'rowid'], axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练随机森林模型
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 预测测试集
y_pred = rf.predict(X_test)

# 评估模型
print(classification_report(y_test, y_pred, target_names=le_species.classes_))
print("Accuracy:", accuracy_score(y_test, y_pred))

import numpy as np

# 计算训练集中每个类别的比例
class_proportions = y_train.value_counts(normalize=True)

# 生成随机预测，根据训练集中的类别分布
random_predictions = np.random.choice(class_proportions.index, size=len(y_test), p=class_proportions.values)

# 计算随机预测的准确率
random_accuracy = accuracy_score(y_test, random_predictions)

print(f"Random guessing accuracy: {random_accuracy:.4f}")
