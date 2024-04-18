from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.pipeline import make_pipeline
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

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
le_island = LabelEncoder()
df['island'] = le_island.fit_transform(df['island'])
le_sex = LabelEncoder()
df['sex'] = le_sex.fit_transform(df['sex'])

X = df.drop(['species', 'rowid'], axis=1)
y = df['species']

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用管道创建逻辑回归模型并应用标准化
# LogisticRegression默认使用一对多策略来处理多类分类问题
logreg_model = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))

# 训练模型
logreg_model.fit(X_train, y_train)

# 进行预测
y_pred_logreg = logreg_model.predict(X_test)

# 输出分类报告和准确率
print(classification_report(y_test, y_pred_logreg, target_names=le_species.classes_))
print("Accuracy:", accuracy_score(y_test, y_pred_logreg))


import numpy as np

# 计算训练集中每个类别的比例
class_proportions = y_train.value_counts(normalize=True)

# 生成随机预测，根据训练集中的类别分布
random_predictions = np.random.choice(class_proportions.index, size=len(y_test), p=class_proportions.values)

# 计算随机预测的准确率
random_accuracy = accuracy_score(y_test, random_predictions)

print(f"Random guessing accuracy: {random_accuracy:.4f}")
