from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

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

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 应用K-均值聚类
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# 获取聚类标签
cluster_labels = kmeans.labels_

# 评估聚类效果，使用轮廓系数
silhouette_avg = silhouette_score(X_scaled, cluster_labels)
print("For n_clusters=3, the average silhouette_score is :", silhouette_avg)

# 将聚类结果添加到数据框中进行比较
df['cluster'] = cluster_labels

# 查看每个聚类中的种类分布
print(df.groupby('cluster')['species'].value_counts())
