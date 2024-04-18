import pandas as pd

file_path = './penguins.csv'
df = pd.read_csv(file_path)

# 查看数据集的前几行以了解其结构
df_head = df.head()

# 获取数据集的描述性统计信息
df_description = df.describe()

# 检查缺失值情况
df_missing_values = df.isnull().sum()

# 查看每个列的数据类型
df_dtypes = df.dtypes

# df_head, df_description, df_missing_values, df_dtypes
print(df_head, df_description, df_missing_values, df_dtypes)
