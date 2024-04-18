import pandas as pd

file_path = './penguins.csv'
df = pd.read_csv(file_path)

import matplotlib.pyplot as plt
import seaborn as sns

# # 设置绘图风格
sns.set(style="whitegrid")
#
# # 创建一个画布，后续的图形将在不同的子图中显示
# fig, axs = plt.subplots(2, 2, figsize=(15, 10))
#
# # 喙长和喙深的关系，按种类着色
# sns.scatterplot(data=df, x="bill_length_mm", y="bill_depth_mm", hue="species", ax=axs[0, 0])
#
# # 翼长和体重的关系，按种类着色
# sns.scatterplot(data=df, x="flipper_length_mm", y="body_mass_g", hue="species", ax=axs[0, 1])
#
# # 喙长的箱线图，按种类分组
# sns.boxplot(data=df, x="species", y="bill_length_mm", ax=axs[1, 0])
#
# # 翼长的箱线图，按种类分组
# sns.boxplot(data=df, x="species", y="flipper_length_mm", ax=axs[1, 1])
#
# plt.tight_layout()
# plt.show()


# 喙长和喙深的关系，按种类着色
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="bill_length_mm", y="bill_depth_mm", hue="species")
plt.title("Bill Length vs Bill Depth by Species")
plt.show()

# 翼长和体重的关系，按种类着色
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="flipper_length_mm", y="body_mass_g", hue="species")
plt.title("Flipper Length vs Body Mass by Species")
plt.show()

# 喙长的箱线图，按种类分组
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x="species", y="bill_length_mm")
plt.title("Boxplot of Bill Length by Species")
plt.show()

# 翼长的箱线图，按种类分组
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x="species", y="flipper_length_mm")
plt.title("Boxplot of Flipper Length by Species")
plt.show()