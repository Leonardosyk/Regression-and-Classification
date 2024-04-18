import pandas as pd
import matplotlib.pyplot as plt

# 创建数据
data = {
    "Year": ["Year 1", "Year 2", "Year 3", "Year 4", "Year 5"],
    "Robot_Cost": [750, 750, 750, 750, 750],
    "Robots_Count": [10, 50, 150, 200, 300],
    "Shipping_Cost": [10000, 10000, 10000, 10000, 10000],
    "Shipping_Count": [0, 5, 15, 20, 30],
    "Hardware_Cost": [3000, 3000, 3000, 3000, 3000],
    "Hardware_Count": [2, 0, 0, 0, 0],
    "Salaries": [50000, 50000, 50000, 50000, 50000],
    "Staff_Count": [2, 2, 2, 2, 2],
    "Office_Cost": [1500, 1500, 1500, 1500, 1500],
    "Office_Count": [12, 12, 12, 12, 12],
    "Swarm_Sales": [37500, 37500, 37500, 37500, 37500],
    "Sales_Count": [0, 5, 15, 20, 30]
}

# 将数据转换为DataFrame
df = pd.DataFrame(data)

# 计算每年的总开支和收入
df['Total_Outgoings'] = (df['Robot_Cost'] * df['Robots_Count'] +
                         df['Shipping_Cost'] * df['Shipping_Count'] +
                         df['Hardware_Cost'] * df['Hardware_Count'] +
                         df['Salaries'] * df['Staff_Count'] +
                         df['Office_Cost'] * df['Office_Count'])

df['Total_Income'] = df['Swarm_Sales'] * df['Sales_Count']

# 计算利润
df['Total_Profit'] = df['Total_Income'] - df['Total_Outgoings']

# 创建一个更大的图来展示表格，增加figsize和cell的字体大小
fig, ax = plt.subplots(figsize=(40, 10))  # 更大的图表尺寸
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)  # 增加字体大小
table.scale(1.2, 1.2)  # 缩放表格大小

plt.show()

# 绘制类似于图二的收入折线图
plt.figure(figsize=(10, 6))
plt.plot(df['Year'], df['Total_Income'], label='Total Income', marker='o')
plt.plot(df['Year'], df['Total_Outgoings'], label='Total Outgoings', marker='x')
plt.plot(df['Year'], df['Total_Profit'], label='Total Profit', marker='s')
plt.title('Profit Forecast')
plt.xlabel('Year End')
plt.ylabel('Amount')
plt.legend()
plt.grid(True)
plt.show()
