import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

# 加载内置的世界地图数据
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# 为每个国家随机生成一个颜色
world['color'] = np.random.randint(0, 100, len(world))

# 绘制世界地图，每个国家根据'color'列的值着色
fig, ax = plt.subplots(1, 1, figsize=(15, 8))
world.plot(column='color', ax=ax, legend=True, legend_kwds={'label': "Country Color", 'orientation': "horizontal"})

plt.show()
