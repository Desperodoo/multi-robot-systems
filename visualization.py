import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np

# 生成随机的地理空间数据
num_points = 100
x = np.random.rand(num_points) * 10
y = np.random.rand(num_points) * 10
z = np.random.rand(num_points) * 10

# 创建3D图形对象
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制初始地理空间数据
scatter = ax.scatter(x, y, z, c='r', marker='o')

# 定义智能体的数量和位置数据
num_agents = 5
agent_x = np.random.rand(num_agents) * 10
agent_y = np.random.rand(num_agents) * 10
agent_z = np.random.rand(num_agents) * 10

# 定义智能体的颜色和标记
colors = ['b', 'g', 'r', 'c', 'm']
markers = ['o', 's', '^', 'D', 'v']

# 绘制每个智能体
agent_scatters = []
for i in range(num_agents):
    agent_scatter = ax.scatter(agent_x[i], agent_y[i], agent_z[i], c=colors[i], marker=markers[i], s=100)
    agent_scatters.append(agent_scatter)

# 更新函数
def update(frame, scatter, agent_scatters, x, y, z, agent_x, agent_y, agent_z):
    # 更新地图数据
    x += np.random.randn(num_points) * 0.1
    y += np.random.randn(num_points) * 0.1
    z += np.random.randn(num_points) * 0.1
    
    # 更新散点图数据
    scatter._offsets3d = (x, y, z)
    
    # 更新智能体数据
    for i in range(num_agents):
        agent_x[i] += np.random.randn() * 0.1
        agent_y[i] += np.random.randn() * 0.1
        agent_z[i] += np.random.randn() * 0.1
        agent_scatters[i]._offsets3d = ([agent_x[i]], [agent_y[i]], [agent_z[i]])
    
    # 返回更新后的图形对象
    return scatter, *agent_scatters

# 创建动画对象
animation = FuncAnimation(fig, update, frames=2000, fargs=(scatter, agent_scatters, x, y, z, agent_x, agent_y, agent_z), interval=2000)

# 显示动画
plt.show()
