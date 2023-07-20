from Occupied_Grid_Map import OccupancyGridMap
from Occupied_Grid_Map import AStar_3D
from astar import AStar_2D
import math
import matplotlib.pyplot as plt
import numpy as np

# test for 3D astar
distance = math.hypot(2,2,2)
print(distance)
map = OccupancyGridMap(is3D=False,boundaries=(20,20))
map.initailize_obstacle(10,10)
print(map.grid_map)
star = AStar_2D(20,20)
map.set_obstacle((2,0))
path, _ = star.searching((1,1),(15,19),map.obstacles,map.ex_obstacles)
print(f'path: {path}')
x = []
y = []
for pos in map.obstacles:
    x.append(pos[0])
    y.append(pos[1])
x = np.array(x)
y = np.array(y)
for point in path:
    plt.scatter(point[0],point[1])
plt.scatter(x,y)
plt.show()