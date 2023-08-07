from Occupied_Grid_Map import OccupiedGridMap
from Occupied_Grid_Map import AStar_3D
from astar import AStar_2D
import math
import matplotlib.pyplot as plt
import numpy as np

# test for 3D astar
distance = math.hypot(2, 2, 2)
print(distance)
is3D = True
if is3D:
    map = OccupiedGridMap(is3D=True, boundaries=(20, 20, 20))
    map.initailize_obstacle(10, 10)
    print(map.grid_map)
    star = AStar_3D(20, 20, 20)
    map.set_obstacle((2, 0, 1))
    path, _ = star.searching((1, 1, 1),(15, 19, 3), map.obstacles, map.ex_obstacles)
    print(f'path: {path}')
    x = []
    y = []
    z = []
    for pos in map.obstacles:
        x.append(float(pos[0]))
        y.append(float(pos[1]))
        z.append(float(pos[2]))
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    for point in path:
        plt.scatter(point[0], point[1], point[2])
    plt.scatter(x, y, z)
    plt.savefig('3d_map_test.png')
else:
    map = OccupiedGridMap(is3D=False, boundaries=(20, 20))
    map.initailize_obstacle(10, 10)
    print(map.grid_map)
    star = AStar_2D(20, 20)
    map.set_obstacle((2, 1))
    path, _ = star.searching((1, 1),(15, 19), map.obstacles, map.ex_obstacles)
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
    plt.savefig('2d_map_test.png')
