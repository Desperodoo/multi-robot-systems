from Occupied_Grid_Map import OccupancyGridMap
from Occupied_Grid_Map import AStar_3D
from astar import AStar_2D
import math

# test for 3D astar
distance = math.hypot(2,2,2)
print(distance)
x = OccupancyGridMap(is3D=False,boundaries=(20,20))
x.initailize_obstacle(10,10)
print(x.grid_map)
y = AStar_2D(20,20)
path, _ = y.searching((1, 1),(5,5),x.obstacles,x.ex_obstacles)
print(f'path: {path}')
