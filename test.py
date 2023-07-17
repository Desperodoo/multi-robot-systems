from Occupied_Grid_Map import OccupancyGridMap
from Occupied_Grid_Map import AStar_3D
import math

# test for 3D astar
distance = math.hypot(2,2,2)
print(distance)
x = OccupancyGridMap(is3D=False,boundaries=(20,20))
x.initailize_obstacle(10,10)
print(x.grid_map)
y = AStar_3D()
path, _ = y.searching((1, 1),(5,5,5),x.ex_obstacles)
print(f'path: {path}')
