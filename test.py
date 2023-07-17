import math
from Occupied_Grid_Map import OccupancyGridMap
from Occupied_Grid_Map import AStar_3D

# test for 3D astar
distance = math.hypot(2, 2, 2)
print(distance)
x = OccupancyGridMap(is3D=True)
x.set_obstacle((1.1, 1.1, 2.2))
x.set_obstacle((4, 4, 4))
x.extended_obstacles(1)
print(x.local_observation((1, 0, 0), 2))
print(x.grid_map)
y = AStar_3D()
path, _ = y.searching((-1, -2, -1), (5, 5, 5), x.ex_obstacles)
print(f'path: {path}')

