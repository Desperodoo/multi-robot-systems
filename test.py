from Occupied_Grid_Map import OccupancyGridMap
import math
distance = math.hypot(2,2,2)
print(distance)
x = OccupancyGridMap(True)
x.set_obstacle((1.1,1.1,2.2))
x.extended_obstacles(1)
print(x.local_observation((1,0,0), 2))
print(x.grid_map)



