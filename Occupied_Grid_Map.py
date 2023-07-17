import numpy as np
from skimage.segmentation import find_boundaries
import math
import heapq
import random

OBSTACLE = 1
UNOCCUPIED = 0


class OccupancyGridMap:

    def __init__(self, is3D: bool, boundaries:tuple, new_grid=None, obstacles = None,exploration_setting='8N' ) -> None:
        self.is3D = is3D
        self.boundaries = boundaries
        if new_grid == None:
            self.grid_map = np.zeros(shape=boundaries)
        else:
            self.grid_map = new_grid
        if obstacles is None:
            self.obstacles = list()
        else:
            self.obstacles = obstacles

        self.ex_obstacles = list()
        self.moving_obstacles = list()
        self.ex_moving_obstacles = list()
        # obstacles
        self.exploration_setting = exploration_setting
    # return whole map 
    def get_map(self):
        return self.grid_map
    '''
    param:
    center_point -> (int,int,int) the center point of the obstacle
    type: -> str shape of the obstacle
        in 2D case: 
            r -> rectangle, data ->(x, y)
            c -> circle, data ->(radius)
        in 3D case:
            r -> rectangle, data ->(x,y,z)
            s -> sphere, data ->(radius)
            c -> cylindar, data -> (radius, high) 

    '''
    def add_blocker_type(self, center_point:tuple, data: tuple, type:str ):
        if self.is3D:
            if type == 'r':
                x_bound = (int(-data[0] / 2), int(data[0] / 2))
                y_bound = (int(-data[1] / 2), int(data[0] / 2))
                z_bound = (int(-data[2] / 2), int(data[2] / 2))
                for x in range(x_bound[0],x_bound[1]):
                    for y in range(y_bound[0],y_bound[1]):
                        for z in range(z_bound[0],z_bound[1]):
                            if self.in_bound((x + center_point[0],y + center_point[1],z + center_point[2])):
                                self.set_map_info((x + center_point[0],y + center_point[1],z + center_point[2]),OBSTACLE)
            elif type == 's':
                for radius in range(0,data):
                    for phi in range(0,180):
                        for rho in range(0,360):
                            p = math.radians(phi)
                            r = math.radians(rho)
                            z = int(radius * math.cos(p) + center_point[2])
                            x = int(radius * math.sin(p) * math.cos(r) + center_point[0])
                            y = radius * math.sin(p) * math.sin(r) + center_point[1]
                            if self.in_bound((x,y,z)):
                                self.set_map_info((x,y,z),OBSTACLE)
            elif type == 'c':
                for radius in range(0,data[0]):
                    for high in range(0,data[1]):
                        for phi in range(0,360):
                            p = math.radians(phi)
                            x = math.cos(p) * radius + center_point[0]
                            y = math.sin(p) * radius + center_point[1]
                            z = high + center_point[2]
                            if self.in_bound((x,y,z)):
                                self.set_map_info((x,y,z),OBSTACLE)
        else:
            if type == 'r':
                x_bound = (int(-data[0] / 2), int(data[0] / 2))
                y_bound = (int(-data[1] / 2), int(data[0] / 2))
                for x in range(x_bound[0],x_bound[1]):
                    for y in range(y_bound[0],y_bound[1]):
                        if self.in_bound((x + center_point[0],y+center_point[1])):
                            self.set_map_info((x + center_point[0],y + center_point[1]),OBSTACLE)
            elif type == 'c':
                for radius in range(0,data[0]):
                    for phi in range (0,360):
                        p = math.radians(phi)
                        x = math.cos(p) + radius + center_point[0]
                        y = math.sin(p) + radius + center_point[1]
                        if self.in_bound((x,y)):
                            self.set_map_info((x,y),OBSTACLE)
    # convert the float point into the int point
    def initailize_obstacle(self,num, center = 20):
        if self.is3D:
            shape_default = [(3,3,3),(3,3),(3)]
            shape = ['r','c','s']
            for _ in range(num):
                index = random.randrange(len(shape))
                center_point = np.random.normal(center,self.boundaries[0] / 2,3)
                self.add_blocker_type(center_point=center_point,data=shape_default[index],type=shape[index])
        else:
            shape_default = [(3,3,3),(3,3)]
            shape = ['r','c']
            for _ in range(num):
                index = random.randrange(len(shape))
                center_point = np.random.normal(center,self.boundaries[0] / 2,2)
                self.add_blocker_type(center_point=center_point,data=shape_default[index],type=shape[index])
    def round_up(self, pos:tuple) -> tuple:
        if self.is3D:
            point = (round(pos[0]), round(pos[1]), round(pos[2]))  # make sure pos is int

        else:
            point = (round(pos[0]), round(pos[1]))  # make sure pos is int
        return point
    
    def get_point_info(self,pos:tuple) -> int:
        if self.is3D:
            x, y, z = self.round_up(pos)
            return self.grid_map[x][y][z]
        else:
            x, y = self.round_up(pos)
            return self.grid_map[x][y]
    
    def set_map_info(self,pos:tuple, value):
        if self.is3D:
            x,y,z = self.round_up(pos)
            self.grid_map[x][y][z] = value
        else:
            x, y = self.round_up(pos)
            self.grid_map[x][y] = value
    
    def in_bound(self,pos:tuple):
        if self.is3D:
            x, y, z = self.round_up(pos)
            return x < self.boundaries[0] and x >= 0 and y < self.boundaries[1] and y >= 0 and z < self.boundaries[2] and z >=0
        else:
            x , y  = self.round_up(pos)
            return x < self.boundaries[0] and x >= 0 and y < self.boundaries[1] and y >=0

    def is_unoccupied(self, pos) -> bool:
        """
        :param pos: cell position we wish to check
        :param extended: whether consider extended obstacles or not
        :return: True if cell is occupied with obstacle, False else
        """
        point = self.get_pos(pos)
        return self.grid_map.get(point) is None
        # if not self.in_bounds(cell=(x, y)):
        #    raise IndexError("Map index out of bounds")

    def set_moving_obstacle(self, pos: list):
        if self.is3D:
            for (x, y, z, phi, v) in pos:
                point = self.get_pos((x, y, z))
                self.grid_map[point] = OBSTACLE
                if point not in self.moving_obstacles:
                    self.moving_obstacles.append(point)
        else:
            for (x, y, phi, v) in pos:
                point = self.get_pos((x, y))
                self.grid_map[point] = OBSTACLE
                if point not in self.moving_obstacles:
                    self.moving_obstacles.append(point)

    def extended_moving_obstacles(self, extend_dis):
        if self.is3D:
            for (x, y, z) in self.moving_obstacles:
                for x_ in range(x - extend_dis, x + extend_dis + 1):
                    for y_ in range(y - extend_dis, y + extend_dis + 1):
                        for z_ in range(z - extend_dis, z + extend_dis + 1):
                            point = self.get_pos((x_, y_, z_))
                            self.grid_map[point] = OBSTACLE
                            if point not in self.moving_obstacles:
                                if point not in self.ex_moving_obstacles:
                                    self.ex_moving_obstacles.append(point)
        else:
            for (x, y) in self.moving_obstacles:
                for x_ in range(x - extend_dis, x + extend_dis + 1):  # extend
                    for y_ in range(y - extend_dis, y + extend_dis + 1):  # extend
                        if self.in_bounds(cell=(x_, y_)):  # whether is bounded
                            point = self.get_pos((x, y))
                            self.occupancy_grid_map[point] = OBSTACLE
                            if point not in self.moving_obstacles:
                                if point not in self.ex_moving_obstacles:
                                    self.ex_moving_obstacles.append(point)

    def set_obstacle(self, pos):
        """
        :param pos: cell position we wish to set obstacle
        :return: None
        """
        point = self.round_up(pos)
        self.set_map_info(point,OBSTACLE)
        if point not in self.obstacles:
            self.obstacles.append(point)

    def set_extenhded_obstacle(self, pos):
        """
        :param pos: cell position we wish to set obstacle
        :return: None
        """
        point = self.get_pos(pos)
        if point not in self.obstacles:
            if point not in self.ex_obstacles:
                self.ex_obstacles.append(point)

    def extended_obstacles(self, extend_range: int):
        if self.is3D:
            for (x, y, z) in self.obstacles:
                point = self.get_pos((x, y, z))
                for x_ in range(point[0] - extend_range, point[0] + extend_range):
                    for y_ in range(point[1] - extend_range, point[1] + extend_range):
                        for z_ in range(point[2] - extend_range, point[2] + extend_range):
                            self.grid_map[(x_, y_, z_)] = OBSTACLE
                            if (x_, y_, z_) not in self.obstacles:
                                if (x_, y_, z_) not in self.ex_obstacles:
                                    self.ex_obstacles.append((x_, y_, z_))
        else:
            for (x, y) in self.obstacles:  # for every static obstacles
                point = self.get_pos((x, y))
                for x_ in range(point[0] - 1, point[0] + 2):  # extend
                    for y_ in range(point[1] - 1, point[1] + 2):  # extend
                        self.grid_map[(x_, y_)] = OBSTACLE
                        if (x_, y_) not in self.obstacles:  # not in original obstacles
                            if (x_, y_) not in self.ex_obstacles:  # not in the extension of other obstacles
                                self.ex_obstacles.append((x_, y_))

    def remove_obstacle(self, pos):
        """ 
        :param pos: position of obstacle
        :return: None
        """
        point = self.round_up(pos)
        self.obstacles.remove(pos)
        self.set_map_info(point,UNOCCUPIED)

    def local_observation(self, global_position: tuple, view_range: int) -> dict:
        """
        :param global_position: position of robot in the global map frame
        :param view_range: how far ahead we should look
        :return: dictionary of new observations
        """
        pos_obstacle = {}
        point = self.get_pos(global_position)
        if self.is3D:
            for x_ in range(point[0] - view_range, point[0] + view_range):
                for y_ in range(point[1] - view_range, point[1] + view_range):
                    for z_ in range(point[2] - view_range, point[2] + view_range):
                        if np.linalg.norm([point[0] - x_, point[1] - y_, point[2] - z_]) <= view_range:
                            if not self.is_unoccupied((x_, y_, z_)):
                                pos_obstacle[(x_, y_, z_)] = OBSTACLE
            return pos_obstacle
        else:
            for x_ in range(point[0] - view_range, point[0] + view_range):
                for y_ in range(point[1] - view_range, point[1] + view_range):
                    if np.linalg.norm([point[0] - x_, point[1] - y_]) <= view_range:
                        if not self.is_unoccupied((x_, y_)):
                            pos_obstacle[(x_, y_)] = OBSTACLE
            return pos_obstacle


class AStar_3D:
    # default using manhattan distance
    def __init__(self) -> None:
        
        self.u_set = [(-1, -1, -1), (-1, -1, 0), (-1, -1, 1), (-1, 0, -1), 
                      (-1, 0, 0), (-1, 0, 1), (-1, 1, -1), (-1, 1, 0),
                      (-1, 1, 1), (0, -1, -1), (0, -1, 0), (0, -1, 1),
                      (0, 0, -1), (0, 0, 1), (0, 1, -1), (0, 1, 0),
                      (0, 1, 1), (1, -1, -1), (1, -1, 0), (1, -1, 1),
                      (1, 0, -1), (1, 0, 1), (1, 1, -1), (1, 1, 0),
                      (1, 1, 1)
                    ]
        
        self.action_set = {(-1, -1, -1) : 0, (-1, -1, 0) : 1, (-1, -1, 1) : 2, (-1, 0, -1) : 3, 
                      (-1, 0, 0) : 4, (-1, 0, 1) : 5, (-1, 1, -1) : 6, (-1, 1, 0) : 7,
                      (-1, 1, 1) : 8, (0, -1, -1) : 9, (0, -1, 0) : 10, (0, -1, 1) : 11,
                      (0, 0, -1) : 12, (0, 0, 1) : 13, (0, 1, -1) : 14, (0, 1, 0) : 15,
                      (0, 1, 1) : 16, (1, -1, -1) : 17, (1, -1, 0) : 18, (1, -1, 1) : 19,
                      (1, 0, -1) : 20, (1, 0 , 0) : 21, (1, 0, 1) : 22, (1, 1, -1) : 23, (1, 1, 0) : 24,
                      (1, 1, 1) : 25}
        self.s_start = None
        self.s_goal = None
        self.obs = None
        self.s_start = None
        self.s_goal = None
        self.obs = None  # position of obstacles
        self.extended_obs = None
        self.OPEN = None  # priority queue / OPEN set
        self.CLOSED = None  # CLOSED set / VISITED order
        self.PARENT = None  # recorded parent
        self.g = None  # cost to come
        # self.path = None

    def is_collision(self, s_start, s_end):
        if s_start in self.extended_obs or s_end in self.extended_obs:
            return True
        return False

    def heuristic(self, s):
        goal = self.s_goal
        sum = 0
        for i in range(0, 3):
            sum += abs(goal[i] - s[i])
        return sum

    def cost(self, s_start, s_goal):
        if self.is_collision(s_start, s_goal):
            return math.inf
        distance = math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1], s_goal[2] - s_start[2])
        return distance

    def extract_path(self, PARENT):
        """
        Extract the path based on the PARENT set.
        :return: The planning path
        """

        path = [self.s_goal]
        s = self.s_goal

        while True:
            s = PARENT[s]
            path.append(s)

            if s == self.s_start:
                break

        return list(path)

    def f_value(self, s):
        return self.g[s] + self.heuristic(s)

    def get_neighbor(self, s):
        return [(s[0] + u[0], s[1] + u[1], s[2] + u[2]) for u in self.u_set]

    def extract_path(self, PARENT):
        path = []
        s = self.s_goal
        while True:
            point = PARENT[s]
            (x , y, z) = (point[0] - s[0], point[1] - s[1], point[2] - s[2])
            s = point
            path.append(self.action_set.get((x,y,z)))
            if s == self.s_start:
                break
        return list(path)

    def searching(self, s_start: tuple, s_goal: tuple, extended_obs):
        self.s_start = s_start
        self.s_goal = s_goal
        self.extended_obs = extended_obs
        self.OPEN = []
        self.CLOSED = []
        self.PARENT = dict()
        self.g = dict()
        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0
        self.g[self.s_goal] = math.inf
        heapq.heappush(self.OPEN, (self.f_value(self.s_start), self.s_start))
        while self.OPEN:
            _, s = heapq.heappop(self.OPEN)
            self.CLOSED.append(s)
            if s == self.s_goal:
                break
            for neighbor in self.get_neighbor(s):
                new_cost = self.g[s] + self.cost(s, neighbor)
                if neighbor not in self.g:
                    self.g[neighbor] = math.inf
                if new_cost < self.g[neighbor]:
                    self.g[neighbor] = new_cost
                    self.PARENT[neighbor] = s
                    heapq.heappush(self.OPEN, (self.f_value(neighbor), neighbor))
        path = self.extract_path(self.PARENT)
        return path, self.CLOSED
