import numpy as np
import copy
from skimage.segmentation import find_boundaries
import math
import heapq

OBSTACLE = 1
UNOCCUPIED = 0


class OccupancyGridMap:
    def __init__(self, is3D: bool, new_grid=None, obstacles=None, exploration_setting='8N') -> None:
        self.is3D = is3D
        if new_grid is None:
            self.grid_map = {}
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

    def get_map(self) -> dict:
        return self.grid_map

    # convert the float point into the int point 
    def get_pos(self, pos: tuple) -> tuple:
        if self.is3D:
            point = (round(pos[0]), round(pos[1]), round(pos[2]))  # make sure pos is int

        else:
            point = (round(pos[0]), round(pos[1]))  # make sure pos is int
        return point

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
        point = self.get_pos(pos)
        print(point)
        self.grid_map[point] = OBSTACLE
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
        point = self.get_pos(pos)
        self.grid_map.pop(point)

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
        path = [self.s_goal]
        s = self.s_goal
        while True:
            s = PARENT[s]
            path.append(s)

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
