import numpy as np
from copy import deepcopy
from typing import Dict, List
from skimage.segmentation import find_boundaries

OBSTACLE = 1
UNOCCUPIED = 0


def generate_obstacle_map(x_dim, y_dim, num_obstacles=5, obstacle_shape=(6, 7)):
    # 创建一个空的二维数组，用于存储地图信息
    obstacle_map = np.zeros((y_dim, x_dim), dtype=int)

    # 定义一些预定义的障碍物形状
    obstacle_shapes = [
        np.ones(shape=obstacle_shape, dtype=int),  # 矩形
    ]

    # 随机选取一些障碍物形状，并将它们随机排列组合
    for i in range(num_obstacles):
        shape = obstacle_shapes[np.random.randint(0, len(obstacle_shapes))]
        y = np.random.randint(10, y_dim - shape.shape[0] - 10)
        x = np.random.randint(10, x_dim - shape.shape[1] - 10)
        obstacle_map[y:y + shape.shape[0], x:x + shape.shape[1]] += shape

    # 将障碍物内部的点设为1
    obstacle_map[obstacle_map > 0] = 1
    # 提取障碍物的边界
    boundary_map = find_boundaries(obstacle_map, mode='inner')

    obstacles = list()
    boundary_obstacles = list()
    for i in range(y_dim):  # Attention!
        for j in range(x_dim):  # Attention!
            if obstacle_map[i, j]:
                obstacles.append((i, j))
            if boundary_map[i, j]:
                boundary_obstacles.append((i, j))

    return obstacle_map, boundary_map, obstacles, boundary_obstacles

# in generate_obstacle_map
# (height, width) => (y_dim, x_dim) => (numpy.shape[0], numpy.shape[1])


class OccupancyGridMap:
    def __init__(self, x_dim, y_dim, new_ogrid=None, obstacles=None, exploration_setting='8N'):
        """
        set initial values for the map occupancy grid
        |----------> y, column
        |           (x=0,y=2)
        |
        V (x=2, y=0)
        x, row
        :param x_dim: dimension in the x direction
        :param y_dim: dimension in the y direction
        """
        self.x_dim = x_dim
        self.y_dim = y_dim
        # the map extents in units [m]

        # the obstacle map
        if new_ogrid is None:
            self.map_extents = (x_dim, y_dim)
            self.occupancy_grid_map = np.zeros(shape=self.map_extents)
        else:
            self.occupancy_grid_map = new_ogrid

        if obstacles is None:
            self.obstacles = list()
        else:
            self.obstacles = obstacles

        self.ex_obstacles = list()
        self.moving_obstacles = list()
        self.ex_moving_obstacles = list()
        # obstacles
        self.exploration_setting = exploration_setting

    def get_map(self):
        """
        :return: return the current occupancy grid map
        """
        return self.occupancy_grid_map

    def is_unoccupied(self, pos: [int, int]) -> bool:
        """
        :param pos: cell position we wish to check
        :param extended: whether consider extended obstacles or not
        :return: True if cell is occupied with obstacle, False else
        """
        (x, y) = (round(pos[0]), round(pos[1]))  # make sure pos is int

        # if not self.in_bounds(cell=(x, y)):
        #    raise IndexError("Map index out of bounds")
        return self.occupancy_grid_map[x][y] == UNOCCUPIED

    def in_bounds(self, cell: (int, int)) -> bool:
        """
        Checks if the provided coordinates are within
        the bounds of the grid map
        :param cell: cell position (x,y)
        :return: True if within bounds, False else
        """
        (x, y) = cell
        return 0 <= x < self.x_dim and 0 <= y < self.y_dim

    def set_moving_obstacle(self, pos: list):
        for (x, y, phi, v) in pos:
            (row, col) = (round(x), round(y))  # make sure pos is int
            if self.in_bounds(cell=(row, col)):  # make sure pos is in bounds
                self.occupancy_grid_map[row, col] = OBSTACLE
                if (row, col) not in self.moving_obstacles:
                    self.moving_obstacles.append((row, col))

    def extended_moving_obstacles(self, extend_dis):
        for (x, y) in self.moving_obstacles:
            for x_ in range(x - extend_dis, x + extend_dis + 1):  # extend
                for y_ in range(y - extend_dis, y + extend_dis + 1):  # extend
                    if self.in_bounds(cell=(x_, y_)):  # whether is bounded
                        self.occupancy_grid_map[x_, y_] = OBSTACLE
                        if (x_, y_) not in self.moving_obstacles:
                            if (x_, y_) not in self.ex_moving_obstacles:
                                self.ex_moving_obstacles.append((x_, y_))

    def set_obstacle(self, pos: (int, int)):
        """
        :param pos: cell position we wish to set obstacle
        :return: None
        """
        (x, y) = (round(pos[0]), round(pos[1]))  # make sure pos is int
        (row, col) = (x, y)
        self.occupancy_grid_map[row, col] = OBSTACLE
        if (x, y) not in self.obstacles:
            self.obstacles.append((x, y))

    def set_extended_obstacle(self, pos: (int, int)):
        """
        :param pos: cell position we wish to set obstacle
        :return: None
        """
        (x, y) = (round(pos[0]), round(pos[1]))  # make sure pos is int
        (row, col) = (x, y)
        self.occupancy_grid_map[row, col] = OBSTACLE
        if (x, y) not in self.obstacles:
            if (x, y) not in self.ex_obstacles:
                self.ex_obstacles.append((x, y))

    def remove_obstacle(self, pos: (int, int)):
        """
        :param pos: position of obstacle
        :return: None
        """
        (x, y) = (round(pos[0]), round(pos[1]))  # make sure pos is int
        (row, col) = (x, y)
        self.occupancy_grid_map[row, col] = UNOCCUPIED

    def local_observation(self, global_position: (int, int), view_range: int) -> Dict:
        """
        :param global_position: position of robot in the global map frame
        :param view_range: how far ahead we should look
        :return: dictionary of new observations
        """
        px = round(global_position[0])
        py = round(global_position[1])

        nodes = list()
        for x in range(px - view_range, px + view_range + 1):
            for y in range(py - view_range, py + view_range + 1):
                if self.in_bounds((x, y)) and np.linalg.norm([px - x, py - y]) <= view_range:
                    nodes.append((x, y))
        return {node: UNOCCUPIED if self.is_unoccupied(pos=node) else OBSTACLE for node in nodes}

    def extended_obstacles(self):
        for (x, y) in self.obstacles:  # for every static obstacles
            for x_ in range(x - 1, x + 2):  # extend
                for y_ in range(y - 1, y + 2):  # extend
                    if self.in_bounds(cell=(x_, y_)):  # whether is bounded
                        self.occupancy_grid_map[x_, y_] = OBSTACLE
                        if (x_, y_) not in self.obstacles:  # not in original obstacles
                            if (x_, y_) not in self.ex_obstacles:  # not in the extension of other obstacles
                                self.ex_obstacles.append((x_, y_))


class SLAM:
    def __init__(self, global_map: OccupancyGridMap, view_range: int):
        self.ground_truth_map = global_map
        # generate a new map
        self.slam_map = None
        self.view_range = view_range

    def rescan(self, global_position: (int, int), moving_obstacles, time_step, extend_dis):
        """
        :param global_position:
        :param time_step:
        :param moving_obstacles:
        :return: slam_map: updated local map
                 pred_map: the map with extended obstacles
        """
        pred_map = deepcopy(self.ground_truth_map)
        pred_map.extended_obstacles()
        pred_map.set_moving_obstacle(pos=moving_obstacles)
        pred_map.extended_moving_obstacles(extend_dis=extend_dis)

        self.slam_map = deepcopy(self.ground_truth_map)
        self.slam_map.extended_obstacles()
        local_observation = self.slam_map.local_observation(global_position=global_position, view_range=self.view_range)
        for node, value in local_observation.items():
            if value == UNOCCUPIED:
                if not pred_map.is_unoccupied(node):
                    if (node in pred_map.moving_obstacles) or (node in pred_map.obstacles):
                        self.slam_map.set_obstacle(node)
                    else:
                        self.slam_map.set_extended_obstacle(node)
        return self.slam_map, pred_map
