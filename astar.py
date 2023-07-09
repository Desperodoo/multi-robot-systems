"""
A_star 2D
@author: huiming zhou
"""

import os
import sys
import math
import heapq

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Search_based_Planning/")

class AStar_2D:
    """AStar set the cost + heuristics as the priority
    """
    def __init__(self, width, height, heuristic_type="manhattan"):
        self.heuristic_type = heuristic_type

        self.u_set = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                      (1, 0), (1, -1), (0, -1), (-1, -1)]

        self.s_start = None
        self.s_goal = None
        self.obs = None  # position of obstacles
        self.extended_obs = None
        self.OPEN = None  # priority queue / OPEN set
        self.CLOSED = None  # CLOSED set / VISITED order
        self.PARENT = None  # recorded parent
        self.g = None  # cost to come
        # self.path = None
        self.width = width
        self.height = height

    def searching(self, s_start: tuple, s_goal: tuple, obs, extended_obs, worker_id):
        """
        A_star Searching.
        :return: path, visited order
        """
        self.s_start = s_start
        while self.s_start in extended_obs:
            extended_obs.remove(self.s_start)
        # if worker_id >= 0:
            # print(f'worker_{worker_id} remove')

        self.s_goal = s_goal
        self.obs = obs  # position of obstacles
        self.extended_obs = extended_obs
        self.OPEN = []  # priority queue / OPEN set
        self.CLOSED = []  # CLOSED set / VISITED order
        self.PARENT = dict()  # recorded parent
        self.g = dict()  # cost to come

        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0
        self.g[self.s_goal] = math.inf
        heapq.heappush(self.OPEN,
                       (self.f_value(self.s_start), self.s_start))

        if s_goal in obs + extended_obs:
            path = [s_start]
            # if worker_id >= 0:
                # print(f'worker_{worker_id} goal is occupied')

        else:
            # if worker_id >= 0:
                # print(f'worker_{worker_id} find path')
            # search_time = 0
            while self.OPEN:
                # search_time += 1
                _, s = heapq.heappop(self.OPEN)
                self.CLOSED.append(s)
                if s == self.s_goal:  # stop condition
                    break

                for s_n in self.get_neighbor(s):
                    new_cost = self.g[s] + self.cost(s, s_n)

                    if s_n not in self.g:
                        self.g[s_n] = math.inf

                    if new_cost < self.g[s_n]:  # conditions for updating Cost
                        self.g[s_n] = new_cost
                        self.PARENT[s_n] = s
                        heapq.heappush(self.OPEN, (self.f_value(s_n), s_n))
            # print(f'worker_{worker_id} search time: {search_time}')
            
            # if worker_id >= 0:
                # print(f'worker_{worker_id} path found')

            try:
                # self.path = self.extract_path(self.PARENT)
                path = self.extract_path(self.PARENT)
            except:
                # print('PATH NOT FOUND!')
                # self.path = [s_start]
                path = [s_start]

        return path, self.CLOSED

    def get_neighbor(self, s):
        """
        find neighbors of state s that not in obstacles.
        :param s: state
        :return: neighbors
        """

        return [(s[0] + u[0], s[1] + u[1]) for u in self.u_set]

    def cost(self, s_start, s_goal):
        """
        Calculate Cost for this motion
        :param s_start: starting node
        :param s_goal: end node
        :return:  Cost for this motion
        :note: Cost function could be more complicate!
        """

        if self.is_collision(s_start, s_goal):
            return math.inf

        return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

    def is_collision(self, s_start, s_end):
        """
        check if the line segment (s_start, s_end) is collision.
        :param s_start: start node
        :param s_end: end node
        :return: True: is collision / False: not collision
        """

        if s_start in self.extended_obs or s_end in self.extended_obs:
            return True

        if s_start[0] < 0 or s_start[0] > self.width or s_start[1] < 0 or s_start[1] > self.height:
            return True
            
        if s_end[0] < 0 or s_end[0] > self.width or s_end[1] < 0 or s_end[1] > self.height:
            return True
            
        if s_start[0] != s_end[0] and s_start[1] != s_end[1]:
            if s_end[0] - s_start[0] == s_start[1] - s_end[1]:
                s1 = (min(s_start[0], s_end[0]), min(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
            else:
                s1 = (min(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), min(s_start[1], s_end[1]))

            if s1 in self.extended_obs or s2 in self.extended_obs:
                return True

        return False

    def f_value(self, s, e=2.5):
        """
        f = g + h. (g: Cost to come, h: heuristic value)
        :param s: current state
        :param e: hyperparameter
        :return: f
        """

        return self.g[s] + e * self.heuristic(s)

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

    def heuristic(self, s):
        """
        Calculate heuristic.
        :param s: current node (state)
        :return: heuristic function value
        """

        heuristic_type = self.heuristic_type  # heuristic type
        goal = self.s_goal  # goal node

        if heuristic_type == "manhattan":
            return abs(goal[0] - s[0]) + abs(goal[1] - s[1])
        else:
            return math.hypot(goal[0] - s[0], goal[1] - s[1])

