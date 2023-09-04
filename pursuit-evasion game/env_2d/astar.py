import math
import heapq


class AStar_2D:
    """AStar set the cost + heuristics as the priority
    """
    def __init__(self, width, height, heuristic_type="manhattan"):
        self.heuristic_type = heuristic_type

        self.u_set = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                      (1, 0), (1, -1), (0, -1), (-1, -1)]
        self.action_set = {(-1,0) : 0, (-1,1) : 1, (0, 1) : 2, (1,1) : 3,
                           (1,0) : 4, (1,-1) : 5, (0, -1) : 6, (-1, -1) : 7}
        self.s_start = None
        self.s_goal = None
        self.obs = None  # position of obstacles
        self.OPEN = None  # priority queue / OPEN set
        self.CLOSED = None  # CLOSED set / VISITED order
        self.PARENT = None  # recorded parent
        self.g = None  # cost to come
        # self.path = None
        self.width = width
        self.height = height

    def searching(self, s_start: tuple, s_goal: tuple, obs):
        """
        A_star Searching.
        :return: path, visited order
        """
        self.s_start = s_start

        self.s_goal = s_goal
        self.obs = obs  # position of obstacles
        self.OPEN = []  # priority queue / OPEN set
        self.CLOSED = []  # CLOSED set / VISITED order
        self.PARENT = dict()  # recorded parent
        self.g = dict()  # cost to come

        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0
        self.g[self.s_goal] = math.inf
        heapq.heappush(self.OPEN,
                       (self.f_value(self.s_start), self.s_start))

        if s_goal in obs:
            path = [s_start]

        else:
            while self.OPEN:
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

            try:
                path = self.extract_path(self.PARENT)
            except:
                print('No path found')
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

        if s_start in self.obs:
            return True

        if s_start[0] < 0 or s_start[0] > self.width or s_start[1] < 0 or s_start[1] > self.height:
            return True
            
        if s_end[0] < 0 or s_end[0] > self.width or s_end[1] < 0 or s_end[1] > self.height:
            return True
            
        if s_end in self.obs:
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

        
        s = self.s_goal
        path = [s]
        while True:
            p = PARENT[s]
            key = (p[0] - s[0], p[1] - s[1])
            path.append(p)
            s = p
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


class AStar_3D:
    # default using manhattan distance
    def __init__(self,width,height,depth) -> None:
        
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
        self.width = width
        self.height = height
        self.depth = depth
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
        if s_start[0] < 0 or s_start[0] >= self.width or s_start[1] < 0 or s_start[1] >= self.height or s_start[2] < 0 or s_start[2] >= self.depth:
            return True
        if s_end[0] < 0 or s_end[0] >= self.width or s_end[1] < 0 or s_end[1] >= self.height or s_end[2] < 0 or s_end[2] >= self.depth:
            return True
        if s_end in self.extended_obs or s_end in self.obs:
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

    '''
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
'''
    def searching(self, s_start: tuple, s_goal: tuple, obs, extended_obs):
        self.s_start = s_start
        self.s_goal = s_goal
        self.extended_obs = extended_obs
        self.obs = obs
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
