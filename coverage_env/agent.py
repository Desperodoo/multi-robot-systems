from copy import deepcopy
from typing import List

from omegaconf import DictConfig

from utils.astar import AStar_2D
import numpy as np
from utils.occupied_grid_map import OccupiedGridMap
from utils.sensor import Sensor
from abc import abstractmethod
import math
from argparse import Namespace


# Discrete Version
class Agent:
    """_summary_
    Base class for agents, providing state transformation with dynamics
    """
    def __init__(
        self, 
        step_size: float, tau: float,
        pos: List[float],
        vel: List[float],
        v_max: float,
        sens_range: int, comm_range: int,
    ):
        """_summary_

        Args:
            step_size (float): simulation time step
            tau (float): time-delay in first-order dynamic
            theta (float): the angle between velocity vector and x-axis, calculated through (vx, vy)
            v_max (float): the max velocity
            sens_range (int): sensor range
            comm_range (int): communication range
        """
        self.step_size = step_size
        self.tau = tau
        
        self.pos = list(pos)
        self.vel = list(vel)
        self.v_max = v_max

        # TODO: whether set theta arbitrarily
        v = np.clip(np.linalg.norm(self.vel), 0, self.v_max)
        if math.isclose(v, 0, rel_tol=1e-3):
            self.theta = 0.
        else:
            self.theta = np.sign(self.vel[0]) * np.arccos(self.vel[0] / v)
        
        self.sens_range = sens_range
        self.comm_range = comm_range

        self.active = True
        self.slam = Sensor(
            num_beams=36,
            radius=self.sens_range,
            horizontal_fov=2 * np.pi
        )
        
        self.theta_list = [i * np.pi / 4 for i in range(0, 8)]  # discretize to 8 angles
        
        self.actions_mat = [[np.cos(t) * self.v_max, np.sin(t) * self.v_max] for t in self.theta_list]
        self.actions_mat.append([0., 0.])

    def step(self, action):
        # index
        next_state = self.dynamic(u=self.actions_mat[action])
        return next_state
        
    def apply_update(self, next_state: list):
        if self.active:
            # state: [x, y, vx, vy, theta]
            self.pos[0], self.pos[1], self.vel[0], self.vel[1], self.theta = tuple(next_state)

    def dynamic(self, u: List[float]):
        """The dynamic of the agent is considered as a 1-order system with 2/3 DOF.
        The input dimension is the same as the state dimension.

        Args:
            u (float): The desired velocity.
            # DOF (int, optional): Degree of freedom. Defaults to 2.
        """
        curr_vx, curr_vy = self.vel
        k1vx = (u[0] - curr_vx) / self.tau
        k2vx = (u[0] - (curr_vx + self.step_size * k1vx / 2)) / self.tau
        k3vx = (u[0] - (curr_vx + self.step_size * k2vx / 2)) / self.tau
        k4vx = (u[0] - (curr_vx + self.step_size * k3vx)) / self.tau
        vx = curr_vx + (k1vx + 2 * k2vx + 2 * k3vx + k4vx) * self.step_size / 6
        k1vy = (u[1] - curr_vy) / self.tau
        k2vy = (u[1] - (curr_vy + self.step_size * k1vy / 2)) / self.tau
        k3vy = (u[1] - (curr_vy + self.step_size * k2vy / 2)) / self.tau
        k4vy = (u[1] - (curr_vy + self.step_size * k3vy)) / self.tau
        vy = curr_vy + (k1vy + 2 * k2vy + 2 * k3vy + k4vy) * self.step_size / 6

        v = np.linalg.norm([vx, vy])
        v = np.clip(v, 0, self.v_max)
        
        if math.isclose(v, 0, rel_tol=1e-3):
            theta = 0
        else:
            theta = np.sign(vx) * np.arccos(vx / v)
        
        x = self.pos[0] + vx * self.step_size
        y = self.pos[1] + vy * self.step_size
        
        return [x, y, vx, vy, theta]

    def dead(self):
        self.active = False
        self.set_pos(-1000, 1000)
        self.set_vel(0, 0)
        self.theta = 0

    def set_vel(self, vx, vy):
        self.vel = [vx, vy]

    def set_pos(self, x, y):
        self.pos = [x, y]


class Navigator:
    """_summary_
    # TODO: is this class necessary?
    Providing ability of sensing
    """
    def __init__(
        self, 
        step_size: float, tau: float,
        sens_range: int, local_map: OccupiedGridMap,
    ):
        # TODO
        self.slam = Sensor(radius=sens_range, horizontal_fov=2 * np.pi, num_beams=60)
        self.map = local_map
    
    def sensor(self, boundary_obstacles, evader_pos, max_boundary_obstacle_num):
        # return obstacle_adj as list with shape of (obstacle_num, )
        if int(self.x) == 1000:
            obstacle_adj = [0] * max_boundary_obstacle_num
            evader_adj = [0] * len(evader_pos)
        else:
            obstacle_adj = self.raser_map[int(self.x)][int(self.y)].tolist()
            evader_adj = self.radar.rescan(
                x=int(self.x), 
                y=int(self.y), 
                boundary_obstacles=boundary_obstacles, 
                evader_pos=evader_pos, 
                obstacle_adj=obstacle_adj
            )
        return obstacle_adj, evader_adj
            

class Relay(Agent):
    """_summary_
    Relay agent(robot) aiming to provide communication connectivity
    """
    def __init__(
        self,
        pos: List[float],
        vel: List[float],
        config,  # TODO: what's the type
    ):
        super().__init__(
            pos=pos, vel=vel,
            v_max=config.v_max, step_size=config.step_size, tau=config.tau,
            sens_range=config.sens_range, comm_range=config.comm_range
        )

    def find_client(self, occupied_grid_map: OccupiedGridMap, pos: tuple, client_pos: tuple):
        if np.linalg.norm([pos[0] - client_pos[0], pos[1] - client_pos[1]]) > self.sens_range:
            return [0]
        else:
            pos = np.round(np.array(occupied_grid_map.get_pos(pos)))
            client_pos = np.round(np.array(occupied_grid_map.get_pos(client_pos)))
            # TODO: get from OccupancyGridMap
            ray_indices = bresenham_line(pos[0], pos[1], client_pos[0], client_pos[1])
            for index in ray_indices:
                if not occupied_grid_map.in_bound(index):
                    print("Error: ray index is not in bound!")
                if occupied_grid_map.get_map()[index] == 1:
                    return [0]
            return [1]
        
    
class Client(Agent, Navigator):
    """_summary_
    Moving target requiring communication signal from the built network, with ability of path planning
    """
    def __init__(
        self,
        pos: List[float],
        vel: List[float],
        config,  # TODO: what's the type
        global_map: OccupiedGridMap,
        target: list
    ):
        """_summary_

        Args:
            target (list): a 1x3 list representing the xyz position of the target
        """
        super().__init__(
            pos=pos, vel=vel,
            v_max=config.v_max, step_size=config.step_size, tau=config.tau,
            sens_range=config.sens_range, comm_range=config.comm_range
        )
        self.target = target
        self.path = []
        self.astar = AStar_2D(width=global_map.boundaries[0], height=global_map.boundaries[1])
    
    def replan(self, target: list, global_map: OccupiedGridMap):
        """_summary_

        Args:
            global_map:
            target (list): a 1x3 list representing the xyz position of the target
        Returns:
            self.path (list): a list of waypoint
        """
        self.target = target
        self.path = self.astar.searching(
            s_start=tuple(self.pos),
            s_goal=tuple(target),
            obs=global_map.obstacles,
            extended_obs=global_map.ex_obstacles
        )
        return self.path
        
        
def bresenham_line(x0, y0, x1, y1):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    err = dx - dy
    line = []

    while True:
        line.append((x0, y0))

        if math.isclose(x0, x1) and math.isclose(y0, y1):
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return line