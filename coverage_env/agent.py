from copy import deepcopy
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
        x: float, y: float,
        vx: float, vy: float,
        vmax: float, 
        sen_range: int, comm_range: int
    ):
        """_summary_

        Args:
            idx (int): agent id
            step_size (float): simulation time step
            tau (float): time-delay in first-order dynamic
            DOF: the dimension of freedom
            x (float): x position
            y (float): y position
            vx (float): velocity in x-axis
            vy (float): velocity in y-axis
            theta (float): the angle between velocity vector and x-axis
            v_max (float): the max velocity
            sen_range (int): sensor range
            comm_range (int): communication range
        """
        self.step_size = step_size
        self.tau = tau
        
        self.x = x
        self.y = y
        
        self.vx = vx
        self.vy = vy
        self.vmax = vmax
        
        self.sen_range = sen_range
        self.comm_range = comm_range

        self.active = True
        self.slam = Sensor(
            num_beams=36,
            radius=self.sen_range,
            horizontal_fov=2 * np.pi
        )
        
        self.theta_list = [i * np.pi / 4 for i in range(0, 8)]
        
        self.actions_mat = [[np.cos(t) * self.vmax, np.sin(t) * self.vmax] for t in self.theta_list]
        self.actions_mat.append([0., 0.])

    def step(self, action: int):
        """Transform discrete action to desired velocity
        Args:
            action (int): an int belong to [0, 1, 2, ..., 8] - 2D
        """
        desired_velocity = self.actions_mat[action]
        next_state = self.dynamic(u=desired_velocity)
        return next_state
        
    def apply_update(self, next_state):
        if self.active:
            self.x, self.y, self.vx, self.vy, self.theta = next_state

    def dynamic(self, u: float = 0):
        """The dynamic of the agent is considered as a 1-order system with 2/3 DOF.
        The input dimension is the same as the state dimension.

        Args:
            u (float): The desired velocity.
            # DOF (int, optional): Degree of freedom. Defaults to 2.
        """
        k1vx = (u[0] - self.vx) / self.tau
        k2vx = (u[0] - (self.vx + self.step_size * k1vx / 2)) / self.tau
        k3vx = (u[0] - (self.vx + self.step_size * k2vx / 2)) / self.tau
        k4vx = (u[0] - (self.vx + self.step_size * k3vx)) / self.tau
        vx = self.vx + (k1vx + 2 * k2vx + 2 * k3vx + k4vx) * self.step_size / 6
        k1vy = (u[1] - self.vy) / self.tau
        k2vy = (u[1] - (self.vy + self.step_size * k1vy / 2)) / self.tau
        k3vy = (u[1] - (self.vy + self.step_size * k2vy / 2)) / self.tau
        k4vy = (u[1] - (self.vy + self.step_size * k3vy)) / self.tau
        vy = self.vy + (k1vy + 2 * k2vy + 2 * k3vy + k4vy) * self.step_size / 6

        v = np.linalg.norm([vx, vy])
        v = np.clip(v, 0, self.vmax)
        
        if math.isclose(v, 0, rel_tol=1e-3):
            theta = 0
        else:
            theta = np.sign(vx) * np.arccos(vx / v)
        
        x = self.x + vx * self.step_size
        y = self.y + vy * self.step_size
        
        return [x, y, vx, vy, theta]

    def dead(self):
        self.active = False
        self.x = -1000
        self.y = -1000
        self.vx = 0
        self.vy = 0
        self.theta = 0


class Navigator(Agent):
    """_summary_
    Providing ability of sensing
    """
    def __init__(
        self, 
        time_step: float, tau: float, DOF: int,
        x: float, y: float, z: float, 
        v: float, theta: float, phi: float, 
        d_v_lmt: float, d_theta_lmt: float, d_phi_lmt: float, v_max: float, 
        sen_range: int, comm_range: int, global_map: OccupiedGridMap,
    ):
        super().__init__(
            time_step=time_step, tau=tau, DOF=DOF,
            x=x, y=y, z=z,
            v=v, theta=theta, phi=phi, 
            d_v_lmt=d_v_lmt, d_theta_lmt=d_theta_lmt, d_phi_lmt=d_phi_lmt, v_max=v_max,
            sen_range=sen_range, comm_range=comm_range, global_map=global_map
        )

        # TODO
        self.slam = Sensor(view_range=sen_range, box_width=1, map_size=[global_map.x_dim, global_map.y_dim])
    
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
        x: float, y: float, 
        vx: float, vy: float,
        config: Namespace,
    ):
        super().__init__(
            x=x, y=y, vx=vx, vy=vy, 
            vmax=config.vmax, step_size=config.step_size, tau=config.tau,
            sen_range=config.sen_range, comm_range=config.comm_range
        )
        
    def specific_operation(self):
        pass
        
    
class Client(Agent):
    """_summary_
    Moving target requiring communication signal from the built network, with ability of path planning
    """
    def __init__(
        self, 
        idx: int, time_step: float, tau: float, DOF: int,
        x: float, y: float, z: float, 
        v: float, theta: float, phi: float, 
        d_v_lmt: float, d_theta_lmt: float, d_phi_lmt: float, v_max: float, 
        sen_range: int, comm_range: int, global_map: OccupiedGridMap,
        target: list
    ):
        """_summary_

        Args:
            target (list): a 1x3 list representing the xyz position of the target
        """
        super().__init__(
            idx=idx, time_step=time_step, tau=tau, DOF=DOF,
            x=x, y=y, z=z,
            v=v, theta=theta, phi=phi, 
            d_v_lmt=d_v_lmt, d_theta_lmt=d_theta_lmt, d_phi_lmt=d_phi_lmt, v_max=v_max,
            sen_range=sen_range, comm_range=comm_range, global_map=global_map
        )
        self.target = target
        self.path = []
        self.astar = AStar_2D(width=global_map.x_dim, height=global_map.y_dim)
    
    def replan(self, target: list, map: OccupiedGridMap):
        """_summary_

        Args:
            target (list): a 1x3 list representing the xyz position of the target
            map (OccupancyGridMap): 

        Returns:
            self.path (list): a list of waypoint
        """
        self.target = target
        self.path = self.astar.searching(
            s_start=(self.x, self.y, self.z),
            s_goal=target,
            obs=map.obs
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