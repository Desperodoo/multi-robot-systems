from copy import deepcopy
import numpy as np
from Occupied_Grid_Map import OccupiedGridMap
from abc import abstractmethod
from astar import AStar_2D
import math
from Sensor import Sensor
from argparse import Namespace


# Discrete Version
class Agent:
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


class Navigator(Agent):
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
            

class Pursuer(Agent):
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
        
    def find_attacker(self, occupied_grid_map: OccupiedGridMap, pos: tuple, attacker_pos: tuple):
        if np.linalg.norm([pos[0] - attacker_pos[0], pos[1] - attacker_pos[1]]) > self.sen_range:
            return [0]
        else:
            pos = np.array(occupied_grid_map.get_pos(pos))
            # TODO: get from OccupancyGridMap
            ray_indices = bresenham_line(pos[0], pos[1], attacker_pos[0], attacker_pos[1])
            for index in ray_indices:
                if not occupied_grid_map.in_bound(index):
                    print("Error: ray indice is not in bound!")
                if occupied_grid_map.get_map()[index] == 1:
                    return [0]
            return [1]


class Evader(Agent):
    def __init__(
        self, 
        x: float, y: float, 
        vx: float, vy: float, 
        target: tuple,
        config: Namespace
    ):
        """_summary_

        Args:
            target (list): a 1x3 list representing the xyz position of the target
        """
        super().__init__(
            x=x, y=y, vx=vx, vy=vy, 
            vmax=config.vmax, step_size=config.step_size, tau=config.tau,
            sen_range=config.sen_range, comm_range=config.comm_range
        )
        self.is_pursuer = False
        self.target = target
        self.astar = AStar_2D(width=config.x_dim, height=config.y_dim)
        self.extend_dis = config.extend_dis
        
    def step(self, action):
        # a belong to [-phi, phi]
        next_state = self.dynamic(u=action)
        return next_state

    def rescan(self, occupied_map: OccupiedGridMap, extend_dis: int, moving_obs: list, pos: tuple):
        """_summary_

        Args:
            pred_map: (OccupiedGridMap): the map with global information
            dynamic_map (OccupiedGridMap): the map with global obstacle information and local moving obstacle information
            extend_dis (int): for safe navigation
            moving_obs (list): position information of the moving obstacle
            pos (tuple): current position of the attacker
        Returns:
            OccupiedGridMap: _description_
        """
        # For visualization
        dynamic_map = deepcopy(occupied_map)
        dynamic_map.extended_obstacles(extend_dis=extend_dis)
        
        pred_map = deepcopy(dynamic_map)
        pred_map.set_moving_obstacle(pos=moving_obs)
        pred_map.extended_moving_obstacles(extend_dis=extend_dis)

        local_observation = dynamic_map.local_observation(global_position=pos, view_range=self.sen_range)
        for node, value in local_observation.items():
            if value == 0:  # unoccupied
                if not pred_map.is_unoccupied(node):
                    if (node in pred_map.moving_obstacles) or (node in pred_map.obstacles):
                        dynamic_map.set_obstacle(node)
                    else:
                        dynamic_map.set_extended_obstacle(node)
        return dynamic_map, pred_map
        
    def replan(self, moving_obs, occupied_map):
        current_x = round(self.x)
        current_y = round(self.y)
        
        extend_dis = self.extend_dis
        # TODO: to be adapted to Sensor class
        while extend_dis >= 0:
            dynamic_map, pred_map = self.rescan(
                occupied_map=occupied_map,
                pos=(current_x, current_y),
                moving_obs=moving_obs,
                extend_dis=extend_dis
            )

            path, _ = self.astar.searching(
                s_start=(current_x, current_y),
                s_goal=self.target,
                obs=dynamic_map.obstacles + dynamic_map.ex_obstacles,
            )

            if len(path) >= 2:
                break
            
            extend_dis -= 1

        if len(path) >= 2:
            next_way_point = path[-2]
        else:
            next_way_point = path[0]

        return path, next_way_point, pred_map

    def waypoint2phi(self, way_point):
        """
        :param way_point:
        :return phi: angle belong to (-pi, pi)
        """
        radius = np.linalg.norm([way_point[0] - self.x, way_point[1] - self.y])
        if math.isclose(radius, 0.0, abs_tol=0.01):
            phi = 0
        else:
            phi = np.sign(way_point[1] - self.y) * np.arccos((way_point[0] - self.x) / (radius + 1e-3))
        return phi
    
    
class Client(Agent):
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