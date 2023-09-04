import numpy as np
from grid import OccupancyGridMap
from abc import abstractmethod
from utils.astar import AStar_2D, AStar_3D
import math
from utils.Sensor import Sensor
from argparse import Namespace

def intersect(line1, line2):
    (a1, a2) = line1[0]
    (b1, b2) = line1[1]
    (c1, c2) = line2[0]
    (d1, d2) = line2[1]

    AB = (b1-a1, b2-a2)
    CD = (d1-c1, d2-c2)
    AC = (c1-a1, c2-a2)
    AD = (d1-a1, d2-a2)
    CA = (a1-c1, a2-c2)
    CB = (b1-c1, b2-c2)
    BC = (c1-b1, c2-b2)

    if (AB[0]*AC[1]-AB[1]*AC[0])*(AB[0]*AD[1]-AB[1]*AD[0]) < 0 and (CD[0]*CA[1]-CD[1]*CA[0])*(CD[0]*CB[1]-CD[1]*CB[0]) < 0:
        if (AB[0]*CD[1]-AB[1]*CD[0]) == 0:  # collineation
            if (BC[0]*CD[1]-BC[1]*CD[0]) == 0 and (a1 <= c1 <= b1 or a1 <= d1 <= b1):
                return True
            else:
                return False
        else:
            return True
    else:
        return False


def vertex(x, y, box_width):
    return [
        (x - box_width / 2, y - box_width / 2),
        (x - box_width / 2, y + box_width / 2),
        (x + box_width / 2, y + box_width / 2),
        (x + box_width / 2, y - box_width / 2)
    ]


class Radar:
    def __init__(self, view_range, box_width, map_size):
        self.range = view_range
        self.obstacles = None
        self.box_width = box_width
        self.beam_num = 40
        self.map_size = map_size

    def rescan(self, x, y, boundary_obstacles, evader_pos, obstacle_adj):
        """_summary_

        Args:
            x (_type_): current x position
            y (_type_): current y position
            boundary_obstacles (_type_): /
            evader_pos (_type_): evader_position
            max_boundary_obstacle_num (_type_): /

        Returns:
            obj_adj: 1-d list
            evader_adj: 1-d list
        """
        evader_num = len(evader_pos)
        evader_adj = np.zeros(shape=(evader_num,))
 
        for idx, pos in enumerate(evader_pos):
            if np.linalg.norm([pos[0] - x, pos[1] - y]) <= self.range:
                # since only one evader, so the local obstacle detection is move within this for loop
                local_obstacles = list()
                for idx_2, obstacle in enumerate(boundary_obstacles):
                    if obstacle_adj[idx_2] == 1:
                        local_obstacles.append(obstacle)
                if not self.is_evader_occluded(pos[0], pos[1], x, y, local_obstacles):
                    evader_adj[idx] = 1

        evader_adj = evader_adj.tolist()
        return evader_adj

    def is_evader_occluded(self, tx, ty, x, y, obstacles):
        """_summary_

        Args:
            tx (int): x position of the checked obstacle
            ty (int): y position of ...
            x (int): x current position
            y (int): y ...
            obstacles (2-d list): local obstacles 

        Returns:
            occlued (bool): /
        """
        target_vertex = [(tx, ty)]
        occluded_list = list()
        for (v_x, v_y) in target_vertex:
            line1 = [(v_x, v_y), (x, y)]
            occluded = False
            for obstacle in obstacles:
                if [tx, ty] == obstacle:
                    continue

                obstacle_vertex = vertex(obstacle[0], obstacle[1], self.box_width)
                margin = [
                    [obstacle_vertex[0], obstacle_vertex[1]],
                    [obstacle_vertex[1], obstacle_vertex[2]],
                    [obstacle_vertex[2], obstacle_vertex[3]],
                    [obstacle_vertex[3], obstacle_vertex[0]]
                ]
                for line2 in margin:
                    occluded = intersect(line1, line2)
                    if occluded:
                        break
                if occluded:
                    break
            occluded_list.append(occluded)
        if all(occluded_list):
            return True
        else:
            return False
        

# Discrete Version
class Agent:
    def __init__(
        self, 
        time_step: float, tau: float, DOF: int,
        x: float, y: float, z: float, 
        vx: float, vy: float, vz: float, 
        v_max: float, 
        sen_range: int, comm_range: int
    ):
        """_summary_

        Args:
            idx (int): agent id
            time_step (float): simulation time step
            tau (float): time-delay in first-order dynamic
            DOF: the dimension of freedom
            x (float): x position
            y (float): y position
            z (float): z position
            vx (float): velocity in x-axis
            vy (float): velocity in y-axis
            vz (float): velocity in z-axis
            theta (float): the angle between velocity vector and z-axis
            phi (float): the angle between the x-y plane projection of velocity vector and x-axis
            v_max (float): the max velocity
            sen_range (int): sensor range
            comm_range (int): communication range
        """
        self.time_step = time_step
        self.tau = tau
        self.DOF = DOF
        
        self.x = x
        self.y = y
        self.z = z
        
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.v_max = v_max
        
        self.sensor_range = sen_range
        self.comm_range = comm_range

        self.active = True
        self.slam = Sensor(
            num_beams=36,
            radius=self.sensor_range,
            horizontal_fov=2 * np.pi
        )
        
        self.theta_list = [i * np.pi / 4 for i in range(0, 8)]
        self.phi_list = [i * np.pi / 4 for i in range(-1, 2)]
        
        if self.DOF == 2:
            self.actions_mat = [[np.cos(t), np.sin(t)] for t in self.theta_list]
            self.actions_mat.append([0., 0.])
        else:
            self.actions_mat = [[np.cos(t), np.sin(t), np.cos(p)] for t in self.theta_list for p in self.phi_list]
            self.actions_mat = self.actions_mat + [[0., 0., 0.], [0., 0., 1.], [0., 0., -1]]


    def step(self, action: int):
        """Transform discrete action to desired velocity
        Args:
            action (int): an int belong to [0, 1, 2, ..., 8] - 2D, or [0, 1, 2, ..., 26] - 3D
        """
        desired_velocity = self.actions_mat[action] * self.v_max
        self.dynamic(u=desired_velocity)
        
    def apply_update(self, next_state):
        if self.DOF == 2:
            [self.x, self.y, self.vx, self.vy, self.theta] = next_state
        else:
            [self.x, self.y, self.z, self.vx, self.vy, self.vz, self.theta, self.phi] = next_state

    def dynamic(self, u: float = 0):
        """The dynamic of the agent is considered as a 1-order system with 2/3 DOF.
        The input dimension is the same as the state dimension.

        Args:
            u (float): The desired velocity.
            # DOF (int, optional): Degree of freedom. Defaults to 2.
        """
        vx = u[0] * (1 - np.exp(self.time_step / self.tau)) + self.vx * np.exp(self.time_step / self.tau)
        vy = u[1] * (1 - np.exp(self.time_step / self.tau)) + self.vy * np.exp(self.time_step / self.tau)
        if self.DOF == 3:
            vz = u[3] * (1 - np.exp(self.time_step / self.tau)) + self.vz * np.exp(self.time_step / self.tau)

        v = np.linalg.norm([vx, vy, vz])
        v = np.clip(v, 0, self.v_max)
        
        if math.isclose(v, 0, rel_tol=1e-3):
            theta = 0
            phi = 0
        else:
            theta = np.arccos(vz / (v))
            v_proj_xy = v * np.sin(theta)
            if math.isclose(v_proj_xy, 0, rel_tol=1e-3):
                phi = 0
            else:
                phi = np.sign(vy) * np.arccos(vx / v_proj_xy)
        
        x = self.x + vx * self.time_step
        y = self.y + vy * self.time_step
        z = self.z + vz * self.time_step
        
        return [x, y, z, vx, vy, vz, ]

class Navigator(Agent):
    def __init__(
        self, 
        time_step: float, tau: float, DOF: int,
        x: float, y: float, z: float, 
        v: float, theta: float, phi: float, 
        d_v_lmt: float, d_theta_lmt: float, d_phi_lmt: float, v_max: float, 
        sen_range: int, comm_range: int, global_map: OccupancyGridMap,
    ):
        super().__init__(
            time_step=time_step, tau=tau, DOF=DOF,
            x=x, y=y, z=z,
            v=v, theta=theta, phi=phi, 
            d_v_lmt=d_v_lmt, d_theta_lmt=d_theta_lmt, d_phi_lmt=d_phi_lmt, v_max=v_max,
            sen_range=sen_range, comm_range=comm_range, global_map=global_map
        )

        # TODO
        self.radar = Radar(view_range=sen_range, box_width=1, map_size=[global_map.x_dim, global_map.y_dim])
    
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
        x: float, y: float, z: float, 
        vx: float, vy: float, vz: float, 
        config: Namespace
    ):
        super().__init__(
            x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, 
            vmax=config.vmax, step_size=config.step_size, tau=config.tau, DOF=config.DOF,
            sen_range=config.sen_range, comm_range=config.comm_range
        )


class Evader(Agent):
    def __init__(
        self, 
        x: float, y: float, z: float, 
        vx: float, vy: float, vz: float, 
        target: tuple,
        config: Namespace
    ):
        """_summary_

        Args:
            target (list): a 1x3 list representing the xyz position of the target
        """
        super().__init__(
            x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, 
            vmax=config.vmax, time_step=config.time_step, tau=config.tau, DOF=config.DOF,
            sen_range=config.sen_range, comm_range=config.comm_range
        )
        self.is_pursuer = False
        self.target = target
        if self.DOF == 2:
            self.astar = AStar_2D(width=config.x_dim, height=config.y_dim)
        else:
            self.astar = AStar_3D()
        
    def step(self, a):
        # a belong to [-1, 1]
        v = self.v_max
        sign_a_phi = np.sign(a * self.phi)  # 判断同号异号
        if sign_a_phi >= 0:
            delta_phi = abs(a - self.phi)
            sign = np.sign(a - self.phi)
        else:
            if abs(a - self.phi) < 2 * np.pi - abs(a - self.phi):
                delta_phi = abs(a - self.phi)
                sign = np.sign(a - self.phi)
            else:
                delta_phi = 2 * np.pi - abs(a - self.phi)
                sign = -np.sign(a - self.phi)

        delta_phi = np.clip(delta_phi, 0, self.delta_phi_max)
        self.phi = self.phi + sign * delta_phi

        if self.phi > np.pi:
            self.phi -= 2 * np.pi
        elif self.phi < -np.pi:
            self.phi += 2 * np.pi

        if self.active:
            x = self.x + v * np.cos(self.phi) * self.step_size
            y = self.y + v * np.sin(self.phi) * self.step_size
            # TODO:
            if self.grid_map.in_bounds((int(x), int(y))):
                if self.grid_map.is_unoccupied((int(x), int(y))):
                    self.x = x
                    self.y = y
                    self.v = v

    def replan(self, p_states, worker_id):
        # return vertices and slam_map
        current_x = round(self.x)
        current_y = round(self.y)
        
        extend_dis = 2
        
        # TODO: to be adapted to Sensor class
        while extend_dis >= 2:
            slam_map, pred_map = self.slam.rescan(
                global_position=(current_x, current_y),
                moving_obstacles=p_states,
                time_step=self.step_size,
                extend_dis=extend_dis
            )

            path, _ = self.astar.searching(
                s_start=(current_x, current_y),
                s_goal=self.target,
                obs=slam_map.obstacles,
                extended_obs=slam_map.obstacles + slam_map.ex_obstacles,
                worker_id=worker_id
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
        sen_range: int, comm_range: int, global_map: OccupancyGridMap,
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
        if DOF == 2:
            self.astar = AStar_2D(width=global_map.x_dim, height=global_map.y_dim)
        else:
            self.astar = AStar_3D()
    
    def replan(self, target: list, map: OccupancyGridMap):
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
        
        
