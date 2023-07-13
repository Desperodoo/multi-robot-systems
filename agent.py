import numpy as np
from grid import OccupancyGridMap, SLAM
from abc import abstractmethod
from astar import AStar_2D
import math
from Occupied_Grid_Map import AStar_3D

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
        idx: int, time_step: float, tau: float, DOF: int,
        x: float, y: float, z: float, 
        v: np.ndarray, theta: float, phi: float, 
        d_v_lmt: float, d_theta_lmt: float, d_phi_lmt: float, v_max: float, 
        sen_range: int, comm_range: int, global_map: OccupancyGridMap
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
            v (np.ndarray): [vx, vy, vz]
            theta (float): the angle between velocity vector and z-axis
            phi (float): the angle between the x-y plane projection of velocity vector and x-axis
            d_v_lmt (float): the limitation of dot velocity
            d_theta_lmt (float): the limitation of dot theta
            d_phi_lmt (float): the limitation of dot phi
            v_max (float): the max velocity
            sen_range (int): sensor range
            comm_range (int): communication range
            global_map (OccupancyGridMap): map
        """
        self.time_step = time_step
        self.tau = tau
        self.idx = idx
        self.DOF = DOF
        
        self.x = x
        self.y = y
        self.z = z
        
        self.v = v
        self.theta = theta
        self.phi = phi
        
        self.d_v_lmt = d_v_lmt
        self.d_theta_lmt = d_theta_lmt
        self.d_phi_lmt = d_phi_lmt
        self.v_max = v_max
        
        self.sensor_range = sen_range
        self.comm_range = comm_range
        self.grid_map = global_map

        self.active = True
        self.slam = SLAM(global_map=global_map, view_range=self.sensor_range)

    def step(self, action: int):
        """Transform discrete action to desired velocity

        Args:
            action (int): an int belong to [0, 1, 2, ..., 8] - 2D, or [0, 1, 2, ..., 26] - 3D
        """
        if self.DOF == 2:
            actions_mat = [[i, j] for i in range(-1, 2, 1) for j in range(-1, 2, 1)]
        else:
            actions_mat = [[i, j, k] for i in range(-1, 2, 1) for j in range(-1, 2, 1) for k in range(-1, 2, 1)]
        
        desired_velocity = actions_mat[action]
        self.dynamic(u=desired_velocity)    

    def dynamic(self, u: float = 0, order: int = 1):
        """The dynamic of the agent is considered as a 1-order system with 2/3 DOF.
        The input dimension is the same as the state dimension.

        Args:
            u (float): The desired velocity.
            order (int, optional): The order of the response characteristic of the velocity. Defaults to 1.
            # DOF (int, optional): Degree of freedom. Defaults to 2.
        """
        self.v = u * (1 - np.exp(self.time_step / self.tau)) + self.v * np.exp(self.time_step / self.tau)
        self.v = np.clip(self.v, -self.v_max, self.v_max)
        v_abs = np.linalg.norm(self.v)
        
        if math.isclose(v_abs, 0, rel_tol=1e-3):
            self.theta = 0
            self.phi = 0
        else:
            vx, vy, vz = self.v
            self.theta = np.arccos(vz / (v_abs))
            v_proj_xy = v_abs * np.sin(self.theta)
            if math.isclose(v_proj_xy, 0, rel_tol=1e-3):
                self.phi = 0
            else:
                self.phi = np.sign(vy) * np.arccos(vx / v_proj_xy)
        
        self.x += self.v[0] * self.time_step
        self.y += self.v[1] * self.time_step
        self.z += self.v[2] * self.time_step
        

class Navigator(Agent):
    def __init__(
        self, 
        idx: int, time_step: float, tau: float, DOF: int,
        x: float, y: float, z: float, 
        v: float, theta: float, phi: float, 
        d_v_lmt: float, d_theta_lmt: float, d_phi_lmt: float, v_max: float, 
        sen_range: int, comm_range: int, global_map: OccupancyGridMap,
    ):
        super().__init__(
            idx=idx, time_step=time_step, tau=tau, DOF=DOF,
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
            

class Pursuer(Navigator):
    def __init__(
        self, 
        idx: int, time_step: float, tau: float, DOF: int,
        x: float, y: float, z: float, 
        v: float, theta: float, phi: float, 
        d_v_lmt: float, d_theta_lmt: float, d_phi_lmt: float, v_max: float, 
        sen_range: int, comm_range: int, global_map: OccupancyGridMap,
        raser_map: list  
    ):
        super().__init__(
            idx=idx, time_step=time_step, tau=tau, DOF=DOF,
            x=x, y=y, z=z,
            v=v, theta=theta, phi=phi, 
            d_v_lmt=d_v_lmt, d_theta_lmt=d_theta_lmt, d_phi_lmt=d_phi_lmt, v_max=v_max,
            sen_range=sen_range, comm_range=comm_range, global_map=global_map
        )
        self.is_pursuer = True


class Evader(Agent):
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
        self.is_pursuer = False
        self.target = target
        if DOF == 2:
            self.astar = AStar_2D(width=global_map.x_dim, height=global_map.y_dim)
        else:
            self.astar = AStar_3D()
        
    def step(self, step_size, a):
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
            x = self.x + v * np.cos(self.phi) * step_size
            y = self.y + v * np.sin(self.phi) * step_size
            # TODO:
            if self.grid_map.in_bounds((int(x), int(y))):
                if self.grid_map.is_unoccupied((int(x), int(y))):
                    self.x = x
                    self.y = y
                    self.v = v

    def replan(self, p_states, time_step, worker_id):
        # return vertices and slam_map
        current_x = round(self.x)
        current_y = round(self.y)
        
        extend_dis = 2
        
        while extend_dis >= 2:
            slam_map, pred_map = self.slam.rescan(
                global_position=(current_x, current_y),
                moving_obstacles=p_states,
                time_step=time_step,
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
        
        
