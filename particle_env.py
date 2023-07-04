import time
import numpy as np
from numpy.linalg import norm
from astar import AStar
from copy import deepcopy
import random
import math
from abc import abstractmethod
from grid import OccupancyGridMap, SLAM, generate_obstacle_map
from pre_generate_map import get_raser_map


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

    # def rescan(self, x, y, grid_map, boundary_obstacles, evader_pos, max_boundary_obstacle_num):
    #     """_summary_

    #     Args:
    #         x (_type_): current x position
    #         y (_type_): current y position
    #         boundary_obstacles (_type_): /
    #         evader_pos (_type_): evader_position
    #         max_boundary_obstacle_num (_type_): /

    #     Returns:
    #         obj_adj: 1-d list
    #         evader_adj: 1-d list
    #     """
    #     # Attention! Attention! Here we use the maximum obstacle number.
        
    #     obstacle_adj = np.zeros(shape=(max_boundary_obstacle_num,))
    #     int_evader_pos = [[int(pos[0]), int(pos[1])] for pos in evader_pos]
    #     evader_num = len(evader_pos)
    #     evader_adj = np.zeros(shape=(evader_num,))
    #     local_obstacles = list()
    #     for obstacle in boundary_obstacles:
    #         if np.linalg.norm([obstacle[0] - x, obstacle[1] - y]) <= self.range:
    #             local_obstacles.append(obstacle)

    #     for beam in range(self.beam_num):
    #         beam_angle = beam * 2 * np.pi / self.beam_num
    #         beam_dir_x = np.cos(beam_angle)
    #         beam_dir_y = np.sin(beam_angle)
    #         for beam_range in range(self.range):
    #             beam_current_x = x + beam_range * beam_dir_x
    #             beam_current_y = y + beam_range * beam_dir_y
    #             if (beam_current_x < 0 or beam_current_x >= self.map_size[0] or beam_current_y < 0 or beam_current_y >= self.map_size[1]):
    #                 break
                
    #             beam_current_pos = [int(beam_current_x), int(beam_current_y)]
    #             if not grid_map.is_unoccupied(beam_current_pos):
    #                 idx = boundary_obstacles.index(beam_current_pos)
    #                 obstacle_adj[idx] = 1
    #                 break
                
    #             if beam_current_pos in int_evader_pos:
    #                 idx = int_evader_pos.index(beam_current_pos)
    #                 evader_adj[idx] = 1
    #     # for obstacle in local_obstacles:
    #     #     if not self.is_obstacle_occluded(obstacle[0], obstacle[1], x, y, local_obstacles):
    #     #         idx = boundary_obstacles.index(obstacle)
    #     #         obstacle_adj[idx] = 1
 
    #     # for idx, pos in enumerate(evader_pos):
    #     #     if (not self.is_obstacle_occluded(pos[0], pos[1], x, y, local_obstacles)) and \
    #     #             (np.linalg.norm([pos[0] - x, pos[1] - y]) <= self.range):
    #     #         evader_adj[idx] = 1

    #     obstacle_adj = obstacle_adj.tolist()
    #     evader_adj = evader_adj.tolist()
    #     return obstacle_adj, evader_adj

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
        # Attention! Attention! Here we use the maximum obstacle number.
        
        evader_num = len(evader_pos)
        evader_adj = np.zeros(shape=(evader_num,))
        # local_obstacles = list()
        # for obstacle in boundary_obstacles:
        #     if np.linalg.norm([obstacle[0] - x, obstacle[1] - y]) <= self.range:
        #         local_obstacles.append(obstacle)
 
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


    # def is_obstacle_occluded(self, tx, ty, x, y, obstacles):
    #     """_summary_

    #     Args:
    #         tx (int): x position of the checked obstacle
    #         ty (int): y position of ...
    #         x (int): x current position
    #         y (int): y ...
    #         obstacles (2-d list): local obstacles 

    #     Returns:
    #         occlued (bool): /
    #     """
    #     target_vertex = vertex(tx, ty, self.box_width)
    #     occluded_list = list()
    #     for (v_x, v_y) in target_vertex:
    #         line1 = [(v_x, v_y), (x, y)]
    #         occluded = False
    #         for obstacle in obstacles:
    #             # exclude obstacles that far away from the checked point
    #             # line_center = [(obstacle[0], obstacle[1]), (x, y)]
    #             # los_angle = self.arccos(line1=line1, line2=line_center)
    #             # if (los_angle > np.pi / 6) or ([tx, ty] == obstacle):
    #             #     continue
                
    #             if [tx, ty] == obstacle:
    #                 continue

    #             obstacle_vertex = vertex(obstacle[0], obstacle[1], self.box_width)
    #             margin = [
    #                 [obstacle_vertex[0], obstacle_vertex[1]],
    #                 [obstacle_vertex[1], obstacle_vertex[2]],
    #                 [obstacle_vertex[2], obstacle_vertex[3]],
    #                 [obstacle_vertex[3], obstacle_vertex[0]]
    #             ]
    #             for line2 in margin:
    #                 occluded = intersect(line1, line2)
    #                 if occluded:
    #                     break
    #             if occluded:
    #                 break
    #         occluded_list.append(occluded)
    #     if all(occluded_list):
    #         return True
    #     else:
    #         return False

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
                # exclude obstacles that far away from the checked point
                # line_center = [(obstacle[0], obstacle[1]), (x, y)]
                # los_angle = self.arccos(line1=line1, line2=line_center)
                # if (los_angle > np.pi / 6) or ([tx, ty] == obstacle):
                #     continue
                
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

    @staticmethod
    def arccos(line1, line2):
        line1 = np.array(line1)
        line2 = np.array(line2)
        vector1 = line1[0] - line1[1]
        vector2 = line2[0] - line2[1]
        cosin = np.dot(vector1, vector2) / (norm(vector1) * norm(vector2))
        if cosin >= 1 or cosin <= -1:
            print('cosin', cosin)
            cosin = np.clip(cosin, -1, 1)
        return np.arccos(cosin)
    

# Discrete Version
class Agent:
    def __init__(self, idx, x, y, phi, phi_lmt, v, v_max, v_lmt, sen_range, comm_range, global_map, time_step=None, tau=None):
        self.time_step = time_step
        self.tau = tau
        self.idx = idx
        self.x = x
        self.y = y
        self.phi = phi
        self.delta_phi_max = phi_lmt
        self.v = v
        self.v_max = v_max
        self.v_lmt = v_lmt
        self.sensor_range = sen_range
        self.comm_range = comm_range
        self.active = True
        self.grid_map = global_map
        self.slam = SLAM(global_map=global_map, view_range=self.sensor_range)

    @abstractmethod
    def step(self, step_size, a):
        raise NotImplementedError        


    def dynamic(self, u, order=1, DOF=2):
        """The dynamic of the agent is considered as a 1-order system with 2/3 DOF.
        The input dimension is the same as the state dimension.

        Args:
            u (float): The desired velocity.
            order (int, optional): The order of the response characteristic of the velocity. Defaults to 1.
            DOF (int, optional): Degree of freedom. Defaults to 2.
        """
        self.v = (u - self.v) * (1 - np.exp(self.time_step / self.tau)) + self.v * np.exp(self.time_step / self.tau)
        self.x += self.v[0] * self.time_step
        self.y += self.v[1] * self.time_step
        if DOF == 3:
            self.z += self.v[2] * self.time_step
            

class Pursuer(Agent):
    def __init__(self, idx, x, y, phi, phi_lmt, v, v_max, v_lmt, sen_range, comm_range, global_map, raser_map):
        super().__init__(
            idx=idx,
            x=x, y=y,
            phi=phi, phi_lmt=phi_lmt,
            v=v, v_max=v_max, v_lmt=v_lmt,
            sen_range=sen_range, comm_range=comm_range, global_map=global_map
        )
        self.is_pursuer = True
        self.radar = Radar(view_range=sen_range, box_width=1, map_size=[global_map.x_dim, global_map.y_dim])
        self.raser_map = raser_map

    def step(self, step_size, a):
        # a belong to [0, 1, 2, 3, 4, 5, 6, 7, 8]
        if a == 0:
            v = 0
        else:
            # v clip
            v = self.v_max
            delta_v = v - self.v
            delta_v = np.clip(delta_v, -self.v_lmt, self.v_lmt)
            v = self.v + delta_v
            # phi clip
            a = a * np.pi / 4
            if a > np.pi:
                a -= 2 * np.pi
            sign_a_phi = np.sign(a * self.phi)
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
            if self.grid_map.in_bounds((int(x), int(y))):
                self.x = x
                self.y = y
                self.v = v

    # def sensor(self, boundary_obstacles, evader_pos, max_boundary_obstacle_num):
    #     obstacle_adj, evader_adj = self.radar.rescan(
    #         x=int(self.x), 
    #         y=int(self.y), 
    #         grid_map=self.grid_map,
    #         boundary_obstacles=boundary_obstacles, 
    #         evader_pos=evader_pos, 
    #         max_boundary_obstacle_num=max_boundary_obstacle_num
    #     )
    #     # return obstacle_adj as list with shape of (obstacle_num, )
    #     if obstacle_adj != self.raser_map[int(self.x)][int(self.y)].tolist():
    #         print('obstacle_adj: ', obstacle_adj)
    #         print('raser_map: ', self.raser_map[int(self.x)][int(self.y)])
    #     assert obstacle_adj == self.raser_map[int(self.x)][int(self.y)].tolist()
    #     return obstacle_adj, evader_adj
    
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


class Evader(Agent):
    def __init__(self, idx, x, y, phi, phi_lmt, v, v_max, v_lmt, sen_range, comm_range, global_map, target):
        super().__init__(
            idx=idx,
            x=x, y=y,
            phi=phi, phi_lmt=phi_lmt,
            v=v, v_max=v_max, v_lmt=v_lmt,
            sen_range=sen_range, comm_range=comm_range, global_map=global_map
        )
        self.is_pursuer = False
        self.astar = AStar(width=global_map.x_dim, height=global_map.y_dim)
        self.target = target
        # self.path = []
        
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
            # self.path = path
        else:
            # path not found, path = [s_start]
            # if len(self.path) < 2:
            #     next_way_point = (self.x, self.y)
            # elif len(self.path) >= 3:
            #     self.path.pop()
            #     next_way_point = self.path[-2]
            # else:
            #     next_way_point = self.path[-2]
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


class ParticleEnv:
    # def __init__(self, cfg: dict):
    def __init__(self):
        cfg = {
            'p_vmax': 0.5,
            'e_vmax': 1,
            'width': 50,
            'height': 50,
            'box_width': 1,
            'p_sen_range': 6,
            'p_comm_range': 12,
            'e_sen_range': 6,
            'e_comm_range': 12,
            # 'target': [17.5, 17.5],
            'kill_radius': 0.5,
            'collision_radius': 1,
            'phi_lmt': np.pi / 4,
            'p_v_lmt': 0.2,
            'e_v_lmt': 0.4
        }
        self.p_obs_dim = 4
        self.e_obs_dim = 4
        self.env_name = 'ParticleEnvBoundGra'
        self.p_vmax = cfg['p_vmax']
        self.e_vmax = cfg['e_vmax']
        self.p_v_lmt = cfg['p_v_lmt']
        self.e_v_lmt = cfg['e_v_lmt']
        self.p_phi_lmt = cfg['phi_lmt']
        self.e_phi_lmt = cfg['phi_lmt']
        self.width = cfg['width']
        self.height = cfg['height']
        self.box_width = cfg['box_width']
        self.p_sen_range = cfg['p_sen_range']
        self.p_comm_range = cfg['p_comm_range']
        self.e_sen_range = cfg['e_sen_range']
        self.e_comm_range = cfg['e_comm_range']

        self.target = None
        self.kill_radius = cfg['kill_radius']
        self.collision_radius = cfg['collision_radius']
        self.random = np.random
        self.collision = None
        # self.random.seed(10086)

        self.state_dim = 4 + 4
        # action space
        self.action_dim = 1

        self.n_episode = 0
        self.episode_limit = 250
        self.shadow_epi = 1000
        self.target_return = 1000

        self.step_size = 0.5
        self.time_step = 0

        self.curriculum = False
        self.p_num = None
        self.e_num = None
        self.p_list = {}
        self.p_idx = []
        self.e_list = {}
        self.e_idx = []
        self.state = None

        self.global_map = None
        self.global_obstacles = list()
        self.obstacle_num = None
        
        self.block_num = 5
        self.obstacle_shape = (6, 7)
        self.max_boundary_obstacle_num = self.block_num * (6 * 7 - 4 * 5)
        self.boundary_obstacles = list()
        self.boundary_obstacle_num = None

    # def gen_init_p_pos(self):
    #     min_dist = 4
    #     sample = []
    #     while len(sample) < self.p_num:
    #         newp = np.random.rand(2) * np.array([self.width - 1, self.height - 1])
    #         if self.global_map.is_unoccupied(pos=newp):
    #             collision = 0
    #             for p in sample:
    #                 if np.linalg.norm(newp - p) < min_dist:
    #                     collision += 1
    #                     break
                    
    #             if collision == 0:
    #                 sample.append(newp)
    #     return sample
    
    def gen_init_p_pos(self):
        min_dist = 4
        sample = []
        while len(sample) < self.p_num:
            newp = np.random.rand(2) * np.array([self.width - 1, self.height - 1])
            if len(sample) == 0:
                sample.append(newp)
            else:
                if self.global_map.is_unoccupied(pos=newp):
                    dists = list()
                    collision = 0
                    connectivity = 0
                    for p in sample:
                        dists.append(np.linalg.norm(newp - p))
                    
                    for dist in dists:
                        if dist < min_dist:
                            collision += 1
                        if dist < self.p_comm_range:
                            connectivity += 1
                    
                    if (collision == 0) and (connectivity > 0) and (connectivity <= 2):
                        sample.append(newp)
        return sample

    def gen_init_e_pos(self, pos, dynamic_map):
        obs = dynamic_map.obstacles
        ex_obs = dynamic_map.ex_obstacles
        moving_obs = dynamic_map.moving_obstacles
        ex_moving_obs = dynamic_map.ex_moving_obstacles
        total_obs = obs + ex_obs + moving_obs + ex_moving_obs

        min_dist = 2
        sample = []
        while len(sample) < self.e_num:
            newp = np.random.normal(loc=pos, scale=5, size=(2,)).clip([0, 0], [self.width - 1, self.height - 1])
            collision = 0
            for obstacle in total_obs:
                if np.linalg.norm(newp - np.array(obstacle)) < min_dist:
                    collision += 1
                    break
            if collision == 0:
                sample.append(newp)
        return sample

    def reset(self, p_num=15, e_num=1, worker_id=-1, map_info=None):
        self.collision = False
        if map_info is None:
            # initialize global map
            obstacle_map, boundary_map, obstacles, boundary_obstacles = generate_obstacle_map(
                x_dim=self.width,
                y_dim=self.height,
                num_obstacles=self.block_num,
                obstacle_shape=self.obstacle_shape
            )
            # array, array, list

            self.global_map = OccupancyGridMap(
                x_dim=self.width,
                y_dim=self.height,
                new_ogrid=obstacle_map,
                obstacles=obstacles
            )
            
            self.boundary_map = OccupancyGridMap(
                x_dim=self.width,
                y_dim=self.height,
                new_ogrid=boundary_map,
                obstacles=boundary_obstacles
            )
            self.raser_map = np.array(get_raser_map(self.boundary_map, height=self.height, width=self.width, max_num_obstacle=self.max_boundary_obstacle_num))
        
        else:
            [obstacle_map, boundary_map, obstacles, boundary_obstacles, self.raser_map] = map_info
            
            self.global_map = OccupancyGridMap(
                x_dim=self.width,
                y_dim=self.height,
                new_ogrid=obstacle_map,
                obstacles=obstacles
            )
            
            self.boundary_map = OccupancyGridMap(
                x_dim=self.width,
                y_dim=self.height,
                new_ogrid=boundary_map,
                obstacles=boundary_obstacles
            )
            
        self.boundary_obstacles = boundary_obstacles
        self.global_obstacles = obstacles
        self.obstacle_num = len(obstacles)
        self.boundary_obstacle_num = len(boundary_obstacles)
        dynamic_map = deepcopy(self.global_map)
        dynamic_map.extended_obstacles()

        # initialize target
        self.target = None
        while self.target is None:
            target = (
                random.randint(0, self.width - 1),
                random.randint(0, self.height - 1)
            )
            if dynamic_map.is_unoccupied(target):
                self.target = target

        # initialize pursuer and evader
        self.p_num = p_num
        self.e_num = e_num
        self.e_list = {}
        self.e_idx = []
        self.p_list = {}
        self.p_idx = []

        self.time_step = 0
        self.n_episode += 1

        p_pos = self.gen_init_p_pos()  # list of array
        for i in range(self.p_num):
            self.p_list[f'{i}'] = Pursuer(
                idx=i,
                x=p_pos[i][0],
                y=p_pos[i][1],
                # phi=(2 * np.random.rand() - 1) * np.pi,
                phi=np.pi / 4,
                phi_lmt=self.p_phi_lmt,
                v=0,
                v_max=self.p_vmax,
                v_lmt=self.p_v_lmt,
                sen_range=self.p_sen_range,
                comm_range=self.p_comm_range,
                global_map=self.boundary_map,
                raser_map=self.raser_map,
                )
            self.p_idx.append(i)

        p_state = self.get_team_state(is_pursuer=True, active=True)
        dynamic_map.set_moving_obstacle(pos=p_state)
        dynamic_map.extended_moving_obstacles(extend_dis=2)

        e_pos = [self.width - 1 - self.target[0], self.height - 1 - self.target[1]]
        e_pos = self.gen_init_e_pos(e_pos, dynamic_map)

        for i in range(self.e_num):
            self.e_list[f'{i}'] = Evader(
                idx=i,
                x=e_pos[i][0],
                y=e_pos[i][1],
                phi=np.pi / 4,
                phi_lmt=self.e_phi_lmt,
                v=self.e_vmax,
                v_max=self.e_vmax,
                v_lmt=self.e_v_lmt,
                sen_range=self.e_sen_range,
                comm_range=self.e_comm_range,
                global_map=deepcopy(self.global_map),
                target=self.target
                )
            self.e_idx.append(i)

    def step(self, action):
        self.time_step += 1
        idx = 0
        for pursuer_idx in self.p_idx:
            pur = self.p_list[f'{pursuer_idx}']
            pur.step(self.step_size, action[idx])
            idx += 1

        reward = self.reward(True)
        self.update_agent_active()
        active = self.get_active()
        done = True if self.get_done() or self.time_step >= self.episode_limit else False

        return reward, done, active

    def evader_step(self, worker_id=-1):
        # Based On Dynamic D Star Algorithm
        p_state = self.get_team_state(is_pursuer=True, active=True)
        for evader_idx in self.e_idx:
            evader = self.e_list[f'{evader_idx}']
            path, way_point, pred_map = evader.replan(p_states=p_state, time_step=self.time_step, worker_id=worker_id)
            # if worker_id >= 0:
            #     print(f'worker_{worker_id} replan')

            phi = evader.waypoint2phi(way_point)
            evader.step(self.step_size, phi)
        return path, pred_map

    def get_done(self):
        for idx in self.e_idx:
            agent = self.e_list[f'{idx}']
            if np.linalg.norm([agent.x - self.target[0], agent.y - self.target[1]]) <= self.kill_radius:
                return True

        p_alive, e_alive = 0, 0
        for idx in self.p_idx:
            if self.p_list[f'{idx}'].active:
                p_alive += 1
        if p_alive == 0:
            return True

        for idx in self.e_idx:
            if self.e_list[f'{idx}'].active:
                e_alive += 1
        if e_alive == 0:
            return True

        return False

    def get_active(self):
        active = []
        for idx in self.p_idx:
            active.append(1 if self.p_list[f'{idx}'].active else 0)
        return active

    def agent_reward(self, agent_idx, is_pursuer=True):
        reward = 0
        is_collision = self.collision_detection(agent_idx=agent_idx, is_pursuer=is_pursuer, obstacle_type='evaders')
        reward += sum(is_collision) * 1

        inner_collision = self.collision_detection(agent_idx=agent_idx, is_pursuer=is_pursuer, obstacle_type='pursuers')
        reward -= (sum(inner_collision) - 1) * 1

        obstacle_collision = self.collision_detection(agent_idx=agent_idx, is_pursuer=is_pursuer, obstacle_type='static_obstacles')
        reward -= (sum(obstacle_collision)) * 1
        
        if sum(obstacle_collision) + sum(inner_collision) - 1 > 0:
            self.collision = True
        return reward

    def reward(self, is_pursuer):
        reward = []
        for idx in self.p_idx:
            if self.p_list[f'{idx}'].active:
                reward.append(self.agent_reward(idx, is_pursuer))
            else:
                reward.append(0)
        return reward

    def update_agent_active(self):
        p_idx_list = list()
        p_alive_list = list()
        for idx in self.p_idx:
            if self.p_list[f'{idx}'].active:
                p_idx_list.append(idx)
                p_p_collision = self.collision_detection(agent_idx=idx, is_pursuer=True, obstacle_type='pursuers')
                p_e_collision = self.collision_detection(agent_idx=idx, is_pursuer=True, obstacle_type='evaders')
                p_o_collision = self.collision_detection(agent_idx=idx, is_pursuer=True, obstacle_type='static_obstacles')
                p_alive_list.append(bool(sum(p_p_collision) - 1 + sum(p_e_collision) + sum(p_o_collision)))

        e_idx_list = list()
        e_alive_list = list()
        for idx in self.e_idx:
            if self.e_list[f'{idx}'].active:
                e_idx_list.append(idx)
                e_e_collision = self.collision_detection(agent_idx=idx, is_pursuer=False, obstacle_type='evaders')
                e_p_collision = self.collision_detection(agent_idx=idx, is_pursuer=False, obstacle_type='pursuers')
                e_o_collision = self.collision_detection(agent_idx=idx, is_pursuer=False, obstacle_type='static_obstacles')
                e_alive_list.append(bool(sum(e_e_collision) - 1 + sum(e_p_collision) + sum(e_o_collision)))
        
        for i, idx in enumerate(p_idx_list):
            agent = self.p_list[f'{idx}']
            if p_alive_list[i]:
                agent.active = False
                agent.x = 1000
                agent.y = 1000
                agent.phi = 0
                agent.v = 0

        for i, idx in enumerate(e_idx_list):
            agent = self.e_list[f'{idx}']
            if e_alive_list[i]:
                agent.active = False
                agent.x = 1000
                agent.y = 1000
                agent.phi = 0
                agent.v = 0

    def get_agent_state(self, is_pursuer, idx, relative=False):
        agent = self.p_list[f'{idx}'] if is_pursuer else self.e_list[f'{idx}']
        if relative:
            phi = agent.phi
            v = agent.v
            vx = v * np.cos(phi)
            vy = v * np.sin(phi)
            return [agent.x, agent.y, vx, vy]
        else:
            return [agent.x, agent.y, agent.phi, agent.v]

    def get_team_state(self, is_pursuer, active=True, relative=False):
        if active:
            team_state = list()
            idx_list = self.p_idx if is_pursuer else self.e_idx
            agent_list = self.p_list if is_pursuer else self.e_list
            for idx in idx_list:
                if agent_list[f'{idx}'].active:
                    team_state.append(self.get_agent_state(is_pursuer, idx, relative=relative))
        else:
            idx_list = self.p_idx if is_pursuer else self.e_idx
            team_state = [self.get_agent_state(is_pursuer, idx, relative=relative) for idx in idx_list]
        return team_state

    def communicate(self):
        """
        the obstacles have no impact on the communication between agents
        :return: adj_mat: the adjacent matrix of the agents
        """
        p_states = self.get_team_state(is_pursuer=True, active=False)
        adj_mat = np.zeros(shape=(self.p_num, self.p_num))
        for i, item_i in enumerate(p_states):
            agent_list = self.p_list
            if agent_list[f'{i}'].active:
                for j, item_j in enumerate(p_states):
                    if agent_list[f'{j}'].active:
                        if np.linalg.norm([item_i[0] - item_j[0], item_i[1] - item_j[1]]) <= self.p_comm_range:
                            adj_mat[i, j] = 1
        adj_mat = adj_mat.tolist()
        return adj_mat

    def sensor(self, evader_pos):
        obstacle_adj_list = list()
        evader_adj_list = list()
        for idx in self.p_idx:
            pursuer = self.p_list[f'{idx}']
            obstacle_adj, evader_adj = pursuer.sensor(
                boundary_obstacles=self.boundary_obstacles,
                evader_pos=evader_pos, 
                max_boundary_obstacle_num=self.max_boundary_obstacle_num
            )
            obstacle_adj_list.append(obstacle_adj)
            evader_adj_list.append(evader_adj)
        return obstacle_adj_list, evader_adj_list

    def collision_detection(self, agent_idx, is_pursuer, obstacle_type: str = 'static_obstacles'):
        agent_state = self.get_agent_state(is_pursuer, agent_idx)
        if obstacle_type == 'static_obstacles':
            obstacles = self.global_obstacles  # list containing coordination of static obstacles
        elif obstacle_type == 'pursuers':
            obstacles = self.get_team_state(is_pursuer=True, active=True)  # only active pursuer is referenced
        else:
            obstacles = self.get_team_state(is_pursuer=False, active=True)

        is_collision = list()
        for obstacle in obstacles:
            if np.linalg.norm([obstacle[0] - agent_state[0], obstacle[1] - agent_state[1]]) <= self.kill_radius:
                is_collision.append(1)
            else:
                is_collision.append(0)

        return is_collision
