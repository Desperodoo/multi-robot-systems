from __future__ import annotations
import time
import numpy as np
from skimage.segmentation import find_boundaries
from copy import deepcopy
import random
import math
from abc import abstractmethod
from .agent import Client, Relay
from utils.occupied_grid_map import OccupiedGridMap
from numba import njit, prange, float64, int32, boolean

def intersect(line1, line2) -> bool:
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


def get_boundary_map(occupied_grid_map: OccupiedGridMap, max_num) -> OccupiedGridMap:
    boundary_map = deepcopy(occupied_grid_map)
    boundary_map.grid_map = find_boundaries(occupied_grid_map.grid_map, mode='inner')
    boundary_map.obstacles = np.argwhere(boundary_map.grid_map == 1).tolist()
    obstacle_agent = deepcopy(boundary_map.obstacles)
    for obstacle in obstacle_agent:
        obstacle.append(0)
        obstacle.append(0)
    obstacle_agent = obstacle_agent + [[0, 0, 0, 0] for _ in range(max_num - len(boundary_map.obstacles))]
    boundary_map.obstacle_agent = obstacle_agent
    return boundary_map


@njit()    
def get_raser_map(boundary_map: np.ndarray, obstacles: np.ndarray, num_beams: int32, radius: int32, width: int32, height: int32, num_obstacles: int32) -> np.ndarray:
    """_summary_
    Returns:
        cache of raser sensing result, map[x][y][obstacle index] -> whether the obstacle is visible from (x,y)
    """
    hash_map = np.zeros((width, height, num_obstacles))
    blocks = width * height
    for idx in prange(blocks):
        x = idx // height
        y = idx % height
        for beam in range(num_beams):
            beam_angle = beam * 2 * np.pi / num_beams
            beam_dir_x = np.cos(beam_angle)
            beam_dir_y = np.sin(beam_angle)
            for beam_range in range(radius):
                beam_current_x = x + beam_range * beam_dir_x
                beam_current_y = y + beam_range * beam_dir_y
                if beam_current_x < 0 or beam_current_x >= width or beam_current_y < 0 or beam_current_y >= height:
                    break
                beam_current_x = int32(beam_current_x)
                beam_current_y = int32(beam_current_y)
                if boundary_map[beam_current_x, beam_current_y] == 1:
                    for idx in range(num_obstacles):
                        if obstacles[idx][0] == beam_current_x and obstacles[idx][1] == beam_current_y:
                            hash_map[x, y, idx] = 1
                            break
                    break
    return hash_map


class Radar:
    def __init__(self, view_range, box_width):
        self.range = view_range
        self.obstacles = None
        self.box_width = box_width

    def rescan(self, x, y, boundary_obstacles, evader_pos, max_boundary_obstacle_num):
        # Attention! Attention! Here we use the maximum obstacle number.
        
        obstacle_adj = np.zeros(shape=(max_boundary_obstacle_num,))
        evader_num = len(evader_pos)
        evader_adj = np.zeros(shape=(evader_num,))
        local_obstacles = list()
        for obstacle in boundary_obstacles:
            if np.linalg.norm([obstacle[0] - x, obstacle[1] - y]) <= self.range:
                local_obstacles.append(obstacle)
        # front_time = time.time()
        for obstacle in local_obstacles:
            # front_time2 = time.time()
            if not self.is_obstacle_occluded(obstacle[0], obstacle[1], x, y, local_obstacles):
                idx = boundary_obstacles.index(obstacle)
                obstacle_adj[idx] = 1
        #     print('time_cost_2: ', time.time() - front_time2)
        # print('time_cost_1: ', time.time() - front_time)      
        for idx, pos in enumerate(evader_pos):
            if (not self.is_obstacle_occluded(pos[0], pos[1], x, y, boundary_obstacles)) and \
                    (np.linalg.norm([pos[0] - x, pos[1] - y]) <= self.range):
                evader_adj[idx] = 1
        obstacle_adj = obstacle_adj.tolist()
        evader_adj = evader_adj.tolist()
        return obstacle_adj, evader_adj

    def is_obstacle_occluded(self, tx, ty, x, y, obstacles):
        target_vertex = vertex(tx, ty, self.box_width)
        occluded_list = list()
        for (v_x, v_y) in target_vertex:
            line1 = [(v_x, v_y), (x, y)]
            occluded = False
            for obstacle in obstacles:
                if [tx, ty] != obstacle:
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


class CoverageEnv:
    def __init__(self, cfg):
        cfg = {
            'p_vmax': 0.5,
            'c_vmax': 1,
            'width': 50,
            'height': 50,
            'box_width': 1,
            'p_sen_range': 6,
            'p_comm_range': 12,
            'c_sen_range': 6,
            'c_comm_range': 12,
            # 'target': [17.5, 17.5],
            'kill_radius': 0.5,
            'phi_lmt': np.pi / 4,
            'p_v_lmt': 0.2,
            'c_v_lmt': 0.4
        }
        self.p_obs_dim = 4
        self.c_obs_dim = 4
        self.env_name = 'ParticleEnvBoundGra'
        self.p_vmax = cfg['p_vmax']
        self.c_vmax = cfg['c_vmax']
        self.p_v_lmt = cfg['p_v_lmt']
        self.c_v_lmt = cfg['c_v_lmt']
        self.p_phi_lmt = cfg['phi_lmt']
        self.c_phi_lmt = cfg['phi_lmt']
        self.width = cfg['width']
        self.height = cfg['height']
        self.box_width = cfg['box_width']
        self.p_sen_range = cfg['p_sen_range']
        self.p_comm_range = cfg['p_comm_range']
        self.c_sen_range = cfg['c_sen_range']
        self.c_comm_range = cfg['c_comm_range']

        self.target = None
        self.kill_radius = cfg['kill_radius']
        self.random = np.random
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
        self.c_num = None
        self.p_list = {}
        self.p_idx = []
        self.c_list = {}
        self.c_idx = []
        self.state = None

        self.global_map = OccupiedGridMap(
            is3D=False,
            boundaries=(self.width, self.height)
        )
        self.global_obstacles = list()
        self.obstacle_num = None
        
        self.block_num = 5
        self.obstacle_shape = (6, 7)
        self.max_boundary_obstacle_num = self.block_num * (6 * 7 - 4 * 5)
        self.boundary_obstacles = list()
        self.boundary_obstacle_num = None
        
    def reset(self, p_num=3, c_num=6, worker_id=-1):
        self.time_step = 0
        self.n_episode += 1
        self.collision = False
        self.caught = False
        
        # Initialize map and corresponding sensor cache
        inflated_map = self.init_map()
        
        self.boundary_map = get_boundary_map(self.occupied_map, self.max_num_obstacle)
        self.inflated_boundary_map = deepcopy(self.boundary_map)
        self.inflated_boundary_map.extended_obstacles(extend_dis=1)
        self.num_obstacle = len(self.boundary_map.obstacles)
        
        self.raser_map = get_raser_map(
            boundary_map=self.boundary_map.grid_map,
            obstacles=np.array(self.boundary_map.obstacles), 
            num_beams=self.sensor_config.num_beams, 
            radius=self.relay_config.sen_range, 
            width=self.boundary_map.boundaries[0], 
            height=self.boundary_map.boundaries[1], 
            num_obstacles=len(self.boundary_map.obstacles)
        ).tolist()
        
        self.inflated_map = deepcopy(inflated_map)
        
        # Initialize entities
        inflated_map = self.init_relays(min_dist=4, inflated_map=inflated_map)
        self.init_attacker(inflated_map=inflated_map, is_percepted=True, target_list=self.target)

        # initialize client
        self.c_num = c_num
        self.c_list = dict()
        self.c_idx = list()
        self.initialize_client()

        # initialize pursuer and evader
        self.p_num = p_num
        self.p_list = {}
        self.p_idx = []
        self.initialize_server()
        
        self.time_step = 0
        self.n_episode += 1
        
    def step(self, action) -> tuple[list, bool, object]:
        next_state = list()
        rewards = list()
        self.time_step += 1
        for idx, relay in enumerate(self.relay_list):
            next_state.append(relay.step(action[idx]))
            
        for idx, relay in enumerate(self.relay_list):
            state = next_state[idx]
            if relay.active:
                reward, collision, caught = self.relay_reward(state, next_state)
                rewards.append(reward)
                if caught:
                    self.caught = caught
                if collision:
                    relay.dead()
                else:
                    relay.apply_update(state)
            else:
                rewards.append(0)
        
        done = self.is_done()
        info = None
        return rewards, done, info
    
    def get_active(self) -> list[int]:
        active = []
        for relay in self.relay_list:
            active.append(1 if relay.active else 0)
        return active
    
    def collision_detection(self, state, obstacle_type: str = 'obstacle', next_state: list = None):
        if obstacle_type == 'obstacle':
            collision = False
            for i in range(-1, 2, 1):
                for j in range(-1, 2, 1):
                    inflated_pos = (state[0] + i * self.relay_config.collision_radius, state[1] + j * self.relay_config.collision_radius)
                    if self.occupied_map.in_bound(inflated_pos):
                        collision = not self.occupied_map.is_unoccupied(inflated_pos)
                    if collision == True:
                        break
                if collision == True:
                    break
            return collision
        else:
            if obstacle_type == 'relay':
                obstacles = next_state
            else:  # attacker
                obstacles = self.get_state(agent_type=obstacle_type)
            collision = list()
            for obstacle in obstacles:
                if np.linalg.norm([obstacle[0] - state[0], obstacle[1] - state[1]]) <= self.relay_config.collision_radius:
                    collision.append(1)
                else:
                    collision.append(0)
            return collision

    def is_done(self) -> bool:
        return True if (self.time_step >= self.max_steps) or self.caught or (sum(self.get_active()) == 0) else False
        
    def get_agent_state(self, agent):
        return [agent.x, agent.y, agent.vx, agent.vy]

    def communicate(self):
        """
        the obstacles have no impact on the communication between agents
        :return: adj_mat: the adjacent matrix of the agents
        """
        active = self.get_active()
        states = self.get_state(agent_type='relay')
        adj_mat = np.zeros(shape=(self.num_relay, self.num_relay))
        for i, item_i in enumerate(states):
            for j, item_j in enumerate(states):
                if active[i] and active[j] and (i <= j) and (np.linalg.norm([item_i[0] - item_j[0], item_i[1] - item_j[1]]) <= self.relay_config.comm_range):
                    adj_mat[i, j] = 1
                    adj_mat[j, i] = 1
        adj_mat = adj_mat.tolist()
        return adj_mat
    
    def sensor(self):
        obstacle_adj_list = list()
        attacker_adj_list = list()
        for relay in self.relay_list:
            if relay.active:
                obstacle_adj = self.raser_map[int(relay.x)][int(relay.y)] + [0] * (self.max_num_obstacle - self.num_obstacle)
                attacker_adj = relay.find_attacker(
                    occupied_grid_map=self.occupied_map, 
                    pos=(round(relay.x), round(relay.y)),
                    attacker_pos=(round(self.attacker_list[0].x), round(self.attacker_list[0].y)),
                )
            else:
                obstacle_adj = [0] * self.max_num_obstacle
                attacker_adj = [0]
            obstacle_adj_list.append(obstacle_adj)
            attacker_adj_list.append(attacker_adj)
        return obstacle_adj_list, attacker_adj_list
    
    def init_map(self):
        """parsed args 

        Args:
            map_config (Namespace): 
        """
        num = self.map_config.num_obstacle_block
        center = self.map_config.center
        variance = self.map_config.variance
        self.occupied_map.initailize_obstacle(num=num, center=center, variance=variance)
        inflated_map = deepcopy(self.occupied_map)
        inflated_map.extended_obstacles(extend_dis=1)
        return inflated_map
    
    def init_relays(self, min_dist: int, inflated_map: OccupiedGridMap):
        """initialize the relay agents

        Args:
            min_dist (int): configure the density of the collective.
            inflated_map (OccupiedGridMap): the inflated obstacle map
            relay_config (dict): the configuration of the relay group
        Returns:
            inflated_map: occupied grid map with the relay as moving obstacle, and all obstacles are inflated.
        """
        map_size = inflated_map.boundaries
        width = map_size[0]
        height = map_size[1]
        self.relay_list = list()
        position_list = list()
        while len(position_list) < self.num_relay:
            label = False  # whether the new position can be appended
            pos = tuple(np.random.rand(2) * np.array([width - 1, height - 1]))
            if inflated_map.is_unoccupied(pos=pos):
                if len(self.relay_list) == 0:
                    label = True
                else:
                    dists = list()
                    collision = 0
                    connectivity = 0
                    for p in position_list:
                        dists.append(np.linalg.norm((pos[0] - p[0], pos[1] - p[1])))
                    
                    for dist in dists:
                        if dist < min_dist:
                            collision += 1
                        if dist < self.relay_config.comm_range:
                            connectivity += 1
                    
                    if (collision == 0) and (connectivity > 0) and (connectivity <= 2):
                        label = True
            
            if label:
                position_list.append(pos)
                agent = self.relay_class(
                    x=pos[0], y=pos[1],
                    vx=0., vy=0.,
                    config=self.relay_config                    
                )
                self.relay_list.append(agent)
                inflated_map.set_moving_obstacle([pos])
                inflated_map.extended_moving_obstacles(extend_dis=2)
        
        return inflated_map

    def init_attacker(self, inflated_map: OccupiedGridMap, is_percepted: bool, target_list: list):
        """initialize the attacker.

        Args:
            num_attacker (int): 
            inflated_map (OccupiedGridMap): occupied grid map with the relay as moving obstacle, and all obstacles are inflated.
            is_percepted (bool): whether the attacker should be percepted in the initial state, true in pursuit, false in navigation and coverage.
        """
        agent_block = inflated_map.moving_obstacles        
        position_list = list()
        self.attacker_list = list()
        map_size = inflated_map.boundaries
        width = map_size[0]
        height = map_size[1]
        while len(position_list) < self.num_attacker:
            pos = tuple(np.random.rand(2) * np.array([width - 1, height - 1]))
            if inflated_map.is_unoccupied(pos=pos):
                if is_percepted:
                    for block in agent_block:
                        dist = np.linalg.norm([block[0] - pos[0], block[1] - pos[1]])
                        if dist < self.relay_config.sen_range:
                            position_list.append(pos)
                            attacker = self.attacker_class(
                                x=pos[0], y=pos[1],
                                vx=0., vy=0.,
                                target=target_list[0],  # TODO: to be implemented
                                attacker_config=self.attacker_config,
                                map_config=self.map_config
                            )
                            self.attacker_list.append(attacker)
                            break
                else:
                    position_list.append(pos)
                    attacker = self.attacker_class(
                        x=pos[0], y=pos[1],
                        vx=0., vy=0.,
                        target=target_list[0],
                        attacker_config=self.attacker_config,
                        map_config=self.map_config
                    )
                    self.attacker_list.append(attacker)

    def demon(self):
        theta_list = [i * np.pi / 4 for i in range(0, 8)]
        actions_mat = [[np.cos(t), np.sin(t)] for t in theta_list]
        actions_mat.append([0., 0.])
        action_list = list()
        e_x = self.attacker_list[0].x
        e_y = self.attacker_list[0].y
        for relay in self.relay_list:
            x = relay.x
            y = relay.y
            radius = np.linalg.norm([x - e_x, y - e_y])
            if math.isclose(radius, 0.0, abs_tol=0.01):
                action = [0., 0.]
            else:
                phi = np.sign(e_y - y) * np.arccos((e_x - x) / (radius + 1e-3))
                action = [np.cos(phi), np.sin(phi)]
            middle_a = [np.linalg.norm((a[0] - action[0], a[1] - action[1])) for a in actions_mat]
            action_list.append(middle_a.index(min(middle_a)))
        return action_list
