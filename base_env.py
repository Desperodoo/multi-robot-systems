import time
import numpy as np
from numpy.linalg import norm
from skimage.segmentation import find_boundaries
from copy import deepcopy
import random
import math
from abc import abstractmethod
from Occupied_Grid_Map import OccupiedGridMap
from argparse import Namespace
from abc import ABCMeta,abstractmethod


class BaseEnv(metaclass=ABCMeta):
    def __init__(self, env_config):
        # Agent config
        self.vel_max_d = env_config.vel_max_d
        self.vel_max_a = env_config.vel_max_a
        self.tau = env_config.tau
        
        self.sen_range = env_config.sen_range
        self.comm_range = env_config.comm_range
        self.collision_radius = env_config.collision_radius
        # Simulation config
        self.time_step = 0
        self.n_episode = 0
        self.max_steps = env_config.max_steps
        self.step_size = env_config.step_size

        self.num_target = env_config.num_target
        self.num_defender = env_config.num_defender
        self.num_attacker = env_config.num_attacker

    def init_map(self, map_config: Namespace):
        """parsed args 

        Args:
            map_config (Namespace): 
        """
        num = map_config.obstacle_num
        center = map_config.center
        self.occupied_map = OccupiedGridMap(is3D=self.is3D,boundaries=self.map_size)
        self.occupied_map.initailize_obstacle(num=num, center=center)
        inflated_map = self.occupied_map.extended_obstacles(extend_range=2)
        return inflated_map
    
    def init_target(self, inflated_map: OccupiedGridMap):
        """initialize the target position

        Args:
            num_target (int): number of target points
            inflated_map (OccupiedGridMap): obstacle inflation
        """
        self.target = list()
        while len(target) < self.num_target:
            target = (
                random.randint(0, self.width - 1),
                random.randint(0, self.height - 1)
            )
            if inflated_map.is_unoccupied(target):
                self.target.append(target)
    
    def init_defender(self, min_dist: int, inflated_map: OccupiedGridMap):
        """initialize the defender

        Args:
            min_dist (int): configure the density of the collective.
            num_defender (int): 
            inflated_map (OccupiedGridMap): inflated obstacle map

        Returns:
            inflated_map: occupied grid map with the defender as moving obstacle, and all obstacles are inflated.
        """
        self.defender_list = list()
        position_list = list()
        while len(position_list) < self.num_defender:
            label = False
            pos = tuple(np.random.rand(2) * np.array([self.width - 1, self.height - 1]))
            if inflated_map.is_unoccupied(pos=pos):
                if len(self.defender_list) == 0:
                    label = True
                else:
                    dists = list()
                    collision = 0
                    connectivity = 0
                    for p in position_list:
                        dists.append(np.linalg.norm(pos - p))
                    
                    for dist in dists:
                        if dist < min_dist:
                            collision += 1
                        if dist < self.comm_range:
                            connectivity += 1
                    
                    if (collision == 0) and (connectivity > 0) and (connectivity <= 2):
                        label = True
            if label:
                position_list.append(pos)
                agent = Defender()
                self.defender_list.append(agent)
                inflated_map.set_moving_obstacle(pos)
                inflated_map.extended_moving_obstacles(pos)
        
        return inflated_map

    def init_attacker(self, inflated_map: OccupiedGridMap, is_percepted: bool):
        """initialize the attacker.

        Args:
            num_attacker (int): 
            inflated_map (OccupiedGridMap): occupied grid map with the defender as moving obstacle, and all obstacles are inflated.
            is_percepted (bool): whether the attacker should be percepted in the initial state, true in pursuit, false in navigation and coverage.
        """
        agent_block = inflated_map.moving_obstacles        
        position_list = list()
        self.attacker_list = list()
        
        while len(position_list) < self.num_attacker:
            pos = tuple(np.random.normal(loc=pos, scale=5, size=(2,)).clip([0, 0], [self.width - 1, self.height - 1]))
            if inflated_map.is_unoccupied(pos=pos):
                if is_percepted:
                    for block in agent_block:
                        dist = np.linalg.norm([block[0] - pos[0], block[1] - pos[1]])
                        if dist < self.sensor_range:
                            position_list.append(pos)
                            attacker = Attacker()
                            self.attacker_list.append(attacker)
                            break
    
    @abstractmethod
    def reset(self):
        self.time_step = 0
        self.n_episode += 1
        
        inflated_map = self.init_map(map_config=None)  # TODO: to be configured
        self.init_target(num_target=None, inflated_map=inflated_map)  # TODO: to be configured
        inflated_map = self.init_defender(num_defender=None, inflated_map=inflated_map)  # TODO: to be configured
        # TODO: the target should be assigned to the attacker manually
        self.init_attacker(num_attacker=None, inflated_map=inflated_map)  # TODO: to be configured

    @abstractmethod
    def step(self, action):
        self.time_step += 1
        idx = 0
        for pursuer_idx in self.p_idx:
            pur = self.p_list[f'{pursuer_idx}']
            pur.step(self.step_size, action[idx])
            idx += 1

        reward = self.reward(True)
        self.update_agent_active()
        done = True if self.get_done() or self.time_step >= self.episode_limit else False
        info = None

        return reward, done, info

    def attacker_step(self, inflated_map):
        # Based On A-Star Algorithm
        for idx, attacker in enumerate(self.attacker_list):
            path, way_point = attacker.replan(inflated_map=inflated_map, time_step=self.time_step)
            phi = attacker.waypoint2phi(way_point)
            attacker.step(self.step_size, phi)
        return path

    @abstractmethod
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

    @abstractmethod
    def defender_reward(self, agent_idx, is_pursuer=True):
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

    def get_reward(self):
        reward = []
        for defender in self.defender_list:
            reward.append(self.defender_reward(defender))
        return reward

    @abstractmethod
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

    def get_state(self, agent_type: str):
        """get states of the collective

        Args:
            relative (bool, optional): whether to transform the state from global coordinate to local coordinate. Defaults to True.

        Returns:
            state: list of tuple
        """
        agent_list = getattr(self, agent_type + '_list')
        state = [self.get_agent_state(agent) for agent in agent_list]
        return state

    @abstractmethod
    def communicate(self):
        """
        the obstacles have no impact on the communication between agents
        :return: adj_mat: the adjacent matrix of the agents
        """
        p_states = self.get_agent_state(relative=False)
        adj_mat = np.zeros(shape=(self.num_defender, self.num_defender))
        for i, item_i in enumerate(p_states):
            for j, item_j in enumerate(p_states):
                if np.linalg.norm([item_i[0] - item_j[0], item_i[1] - item_j[1]]) <= self.p_comm_range:
                    adj_mat[i, j] = 1
        adj_mat = adj_mat.tolist()
        return adj_mat

    def collision_detection(self, state, obstacle_type: str = 'obstacle'):
        if obstacle_type == 'obstacle':
            obstacles = self.global_obstacles  # list containing coordination of static obstacles
        else:
            obstacles = self.get_state(agent_type=obstacle_type)

        is_collision = list()
        for obstacle in obstacles:
            if np.linalg.norm([obstacle[0] - state[0], obstacle[1] - state[1]]) <= self.collision_radius:
                is_collision.append(1)
            else:
                is_collision.append(0)

        return is_collision
