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
from agent import *


class BaseEnv(metaclass=ABCMeta):
    def __init__(self, map_config, env_config, defender_config, attacker_config, sensor_config):
        # Agent config
        self.defender_class = eval(env_config.defender_class)
        self.attacker_class = eval(env_config.attacker_class)
        
        # Simulation config
        self.time_step = 0
        self.n_episode = 0
        self.max_steps = env_config.max_steps
        self.step_size = env_config.step_size

        self.num_target = env_config.num_target
        self.num_defender = env_config.num_defender
        self.num_attacker = env_config.num_attacker

        self.map_config = map_config
        self.env_config = env_config
        self.defender_config = defender_config
        self.attacker_config = attacker_config
        self.sensor_config = sensor_config
        
    def init_map(self):
        """parsed args 

        Args:
            map_config (Namespace): 
        """
        num = self.map_config.num_obstacle_block
        center = self.map_config.center
        variance = self.map_config.variance
        self.occupied_map = OccupiedGridMap(is3D=False, boundaries=self.map_config.map_size)
        self.occupied_map.initailize_obstacle(num=num, center=center, variance=variance)
        inflated_map = deepcopy(self.occupied_map)
        inflated_map.extended_obstacles(extend_dis=2)
        return inflated_map
    
    def init_target(self, inflated_map: OccupiedGridMap):
        """initialize the target position

        Args:
            num_target (int): number of target points
            inflated_map (OccupiedGridMap): obstacle inflation
        """
        self.target = list()
        map_size = inflated_map.boundaries
        width = map_size[0]
        height = map_size[1]
        while len(self.target) < self.num_target:
            target = (
                random.randint(0, width - 1),
                random.randint(0, height - 1)
            )
            # target = (0, 0)
            if inflated_map.is_unoccupied(target):
                self.target.append(target)
    
    def init_defender(self, min_dist: int, inflated_map: OccupiedGridMap):
        """initialize the defender

        Args:
            min_dist (int): configure the density of the collective.
            inflated_map (OccupiedGridMap): the inflated obstacle map
            defender_config (dict): the configuration of the defender group
        Returns:
            inflated_map: occupied grid map with the defender as moving obstacle, and all obstacles are inflated.
        """
        map_size = inflated_map.boundaries
        width = map_size[0]
        height = map_size[1]
        self.defender_list = list()
        position_list = list()
        while len(position_list) < self.num_defender:
            label = False  # whether the new position can be appended
            pos = tuple(np.random.rand(2) * np.array([width - 1, height - 1]))
            if inflated_map.is_unoccupied(pos=pos):
                if len(self.defender_list) == 0:
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
                        if dist < self.defender_config.comm_range:
                            connectivity += 1
                    
                    if (collision == 0) and (connectivity > 0) and (connectivity <= 2):
                        label = True
            
            if label:
                position_list.append(pos)
                agent = self.defender_class(
                    x=pos[0], y=pos[1],
                    vx=0., vy=0.,
                    config=self.defender_config                    
                )
                self.defender_list.append(agent)
                inflated_map.set_moving_obstacle([pos])
                inflated_map.extended_moving_obstacles(extend_dis=2)
        
        return inflated_map

    def init_attacker(self, inflated_map: OccupiedGridMap, is_percepted: bool, target_list: list):
        """initialize the attacker.

        Args:
            num_attacker (int): 
            inflated_map (OccupiedGridMap): occupied grid map with the defender as moving obstacle, and all obstacles are inflated.
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
                        if dist < self.defender_config.sen_range:
                            position_list.append(pos)
                            attacker = self.attacker_class(
                                x=pos[0], y=pos[1],
                                vx=0., vy=0.,
                                target=target_list[0],  # TODO: to be implemented
                                config=self.attacker_config 
                            )
                            self.attacker_list.append(attacker)
                            break
                else:
                    position_list.append(pos)
                    attacker = self.attacker_class(
                        x=pos[0], y=pos[1],
                        vx=0., vy=0.,
                        target=target_list[0],
                        config=self.attacker_config 
                    )
                    self.attacker_list.append(attacker)

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    def attacker_step(self, inflated_map):
        # Based On A-Star Algorithm
        for idx, attacker in enumerate(self.attacker_list):
            path, way_point = attacker.replan(inflated_map=inflated_map, time_step=self.time_step)
            phi = attacker.waypoint2phi(way_point)
            attacker.step(self.step_size, phi)
        return path

    @abstractmethod
    def get_done(self):
        pass

    @abstractmethod
    def defender_reward(self, agent_idx, is_pursuer=True):
        pass
    
    def get_reward(self):
        reward = []
        for defender in self.defender_list:
            reward.append(self.defender_reward(defender))
        return reward

    @abstractmethod
    def get_agent_state(self, is_pursuer, idx, relative=False):
        pass

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
        pass

            
        
        
        
            