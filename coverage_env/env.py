from __future__ import annotations
import time
import numpy as np
from skimage.segmentation import find_boundaries
from copy import deepcopy
import random
import math
from abc import abstractmethod
from coverage_env.agent import Client, Relay
from utils.occupied_grid_map import OccupiedGridMap
from numba import njit, prange, float64, int32, boolean


def intersect(line1, line2) -> bool:
    (a1, a2) = line1[0]
    (b1, b2) = line1[1]
    (c1, c2) = line2[0]
    (d1, d2) = line2[1]

    AB = (b1 - a1, b2 - a2)
    CD = (d1 - c1, d2 - c2)
    AC = (c1 - a1, c2 - a2)
    AD = (d1 - a1, d2 - a2)
    CA = (a1 - c1, a2 - c2)
    CB = (b1 - c1, b2 - c2)
    BC = (c1 - b1, c2 - b2)

    if (AB[0] * AC[1] - AB[1] * AC[0]) * (AB[0] * AD[1] - AB[1] * AD[0]) < 0 and (CD[0] * CA[1] - CD[1] * CA[0]) * (
            CD[0] * CB[1] - CD[1] * CB[0]) < 0:
        if (AB[0] * CD[1] - AB[1] * CD[0]) == 0:  # collineation
            if (BC[0] * CD[1] - BC[1] * CD[0]) == 0 and (a1 <= c1 <= b1 or a1 <= d1 <= b1):
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
    """
    Get the boundary map of the given OGM, also by type OccupiedGridMap, including only boundaries
    """
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
def get_lidar_map(boundary_map: np.ndarray, obstacles: np.ndarray, num_beams: int32, radius: int32, width: int32,
                  height: int32, num_obstacles: int32) -> np.ndarray:
    # TODO: translate this function into Sensor
    """_summary_
    Returns:
        cache of lidar sensing result, map[x][y][obstacle index] -> whether the obstacle is visible from (x,y)
    """
    hash_map = np.zeros((width, height, num_obstacles))
    blocks = width * height
    for i in prange(blocks):
        x = i // height
        y = i % height
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


class CoverageEnv:
    def __init__(self, cfg):
        self.inflated_map = list()
        self.lidar_map = list()
        self.map_config = cfg.map
        self.env_config = cfg.env

        # Agent config
        self.relay_config = cfg.relay
        self.relay_class = eval(self.env_config.relay_class)
        self.client_config = cfg.client
        self.client_class = eval(self.env_config.client_class)
        self.sensor_config = cfg.sensor

        # Simulation config
        self.time_step = 0
        self.n_episode = 0
        self.max_steps = self.env_config.max_steps
        self.step_size = self.env_config.step_size

        self.num_relay = self.env_config.num_relay
        self.num_client = self.env_config.num_client

        self.relay_o_dim = 4
        self.client_o_dim = 4
        self.env_name = 'ParticleEnvBoundGra'
        self.relay_v_max = cfg['relay']['v_max']
        self.client_v_max = cfg['client']['v_max']

        self.random = np.random
        # self.random.seed(0)

        self.curriculum = False

        # initialize new empty map
        self.global_map = OccupiedGridMap(
            is3D=False,
            boundaries=tuple(self.map_config.map_size)
        )

    def reset(self):
        self.time_step = 0
        self.n_episode += 1
        self.collision = False

        # Initialize map and corresponding sensor cache (lidar map)
        self.global_map.initailize_obstacle(self.map_config.obstacle_num)
        self.boundary_map = get_boundary_map(self.global_map, len(self.global_map.obstacles))

        # TODO: how to handle dynamic entities/obstacles
        self.lidar_map = get_lidar_map(
            boundary_map=self.boundary_map.grid_map,
            obstacles=np.array(self.boundary_map.obstacles),
            num_beams=self.sensor_config.num_beams,
            radius=self.relay_config.sens_range,
            width=self.boundary_map.boundaries[0],
            height=self.boundary_map.boundaries[1],
            num_obstacles=len(self.boundary_map.obstacles)
        ).tolist()

        self.inflated_map = deepcopy(self.global_map)

        # Initialize entities
        self.init_relays(min_dist=4, min_connection=1, inflated_map=self.inflated_map)
        self.init_client(inflated_map=self.inflated_map, target_list=[])

    def step(self, action) -> tuple[list, bool, object]:
        next_state = list()
        rewards = list()
        self.time_step += 1
        for idx, relay in enumerate(self.relay_list):
            next_state.append(relay.step(action[idx]))

        for idx, relay in enumerate(self.relay_list):
            state = next_state[idx]
            if relay.active:
                # reward, collision, caught = self.relay_reward(state, next_state)
                reward, collision, caught = 0, False, False
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
                    inflated_pos = (state[0] + i * self.relay_config.collision_radius,
                                    state[1] + j * self.relay_config.collision_radius)
                    if self.global_map.in_bound(inflated_pos):
                        collision = not self.global_map.is_unoccupied(inflated_pos)
                    if collision:
                        return collision
            return collision
        else:
            if obstacle_type == 'relay':
                obstacles = next_state
            else:  # client
                obstacles = self.get_state(agent_type=obstacle_type)
            collision = list()
            for obstacle in obstacles:
                if np.linalg.norm(
                        [obstacle[0] - state[0], obstacle[1] - state[1]]) <= self.relay_config.collision_radius:
                    collision.append(1)
                else:
                    collision.append(0)
            return collision

    def is_done(self) -> bool:
        return True if (self.time_step >= self.max_steps) or (sum(self.get_active()) == 0) else False

    def get_agent_state(self, agent):
        return [agent.pos[0], agent.pos[1], agent.vel[0], agent.vel[1]]

    def get_state(self, agent_type: str):
        """get states of the collective

        Args:
            agent_type: relay OR client
        Returns:
            state: list of tuple
        """
        agent_list = getattr(self, agent_type + '_list')
        state = [self.get_agent_state(agent) for agent in agent_list]
        return state

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
                if active[i] and active[j] and (i <= j) and (
                        np.linalg.norm([item_i[0] - item_j[0], item_i[1] - item_j[1]]) <= self.relay_config.comm_range):
                    adj_mat[i, j] = 1
                    adj_mat[j, i] = 1
        adj_mat = adj_mat.tolist()
        return adj_mat

    def sensor(self):
        # FIXME: what's the purpose of this function? why obstacle list must be at a fixed size
        obstacle_adj_list = list()
        client_adj_list = list()
        for relay in self.relay_list:
            if relay.active:
                obstacle_adj = self.lidar_map[int(relay.pos[0])][int(relay.pos[1])] + [0] * (
                        self.map_config.max_obstacle_num - self.map_config.obstacle_num)
                client_adj = relay.find_client(
                    occupied_grid_map=self.global_map,
                    pos=relay.pos,
                    client_pos=self.client_list[0].pos,
                )
            else:
                obstacle_adj = [0] * self.map_config.max_obstacle_num
                client_adj = [0]
            obstacle_adj_list.append(obstacle_adj)
            client_adj_list.append(client_adj)
        return obstacle_adj_list, client_adj_list

    def init_relays(self, min_dist: float, min_connection: int, inflated_map: OccupiedGridMap):
        """
        Initialize the relay agents and modify the input map correspondingly
        Args:
            min_dist (float): configure the density of the collective.
            min_connection (int): configure the lowest requirement for newly appended relay
            inflated_map (OccupiedGridMap): the inflated obstacle map
        Returns:
            inflated_map: occupied grid map with the relay as moving obstacle, and all obstacles are inflated.
        """
        map_size = inflated_map.boundaries
        width = map_size[0]
        height = map_size[1]

        self.relay_list = list()
        position_list = list()
        while len(self.relay_list) < self.num_relay:
            pos = list(np.random.rand(2) * np.array([width - 1, height - 1]))

            # collision checking
            if not inflated_map.is_unoccupied(pos=pos):
                continue

            if len(self.relay_list) > 0:
                # initial safe distance and connectivity constraints checking
                dists = [np.linalg.norm((pos[0] - p[0], pos[1] - p[1])) for p in position_list]
                # TODO: some bugs here, connection check for early generated relays are kind of weired
                if min(dists) < min_dist or sum(d < self.relay_config.comm_range for d in dists) < min_connection:
                    continue

            position_list.append(pos)
            self.relay_list.append(self.relay_class(
                pos=pos, vel=[0., 0.],
                config=self.relay_config
            ))
            inflated_map.set_moving_obstacle([pos])
            inflated_map.extended_moving_obstacles(extend_dis=2)

    def init_client(self, inflated_map: OccupiedGridMap, target_list: list):
        """initialize the client.

        Args:
            inflated_map: occupied grid map with the relay as moving obstacle, and all obstacles are inflated.
            target_list: the navigation goals for clients
            # TODO: dynamic target list (predefined multi-goal OR automatically recalculate)
        """
        agent_block = inflated_map.moving_obstacles
        position_list = list()
        self.client_list = list()
        map_size = inflated_map.boundaries
        width = map_size[0]
        height = map_size[1]
        while len(position_list) < self.num_client:
            pos = list(np.random.rand(2) * np.array([width - 1, height - 1]))
            if inflated_map.is_unoccupied(pos=pos):
                position_list.append(pos)
                client = self.client_class(
                    pos=pos, vel=[0., 0.],
                    target=pos,
                    config=self.client_config,
                    global_map=self.global_map
                )
                self.client_list.append(client)

    def client_step(self):
        # Based On A Star Algorithm
        path_list = list()
        for client in self.client_list:
            # path = client.replan(global_map=self.global_map, target=client.target)
            path = [client.pos]
            action = int(client.theta / (2 * np.pi / len(client.theta_list)))
            [x, y, vx, vy, theta] = client.step(action=action)
            if self.global_map.in_bound((x, y)) and self.global_map.is_unoccupied((x, y)):
                client.apply_update([x, y, vx, vy, theta])
            path_list.append(path)
        return path_list

    def demon(self):
        theta_list = [i * np.pi / 4 for i in range(0, 8)]
        actions_mat = [[np.cos(t), np.sin(t)] for t in theta_list]
        actions_mat.append([0., 0.])
        action_list = list()
        client_pos = self.client_list[0].pos
        c_x, c_y = client_pos
        for relay in self.relay_list:
            x, y = relay.pos
            radius = np.linalg.norm([x - c_x, y - c_y])
            if math.isclose(radius, 0.0, abs_tol=0.01):
                action = [0., 0.]
            else:
                phi = np.sign(c_y - y) * np.arccos((c_x - x) / (radius + 1e-3))
                action = [np.cos(phi), np.sin(phi)]
            middle_a = [np.linalg.norm((a[0] - action[0], a[1] - action[1])) for a in actions_mat]
            action_list.append(middle_a.index(min(middle_a)))
        return action_list
