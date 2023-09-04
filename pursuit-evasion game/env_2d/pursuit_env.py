import time, math
import os, sys
import random
import argparse
import numpy as np
from copy import deepcopy
from numba import jit
from base_env import BaseEnv
from gif_plotting import sim_moving
from Occupied_Grid_Map import OccupiedGridMap
from skimage.segmentation import find_boundaries


def get_boundary_map(occupied_grid_map: OccupiedGridMap) -> OccupiedGridMap:
    boundary_map = deepcopy(occupied_grid_map)
    boundary_map.grid_map = find_boundaries(occupied_grid_map.grid_map, mode='inner')
    boundary_map.obstacles = np.argwhere(boundary_map.grid_map == 1).tolist()
    return boundary_map
    
    
def get_raser_map(boundary_map: OccupiedGridMap, num_beams: int, radius: int, max_num_obstacle: int):
    hash_map = np.zeros((*boundary_map.boundaries, max_num_obstacle))
    (width, height) = boundary_map.boundaries
    for x in range(width):
        for y in range(height):
            local_obstacles = list()
            for obstacle in boundary_map.obstacles:
                if np.linalg.norm([obstacle[0] - x, obstacle[1] - y]) <= radius:
                    local_obstacles.append(obstacle)

            for beam in range(num_beams):
                beam_angle = beam * 2 * np.pi / num_beams
                beam_dir_x = np.cos(beam_angle)
                beam_dir_y = np.sin(beam_angle)
                for beam_range in range(radius):
                    beam_current_x = x + beam_range * beam_dir_x
                    beam_current_y = y + beam_range * beam_dir_y
                    if (beam_current_x < 0 or beam_current_x >= width or beam_current_y < 0 or beam_current_y >= height):
                        break
                    
                    beam_current_pos = [int(beam_current_x), int(beam_current_y)]
                    if not boundary_map.is_unoccupied(beam_current_pos):
                        idx = boundary_map.obstacles.index(beam_current_pos)
                        hash_map[x, y, idx] = 1
                        break
    
    hash_map = hash_map.tolist()
    return hash_map


class Pursuit_Env(BaseEnv):
    def __init__(self, map_config, env_config, defender_config, attacker_config, sensor_config):
        super().__init__(map_config, env_config, defender_config, attacker_config, sensor_config)

    def reset(self):
        self.time_step = 0
        self.n_episode += 1
        self.collision = False
        
        inflated_map = self.init_map()
        self.boundary_map = get_boundary_map(self.occupied_map)
        start_time = time.time()
        self.raser_map = get_raser_map(boundary_map=self.boundary_map, num_beams=self.sensor_config.num_beams, radius=self.sensor_config.radius, max_num_obstacle=map_config.max_num_obstacle)
        print('time cost 1:', time.time() - start_time)
        self.inflated_map = deepcopy(inflated_map)
        # No need for navigation and coverage
        self.init_target(inflated_map=inflated_map)
        inflated_map = self.init_defender(min_dist=4, inflated_map=inflated_map)
        # TODO: the target should be assigned to the attacker manually
        self.init_attacker(inflated_map=inflated_map, is_percepted=True, target_list=self.target)
        
    def attacker_step(self):
        # Based On A Star Algorithm
        state = self.get_state(agent_type='defender')
        for attacker in self.attacker_list:
            path, way_point, pred_map = attacker.replan(moving_obs=state, occupied_map=self.occupied_map)
            phi = attacker.waypoint2phi(way_point)
            action = [np.cos(phi) * self.attacker_config.vmax, np.sin(phi) * self.attacker_config.vmax]
            [x, y, vx, vy, theta] = attacker.step(action=action)
            if self.occupied_map.in_bound((x, y)) and self.occupied_map.is_unoccupied((x, y)):
                attacker.apply_update([x, y, vx, vy, theta])
            if np.linalg.norm((self.target[0][0] - x, self.target[0][1] - y)) <= self.attacker_config.collision_radius:
                self.init_target(inflated_map=self.inflated_map)
                attacker.target = self.target[0]
            
        return path, pred_map    
        
    def step(self, action):
        next_state = list()
        rewards = list()
        can_applys = list()
        self.time_step += 1
        for idx, defender in enumerate(self.defender_list):
            next_state.append(defender.step(action[idx]))
            
        for state in next_state:
            reward, can_apply = self.defender_reward(state, next_state)
            rewards.append(reward)
            can_applys.append(can_apply)
        
        for idx, defender in enumerate(self.defender_list):
            if can_applys[idx]:
                defender.apply_update(next_state[idx])
                
        done = True if self.time_step >= self.max_steps else False
        info = None
        return rewards, done, info

    def get_done(self):
        pass
    
    def defender_reward(self, state, next_state):
        reward = 0
        can_apply = True
        
        inner_collision = self.collision_detection(state, obstacle_type='defender', next_state=next_state)
        reward -= (sum(inner_collision) - 1) * 1

        obstacle_collision = self.collision_detection(state, obstacle_type='obstacle')
        reward -= obstacle_collision
        
        if reward < 0:
            can_apply = False
            self.collision = True
            return reward, can_apply
        
        boundaries = self.map_config.map_size
        state[0] = np.clip(state[0], 0, boundaries[0] - 1)
        state[1] = np.clip(state[1], 0, boundaries[1] - 1)
        is_collision = self.collision_detection(state, obstacle_type='attacker')
        reward += sum(is_collision) * 1

        return reward, can_apply
    
    def collision_detection(self, state, obstacle_type: str = 'obstacle', next_state: list = None):
        if obstacle_type == 'obstacle':
            collision = False
            for i in range(-1, 2, 1):
                for j in range(-1, 2, 1):
                    inflated_pos = (state[0] + i * self.defender_config.collision_radius, state[1] + j * self.defender_config.collision_radius)
                    if self.occupied_map.in_bound(inflated_pos):
                        collision = not self.occupied_map.is_unoccupied(inflated_pos)
                    if collision == True:
                        break
                if collision == True:
                    break
            return collision
        else:
            if obstacle_type == 'defender':
                obstacles = next_state
            else:
                obstacles = self.get_state(agent_type=obstacle_type)

            collision = list()
            for obstacle in obstacles:
                if np.linalg.norm([obstacle[0] - state[0], obstacle[1] - state[1]]) <= self.defender_config.collision_radius:
                    collision.append(1)
                else:
                    collision.append(0)

            return collision
    
    def get_agent_state(self, agent):
        return [agent.x, agent.y, agent.vx, agent.vy]

    def communicate(self):
        """
        the obstacles have no impact on the communication between agents
        :return: adj_mat: the adjacent matrix of the agents
        """
        states = self.get_state(agent_type='defender')
        adj_mat = np.zeros(shape=(self.num_defender, self.num_defender))
        for i, item_i in enumerate(states):
            for j, item_j in enumerate(states):
                if (i <= j) and (np.linalg.norm([item_i[0] - item_j[0], item_i[1] - item_j[1]]) <= self.defender_config.comm_range):
                    adj_mat[i, j] = 1
                    adj_mat[j, 1] = 1
        adj_mat = adj_mat.tolist()
        return adj_mat
    
    def sensor(self):
        obstacle_adj_list = list()
        attacker_adj_list = list()
        for defender in self.defender_list:
            obstacle_adj = self.raser_map[int(defender.x)][int(defender.y)]
            attacker_adj = defender.find_attacker(
                occupied_grid_map=self.occupied_map, 
                pos=(round(defender.x), round(defender.y)),
                attacker_pos=(round(self.attacker_list[0].x), round(self.attacker_list[0].y)),
            )
            obstacle_adj_list.append(obstacle_adj)
            attacker_adj_list.append(attacker_adj)
        return obstacle_adj_list, attacker_adj_list

    def demon(self):
        theta_list = [i * np.pi / 4 for i in range(0, 8)]
        actions_mat = [[np.cos(t), np.sin(t)] for t in theta_list]
        actions_mat.append([0., 0.])
        action_list = list()
        e_x = self.attacker_list[0].x
        e_y = self.attacker_list[0].y
        for defender in self.defender_list:
            x = defender.x
            y = defender.y
            radius = np.linalg.norm([x - e_x, y - e_y])
            if math.isclose(radius, 0.0, abs_tol=0.01):
                action = [0., 0.]
            else:
                phi = np.sign(e_y - y) * np.arccos((e_x - x) / (radius + 1e-3))
                action = [np.cos(phi), np.sin(phi)]
            middle_a = [np.linalg.norm((a[0] - action[0], a[1] - action[1])) for a in actions_mat]
            action_list.append(middle_a.index(min(middle_a)))
        return action_list
    

if __name__ == '__main__':
    os.chdir(sys.path[0])
    # Env Config
    parser = argparse.ArgumentParser("Configuration Setting of the MRS Environment")
    parser.add_argument("--env_name", type=str, default='Pursuit-Evasion Game')
    parser.add_argument("--max_steps", type=int, default=250, help='The maximum time steps in a episode')
    parser.add_argument("--step_size", type=float, default=0.1, help='The size of simulation time step')
    parser.add_argument("--num_target", type=int, default=1, help='The number of target point, 1 in pursuit, n in coverage, 0 in navigation')
    parser.add_argument("--num_defender", type=int, default=15, help='The number of defender (pursuer/server/navigator)')
    parser.add_argument("--num_attacker", type=int, default=1, help='The number of attacker (evader/client/target)')

    parser.add_argument("--defender_class", type=str, default='Pursuer', help='The class of the defender')
    parser.add_argument("--attacker_class", type=str, default='Evader', help='The class of the attacker')

    env_config = parser.parse_args()
    
    # Map Config
    parser = argparse.ArgumentParser("Configuration Setting of the Map")
    parser.add_argument("--resolution", type=int, default=1, help='The resolution of the map')
    parser.add_argument("--num_obstacle_block", type=int, default=5, help='The number of the obstacles')
    parser.add_argument("--center", type=int, default=(30, 30), help='The center of the obstacles')
    parser.add_argument("--variance", type=int, default=10, help='The varience of normal distribution that generate the position of obstacle block')
    parser.add_argument("--map_size", type=tuple, default=(60, 60), help='The size of the map')
    parser.add_argument("--is3D", type=bool, default=False, help='The dimension of freedom, 2 or 3')
    parser.add_argument("--max_num_obstacle", type=int, default=110, help='The max number of boundary obstacle, equivalent to num_obs_block * 22 (boundary of a 6x7 rectangle)')

    map_config = parser.parse_args()
    
    # Pursuer Config
    parser = argparse.ArgumentParser("Configuration Setting of the Defender")
    parser.add_argument("--sen_range", type=int, default=8, help='The sensor range of the agents')
    parser.add_argument("--comm_range", type=int, default=16, help='The communication range of the agents')
    parser.add_argument("--collision_radius", type=float, default=0.5, help='The smallest distance at which a collision can occur between two agents')
    parser.add_argument("--step_size", type=float, default=0.1, help='The size of simulation time step')

    parser.add_argument("--vmax", type=float, default=2, help='The limitation of the velocity of the defender')
    parser.add_argument("--tau", type=float, default=0.2, help='The time constant of first-order dynamic system')
    parser.add_argument("--DOF", type=int, default=2, help='The dimension of freedom, 2 or 3')

    defender_config = parser.parse_args()
    
    # Evader Config
    parser = argparse.ArgumentParser("Configuration Setting of the Attacker")
    parser.add_argument("--sen_range", type=int, default=8, help='The sensor range of the agents')
    parser.add_argument("--comm_range", type=int, default=16, help='The communication range of the agents')
    parser.add_argument("--collision_radius", type=float, default=0.5, help='The smallest distance at which a collision can occur between two agents')
    parser.add_argument("--step_size", type=float, default=0.1, help='The size of simulation time step')

    parser.add_argument("--vmax", type=float, default=4, help='The limitation of the velocity of the defender')
    parser.add_argument("--tau", type=float, default=0.2, help='The time constant of first-order dynamic system')
    parser.add_argument("--DOF", type=int, default=2, help='The dimension of freedom, 2 or 3')

    parser.add_argument("--x_dim", type=int, default=map_config.map_size[0], help='The x-dimension of map')
    parser.add_argument("--y_dim", type=int, default=map_config.map_size[1], help='The y-dimension of map')
    
    parser.add_argument("--extend_dis", type=int, default=3, help='The extend distance for astar')
    attacker_config = parser.parse_args()
    
    # Sensor Config
    parser = argparse.ArgumentParser("Configuration Setting of the Sensor")
    parser.add_argument("--num_beams", type=int, default=36, help='The number of beams in LiDAR')
    parser.add_argument("--radius", type=int, default=defender_config.sen_range, help='The radius of beams in LiDAR')

    sensor_config = parser.parse_args()
    
    # Initialize the Env
    env = Pursuit_Env(map_config=map_config, env_config=env_config, defender_config=defender_config, attacker_config=attacker_config, sensor_config=sensor_config)
    env.reset()
    done = False
    acc_reward = 0
    # Evaluate Matrix
    epi_obs_p = list()
    epi_obs_e = list()
    epi_target = list()
    epi_r = list()
    epi_path = list()
    epi_p_o_adj = list()
    epi_p_e_adj = list()
    epi_p_p_adj = list()
    epi_extended_obstacles = list()
    win_tag = False
    start_time = time.time()
    idx = 0
    while not done:
        print(idx)
        idx += 1
        state = env.get_state(agent_type='attacker')
        p_p_adj = env.communicate()
        p_o_adj, p_e_adj = env.sensor()
        # action = [random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8]) for _ in range(env_config.num_defender)]
        action = env.demon()
        path, pred_map = env.attacker_step()
        rewards, done, info = env.step(action)
        acc_reward += sum(rewards)
        # Store Evaluate Matrix
        epi_obs_p.append(env.get_state(agent_type='defender'))
        epi_obs_e.append(env.get_state(agent_type='attacker'))
        epi_target.append(env.target[0])
        epi_r.append(sum(rewards))
        epi_path.append(path)
        epi_p_p_adj.append(p_p_adj)
        epi_p_e_adj.append(p_e_adj)
        epi_p_o_adj.append(p_o_adj)
        epi_extended_obstacles.append(pred_map.ex_moving_obstacles + pred_map.ex_obstacles)
        if done:
            # Print Game Result
            print('DONE!')
            print('time cost: ', time.time() - start_time)
            print(f'reward: {acc_reward}')

            epi_obs_p = np.array(epi_obs_p)
            epi_obs_e = np.array(epi_obs_e)
            # Plotting
            sim_moving(
                step=env.time_step,
                height=map_config.map_size[0],
                width=map_config.map_size[1],
                obstacles=env.occupied_map.obstacles,
                boundary_obstacles=env.boundary_map.obstacles,
                extended_obstacles=epi_extended_obstacles,
                box_width=map_config.resolution,
                n_p=env_config.num_defender,
                n_e=1,
                p_x=epi_obs_p[:, :, 0],
                p_y=epi_obs_p[:, :, 1],
                e_x=epi_obs_e[:, :, 0],
                e_y=epi_obs_e[:, :, 1],
                path=epi_path,
                target=epi_target,
                e_ser=attacker_config.sen_range,
                c_r=defender_config.collision_radius,
                p_p_adj=epi_p_p_adj,
                p_e_adj=epi_p_e_adj,
                p_o_adj=epi_p_o_adj,
                dir='sim_moving' + str(time.time())
            )
            break
