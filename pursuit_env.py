import argparse
import numpy as np
from base_env import BaseEnv


class Pursuit_Env(BaseEnv):
    def __init__(self, env_config):
        super.__init__(env_config)

    def reset(self):
        self.time_step = 0
        self.n_episode += 1
        self.collision = False
        
        inflated_map = self.init_map(map_config=None)
        # No need for navigation and coverage
        self.init_target(inflated_map=inflated_map)
        inflated_map = self.init_defender(inflated_map=inflated_map)
        # TODO: the target should be assigned to the attacker manually
        self.init_attacker(inflated_map=inflated_map)
        
    def step(self, action):
        self.time_step += 1
        for idx, defender in self.defender_list:
            defender.step(self.step_size, action[idx])

        reward = self.get_reward()
        done = True if self.get_done() or self.time_step >= self.episode_limit else False
        info = None
        return reward, done, info

    def defender_reward(self, defender, is_pursuer=True):
        reward = 0
        state = self.get_agent_state(defender)
        is_collision = self.collision_detection(state, obstacle_type='attacker')
        reward += sum(is_collision) * 1

        inner_collision = self.collision_detection(state, obstacle_type='defender')
        reward -= (sum(inner_collision) - 1) * 1

        obstacle_collision = self.collision_detection(state, obstacle_type='obstacle')
        reward -= (sum(obstacle_collision)) * 1
        
        if sum(obstacle_collision) + sum(inner_collision) - 1 > 0:
            self.collision = True
        return reward
    
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
                if (i <= j) and (np.linalg.norm([item_i[0] - item_j[0], item_i[1] - item_j[1]]) <= self.comm_range):
                    adj_mat[i, j] = 1
                    adj_mat[j, 1] = 1
        adj_mat = adj_mat.tolist()
        return adj_mat


parser = argparse.ArgumentParser("Configuration Setting in MRS Environment")
# Agent Config
parser.add_argument("--sen_range", type=int, default=6, help='The sensor range of the agents')
parser.add_argument("--comm_range", type=int, default=12, help='The communication range of the agents')
parser.add_argument("--collision_radius", type=float, default=0.5, help='The smallest distance at which a collision can occur between two agents')

parser.add_argument("--vel_max_d", type=float, default=0.5, help='The limitation of the velocity of the defender')
parser.add_argument("--vel_max_a", type=float, default=1.0, help='The limitation of the velocity of the attacker')
parser.add_argument("--tau", type=float, default=0.01, help='The time constant of first-order dynamic system')

# Env Config
parser.add_argument("--env_name", type=str, default='Pursuit-Evasion Game')
parser.add_argument("--max_steps", type=int, default=250, help='The maximum time steps in a episode')
parser.add_argument("--step_size", type=float, default=0.1, help='The size of simulation time step')
parser.add_argument("--num_target", type=int, default=1, help='The number of target point, 1 in pursuit, n in coverage, 0 in navigation')
parser.add_argument("--num_defender", type=int, default=15, help='The number of defender (pursuer/server/navigator)')
parser.add_argument("--num_attacker", type=int, default=15, help='The number of attacker (evader/client/target)')
