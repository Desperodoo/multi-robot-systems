import random
import argparse
import numpy as np
from base_env import BaseEnv


class Pursuit_Env(BaseEnv):
    def __init__(self, env_config, defender_config, attacker_config):
        super.__init__(env_config, defender_config, attacker_config)

    def reset(self):
        self.time_step = 0
        self.n_episode += 1
        self.collision = False
        
        inflated_map = self.init_map(map_config=None)
        # No need for navigation and coverage
        self.init_target(inflated_map=inflated_map)
        inflated_map = self.init_defender(min_dist=4, inflated_map=inflated_map, defender_config=defender_config)
        # TODO: the target should be assigned to the attacker manually
        self.init_attacker(inflated_map=inflated_map, is_percepted=True, attacker_config=attacker_config)
        
    def step(self, action):
        next_state = list()
        rewards = list()
        can_applys = list()
        self.time_step += 1
        for idx, defender in enumerate(self.defender_list):
            next_state.append(defender.step(self.step_size, action[idx]))
            
        for state in next_state:
            reward, can_apply = self.defender_reward(state, next_state)
            rewards.append(reward)
            can_applys.append(can_apply)
        
        for idx, defender in enumerate(self.defender_list):
            if can_applys[idx]:
                defender.apply_update(next_state[idx])
                
        done = True if self.time_step >= self.episode_limit else False
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
        
        is_collision = self.collision_detection(state, obstacle_type='attacker')
        reward += sum(is_collision) * 1

        return reward
    
    def collision_detection(self, state, obstacle_type: str = 'obstacle', next_state: list = None):
        if obstacle_type == 'obstacle':
            collision = False
            for i in range(-1, 2, 1):
                for j in range(-1, 2, 1):
                    inflated_pos = (state[0] + i * self.collision_radius, state[1] + j * self.collision_radius)
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
                if np.linalg.norm([obstacle[0] - state[0], obstacle[1] - state[1]]) <= self.collision_radius:
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
                if (i <= j) and (np.linalg.norm([item_i[0] - item_j[0], item_i[1] - item_j[1]]) <= self.comm_range):
                    adj_mat[i, j] = 1
                    adj_mat[j, 1] = 1
        adj_mat = adj_mat.tolist()
        return adj_mat


if __name__ == '__main__':
    # Pursuer Config
    parser = argparse.ArgumentParser("Configuration Setting of the Defender")
    parser.add_argument("--sen_range", type=int, default=6, help='The sensor range of the agents')
    parser.add_argument("--comm_range", type=int, default=12, help='The communication range of the agents')
    parser.add_argument("--collision_radius", type=float, default=0.5, help='The smallest distance at which a collision can occur between two agents')
    parser.add_argument("--step_size", type=float, default=0.1, help='The size of simulation time step')

    parser.add_argument("--vmax", type=float, default=2, help='The limitation of the velocity of the defender')
    parser.add_argument("--tau", type=float, default=0.01, help='The time constant of first-order dynamic system')
    parser.add_argument("--DOF", type=int, default=2, help='The dimension of freedom, 2 or 3')

    defender_config = parser.parse_args()
    
    # Evader Config
    parser = argparse.ArgumentParser("Configuration Setting of the Attacker")
    parser.add_argument("--sen_range", type=int, default=6, help='The sensor range of the agents')
    parser.add_argument("--comm_range", type=int, default=12, help='The communication range of the agents')
    parser.add_argument("--collision_radius", type=float, default=0.5, help='The smallest distance at which a collision can occur between two agents')
    parser.add_argument("--step_size", type=float, default=0.1, help='The size of simulation time step')

    parser.add_argument("--vmax", type=float, default=4, help='The limitation of the velocity of the defender')
    parser.add_argument("--tau", type=float, default=0.01, help='The time constant of first-order dynamic system')
    parser.add_argument("--DOF", type=int, default=2, help='The dimension of freedom, 2 or 3')

    attacker_config = parser.parse_args()
    
    # Env Config
    parser = argparse.ArgumentParser("Configuration Setting of the MRS Environment")
    parser.add_argument("--env_name", type=str, default='Pursuit-Evasion Game')
    parser.add_argument("--max_steps", type=int, default=250, help='The maximum time steps in a episode')
    parser.add_argument("--step_size", type=float, default=0.1, help='The size of simulation time step')
    parser.add_argument("--num_target", type=int, default=1, help='The number of target point, 1 in pursuit, n in coverage, 0 in navigation')
    parser.add_argument("--num_defender", type=int, default=15, help='The number of defender (pursuer/server/navigator)')
    parser.add_argument("--num_attacker", type=int, default=15, help='The number of attacker (evader/client/target)')

    parser.add_argument("--defender_class", type=str, default='Pursuer', help='The class of the defender')
    parser.add_argument("--attacker_class", type=str, default='Attacker', help='The class of the attacker')

    env_config = parser.parse_args()
    
    # Map Config
    parser = argparse.ArgumentParser("Configuration Setting of the Map")
    parser.add_argument("--num_obstacle", type=int, default=5, help='The number of the obstacles')
    parser.add_argument("--center", type=int, defaule=10, help='The center of the obstacles')
    
    map_config = parser.parse_args()
    
    # Initialize the Env
    env = Pursuit_Env(map_config=map_config, env_config=env_config, defender_config=defender_config, attacker_config=attacker_config)
    env.reset()
    done = False
    while not done:
        state = env.get_state()
        action = [random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8]) for _ in range(env_config.num_defender)]
    