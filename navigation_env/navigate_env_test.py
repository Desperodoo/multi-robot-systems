from navigate_env import navigate_env
from navigate_env import Pursuer
from config import flocking_config
import numpy as np




if __name__ == '__main__':
    map_config = {
        "map_size": (20,20),
        "is3D": False,
        "spread": 3,
        "field_radius" : 5,
        "field_cof" : -5,
        "time_limit": 10
    }
    agent_config = {
        "phi" : 0.2,
        "phi_lmt" : 10,
        "v": 1,
        "v_max" : 2,
        "v_lmt" : 3,
        "agent_num": 2,
        "sen_range" : 10,
        "fov" : 2 * np.pi
    }
    env = navigate_env(map_config=map_config,agent_config=agent_config)
    env.reset()
    states, agent_map, reward, done = env.step([1,1])
    print(states)