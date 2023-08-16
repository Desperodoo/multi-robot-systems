from base_env import BaseEnv
import argparse


parser = argparse.ArgumentParser("Configuration Setting in MRS Environment")
# Agent Config
parser.add_argument("--sensor_range", type=int, default=6, help='The sensor range of the agents')
parser.add_argument("--comm_range", type=int, default=12, help='The communication range of the agents')
parser.add_argument("--collision_radius", type=float, default=0.5, help='The smallest distance at which a collision can occur between two agents')

parser.add_argument("--vel_lmt_d", type=float, default=0.5, help='The limitation of the velocity of the defender')
parser.add_argument("--vel_lmt_a", type=float, default=1.0, help='The limitation of the velocity of the attacker')
parser.add_argument("--tau", type=float, default=0.01, help='The time constant of first-order dynamic system')

# Env Config
parser.add_argument("--env_name", type=str, default='Pursuit-Evasion Game')
parser.add_argument("--max_steps", type=int, default=250, help='The maximum time steps in a episode')
parser.add_argument("--step_size", type=float, default=0.1, help='The size of simulation time step')
parser.add_argument("--num_target", type=int, default=1, help='The number of target point, 1 in pursuit, n in coverage, 0 in navigation')
parser.add_argument("--num_defender", type=int, default=15, help='The number of defender (pursuer/server/navigator)')
parser.add_argument("--num_attacker", type=int, default=15, help='The number of attacker (evader/client/target)')


class Pursuit_Env(BaseEnv):
    def __init__(self, env_config):
        super.__init__(env_config)
        
