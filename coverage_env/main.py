import hydra
import time
import warnings

import numpy as np
from numba.core.errors import NumbaDeprecationWarning, NumbaWarning

from coverage_env.env import CoverageEnv
from coverage_env.gif_plotting import sim_moving


@hydra.main(config_path='./', config_name='config.yaml', version_base=None)
def main(cfg):
    warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
    warnings.simplefilter('ignore', category=NumbaWarning)

    # Initialize the Env
    env = CoverageEnv(cfg)
    env.reset()
    done = False
    acc_reward = 0

    # Evaluate Matrix
    epi_observation_relay = list()
    epi_observation_client = list()
    epi_target = list()
    epi_r = list()
    epi_path = list()
    epi_r2o_adj = list()
    epi_p2e_adj = list()
    epi_p2p_adj = list()
    start_time = time.time()
    idx = 0
    while not done:
        print(idx)
        idx += 1
        state = env.get_state(agent_type='client')
        p_p_adj = env.communicate()
        p_o_adj, p_e_adj = env.sensor()
        # action = [random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8]) for _ in range(env_config.num_defender)]
        action = env.demon()
        path = env.client_step()
        rewards, done, info = env.step(action)
        acc_reward += sum(rewards)

        # Store Evaluate Matrix
        epi_observation_relay.append(env.get_state(agent_type='relay'))
        epi_observation_client.append(env.get_state(agent_type='client'))
        epi_target.append([client.target for client in env.client_list])
        epi_r.append(sum(rewards))
        epi_path.append(path)
        epi_p2p_adj.append(p_p_adj)
        epi_p2e_adj.append(p_e_adj)
        epi_r2o_adj.append(p_o_adj)
        # epi_extended_obstacles.append(pred_map.ex_moving_obstacles + pred_map.ex_obstacles)
        if done:
            # Print Game Result
            print('DONE!')
            print('time cost: ', time.time() - start_time)
            print(f'reward: {acc_reward}')

            epi_observation_relay = np.array(epi_observation_relay)
            epi_observation_client = np.array(epi_observation_client)
            # Plotting
            sim_moving(
                step=env.time_step,
                height=cfg.map.map_size[0],
                width=cfg.map.map_size[1],
                obstacles=env.global_map.obstacles,
                boundary_obstacles=env.boundary_map.obstacles,
                # extended_obstacles=epi_extended_obstacles,
                box_width=cfg.map.resolution,
                n_p=cfg.env.num_relay,
                n_e=1,
                p_x=epi_observation_relay[:, :, 0],
                p_y=epi_observation_relay[:, :, 1],
                e_x=epi_observation_client[:, :, 0],
                e_y=epi_observation_client[:, :, 1],
                path=epi_path,
                target=epi_target,
                e_ser=cfg.client.sens_range,
                c_r=cfg.relay.collision_radius,
                p_p_adj=epi_p2p_adj,
                p_e_adj=epi_p2e_adj,
                p_o_adj=epi_r2o_adj,
                dir='sim_moving' + str(time.time())
            )
            break


if __name__ == '__main__':
    main()
