import warnings, hydra, time
import numpy as np
from numba.core.errors import NumbaDeprecationWarning, NumbaWarning
from .env import CoverageEnv
from .gif_plotting import sim_moving


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
    epi_obs_p = list()
    epi_obs_e = list()
    epi_target = list()
    epi_r = list()
    epi_path = list()
    epi_p_o_adj = list()
    epi_p_e_adj = list()
    epi_p_p_adj = list()
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
        path = env.attacker_step()
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
        # epi_extended_obstacles.append(pred_map.ex_moving_obstacles + pred_map.ex_obstacles)
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
                height=cfg.map.map_size[0],
                width=cfg.map.map_size[1],
                obstacles=env.occupied_map.obstacles,
                boundary_obstacles=env.boundary_map.obstacles,
                # extended_obstacles=epi_extended_obstacles,
                box_width=cfg.map.resolution,
                n_p=cfg.env.num_defender,
                n_e=1,
                p_x=epi_obs_p[:, :, 0],
                p_y=epi_obs_p[:, :, 1],
                e_x=epi_obs_e[:, :, 0],
                e_y=epi_obs_e[:, :, 1],
                path=epi_path,
                target=epi_target,
                e_ser=cfg.attacker.sen_range,
                c_r=cfg.defender.collision_radius,
                p_p_adj=epi_p_p_adj,
                p_e_adj=epi_p_e_adj,
                p_o_adj=epi_p_o_adj,
                dir='sim_moving' + str(time.time())
            )
            break


if __name__ == '__main__':
    main()