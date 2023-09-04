from particle_env import ParticleEnv
import time
import random
import numpy as np
from utils.gif_plotting import sim_moving
from particle_env import ParticleEnv
import os


if __name__ == "__main__":
    """
    NOMENCLATURE:
    p: Pursuer
    e: Evader
    o: Obstacle
    r: reward
    epi: Episode
    pred: predict
    
    """
    # Simulation Time
    begin_time = time.time()
    # Store States, Rewards, Adjacent Matrix, Path, Obstacles
    epi_obs_p = []
    epi_obs_e = []
    epi_r = []
    epi_path = []
    epi_p_o_adj = []
    epi_p_e_adj = []
    epi_p_p_adj = []
    epi_extended_obstacles = []
    episode_reward = 0
    win_tag = False
    p_num = 15
    # Initialize the Environment
    env = ParticleEnv()
    env.reset(p_num=15)
    # Loop
    for step in range(env.episode_limit):
        # Obtain States
        p_state = env.get_team_state(True, False)  # p_state.shape=(pursuer_num, obs_dim), where N is the number of the pursuers/agents
        e_state = env.get_team_state(False, False)  # e_state.shape=(evader_num, obs_dim)
        # Obtain adjacent matrices of pursuer-pursuer, pursuer-obstacle, pursuer-evader
        p_p_adj = env.communicate()  # p_p_adj.shape of (pursuer_num, pursuer_num)
        p_o_adj, p_e_adj = env.sensor(evader_pos=e_state)  # p_p_adj.shape of (N, N), p_e_adj.shape of (N, 1)
        # Evader step and return its local predicted map
        path, pred_map = env.evader_step()
        # Save extended obstacles which has different color (gray), and the color of real obstacles is black
        extended_obstacles = pred_map.ex_obstacles + pred_map.ex_moving_obstacles
        # Generate random action for agents
        a_n = [random.randint(0, 8) for i in range(p_num)]
        # Step
        r, done, info = env.step(a_n)  # Take a step
        # Save datas for plotting
        episode_reward += sum(r)
        epi_obs_p.append(p_state)
        epi_obs_e.append(e_state)
        epi_path.append(path)
        epi_extended_obstacles.append(extended_obstacles)
        epi_p_p_adj.append(p_p_adj)
        epi_p_e_adj.append(p_e_adj)
        epi_p_o_adj.append(p_o_adj)
        
        if done:
            # Whether evader is cought
            e_col = 1 if done and not env.e_list['0'].active else 0
            # Whether the agent is collided with obstacles or teamates
            collision = bool(p_num - sum(info) - e_col)
            # Print Game Result
            print('DONE!')
            print('time cost: ', time.time() - begin_time)
            print(f'win: {bool(win_tag)}    collision: {bool(collision)}    reward: {episode_reward}    steps: {step}')

            epi_obs_p = np.array(epi_obs_p)
            epi_obs_e = np.array(epi_obs_e)
            # Plotting
            sim_moving(
                step=step + 1,
                height=env.height,
                width=env.width,
                obstacles=env.global_obstacles,
                boundary_obstacles=env.boundary_obstacles,
                extended_obstacles=epi_extended_obstacles,
                box_width=env.box_width,
                n_p=p_num,
                n_e=1,
                p_x=epi_obs_p[:, :, 0],
                p_y=epi_obs_p[:, :, 1],
                e_x=epi_obs_e[:, :, 0],
                e_y=epi_obs_e[:, :, 1],
                path=epi_path,
                target=env.target,
                e_ser=env.e_sen_range,
                c_r=env.kill_radius,
                p_p_adj=epi_p_p_adj,
                p_e_adj=epi_p_e_adj,
                p_o_adj=epi_p_o_adj,
                dir='sim_moving' + str(time.time())
            )
            break
