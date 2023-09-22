class ParticleEnv:
    def __init__(self):
        cfg = {
            'p_vmax': 0.5,
            'c_vmax': 1,
            'width': 50,
            'height': 50,
            'box_width': 1,
            'p_sen_range': 6,
            'p_comm_range': 12,
            'c_sen_range': 6,
            'c_comm_range': 12,
            # 'target': [17.5, 17.5],
            'kill_radius': 0.5,
            'phi_lmt': np.pi / 4,
            'p_v_lmt': 0.2,
            'c_v_lmt': 0.4
        }
        self.p_obs_dim = 4
        self.c_obs_dim = 4
        self.env_name = 'ParticleEnvBoundGra'
        self.p_vmax = cfg['p_vmax']
        self.c_vmax = cfg['c_vmax']
        self.p_v_lmt = cfg['p_v_lmt']
        self.c_v_lmt = cfg['c_v_lmt']
        self.p_phi_lmt = cfg['phi_lmt']
        self.c_phi_lmt = cfg['phi_lmt']
        self.width = cfg['width']
        self.height = cfg['height']
        self.box_width = cfg['box_width']
        self.p_sen_range = cfg['p_sen_range']
        self.p_comm_range = cfg['p_comm_range']
        self.c_sen_range = cfg['c_sen_range']
        self.c_comm_range = cfg['c_comm_range']

        self.target = None
        self.kill_radius = cfg['kill_radius']
        self.random = np.random
        # self.random.seed(10086)

        self.state_dim = 4 + 4
        # action space
        self.action_dim = 1

        self.n_episode = 0
        self.episode_limit = 250
        self.shadow_epi = 1000
        self.target_return = 1000

        self.step_size = 0.5
        self.time_step = 0

        self.curriculum = False
        self.p_num = None
        self.c_num = None
        self.p_list = {}
        self.p_idx = []
        self.c_list = {}
        self.c_idx = []
        self.state = None

        self.global_map = None
        self.global_obstacles = list()
        self.obstacle_num = None
        
        self.block_num = 5
        self.obstacle_shape = (6, 7)
        self.max_boundary_obstacle_num = self.block_num * (6 * 7 - 4 * 5)
        self.boundary_obstacles = list()
        self.boundary_obstacle_num = None

    def initialize_client(self):
        # generate collision-free initial positions
        sample = list()
        while len(sample) < self.c_num:
            pos = np.random.normal(loc=self.width / 2, scale=5, size=(2,)).clip([0, 0], [self.width - 1, self.height - 1])
            if self.global_map.is_unoccupied(tuple(pos)):
                collision = 0
                for p in sample:
                    if np.linalg.norm(pos - p) < self.kill_radius:
                        collision += 1
                        break
                if collision == 0:
                    sample.append(pos)
        
        for idx, p in enumerate(sample):
            x, y = p
            self.c_list.append(
                Client(
                    idx=idx, time_step=self.time_step, tau=self.tau, DOF=2,
                    x=x, y=y, z=0, 
                    v=0, theta=0, phi=0,
                    d_v_lmt=self.c_v_lmt, d_theta_lmt=self.c_theta_lmt, d_phi_lmt=self.c_phi_lmt,
                    v_max=self.c_vmax, sen_range=self.c_sen_range, comm_range=self.c_comm_range
                )
            )
            self.c_idx.append(idx)

    def initialize_server(self):
        pass

    def reset(self, p_num=3, c_num=6, worker_id=-1):
        # initialize global map
        self.global_map = OccupiedGridMap(
            is3D=False,
            boundaries=(self.width, self.height)
        )
        
        self.global_map.initailize_obstacle(num=10, center=self.width / 2)
        
        obstacle_map, boundary_map, obstacles, boundary_obstacles = self.global_map.get_map()
        # array, array, list, list

        self.boundary_obstacles = boundary_obstacles
        self.global_obstacles = obstacles
        self.obstacle_num = len(obstacles)
        self.boundary_obstacle_num = len(boundary_obstacles)

        # initialize client
        self.c_num = c_num
        self.c_list = dict()
        self.c_idx = list()
        self.initialize_client()

        # initialize pursuer and evader
        self.p_num = p_num
        self.p_list = {}
        self.p_idx = []
        self.initialize_server()
        
        self.time_step = 0
        self.n_episode += 1

        p_state = self.get_team_state(is_pursuer=True, active=True)
        dynamic_map.set_moving_obstacle(pos=p_state)
        dynamic_map.extended_moving_obstacles()

        front_time = time.time()
        e_pos = [self.width - 1 - self.target[0], self.height - 1 - self.target[1]]
        e_pos = self.gen_init_e_pos(e_pos, dynamic_map)
        # print(f'worker_{worker_id} generate evader positioin, time cost = {front_time - time.time()}')

        for i in range(self.e_num):
            self.e_list[f'{i}'] = Evader(
                idx=i,
                x=e_pos[i][0],
                y=e_pos[i][1],
                phi=np.pi / 4,
                phi_lmt=self.e_phi_lmt,
                v=self.e_vmax,
                v_max=self.e_vmax,
                v_lmt=self.e_v_lmt,
                sen_range=self.e_sen_range,
                comm_range=self.e_comm_range,
                global_map=deepcopy(self.global_map),
                target=self.target
                )
            self.e_idx.append(i)

    def step(self, action):
        self.time_step += 1
        idx = 0
        for pursuer_idx in self.p_idx:
            pur = self.p_list[f'{pursuer_idx}']
            pur.step(self.step_size, action[idx])
            idx += 1

        reward = self.reward(True)
        self.update_agent_active()
        active = self.get_active()
        done = True if self.get_done() or self.time_step >= self.episode_limit else False

        return reward, done, active

    def evader_step(self, worker_id=-1):
        # Based On Dynamic D Star Algorithm
        p_state = self.get_team_state(is_pursuer=True, active=True)
        for evader_idx in self.e_idx:
            evader = self.e_list[f'{evader_idx}']
            path, way_point, pred_map = evader.replan(p_states=p_state, time_step=self.time_step, worker_id=worker_id)
            # if worker_id >= 0:
            #     print(f'worker_{worker_id} replan')

            phi = evader.waypoint2phi(way_point)
            evader.step(self.step_size, phi)
        return path, pred_map

    def get_done(self):
        for idx in self.e_idx:
            agent = self.e_list[f'{idx}']
            if np.linalg.norm([agent.x - self.target[0], agent.y - self.target[1]]) <= self.kill_radius:
                return True

        p_alive, e_alive = 0, 0
        for idx in self.p_idx:
            if self.p_list[f'{idx}'].active:
                p_alive += 1
        if p_alive == 0:
            return True

        for idx in self.e_idx:
            if self.e_list[f'{idx}'].active:
                e_alive += 1
        if e_alive == 0:
            return True

        return False

    def get_active(self):
        active = []
        for idx in self.p_idx:
            active.append(1 if self.p_list[f'{idx}'].active else 0)
        return active

    def agent_reward(self, agent_idx, is_pursuer=True):
        reward = 0
        is_collision = self.collision_detection(agent_idx=agent_idx, is_pursuer=is_pursuer, obstacle_type='evaders')
        reward += sum(is_collision) * 1

        inner_collision = self.collision_detection(agent_idx=agent_idx, is_pursuer=is_pursuer, obstacle_type='pursuers')
        reward -= (sum(inner_collision) - 1) * 1

        obstacle_collision = self.collision_detection(agent_idx=agent_idx, is_pursuer=is_pursuer, obstacle_type='static_obstacles')
        reward -= (sum(obstacle_collision)) * 1
        return reward

    def reward(self, is_pursuer):
        reward = []
        for idx in self.p_idx:
            if self.p_list[f'{idx}'].active:
                reward.append(self.agent_reward(idx, is_pursuer))
            else:
                reward.append(0)
        return reward

    def update_agent_active(self):
        p_idx_list = list()
        p_alive_list = list()
        for idx in self.p_idx:
            if self.p_list[f'{idx}'].active:
                p_idx_list.append(idx)
                p_p_collision = self.collision_detection(agent_idx=idx, is_pursuer=True, obstacle_type='pursuers')
                p_e_collision = self.collision_detection(agent_idx=idx, is_pursuer=True, obstacle_type='evaders')
                p_o_collision = self.collision_detection(agent_idx=idx, is_pursuer=True, obstacle_type='static_obstacles')
                p_alive_list.append(bool(sum(p_p_collision) - 1 + sum(p_e_collision) + sum(p_o_collision)))

        e_idx_list = list()
        e_alive_list = list()
        for idx in self.e_idx:
            if self.e_list[f'{idx}'].active:
                e_idx_list.append(idx)
                e_e_collision = self.collision_detection(agent_idx=idx, is_pursuer=False, obstacle_type='evaders')
                e_p_collision = self.collision_detection(agent_idx=idx, is_pursuer=False, obstacle_type='pursuers')
                e_o_collision = self.collision_detection(agent_idx=idx, is_pursuer=False, obstacle_type='static_obstacles')
                e_alive_list.append(bool(sum(e_e_collision) - 1 + sum(e_p_collision) + sum(e_o_collision)))
        
        for i, idx in enumerate(p_idx_list):
            agent = self.p_list[f'{idx}']
            if p_alive_list[i]:
                agent.active = False
                agent.x = 1000
                agent.y = 1000
                agent.phi = 0
                agent.v = 0

        for i, idx in enumerate(e_idx_list):
            agent = self.e_list[f'{idx}']
            if e_alive_list[i]:
                agent.active = False
                agent.x = 1000
                agent.y = 1000
                agent.phi = 0
                agent.v = 0

    def get_agent_state(self, is_pursuer, idx, relative=False):
        agent = self.p_list[f'{idx}'] if is_pursuer else self.e_list[f'{idx}']
        if relative:
            phi = agent.phi
            v = agent.v
            vx = v * np.cos(phi)
            vy = v * np.sin(phi)
            return [agent.x, agent.y, vx, vy]
        else:
            return [agent.x, agent.y, agent.phi, agent.v]

    def get_team_state(self, is_pursuer, active=True, relative=False):
        if active:
            team_state = list()
            idx_list = self.p_idx if is_pursuer else self.e_idx
            agent_list = self.p_list if is_pursuer else self.e_list
            for idx in idx_list:
                if agent_list[f'{idx}'].active:
                    team_state.append(self.get_agent_state(is_pursuer, idx, relative=relative))
        else:
            idx_list = self.p_idx if is_pursuer else self.e_idx
            team_state = [self.get_agent_state(is_pursuer, idx, relative=relative) for idx in idx_list]
        return team_state

    def communicate(self):
        """
        the obstacles have no impact on the communication between agents
        :return: adj_mat: the adjacent matrix of the agents
        """
        p_states = self.get_team_state(is_pursuer=True, active=False)
        adj_mat = np.zeros(shape=(self.p_num, self.p_num))
        for i, item_i in enumerate(p_states):
            agent_list = self.p_list
            if agent_list[f'{i}'].active:
                for j, item_j in enumerate(p_states):
                    if agent_list[f'{j}'].active:
                        if np.linalg.norm([item_i[0] - item_j[0], item_i[1] - item_j[1]]) <= self.p_comm_range:
                            adj_mat[i, j] = 1
        adj_mat = adj_mat.tolist()
        return adj_mat

    def sensor(self, evader_pos):
        obstacle_adj_list = list()
        evader_adj_list = list()
        for idx in self.p_idx:
            pursuer = self.p_list[f'{idx}']
            obstacle_adj, evader_adj = pursuer.sensor(
                boundary_obstacles=self.boundary_obstacles,
                evader_pos=evader_pos, 
                max_boundary_obstacle_num=self.max_boundary_obstacle_num
            )
            obstacle_adj_list.append(obstacle_adj)
            evader_adj_list.append(evader_adj)
        return obstacle_adj_list, evader_adj_list

    def collision_detection(self, agent_idx, is_pursuer, obstacle_type: str = 'static_obstacles'):
        agent_state = self.get_agent_state(is_pursuer, agent_idx)
        if obstacle_type == 'static_obstacles':
            obstacles = self.global_obstacles  # list containing coordination of static obstacles
        elif obstacle_type == 'pursuers':
            obstacles = self.get_team_state(is_pursuer=True, active=True)  # only active pursuer is referenced
        else:
            obstacles = self.get_team_state(is_pursuer=False, active=True)

        is_collision = list()
        for obstacle in obstacles:
            if np.linalg.norm([obstacle[0] - agent_state[0], obstacle[1] - agent_state[1]]) <= self.kill_radius:
                is_collision.append(1)
            else:
                is_collision.append(0)

        return is_collision
