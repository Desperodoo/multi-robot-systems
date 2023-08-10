from Occupied_Grid_Map import OccupiedGridMap
from Sensor import Sensor
import numpy as np
import random
            

class Pursuer():
    def __init__(self, idx:int, x:int, y:int, config,phi, phi_lmt, v, v_max, v_lmt, sen_range, fov):
        self.time_step = 0
        self.idx = idx
        self.x = x
        self.y = y
        self.phi = config['phi']
        self.delta_phi_max = config['phi_lmt']
        self.v = config['v']
        self.v_max = config['v_max']
        self.v_lmt = config['v_lmt']
        self.sensor_range = config['sen_range']
        self.active = True
        self.slam = Sensor(36,self.sensor_range,config['fov'])

    def step(self, step_size, a):
        # a belong to [0, 1, 2, 3, 4, 5, 6, 7, 8]
        if a == 0:
            v = 0
        else:
            # v clip
            v = self.v_max
            delta_v = v - self.v
            delta_v = np.clip(delta_v, -self.v_lmt, self.v_lmt)
            v = self.v + delta_v
            # phi clip
            a = a * np.pi / 4
            if a > np.pi:
                a -= 2 * np.pi
            sign_a_phi = np.sign(a * self.phi)
            if sign_a_phi >= 0:
                delta_phi = abs(a - self.phi)
                sign = np.sign(a - self.phi)
            else:
                if abs(a - self.phi) < 2 * np.pi - abs(a - self.phi):
                    delta_phi = abs(a - self.phi)
                    sign = np.sign(a - self.phi)
                else:
                    delta_phi = 2 * np.pi - abs(a - self.phi)
                    sign = -np.sign(a - self.phi)

            delta_phi = np.clip(delta_phi, 0, self.delta_phi_max)
            self.phi = self.phi + sign * delta_phi

            if self.phi > np.pi:
                self.phi -= 2 * np.pi
            elif self.phi < -np.pi:
                self.phi += 2 * np.pi
        
        if self.active:
            x = self.x + v * np.cos(self.phi) * step_size
            y = self.y + v * np.sin(self.phi) * step_size
            if self.grid_map.in_bounds((int(x), int(y))):
                self.x = x
                self.y = y
                self.v = v

    def dynamic(self, u, order=1, DOF=2):
      """The dynamic of the agent is considered as a 1-order system with 2/3 DOF.
      The input dimension is the same as the state dimension
      Args:
          u (float): The desired velocity.
          order (int, optional): The order of the response characteristic of the velocity. Defaults to 1.
          DOF (int, optional): Degree of freedom. Defaults to 2.
      """
      self.v = (u - self.v) * (1 - np.exp(self.time_step / self.tau)) + self.v * np.exp(self.time_step / self.tau)
      self.x += self.v[0] * self.time_step
      self.y += self.v[1] * self.time_step
      if DOF == 3:
          self.z += self.v[2] * self.time_step
    
    def sensor(self, boundary_obstacles, evader_pos, max_boundary_obstacle_num):
        # return obstacle_adj as list with shape of (obstacle_num, )
        if int(self.x) == 1000:
            obstacle_adj = [0] * max_boundary_obstacle_num
            evader_adj = [0] * len(evader_pos)
        else:
            obstacle_adj = self.raser_map[int(self.x)][int(self.y)].tolist()
            evader_adj = self.radar.rescan(
                x=int(self.x), 
                y=int(self.y), 
                boundary_obstacles=boundary_obstacles, 
                evader_pos=evader_pos, 
                obstacle_adj=obstacle_adj
            )
        return obstacle_adj, evader_adj

class navigate_env:
    def __init__(self,map_size:tuple, is3D:bool, agent_num:int,spread:int,map_config, agent_config) -> None:
        """
        Args:
          map_size: (x,y) size of the map
          agent_num: int number of agent in the environment
          spread: int when agents are initialized, spread measure how close the agent will stick together
        """
        self.map_size = map_config['map_size']
        self.is3D = map_config['is3D']
        self.spread = map_config['spread']
        self.agent_num = agent_num
        self.agent_config = agent_config
        self.field_radius = map_config['field_radius']
        self.field_cof = map_config['field_cof']
        self.agent_list = []
    def init_client(self):
        center_x = random.randrange(self.map_size[0])
        center_y = random.randrange(self.map_size[1])
        for i in range(self.agent_num):
            point_x = np.random.normal(center_x,self.spread,1).clip(center_x - 10, center_x + 10)
            point_y = np.random.normal(center_y,self.spread,1).clip(center_y - 10,center_y + 10)
            agent = Pursuer(i,point_x,point_y,self.agent_config)
            self.agent_list.append(agent)
            

    def reset(self):
        self.occupaied_map = OccupiedGridMap(is3D=bool,boundaries=self.map_size)
        self.occupaied_map.initailize_obstacle(5)
        self.agent_list = []
        self.init_client()
        return 
        "return state: still not defined yet"

    def get_force_field(self,pos_x,pos_y): 
        radar = Sensor(num_beams=36,radius=self.field_radius,horizontal_fov= 2 * np.pi)
        beam_ranges, local_map, obstacle_positions,direction = radar.get_local_sensed_map(self.occupaied_map, (pos_x,pos_y), 0.0)
        sum_force_x = 0
        sum_force_y = 0
        for angle, obstacles in enumerate(direction,obstacles):
            distance = np.sqrt(abs(obstacles[0] - pos_x[0]) ** 2 + abs(obstacles[1] - pos_y) ** 2)
            value_x = self.field_cof * distance * np.sin(angle)
            value_y = self.field_cof * distance * np.cos(angle)
            sum_force_x += value_x
            sum_force_y += value_y
        total_force = np.sqrt(sum_force_x ** 2 + sum_force_y ** 2)
        
        angle = np.arcsin(total_force / sum_force_y)
        return total_force, angle

            

