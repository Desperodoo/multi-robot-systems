import sys
sys.path.append('utils')
from Occupied_Grid_Map import OccupiedGridMap
from Sensor import Sensor
from flocking import flocking_ago
from agent import Pursuer
import numpy as np
import random
            
"""
agent in the env
"""


class navigate_env:
    def __init__(self,map_config, agent_config,flocking_config) -> None:
        """
        Args:
          map_size: (x,y) size of the map
          agent_num: int number of agent in the environment
          spread: int when agents are initialized, spread measure how close the agent will stick together
        """
        self.map_size = map_config['map_size']
        self.is3D = map_config['is3D']
        self.spread = map_config['spread']
        self.agent_num = agent_config['agent_num']
        self.agent_config = agent_config
        self.field_radius = map_config['field_radius']
        self.field_cof = map_config['field_cof']
        self.time_limit = map_config['time_limit']
        self.agent_list = []
        self.flock_ago = flocking_ago(flocking_config)
        

    def init_client(self):
        """
        initailize the agent in the occupied in the occupancy grid map 
        1. pick a center point randomly 
        2. init the agent around the center point with normal distribution 
        """
        center_x = random.randrange(self.map_size[0])
        center_y = random.randrange(self.map_size[1])
        for i in range(self.agent_num):
            point_x = np.random.normal(center_x,self.spread,1).clip(0, self.map_size[0] - 1)[0]
            point_y = np.random.normal(center_y,self.spread,1).clip(0, self.map_size[1] - 1)[0]
            agent = Pursuer(i,point_x,point_y,self.agent_config)
            self.occupaied_map.set_map_info((point_x,point_y), 2)
            self.agent_list.append(agent)
            

    def reset(self):
        self.time = 0
        self.occupaied_map = OccupiedGridMap(is3D=self.is3D,boundaries=self.map_size)
        self.occupaied_map.initailize_obstacle(5)
        self.agent_list = []
        self.agent_map = np.zeros(self.map_size)
        # does not consider the distance from starting point to the endpoint 
        self.end_point =( random.randrange(self.map_size[0]),random.randrange(self.map_size[1]))
        self.init_client()
        "return state: still not defined yet"
    
    def get_done(self):
        if self.time >= self.time_limit: 
            return True
        for agent in self.agent_list:
            if np.linalg.norm([agent.x - self.end_point[0],agent.y - self.end_point[1]]) < 2:
                return True
        return False
    
    
    def get_reward(self, collision, agent, curpos):
        """
            reward function: currently based on the distance between ending point and agent's position
            when the agent hits each other or the obstacles will get a penalty 
        """
        if collision:
            return -10
        distance = np.linalg.norm([agent.x - curpos[0], agent.y - curpos[1]]) 
        if distance > 0:
            return 10
        else:
            return -5       

    def get_force_field(self,pos_x,pos_y): 
        """
        calculate the force at point (pos_x, pos_y)
        the magnitude of the force depend on the distance between obstacles and point
        """
        #only consider the obstacle did not consider other agent's position
        radar = Sensor(num_beams=36,radius=self.field_radius,horizontal_fov= 2 * np.pi)
        beam_ranges, local_map, obstacle_positions,direction = radar.get_local_sensed_map(self.occupaied_map, (pos_x,pos_y), 0.0)
        sum_force_x = 0
        sum_force_y = 0
        for idx in range(len(direction)):
            distance = np.sqrt(abs(obstacle_positions[idx][0] - pos_x) ** 2 + abs(obstacle_positions[idx][1] - pos_y) ** 2)
            # get the force from X and Y 
            value_x = self.field_cof * distance * np.sin(direction[idx])
            value_y = self.field_cof * distance * np.cos(direction[idx])
            sum_force_x += value_x
            sum_force_y += value_y
        # compute the sum force 
        total_force = np.sqrt(sum_force_x ** 2 + sum_force_y ** 2) * self.field_cof
        # find the angle 
        fin_angle = 0
        if sum_force_y == 0:
            fin_angle = 0
        elif sum_force_x == 0:
            fin_angle = np.pi / 2
        else:
            fin_angle = np.arcsin(sum_force_x / sum_force_y)
        return total_force, fin_angle
    def is_collision(self, x, y):
        if not self.occupaied_map.in_bound((x,y)):
            return True
        return not self.occupaied_map.is_unoccupied((x, y))
    
    def get_agent_state(self,idx):
        agent = self.agent_list[idx]
        beam_ranges, local_map, obstacle_positions,bream_direction = agent.slam.get_local_sensed_map(self.occupaied_map, (agent.x, agent.y), 0.0)
        return [agent.x, agent.y, agent.phi, agent.v], obstacle_positions
    
    # 返回一个所有AGENT共享的全局地图
    def communicate(self):
        agent_states = []
        for idx in range(self.agent_num):
            agent_state, obstacle_pos = self.get_agent_state(idx)
            agent_states.append(agent_state)
            # 共享地图的实现并不完善 无法识别移动物体
            for pos in obstacle_pos:
                x = round(pos[0])
                y = round(pos[1])
                self.agent_map[x - 1][y - 1] = 1
        return agent_states
    
    def step(self,action_list):
        self.time += 1
        idx = 0
        reward = []
        flock_force, flock_angle = self.flock_ago.flocking_algorithm(self.agent_list)
        for agent, force, angle in zip(self.agent_list, flock_force,flock_angle):
            force, angle = self.get_force_field(agent.x,agent.y)
            x, y, v = agent.step(0.5,action_list[idx], force,angle)
            x = x + force * np.cos(angle)
            y = y + force * np.sin(angle)
            if not self.is_collision(x,y):
                self.occupaied_map.set_map_info((agent.x, agent.y),0)
                agent.x = x
                agent.y = y
                agent.v = v
                self.occupaied_map.set_map_info((agent.x, agent.y),2)
                reward.append(self.get_reward(False,agent,(x,y)))
            else:
                reward.append(self.get_reward(True,None,None))
            idx += 1
        done = self.get_done()
        states = self.communicate()
        return states, self.agent_map, reward, done, 

    

