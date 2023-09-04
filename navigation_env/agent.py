import numpy as np
from utils.Sensor import Sensor

class Pursuer():
    def __init__(self, idx:int, x:int, y:int, config):
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

    def step(self, step_size, a, force, angle):
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
        
        # not mature idea adding the force directly to the position
        if self.active:
            x = self.x + v * np.cos(self.phi) * step_size + force * np.cos(angle)
            y = self.y + v * np.sin(self.phi) * step_size + force * np.sin(angle)        
        return x, y, v

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
    