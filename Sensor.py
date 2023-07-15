import math

import numpy as np
import matplotlib.pyplot as plt
from Occupied_Grid_Map import OccupancyGridMap


class Sensor:
    def __init__(self):
        pass

    def get_local_sensed_map(self, got_global_map, pos, direction, is3D, phi_range=None, fov=None):
        global_map = got_global_map.get_map()

        num_beams = 36
        max_range = 50.0
        pos = np.array(got_global_map.get_pos(pos))
        beam_angles = np.linspace(direction - fov / 2, direction + fov / 2, num_beams)
        beam_directions = np.column_stack((np.cos(beam_angles), np.sin(beam_angles)))

        beam_ranges = np.full(num_beams, max_range)

        for i, beam_direction in enumerate(beam_directions):
            start_point = pos
            end_point = got_global_map.get_pos(pos + beam_direction * max_range)

            ray_indices = self.bresenham_line(start_point[0], start_point[1], end_point[0], end_point[1])

            for index in ray_indices:
                if global_map.get(index) is not None:
                    beam_ranges[i] = np.linalg.norm(index - start_point)
                    break

        return beam_ranges

    def bresenham_line(self, x0, y0, x1, y1):
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        err = dx - dy
        line = []

        while True:
            line.append((x0, y0))

            if math.isclose(x0, x1) and math.isclose(y0, y1):
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        return line


def test_lidar_simulator():
    # 创建一个2D地图
    map_size = (100, 100)
    global_map = OccupancyGridMap(is3D=False)

    # 在地图上设置障碍物
    grid_map_vis = np.ones(map_size)
    for r in range(55, 65):
        for c in range(55, 65):
            global_map.set_obstacle((r, c))
            grid_map_vis[r, c] = 0

    # 创建 Sensor 对象
    lidar_simulator = Sensor()

    # 设置激光雷达位置和朝向
    pos = (50, 50)
    direction = 0.0

    # 设置是否为3D情况和视野范围
    is3D = False
    fov = np.pi

    # 获取每条激光束的距离信息
    beam_ranges = lidar_simulator.get_local_sensed_map(global_map, pos, direction, is3D=is3D, fov=fov)

    # 可视化结果
    plt.figure(figsize=(6, 6))

    plt.imshow(grid_map_vis, cmap='gray', origin='lower')
    plt.plot(pos[0], pos[1], 'ro')
    plt.arrow(pos[0], pos[1], np.cos(direction), np.sin(direction), color='r', width=0.1, head_width=0.3, alpha=0.5)
    for angle, beam_range in zip(np.linspace(direction - fov / 2, direction + fov / 2, len(beam_ranges)), beam_ranges):
        beam_end_x = pos[0] + beam_range * np.cos(angle)
        beam_end_y = pos[1] + beam_range * np.sin(angle)
        plt.plot([pos[0], beam_end_x], [pos[1], beam_end_y], 'g-', alpha=0.5)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Sensor Simulation')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    test_lidar_simulator()
