import math

import numpy as np
import matplotlib.pyplot as plt
from Occupied_Grid_Map import OccupiedGridMap


def bresenham_line(x0, y0, x1, y1):
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


def bresenham_line_3d(x0, y0, z0, x1, y1, z1):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    dz = abs(z1 - z0)
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    sz = -1 if z0 > z1 else 1

    if dx >= dy and dx >= dz:
        err1 = 2 * dy - dx
        err2 = 2 * dz - dx
        points = [(x0, y0, z0)]

        while not math.isclose(x0, x1):
            if err1 > 0:
                y0 += sy
                err1 -= 2 * dx

            if err2 > 0:
                z0 += sz
                err2 -= 2 * dx

            err1 += 2 * dy
            err2 += 2 * dz
            x0 += sx
            points.append((x0, y0, z0))
    elif dy >= dx and dy >= dz:
        err1 = 2 * dx - dy
        err2 = 2 * dz - dy
        points = [(x0, y0, z0)]

        while not math.isclose(y0, y1):
            if err1 > 0:
                x0 += sx
                err1 -= 2 * dy

            if err2 > 0:
                z0 += sz
                err2 -= 2 * dy

            err1 += 2 * dx
            err2 += 2 * dz
            y0 += sy
            points.append((x0, y0, z0))
    else:
        err1 = 2 * dx - dz
        err2 = 2 * dy - dz
        points = [(x0, y0, z0)]

        while not math.isclose(z0, z1):
            if err1 > 0:
                x0 += sx
                err1 -= 2 * dz

            if err2 > 0:
                y0 += sy
                err2 -= 2 * dz

            err1 += 2 * dx
            err2 += 2 * dy
            z0 += sz
            points.append((x0, y0, z0))

    return points


class Sensor:
    MAP_UNCHECKED = 2
    MAP_OCCUPIED = 1
    MAP_FREE = 0

    def __init__(self, num_beams: int, radius: float, horizontal_fov: float, vertical_fov=None):
        self.fov = horizontal_fov
        self.phi = vertical_fov
        self.num_beams = num_beams
        self.radius = radius

    def get_local_sensed_map(self, occupancy_grid_map: OccupiedGridMap, pos: tuple, direction: float):
        """
        Get local sensed map through ray casting algorithm
        @param occupancy_grid_map: map object providing pos translation and occupancy check method, also providing resolution information
        @param pos: position in coordination (x, y(, z))
        @param direction: theta-form direction of current sensor in local coordination
        @return: beam_ranges(distance measured by each beam), obstacle coordination list [(x, y(, z))], None for free beam
        """
        pos = np.array(occupancy_grid_map.get_pos(pos))
        beam_angles = np.linspace(direction - self.fov / 2, direction + self.fov / 2, self.num_beams)
        beam_directions = np.column_stack((np.cos(beam_angles), np.sin(beam_angles)))
        print(beam_directions)
        beam_ranges = np.full(self.num_beams, self.radius)
        obstacle_positions = []
        # TODO: get from OccupancyGridMap
        local_map = np.full((100, 100), self.MAP_UNCHECKED)
        obstacle_direction = []
        for i, beam_direction in enumerate(beam_directions):
            start_point = pos
            end_point = occupancy_grid_map.get_pos(pos + beam_direction * self.radius)

            ray_indices = bresenham_line(start_point[0], start_point[1], end_point[0] - 1, end_point[1] - 1)
            for index in ray_indices:
                if occupancy_grid_map.get_map()[index] != 0:
                    beam_ranges[i] = np.linalg.norm(index - start_point)
                    obstacle_positions.append(occupancy_grid_map.index_to_pos(index))
                    local_map[index] = 1
                    local_map[index] = self.MAP_OCCUPIED
                    obstacle_direction.append(np.arcsin(beam_direction[0]))
                    break
                else:
                    local_map[index] = self.MAP_FREE

        return beam_ranges, local_map, obstacle_positions, obstacle_direction


def test_lidar_simulator():
    # 创建一个2D地图
    map_size = (100, 100)
    global_map = OccupiedGridMap(is3D=False, boundaries=map_size)

    # 在地图上设置障碍物
    grid_map_vis = np.ones(map_size)
    for r in range(55, 65):
        for c in range(55, 65):
            global_map.set_obstacle((r, c))
            grid_map_vis[r, c] = 0


    is3D = False
    fov = 2 * np.pi 
    # 创建 Sensor 对象
    lidar_simulator = Sensor(num_beams=36, radius=50, horizontal_fov=fov)

    # 设置激光雷达位置和朝向
    pos = (50, 50)
    direction = 0.0

    # 获取每条激光束的距离信息
    beam_ranges, local_map, obstacle_positions,bream_direction = lidar_simulator.get_local_sensed_map(global_map, pos, direction)
    # 可视化结果
    print(obstacle_positions)
    print(bream_direction)
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
