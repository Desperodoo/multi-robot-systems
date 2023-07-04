import numpy as np
from skimage.segmentation import find_boundaries
from grid import generate_obstacle_map, OccupancyGridMap
# from grid import generate_obstacle_map, OccupancyGridMap


def get_raser_map(boundary_map, height, width, max_num_obstacle):
    boundary_obstacles = boundary_map.obstacles
    hash_map = np.zeros((width, height, max_num_obstacle))
    beam_num = 40
    view_range = 6
    
    for x in range(width):
        for y in range(height):
            local_obstacles = list()
            for obstacle in boundary_obstacles:
                if np.linalg.norm([obstacle[0] - x, obstacle[1] - y]) <= view_range:
                    local_obstacles.append(obstacle)

            for beam in range(beam_num):
                beam_angle = beam * 2 * np.pi / beam_num
                beam_dir_x = np.cos(beam_angle)
                beam_dir_y = np.sin(beam_angle)
                for beam_range in range(view_range):
                    beam_current_x = x + beam_range * beam_dir_x
                    beam_current_y = y + beam_range * beam_dir_y
                    if (beam_current_x < 0 or beam_current_x >= width or beam_current_y < 0 or beam_current_y >= height):
                        break
                    
                    beam_current_pos = (int(beam_current_x), int(beam_current_y))
                    if not boundary_map.is_unoccupied(beam_current_pos):
                        idx = boundary_obstacles.index(beam_current_pos)
                        hash_map[x, y, idx] = 1
                        break
    
    hash_map = hash_map.tolist()
    return hash_map

    
if __name__ == '__main__':
    width = 50
    height = 50
    block_num = 5
    max_boundary_obstacle_num = block_num * (6 * 7 - 4 * 5)
    obstacle_map_list = list()
    boundary_map_list = list()
    obstacle_list = list()
    obstacle_num_list = list()
    boundary_obstacle_list = list()
    boundary_obstacle_num_list = list()
    hash_map_list = list()
    # obstacle_map.shape = (width, height), boundary_map.shape = (width, height), obstacle_map.shape = (n, 2), boundary_obstacles = (m, 2)
    for i in range(10):
        obstacle_map, boundary_map, obstacles, boundary_obstacles = generate_obstacle_map(
            x_dim=width,
            y_dim=height,
            num_obstacles=block_num
        )
        
        obstacle_map_list.append(obstacle_map.tolist())
        boundary_map_list.append(boundary_map.tolist())
        obstacle_num_list.append(len(obstacles))
        boundary_obstacle_num_list.append(len(boundary_obstacles))
        obstacle_list = obstacle_list + obstacles
        boundary_obstacle_list = boundary_obstacle_list + boundary_obstacles\
    
        grid_map = OccupancyGridMap(
            x_dim=width,
            y_dim=height,
            new_ogrid=boundary_map,
            obstacles=boundary_obstacles
        )

        hash_map = get_raser_map(boundary_map=grid_map, height=height)
        hash_map_list.append(hash_map)
    
    obstacle_map_list = np.array(obstacle_map_list)
    boundary_map_list = np.array(boundary_map_list)
    obstacle_num_list = np.array(obstacle_num_list)
    boundary_obstacle_num_list = np.array(boundary_obstacle_num_list)
    obstacle_list = np.array(obstacle_list)
    boundary_obstacle_list = np.array(boundary_obstacle_list)
    
    np.save('obstacle_map_list.npy', obstacle_map_list)
    np.save('boundary_map_list.npy', boundary_map_list)
    np.save('obstacle_num_list.npy', obstacle_num_list)
    np.save('boundary_obstacle_num_list.npy', boundary_obstacle_num_list)
    np.save('obstacle_list.npy', obstacle_list)
    np.save('boundary_obstacle_list.npy', boundary_obstacle_list)