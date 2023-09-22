# _*_ coding: utf-8 _*_
import imageio.v2 as imageio
import string
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from scipy.interpolate import make_interp_spline

"""
from envi import sim_moving

# 画出来的是仿真动态图，gif格式
sim_moving(step, n, h_x, h_y, h_e, ep, e_ser, c_r)

# step：整个仿真运行的步数
# n：Pursuer的个数
# h_x：记录下来的每一个Pursuer的x坐标值（历史数据），n行，step列
# h_y：记录下来的每一个Pursuer的y坐标值（历史数据），n行，step列
# h_e：记录下来的Evader的xy坐标值（历史数据），step行，2列
# ep：Evader要到达的目标点位置坐标
# e_ser：Evader的感知半径
# c_r：Pursuer的抓捕半径

"""


def draw_obstacle(x, y, box_width, color):
    left, bottom, wid, hei = (x - box_width / 2, y - box_width / 2, box_width, box_width)
    rect = patches.Rectangle(xy=(left, bottom), width=wid, height=hei, color=color)
    return rect


def draw_circle(x, y, r, color, line_style, line_width):
    circle = patches.Circle(xy=(x, y), radius=r, edgecolor=color, linestyle=line_style, linewidth=line_width,
                            facecolor='none')
    return circle


def draw_waypoint(x, y):
    path = Path(vertices=np.array([[x - 0.15, y], [x, y - 0.15], [x + 0.25, y], [x, y + 0.25]]))
    waypoint = patches.PathPatch(path, color='red', alpha=0.5)
    return waypoint


# 绘制动态图
def sim_moving(step, width, height, obstacles, boundary_obstacles, box_width, n_p, n_e, p_x, p_y, e_x, e_y, path,
               target, e_ser, c_r, p_p_adj, p_e_adj, p_o_adj, dir, extended_obstacles=None):
    """

    :param step: time steps, type=int
    :param height: height of the map, type=int
    :param width: width of the map, type=int
    :param obstacles: list of obstacles, shape=(n, 2)
    :param boundary obstacles: list of boundary of the obstacles
    :param exteneded obstacles : list of extended obstacles
    :param box_width: describe the mesh density, type=int
    :param n_p: number of pursuers
    :param n_e: number of evaders
    :param p_x: x position of pursuers, shape=(steps, n_p)
    :param p_y: y position of pursuers, shape=(steps, n_p)
    :param e_x: x position of evaders, shape=(steps, n_e)
    :param e_y: y position of evaders, shape=(steps, n_e)
    :param path: list of waypoints of evader, shape=(steps, n, 2)
    :param target: goal of the evader
    :param e_ser: sensor range of the evader
    :param c_r: kill radius of the pursuers
    :param p_p_adj: adjacent matrix of pursuers
    :param p_e_adj: adjacent matrix of evader
    :param p_o_adj: adjacent matrix of obstacles
    :param dir: save dir
    :return:
    """
    fig3 = plt.figure(3, figsize=(5, 5 * 0.9))
    ax3 = fig3.add_axes(rect=[0.12, 0.1, 0.8, 0.82])
    image_list = list()

    # p_p_adj = np.array(p_p_adj)
    # p_e_adj = np.array(p_e_adj)
    # p_o_adj = np.array(p_o_adj)

    # p_p_adj = p_p_adj[:, 0, :].tolist()
    # p_e_adj = p_e_adj[:, 0, :].tolist()
    # p_o_adj = p_o_adj[:, 0, :].tolist()

    for i in range(step):
        ax3.cla()

        ax3.set_title('Trajectory', size=12)
        ax3.set_xlabel('x/(m)', size=12)
        ax3.set_ylabel('y/(m)', size=12)
        ax3.grid(color='black', linestyle='--', linewidth=0.8)
        ax3.axis([0, height - 1, 0, width - 1])
        ax3.set_aspect('equal')
        # draw evader start point and target point
        ax3.scatter(e_x[0], e_y[0], color='green', edgecolors='white', marker='s')
        ax3.scatter(target[i][0], target[i][1], color='white', edgecolors='red', marker='^')
        # for j in range(n_e):
        #     ax3.plot([e_x[0, j], target[0]], [e_y[0, j], target[1]], color='purple', linestyle='--', alpha=0.3)

        # draw extended obstacles
        if extended_obstacles is not None:
            for obstacle in extended_obstacles[i]:
                rect = draw_obstacle(x=obstacle[0], y=obstacle[1], box_width=box_width, color='grey')
                ax3.add_patch(rect)
        
        # draw trajectory
        k = 40
        for j in range(n_p):
            if i >= 1:
                if i >= k:
                    p_x_data = p_x[i - k:i, j]
                    p_y_data = p_y[i - k:i, j]
                    # e_x_data = p_x[i - k:i, j]
                    # e_y_data = p_y[i - k:i, j]
                else:
                    p_x_data = p_x[:i + 1, j]
                    p_y_data = p_y[:i + 1, j]
                    # e_x_data = p_x[:i, j]
                    # e_y_data = p_y[:i, j]            
                for l in range(len(p_x_data) - 1):                    
                    ax3.plot(p_x_data[l:l+2], p_y_data[l:l+2], color='green', linewidth=0.5, alpha=l / len(p_x_data) / 4)
        k = 30
        for j in range(n_e):
            if i >= 1:
                if i >= k:
                    e_x_data = e_x[i - k:i, j]
                    e_y_data = e_y[i - k:i, j]
                else:
                    e_x_data = e_x[:i + 1, j]
                    e_y_data = e_y[:i + 1, j]
                for l in range(len(e_x_data) - 1):                    
                    ax3.plot(e_x_data[l:l+2], e_y_data[l:l+2], color='red', linewidth=0.5, alpha=l / len(p_x_data) / 4)
        
        # draw obstacles
        for obstacle in obstacles:
            rect = draw_obstacle(x=obstacle[0], y=obstacle[1], box_width=box_width, color='black')
            ax3.add_patch(rect)

        # for j in range(n_p):
        #     rect = draw_obstacle(x=round(p_x[i, j]), y=round(p_y[i, j]), box_width=box_width, color='grey')
        #     ax3.add_patch(rect)

        # draw agents
        ax3.plot(p_x[i, :], p_y[i, :], 'o', color='green', markersize=4 * c_r)
        ax3.plot(e_x[i, :], e_y[i, :], 'o', color='red', markersize=4 * c_r)

        # # draw pursuer kill radius
        # for j in range(n_p):
        #     circle = draw_circle(x=p_x[i, j], y=p_y[i, j], r=c_r, color='green', line_style='-', line_width=0.5)
        #     ax3.add_patch(circle)

        # draw evader sensor range
        for j in range(n_e):
            circle = draw_circle(x=e_x[i, j], y=e_y[i, j], r=e_ser, color='red', line_style='--', line_width=0.5)
            ax3.add_patch(circle)

        # draw path
        for j in range(n_e):
            for waypoint in path[i][j]:
                wp = draw_waypoint(waypoint[0], waypoint[1])
                ax3.add_patch(wp)

        # draw communication
        p_adj = p_p_adj[i]
        e_adj = p_e_adj[i]
        o_adj = p_o_adj[i]
        for p1, a in enumerate(p_adj):
            for p2, connected in enumerate(a):
                if (p1 > p2) and connected:
                    ax3.plot([p_x[i, p1], p_x[i, p2]], [p_y[i, p1], p_y[i, p2]], color='green', linestyle='--', alpha=0.3)

        for p1, a in enumerate(e_adj):
            for p2, connected in enumerate(a):
                if connected:
                    ax3.plot([p_x[i, p1], e_x[i, p2]], [p_y[i, p1], e_y[i, p2]], color='red', linestyle='--', alpha=0.3)
  
        for p1, a in enumerate(o_adj):
            for p2, connected in enumerate(a):
                if connected:
                    ax3.plot([p_x[i, p1], boundary_obstacles[p2][0]], [p_y[i, p1], boundary_obstacles[p2][1]], color='black', linestyle='--', alpha=0.2)

        # for j, connected in enumerate(p_p_adj[i]):
        #     if connected:
        #         ax3.plot([p_x[i, 0], p_x[i, j]], [p_y[i, 0], p_y[i, j]], color='green', linestyle='--', alpha=0.5)

        # for j, connected in enumerate(p_e_adj[i]):
        #     if connected:
        #         ax3.plot([p_x[i, 0], e_x[i, j]], [p_y[i, 0], e_y[i, j]], color='red', linestyle='--', alpha=0.5)

        # for j, connected in enumerate(p_o_adj[i]):
        #     if connected:
        #         ax3.plot([p_x[i, 0], boundary_obstacles[j][0]], [p_y[i, 0], boundary_obstacles[j][1]], color='black', linestyle='--', alpha=0.5)

        fig3.savefig(dir + '.png')
        image_list.append(imageio.imread(dir + '.png'))

    imageio.mimsave(dir + '.gif', image_list, 'GIF', duration=0.07)


def draw_current_map(width, height, box_width, p_x, p_y, e_x, e_y, target, obstacles, path):
    fig3 = plt.figure(4, figsize=(5, 5 * 0.9))
    ax3 = fig3.add_axes(rect=[0.12, 0.1, 0.8, 0.82])

    ax3.set_title('Trajectory', size=12)
    ax3.set_xlabel('x/(m)', size=12)
    ax3.set_ylabel('y/(m)', size=12)
    ax3.grid(color='b', linestyle='--', linewidth=0.25)
    ax3.axis([0, width, 0, height])
    ax3.set_aspect('equal')

    ax3.scatter(target[0], target[1], color='white', edgecolors='red', marker='^')

    ax3.plot(p_x, p_y, 'o', color='green')
    ax3.plot(e_x, e_y, 'o', color='blue')

    for obstacle in obstacles:
        rect = draw_obstacle(x=obstacle[0], y=obstacle[1], box_width=box_width, color='black')
        ax3.add_patch(rect)

    path = np.array(path).T
    ax3.scatter(path[0], path[1], color='white', edgecolors='blue', marker='^')

    fig3.savefig('current_map.png')