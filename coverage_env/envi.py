# _*_ coding: utf-8 _*_
import imageio.v2 as imageio
import string
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.patches import PathPatch

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
    path = Path(vertices=np.array([[x - 0.5, y], [x, y - 0.5], [x + 0.5, y], [x, y + 0.5]]))
    waypoint = patches.PathPatch(path, color='red')
    return waypoint


# 绘制动态图
def sim_moving(step, height, width, obstacles, extended_obstacles, box_width, n_p, n_e, p_x, p_y, e_x, e_y, path,
               target, e_ser, c_r, p_p_adj, p_e_adj, p_o_adj):
    """

    :param step: time steps, type=int
    :param height: height of the map, type=int
    :param width: width of the map, type=int
    :param obstacles: list of obstacles, shape=(n, 2)
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
    :return:
    """
    fig3 = plt.figure(3, figsize=(5, 5 * 0.9))
    ax3 = fig3.add_axes(rect=[0.12, 0.1, 0.8, 0.82])
    image_list = list()

    p_p_adj = np.array(p_p_adj)
    p_e_adj = np.array(p_e_adj)
    p_o_adj = np.array(p_o_adj)

    p_p_adj = p_p_adj[:, 0, :].tolist()
    p_e_adj = p_e_adj[:, 0, :].tolist()
    p_o_adj = p_o_adj[:, 0, :].tolist()

    for i in range(step):
        ax3.cla()

        ax3.set_title('Trajectory', size=12)
        ax3.set_xlabel('x/(m)', size=12)
        ax3.set_ylabel('y/(m)', size=12)
        ax3.grid(color='black', linestyle='--', linewidth=0.8)
        ax3.axis([0, width, 0, height])
        ax3.set_aspect('equal')
        # draw evader start point and target point
        ax3.scatter(e_x[0], e_y[0], color='green', edgecolors='white', marker='s')
        ax3.scatter(target[0], target[1], color='white', edgecolors='red', marker='^')
        for j in range(n_e):
            ax3.plot([e_x[0, j], target[0]], [e_y[0, j], target[1]], color='purple', linestyle='--', alpha=0.3)

        # draw extended obstacles
        for obstacle in extended_obstacles[i]:
            rect = draw_obstacle(x=obstacle[0], y=obstacle[1], box_width=box_width, color='grey')
            ax3.add_patch(rect)

        # draw obstacles
        for obstacle in obstacles:
            rect = draw_obstacle(x=obstacle[0], y=obstacle[1], box_width=box_width, color='black')
            ax3.add_patch(rect)

        for j in range(n_p):
            rect = draw_obstacle(x=round(p_x[i, j]), y=round(p_y[i, j]), box_width=box_width, color='grey')
            ax3.add_patch(rect)

        # draw agents
        ax3.plot(p_x[i, :], p_y[i, :], 'o', color='green')
        ax3.plot(e_x[i, :], e_y[i, :], 'o', color='red')

        # draw pursuer kill radius
        for j in range(n_p):
            circle = draw_circle(x=p_x[i, j], y=p_y[i, j], r=c_r, color='green', line_style='-', line_width=0.5)
            ax3.add_patch(circle)

        # draw evader sensor range
        for j in range(n_e):
            circle = draw_circle(x=e_x[i, j], y=e_y[i, j], r=e_ser, color='red', line_style='--', line_width=0.5)
            ax3.add_patch(circle)

        # draw path
        for waypoint in path[i]:
            wp = draw_waypoint(waypoint[0], waypoint[1])
            ax3.add_patch(wp)

        # draw communication
        for j, connected in enumerate(p_p_adj[i]):
            if connected:
                ax3.plot([p_x[i, 0], p_x[i, j]], [p_y[i, 0], p_y[i, j]], color='green', linestyle='--', alpha=0.5)

        for j, connected in enumerate(p_e_adj[i]):
            if connected:
                ax3.plot([p_x[i, 0], e_x[i, j]], [p_y[i, 0], e_y[i, j]], color='red', linestyle='--', alpha=0.5)

        for j, connected in enumerate(p_o_adj[i]):
            if connected:
                ax3.plot([p_x[i, 0], obstacles[j][0]], [p_y[i, 0], obstacles[j][1]], color='white', linestyle='--', alpha=0.5)

        fig3.savefig('temp.png')
        image_list.append(imageio.imread('temp.png'))

    imageio.mimsave('sim_moving.gif', image_list, 'GIF', duration=0.1)


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