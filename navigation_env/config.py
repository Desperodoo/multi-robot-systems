import argparse
import numpy as np
map_args = argparse.ArgumentParser("Settings for map")

map_args.add_argument("--boundaries", type=tuple, default=(30,30), help="the boundaries of the map, no negative number can be inputted")
map_args.add_argument("--time_limits",type=int, default=1000, help="the simulation step in the map")
map_args.add_argument("--obstacles_num",type=int,default= 10, help="number of obstacles in the map")
map_args.add_argument("--min_range",type=float, default=100, help="the minimum distance from agents' start point to its destination")
map_args.add_argument("--map", type=list, default= None, help=" the specific map")

agent_config = argparse.ArgumentParser("Setting for the agent ")

flocking_config = argparse.ArgumentParser("setting for the flocking algorithm")
flocking_config.add_argument("--radius", type=float, default=2.0, help="the radius that other agent will be considered as nearby agent")
flocking_config.add_argument("--separate_cof",type=float, default=1, help="the cofficient of the separate varaible")
flocking_config.add_argument("--cohesion_cof",type=float, default=1, help="the cofficient of the cohesion varaible")
flocking_config.add_argument("--alignment_cof",type=float, default=1, help="the cofficient of the alignment varaible")



