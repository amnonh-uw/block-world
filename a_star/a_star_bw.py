# -*- coding: utf-8 -*-
from a_star.a_star import aStarBase
import numpy as np
import argparse

class Node:
    def __init__(self, point, action=None, action_cost=1):
        self.point = point
        self.action = action
        self.action_cost = action_cost
        self.parent = None
        self.H = 0  # heuristic cost to get from here to goal
        self.G = 0  # cost to get here from start

    @staticmethod
    def origin():
        return Node(np.array([0,0,0]))

    def __eq__(self, other):
        eq =  (self.point != other.point).sum() == 0
        x = "=" if eq else "!="
        print("{}{}{}".format(self.point,x,other.point))
        return eq

    def __ne__(self, other):
        return (not self.__eq__(other))

    def __hash__(self):
        return hash(tuple(self.point))

    def __str__(self):
        return str(self.point)

class aStar(aStarBase):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def at_goal(self, node):
        # return true if node has reached our goal

        return  (abs(self.goal - node.point*self.step_size) > self.reach_minimum).sum() == 0

    def children(self, node):
        # return a list of the children of node
        out = []
        for x in range(-1, 2):
            for y in range(-1, 2):
                for z in range(-1, 2):

                    if x != 0 or y != 0 or z != 0:
                        act = np.array([x, y, z])
                        point = node.point + act
                        out.append(Node(point, act))
        return out

    def invalid(self, node):
        # return true if the node is invalid (outside of space, or colliding with something)

        t =  self.env.check_occupied(node.point * self.step_size)
        x = "occupied" if t else ""
        print("occpancy check {} {}".format(node.point, x))

        if t:
            print("OCCUPIED!!!!!!!")
        return t

    def heuristic(self, node):
        # return the heuristic value of distance from node to the goal
        d1 = abs(node.point[0] - self.goal[0])
        d2 = abs(node.point[1] - self.goal[1])

        return max(d1, d2)

def find_next_step(env, goal, step_size=0.1, reach_minimum=0.1):
    start = env.finger_pos
    print("find_path {}->{}".format(start, goal))

    astar = aStar(env=env, step_size=step_size, reach_minimum=reach_minimum)

    path = astar.find_path_from_origin(goal - start, Node)

    print("path {}".format(list(map(lambda x: x.point + start, path))))
    print("path length {}".format(len(path)))

    return(path[1] * env.step_size)
