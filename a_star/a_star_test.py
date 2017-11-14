# -*- coding: utf-8 -*-
import sys
from a_star import aStarBase
import argparse
import numpy as np

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
        return Node(np.array([0,0]))

    def __eq__(self, other):
        return (self.point != other.point).sum() == 0

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
        return (node.point != self.goal).sum() == 0

    def children(self, node):
        # return a list of the children of node
        out = []
        for x in range(-1, 2):
            for y in range(-1, 2):

                if x != 0 or y != 0:
                    act = np.array([x, y])
                    point = node.point + act
                    out.append(Node(point, act))
        return out

    def invalid(self, node):
        # return true if the node is invalid (outside of space, or colliding with something)
        row = node.point[0] + self.start[0]
        if row < 0 or row >= len(self.maze):
            return True

        s = self.maze[row]
        col = node.point[1] + self.start[1]
        if col < 0 or col >= len(s):
            return True

        if s[col] != ' ':
            return True

        return False

    def heuristic(self, node):
        # return the heuristic value of distance from node to the goal
        d1 = abs(node.point[0] - self.goal[0])
        d2 = abs(node.point[1] - self.goal[1])

        return max(d1, d2)

def str_to_pos(s):
    p = s.split(",", 2)
    return np.array([int(p[0]), int(p[1])])

def main(argv):
    parser = argparse.ArgumentParser(description='a_star_test')
    parser.add_argument('--maze', type=str, default="maze")
    parser.add_argument('--start', type=str, default="0,0")
    parser.add_argument('--goal', type=str, default="19,19")

    args = parser.parse_args(argv)

    with open(args.maze) as f:
        maze = f.readlines()


    start = str_to_pos(args.start)
    goal = str_to_pos(args.goal)

    astar = aStar(maze = maze, start = start)

    print(" from {} to {}".format(start, goal))
    path = astar.find_path_from_origin(goal - start, Node)
    print(list(map(lambda x: x.point + start, path)))
    print("path length {}".format(len(path)))

if __name__ == "__main__":
    main(sys.argv[1:])

