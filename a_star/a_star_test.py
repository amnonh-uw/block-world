# -*- coding: utf-8 -*-
import sys
from a_star import aStarBase
import argparse

class Node:
    def __init__(self, point, action=None, action_cost=1):
        self.point = point
        self.action = action
        self.action_cost = action_cost
        self.parent = None
        self.H = 0  # heuristic cost to get from here to goal
        self.G = 0  # cost to get here from start

    def __eq__(self, other):
        return self.point == other.point

    def __ne__(self, other):
        return (not self.__eq__(other))

    def __hash__(self):
        return hash(self.point)

class aStar(aStarBase):
    def __init__(self, maze):
        self.maze = maze

    def at_goal(self, node):
        # return true if node has reached our goal
        return node.point == self.goal

    def children(self, node):
        # return a list of the children of node
        actions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        out = []
        for act in actions:
            point = (act[0] + node.point[0], act[1] + node.point[1])
            out.append(Node(point, act))

        return out

    def invalid(self, node):
        # return true if the node is invalid (outside of space, or colliding with something)
        row = node.point[0]
        if row < 0 or row >= len(self.maze):
            return True

        s = self.maze[row]
        col = node.point[1]
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
    return (int(p[0]), int(p[1]))

def main(argv):
    parser = argparse.ArgumentParser(description='a_star_test')
    parser.add_argument('--maze', type=str, default="maze")
    parser.add_argument('--start', type=str, default="0,0")
    parser.add_argument('--goal', type=str, default="19,19")

    args = parser.parse_args(argv)

    with open(args.maze) as f:
        maze = f.readlines()

    astar = aStar(maze)

    start = str_to_pos(args.start)
    goal = str_to_pos(args.goal)

    print(" from {} to {}".format(start, goal))
    path = astar.find_path(start, goal, Node)
    print(list(map(lambda x: x.point, path)))
    print("path length {}".format(len(path)))

if __name__ == "__main__":
    main(sys.argv[1:])

