# -*- coding: utf-8 -*-

class aStarBase:
    def at_goal(self, node):
        # return true if node has reached our goal
        pass

    def children(self, node):
        # return a list of the children of node
        pass

    def invalid(self, node):
        # return true if the node is invalid (outside of space, or colliding with something)
        pass

    def heuristic(self, node):
        # return the heuristic value of distance from node to the goal
        # the heuristic cannot over estimate the distance
        pass

    def find_path(self, start_point, goal_point, Node):
        start = Node(start_point)
        self.goal = goal_point

        openset = set()
        closedset = set()

        current = start
        openset.add(current)

        while openset:
            # Find the item in the open set with the lowest G + H score
            current = min(openset, key=lambda o: o.G + o.H)

            if self.at_goal(current):
                path = []
                while current.parent:
                    path.append(current)
                    current = current.parent
                path.append(current)
                return path[::-1]

            openset.remove(current)
            closedset.add(current)

            # Loop through the node's children/siblings
            for node in self.children(current):
                if node in closedset:
                    # already in the closed set, skip it
                    continue

                if self.invalid(node):
                    closedset.add(node)
                    continue

                if node in openset:
                    # already in the open set
                    new_g = current.G + node.action_cost
                    if node.G > new_g:
                        node.G = new_g
                        node.parent = current
                else:
                    # new node
                    node.G = current.G + node.action_cost
                    node.H = self.heuristic(node)
                    node.parent = current
                    openset.add(node)

        print("no path found")