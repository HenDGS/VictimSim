##  ABSTRACT AGENT
### @Author: Tacla (UTFPR)
### It has the default methods for all the agents supposed to run in
### the environment

from itertools import count
from math import sqrt
import os
import random
import heapq
from abc import ABC, abstractmethod
from physical_agent import PhysAgent


class AbstractAgent:
    """ This class represents a generic agent and must be implemented by a concrete class. """
    
    
    def __init__(self, env, config_file):
        """ 
        Any class that inherits from this one will have these attributes available.
        @param env referencia o ambiente
        @param config_file: the absolute path to the agent's config file
        """
       
        self.env = env              # ref. to the environment
        self.body = None            # ref. to the physical part of the agent in the environment
        self.NAME = ""              # the name of the agent
        self.TLIM = 0.0             # time limit to execute (cannot be exceeded)
        self.COST_LINE = 0.0        # cost to walk one step hor or vertically
        self.COST_DIAG = 0.0        # cost to walk one step diagonally
        self.COST_READ = 0.0        # cost to read a victim's vital sign
        self.COST_FIRST_AID = 0.0   # cost to drop the first aid package to a victim
        self.COLOR = (100,100,100)  # color of the agent
        self.TRACE_COLOR = (140,140,140) # color for the visited cells
        
        # Read agents config file for controlling time
        with open(config_file, "r") as file:

            # Read each line of the file
            for line in file:
                # Split the line into words
                words = line.split()

                # Get the keyword and value
                keyword = words[0]
                if keyword=="NAME":
                    self.NAME = words[1]
                elif keyword=="COLOR":
                    r = int(words[1].strip('(), '))
                    g = int(words[2].strip('(), '))
                    b = int(words[3].strip('(), '))
                    self.COLOR=(r,g,b)  # a tuple
                elif keyword=="TRACE_COLOR":
                    r = int(words[1].strip('(), '))
                    g = int(words[2].strip('(), '))
                    b = int(words[3].strip('(), '))
                    self.TRACE_COLOR=(r,g,b)  # a tuple
                elif keyword=="TLIM":
                    self.TLIM = float(words[1])
                elif keyword=="COST_LINE":
                    self.COST_LINE = float(words[1])
                elif keyword=="COST_DIAG":
                    self.COST_DIAG = float(words[1])
                elif keyword=="COST_FIRST_AID":
                    self.COST_FIRST_AID = float(words[1])
                elif keyword=="COST_READ":    
                    self.COST_READ = float(words[1])
                    
        # Register within the environment - creates a physical body
        # Starts in the ACTIVE state
        self.body = env.add_agent(self, PhysAgent.ACTIVE)

    
    def Heuristic(self,position,goal) -> float:
        """ 
            Calculates the heuristic value from the target position to the goal
        """
        px, py = position[0], position[1]
        dx, dy = goal[0], goal[1]
        return sqrt(pow(abs(px - dx),2) + pow(abs(py - dy),2))

    def get_neighbors(self,node, grid):
        x,y = node.position
        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0),(-1, -1),(-1, 1),(1, -1),(1, 1)]:
            nx,ny = x + dx, y + dy
            if (nx,ny) in grid:
                cost = 1.0
                if dx != 0 & dy != 0:
                    cost = 1.5
                neighbors.append(Node((nx,ny),parent=node,cost=cost + node.cost))
        
        return neighbors
    
    def calculatePathCost(self,path) -> float:
        curentPos = (self.body.x,self.body.y)
        counter = 0.0
        for Pos in path:
            dx,dy = curentPos[0] - Pos[0], curentPos[1] - Pos[1]
            curentPos = Pos
            if (dx != 0) & (dy!= 0):
                counter += self.COST_DIAG
            else:
                counter += self.COST_LINE  
        return counter

    def astar(self,start, goal, grid) -> list:
        open_list = []
        closed_list = set()

        heapq.heappush(open_list, (start.cost, start))
    
        while open_list:
            current_cost, current_node = heapq.heappop(open_list)

            if current_node.position == goal.position:
                # Goal reached, construct and return the path
                path = []
                while current_node:
                    path.append(current_node.position)
                    current_node = current_node.parent
                path = path[::-1]
                return path[1:]

            closed_list.add(current_node.position)

            for neighbor in self.get_neighbors(current_node,grid):
                if neighbor.position in closed_list:
                    continue

                moveCost = 1.0
                if current_node.position[0] - neighbor.position[0] != 0 & current_node.position[1] - neighbor.position[1] != 0:
                    moveCost = 1.5
                new_cost = current_node.cost + moveCost
                if neighbor not in open_list:
                    heapq.heappush(open_list, (new_cost + self.Heuristic(neighbor.position, goal.position), neighbor))
                elif new_cost < neighbor.cost:
                    neighbor.cost = new_cost
                    neighbor.parent = current_node
        return []

    @abstractmethod
    def deliberate(self) -> bool:
        """ This is the choice of the next action. The simulator calls this
        method at each reasonning cycle if the agent is ACTIVE.
        Must be implemented in every agent
        @return True: there's one or more actions to do
        @return False: there's no more action to do """

        pass
    
class Node:
    """ This class represents a node in a graphic. """
    def __init__(self, position, parent=None, cost=0.0):
        self.position = position
        self.parent = parent
        self.cost = cost
        
    def __lt__(self, other):
        return self.cost < other.cost
