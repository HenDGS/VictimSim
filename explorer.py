# @Author: Tacla, UTFPR
# It walks randomly in the environment looking for victims.

import random
from abstract_agent import AbstractAgent
from physical_agent import PhysAgent


class Explorer(AbstractAgent):
    activeExplorers = []

    def __init__(self, env, config_file, resc, agentnumber):
        """ Construtor do agente random on-line
        @param env referencia o ambiente
        @config_file: the absolute path to the explorer's config file
        @param resc referencia o rescuer para poder acorda-lo
        """

        super().__init__(env, config_file)

        # Specific initialization for the rescuer
        self.resc = resc  # reference to the rescuer agent
        self.rtime = self.TLIM  # remaining time to explore
        self.agentnumber = agentnumber  # number of the agent
        Explorer.activeExplorers.append(self.agentnumber)
        # Initialize the stack with the base position.
        self.stack = [(self.body.x_base, self.body.y_base)]
        # Initialize the set of visited positions with the base position.
        self.visited = {(self.body.x_base, self.body.y_base)}

    def deliberate(self) -> bool:
        """
        The agent chooses the next action. Execute the exploration using a depth-first search (DFS) algorithm
        """

        # No more actions, time almost ended
        if self.rtime < 10.0:
            print(f"{self.NAME} I believe I've remaining time of {self.rtime:.1f}")
            Explorer.activeExplorers.remove(self.agentnumber)
            if len(Explorer.activeExplorers) == 0:
                self.resc.go_save_victims([], [])
            return False

        # Make moves in all possible directions, -1 moves in left or up
        if self.agentnumber == 1:
            # down as first move
            dxs, dys = [0, 1, 0, -1, 1, 1, -1, -1], [1, 0, -1, 0, 1, -1, 1, -1]
        elif self.agentnumber == 2:
            # up as first move
            dxs, dys = [0, -1, 0, 1, 1, 1, -1, -1], [-1, 0, 1, 0, 1, -1, 1, -1]
        elif self.agentnumber == 3:
            # right as first move
            dxs, dys = [1, 0, -1, 0, 1, -1, 1, -1], [0, 1, 0, -1, 1, 1, -1, -1]
        elif self.agentnumber == 4:
            # left as first move
            dxs, dys = [-1, 0, 1, 0, 1, -1, 1, -1], [0, 1, 0, -1, 1, 1, -1, -1]

        for dx, dy in zip(dxs, dys):
            obstacle_list = self.body.check_obstacles()

            # map obstacle list to dict with "up", "up-right", "right", "down-right", "down", "down-left", "left",
            # "up-left"
            obstacle_dict = dict(
                zip(["up", "up-right", "right", "down-right", "down", "down-left", "left", "up-left"], obstacle_list))

            new_x, new_y = self.stack[-1][0] + dx, self.stack[-1][1] + dy

            # map dx, dy to direction
            if dx == 0 and dy == 1:
                direction = "down"
            elif dx == 1 and dy == 1:
                direction = "down-right"
            elif dx == 1 and dy == 0:
                direction = "right"
            elif dx == 1 and dy == -1:
                direction = "up-right"
            elif dx == 0 and dy == -1:
                direction = "up"
            elif dx == -1 and dy == -1:
                direction = "up-left"
            elif dx == -1 and dy == 0:
                direction = "left"
            elif dx == -1 and dy == 1:
                direction = "down-left"

            # if direction in obstacle_dict key is 1 or 2 it's obstacle
            if obstacle_dict[direction] == 1 or obstacle_dict[direction] == 2:
                continue

            if 0 <= new_x < self.env.dic["GRID_WIDTH"] and 0 <= new_y < self.env.dic["GRID_HEIGHT"] and (
                    new_x, new_y) not in self.visited and self.body.walk(dx, dy) == PhysAgent.EXECUTED:
                self.check_for_victim()
                self.visited.add((new_x, new_y))
                self.stack.append((new_x, new_y))
                self.update_remaining_time(dx, dy)
                break
        else:
            # If we are here it means we got stuck, unroll with stack
            if len(self.stack) > 1:
                self.stack.pop(-1)
                # move back to the previous cell
                prev_x, prev_y = self.stack[-1]
                dx, dy = prev_x - self.body.x, prev_y - self.body.y
                move_result = self.body.walk(dx, dy)
                if move_result == PhysAgent.EXECUTED:
                    self.check_for_victim()
                    self.update_remaining_time(dx, dy)
                elif move_result == PhysAgent.BUMPED:
                    print(f"{self.NAME} I bumped into a wall at {self.body.x},{self.body.y}")
        return True

    def check_for_victim(self):
        seq = self.body.check_for_victim()
        vs = []
        if seq >= 0:
            vs = self.body.read_vital_signals(seq)
            self.rtime -= self.COST_READ
            # print(f"Exp: read vital signals of {seq}")
            # print(vs)

    def update_remaining_time(self, dx, dy):
        """ Updates the remaining time of the agent after walking dx, dy steps """
        if dx != 0 and dy != 0:
            self.rtime -= self.COST_DIAG
        else:
            self.rtime -= self.COST_LINE

    def check_obstacle(self, dx, dy):
        obstacle_list = self.body.check_obstacles()
        obstacle_dict = dict(
            zip(["up", "up-right", "right", "down-right", "down", "down-left", "left", "up-left"], obstacle_list))

        if dx == 0 and dy == 1:
            direction = "down"
        elif dx == 1 and dy == 1:
            direction = "down-right"
        elif dx == 1 and dy == 0:
            direction = "right"
        elif dx == 1 and dy == -1:
            direction = "up-right"
        elif dx == 0 and dy == -1:
            direction = "up"
        elif dx == -1 and dy == -1:
            direction = "up-left"
        elif dx == -1 and dy == 0:
            direction = "left"
        elif dx == -1 and dy == 1:
            direction = "down-left"

        # if direction in obstacle_dict key is 1 or 2 it's obstacle
        if obstacle_dict[direction] == 1 or obstacle_dict[direction] == 2:
            return True
        else:
            return False
