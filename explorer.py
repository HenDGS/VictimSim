# @Author: Tacla, UTFPR
# It walks randomly in the environment looking for victims.

import random
from abstract_agent import AbstractAgent
from physical_agent import PhysAgent
from abstract_agent import Node


class Explorer(AbstractAgent):
    activeExplorers = [] #Lista de exploradores ainda ativos
    completeMap = set() #Mapa completo
    standbyRescuers = [] #Lista de regate ainda esperando
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
        Explorer.activeExplorers.append(self.agentnumber);
        # Initialize the stack with the base position.
        self.stack = [(self.body.x_base, self.body.y_base)]
        # Initialize the set of visited positions with the base position.
        self.visited = {(self.body.x_base, self.body.y_base)}
        #Coleciona as vitimas encontradas
        self.victims = []
        #Sequencia de passos para voltar a base
        self.wayback = []
        #Contagem de passos dados
        self.stepcount = 0.0
        #Drone retornou a base
        self.returned = False

    def deliberate(self) -> bool:
        """
        The agent chooses the next action. Execute the exploration using a depth-first search (DFS) algorithm
        """
        
        # Se existir um caminho para voltar, então ele é seguido
        if len(self.wayback) > 0:
            nextpos = self.wayback.pop(0)
            self.body.walk(nextpos[0] - self.body.x, nextpos[1] - self.body.y)
            self.update_remaining_time(nextpos[0] - self.body.x,nextpos[1] - self.body.y)
            self.returned = True
            return True
        
        # Voltou para a base
        if self.returned:
            print(f"{self.NAME} {self.agentnumber} I believe I've remaining time of {self.rtime:.1f}")
            Explorer.activeExplorers.remove(self.agentnumber)
            #Adiciona todos os blocos visitados para o mapa geral
            for x in self.visited:
                Explorer.completeMap.add(x)
            #Inicia os Rescuers quando todos os explorers forem finalizados
            if len(Explorer.activeExplorers) == 0:
                self.resc.go_save_victims(Explorer.completeMap, self.victims)
                for duo in Explorer.standbyRescuers:
                    rescuer,victims = duo
                    rescuer.go_save_victims(Explorer.completeMap,victims)
            else:
                #Adds the rescuers to the standby list
                Explorer.standbyRescuers.append((self.resc, self.victims))
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
                    if (self.body.x,self.body.y) not in self.victims:
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
            self.victims.append((self.body.x,self.body.y))
            # print(f"Exp: read vital signals of {seq}")
            # print(vs)

    def update_remaining_time(self, dx, dy):
        """ Updates the remaining time of the agent after walking dx, dy steps """
        if dx != 0 and dy != 0:
            self.rtime -= self.COST_DIAG
            self.stepcount = self.stepcount + self.COST_DIAG
        else:
            self.rtime -= self.COST_LINE
            self.stepcount = self.stepcount + self.COST_LINE
        
        #Calculates the time required to return to base
        timer = 0.0
        counter = self.rtime - self.stepcount
        if self.stepcount > self.rtime:     
            path = self.astar(Node((self.body.x,self.body.y)),Node((self.body.x_base,self.body.y_base)),self.visited)
            if (self.calculatePathCost(path) * 2 >= self.rtime) & (len(self.wayback) == 0):
               for x in path:
                  self.wayback.append(x)
            



