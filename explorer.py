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
        self.ttcb = 0.15 * self.TLIM     # (time to come back) indicate the time the agend needs start come back to base


        # Initialize the stack with the base position.
        self.stack = [(self.body.x_base, self.body.y_base)]
        # Initialize the set of visited positions with the base position.
        self.visited = {(self.body.x_base, self.body.y_base)}

    def euclidianDistance(self, pos1, pos2):
        dist = (((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)**0.5)
        return dist 
    
    #Implementação do A* com a distância euclediana como euristica.
    def come_back_base(self):
        
        while self.stack[-1] != self.stack[0]:
            
            #pegamos as posições do agente no momento
            agent_x = int(self.stack[-1][0])
            agent_y = int(self.stack[-1][1])
            #criamos um dicionário para guardar as informações da melhor escolha
            x_nxt_move = None
            y_nxt_move = None
            Eucledian_dist = float('inf') 

            #print(f"esse é o X {agent_x} e esse é o y {agent_y} da posição atual do agente!!")
            for x in range(-1, 2):
                for y in range(- 1, 2):
                    #selecionamos apenas os pontos ao redor do agente
                    if x != 0 or y != 0:                            
                        #print(f'Posição: ({x}, {y})')
                        #verifica se o caminho está livre ou tem obstaculo
                        if self.check_obstacle(x, y) == False:
                            if self.euclidianDistance((agent_x + x, agent_y + y), self.stack[0]) < Eucledian_dist:
                                #print(f"Distancia euclediana antes: {Eucledian_dist} e agora {self.euclidianDistance((agent_x + x, agent_y + y), self.stack[0])}")
                                Eucledian_dist = self.euclidianDistance((agent_x + x, agent_y + y), self.stack[0])
                                x_nxt_move = x #agent_x + x
                                y_nxt_move = y #agent_y + y
                                #print(f'Posição: ({x_nxt_move}, {y_nxt_move})')
            
            if self.body.walk(x_nxt_move, y_nxt_move) == PhysAgent.EXECUTED:
                self.stack.append((agent_x + x_nxt_move, agent_y + y_nxt_move)) 
                #print(f"Sucesso!!!!!!")                   

            #Se o movimento não foi efetivado significa que acabou a bateria agente
            else:
                print("movimento não realizado, acabou o tempo")
                return
        return

    def deliberate(self) -> bool:
        """
        The agent chooses the next action. Execute the exploration using a depth-first search (DFS) algorithm
        """

        # No more actions, time almost ended
        if self.rtime < self.ttcb:
            #print(f"distância euclediana: {self.euclidianDistance(self.stack[-1], self.stack[0])} ||| TimeToComeBack {self.ttcb}")
            #print(f"Tempo restante: {self.rtime} ||| TimeToComeBack {self.ttcb}\n")

            print(f"{self.NAME} I believe I've remaining time of {self.rtime:.1f}, Agent Number {self.agentnumber}")
            self.come_back_base()

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
