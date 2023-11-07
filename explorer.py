# @Author: Tacla, UTFPR
# It walks randomly in the environment looking for victims.

import random
from abstract_agent import AbstractAgent
from physical_agent import PhysAgent
from abstract_agent import Node
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class Explorer(AbstractAgent):
    activeExplorers = []  # Lista de exploradores ainda ativos
    completeMap = set()  # Mapa completo
    standbyRescuers = []  # Lista de regate ainda esperando
    allvictims = []
    totalExplorers = 0

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
        # Coleciona as vitimas encontradas
        self.victims = []
        # Sequencia de passos para voltar a base
        self.wayback = []
        # Contagem de passos dados
        self.stepcount = 0.0
        # Drone retornou a base
        self.returned = False
        # modelo de classificação
        self.modelo_arvore_decisao = None

        Explorer.totalExplorers += 1

    def deliberate(self) -> bool:
        """
        The agent chooses the next action. Execute the exploration using a depth-first search (DFS) algorithm
        """

        # Se existir um caminho para voltar, então ele é seguido
        if len(self.wayback) > 0:
            nextpos = self.wayback.pop(0)
            self.body.walk(nextpos[0] - self.body.x, nextpos[1] - self.body.y)
            self.update_remaining_time(nextpos[0] - self.body.x, nextpos[1] - self.body.y)
            self.returned = True
            return True

        # Voltou para a base
        if self.returned:
            print(f"{self.NAME} {self.agentnumber} I believe I've remaining time of {self.rtime:.1f}")
            Explorer.activeExplorers.remove(self.agentnumber)
            # Adiciona todos os blocos visitados para o mapa geral
            for x in self.visited:
                Explorer.completeMap.add(x)

            for x in self.victims:
                if x not in Explorer.allvictims:
                    Explorer.allvictims.append(x)
            # Inicia os Rescuers quando todos os explorers forem finalizados
            if len(Explorer.activeExplorers) == 0:
                print(f"total of victims found: {len(Explorer.allvictims)}")
                print(f"total of cells explored: {len(Explorer.completeMap)}")

                # --------Treinar o modelo de classificação e adicionar a classificação (label) para cada vítima
                # A função Classification retorna um dataframe com as seguintes colunas ["x", "y", "id", "pSist", "pDiast", "qPA", "pulso", "fResp", "grav", "prof_label", "label"]
                # aonde a coluna label indica a críticidade classificada para determinada vitima
                Victims_Label = self.Classification(Explorer.allvictims)

                # --------

                clusters = self.Cluster(Explorer.totalExplorers, Explorer.allvictims)
                self.resc.go_save_victims(Explorer.completeMap, clusters[0])
                cluster = 1
                for duo in Explorer.standbyRescuers:
                    rescuer, victims = duo
                    rescuer.go_save_victims(Explorer.completeMap, clusters[cluster])
                    cluster += 1
            else:
                # Adds the rescuers to the standby list
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
                    if (self.body.x, self.body.y) not in self.victims:
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
            self.victims.append((self.body.x, self.body.y, vs))

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

        # Calculates the time required to return to base
        timer = 0.0
        counter = self.rtime - self.stepcount
        if (self.stepcount > self.rtime):
            # if self.lastDistanceSteps >= self.lastDistance/10:
            path = self.astar(Node((self.body.x, self.body.y)), Node((self.body.x_base, self.body.y_base)),
                              self.visited)
            lastDistance = self.calculatePathCost(path) + self.COST_DIAG * 3
            if (lastDistance >= self.rtime) & (len(self.wayback) == 0):
                print(f"Agent {self.agentnumber}: going back because distance {lastDistance}, time {self.rtime}")
                for x in path:
                    self.wayback.append(x)

    def Classification(self, allVictims):

        colunas = ["x", "y", "id", "pSist", "pDiast", "qPA", "pulso", "fResp", "grav", "prof_label"]

        # Extrai apenas as listas da tupla
        coluna_0 = [item[0] for item in allVictims]
        coluna_1 = [item[1] for item in allVictims]
        coluna_2_e_seguintes = [item[2] for item in allVictims]

        # Cria um DataFrame com as infos da tupla
        Vitimas = pd.DataFrame({'x': coluna_0, 'y': coluna_1,
                                **{f'Coluna_{i}': [x[i] for x in coluna_2_e_seguintes] for i in
                                   range(len(coluna_2_e_seguintes[0]))}})
        Vitimas.columns = colunas
        self.TrainingModel()

        Vitimas_predição = Vitimas[["qPA", "pulso", "fResp"]]

        # Use o modelo para fazer a previsão
        previsao_novas_vitimas = self.modelo_arvore_decisao.predict(Vitimas_predição)
        print(f'\nMAYCOM: tamanho previsão {len(previsao_novas_vitimas)}')

        Vitimas['label'] = previsao_novas_vitimas
        # print(f'\nNova Base Maycom: \n {Vitimas}\n')

        # Calcule a precisão do modelo
        precisao = accuracy_score(Vitimas['prof_label'], Vitimas['label'])
        print(f"MAYCOM Precisão do modelo fora de cenários simulados: {precisao}\n")

        return Vitimas

    def TrainingModel(self):
        colunas = ["id", "pSist", "pDiast", "qPA", "pulso", "fResp", "grav", "label"]

        data = pd.read_csv("./datasets/data_800vic/sinais_vitais_com_label.txt", header=None, names=colunas)
        # data.append(pd.read_csv("./datasets/data_100x80_132vic/sinais_vitais.txt", header=None, names = colunas))
        # data.append(pd.read_csv("./datasets/data_20x20_42vic/sinais_vitais_com_label.txt", header=None, names = colunas))
        # data.append(pd.read_csv("./datasets/data_12x12_10vic/sinais_vitais_com_label.txt", header=None, names = colunas))

        # print(f'Dataframe: \n{data}')
        # print(f'\nMAYCOM LEN: {data.dtypes}\n')

        # Organiza os dados em uma única matriz
        X = list(zip(data.qPA, data.pulso, data.fResp))
        y = data.label

        # Divide os dados em conjuntos de treinamento e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1, random_state=42)

        # Crie o modelo de árvore de decisão
        modelo_arvore_decisao = tree.DecisionTreeClassifier(max_depth=30)

        # Treina o modelo com os dados de treinamento
        self.modelo_arvore_decisao = modelo_arvore_decisao.fit(X_train, y_train)

        # Faz previsões com o modelo
        previsoes = modelo_arvore_decisao.predict(X_test)

        # Calcula a precisão do modelo
        precisao = accuracy_score(y_test, previsoes)
        print("Precisão do modelo treinamento:", precisao)

        return

    def Cluster(self, num, allVictims):
        centers = []
        clusters = []
        # Gerar os centros aleatorios
        for i in range(num):
            added = False
            while not added:
                position = allVictims[random.randrange(0, len(allVictims) - 1)]
                if position not in centers:
                    centers.append(position)
                    added = True
                    # inicia o cluster com os centros
        for center in centers:
            clusters.append([center])
        # adiciona cada vitima a um cluster com base nos centros
        for victim in allVictims:
            if victim in centers:
                continue
            nearestDistance = float('inf')
            nearestCluster = -1
            currentCluster = 0
            # acha o cluster mais proximo a vitima
            for cluster in clusters:
                distance = self.Heuristic(victim, cluster[0]) * (
                            5 - victim[2][6])
                if distance < nearestDistance:
                    nearestDistance = distance
                    nearestCluster = currentCluster
                currentCluster += 1
            clusters[nearestCluster].append(victim)
        return clusters