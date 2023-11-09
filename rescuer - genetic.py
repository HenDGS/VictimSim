##  RESCUER AGENT
### @Author: Tacla (UTFPR)
### Demo of use of VictimSim

import os
import random
from abstract_agent import AbstractAgent
from physical_agent import PhysAgent
from abc import ABC, abstractmethod
from abstract_agent import Node
import pyswarms as ps
import pygad
from sklearn.cluster import KMeans
import numpy as np


## Classe que define o Agente Rescuer com um plano fixo
class Rescuer(AbstractAgent):
    rescuedVictims = []  # Set de vitimas resgatadas com sucesso
    activeRescuers = []  # Lista de rescuer ativos

    def __init__(self, env, config_file, agentNumber):
        """
        @param env: a reference to an instance of the environment class
        @param config_file: the absolute path to the agent's config file"""

        super().__init__(env, config_file)

        # Specific initialization for the rescuer
        self.plan = []  # a list of planned actions
        self.rtime = self.TLIM  # for controlling the remaining time
        self.map = []
        self.victims = []
        self.agentNumber = agentNumber
        # Starts in IDLE state.
        # It changes to ACTIVE when the map arrives
        self.body.set_state(PhysAgent.IDLE)

        # planning
        self.__planner()

    def go_save_victims(self, walls, victims):
        """ The explorer sends the map containing the walls and
        victims' location. The rescuer becomes ACTIVE. From now,
        the deliberate method is called by the environment"""
        Rescuer.activeRescuers.append(self.agentNumber)
        self.map = walls
        self.victims = victims
        self.body.set_state(PhysAgent.ACTIVE)
        self.__planner()

    # def __planner(self):
    #     """ A private method that calculates the walk actions to rescue the
    #     victims. Further actions may be necessary and should be added in the
    #     deliberata method"""
    #
    #     # This is a off-line trajectory plan, each element of the list is
    #     # a pair dx, dy that do the agent walk in the x-axis and/or y-axis
    #
    #     # Calculate the order of the victims
    #     allvictims = self.victims.copy()
    #     victimsPath = []
    #     lastPosition = (self.body.x, self.body.y)
    #
    #     # Enquanto tiver vitimas
    #     # while allvictims:
    #     #     shortest = 99999.0
    #     #     nearestVictim = (0, 0)
    #     #     # Compara a distancia entre todas as vitimas para achar a mais proxima
    #     #     for victim in allvictims:
    #     #         distance = len(
    #     #             self.astar(Node((lastPosition[0], lastPosition[1])), Node((victim[0], victim[1])), self.map))
    #     #         if distance < shortest:
    #     #             shortest = distance
    #     #             nearestVictim = victim
    #     #     # Adiciona a vitima mais proxima a lista e usa ela como referencia para achar a proxima vitima
    #     #     victimsPath.append(nearestVictim)
    #     #     allvictims.remove(nearestVictim)
    #     #     lastPosition = nearestVictim
    #     # lastPosition = (self.body.x, self.body.y)
    #     # victimsPath.append((self.body.x_base, self.body.y_base))
    #     # # Adiciona a ordem de movimento no self.plan
    #     # for victim in victimsPath:
    #     #     path = self.astar(Node((lastPosition[0], lastPosition[1])), Node((victim[0], victim[1])), self.map)
    #     #     lastCalculatedPos = lastPosition
    #     #     for x in path:
    #     #         self.plan.append((x[0] - lastCalculatedPos[0], x[1] - lastCalculatedPos[1]))
    #     #         lastCalculatedPos = x
    #     #     lastPosition = victim
    #
    #     while allvictims:
    #         shortest = float('inf')
    #         nearestVictim = (0, 0, 0)
    #         # Compara a distancia entre todas as vitimas para achar a mais proxima
    #         for victim in allvictims:
    #             distance = len(
    #                 self.astar(Node((lastPosition[0], lastPosition[1])), Node((victim[0], victim[1])), self.map)
    #             )
    #             modified_distance = distance * victim[-1][7]
    #             if modified_distance < shortest:
    #                 shortest = modified_distance
    #                 nearestVictim = victim
    #         # Adiciona a vitima mais proxima a lista e usa ela como referencia para achar a proxima vitima
    #         victimsPath.append(nearestVictim)
    #         allvictims.remove(nearestVictim)
    #         lastPosition = nearestVictim[:2]
    #     lastPosition = (self.body.x, self.body.y)
    #     victimsPath.append((self.body.x_base, self.body.y_base))
    #     # Adiciona a ordem de movimento no self.plan
    #     for victim in victimsPath:
    #         path = self.astar(Node((lastPosition[0], lastPosition[1])), Node((victim[0], victim[1])), self.map)
    #         lastCalculatedPos = lastPosition
    #         for x in path:
    #             self.plan.append((x[0] - lastCalculatedPos[0], x[1] - lastCalculatedPos[1]))
    #             lastCalculatedPos = x
    #         lastPosition = victim
    #
    #     """
    #     self.plan.append((0,1))
    #     self.plan.append((1,1))
    #     self.plan.append((1,0))
    #     self.plan.append((1,-1))
    #     self.plan.append((0,-1))
    #     self.plan.append((-1,0))
    #     self.plan.append((-1,-1))
    #     self.plan.append((-1,-1))
    #     self.plan.append((-1,1))
    #     self.plan.append((1,1))
    #     """

    def __planner(self):
        """ A private method that calculates the walk actions to rescue the
        victims. Further actions may be necessary and should be added in the
        deliberate method"""

        all_victims = self.victims.copy()
        number_of_victims = len(all_victims)

        # Genetic Algorithm specifics
        population_size = 100
        generations = 100

        # creating initial population
        population = [self.create_gnome(number_of_victims) for _ in range(population_size)]

        for _ in range(generations):
            population = sorted(population, key=self.calculate_fitness)

            # If the best chromosome has a fitness equals to 0, our Genetic Algorithm has found the optimal solution
            if self.calculate_fitness(population[0]) == 0:
                break

            new_generation = []

            # Elitism: We are going to keep the 10% of our population
            elitism_size = (10 * population_size) // 100
            new_generation.extend(population[:elitism_size])

            # The rest 90% of our population is going to be filled with the offspring of two parents
            crossover_size = (90 * population_size) // 100
            for _ in range(crossover_size):
                parent1 = random.choice(population[:population_size // 2])
                parent2 = random.choice(population[:population_size // 2])
                child_chromosome = self.crossover_OX1(parent1, parent2)
                new_generation.append(child_chromosome)

            population = new_generation

        # At this point, our first chromosome on the population list should be the one with the optimal solution
        optimal_chromosome = population[0]

        # We are going to sort the victims list based on our optimal chromosome
        victims_path = [all_victims[i] for i in optimal_chromosome]

        lastPosition = (self.body.x, self.body.y)
        victims_path.append((self.body.x_base, self.body.y_base))

        # Adiciona a ordem de movimento no self.plan
        for victim in victims_path:
            path = self.astar(Node((lastPosition[0], lastPosition[1])), Node((victim[0], victim[1])), self.map)
            last_calculated_pos = lastPosition
            for x in path:
                self.plan.append((x[0] - last_calculated_pos[0], x[1] - last_calculated_pos[1]))
                last_calculated_pos = x
            lastPosition = victim
        ...

    @staticmethod
    def create_gnome(len_victims):
        gnome = np.arange(len_victims)
        np.random.shuffle(gnome)
        return gnome.tolist()

    def calculate_fitness(self, chromosome):
        return sum([abs(i - v[2][7]) for v, i in zip(self.victims, chromosome)])

    def crossover(self, parent1, parent2):
        child_chromosome = []
        for gene1, gene2 in zip(parent1, parent2):

            prob = random.random()

            if prob < 0.45:
                child_chromosome.append(gene1)
            elif prob < 0.90:
                child_chromosome.append(gene2)
            else:
                child_chromosome.append(self.mutated_genes(len(parent1)))
        return child_chromosome

    def crossover_OX1(self, parent1, parent2):
        """ Creates a child using Order Crossover """
        child = [None] * len(parent1)

        # Select a random range within chromosome
        start, stop = sorted(random.sample(range(len(parent1)), 2))

        # Copy the subsequence from first parent to child
        child[start:stop] = parent1[start:stop]

        # Take genes from second parent not already in the child, maintain their order
        p2_remain = [gene for gene in parent2 if gene not in child]

        # Add these genes to child, before and after the copied subsequence
        child[:start] = p2_remain[:start]
        child[stop:] = p2_remain[start:]
        return child

    @staticmethod
    def mutated_genes(length):
        return np.random.randint(low=0, high=length)

    def deliberate(self) -> bool:
        """ This is the choice of the next action. The simulator calls this
        method at each reasonning cycle if the agent is ACTIVE.
        Must be implemented in every agent
        @return True: there's one or more actions to do
        @return False: there's no more action to do """

        # No more actions to do
        if self.plan == []:  # empty list, no more actions to do
            Rescuer.activeRescuers.remove(self.agentNumber)
            if len(Rescuer.activeRescuers) == 0:
                print(f"Vitimas resgatadas ({len(Rescuer.rescuedVictims)}):\n(id,x,y,gravidade,label)")
                for x, y, data in Rescuer.rescuedVictims:
                    print(f"{data[0]},{x},{y},{data[6]},{data[7]}")
            return False

        # Takes the first action of the plan (walk action) and removes it from the plan
        dx, dy = self.plan.pop(0)

        # Walk - just one step per deliberation
        result = self.body.walk(dx, dy)

        # Rescue the victim at the current position
        if result == PhysAgent.EXECUTED:
            # check if there is a victim at the current position
            seq = self.body.check_for_victim()
            if seq >= 0:
                res = self.body.first_aid(seq)  # True when rescued
                if res:
                    for victim in self.victims:
                        if [self.body.x, self.body.y] == [victim[0], victim[1]]:
                            if victim not in Rescuer.rescuedVictims:
                                Rescuer.rescuedVictims.append(victim)
                            break
        return True
