import numpy as np
import random
import copy


def create_individual(victims):
    # Creates a list of victims in random order
    return random.sample(victims, len(victims))


def create_population(victims, size):
    # Creates a population of different orders of victims
    return [create_individual(victims) for _ in range(size)]


def fitness(individual, base, map):
    total_distance = 0
    lastPosition = base
    for victim in individual:
        distance = len(self.astar(Node((lastPosition[0], lastPosition[1])), Node((victim[0], victim[1])), map))
        total_distance += distance
        lastPosition = victim
    total_distance += len(self.astar(Node((lastPosition[0], lastPosition[1])), Node((base[0], base[1])), map))
    return 1 / total_distance  # Fitness is higher for shorter distances


def crossover(parent1, parent2):
    # Mixing the orders from two parents to create a child
    half = len(parent1) // 2
    child = parent1[:half] + parent2[half:]
    return child


def mutate(individual, mutation_rate):
    # Swaps two victims' places with a certain chance
    for swapped in range(len(individual)):
        if (random.random() < mutation_rate):
            swap_with = int(random.random() * len(individual))

            individual[swapped], individual[swap_with] = individual[swap_with], individual[swapped]
    return individual


def next_generation(population, base, map, mutation_rate):
    scores = [(fitness(i, base, map), i) for i in population]
    scores = [i[1] for i in sorted(scores)]
    retained = scores[int(.1 * len(scores)):]  # Retaining top 10%
    retain_length = len(retained)
    desired_length = len(population) - retain_length
    children = []
    while len(children) < desired_length:
        parent1 = np.random.randint(0, retain_length - 1)
        parent2 = np.random.randint(0, retain_length - 1)
        if parent1 != parent2:
            parent1_genes = retained[parent1]
            parent2_genes = retained[parent2]
            child = crossover(parent1_genes, parent2_genes)
            child = mutate(child, mutation_rate)
            children.append(child)
    parents_and_children = retained + children
    return parents_and_children