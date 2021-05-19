import random

from utils import fitness, random_solution
from greedy import Greedy

class GeneticAlgo(object):
    def __init__(self, coords, population_size=10, stopping_iter=-1, k=4, p_cross=0.1, p_mutate=0.1):
        self.coords = coords
        self.N = len(coords)
        self.stopping_iter = 100000 if stopping_iter == -1 else stopping_iter
        self.iteration = 1
        self.population_size = population_size

        self.nodes = [i for i in range(self.N)]

        self.cur_solution = None
        self.cur_fitness = None

        self.best_solution = None
        self.best_fitness = float("Inf")
        self.fitness_list = []

        self.cur_population = None
        self.k = k
        self.p_cross = p_cross
        self.p_mutate = p_mutate

    def initial_population(self):
        """
        Initial Population
        """
        # Initialize with the greedy solution.
        # population = [Greedy(self.coords).greedy()[0] for _ in range(self.population_size)]

        # Initialize with the random solution.
        population = [random_solution(self.coords)[0] for _ in range(self.population_size)]
        return population

    def evaluation(self):
        """
        Evaluate the chosen probability of population
        """
        population_fitness = [fitness(self.coords, chromosome) for chromosome in self.cur_population]
        self.cur_fitness = min(population_fitness)
        self.cur_solution = self.cur_population[population_fitness.index(self.cur_fitness)]

        if self.cur_fitness < self.best_fitness:
            self.best_solution = self.cur_solution
            self.best_fitness = self.cur_fitness

        distance = [population_fitness[i] - self.cur_fitness for i in range(len(population_fitness))]
        max_distance = max(distance)
        chosen_prob = [ (max_distance-distance[i])/(max_distance+1e-6) for i in range(len(distance))]
        if max(chosen_prob) < 1e-6:
            chosen_prob = [1] * len(chosen_prob)
        return chosen_prob

    def select(self):
        """
        Select parents from population
        """
        chosen_prob = self.evaluation()
        parents = []
        while len(parents) < self.k:
            idx = random.randint(0, len(chosen_prob)-1)
            if chosen_prob[idx] > random.uniform(0, 1):
                parents.append(self.cur_population[idx])

        return parents

    def crossover(self, parents):
        """
        Crossover with probability p_corss
        """
        children = []
        for _ in range(self.population_size):
            if random.uniform(0,1) > self.p_cross:
                children.append(self.cur_population[random.randint(0, self.population_size-1)])
            else:
                parent1 = parents[random.randint(0, self.k-1)]
                parent2 = parents[random.randint(0, self.k-1)]
                start = random.randint(0, self.N-1)
                end = random.randint(0, self.N-1)
                if start > end:
                    swap = start
                    start = end
                    end = swap
                child = [None] * self.N
                for i in range(start, end+1, 1):
                    child[i] = parent1[i]
                pointer = 0
                for i in range(self.N):
                    if child[i] is None:
                        while parent2[pointer] in child:
                            pointer += 1
                        child[i] = parent2[pointer]
                children.append(child)

        return children

    def mutate(self, children):
        """
        Mutate with probability p_mutate
        """
        new_children = []
        for child in children:
            if random.uniform(0, 1) < self.p_mutate:
                idx1 = 0
                idx2 = 0
                while idx1 == idx2:
                    idx1 = random.randint(0, self.N-1)
                    idx2 = random.randint(0, self.N-1)
                swap = child[idx1]
                child[idx1] = child[idx2]
                child[idx2] = swap
                new_children.append(child)
            else:
                new_children.append(child)
        return new_children


    def evolution(self):
        """
        Starting GA
        """
        print("Starting GA.")
        self.cur_population = self.initial_population()
        while self.iteration < self.stopping_iter:
            parents = self.select()
            children = self.crossover(parents)
            self.cur_population = self.mutate(children)

            self.iteration += 1
            self.fitness_list.append(self.cur_fitness)

        print("Best fitness obtained: ", self.best_fitness)
        improvement = 100 * (self.fitness_list[0] - self.best_fitness) / (self.fitness_list[0])
        print(f"Improvement over greedy heuristic: {improvement : .2f}%")

        return self.best_solution, self.fitness_list