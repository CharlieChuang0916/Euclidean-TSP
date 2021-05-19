import random

from utils import fitness, random_solution
from greedy import Greedy

class TabuSearch(object):
    def __init__(self, coords, stopping_iter=-1):
        self.coords = coords
        self.N = len(coords)
        self.stopping_iter = 100000 if stopping_iter == -1 else stopping_iter
        self.iteration = 1

        self.nodes = [i for i in range(self.N)]

        self.cur_solution = None
        self.cur_fitness = None

        self.best_solution = None
        self.best_fitness = float("Inf")
        self.fitness_list = []

        self.tabu_list = []

    def two_opt_swap(self, solution):
        """
        All 2-opt neighbors of the current solution.
        """
        neighbors = []

        for i in range(self.N):
            node1 = 0
            node2 = 0

            while node1 == node2:
                node1 = random.randint(1, self.N - 1)
                node2 = random.randint(1, self.N - 1)

            if node1 > node2:
                swap = node1
                node1 = node2
                node2 = swap

            tmp = solution[node1: node2]
            neighbors.append(solution[:node1] + tmp[::-1] + solution[node2:])

        return neighbors

    def search(self):
        """
        Execute tabu search algorithm.
        """
        # Initialize with the greedy solution.
        # self.cur_solution, self.cur_fitness = Greedy(self.coords).greedy()

        # Initialize with the random solution.
        self.cur_solution, self.cur_fitness = random_solution(self.coords)

        print("Starting search.")
        self.best_solution = self.cur_solution
        self.best_fitness = self.cur_fitness
        best_candidate = self.cur_solution
        self.tabu_list.append(best_candidate)
        while self.iteration < self.stopping_iter:
            neighborhood = self.two_opt_swap(best_candidate)
            best_candidate = neighborhood[0]
            for candidate in neighborhood:
                if candidate not in self.tabu_list and fitness(self.coords, candidate) < fitness(self.coords, best_candidate):
                    best_candidate = candidate

            candidate_fitness = fitness(self.coords, best_candidate)
            if candidate_fitness < self.best_fitness:
                self.best_solution = best_candidate
                self.best_fitness = candidate_fitness

            self.cur_solution, self.cur_fitness = best_candidate, candidate_fitness

            self.tabu_list.append(best_candidate)
            self.iteration += 1

            self.fitness_list.append(self.cur_fitness)

        print("Best fitness obtained: ", self.best_fitness)
        improvement = 100 * (self.fitness_list[0] - self.best_fitness) / (self.fitness_list[0])
        print(f"Improvement over greedy heuristic: {improvement : .2f}%")

        return self.best_solution, self.fitness_list