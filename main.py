from anneal import SimAnneal
from tabu import TabuSearch
from genetic import GeneticAlgo
from greedy import Greedy
from approx import ApproxTSP

from utils import visualize_routes, plot_learning

import random
import os
import sys

def generate_random_coords(num_nodes):
    return [[random.uniform(-1000, 1000), random.uniform(-1000, 1000)] for i in range(num_nodes)]

def read_elem(filename):
    with open(filename) as f:
        return [str(elem) for elem in f.read().split()]

def read_file(filename):
    file_it = iter(read_elem(filename))

    # The input files follow the TSPLib "explicit" format.
    nb_cities = 0
    coords = []
    validtype = False
    for pch in file_it:
        if pch == "DIMENSION:":
            nb_cities = int(next(file_it))
        if pch == "DIMENSION":
            next(file_it)
            nb_cities = int(next(file_it))
        if pch == "EDGE_WEIGHT_TYPE:":
            if str(next(file_it))=="EUC_2D":
                validtype = True
        if pch == "EDGE_WEIGHT_TYPE":
            next(file_it)
            if str(next(file_it))=="EUC_2D":
                validtype = True
        if pch == "NODE_COORD_SECTION":
            if validtype is True:
                for i in range(nb_cities):
                    idx = int(next(file_it))
                    x = float(next(file_it))
                    y = float(next(file_it))
                    coords.append([x, y])

    return validtype, coords, nb_cities

if __name__ == "__main__":
    path = sys.argv[1]
    algo = sys.argv[2]
    print(algo)
    if algo not in ["SA", "tabu", "GA", "greedy", "approx"]:
        raise ValueError("invalid algorithm")
    for file in os.listdir(path):
        filename = os.path.join(path, file)
        root, extension = os.path.splitext(filename)
        if extension == ".tsp":
            valid, coords, nb_cities = read_file(filename) # generate_random_coords(num_nodes)
            if valid is True and nb_cities < 2000:
                print("Path = :", filename)
                print("#City = :", nb_cities)
                if algo == "SA":
                    sa = SimAnneal(coords)
                    solution, fitness_list = sa.anneal()
                if algo == "tabu":
                    ts = TabuSearch(coords, stopping_iter=500)
                    solution, fitness_list = ts.search()
                if algo == "GA":
                    ga = GeneticAlgo(coords, stopping_iter=1000)
                    solution, fitness_list = ga.evolution()
                if algo == "greedy":
                    g = Greedy(coords)
                    solution, fitness = g.greedy()
                    fitness_list = [fitness, fitness]
                if algo == "approx":
                    ap = ApproxTSP(coords)
                    solution, fitness = ap.approximate()
                    fitness_list = [fitness, fitness]
                visualize_routes(solution, coords)
                plot_learning(fitness_list)