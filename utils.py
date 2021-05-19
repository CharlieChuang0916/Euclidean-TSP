import math
import random
import matplotlib.pyplot as plt
import visualize_tsp

def dist(coord_0, coord_1):
    """
    Euclidean distance between two nodes.
    """
    return math.sqrt((coord_0[0] - coord_1[0]) ** 2 + (coord_0[1] - coord_1[1]) ** 2)

def fitness(coords, solution):
    """
    Total distance of the current solution path.
    """
    N = len(coords)
    cur_fit = 0
    for i in range(N):
        cur_fit += dist(coords[solution[i % N]], coords[solution[(i + 1) % N]])
    return cur_fit

def visualize_routes(solution, coords):
    """
    Visualize the TSP route with matplotlib.
    """
    visualize_tsp.plotTSP([solution], coords)

def plot_learning(fitness_list):
    """
    Plot the fitness through iterations.
    """
    plt.plot([i for i in range(len(fitness_list))], fitness_list)
    plt.ylabel("Fitness")
    plt.xlabel("Iteration")
    plt.show()

def random_solution(coords):
    solution = list(range(len(coords)))
    random.shuffle(solution)
    fit = fitness(coords, solution)
    return solution, fit
