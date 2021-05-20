from utils import dist, fitness

class ApproxTSP(object):
    def __init__(self, coords, stopping_iter=-1):
        self.coords = coords
        self.N = len(coords)

        self.nodes = [i for i in range(self.N)]

        self.best_solution = None
        self.best_fitness = float("Inf")

        self.W = []

    def prim_mst(self):
        d = []
        for i in range(self.N):
            di = []
            for j in range(self.N):
                dij = float("Inf") if i==j else dist(self.coords[i], self.coords[j])
                di.append(dij)
            d.append(di)

        MST_parent = [None] * self.N
        cost = [float("Inf")] * self.N
        visit = [False] * self.N

        cost[0] = 0
        MST_parent[0] = -1
        for edge_count in range(self.N-1):

            min_dist = float("Inf")
            cur_node = None
            for node in range(self.N):
                if not visit[node] and cost[node] < min_dist:
                    cur_node = node
                    min_dist = cost[node]
            visit[cur_node] = True

            for adj_node in range(self.N):
                if not visit[adj_node] and d[cur_node][adj_node] < cost[adj_node]:
                    MST_parent[adj_node] = cur_node
                    cost[adj_node] = d[cur_node][adj_node]

        T = [[] for _ in range(self.N)]
        for i in range(1, self.N):
            j = MST_parent[i]
            if j != -1:
                T[j].append(i)
        return T


    def preorder_tree_walk(self, node, T):
        self.W.append(node)
        for adj in T[node]:
            self.preorder_tree_walk(adj, T)


    def approximate(self):
        """
        2-approximation algorithm
        """
        MST = self.prim_mst()
        self.preorder_tree_walk(0, MST)

        visit = [False] * self.N

        solution = []
        for node in self.W:
            if not visit[node]:
                solution.append(node)
                visit[node] = True

        self.best_solution = solution
        self.best_fitness = fitness(self.coords, self.best_solution)

        return self.best_solution, self.best_fitness