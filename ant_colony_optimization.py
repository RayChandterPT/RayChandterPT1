import numpy as np

class AntColonyOptimization:
    def __init__(self, num_ants, num_nodes, pheromone_init, alpha=1.0, beta=1.0, evaporation_rate=0.5, num_iterations=100):
        self.num_ants = num_ants
        self.num_nodes = num_nodes
        self.pheromone_init = pheromone_init
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.num_iterations = num_iterations
        self.distance_matrix = np.zeros((num_nodes, num_nodes))
        self.pheromone_matrix = np.full((num_nodes, num_nodes), pheromone_init)
        self.best_path = []
        self.best_distance = float('inf')

    def calculate_distance(self, path):
        distance = 0
        for i in range(len(path) - 1):
            distance += self.distance_matrix[path[i]][path[i + 1]]
        return distance

    def update_pheromone(self, paths):
        self.pheromone_matrix *= (1 - self.evaporation_rate)
        for path in paths:
            distance = self.calculate_distance(path)
            for i in range(len(path) - 1):
                node1 = path[i]
                node2 = path[i + 1]
                self.pheromone_matrix[node1][node2] += 1 / distance

    def select_next_node(self, ant, available_nodes):
        pheromone_values = self.pheromone_matrix[ant.current_node][available_nodes]
        heuristic_values = 1 / self.distance_matrix[ant.current_node][available_nodes]
        probabilities = np.power(pheromone_values, self.alpha) * np.power(heuristic_values, self.beta)
        probabilities /= np.sum(probabilities)
        next_node = np.random.choice(available_nodes, p=probabilities)
        return next_node

    def construct_solution(self):
        ants = []
        for _ in range(self.num_ants):
            ant = Ant(start_node=0, num_nodes=self.num_nodes)
            ants.append(ant)

        for _ in range(self.num_nodes - 1):
            for ant in ants:
                available_nodes = ant.get_available_nodes()
                next_node = self.select_next_node(ant, available_nodes)
                ant.visit_node(next_node)

        paths = [ant.visited_nodes for ant in ants]
        return paths

    def optimize(self, distance_matrix):
        self.distance_matrix = distance_matrix

        for _ in range(self.num_iterations):
            paths = self.construct_solution()

            for path in paths:
                distance = self.calculate_distance(path)
                if distance < self.best_distance:
                    self.best_distance = distance
                    self.best_path = path

            self.update_pheromone(paths)

        return self.best_path, self.best_distance

class Ant:
    def __init__(self, start_node, num_nodes):
        self.start_node = start_node
        self.num_nodes = num_nodes
        self.visited_nodes = [start_node]
        self.available_nodes = list(range(num_nodes))
        self.available_nodes.remove(start_node)
        self.current_node = start_node

    def get_available_nodes(self):
        return self.available_nodes

    def visit_node(self, node):
        self.visited_nodes.append(node)
        self.available_nodes.remove(node)
        self.current_node = node

if __name__ == '__main__':
    # Example usage: Traveling Salesman Problem (TSP)

    distance_matrix = np.array([[0, 2, 9, 10],
                                [1, 0, 6, 4],
                                [15, 7, 0, 8],
                                [6, 3, 12, 0]])

    num_ants = 5
    num_nodes = distance_matrix.shape[0]
    pheromone_init = 0.1
    alpha = 1.0
    beta = 2.0
    evaporation_rate = 0.5
    num_iterations = 100

    aco = AntColonyOptimization(num_ants, num_nodes, pheromone_init, alpha, beta, evaporation_rate, num_iterations)
    best_path, best_distance = aco.optimize(distance_matrix)

    print("Best path:", best_path)
    print("Best distance:", best_distance)
