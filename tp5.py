# # import numpy as np
# # import matplotlib.pyplot as plt
# # import random

# # # Define grid size
# # grid_size = 8
# # x_vals = np.linspace(0.10, 0.90, grid_size + 1)
# # y_vals = np.linspace(0.10, 0.90, grid_size + 1)
# # grid_points = [[x, y, 1] for x in x_vals for y in y_vals]

# # # Genetic Algorithm Parameters
# # POP_SIZE = 100
# # GENS = 200
# # MUTATION_RATE = 0.1
# # ELITISM = 5  # Keep top 5 solutions

# # # Calculate path length
# # def path_length(path):
# #     return sum(np.linalg.norm(np.array(path[i]) - np.array(path[i+1])) for i in range(len(path) - 1))

# # # Fitness function (Inverse of path length)
# # def fitness_function(path):
# #     return 1 / (path_length(path) + 1e-5)  # Ensure no division by zero

# # # Nearest Neighbor Initialization
# # def nearest_neighbor_path(start=0):
# #     unvisited = grid_points[:]
# #     path = [unvisited.pop(start)]
# #     while unvisited:
# #         nearest = min(unvisited, key=lambda p: np.linalg.norm(np.array(path[-1]) - np.array(p)))
# #         path.append(nearest)
# #         unvisited.remove(nearest)
# #     return path

# # # Initialize population with NN Heuristic
# # def initialize_population(size):
# #     population = [nearest_neighbor_path()]
# #     for _ in range(size - 1):
# #         random.shuffle(grid_points)
# #         population.append(grid_points[:])
# #     return population

# # # Tournament Selection
# # def tournament_selection(population, fitnesses, k=5):
# #     selected = random.sample(list(zip(population, fitnesses)), k)
# #     return max(selected, key=lambda x: x[1])[0]

# # # Crossover (Ordered Crossover)
# # def crossover(parent1, parent2):
# #     size = len(parent1)
# #     start, end = sorted(random.sample(range(size), 2))
# #     child = [None] * size
# #     child[start:end] = parent1[start:end]
# #     remaining = [gene for gene in parent2 if gene not in child]
    
# #     idx = 0
# #     for i in range(size):
# #         if child[i] is None:
# #             child[i] = remaining[idx]
# #             idx += 1
    
# #     return child

# # # 2-Opt Optimization
# # def two_opt(path):
# #     best = path[:]
# #     best_length = path_length(best)
# #     for i in range(1, len(path) - 2):
# #         for j in range(i + 1, len(path)):
# #             if j - i == 1: continue  # No adjacent swaps
# #             new_path = path[:]
# #             new_path[i:j] = path[j-1:i-1:-1]  # Reverse segment
# #             new_length = path_length(new_path)
# #             if new_length < best_length:
# #                 best = new_path[:]
# #                 best_length = new_length
# #     return best

# # # Mutation (Swap Two Points + 2-Opt)
# # def mutate(path):
# #     if random.random() < MUTATION_RATE:
# #         i, j = random.sample(range(len(path)), 2)
# #         path[i], path[j] = path[j], path[i]
# #     return two_opt(path)  # Apply 2-Opt after mutation

# # # Genetic Algorithm
# # def genetic_algorithm():
# #     population = initialize_population(POP_SIZE)
# #     best_individual = None
# #     best_fitness = float('-inf')

# #     for gen in range(GENS):
# #         fitnesses = [fitness_function(ind) for ind in population]
# #         new_population = []

# #         # Elitism: Preserve best individuals
# #         best_indices = np.argsort(fitnesses)[-ELITISM:]
# #         for idx in best_indices:
# #             new_population.append(population[idx])

# #         while len(new_population) < POP_SIZE:
# #             parent1 = tournament_selection(population, fitnesses)
# #             parent2 = tournament_selection(population, fitnesses)
# #             child = crossover(parent1, parent2)
# #             child = mutate(child)
# #             new_population.append(child)

# #         population = new_population
# #         best_individual = population[np.argmax(fitnesses)]
# #         best_fitness = max(fitnesses)

# #         if gen % 20 == 0:
# #             print(f"Generation {gen} - Best Path Length: {path_length(best_individual):.4f}")

# #     return best_individual

# # # Zigzag Toolpath
# # def zigzag_path():
# #     path = []
# #     for i, x in enumerate(x_vals):
# #         if i % 2 == 0:
# #             path.extend([[x, y, 1] for y in y_vals])
# #         else:
# #             path.extend([[x, y, 1] for y in reversed(y_vals)])
# #     return path

# # # Get paths
# # zigzag_toolpath = zigzag_path()
# # optimized_toolpath = genetic_algorithm()

# # # Calculate path lengths
# # zigzag_length = path_length(zigzag_toolpath)
# # optimized_length = path_length(optimized_toolpath)

# # print(f"Zigzag Path Length: {zigzag_length:.4f}")
# # print(f"GA-Optimized Path Length: {optimized_length:.4f}")
# # print(f"Improvement: {(1 - optimized_length / zigzag_length) * 100:.2f}%")

# # # Visualization
# # fig = plt.figure(figsize=(15, 8))
# # ax1 = fig.add_subplot(121, projection='3d')
# # x_vals_z, y_vals_z, z_vals_z = zip(*zigzag_toolpath)
# # ax1.plot(x_vals_z, y_vals_z, z_vals_z, color='blue', marker='o', linestyle='-', label="Zigzag")
# # ax1.set_xlim(0, 1)
# # ax1.set_ylim(0, 1)
# # ax1.set_zlim(0, 1.1)
# # ax1.set_xlabel("X Axis")
# # ax1.set_ylabel("Y Axis")
# # ax1.set_zlabel("Z Axis")
# # ax1.set_title("Zigzag Toolpath")
# # ax1.legend()

# # ax2 = fig.add_subplot(122, projection='3d')
# # x_vals_opt, y_vals_opt, z_vals_opt = zip(*optimized_toolpath)
# # ax2.plot(x_vals_opt, y_vals_opt, z_vals_opt, color='red', marker='o', linestyle='-', label="Optimized")
# # ax2.set_xlim(0, 1)
# # ax2.set_ylim(0, 1)
# # ax2.set_zlim(0, 1.1)
# # ax2.set_xlabel("X Axis")
# # ax2.set_ylabel("Y Axis")
# # ax2.set_zlabel("Z Axis")
# # ax2.set_title("GA-Optimized Toolpath")
# # ax2.legend()

# # plt.tight_layout()
# # plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# import random

# # Define grid size
# grid_size = 8
# x_vals = np.linspace(0.10, 0.90, grid_size + 1)
# y_vals = np.linspace(0.10, 0.90, grid_size + 1)
# grid_points = np.array([[x, y, 1] for x in x_vals for y in y_vals])

# # Genetic Algorithm Parameters
# POP_SIZE = 50
# GENS = 100
# MUTATION_RATE = 0.05
# ELITISM = 5  # Keep top 5 solutions

# # Calculate path length using NumPy
# def path_length(path):
#     diffs = np.diff(path, axis=0)
#     return np.sum(np.linalg.norm(diffs, axis=1))

# # Fitness function (Inverse of path length)
# def fitness_function(path):
#     return 1 / (path_length(path) + 1e-5)  # Avoid division by zero

# # Nearest Neighbor Initialization
# def nearest_neighbor_path(start=0):
#     unvisited = grid_points.tolist()
#     path = [unvisited.pop(start)]
#     while unvisited:
#         nearest = min(unvisited, key=lambda p: np.linalg.norm(np.array(path[-1]) - np.array(p)))
#         path.append(nearest)
#         unvisited.remove(nearest)
#     return np.array(path)

# # Initialize population
# def initialize_population(size):
#     population = [nearest_neighbor_path()]
#     for _ in range(size - 1):
#         np.random.shuffle(grid_points)
#         population.append(grid_points.copy())
#     return population

# # Roulette Wheel Selection
# def roulette_wheel_selection(population, fitnesses):
#     total_fitness = sum(fitnesses)
#     selection_probs = [f / total_fitness for f in fitnesses]
#     return population[np.random.choice(len(population), p=selection_probs)]

# # Crossover (Ordered Crossover)
# def crossover(parent1, parent2):
#     size = len(parent1)
#     start, end = sorted(random.sample(range(size), 2))
#     child = [None] * size
#     child[start:end] = parent1[start:end]
#     remaining = [gene for gene in parent2 if gene not in child]
#     idx = 0
#     for i in range(size):
#         if child[i] is None:
#             child[i] = remaining[idx]
#             idx += 1
#     return np.array(child)

# # 2-Opt Optimization (applied only to top 10% of population)
# def two_opt(path):
#     best = path.copy()
#     best_length = path_length(best)
#     for i in range(1, len(path) - 2):
#         for j in range(i + 1, len(path)):
#             if j - i == 1:
#                 continue  # No adjacent swaps
#             new_path = path.copy()
#             new_path[i:j] = path[j - 1 : i - 1 : -1]  # Reverse segment
#             new_length = path_length(new_path)
#             if new_length < best_length:
#                 best = new_path.copy()
#                 best_length = new_length
#     return best

# # Mutation (Swap Two Points)
# def mutate(path):
#     if random.random() < MUTATION_RATE:
#         i, j = np.random.choice(len(path), 2, replace=False)
#         path[[i, j]] = path[[j, i]]
#     return path

# # Genetic Algorithm
# def genetic_algorithm():
#     population = initialize_population(POP_SIZE)
#     best_individual = None
#     best_fitness = float('-inf')
    
#     for gen in range(GENS):
#         fitnesses = np.array([fitness_function(ind) for ind in population])
#         new_population = []
        
#         # Elitism: Preserve best individuals
#         elite_indices = fitnesses.argsort()[-ELITISM:]
#         for idx in elite_indices:
#             new_population.append(population[idx])
            
#         while len(new_population) < POP_SIZE:
#             parent1 = roulette_wheel_selection(population, fitnesses)
#             parent2 = roulette_wheel_selection(population, fitnesses)
#             child = crossover(parent1, parent2)
#             child = mutate(child)
#             new_population.append(child)
        
#         # Apply 2-Opt to top 10% of the population
#         top_10_percent = int(0.1 * POP_SIZE)
#         for i in range(top_10_percent):
#             new_population[i] = two_opt(new_population[i])
        
#         population = new_population
#         best_individual = population[np.argmax(fitnesses)]
#         best_fitness = max(fitnesses)
        
#         if gen % 20 == 0:
#             print(f"Generation {gen} - Best Path Length: {path_length(best_individual):.4f}")
    
#     return best_individual

# # Zigzag Toolpath
# def zigzag_path():
#     path = []
#     for i, x in enumerate(x_vals):
#         if i % 2 == 0:
#             path.extend([[x, y, 1] for y in y_vals])
#         else:
#             path.extend([[x, y, 1] for y in reversed(y_vals)])
#     return np.array(path)

# # Get paths
# zigzag_toolpath = zigzag_path()
# optimized_toolpath = genetic_algorithm()

# # Calculate path lengths
# zigzag_length = path_length(zigzag_toolpath)
# optimized_length = path_length(optimized_toolpath)

# print(f"Zigzag Path Length: {zigzag_length:.4f}")
# print(f"GA-Optimized Path Length: {optimized_length:.4f}")
# print(f"Improvement: {(1 - optimized_length / zigzag_length) * 100:.2f}%")

# # Visualization
# fig = plt.figure(figsize=(15, 8))
# ax1 = fig.add_subplot(121, projection='3d')
# x_vals_z, y_vals_z, z_vals_z = zip(*zigzag_toolpath)
# ax1.plot(x_vals_z, y_vals_z, z_vals_z, color='blue', marker='o', linestyle='-', label="Zigzag")
# ax1.set_xlabel("X Axis")
# ax1.set_ylabel("Y Axis")
# ax1.set_zlabel("Z Axis")
# ax1.set_title("Zigzag Toolpath")
# ax1.legend()

# ax2 = fig.add_subplot(122, projection='3d')
# x_vals_opt, y_vals_opt, z_vals_opt = zip(*optimized_toolpath)
# ax2.plot(x_vals_opt, y_vals_opt, z_vals_opt, color='red', marker='o', linestyle='-', label="Optimized")
# ax2.set_xlabel("X Axis")
# ax2.set_ylabel("Y Axis")
# ax2.set_zlabel("Z Axis")
# ax2.set_title("GA-Optimized Toolpath")
# ax2.legend()

# plt.tight_layout()
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
import random

# Define grid size
grid_size = 8
x_vals = np.linspace(0.10, 0.90, grid_size + 1)
y_vals = np.linspace(0.10, 0.90, grid_size + 1)
grid_points = [[x, y, 1] for x in x_vals for y in y_vals]

# Genetic Algorithm Parameters
POP_SIZE = 50   # Reduced for speed
GENS = 150      # More generations for better convergence
MUTATION_RATE = 0.2
ELITISM = 5     # Keep best 5 individuals

# Calculate path length
def path_length(path):
    return sum(np.linalg.norm(np.array(path[i]) - np.array(path[i+1])) for i in range(len(path) - 1))

# Fitness function (Inverse of path length)
def fitness_function(path):
    return 1 / (path_length(path) + 1e-5)  # Prevent division by zero

# Nearest Neighbor Initialization (Better initial solutions)
def nearest_neighbor_path(start=0):
    unvisited = grid_points[:]
    path = [unvisited.pop(start)]
    while unvisited:
        nearest = min(unvisited, key=lambda p: np.linalg.norm(np.array(path[-1]) - np.array(p)))
        path.append(nearest)
        unvisited.remove(nearest)
    return path

# Initialize population (Mix of NN and random solutions)
def initialize_population(size):
    population = [nearest_neighbor_path()]
    for _ in range(size - 1):
        shuffled_path = grid_points[:]
        random.shuffle(shuffled_path)
        population.append(shuffled_path)
    return population

# Tournament Selection (More efficient than roulette wheel)
def tournament_selection(population, fitnesses, k=5):
    selected = random.sample(list(zip(population, fitnesses)), k)
    return max(selected, key=lambda x: x[1])[0]

# Crossover (Fixed Ordered Crossover)
def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    
    child = [None] * size
    child[start:end] = parent1[start:end]
    
    # Fix: Convert child to tuple-based comparison
    child_tuples = [tuple(point) if point is not None else None for point in child]
    
    remaining = [gene for gene in parent2 if tuple(gene) not in child_tuples]
    
    idx = 0
    for i in range(size):
        if child[i] is None:
            child[i] = remaining[idx]
            idx += 1
    
    return child

# 2-Opt Optimization (Improves local paths)
def two_opt(path):
    best = path[:]
    best_length = path_length(best)
    
    for i in range(1, len(path) - 2):
        for j in range(i + 1, len(path)):
            if j - i == 1: continue  # No adjacent swaps
            
            new_path = path[:]
            new_path[i:j] = path[j-1:i-1:-1]  # Reverse segment
            new_length = path_length(new_path)
            
            if new_length < best_length:
                best = new_path[:]
                best_length = new_length
    
    return best

# Mutation (Swap Two Points + 2-Opt)
def mutate(path):
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(len(path)), 2)
        path[i], path[j] = path[j], path[i]
    
    return two_opt(path)  # Apply 2-Opt after mutation

# Genetic Algorithm
def genetic_algorithm():
    population = initialize_population(POP_SIZE)
    best_individual = None
    best_fitness = float('-inf')

    for gen in range(GENS):
        fitnesses = [fitness_function(ind) for ind in population]
        new_population = []

        # Elitism: Preserve best individuals
        best_indices = np.argsort(fitnesses)[-ELITISM:]
        for idx in best_indices:
            new_population.append(population[idx])

        while len(new_population) < POP_SIZE:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = new_population
        best_individual = population[np.argmax(fitnesses)]
        best_fitness = max(fitnesses)

        if gen % 10 == 0:
            print(f"Generation {gen} - Best Path Length: {path_length(best_individual):.4f}")

    return best_individual

# Zigzag Toolpath
def zigzag_path():
    path = []
    for i, x in enumerate(x_vals):
        if i % 2 == 0:
            path.extend([[x, y, 1] for y in y_vals])
        else:
            path.extend([[x, y, 1] for y in reversed(y_vals)])
    return path

# Get paths
zigzag_toolpath = zigzag_path()
optimized_toolpath = genetic_algorithm()

# Calculate path lengths
zigzag_length = path_length(zigzag_toolpath)
optimized_length = path_length(optimized_toolpath)

print(f"Zigzag Path Length: {zigzag_length:.4f}")
print(f"GA-Optimized Path Length: {optimized_length:.4f}")
print(f"Improvement: {(1 - optimized_length / zigzag_length) * 100:.2f}%")

# Visualization
fig = plt.figure(figsize=(15, 8))
ax1 = fig.add_subplot(121, projection='3d')
x_vals_z, y_vals_z, z_vals_z = zip(*zigzag_toolpath)
ax1.plot(x_vals_z, y_vals_z, z_vals_z, color='blue', marker='o', linestyle='-', label="Zigzag")
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_zlim(0, 1.1)
ax1.set_xlabel("X Axis")
ax1.set_ylabel("Y Axis")
ax1.set_zlabel("Z Axis")
ax1.set_title("Zigzag Toolpath")
ax1.legend()

ax2 = fig.add_subplot(122, projection='3d')
x_vals_opt, y_vals_opt, z_vals_opt = zip(*optimized_toolpath)
ax2.plot(x_vals_opt, y_vals_opt, z_vals_opt, color='red', marker='o', linestyle='-', label="Optimized")
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_zlim(0, 1.1)
ax2.set_xlabel("X Axis")
ax2.set_ylabel("Y Axis")
ax2.set_zlabel("Z Axis")
ax2.set_title("GA-Optimized Toolpath")
ax2.legend()

plt.tight_layout()
plt.show()
