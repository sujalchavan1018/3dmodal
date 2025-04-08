# # output i nice figure_3 ouptut

# import numpy as np
# import matplotlib.pyplot as plt
# import random

# # Define the 5x5 grid
# grid_size = 5
# x = np.arange(grid_size)
# y = np.arange(grid_size)
# X, Y = np.meshgrid(x, y)
# points = list(zip(X.flatten(), Y.flatten()))

# # Parameters for the Genetic Algorithm
# POPULATION_SIZE = 50
# GENERATIONS = 100
# MUTATION_RATE = 0.1

# # Fitness function: Total Euclidean distance of the toolpath
# def calculate_fitness(toolpath):
#     total_distance = 0
#     for i in range(len(toolpath) - 1):
#         x1, y1 = toolpath[i]
#         x2, y2 = toolpath[i + 1]
#         total_distance += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
#     return total_distance

# # Generate a random toolpath
# def generate_random_toolpath():
#     return random.sample(points, len(points))

# # Selection: Select the best individuals based on fitness
# def selection(population, fitness_scores):
#     return random.choices(population, weights=[1 / score for score in fitness_scores], k=POPULATION_SIZE)

# # Crossover: Combine two toolpaths to create a new one
# def crossover(parent1, parent2):
#     crossover_point = random.randint(1, len(parent1) - 1)
#     child = parent1[:crossover_point]
#     for point in parent2:
#         if point not in child:
#             child.append(point)
#     return child

# # Mutation: Randomly swap two points in the toolpath
# def mutate(toolpath):
#     if random.random() < MUTATION_RATE:
#         idx1, idx2 = random.sample(range(len(toolpath)), 2)
#         toolpath[idx1], toolpath[idx2] = toolpath[idx2], toolpath[idx1]
#     return toolpath

# # Genetic Algorithm
# def genetic_algorithm():
#     # Initialize population
#     population = [generate_random_toolpath() for _ in range(POPULATION_SIZE)]
    
#     for generation in range(GENERATIONS):
#         # Calculate fitness for each individual
#         fitness_scores = [calculate_fitness(toolpath) for toolpath in population]
        
#         # Select the best individuals
#         selected_population = selection(population, fitness_scores)
        
#         # Create the next generation
#         new_population = []
#         for _ in range(POPULATION_SIZE):
#             parent1, parent2 = random.sample(selected_population, 2)
#             child = crossover(parent1, parent2)
#             child = mutate(child)
#             new_population.append(child)
        
#         population = new_population
    
#     # Return the best toolpath
#     best_toolpath = min(population, key=calculate_fitness)
#     return best_toolpath

# # Generate zig-zag toolpath for comparison
# def generate_zig_zag_toolpath():
#     toolpath = []
#     for row in range(grid_size):
#         if row % 2 == 0:
#             toolpath.extend([(col, row) for col in range(grid_size)])
#         else:
#             toolpath.extend([(col, row) for col in range(grid_size - 1, -1, -1)])
#     return toolpath

# # Compare the genetic algorithm toolpath with the zig-zag toolpath
# def compare_toolpaths():
#     # Generate zig-zag toolpath
#     zig_zag_toolpath = generate_zig_zag_toolpath()
#     zig_zag_distance = calculate_fitness(zig_zag_toolpath)
    
#     # Generate optimized toolpath using genetic algorithm
#     optimized_toolpath = genetic_algorithm()
#     optimized_distance = calculate_fitness(optimized_toolpath)
    
#     # Print results
#     print("Zig-Zag Toolpath Distance:", zig_zag_distance)
#     print("Optimized Toolpath Distance:", optimized_distance)
    
#     # Plot both toolpaths
#     plt.figure(figsize=(12, 6))
    
#     # Plot zig-zag toolpath
#     plt.subplot(1, 2, 1)
#     zig_zag_x, zig_zag_y = zip(*zig_zag_toolpath)
#     plt.plot(zig_zag_x, zig_zag_y, marker='o', color='red', label='Zig-Zag Toolpath')
#     plt.scatter(X, Y, color='blue')
#     plt.title(f'Zig-Zag Toolpath (Distance: {zig_zag_distance:.2f})')
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.grid(True)
    
#     # Plot optimized toolpath
#     plt.subplot(1, 2, 2)
#     optimized_x, optimized_y = zip(*optimized_toolpath)
#     plt.plot(optimized_x, optimized_y, marker='o', color='green', label='Optimized Toolpath')
#     plt.scatter(X, Y, color='blue')
#     plt.title(f'Optimized Toolpath (Distance: {optimized_distance:.2f})')
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.grid(True)
    
#     plt.tight_layout()
#     plt.show()

# # Run the comparison
# compare_toolpaths()

