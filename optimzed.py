import numpy as np
import matplotlib.pyplot as plt
import random

# Define the 5x5 grid
grid_size = 5
x = np.arange(grid_size)
y = np.arange(grid_size)
X, Y = np.meshgrid(x, y)
points = list(zip(X.flatten(), Y.flatten()))

# Define obstacles (list of (x, y) coordinates)
obstacles = [(1, 1), (2, 3), (3, 1)]

# Remove obstacles from the list of valid points
valid_points = [p for p in points if p not in obstacles]

# Parameters for Genetic Algorithm
POPULATION_SIZE = 1000
GENERATIONS = 700
MUTATION_RATE = 0.05  # Reduced mutation rate
ELITE_SIZE = 5  # More elitism to retain best solutions

# Fitness function: Total Euclidean distance of the toolpath
def calculate_fitness(toolpath):
    total_distance = 0
    for i in range(len(toolpath) - 1):
        x1, y1 = toolpath[i]
        x2, y2 = toolpath[i + 1]
        total_distance += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return total_distance

# Generate a random toolpath from valid points
def generate_random_toolpath():
    return random.sample(valid_points, len(valid_points))

# Tournament Selection
def selection(population):
    selected = []
    for _ in range(POPULATION_SIZE):
        i, j = random.sample(range(len(population)), 2)
        selected.append(population[i] if calculate_fitness(population[i]) < calculate_fitness(population[j]) else population[j])
    return selected

# Ordered Crossover: Preserves sequence order
def ordered_crossover(parent1, parent2):
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child = [None] * len(parent1)
    child[start:end] = parent1[start:end]
    
    ptr = end
    for i in range(len(parent2)):
        if parent2[i] not in child:
            child[ptr] = parent2[i]
            ptr = (ptr + 1) % len(parent1)
    return child

# Mutation: Swap two random points
def mutate(toolpath):
    if random.random() < MUTATION_RATE:
        idx1, idx2 = random.sample(range(len(toolpath)), 2)
        toolpath[idx1], toolpath[idx2] = toolpath[idx2], toolpath[idx1]
    return toolpath

# Genetic Algorithm
def genetic_algorithm():
    # Initialize population
    population = [generate_random_toolpath() for _ in range(POPULATION_SIZE)]
    
    # Store fitness values for each generation
    best_fitness_per_generation = []
    
    for generation in range(GENERATIONS):
        fitness_scores = [calculate_fitness(toolpath) for toolpath in population]
        
        # Keep the best individuals (Elitism)
        sorted_population = [toolpath for _, toolpath in sorted(zip(fitness_scores, population), key=lambda x: x[0])]
        elite = sorted_population[:ELITE_SIZE]
        
        # Select new population
        selected_population = selection(population)
        
        # Create next generation
        new_population = elite.copy()
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = random.sample(selected_population, 2)
            child = ordered_crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)
        
        population = new_population
        
        # Record the best fitness value for this generation
        best_fitness = min(fitness_scores)
        best_fitness_per_generation.append(best_fitness)
    
    # Plot the fitness values over generations
    plt.figure(figsize=(10, 6))
    plt.plot(range(GENERATIONS), best_fitness_per_generation, label='Best Fitness', color='blue')
    plt.xlabel('Generation')
    plt.ylabel('Fitness (Total Distance)')
    plt.title('Fitness Value Over Generations')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Return the top five toolpaths and their fitness values
    sorted_population = [toolpath for _, toolpath in sorted(zip(fitness_scores, population), key=lambda x: x[0])]
    top_five_toolpaths = sorted_population[:5]
    top_five_fitness = [calculate_fitness(toolpath) for toolpath in top_five_toolpaths]
    
    return top_five_toolpaths, top_five_fitness

# Generate zig-zag toolpath for comparison
def generate_zig_zag_toolpath():
    toolpath = []
    for row in range(grid_size):
        if row % 2 == 0:
            toolpath.extend([(col, row) for col in range(grid_size) if (col, row) not in obstacles])
        else:
            toolpath.extend([(col, row) for col in range(grid_size - 1, -1, -1) if (col, row) not in obstacles])
    return toolpath

# Compare toolpaths
def compare_toolpaths():
    zig_zag_toolpath = generate_zig_zag_toolpath()
    zig_zag_distance = calculate_fitness(zig_zag_toolpath)
    
    # Get the top five toolpaths and their fitness values
    top_five_toolpaths, top_five_fitness = genetic_algorithm()
    
    # Print the top five toolpaths and their fitness values
    print("Top Five Toolpaths and Their Fitness Values:")
    for i, (toolpath, fitness) in enumerate(zip(top_five_toolpaths, top_five_fitness)):
        print(f"Toolpath {i + 1}:")
        print("Path:", toolpath)
        print("Fitness (Total Distance):", fitness)
        print()
    
    # Plot the best toolpath
    best_toolpath = top_five_toolpaths[0]
    best_distance = top_five_fitness[0]
    
    # Print results
    print("Zig-Zag Toolpath Distance:", zig_zag_distance)
    print("Optimized Toolpath Distance:", best_distance)
    
    # Plot toolpaths
    plt.figure(figsize=(12, 6))

    # Zig-Zag Toolpath
    plt.subplot(1, 2, 1)
    zig_zag_x, zig_zag_y = zip(*zig_zag_toolpath)
    plt.plot(zig_zag_x, zig_zag_y, marker='o', color='red', label='Zig-Zag Toolpath')
    plt.scatter(X, Y, color='blue')
    obstacle_x, obstacle_y = zip(*obstacles)
    plt.scatter(obstacle_x, obstacle_y, color='black', marker='x', label='Obstacles')
    plt.title(f'Zig-Zag Toolpath (Distance: {zig_zag_distance:.2f})')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.legend()

    # Optimized Toolpath
    plt.subplot(1, 2, 2)
    optimized_x, optimized_y = zip(*best_toolpath)
    plt.plot(optimized_x, optimized_y, marker='o', color='green', label='Optimized Toolpath')
    plt.scatter(X, Y, color='blue')
    plt.scatter(obstacle_x, obstacle_y, color='black', marker='x', label='Obstacles')
    plt.title(f'Optimized Toolpath (Distance: {best_distance:.2f})')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

# Run the comparison
compare_toolpaths()



# Top Five Toolpaths and Their Fitness Values:
# Toolpath 1:
# Path: [(np.int64(2), np.int64(1)), (np.int64(2), np.int64(2)), (np.int64(3), np.int64(2)), (np.int64(4), np.int64(2)), (np.int64(3), np.int64(3)), (np.int64(4), np.int64(3)), (np.int64(4), np.int64(4)), (np.int64(3), np.int64(4)), (np.int64(2), np.int64(4)), (np.int64(1), np.int64(4)), (np.int64(0), np.int64(4)), (np.int64(0), np.int64(3)), (np.int64(1), np.int64(3)), (np.int64(1), np.int64(2)), (np.int64(0), np.int64(2)), (np.int64(0), np.int64(1)), (np.int64(0), np.int64(0)), (np.int64(1), np.int64(0)), (np.int64(2), np.int64(0)), (np.int64(3), np.int64(0)), (np.int64(4), np.int64(0)), (np.int64(4), np.int64(1))]   
# Fitness (Total Distance): 21.414213562373096

# Toolpath 2:
# Path: [(np.int64(2), np.int64(1)), (np.int64(2), np.int64(2)), (np.int64(3), np.int64(2)), (np.int64(4), np.int64(2)), (np.int64(3), np.int64(3)), (np.int64(4), np.int64(3)), (np.int64(4), np.int64(4)), (np.int64(3), np.int64(4)), (np.int64(2), np.int64(4)), (np.int64(1), np.int64(4)), (np.int64(0), np.int64(4)), (np.int64(0), np.int64(3)), (np.int64(1), np.int64(3)), (np.int64(1), np.int64(2)), (np.int64(0), np.int64(2)), (np.int64(0), np.int64(1)), (np.int64(0), np.int64(0)), (np.int64(1), np.int64(0)), (np.int64(2), np.int64(0)), (np.int64(3), np.int64(0)), (np.int64(4), np.int64(0)), (np.int64(4), np.int64(1))]   
# Fitness (Total Distance): 21.414213562373096

# Toolpath 3:
# Path: [(np.int64(2), np.int64(1)), (np.int64(2), np.int64(2)), (np.int64(3), np.int64(2)), (np.int64(4), np.int64(2)), (np.int64(3), np.int64(3)), (np.int64(4), np.int64(3)), (np.int64(4), np.int64(4)), (np.int64(3), np.int64(4)), (np.int64(2), np.int64(4)), (np.int64(1), np.int64(4)), (np.int64(0), np.int64(4)), (np.int64(0), np.int64(3)), (np.int64(1), np.int64(3)), (np.int64(1), np.int64(2)), (np.int64(0), np.int64(2)), (np.int64(0), np.int64(1)), (np.int64(0), np.int64(0)), (np.int64(1), np.int64(0)), (np.int64(2), np.int64(0)), (np.int64(3), np.int64(0)), (np.int64(4), np.int64(0)), (np.int64(4), np.int64(1))]   
# Fitness (Total Distance): 21.414213562373096

# Toolpath 4:
# Path: [(np.int64(2), np.int64(1)), (np.int64(2), np.int64(2)), (np.int64(3), np.int64(2)), (np.int64(4), np.int64(2)), (np.int64(3), np.int64(3)), (np.int64(4), np.int64(3)), (np.int64(4), np.int64(4)), (np.int64(3), np.int64(4)), (np.int64(2), np.int64(4)), (np.int64(1), np.int64(4)), (np.int64(0), np.int64(4)), (np.int64(0), np.int64(3)), (np.int64(1), np.int64(3)), (np.int64(1), np.int64(2)), (np.int64(0), np.int64(2)), (np.int64(0), np.int64(1)), (np.int64(0), np.int64(0)), (np.int64(1), np.int64(0)), (np.int64(2), np.int64(0)), (np.int64(3), np.int64(0)), (np.int64(4), np.int64(0)), (np.int64(4), np.int64(1))]   
# Fitness (Total Distance): 21.414213562373096

# Toolpath 5:
# Path: [(np.int64(2), np.int64(1)), (np.int64(2), np.int64(2)), (np.int64(3), np.int64(2)), (np.int64(4), np.int64(2)), (np.int64(3), np.int64(3)), (np.int64(4), np.int64(3)), (np.int64(4), np.int64(4)), (np.int64(3), np.int64(4)), (np.int64(2), np.int64(4)), (np.int64(1), np.int64(4)), (np.int64(0), np.int64(4)), (np.int64(0), np.int64(3)), (np.int64(1), np.int64(3)), (np.int64(1), np.int64(2)), (np.int64(0), np.int64(2)), (np.int64(0), np.int64(1)), (np.int64(0), np.int64(0)), (np.int64(1), np.int64(0)), (np.int64(2), np.int64(0)), (np.int64(3), np.int64(0)), (np.int64(4), np.int64(0)), (np.int64(4), np.int64(1))]   
# Fitness (Total Distance): 21.414213562373096

# Zig-Zag Toolpath Distance: 24.0
# Optimized Toolpath Distance: 21.414213562373096