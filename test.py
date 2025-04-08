# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
# from matplotlib.animation import FuncAnimation
# import random

# # Define grid size
# grid_size = 8
# x_vals = np.linspace(0.10, 0.90, grid_size + 1)
# y_vals = np.linspace(0.10, 0.90, grid_size + 1)
# grid_points = [[x, y, 1] for x in x_vals for y in y_vals]

# # Genetic Algorithm Parameters
# POP_SIZE = 60
# GENS = 130
# MUTATION_RATE = 0.2

# # Calculate path length
# def path_length(path):
#     return sum(np.linalg.norm(np.array(path[i]) - np.array(path[i+1])) for i in range(len(path) - 1))

# # Fitness function
# def fitness_function(path):
#     return 1 / (path_length(path) + 1e-5)

# # Nearest Neighbor Heuristic Initialization
# def nearest_neighbor_path(start_point, points):
#     unvisited = points[:]
#     path = [unvisited.pop(unvisited.index(start_point))]
#     while unvisited:
#         nearest = min(unvisited, key=lambda p: np.linalg.norm(np.array(p) - np.array(path[-1])))
#         path.append(nearest)
#         unvisited.remove(nearest)
#     return path

# # Initialize population with NNH paths
# def initialize_population(size):
#     return [nearest_neighbor_path(random.choice(grid_points), grid_points) for _ in range(size)]

# # Selection (Tournament Selection)
# def selection(population, fitnesses):
#     selected = random.choices(list(zip(population, fitnesses)), k=5)
#     return max(selected, key=lambda x: x[1])[0]

# # Ordered Crossover
# def crossover(parent1, parent2):
#     size = len(parent1)
#     start, end = sorted(random.sample(range(size), 2))
#     child = [None] * size
#     child[start:end] = parent1[start:end]
#     remaining = [p for p in parent2 if p not in child]
#     idx = 0
#     for i in range(size):
#         if child[i] is None:
#             child[i] = remaining[idx]
#             idx += 1
#     return child

# # Mutation (Swap Two Points)
# def mutate(path):
#     if random.random() < MUTATION_RATE:
#         i, j = random.sample(range(len(path)), 2)
#         path[i], path[j] = path[j], path[i]
#     return path

# # 2-Opt Optimization
# def two_opt(path):
#     improved = True
#     while improved:
#         improved = False
#         for i in range(1, len(path) - 2):
#             for j in range(i + 1, len(path)):
#                 if j - i == 1: continue  # Skip adjacent swaps
#                 new_path = path[:i] + path[i:j][::-1] + path[j:]
#                 if path_length(new_path) < path_length(path):
#                     path = new_path
#                     improved = True
#     return path

# # Genetic Algorithm
# def genetic_algorithm():
#     population = initialize_population(POP_SIZE)
#     best_individual = None
#     best_fitness = float('-inf')

#     for gen in range(GENS):
#         fitnesses = [fitness_function(ind) for ind in population]
#         new_population = []

#         # Elitism (Keep Best)
#         best_idx = np.argmax(fitnesses)
#         best_individual = population[best_idx]
#         best_fitness = fitnesses[best_idx]
#         new_population.append(best_individual)

#         while len(new_population) < POP_SIZE:
#             parent1, parent2 = selection(population, fitnesses), selection(population, fitnesses)
#             child = crossover(parent1, parent2)
#             child = mutate(child)
#             new_population.append(child)

#         population = new_population

#         # Apply 2-Opt to the best individual every 10 generations
#         if gen % 10 == 0:
#             best_individual = two_opt(best_individual)

#     return best_individual

# # Generate Zigzag Pattern
# def zigzag_path():
#     path = []
#     for i, x in enumerate(x_vals):
#         if i % 2 == 0:
#             path.extend([[x, y, 1] for y in y_vals])
#         else:
#             path.extend([[x, y, 1] for y in reversed(y_vals)])
#     return path

# # Generate both toolpaths
# optimized_toolpath = genetic_algorithm()
# zigzag_toolpath = zigzag_path()

# # Calculate path lengths
# optimized_length = path_length(optimized_toolpath)
# zigzag_length = path_length(zigzag_toolpath)

# print(f"GA-Optimized Path Length: {optimized_length:.4f}")
# print(f"Zigzag Path Length: {zigzag_length:.4f}")
# print(f"Improvement: {(1 - optimized_length/zigzag_length)*100:.2f}%")

# # Create the figure with two subplots
# fig = plt.figure(figsize=(24, 10))
# ax1 = fig.add_subplot(121, projection='3d')
# ax2 = fig.add_subplot(122, projection='3d')

# # Function to setup the 3D scene (common for both subplots)
# def setup_scene(ax):
#     # Define main box vertices
#     vertices = np.array([
#         [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Bottom face
#         [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # Top face
#     ])

#     # Define faces of the main box (excluding top)
#     faces = [
#         [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom
#         [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front
#         [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back
#         [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right
#         [vertices[4], vertices[7], vertices[3], vertices[0]]   # Left
#     ]

#     # Colors for each face
#     colors = ['red', 'blue', 'cyan', 'purple', 'orange']

#     # Add the box faces (excluding the top face)
#     for i in range(len(faces)):
#         ax.add_collection3d(Poly3DCollection([faces[i]], facecolors=colors[i], linewidths=1, edgecolors='black'))

#     # Define pocket parameters
#     pocket_depth = 0.70
#     pocket_vertices = np.array([
#         [0.10, 0.10, 1], [0.90, 0.10, 1], [0.90, 0.90, 1], [0.10, 0.90, 1],
#         [0.10, 0.10, pocket_depth], [0.90, 0.10, pocket_depth], [0.90, 0.90, pocket_depth], [0.10, 0.90, pocket_depth]
#     ])

#     # Define green top surface (around pocket opening)
#     top_surface = [
#         [vertices[4], vertices[5], pocket_vertices[1], pocket_vertices[0]],
#         [pocket_vertices[3], vertices[7], vertices[6], pocket_vertices[2]],
#         [pocket_vertices[0], pocket_vertices[3], vertices[7], vertices[4]],
#         [vertices[5], pocket_vertices[1], pocket_vertices[2], vertices[6]]
#     ]

#     for face in top_surface:
#         ax.add_collection3d(Poly3DCollection([face], facecolors='green', linewidths=1, edgecolors='black'))

#     # Define pocket faces - SOLID GRAY BOTTOM
#     pocket_faces = [
#         [pocket_vertices[4], pocket_vertices[5], pocket_vertices[6], pocket_vertices[7]],
#         [pocket_vertices[0], pocket_vertices[1], pocket_vertices[5], pocket_vertices[4]],
#         [pocket_vertices[2], pocket_vertices[3], pocket_vertices[7], pocket_vertices[6]],
#         [pocket_vertices[1], pocket_vertices[2], pocket_vertices[6], pocket_vertices[5]],
#         [pocket_vertices[3], pocket_vertices[0], pocket_vertices[4], pocket_vertices[7]]
#     ]

#     ax.add_collection3d(Poly3DCollection([pocket_faces[0]], facecolors='gray', linewidths=1, edgecolors='black'))
#     for face in pocket_faces[1:]:
#         ax.add_collection3d(Poly3DCollection([face], facecolors='gray', alpha=0.3, linewidths=1, edgecolors='black'))

#     # Create line segments for the grid
#     xy_lines = []
#     for i in range(grid_size + 1):
#         for j in range(grid_size):
#             xy_lines.append([[x_vals[j], y_vals[i], 1], [x_vals[j + 1], y_vals[i], 1]])  # Horizontal
#             xy_lines.append([[x_vals[i], y_vals[j], 1], [x_vals[i], y_vals[j + 1], 1]])  # Vertical

#     ax.add_collection3d(Line3DCollection(xy_lines, colors='lightgray', linewidths=1.5))
    
#     # Scatter plot for white dots
#     ax.scatter(*zip(*grid_points), color='white', s=20, zorder=15)
    
#     # Adjust view and axis
#     ax.view_init(elev=25, azim=-45)
#     ax.set_xlim(0, 1)
#     ax.set_ylim(0, 1)
#     ax.set_zlim(0, 1.1)
#     ax.set_xlabel('X Axis')
#     ax.set_ylabel('Y Axis')
#     ax.set_zlabel('Z Axis')

# # Setup both scenes
# setup_scene(ax1)
# setup_scene(ax2)

# # Add titles
# ax1.set_title(f'GA-Optimized Toolpath (Length: {optimized_length:.2f})', fontsize=14, pad=20)
# ax2.set_title(f'Zigzag Toolpath (Length: {zigzag_length:.2f})', fontsize=14, pad=20)

# # Create drill and path visualizations for both subplots
# # For GA-optimized path
# drill1, = ax1.plot([], [], [], color='black', marker='o', markersize=10)
# traveled_x1, traveled_y1, traveled_z1 = [], [], []
# red_line1, = ax1.plot([], [], [], color='red', linewidth=2, zorder=5)

# # For zigzag path
# drill2, = ax2.plot([], [], [], color='black', marker='o', markersize=10)
# traveled_x2, traveled_y2, traveled_z2 = [], [], []
# red_line2, = ax2.plot([], [], [], color='red', linewidth=2, zorder=5)

# # Update functions for both animations
# def update1(i):
#     if i < len(optimized_toolpath):
#         x_pos, y_pos, z_pos = optimized_toolpath[i]
#         drill1.set_data([x_pos], [y_pos])
#         drill1.set_3d_properties([z_pos])
        
#         traveled_x1.append(x_pos)
#         traveled_y1.append(y_pos)
#         traveled_z1.append(z_pos)
        
#         red_line1.set_data(traveled_x1, traveled_y1)
#         red_line1.set_3d_properties(traveled_z1)
#     return drill1, red_line1

# def update2(i):
#     if i < len(zigzag_toolpath):
#         x_pos, y_pos, z_pos = zigzag_toolpath[i]
#         drill2.set_data([x_pos], [y_pos])
#         drill2.set_3d_properties([z_pos])
        
#         traveled_x2.append(x_pos)
#         traveled_y2.append(y_pos)
#         traveled_z2.append(z_pos)
        
#         red_line2.set_data(traveled_x2, traveled_y2)
#         red_line2.set_3d_properties(traveled_z2)
#     return drill2, red_line2

# # Create both animations
# ani1 = FuncAnimation(fig, update1, frames=len(optimized_toolpath), interval=100, blit=True)
# ani2 = FuncAnimation(fig, update2, frames=len(zigzag_toolpath), interval=100, blit=True)

# plt.tight_layout()
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.animation import FuncAnimation
import random
from collections import defaultdict

# Define grid size
grid_size = 8
x_vals = np.linspace(0.10, 0.90, grid_size + 1)
y_vals = np.linspace(0.10, 0.90, grid_size + 1)
grid_points = [np.array([x, y, 1]) for x in x_vals for y in y_vals]

# Genetic Algorithm Parameters
POP_SIZE = 100
GENS = 200
MUTATION_RATE = 0.3

# Calculate path length
def path_length(path):
    return sum(np.linalg.norm(path[i] - path[i+1]) for i in range(len(path) - 1))

# Fitness function
def fitness_function(path):
    return 1 / (path_length(path) + 1e-5)

# Helper function to remove array from list
def remove_array_from_list(arr, lst):
    for i, item in enumerate(lst):
        if np.array_equal(item, arr):
            lst.pop(i)
            return
    raise ValueError("Array not found in list")

# Nearest Neighbor Heuristic Initialization
def nearest_neighbor_path(start_point, points):
    unvisited = points.copy()
    path = [start_point]
    remove_array_from_list(start_point, unvisited)
    
    while unvisited:
        last = path[-1]
        nearest = min(unvisited, key=lambda p: np.linalg.norm(p - last))
        path.append(nearest)
        remove_array_from_list(nearest, unvisited)
    return path

# Greedy Path Initialization
def greedy_path(start_point, points):
    unvisited = points.copy()
    path = [start_point]
    remove_array_from_list(start_point, unvisited)
    
    while unvisited:
        last = path[-1]
        scores = []
        for i, p in enumerate(unvisited):
            dist_to_p = np.linalg.norm(p - last)
            if len(unvisited) == 1:
                scores.append(dist_to_p)
            else:
                other_points = unvisited[:i] + unvisited[i+1:]
                dists_to_others = [np.linalg.norm(p - op) for op in other_points]
                scores.append(dist_to_p + min(dists_to_others))
        best_idx = np.argmin(scores)
        path.append(unvisited.pop(best_idx))
    return path

# Initialize population
def initialize_population(size):
    population = []
    for _ in range(size//3):
        start = random.choice(grid_points)
        population.append(nearest_neighbor_path(start, grid_points.copy()))
    for _ in range(size//3):
        start = random.choice(grid_points)
        population.append(greedy_path(start, grid_points.copy()))
    for _ in range(size - len(population)):
        path = grid_points.copy()
        random.shuffle(path)
        population.append(path)
    return population

# Edge Recombination Crossover
def edge_recombination_crossover(parent1, parent2):
    edge_map = defaultdict(set)
    for path in [parent1, parent2]:
        for i, point in enumerate(path):
            point_tuple = tuple(point)
            neighbors = set()
            if i > 0:
                neighbors.add(tuple(path[i-1]))
            if i < len(path)-1:
                neighbors.add(tuple(path[i+1]))
            edge_map[point_tuple].update(neighbors)
    
    child = []
    current = tuple(random.choice(grid_points))
    child.append(np.array(current))
    
    while len(child) < len(grid_points):
        for neighbors in edge_map.values():
            if current in neighbors:
                neighbors.remove(current)
        
        if edge_map[current]:
            next_point = min(edge_map[current], key=lambda x: len(edge_map[x]))
        else:
            remaining = [p for p in grid_points if tuple(p) not in map(tuple, child)]
            next_point = tuple(random.choice(remaining))
        
        child.append(np.array(next_point))
        current = next_point
    
    return child

# Mutation operators
def inversion_mutation(path):
    if random.random() < MUTATION_RATE:
        i, j = sorted(random.sample(range(len(path)), 2))
        path[i:j+1] = path[i:j+1][::-1]
    return path

def swap_mutation(path):
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(len(path)), 2)
        path[i], path[j] = path[j], path[i]
    return path

# 2-opt optimization
def two_opt(path):
    best = path.copy()
    improved = True
    while improved:
        improved = False
        for i in range(1, len(path)-2):
            for j in range(i+1, len(path)):
                if j-i == 1:
                    continue
                new_path = path[:i] + path[i:j][::-1] + path[j:]
                if path_length(new_path) < path_length(best):
                    best = new_path
                    improved = True
        path = best
    return best

# Genetic Algorithm
def genetic_algorithm():
    population = initialize_population(POP_SIZE)
    best_individual = min(population, key=path_length)
    best_length = path_length(best_individual)
    
    for gen in range(GENS):
        ranked = sorted(population, key=path_length)
        selection_probs = [1/(i+1) for i in range(len(ranked))]
        selection_probs = np.array(selection_probs)/sum(selection_probs)
        
        new_population = []
        elites = ranked[:int(0.1*POP_SIZE)]
        new_population.extend(elites)
        
        while len(new_population) < POP_SIZE:
            # Fixed parent selection
            parent_indices = np.random.choice(len(ranked), size=2, p=selection_probs, replace=False)
            parent1, parent2 = ranked[parent_indices[0]], ranked[parent_indices[1]]
            
            if random.random() < 0.9:
                child = edge_recombination_crossover(parent1, parent2)
            else:
                child = random.choice([parent1, parent2])
            
            child = inversion_mutation(child)
            child = swap_mutation(child)
            
            if random.random() < 0.3 or gen % 5 == 0:
                child = two_opt(child)
            
            new_population.append(child)
        
        population = new_population
        current_best = min(population, key=path_length)
        current_length = path_length(current_best)
        
        if current_length < best_length:
            best_individual = current_best
            best_length = current_length
    
    return best_individual

# Generate Zigzag Pattern
def zigzag_path():
    path = []
    for i, x in enumerate(x_vals):
        if i % 2 == 0:
            path.extend([np.array([x, y, 1]) for y in y_vals])
        else:
            path.extend([np.array([x, y, 1]) for y in reversed(y_vals)])
    return path

# Generate both toolpaths
optimized_toolpath = genetic_algorithm()
zigzag_toolpath = zigzag_path()

# Calculate path lengths
optimized_length = path_length(optimized_toolpath)
zigzag_length = path_length(zigzag_toolpath)

print(f"GA-Optimized Path Length: {optimized_length:.4f}")
print(f"Zigzag Path Length: {zigzag_length:.4f}")
print(f"Improvement: {(1 - optimized_length/zigzag_length)*100:.2f}%")

# Visualization (same as before)
fig = plt.figure(figsize=(24, 10))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

def setup_scene(ax):
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ])
    
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[4], vertices[7], vertices[3], vertices[0]]
    ]
    
    colors = ['red', 'blue', 'cyan', 'purple', 'orange']
    for i, face in enumerate(faces):
        ax.add_collection3d(Poly3DCollection([face], facecolors=colors[i], linewidths=1, edgecolors='black'))
    
    pocket_depth = 0.70
    pocket_vertices = np.array([
        [0.10, 0.10, 1], [0.90, 0.10, 1], [0.90, 0.90, 1], [0.10, 0.90, 1],
        [0.10, 0.10, pocket_depth], [0.90, 0.10, pocket_depth], [0.90, 0.90, pocket_depth], [0.10, 0.90, pocket_depth]
    ])
    
    top_surfaces = [
        [vertices[4], vertices[5], pocket_vertices[1], pocket_vertices[0]],
        [pocket_vertices[3], vertices[7], vertices[6], pocket_vertices[2]],
        [pocket_vertices[0], pocket_vertices[3], vertices[7], vertices[4]],
        [vertices[5], pocket_vertices[1], pocket_vertices[2], vertices[6]]
    ]
    for face in top_surfaces:
        ax.add_collection3d(Poly3DCollection([face], facecolors='green', linewidths=1, edgecolors='black'))
    
    pocket_faces = [
        [pocket_vertices[4], pocket_vertices[5], pocket_vertices[6], pocket_vertices[7]],
        [pocket_vertices[0], pocket_vertices[1], pocket_vertices[5], pocket_vertices[4]],
        [pocket_vertices[2], pocket_vertices[3], pocket_vertices[7], pocket_vertices[6]],
        [pocket_vertices[1], pocket_vertices[2], pocket_vertices[6], pocket_vertices[5]],
        [pocket_vertices[3], pocket_vertices[0], pocket_vertices[4], pocket_vertices[7]]
    ]
    ax.add_collection3d(Poly3DCollection([pocket_faces[0]], facecolors='gray', linewidths=1, edgecolors='black'))
    for face in pocket_faces[1:]:
        ax.add_collection3d(Poly3DCollection([face], facecolors='gray', alpha=0.3, linewidths=1, edgecolors='black'))
    
    xy_lines = []
    for i in range(grid_size + 1):
        for j in range(grid_size):
            xy_lines.append([[x_vals[j], y_vals[i], 1], [x_vals[j+1], y_vals[i], 1]])
            xy_lines.append([[x_vals[i], y_vals[j], 1], [x_vals[i], y_vals[j+1], 1]])
    ax.add_collection3d(Line3DCollection(xy_lines, colors='lightgray', linewidths=1.5))
    
    ax.scatter(*zip(*grid_points), color='white', s=20, zorder=15)
    
    ax.view_init(elev=25, azim=-45)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1.1)
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

# Setup both scenes
setup_scene(ax1)
setup_scene(ax2)

ax1.set_title(f'GA-Optimized Toolpath (Length: {optimized_length:.2f})', fontsize=14, pad=20)
ax2.set_title(f'Zigzag Toolpath (Length: {zigzag_length:.2f})', fontsize=14, pad=20)

def init_animation(ax, path):
    line, = ax.plot([], [], [], 'r-', linewidth=2, zorder=5)
    drill, = ax.plot([], [], [], 'ko', markersize=10)
    return line, drill

def update_animation(frame, path, line, drill):
    if frame < len(path):
        x, y, z = path[frame]
        drill.set_data([x], [y])
        drill.set_3d_properties([z])
        
        line.set_data(*zip(*[(p[0], p[1]) for p in path[:frame+1]]))
        line.set_3d_properties([p[2] for p in path[:frame+1]])
    return line, drill

line1, drill1 = init_animation(ax1, optimized_toolpath)
line2, drill2 = init_animation(ax2, zigzag_toolpath)

ani1 = FuncAnimation(fig, lambda i: update_animation(i, optimized_toolpath, line1, drill1),
                    frames=len(optimized_toolpath), interval=100, blit=True)
ani2 = FuncAnimation(fig, lambda i: update_animation(i, zigzag_toolpath, line2, drill2),
                    frames=len(zigzag_toolpath), interval=100, blit=True)

plt.tight_layout()
plt.show()
