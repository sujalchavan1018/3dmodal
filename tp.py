# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
# from matplotlib.animation import FuncAnimation
# import random

# # Define main box vertices
# vertices = np.array([
#     [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Bottom face
#     [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # Top face
# ])

# # Define faces of the main box (excluding top)
# faces = [
#     [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom
#     [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front
#     [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back
#     [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right
#     [vertices[4], vertices[7], vertices[3], vertices[0]]   # Left
# ]

# # Colors for each face
# colors = ['red', 'blue', 'cyan', 'purple', 'orange']

# # Setup figure
# fig = plt.figure(figsize=(12, 10))
# ax = fig.add_subplot(111, projection='3d')

# # Add the box faces (excluding the top face)
# for i in range(len(faces)):
#     ax.add_collection3d(Poly3DCollection([faces[i]], facecolors=colors[i], linewidths=1, edgecolors='black'))

# # Define pocket parameters
# pocket_depth = 0.70
# pocket_vertices = np.array([
#     [0.10, 0.10, 1], [0.90, 0.10, 1], [0.90, 0.90, 1], [0.10, 0.90, 1],
#     [0.10, 0.10, pocket_depth], [0.90, 0.10, pocket_depth], [0.90, 0.90, pocket_depth], [0.10, 0.90, pocket_depth]
# ])

# # Define green top surface (around pocket opening)
# top_surface = [
#     [vertices[4], vertices[5], pocket_vertices[1], pocket_vertices[0]],
#     [pocket_vertices[3], vertices[7], vertices[6], pocket_vertices[2]],
#     [pocket_vertices[0], pocket_vertices[3], vertices[7], vertices[4]],
#     [vertices[5], pocket_vertices[1], pocket_vertices[2], vertices[6]]
# ]

# for face in top_surface:
#     ax.add_collection3d(Poly3DCollection([face], facecolors='green', linewidths=1, edgecolors='black'))

# # Define pocket faces - SOLID GRAY BOTTOM
# pocket_faces = [
#     [pocket_vertices[4], pocket_vertices[5], pocket_vertices[6], pocket_vertices[7]],
#     [pocket_vertices[0], pocket_vertices[1], pocket_vertices[5], pocket_vertices[4]],
#     [pocket_vertices[2], pocket_vertices[3], pocket_vertices[7], pocket_vertices[6]],
#     [pocket_vertices[1], pocket_vertices[2], pocket_vertices[6], pocket_vertices[5]],
#     [pocket_vertices[3], pocket_vertices[0], pocket_vertices[4], pocket_vertices[7]]
# ]

# ax.add_collection3d(Poly3DCollection([pocket_faces[0]], facecolors='gray', linewidths=1, edgecolors='black'))
# for face in pocket_faces[1:]:
#     ax.add_collection3d(Poly3DCollection([face], facecolors='gray', alpha=0.3, linewidths=1, edgecolors='black'))

# # Create 8x8 grid on the top side of the pocket
# grid_size = 8
# x_vals = np.linspace(0.10, 0.90, grid_size + 1)
# y_vals = np.linspace(0.10, 0.90, grid_size + 1)
# grid_points = [[x, y, 1] for x in x_vals for y in y_vals]

# # Scatter plot for white dots
# ax.scatter(*zip(*grid_points), color='white', s=20, zorder=15)

# # Create line segments for the grid
# xy_lines = []
# for i in range(grid_size + 1):
#     for j in range(grid_size):
#         xy_lines.append([[x_vals[j], y_vals[i], 1], [x_vals[j + 1], y_vals[i], 1]])  # Horizontal
#         xy_lines.append([[x_vals[i], y_vals[j], 1], [x_vals[i], y_vals[j + 1], 1]])  # Vertical

# ax.add_collection3d(Line3DCollection(xy_lines, colors='lightgray', linewidths=1.5))

# # Optimized Zigzag Path Generator (without scipy)
# def create_optimized_zigzag(points, x_vals, y_vals):
#     """Create an optimized zigzag path that minimizes transitions between rows"""
    
#     # Group points by their x-coordinate (rows)
#     rows = {}
#     for x in x_vals:
#         rows[x] = [p for p in points if abs(p[0] - x) < 1e-5]
    
#     # Calculate row centers for distance measurement
#     row_centers = []
#     for x in x_vals:
#         row_points = rows[x]
#         center_x = sum(p[0] for p in row_points) / len(row_points)
#         center_y = sum(p[1] for p in row_points) / len(row_points)
#         row_centers.append((center_x, center_y))
    
#     # Simple nearest-neighbor approach for row ordering
#     current_row = 0
#     visited_rows = {current_row}
#     row_order = [current_row]
    
#     while len(visited_rows) < len(x_vals):
#         min_dist = float('inf')
#         next_row = -1
        
#         for i in range(len(x_vals)):
#             if i not in visited_rows:
#                 # Calculate distance between current row center and potential next row
#                 dist = np.sqrt((row_centers[current_row][0] - row_centers[i][0])**2 + 
#                                (row_centers[current_row][1] - row_centers[i][1])**2)
#                 if dist < min_dist:
#                     min_dist = dist
#                     next_row = i
        
#         if next_row == -1:  # Shouldn't happen but just in case
#             next_row = next(i for i in range(len(x_vals)) if i not in visited_rows)
        
#         visited_rows.add(next_row)
#         row_order.append(next_row)
#         current_row = next_row
    
#     # Build the final path
#     optimized_path = []
#     for i, row_idx in enumerate(row_order):
#         x = x_vals[row_idx]
#         row_points = rows[x]
        
#         # Alternate direction based on row index
#         if i % 2 == 0:
#             optimized_path.extend(sorted(row_points, key=lambda p: p[1]))
#         else:
#             optimized_path.extend(sorted(row_points, key=lambda p: p[1], reverse=True))
    
#     return optimized_path

# # Generate optimized path
# optimized_toolpath = create_optimized_zigzag(grid_points, x_vals, y_vals)

# # Verify all points are covered exactly once
# assert len(optimized_toolpath) == len(grid_points)
# assert len(set(tuple(p) for p in optimized_toolpath)) == len(grid_points)

# # Visualization setup
# drill, = ax.plot([], [], [], color='black', marker='o', markersize=10)
# traveled_x, traveled_y, traveled_z = [], [], []
# red_line, = ax.plot([], [], [], color='red', linewidth=2, zorder=5)

# def update(i):
#     if i < len(optimized_toolpath):
#         x_pos, y_pos, z_pos = optimized_toolpath[i]
#         drill.set_data([x_pos], [y_pos])
#         drill.set_3d_properties([z_pos])
        
#         traveled_x.append(x_pos)
#         traveled_y.append(y_pos)
#         traveled_z.append(z_pos)
        
#         red_line.set_data(traveled_x, traveled_y)
#         red_line.set_3d_properties(traveled_z)
#     return drill, red_line

# # Animation
# ani = FuncAnimation(fig, update, frames=len(optimized_toolpath), interval=100, blit=True)

# # Final plot adjustments
# ax.view_init(elev=25, azim=-45)
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
# ax.set_zlim(0, 1.1)
# ax.set_xlabel('X Axis')
# ax.set_ylabel('Y Axis')
# ax.set_zlabel('Z Axis')
# ax.set_title('Optimized Zigzag Toolpath for Pocket Roughing', fontsize=14, pad=20)
# plt.tight_layout()
# plt.show()

# zigzag pattern

import numpy as np
import matplotlib.pyplot as plt
import random

# Define grid size
grid_size = 8
x_vals = np.linspace(0.10, 0.90, grid_size + 1)
y_vals = np.linspace(0.10, 0.90, grid_size + 1)
grid_points = [[x, y, 1] for x in x_vals for y in y_vals]

# Genetic Algorithm Parameters
POP_SIZE = 50
GENS = 100
MUTATION_RATE = 0.2

# Calculate path length
def path_length(path):
    return sum(np.linalg.norm(np.array(path[i]) - np.array(path[i+1])) for i in range(len(path) - 1))

# Fitness function
def fitness_function(path):
    return 1 / (path_length(path) + 1e-5)

# Nearest Neighbor Heuristic Initialization
def nearest_neighbor_path(start_point, points):
    unvisited = points[:]
    path = [unvisited.pop(unvisited.index(start_point))]
    while unvisited:
        nearest = min(unvisited, key=lambda p: np.linalg.norm(np.array(p) - np.array(path[-1])))
        path.append(nearest)
        unvisited.remove(nearest)
    return path

# Initialize population with NNH paths
def initialize_population(size):
    return [nearest_neighbor_path(random.choice(grid_points), grid_points) for _ in range(size)]

# Selection (Tournament Selection)
def selection(population, fitnesses):
    selected = random.choices(list(zip(population, fitnesses)), k=5)
    return max(selected, key=lambda x: x[1])[0]

# Ordered Crossover
def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[start:end] = parent1[start:end]
    remaining = [p for p in parent2 if p not in child]
    idx = 0
    for i in range(size):
        if child[i] is None:
            child[i] = remaining[idx]
            idx += 1
    return child

# Mutation (Swap Two Points)
def mutate(path):
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(len(path)), 2)
        path[i], path[j] = path[j], path[i]
    return path

# 2-Opt Optimization
def two_opt(path):
    improved = True
    while improved:
        improved = False
        for i in range(1, len(path) - 2):
            for j in range(i + 1, len(path)):
                if j - i == 1: continue  # Skip adjacent swaps
                new_path = path[:i] + path[i:j][::-1] + path[j:]
                if path_length(new_path) < path_length(path):
                    path = new_path
                    improved = True
    return path

# Genetic Algorithm
def genetic_algorithm():
    population = initialize_population(POP_SIZE)
    best_individual = None
    best_fitness = float('-inf')

    for gen in range(GENS):
        fitnesses = [fitness_function(ind) for ind in population]
        new_population = []

        # Elitism (Keep Best)
        best_idx = np.argmax(fitnesses)
        best_individual = population[best_idx]
        best_fitness = fitnesses[best_idx]
        new_population.append(best_individual)

        while len(new_population) < POP_SIZE:
            parent1, parent2 = selection(population, fitnesses), selection(population, fitnesses)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = new_population

        # Apply 2-Opt to the best individual every 10 generations
        if gen % 10 == 0:
            best_individual = two_opt(best_individual)

    return best_individual

# Generate Zigzag Pattern
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
ax2 = fig.add_subplot(122, projection='3d')

# Plot Zigzag Path
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

# Plot Optimized Path
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


# Zigzag Path Length: 8.0000
# GA-Optimized Path Length: 8.5236
# Improvement: -6f.55%

# this is output 