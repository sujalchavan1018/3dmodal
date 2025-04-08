import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.animation import FuncAnimation
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

# Generate the optimized toolpath
optimized_toolpath = genetic_algorithm()

# Create the figure and 3D axis
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Function to setup the 3D scene
def setup_scene(ax):
    # Define main box vertices
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Bottom face
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # Top face
    ])

    # Define faces of the main box (excluding top)
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right
        [vertices[4], vertices[7], vertices[3], vertices[0]]   # Left
    ]

    # Colors for each face
    colors = ['red', 'blue', 'cyan', 'purple', 'orange']

    # Add the box faces (excluding the top face)
    for i in range(len(faces)):
        ax.add_collection3d(Poly3DCollection([faces[i]], facecolors=colors[i], linewidths=1, edgecolors='black'))

    # Define pocket parameters
    pocket_depth = 0.70
    pocket_vertices = np.array([
        [0.10, 0.10, 1], [0.90, 0.10, 1], [0.90, 0.90, 1], [0.10, 0.90, 1],
        [0.10, 0.10, pocket_depth], [0.90, 0.10, pocket_depth], [0.90, 0.90, pocket_depth], [0.10, 0.90, pocket_depth]
    ])

    # Define green top surface (around pocket opening)
    top_surface = [
        [vertices[4], vertices[5], pocket_vertices[1], pocket_vertices[0]],
        [pocket_vertices[3], vertices[7], vertices[6], pocket_vertices[2]],
        [pocket_vertices[0], pocket_vertices[3], vertices[7], vertices[4]],
        [vertices[5], pocket_vertices[1], pocket_vertices[2], vertices[6]]
    ]

    for face in top_surface:
        ax.add_collection3d(Poly3DCollection([face], facecolors='green', linewidths=1, edgecolors='black'))

    # Define pocket faces - SOLID GRAY BOTTOM
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

    # Create line segments for the grid
    xy_lines = []
    for i in range(grid_size + 1):
        for j in range(grid_size):
            xy_lines.append([[x_vals[j], y_vals[i], 1], [x_vals[j + 1], y_vals[i], 1]])  # Horizontal
            xy_lines.append([[x_vals[i], y_vals[j], 1], [x_vals[i], y_vals[j + 1], 1]])  # Vertical

    ax.add_collection3d(Line3DCollection(xy_lines, colors='lightgray', linewidths=1.5))
    
    # Scatter plot for white dots
    ax.scatter(*zip(*grid_points), color='white', s=20, zorder=15)
    
    # Adjust view and axis
    ax.view_init(elev=25, azim=-45)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1.1)
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

# Setup the scene
setup_scene(ax)
ax.set_title('GA-Optimized Toolpath', fontsize=14, pad=20)

# Create drill and path visualizations
drill, = ax.plot([], [], [], color='black', marker='o', markersize=10)
traveled_x, traveled_y, traveled_z = [], [], []
red_line, = ax.plot([], [], [], color='red', linewidth=2, zorder=5)

# Update function for animation
def update(i):
    if i < len(optimized_toolpath):
        x_pos, y_pos, z_pos = optimized_toolpath[i]
        drill.set_data([x_pos], [y_pos])
        drill.set_3d_properties([z_pos])
        
        traveled_x.append(x_pos)
        traveled_y.append(y_pos)
        traveled_z.append(z_pos)
        
        red_line.set_data(traveled_x, traveled_y)
        red_line.set_3d_properties(traveled_z)
    return drill, red_line

# Calculate and print path length
optimized_length = path_length(optimized_toolpath)
print(f"GA-Optimized Path Length: {optimized_length:.4f}")

# Create animation
ani = FuncAnimation(fig, update, frames=len(optimized_toolpath), interval=100, blit=True)

plt.tight_layout()
plt.show()