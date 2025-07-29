import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation
import random
from tqdm import tqdm
from matplotlib.colors import LightSource

# Parameters
grid_size = 8
pocket_margin = 0.10
pocket_depth = 0.30
l_width = 3.0
l_height = 3.0
l_depth = 1.5

# Create grid points with pocket
x_vals = np.linspace(pocket_margin, l_width-pocket_margin, grid_size)
y_vals = np.linspace(pocket_margin, l_height-pocket_margin, grid_size)
grid_points = []
for x in x_vals:
    for y in y_vals:
        if (x < 0.8 or y < 0.8):  # L shape condition
            if 0.5 < x < 1.5 and 0.5 < y < 1.5:  # Pocket area
                grid_points.append([x, y, l_depth - pocket_depth])
            else:
                grid_points.append([x, y, l_depth])

# GA Parameters (unchanged)
GA_PARAMS = {
    'pop_size': 70,
    'generations': 150,
    'mutation_rate': 0.5,
    'elitism': 3,
    'tournament_size': 5
}

# ToolpathOptimizer class remains exactly the same
class ToolpathOptimizer:
    def __init__(self, points):
        self.points = points
        self.distance_matrix = self._precompute_distances()
        
    def _precompute_distances(self):
        n = len(self.points)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(np.array(self.points[i]) - np.array(self.points[j]))
                dist_matrix[i][j] = dist_matrix[j][i] = dist
        return dist_matrix
    
    def path_length(self, path_indices):
        return sum(self.distance_matrix[path_indices[i]][path_indices[i+1]] 
               for i in range(len(path_indices)-1))
    
    def fitness(self, path_indices):
        return 1 / (self.path_length(path_indices) + 1e-8)
    
    def initialize_population(self):
        population = []
        for _ in range(GA_PARAMS['pop_size']):
            if random.random() < 0.9:
                start = random.randint(0, len(self.points)-1)
                path = self._nearest_neighbor_path(start)
            else:
                path = list(range(len(self.points)))
                random.shuffle(path)
            population.append(path)
        return population
    
    def _nearest_neighbor_path(self, start_index):
        unvisited = set(range(len(self.points)))
        path = [start_index]
        unvisited.remove(start_index)
        
        while unvisited:
            last = path[-1]
            nearest = min(unvisited, key=lambda x: self.distance_matrix[last][x])
            path.append(nearest)
            unvisited.remove(nearest)
        return path
    
    def evolve(self):
        population = self.initialize_population()
        best_path = None
        best_fitness = -np.inf
        
        with tqdm(total=GA_PARAMS['generations'], desc="Optimizing Path") as pbar:
            for gen in range(GA_PARAMS['generations']):
                fitnesses = [self.fitness(ind) for ind in population]
                
                current_best_idx = np.argmax(fitnesses)
                if fitnesses[current_best_idx] > best_fitness:
                    best_fitness = fitnesses[current_best_idx]
                    best_path = population[current_best_idx]
                
                new_pop = []
                elite_indices = np.argsort(fitnesses)[-GA_PARAMS['elitism']:]
                new_pop.extend([population[i] for i in elite_indices])
                
                while len(new_pop) < GA_PARAMS['pop_size']:
                    parent1 = self._tournament_select(population, fitnesses)
                    parent2 = self._tournament_select(population, fitnesses)
                    child = self._crossover(parent1, parent2)
                    child = self._mutate(child)
                    new_pop.append(child)
                
                population = new_pop
                best_path = self._two_opt(best_path)
                
                if gen % 20 == 0:
                    best_path = self._three_opt(best_path)
                
                pbar.update(1)
        
        return [self.points[i] for i in best_path]
    
    def _tournament_select(self, population, fitnesses):
        contestants = random.sample(list(zip(population, fitnesses)), 
                                GA_PARAMS['tournament_size'])
        return max(contestants, key=lambda x: x[1])[0]
    
    def _crossover(self, parent1, parent2):
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        child = [None]*size
        child[start:end] = parent1[start:end]
        
        ptr = 0
        for i in range(size):
            if child[i] is None:
                while parent2[ptr] in child:
                    ptr += 1
                child[i] = parent2[ptr]
                ptr += 1
        return child
    
    def _mutate(self, path):
        if random.random() < GA_PARAMS['mutation_rate']:
            i, j = random.sample(range(len(path)), 2)
            path[i], path[j] = path[j], path[i]
        return path
    
    def _two_opt(self, path):
        best_path = path
        best_length = self.path_length(path)
        improved = True
        
        while improved:
            improved = False
            for i in range(1, len(path)-2):
                for j in range(i+1, len(path)):
                    if j - i == 1:
                        continue
                    new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
                    new_length = self.path_length(new_path)
                    if new_length < best_length:
                        best_path = new_path
                        best_length = new_length
                        improved = True
            path = best_path
        return best_path
    
    def _three_opt(self, path):
        best_path = path
        best_length = self.path_length(path)
        
        for i in range(1, len(path)-3):
            for j in range(i+1, len(path)-2):
                for k in range(j+1, len(path)-1):
                    new_path = path[:i] + path[i:j][::-1] + path[j:k][::-1] + path[k:]
                    new_length = self.path_length(new_path)
                    if new_length < best_length:
                        best_path = new_path
                        best_length = new_length
        return best_path

def create_spiral_pocket_path():
    """Create a spiral path specifically for the pocket area"""
    pocket_points = [p for p in grid_points if 0.5 < p[0] < 1.5 and 0.5 < p[1] < 1.5]
    
    # Sort points in a spiral pattern
    center_x = (0.5 + 1.5) / 2
    center_y = (0.5 + 1.5) / 2
    
    # Sort by angle and distance from center
    pocket_points_sorted = sorted(pocket_points,
                                key=lambda p: (np.arctan2(p[1]-center_y, p[0]-center_x),
                                np.linalg.norm([p[0]-center_x, p[1]-center_y])))
    
    # Alternate between inner and outer points for spiral effect
    spiral_path = []
    n = len(pocket_points_sorted)
    for i in range(n//2):
        spiral_path.append(pocket_points_sorted[i])
        spiral_path.append(pocket_points_sorted[n-1-i])
    if n % 2 != 0:
        spiral_path.append(pocket_points_sorted[n//2])
    
    return spiral_path

def create_hybrid_path(optimized_path):
    """Combine optimized path with spiral pattern in pocket area"""
    # Separate pocket and non-pocket points
    pocket_indices = [i for i, p in enumerate(grid_points) 
                     if 0.5 < p[0] < 1.5 and 0.5 < p[1] < 1.5]
    non_pocket_indices = [i for i in range(len(grid_points)) if i not in pocket_indices]
    
    # Get optimized order for non-pocket points
    optimized_non_pocket = [i for i in optimized_path if i in non_pocket_indices]
    
    # Create spiral path for pocket
    spiral_pocket = create_spiral_pocket_path()
    spiral_indices = [grid_points.index(p) for p in spiral_pocket]
    
    # Find best insertion point for spiral in optimized path
    if optimized_non_pocket and spiral_indices:
        # Find point in optimized path closest to spiral start
        spiral_start_point = grid_points[spiral_indices[0]]
        closest_idx = min(range(len(optimized_non_pocket)),
                        key=lambda i: np.linalg.norm(
                            np.array(grid_points[optimized_non_pocket[i]]) - 
                            np.array(spiral_start_point)))
        
        # Insert spiral after closest point
        hybrid_indices = (optimized_non_pocket[:closest_idx+1] + 
                         spiral_indices + 
                         optimized_non_pocket[closest_idx+1:])
    else:
        hybrid_indices = optimized_path
    
    return [grid_points[i] for i in hybrid_indices]

class LLetterVisualizer:
    def __init__(self):
        self.fig = plt.figure(figsize=(18, 8), facecolor='white')
        self.ax1 = self.fig.add_subplot(121, projection='3d')  # Hybrid path
        self.ax2 = self.fig.add_subplot(122, projection='3d')  # Path comparison
        self._setup_lighting()
        self.setup_scenes()
    
    def _setup_lighting(self):
        self.light = LightSource(azdeg=225, altdeg=45)
        self.face_colors = {
            'base': '#4682B4', 'front': '#5F9EA0', 'back': '#B0C4DE',
            'right': '#87CEEB', 'left': '#ADD8E6', 'pocket': '#D3D3D3',
            'top': '#98FB98', 'grid': '#FFFFFF'
        }
    
    def _create_l_letter(self, ax):
        # Create L letter geometry
        vertical = np.array([
            [0, 0, 0], [0, l_height, 0], [0.8, l_height, 0], [0.8, 0, 0],
            [0, 0, l_depth], [0, l_height, l_depth], [0.8, l_height, l_depth], [0.8, 0, l_depth]
        ])
        
        horizontal = np.array([
            [0, 0, 0], [0, 0.8, 0], [l_width, 0.8, 0], [l_width, 0, 0],
            [0, 0, l_depth], [0, 0.8, l_depth], [l_width, 0.8, l_depth], [l_width, 0, l_depth]
        ])
        
        def get_faces(vertices):
            return [
                [vertices[0], vertices[1], vertices[2], vertices[3]],
                [vertices[4], vertices[5], vertices[6], vertices[7]],
                [vertices[0], vertices[1], vertices[5], vertices[4]],
                [vertices[2], vertices[3], vertices[7], vertices[6]],
                [vertices[1], vertices[2], vertices[6], vertices[5]],
                [vertices[0], vertices[3], vertices[7], vertices[4]]
            ]
        
        for vertices, color_factor in [(vertical, 1.0), (horizontal, 0.9)]:
            faces = get_faces(vertices)
            shaded_colors = []
            for face in faces:
                v1 = np.array(face[1]) - np.array(face[0])
                v2 = np.array(face[2]) - np.array(face[0])
                normal = np.cross(v1, v2)
                normal = normal / np.linalg.norm(normal)
                intensity = np.dot(normal, [-1, -1, 1])
                intensity = (intensity + 1) / 2
                intensity = max(0.3, min(1.0, intensity))
                shaded_color = np.array([0.6, 0.8, 1.0]) * intensity * color_factor
                shaded_colors.append(shaded_color)
            
            poly = Poly3DCollection(
                faces,
                facecolors=shaded_colors,
                linewidths=1,
                edgecolors='k',
                zorder=1,
                alpha=0.7
            )
            ax.add_collection3d(poly)
        
        xs, ys, zs = zip(*grid_points)
        colors = []
        for point in grid_points:
            if 0.5 < point[0] < 1.5 and 0.5 < point[1] < 1.5:  # Pocket area
                colors.append('red')
            else:
                colors.append(self.face_colors['grid'])
                
        ax.scatter(
            xs, ys, zs, 
            color=colors,
            s=40, 
            edgecolor='k',
            linewidth=1,
            zorder=5
        )
        
        ax.view_init(elev=30, azim=-50)
        ax.set_xlim(0, l_width)
        ax.set_ylim(0, l_height)
        ax.set_zlim(0, l_depth*1.1)
        ax.set_xlabel('X Axis', fontsize=10)
        ax.set_ylabel('Y Axis', fontsize=10)
        ax.set_zlabel('Z Axis', fontsize=10)
        ax.grid(False)
        ax.set_facecolor('white')
    
    def setup_scenes(self):
        self._create_l_letter(self.ax1)
        self._create_l_letter(self.ax2)
        self.ax1.set_title('Hybrid Toolpath (Optimal + Spiral)', fontsize=12, pad=15)
        self.ax2.set_title('Toolpath Comparison', fontsize=12, pad=15)
        self.fig.suptitle('L Letter with Pocket - Hybrid Toolpath', fontsize=16, y=0.95)
    
    def animate_paths(self, hybrid_path, optimized_path, spiral_pocket_path):
        # Initialize paths
        self.hybrid_line, = self.ax1.plot([], [], [], 'g-', linewidth=3, zorder=6)
        self.hybrid_drill, = self.ax1.plot([], [], [], 'ko', markersize=8, zorder=10)
        
        self.opt_line, = self.ax2.plot([], [], [], 'r-', linewidth=1.5, zorder=6, alpha=0.7)
        self.spiral_line, = self.ax2.plot([], [], [], 'b-', linewidth=1.5, zorder=6, alpha=0.7)
        
        # Store path coordinates
        self.hybrid_x, self.hybrid_y, self.hybrid_z = [], [], []
        self.opt_x, self.opt_y, self.opt_z = [], [], []
        self.spiral_x, self.spiral_y, self.spiral_z = [], [], []
        
        # Create animation
        max_frames = max(len(hybrid_path), len(optimized_path), len(spiral_pocket_path))
        self.ani = FuncAnimation(
            self.fig, 
            self._update_animation,
            frames=max_frames,
            interval=100,
            blit=True,
            init_func=lambda: [self.hybrid_line, self.hybrid_drill, 
                             self.opt_line, self.spiral_line],
            fargs=(hybrid_path, optimized_path, spiral_pocket_path)
        )
    
    def _update_animation(self, frame, hybrid_path, opt_path, spiral_path):
        # Update hybrid path
        if frame < len(hybrid_path):
            x, y, z = hybrid_path[frame]
            self.hybrid_drill.set_data([x], [y])
            self.hybrid_drill.set_3d_properties([z])
            self.hybrid_x.append(x)
            self.hybrid_y.append(y)
            self.hybrid_z.append(z)
            self.hybrid_line.set_data(self.hybrid_x, self.hybrid_y)
            self.hybrid_line.set_3d_properties(self.hybrid_z)
        
        # Update comparison paths
        if frame < len(opt_path):
            x, y, z = opt_path[frame]
            self.opt_x.append(x)
            self.opt_y.append(y)
            self.opt_z.append(z)
            self.opt_line.set_data(self.opt_x, self.opt_y)
            self.opt_line.set_3d_properties(self.opt_z)
        
        if frame < len(spiral_path):
            x, y, z = spiral_path[frame]
            self.spiral_x.append(x)
            self.spiral_y.append(y)
            self.spiral_z.append(z)
            self.spiral_line.set_data(self.spiral_x, self.spiral_y)
            self.spiral_line.set_3d_properties(self.spiral_z)
        
        return self.hybrid_line, self.hybrid_drill, self.opt_line, self.spiral_line

# Main execution
if __name__ == "__main__":
    print("Optimizing toolpath...")
    optimizer = ToolpathOptimizer(grid_points)
    optimized_path = optimizer.evolve()
    spiral_pocket_path = create_spiral_pocket_path()
    hybrid_path = create_hybrid_path([grid_points.index(p) for p in optimized_path])
    
    # Calculate path lengths
    opt_length = optimizer.path_length([grid_points.index(p) for p in optimized_path])
    hybrid_length = optimizer.path_length([grid_points.index(p) for p in hybrid_path])
    
    print(f"\nOptimized Path Length: {opt_length:.4f} units")
    print(f"Hybrid Path Length: {hybrid_length:.4f} units")
    print(f"Difference: {hybrid_length-opt_length:.4f} units ({(hybrid_length/opt_length-1)*100:.2f}% longer)")
    
    print("Creating visualization...")
    visualizer = LLetterVisualizer()
    visualizer.animate_paths(hybrid_path, optimized_path, spiral_pocket_path)
    
    plt.tight_layout()
    plt.show()