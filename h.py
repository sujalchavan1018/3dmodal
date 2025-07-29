import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation
import random
from tqdm import tqdm
from matplotlib.colors import LightSource
from scipy.interpolate import make_interp_spline

# Parameters with some randomness
grid_size = 8
pocket_margin = 0.10
pocket_depth = 0.30
h_width = 3.0
h_height = 3.0
h_depth = 1.5

# Create grid points with some randomness
def create_grid_with_randomness():
    x_vals = np.linspace(pocket_margin, h_width-pocket_margin, grid_size)
    y_vals = np.linspace(pocket_margin, h_height-pocket_margin, grid_size)
    grid_points = []
    
    for x in x_vals:
        for y in y_vals:
            # Add some randomness to positions (10% of grid spacing)
            rand_x = random.uniform(-0.03, 0.03)
            rand_y = random.uniform(-0.03, 0.03)
            
            if (x < 0.8 or x > 2.2 or (y > 1.1 and y < 1.9)):
                if 1.0 < x < 2.0 and 1.3 < y < 1.7:
                    grid_points.append([x + rand_x, y + rand_y, h_depth - pocket_depth])
                else:
                    grid_points.append([x + rand_x, y + rand_y, h_depth])
    return grid_points

grid_points = create_grid_with_randomness()

# GA Parameters
GA_PARAMS = {
    'pop_size': 70,
    'generations': 150,
    'mutation_rate': 0.5,
    'elitism': 3,
    'tournament_size': 5
}

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
        
        return best_path
    
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

def create_zigzag_path():
    """Create a simple zigzag path for comparison"""
    # Sort points in a simple zigzag pattern
    sorted_points = sorted(grid_points, key=lambda p: (p[1], p[0] if p[1] % 0.2 < 0.1 else -p[0]))
    return sorted_points

def find_closest_point_index(target_point, points_list):
    """Find the index of the closest point in points_list to target_point"""
    distances = [np.linalg.norm(np.array(target_point) - np.array(p)) for p in points_list]
    return np.argmin(distances)

class HLetterVisualizer:
    def __init__(self):
        self.fig = plt.figure(figsize=(18, 8), facecolor='white')
        self.ax1 = self.fig.add_subplot(121, projection='3d')  # Optimized path
        self.ax2 = self.fig.add_subplot(122, projection='3d')  # Zigzag path
        self._setup_lighting()
        self.setup_scenes()
    
    def _setup_lighting(self):
        self.light = LightSource(azdeg=225, altdeg=45)
        self.face_colors = {
            'base': '#4682B4', 'front': '#5F9EA0', 'back': '#B0C4DE',
            'right': '#87CEEB', 'left': '#ADD8E6', 'pocket': '#D3D3D3',
            'top': '#98FB98', 'grid': '#FFFFFF'
        }
    
    def _create_h_letter(self, ax):
        # Create H letter geometry
        left_front = np.array([
            [0, 0, 0], [0, h_height, 0], [0.8, h_height, 0], [0.8, 0, 0],
            [0, 0, h_depth], [0, h_height, h_depth], [0.8, h_height, h_depth], [0.8, 0, h_depth]
        ])
        
        right_front = np.array([
            [2.2, 0, 0], [2.2, h_height, 0], [h_width, h_height, 0], [h_width, 0, 0],
            [2.2, 0, h_depth], [2.2, h_height, h_depth], [h_width, h_height, h_depth], [h_width, 0, h_depth]
        ])
        
        horizontal = np.array([
            [0.8, 1.1, 0], [0.8, 1.9, 0], [2.2, 1.9, 0], [2.2, 1.1, 0],
            [0.8, 1.1, h_depth], [0.8, 1.9, h_depth], [2.2, 1.9, h_depth], [2.2, 1.1, h_depth]
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
        
        for vertices, color_factor in [(left_front, 1.0), (right_front, 1.0), (horizontal, 0.9)]:
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
            if 1.0 < point[0] < 2.0 and 1.3 < point[1] < 1.7:
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
        ax.set_xlim(0, h_width)
        ax.set_ylim(0, h_height)
        ax.set_zlim(0, h_depth*1.1)
        ax.set_xlabel('X Axis', fontsize=10)
        ax.set_ylabel('Y Axis', fontsize=10)
        ax.set_zlabel('Z Axis', fontsize=10)
        ax.grid(False)
        ax.set_facecolor('white')
    
    def setup_scenes(self):
        self._create_h_letter(self.ax1)
        self._create_h_letter(self.ax2)
        self.ax1.set_title('Optimized Path (Shortest Distance)', fontsize=12, pad=15)
        self.ax2.set_title('Zigzag Path (Inefficient)', fontsize=12, pad=15)
        self.fig.suptitle('Toolpath Comparison: Optimized vs. Zigzag', fontsize=16, y=0.95)
    
    def animate_paths(self, optimized_path, zigzag_path):
        # Initialize paths
        self.opt_line, = self.ax1.plot([], [], [], color='lime', linewidth=3, zorder=6, alpha=0.9)
        self.opt_drill, = self.ax1.plot([], [], [], 'o', color='yellow', markersize=8, 
                                      markeredgecolor='black', linewidth=1, zorder=10)
        
        self.zig_line, = self.ax2.plot([], [], [], color='red', linewidth=3, zorder=6, alpha=0.7)
        self.zig_drill, = self.ax2.plot([], [], [], 'o', color='orange', markersize=8, 
                                      markeredgecolor='black', linewidth=1, zorder=10)
        
        # Store path coordinates
        self.opt_x, self.opt_y, self.opt_z = [], [], []
        self.zig_x, self.zig_y, self.zig_z = [], [], []
        
        # Create animation
        max_frames = max(len(optimized_path), len(zigzag_path))
        self.ani = FuncAnimation(
            self.fig, 
            self._update_animation,
            frames=max_frames,
            interval=50,
            blit=True,
            init_func=lambda: [self.opt_line, self.opt_drill, 
                             self.zig_line, self.zig_drill],
            fargs=(optimized_path, zigzag_path)
        )
    
    def _update_animation(self, frame, opt_path, zig_path):
        # Update optimized path
        if frame < len(opt_path):
            x, y, z = opt_path[frame]
            self.opt_drill.set_data([x], [y])
            self.opt_drill.set_3d_properties([z])
            
            self.opt_x.append(x)
            self.opt_y.append(y)
            self.opt_z.append(z)
            
            if len(self.opt_x) > 1:
                # Create smooth path
                t = np.linspace(0, 1, len(self.opt_x))
                try:
                    spl_x = make_interp_spline(t, self.opt_x, k=3)
                    spl_y = make_interp_spline(t, self.opt_y, k=3)
                    spl_z = make_interp_spline(t, self.opt_z, k=3)
                    t_new = np.linspace(0, 1, len(self.opt_x)*3)
                    x_smooth = spl_x(t_new)
                    y_smooth = spl_y(t_new)
                    z_smooth = spl_z(t_new)
                    self.opt_line.set_data(x_smooth, y_smooth)
                    self.opt_line.set_3d_properties(z_smooth)
                except:
                    self.opt_line.set_data(self.opt_x, self.opt_y)
                    self.opt_line.set_3d_properties(self.opt_z)
        
        # Update zigzag path
        if frame < len(zig_path):
            x, y, z = zig_path[frame]
            self.zig_drill.set_data([x], [y])
            self.zig_drill.set_3d_properties([z])
            
            self.zig_x.append(x)
            self.zig_y.append(y)
            self.zig_z.append(z)
            
            if len(self.zig_x) > 1:
                self.zig_line.set_data(self.zig_x, self.zig_y)
                self.zig_line.set_3d_properties(self.zig_z)
        
        return self.opt_line, self.opt_drill, self.zig_line, self.zig_drill

# Main execution
if __name__ == "__main__":
    print("Optimizing toolpath...")
    optimizer = ToolpathOptimizer(grid_points)
    optimized_path_indices = optimizer.evolve()
    optimized_path = [grid_points[i] for i in optimized_path_indices]
    zigzag_path = create_zigzag_path()
    
    # Calculate path lengths
    opt_length = optimizer.path_length(optimized_path_indices)
    zigzag_indices = [find_closest_point_index(p, grid_points) for p in zigzag_path]
    zigzag_length = optimizer.path_length(zigzag_indices)
    
    print(f"\nOptimized Path Length: {opt_length:.4f} units")
    print(f"Zigzag Path Length: {zigzag_length:.4f} units")
    print(f"Difference: {zigzag_length-opt_length:.4f} units ({(zigzag_length/opt_length-1)*100:.2f}% longer)")
    
    print("Creating visualization...")
    visualizer = HLetterVisualizer()
    visualizer.animate_paths(optimized_path, zigzag_path)
    
    plt.tight_layout()
    plt.show()