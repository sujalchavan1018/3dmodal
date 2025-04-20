import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.animation import FuncAnimation
import random
from tqdm import tqdm
from matplotlib.colors import LightSource
from scipy.interpolate import make_interp_spline

# Parameters
grid_size = 8
pocket_margin = 0.10
pocket_depth = 0.30
l_width = 3.0  # Width of L letter
l_height = 3.0  # Height of L letter
l_depth = 1.5   # Depth of L letter
smoothness_factor = 0.3  # Controls how smooth the curves are (0-1)

# Create grid points on the top surface of the L letter
x_vals = np.linspace(pocket_margin, l_width-pocket_margin, grid_size)
y_vals = np.linspace(pocket_margin, l_height-pocket_margin, grid_size)
grid_points = [[x, y, l_depth] for x in x_vals for y in y_vals 
              if (x < 0.8 or y < 0.8)]  # Only points on L surface

# Optimized GA parameters
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
        total = 0
        for i in range(len(path_indices)-1):
            p1 = np.array(self.points[path_indices[i]])
            p2 = np.array(self.points[path_indices[i+1]])
            total += np.linalg.norm(p1 - p2)
        return total
    
    def fitness(self, path_indices):
        # Penalize both distance and number of turns
        distance = self.path_length(path_indices)
        turns = self.count_turns(path_indices)
        return 1 / (distance + turns*0.2 + 1e-8)  # Weighted fitness
    
    def count_turns(self, path_indices):
        if len(path_indices) < 3:
            return 0
            
        turns = 0
        for i in range(1, len(path_indices)-1):
            v1 = np.array(self.points[path_indices[i-1]]) - np.array(self.points[path_indices[i]])
            v2 = np.array(self.points[path_indices[i+1]]) - np.array(self.points[path_indices[i]])
            angle = np.arccos(np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-8))
            if angle > np.pi/4:  # Count angles > 45 degrees as turns
                turns += 1
        return turns
    
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
        best_fitness = self.fitness(path)
        improved = True
        
        while improved:
            improved = False
            for i in range(1, len(path)-2):
                for j in range(i+1, len(path)):
                    if j - i == 1:
                        continue
                    new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
                    new_fitness = self.fitness(new_path)
                    if new_fitness > best_fitness:
                        best_path = new_path
                        best_fitness = new_fitness
                        improved = True
            path = best_path
        return best_path
    
    def _three_opt(self, path):
        best_path = path
        best_fitness = self.fitness(path)
        
        for i in range(1, len(path)-3):
            for j in range(i+1, len(path)-2):
                for k in range(j+1, len(path)-1):
                    new_path = path[:i] + path[i:j][::-1] + path[j:k][::-1] + path[k:]
                    new_fitness = self.fitness(new_path)
                    if new_fitness > best_fitness:
                        best_path = new_path
                        best_fitness = new_fitness
        return best_path

def create_zigzag_path():
    path = []
    # Create zigzag pattern only on L surface
    for i, x in enumerate(x_vals):
        if i % 2 == 0:
            for y in y_vals:
                if (x < 0.8 or y < 0.8):
                    path.append([x, y, l_depth])
        else:
            for y in reversed(y_vals):
                if (x < 0.8 or y < 0.8):
                    path.append([x, y, l_depth])
    return path

def smooth_path(path, smoothness=0.3):
    """Convert straight path segments into smooth curves"""
    if len(path) < 3:
        return path
    
    # Extract coordinates
    x = [p[0] for p in path]
    y = [p[1] for p in path]
    z = [p[2] for p in path]
    
    # Create spline for each dimension
    t = np.linspace(0, 1, len(path))
    t_new = np.linspace(0, 1, int(len(path)*(1+smoothness)))
    
    try:
        spl_x = make_interp_spline(t, x, k=3)
        spl_y = make_interp_spline(t, y, k=3)
        spl_z = make_interp_spline(t, z, k=3)
        
        x_smooth = spl_x(t_new)
        y_smooth = spl_y(t_new)
        z_smooth = spl_z(t_new)
    except:
        # Fallback if spline fails
        return path
    
    # Combine back into path
    return list(zip(x_smooth, y_smooth, z_smooth))

class LLetterVisualizer:
    def __init__(self):
        self.fig = plt.figure(figsize=(20, 8), facecolor='white')
        self.ax1 = self.fig.add_subplot(121, projection='3d')  # Optimized
        self.ax2 = self.fig.add_subplot(122, projection='3d')  # Zigzag
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
        # Vertical bar vertices
        vertical = np.array([
            [0, 0, 0], [0, l_height, 0], [0.8, l_height, 0], [0.8, 0, 0],
            [0, 0, l_depth], [0, l_height, l_depth], [0.8, l_height, l_depth], [0.8, 0, l_depth]
        ])
        
        # Horizontal bar vertices
        horizontal = np.array([
            [0, 0, 0], [0, 0.8, 0], [l_width, 0.8, 0], [l_width, 0, 0],
            [0, 0, l_depth], [0, 0.8, l_depth], [l_width, 0.8, l_depth], [l_width, 0, l_depth]
        ])
        
        # Define faces for each part
        def get_faces(vertices):
            return [
                [vertices[0], vertices[1], vertices[2], vertices[3]],  # front
                [vertices[4], vertices[5], vertices[6], vertices[7]],  # back
                [vertices[0], vertices[1], vertices[5], vertices[4]],  # top
                [vertices[2], vertices[3], vertices[7], vertices[6]],  # bottom
                [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
                [vertices[0], vertices[3], vertices[7], vertices[4]]   # left
            ]
        
        # Create all parts with shading
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
        
        # Add grid points
        xs, ys, zs = zip(*grid_points)
        ax.scatter(
            xs, ys, zs, 
            color=self.face_colors['grid'], 
            s=40, 
            edgecolor='k',
            linewidth=1,
            zorder=5
        )
        
        # Configure view
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
        self.ax1.set_title('Optimized Toolpath', fontsize=12, pad=15)
        self.ax2.set_title('Zigzag Toolpath', fontsize=12, pad=15)
        self.fig.suptitle('L Letter Toolpath Comparison', fontsize=16, y=0.95)
    
    def animate_paths(self, optimized_path, zigzag_path):
        # Smooth the paths
        opt_smooth = smooth_path(optimized_path, smoothness_factor)
        zz_smooth = smooth_path(zigzag_path, smoothness_factor)
        
        # Initialize both paths
        self.opt_line, = self.ax1.plot([], [], [], 'r-', linewidth=2, zorder=6)
        self.opt_drill, = self.ax1.plot([], [], [], 'ko', markersize=8, zorder=10)
        
        self.zz_line, = self.ax2.plot([], [], [], 'b-', linewidth=2, zorder=6)
        self.zz_drill, = self.ax2.plot([], [], [], 'ko', markersize=8, zorder=10)
        
        # Store path coordinates
        self.opt_x, self.opt_y, self.opt_z = [], [], []
        self.zz_x, self.zz_y, self.zz_z = [], [], []
        
        # Create animation
        max_frames = max(len(opt_smooth), len(zz_smooth))
        self.ani = FuncAnimation(
            self.fig, 
            self._update_animation,
            frames=max_frames,
            interval=100,  # Faster animation
            blit=True,
            init_func=lambda: [self.opt_line, self.opt_drill, self.zz_line, self.zz_drill],
            fargs=(opt_smooth, zz_smooth)
        )
    
    def _update_animation(self, frame, opt_path, zz_path):
        # Update optimized path
        if frame < len(opt_path):
            x, y, z = opt_path[frame]
            self.opt_drill.set_data([x], [y])
            self.opt_drill.set_3d_properties([z])
            self.opt_x.append(x)
            self.opt_y.append(y)
            self.opt_z.append(z)
            self.opt_line.set_data(self.opt_x, self.opt_y)
            self.opt_line.set_3d_properties(self.opt_z)
        
        # Update zigzag path
        if frame < len(zz_path):
            x, y, z = zz_path[frame]
            self.zz_drill.set_data([x], [y])
            self.zz_drill.set_3d_properties([z])
            self.zz_x.append(x)
            self.zz_y.append(y)
            self.zz_z.append(z)
            self.zz_line.set_data(self.zz_x, self.zz_y)
            self.zz_line.set_3d_properties(self.zz_z)
        
        return self.opt_line, self.opt_drill, self.zz_line, self.zz_drill

# Main execution
if __name__ == "__main__":
    print("Optimizing toolpath...")
    optimizer = ToolpathOptimizer(grid_points)
    optimized_path = optimizer.evolve()
    zigzag_path = create_zigzag_path()
    
    # Calculate path lengths
    opt_length = optimizer.path_length([grid_points.index(p) for p in optimized_path])
    zz_length = optimizer.path_length([grid_points.index(p) for p in zigzag_path])
    
    print(f"\nOptimized Path Length: {opt_length:.4f} units")
    print(f"Zigzag Path Length: {zz_length:.4f} units")
    print(f"Improvement: {(1 - opt_length/zz_length)*100:.2f}% shorter")
    
    print("Creating visualization...")
    visualizer = LLetterVisualizer()
    visualizer.animate_paths(optimized_path, zigzag_path)
    
    plt.tight_layout()
    plt.show()

    # toolpath where not so clear path