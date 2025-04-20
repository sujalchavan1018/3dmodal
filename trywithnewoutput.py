import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.animation import FuncAnimation
import random
from tqdm import tqdm
from matplotlib.colors import LightSource
import matplotlib.patches as mpatches

def calculate_distance_matrix(points):
    """Vectorized distance matrix calculation for better performance"""
    points = np.array(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))

# Parameters
grid_size = 8  # Reduced from larger sizes for better performance
pocket_margin = 0.10
pocket_depth = 0.30
box_size = 1.0

def create_shape_points(shape='square'):
    x_vals = np.linspace(pocket_margin, box_size-pocket_margin, grid_size)
    y_vals = np.linspace(pocket_margin, box_size-pocket_margin, grid_size)
    
    if shape == 'square':
        return [[x, y, box_size] for x in x_vals for y in y_vals]
    elif shape == 'X':
        points = []
        for x, y in zip(x_vals, y_vals):
            points.append([x, y, box_size])
            points.append([x, y_vals[-1] - (y - y_vals[0]), box_size])
        return points
    elif shape == 'H':
        points = []
        for x in [x_vals[0], x_vals[-1]]:
            for y in y_vals:
                points.append([x, y, box_size])
        mid_y = y_vals[len(y_vals)//2]
        for x in x_vals[1:-1]:
            points.append([x, mid_y, box_size])
        return points
    elif shape == 'circle':
        center = [box_size/2, box_size/2]
        radius = (box_size - 2*pocket_margin)/2
        angles = np.linspace(0, 2*np.pi, grid_size**2, endpoint=False)
        return [
            [center[0] + radius*np.cos(a), 
             center[1] + radius*np.sin(a), 
             box_size] 
            for a in angles
        ]
    else:
        raise ValueError(f"Unknown shape: {shape}")

# GA Parameters
GA_PARAMS = {
    'pop_size': 20,
    'generations': 50,
    'mutation_rate': 0.3,
    'elitism': 5,
    'tournament_size': 7,
    'turn_penalty_weight': 0.2,
    'cluster_points': False
}

class ToolpathOptimizer:
    def __init__(self, points):
        self.points = np.array(points)
        self.distance_matrix = calculate_distance_matrix(self.points)
        self.clusters = None
        
    def path_length(self, path_indices):
        """Calculate total path length including turn penalties"""
        total = 0
        prev_dir = None
        turn_penalty = 0
        
        for i in range(len(path_indices)-1):
            current = path_indices[i]
            next_p = path_indices[i+1]
            
            # Add distance
            total += self.distance_matrix[current][next_p]
            
            # Calculate turn penalty if we have previous direction
            if prev_dir is not None and i > 0:
                current_dir = self.points[next_p] - self.points[current]
                current_dir = current_dir / np.linalg.norm(current_dir)
                
                # Calculate angle change (0-180 degrees)
                angle_change = np.degrees(np.arccos(np.clip(np.dot(prev_dir, current_dir), -1, 1)))
                turn_penalty += angle_change * GA_PARAMS['turn_penalty_weight']
            
            prev_dir = self.points[next_p] - self.points[current]
            prev_dir = prev_dir / np.linalg.norm(prev_dir)
        
        return total + turn_penalty
    
    def fitness(self, path_indices):
        return 1 / (self.path_length(path_indices) + 1e-8)
    
    def initialize_population(self):
        population = []
        
        # Create diverse initial population
        for _ in range(GA_PARAMS['pop_size']):
            if random.random() < 0.8:  # 80% good starts
                start = random.randint(0, len(self.points)-1)
                path = self._nearest_neighbor_path(start)
            else:
                # Random shuffle
                path = list(range(len(self.points)))
                random.shuffle(path)
            population.append(path)
        return population
    
    def _nearest_neighbor_path(self, start_index, points=None):
        """Create path using nearest neighbor heuristic"""
        if points is None:
            points = list(range(len(self.points)))
        else:
            points = list(points)  # make copy
            
        path = [start_index]
        points.remove(start_index)
        
        while points:
            last = path[-1]
            nearest = min(points, key=lambda x: self.distance_matrix[last][x])
            path.append(nearest)
            points.remove(nearest)
        return path
    
    def evolve(self):
        population = self.initialize_population()
        best_path = None
        best_fitness = -np.inf
        no_improvement = 0
        
        with tqdm(total=GA_PARAMS['generations'], desc="Optimizing Path") as pbar:
            for gen in range(GA_PARAMS['generations']):
                fitnesses = [self.fitness(ind) for ind in population]
                
                current_best_idx = np.argmax(fitnesses)
                if fitnesses[current_best_idx] > best_fitness:
                    best_fitness = fitnesses[current_best_idx]
                    best_path = population[current_best_idx]
                    no_improvement = 0
                else:
                    no_improvement += 1
                
                # Early stopping if no improvement for 20 generations
                if no_improvement >= 20 and gen > 50:
                    pbar.update(GA_PARAMS['generations'] - gen)
                    break
                
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
                
                # Apply local optimization to best path every generation
                best_path = self._two_opt(best_path)
                
                # Occasionally apply more expensive optimization
                if gen % 10 == 0:
                    best_path = self._three_opt(best_path)
                
                pbar.update(1)
                pbar.set_postfix({'best': 1/best_fitness})
        
        return [self.points[i] for i in best_path]
    
    def _tournament_select(self, population, fitnesses):
        contestants = random.sample(list(zip(population, fitnesses)), 
                                GA_PARAMS['tournament_size'])
        return max(contestants, key=lambda x: x[1])[0]
    
    def _crossover(self, parent1, parent2):
        """Order crossover (OX) with tournament selection"""
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
        """Mutation with different operations"""
        if random.random() < GA_PARAMS['mutation_rate']:
            # Choose mutation type
            if random.random() < 0.7:  # 70% swap mutation
                i, j = random.sample(range(len(path)), 2)
                path[i], path[j] = path[j], path[i]
            else:  # 30% inversion mutation
                i, j = sorted(random.sample(range(len(path)), 2))
                path[i:j+1] = path[i:j+1][::-1]
        return path
    
    def _two_opt(self, path):
        """2-opt optimization with turn penalty consideration"""
        best_path = path
        best_score = self.fitness(path)
        improved = True
        
        while improved:
            improved = False
            for i in range(1, len(path)-2):
                for j in range(i+1, len(path)):
                    if j - i == 1:
                        continue
                    new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
                    new_score = self.fitness(new_path)
                    if new_score > best_score:
                        best_path = new_path
                        best_score = new_score
                        improved = True
            path = best_path
        return best_path
    
    def _three_opt(self, path):
        """3-opt optimization for more complex improvements"""
        best_path = path
        best_score = self.fitness(path)
        
        for i in range(1, len(path)-4):
            for j in range(i+2, len(path)-2):
                for k in range(j+2, len(path)-1):
                    # Try all 7 possible 3-opt combinations
                    for variant in range(7):
                        new_path = self._three_opt_variant(path, i, j, k, variant)
                        new_score = self.fitness(new_path)
                        if new_score > best_score:
                            best_path = new_path
                            best_score = new_score
        return best_path
    
    def _three_opt_variant(self, path, i, j, k, variant):
        """Generate different 3-opt variants"""
        a, b, c = path[:i], path[i:j], path[j:k]
        d = path[k:]
        
        if variant == 0: return a + b + c + d
        elif variant == 1: return a + b[::-1] + c + d
        elif variant == 2: return a + b + c[::-1] + d
        elif variant == 3: return a + b[::-1] + c[::-1] + d
        elif variant == 4: return a + c + b + d
        elif variant == 5: return a + c + b[::-1] + d
        elif variant == 6: return a + c[::-1] + b + d
        else: return a + c[::-1] + b[::-1] + d

class BoxVisualizer:
    def __init__(self, shape='square'):
        self.fig = plt.figure(figsize=(20, 8), facecolor='white')
        self.ax1 = self.fig.add_subplot(121, projection='3d')  # Optimized
        self.ax2 = self.fig.add_subplot(122, projection='3d')  # Comparison
        self.shape = shape
        self._setup_lighting()
        self.setup_scenes()
        self.pocket_patches = []  # To store pocket visualization patches
    
    def _setup_lighting(self):
        self.light = LightSource(azdeg=225, altdeg=45)
        self.face_colors = {
            'base': '#4682B4', 'front': '#5F9EA0', 'back': '#B0C4DE',
            'right': '#87CEEB', 'left': '#ADD8E6', 'pocket': '#D3D3D3',
            'top': '#98FB98', 'grid': '#FFFFFF'
        }
    
    def _create_shape_outline(self, ax):
        """Create outline for the specific shape"""
        s = box_size
        m = pocket_margin
        
        if self.shape == 'X':
            lines = [
                [[m, m, s], [s-m, s-m, s]],  # Main diagonal
                [[s-m, m, s], [m, s-m, s]]    # Anti-diagonal
            ]
            ax.add_collection3d(Line3DCollection(
                lines, colors='red', linewidths=3, zorder=5
            ))
        elif self.shape == 'H':
            lines = [
                [[m, m, s], [m, s-m, s]],     # Left vertical
                [[s-m, m, s], [s-m, s-m, s]], # Right vertical
                [[m, s/2, s], [s-m, s/2, s]]  # Horizontal
            ]
            ax.add_collection3d(Line3DCollection(
                lines, colors='blue', linewidths=3, zorder=5
            ))
        elif self.shape == 'circle':
            center = [s/2, s/2]
            radius = (s - 2*m)/2
            angles = np.linspace(0, 2*np.pi, 100)
            circle_points = np.array([
                [center[0] + radius*np.cos(a), 
                 center[1] + radius*np.sin(a), 
                 s] 
                for a in angles
            ])
            ax.plot(circle_points[:,0], circle_points[:,1], circle_points[:,2], 
                   'g-', linewidth=3, zorder=5)
    
    def _create_box(self, ax):
        s = box_size
        m = pocket_margin
        d = pocket_depth
        
        # Main box vertices
        vertices = np.array([
            [0, 0, 0], [s, 0, 0], [s, s, 0], [0, s, 0],  # Bottom
            [0, 0, s], [s, 0, s], [s, s, s], [0, s, s]   # Top
        ])
        
        # Box faces (excluding top)
        faces = {
            'base': [vertices[[0, 1, 2, 3]]],
            'front': [vertices[[0, 1, 5, 4]]],
            'back': [vertices[[2, 3, 7, 6]]],
            'right': [vertices[[1, 2, 6, 5]]],
            'left': [vertices[[3, 0, 4, 7]]]
        }
        
        for name, face_verts in faces.items():
            poly = Poly3DCollection(
                face_verts,
                facecolors=self.face_colors[name],
                linewidths=1,
                edgecolors='k',
                zorder=1,
                alpha=0.7
            )
            ax.add_collection3d(poly)
        
        # Create top surface
        self.top_face = Poly3DCollection(
            [vertices[[4, 5, 6, 7]]],
            facecolors=self.face_colors['top'],
            linewidths=1,
            edgecolors='k',
            zorder=2,
            alpha=0.5
        )
        ax.add_collection3d(self.top_face)
        
        # Configure view
        ax.view_init(elev=30, azim=-50)
        ax.set_xlim(0, s)
        ax.set_ylim(0, s)
        ax.set_zlim(0, s*1.1)
        ax.set_xlabel('X Axis', fontsize=10)
        ax.set_ylabel('Y Axis', fontsize=10)
        ax.set_zlabel('Z Axis', fontsize=10)
        ax.grid(False)
        ax.set_facecolor('white')
    
    def setup_scenes(self):
        self._create_box(self.ax1)
        self._create_box(self.ax2)
        self._create_shape_outline(self.ax1)
        self._create_shape_outline(self.ax2)
        
        self.ax1.set_title('Optimized Toolpath', fontsize=12, pad=15)
        self.ax2.set_title('Comparison Path', fontsize=12, pad=15)
        self.fig.suptitle(f'Toolpath Optimization for {self.shape.upper()} Shape', 
                         fontsize=16, y=0.95)
    
    def _create_pocket_patch(self, ax, x, y):
        """Create a pocket (hole) at the specified position"""
        s = box_size
        d = pocket_depth
        r = 0.02  # Radius of the pocket
        
        # Create cylinder vertices for the pocket
        theta = np.linspace(0, 2*np.pi, 20)
        x_vals = x + r * np.cos(theta)
        y_vals = y + r * np.sin(theta)
        
        # Create vertical cylinder faces
        for z in np.linspace(s, s-d, 5):
            verts = []
            for t, (xi, yi) in enumerate(zip(x_vals, y_vals)):
                verts.append([xi, yi, z])
            
            poly = Poly3DCollection(
                [verts],
                facecolors=self.face_colors['pocket'],
                linewidths=0.5,
                edgecolors='k',
                zorder=3,
                alpha=0.9
            )
            ax.add_collection3d(poly)
            self.pocket_patches.append(poly)
        
        # Create bottom circle of the pocket
        bottom_circle = []
        for xi, yi in zip(x_vals, y_vals):
            bottom_circle.append([xi, yi, s-d])
        
        poly = Poly3DCollection(
            [bottom_circle],
            facecolors=self.face_colors['pocket'],
            linewidths=0.5,
            edgecolors='k',
            zorder=3,
            alpha=0.9
        )
        ax.add_collection3d(poly)
        self.pocket_patches.append(poly)
    
    def animate_paths(self, optimized_path, comparison_path):
        # Initialize both paths
        self.opt_line, = self.ax1.plot([], [], [], 'r-', linewidth=3, zorder=6)
        self.opt_drill, = self.ax1.plot([], [], [], 'ko', markersize=8, zorder=10)
        
        self.comp_line, = self.ax2.plot([], [], [], 'b-', linewidth=3, zorder=6)
        self.comp_drill, = self.ax2.plot([], [], [], 'ko', markersize=8, zorder=10)
        
        # Store path coordinates
        self.opt_x, self.opt_y, self.opt_z = [], [], []
        self.comp_x, self.comp_y, self.comp_z = [], [], []
        
        # Store the paths for animation
        self.optimized_path = optimized_path
        self.comparison_path = comparison_path
        
        # Create legend
        opt_patch = mpatches.Patch(color='red', label='Optimized')
        comp_patch = mpatches.Patch(color='blue', label='Comparison')
        self.fig.legend(handles=[opt_patch, comp_patch], loc='upper right')
        
        # Create animation
        max_frames = max(len(optimized_path), len(comparison_path))
        self.ani = FuncAnimation(
            self.fig, 
            self._update_animation,
            frames=max_frames,
            interval=100,
            blit=False,  # Changed to False to allow for pocket creation
            init_func=lambda: [self.opt_line, self.opt_drill, self.comp_line, self.comp_drill]
        )
    
    def _update_animation(self, frame):
        # Clear previous pocket patches
        for patch in self.pocket_patches:
            patch.remove()
        self.pocket_patches = []
        
        # Update optimized path
        if frame < len(self.optimized_path):
            x, y, z = self.optimized_path[frame]
            self.opt_drill.set_data([x], [y])
            self.opt_drill.set_3d_properties([z])
            self.opt_x.append(x)
            self.opt_y.append(y)
            self.opt_z.append(z)
            self.opt_line.set_data(self.opt_x, self.opt_y)
            self.opt_line.set_3d_properties(self.opt_z)
            
            # Create pocket at this position
            self._create_pocket_patch(self.ax1, x, y)
        
        # Update comparison path
        if frame < len(self.comparison_path):
            x, y, z = self.comparison_path[frame]
            self.comp_drill.set_data([x], [y])
            self.comp_drill.set_3d_properties([z])
            self.comp_x.append(x)
            self.comp_y.append(y)
            self.comp_z.append(z)
            self.comp_line.set_data(self.comp_x, self.comp_y)
            self.comp_line.set_3d_properties(self.comp_z)
            
            # Create pocket at this position
            self._create_pocket_patch(self.ax2, x, y)
        
        return [self.opt_line, self.opt_drill, self.comp_line, self.comp_drill] + self.pocket_patches

def create_comparison_path(points, method='zigzag'):
    points = np.array(points)
    if method == 'zigzag':
        sorted_points = sorted(points, key=lambda p: (p[0], p[1]))
        path = []
        for i, x in enumerate(np.unique([p[0] for p in sorted_points])):
            x_points = [p for p in sorted_points if p[0] == x]
            if i % 2 == 0:
                path.extend(x_points)
            else:
                path.extend(reversed(x_points))
        return path
    elif method == 'spiral':
        center = np.mean(points, axis=0)
        angles = np.arctan2(points[:,1]-center[1], points[:,0]-center[0])
        dists = np.linalg.norm(points[:,:2]-center[:2], axis=1)
        return points[np.lexsort((dists, angles))]
    else:
        raise ValueError(f"Unknown comparison method: {method}")

if __name__ == "__main__":
    shapes = ['square', 'X', 'H', 'circle']
    
    for shape in shapes[:1]:  # Just run square for testing
        print(f"\nProcessing {shape} shape...")
        points = create_shape_points(shape)
        
        print("Optimizing toolpath...")
        optimizer = ToolpathOptimizer(points)
        optimized_path = optimizer.evolve()
        
        print("Creating comparison path...")
        comparison_path = create_comparison_path(points, method='zigzag')
        
        opt_indices = [np.where((points == p).all(axis=1))[0][0] for p in optimized_path]
        comp_indices = [np.where((points == p).all(axis=1))[0][0] for p in comparison_path]
        
        opt_length = optimizer.path_length(opt_indices)
        comp_length = optimizer.path_length(comp_indices)
        
        print(f"\n{shape.upper()} Shape Results:")
        print(f"Optimized Path Length: {opt_length:.4f} units")
        print(f"Comparison Path Length: {comp_length:.4f} units")
        print(f"Improvement: {(1 - opt_length/comp_length)*100:.2f}% shorter")
        
        print("Creating visualization...")
        visualizer = BoxVisualizer(shape)
        visualizer.animate_paths(optimized_path, comparison_path)
        
        plt.tight_layout()
        plt.show()