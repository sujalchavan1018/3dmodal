
"""
Toolpath Optimization Dashboard

A comprehensive application for visualizing and optimizing toolpaths using genetic algorithms.
The tool generates and compares different toolpath strategies (optimized, zigzag, hybrid) for various shapes.

Key Features:
- 3D visualization of toolpaths
- Genetic algorithm optimization
- Multiple shape types (letters, geometric shapes)
- Animation controls
- Data export capabilities

Classes:
    ToolpathOptimizer: Genetic algorithm implementation for path optimization
    ShapeVisualizer: Handles 3D visualization of shapes and toolpaths
    ToolpathDashboard: Main application class with GUI interface

Author: sujal chavan
Date: [Current Date]
Version: 1.0
"""

# Import necessary libraries
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.animation import FuncAnimation, PillowWriter
import random
from tqdm import tqdm
from matplotlib.colors import LightSource
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import csv
import os
from scipy.interpolate import make_interp_spline
from matplotlib.patches import Circle, Polygon, Rectangle
import math
from PIL import Image, ImageTk
import webbrowser

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

# Modern color scheme for the UI
BG_COLOR = "#f0f0f0"          # Background color
ACCENT_COLOR = "#4a6fa5"       # Primary accent color
HIGHLIGHT_COLOR = "#6c8fc7"    # Highlight color
TEXT_COLOR = "#333333"         # Main text color
BUTTON_COLOR = "#5d8bb7"       # Button color
DARK_BG = "#2c3e50"            # Dark background for headers

# Default parameters for the shapes 
grid_size = 8                  # Default grid resolution
pocket_margin = 0.10           # Margin around pockets
pocket_depth = 0.30            # Depth of pockets
letter_width = 3.0             # Default width for letter shapes
letter_height = 3.0            # Default height for letter shapes
letter_depth = 1.5             # Default depth for letter shapes
box_size = 1.0                 # Default size for box shape

# Genetic Algorithm default parameters
GA_PARAMS = {
    'pop_size': 70,            # Population size
    'generations': 150,        # Number of generations
    'mutation_rate': 0.5,      # Mutation probability
    'elitism': 3,              # Number of elite individuals preserved
    'tournament_size': 5       # Tournament selection size
}

# =============================================================================
# TOOLPATH OPTIMIZATION CLASS
# =============================================================================

class ToolpathOptimizer:
    """
    Optimizes toolpaths using a genetic algorithm approach to solve the Traveling Salesman Problem.
    
    Attributes:
        points (list): List of 3D points representing toolpath locations
        distance_matrix (numpy.ndarray): Precomputed distance matrix between all points
    """
    
    def __init__(self, points):
        """
        Initialize the optimizer with a set of points.
        
        Args:
            points (list): List of (x,y,z) coordinates representing toolpath locations
        """
        self.points = points
        self.distance_matrix = self._precompute_distances()
        
    def _precompute_distances(self):
        """Precompute the distance matrix between all points for faster calculations."""
        n = len(self.points)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(np.array(self.points[i]) - np.array(self.points[j]))
                dist_matrix[i][j] = dist_matrix[j][i] = dist
        return dist_matrix
    
    def path_length(self, path_indices):
        """
        Calculate the total length of a path given by indices.
        
        Args:
            path_indices (list): List of point indices representing the path
            
        Returns:
            float: Total length of the path
        """
        return sum(self.distance_matrix[path_indices[i]][path_indices[i+1]] 
               for i in range(len(path_indices)-1))
        
    def fitness(self, path_indices):
        """
        Calculate fitness of a path (inverse of path length).
        
        Args:
            path_indices (list): List of point indices representing the path
            
        Returns:
            float: Fitness value (inverse of path length)
        """
        return 1 / (self.path_length(path_indices) + 1e-8)  # Add small value to avoid division by zero
    
    def initialize_population(self):
        """
        Initialize population with a mix of random paths and nearest neighbor paths.
        
        Returns:
            list: List of paths (each path is a list of indices)
        """
        population = []
        for _ in range(GA_PARAMS['pop_size']):
            if random.random() < 0.9:  # 90% chance to use nearest neighbor
                start = random.randint(0, len(self.points)-1)
                path = self._nearest_neighbor_path(start)
            else:  # 10% chance to use random path
                path = list(range(len(self.points)))
                random.shuffle(path)
            population.append(path)
        return population
    
    def _nearest_neighbor_path(self, start_index):
        """
        Generate a path using the nearest neighbor heuristic.
        
        Args:
            start_index (int): Index of starting point
            
        Returns:
            list: Path generated using nearest neighbor algorithm
        """
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
        """
        Run the genetic algorithm optimization.
        
        Returns:
            list: Optimized path as a list of indices
        """
        population = self.initialize_population()
        best_path = None
        best_fitness = -np.inf
        
        # Progress bar for optimization process
        with tqdm(total=GA_PARAMS['generations'], desc="Optimizing Path") as pbar:
            for gen in range(GA_PARAMS['generations']):
                # Evaluate fitness
                fitnesses = [self.fitness(ind) for ind in population]
                
                # Track best solution
                current_best_idx = np.argmax(fitnesses)
                if fitnesses[current_best_idx] > best_fitness:
                    best_fitness = fitnesses[current_best_idx]
                    best_path = population[current_best_idx]
                
                # Create new population
                new_pop = []
                
                # Elitism: keep top individuals
                elite_indices = np.argsort(fitnesses)[-GA_PARAMS['elitism']:]
                new_pop.extend([population[i] for i in elite_indices])
                
                # Generate offspring
                while len(new_pop) < GA_PARAMS['pop_size']:
                    parent1 = self._tournament_select(population, fitnesses)
                    parent2 = self._tournament_select(population, fitnesses)
                    child = self._crossover(parent1, parent2)
                    child = self._mutate(child)
                    new_pop.append(child)
                
                population = new_pop
                
                # Apply local optimization
                best_path = self._two_opt(best_path)
                
                # Periodically apply more intensive optimization
                if gen % 20 == 0:
                    best_path = self._three_opt(best_path)
                
                pbar.update(1)
        
        return best_path
    
    def _tournament_select(self, population, fitnesses):
        """
        Tournament selection for parent selection.
        
        Args:
            population (list): Current population of paths
            fitnesses (list): Fitness values for each path
            
        Returns:
            list: Selected parent path
        """
        contestants = random.sample(list(zip(population, fitnesses)), 
                                GA_PARAMS['tournament_size'])
        return max(contestants, key=lambda x: x[1])[0]
    
    def _crossover(self, parent1, parent2):
        """
        Ordered crossover for path recombination.
        
        Args:
            parent1 (list): First parent path
            parent2 (list): Second parent path
            
        Returns:
            list: Child path created through crossover
        """
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
        """
        Swap mutation operator.
        
        Args:
            path (list): Path to mutate
            
        Returns:
            list: Mutated path
        """
        if random.random() < GA_PARAMS['mutation_rate']:
            i, j = random.sample(range(len(path)), 2)
            path[i], path[j] = path[j], path[i]
        return path
    
    def _two_opt(self, path):
        """
        2-opt local optimization for path improvement.
        
        Args:
            path (list): Path to optimize
            
        Returns:
            list: Optimized path
        """
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
        """
        3-opt local optimization for path improvement.
        
        Args:
            path (list): Path to optimize
            
        Returns:
            list: Optimized path
        """
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

# =============================================================================
# PATH GENERATION FUNCTIONS
# =============================================================================

def create_zigzag_path(x_vals, y_vals, width, height, depth, shape_type):
    """
    Create a zigzag toolpath for a given shape type.
    
    Args:
        x_vals (list): X-coordinates of grid points
        y_vals (list): Y-coordinates of grid points
        width (float): Width of the shape
        height (float): Height of the shape
        depth (float): Depth of the shape
        shape_type (str): Type of shape ('H', 'I', 'L', etc.)
        
    Returns:
        list: List of (x,y,z) points forming the zigzag path
    """
    path = []
    x_vals = sorted(x_vals)
    y_vals = sorted(y_vals)
    
    # Shape-specific path generation
    if shape_type == "H":
        # Left vertical bar
        for i, y in enumerate(y_vals):
            if i % 2 == 0:
                path.append([x_vals[0], y, depth])
            else:
                path.append([x_vals[1], y, depth])
        
        # Right vertical bar
        for i, y in enumerate(reversed(y_vals)):
            if i % 2 == 0:
                path.append([x_vals[-1], y, depth])
            else:
                path.append([x_vals[-2], y, depth])
        
        # Horizontal bar - connect with smooth transitions
        cross_y1 = height * 0.37
        cross_y2 = height * 0.63
        for x in x_vals[2:-2]:
            path.append([x, cross_y1, depth])
            path.append([x, cross_y2, depth])
    
    elif shape_type == "I":
        # Vertical bar with small horizontal caps
        center_x = width / 2
        bar_width = 0.8
        
        # Main vertical bar
        for i, y in enumerate(y_vals):
            if i % 2 == 0:
                path.append([center_x - bar_width/2, y, depth])
                path.append([center_x + bar_width/2, y, depth])
            else:
                path.append([center_x + bar_width/2, y, depth])
                path.append([center_x - bar_width/2, y, depth])
        
        # Top cap
        for x in np.linspace(center_x - bar_width, center_x + bar_width, len(x_vals)):
            path.append([x, y_vals[-1], depth])
        
        # Bottom cap
        for x in np.linspace(center_x + bar_width, center_x - bar_width, len(x_vals)):
            path.append([x, y_vals[0], depth])
    
    elif shape_type == "L":
        # Vertical part
        for i, y in enumerate(y_vals):
            if i % 2 == 0:
                path.append([x_vals[0], y, depth])
                path.append([x_vals[1], y, depth])
            else:
                path.append([x_vals[1], y, depth])
                path.append([x_vals[0], y, depth])
        
        # Horizontal base
        for x in x_vals[2:]:
            path.append([x, y_vals[0], depth])
    
    elif shape_type == "X":
        # Diagonal paths
        for i, x in enumerate(x_vals):
            y1 = (height/width) * x
            y2 = height - (height/width) * x
            path.append([x, y1, depth])
            path.append([x, y2, depth])
    
    elif shape_type == "T":
        # Horizontal top
        for x in x_vals:
            path.append([x, y_vals[-1], depth])
        
        # Vertical center
        center_x = width / 2
        bar_width = 0.8
        for y in y_vals[:-1]:
            path.append([center_x - bar_width/2, y, depth])
            path.append([center_x + bar_width/2, y, depth])
    
    elif shape_type == "Box":
        # Perimeter path
        margin = 0.2
        # Bottom edge
        for x in np.linspace(margin, width-margin, len(x_vals)):
            path.append([x, margin, depth])
        
        # Right edge
        for y in np.linspace(margin, height-margin, len(y_vals)):
            path.append([width-margin, y, depth])
        
        # Top edge
        for x in np.linspace(width-margin, margin, len(x_vals)):
            path.append([x, height-margin, depth])
        
        # Left edge
        for y in np.linspace(height-margin, margin, len(y_vals)):
            path.append([margin, y, depth])
    
    elif shape_type == "Triangle":
        # Right-angle triangle path
        margin = 0.2
        for i, x in enumerate(np.linspace(margin, width-margin, len(x_vals))):
            y_max = height * (1 - x/width) - margin
            y_points = np.linspace(margin, y_max, len(y_vals))
            if i % 2 == 0:
                for y in y_points:
                    path.append([x, y, depth])
            else:
                for y in reversed(y_points):
                    path.append([x, y, depth])
    
    elif shape_type == "Cylinder":
        # Circular path
        center_x, center_y = width/2, height/2
        radius = min(width, height)/2 * 0.8
        n_rings = 5
        n_points = 20
        
        # Spiral from outside in
        for r in np.linspace(radius, 0.2, n_rings):
            angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
            if (radius - r) % (2 * radius/n_rings) < radius/n_rings:
                angles = angles[::-1]  # Alternate direction
            
            for angle in angles:
                x = center_x + r * np.cos(angle)
                y = center_y + r * np.sin(angle)
                path.append([x, y, depth])
    
    elif shape_type == "Pentagon":
        # Pentagon path
        center_x, center_y = width/2, height/2
        radius = min(width, height)/2 * 0.8
        angles = np.linspace(0, 2*np.pi, 6)[:-1]  # 5 points
        
        # Create concentric pentagons
        n_layers = 5
        for layer in range(n_layers):
            current_radius = radius * (1 - layer/n_layers)
            if layer % 2 == 0:
                current_angles = angles
            else:
                current_angles = angles[::-1]
            
            for angle in current_angles:
                x = center_x + current_radius * np.cos(angle)
                y = center_y + current_radius * np.sin(angle)
                path.append([x, y, depth])
    
    return path

def create_spiral_pocket_path(grid_points, width, height, depth, shape_type):
    """
    Create a spiral path specifically for the pocket area with smooth transitions.
    
    Args:
        grid_points (list): List of all grid points
        width (float): Width of the shape
        height (float): Height of the shape
        depth (float): Depth of the shape
        shape_type (str): Type of shape ('H', 'I', 'L', etc.)
        
    Returns:
        list: List of (x,y,z) points forming the spiral pocket path
    """
    if shape_type == "H":
        pocket_points = [p for p in grid_points if 1.0 < p[0] < width-1.0 and 
                        height*0.37 < p[1] < height*0.63]
    elif shape_type == "I":
        center_x = width / 2
        bar_width = 0.8
        pocket_points = [p for p in grid_points if center_x - bar_width/2 <= p[0] <= center_x + bar_width/2]
    elif shape_type == "L":
        pocket_points = [p for p in grid_points if p[0] < 0.8 or p[1] < 0.8]
    elif shape_type == "X":
        bar_width = 0.5
        pocket_points = []
        for p in grid_points:
            y1 = (height/width) * p[0]
            y2 = height - (height/width) * p[0]
            if abs(p[1] - y1) < bar_width/2 or abs(p[1] - y2) < bar_width/2:
                pocket_points.append(p)
    elif shape_type == "T":
        center_x = width / 2
        bar_width = 0.8
        top_height = 0.3
        pocket_points = [p for p in grid_points if (p[1] > height - top_height) or 
                       (center_x - bar_width/2 <= p[0] <= center_x + bar_width/2)]
    elif shape_type == "Box":
        # Inner box for pocket
        margin = 0.8
        pocket_points = [p for p in grid_points if 
                        margin <= p[0] <= width-margin and 
                        margin <= p[1] <= height-margin]
    elif shape_type == "Triangle":
        # Inner triangle for pocket
        pocket_points = []
        for p in grid_points:
            y_max = height * (1 - p[0]/width) * 0.8  # Smaller triangle
            if p[1] <= y_max:
                pocket_points.append(p)
    elif shape_type == "Cylinder":
        # Circular pocket
        center_x, center_y = width/2, height/2
        radius = min(width, height)/2 * 0.7
        pocket_points = []
        for p in grid_points:
            dist = math.sqrt((p[0]-center_x)**2 + (p[1]-center_y)**2)
            if dist <= radius:
                pocket_points.append(p)
    elif shape_type == "Pentagon":
        # Inner pentagon for pocket
        center_x, center_y = width/2, height/2
        radius = min(width, height)/2 * 0.7
        angles = np.linspace(0, 2*np.pi, 6)[:-1]  # 5 points
        vertices = [(center_x + radius * np.cos(a), center_y + radius * np.sin(a)) for a in angles]
        poly = Polygon(vertices)
        pocket_points = [p for p in grid_points if poly.contains_point((p[0], p[1]))]
    
    if not pocket_points:
        return []
        
    # Calculate center of pocket
    center_x = np.mean([p[0] for p in pocket_points])
    center_y = np.mean([p[1] for p in pocket_points])
    
    # Sort points in a spiral pattern by angle and distance from center
    pocket_points_sorted = sorted(pocket_points,
                                key=lambda p: (np.arctan2(p[1]-center_y, p[0]-center_x),
                                np.hypot(p[0]-center_x, p[1]-center_y)))
    
    # Create smooth spiral path
    spiral_path = []
    angles = [np.arctan2(p[1]-center_y, p[0]-center_x) for p in pocket_points_sorted]
    dists = [np.hypot(p[0]-center_x, p[1]-center_y) for p in pocket_points_sorted]
    
    # Group points by angle sectors for smoother transitions
    angle_sectors = np.linspace(-np.pi, np.pi, 8)
    sector_groups = [[] for _ in range(len(angle_sectors)-1)]
    
    for point in pocket_points_sorted:
        angle = np.arctan2(point[1]-center_y, point[0]-center_x)
        for i in range(len(angle_sectors)-1):
            if angle_sectors[i] <= angle < angle_sectors[i+1]:
                sector_groups[i].append(point)
                break
    
    # Build spiral by connecting groups with smooth transitions
    for i in range(len(sector_groups)):
        if i % 2 == 0:
            sector_groups[i].sort(key=lambda p: np.hypot(p[0]-center_x, p[1]-center_y))
        else:
            sector_groups[i].sort(key=lambda p: -np.hypot(p[0]-center_x, p[1]-center_y))
        spiral_path.extend(sector_groups[i])
    
    # Add intermediate points for smoother curves
    if len(spiral_path) > 3:
        x = [p[0] for p in spiral_path]
        y = [p[1] for p in spiral_path]
        z = [p[2] for p in spiral_path]
        
        # Create spline interpolation
        t = np.linspace(0, 1, len(spiral_path))
        t_new = np.linspace(0, 1, len(spiral_path)*3)
        
        try:
            # Use cubic spline for smoother curves
            spl_x = make_interp_spline(t, x, k=3)
            spl_y = make_interp_spline(t, y, k=3)
            spl_z = make_interp_spline(t, z, k=3)
            
            x_smooth = spl_x(t_new)
            y_smooth = spl_y(t_new)
            z_smooth = spl_z(t_new)
            
            smooth_path = list(zip(x_smooth, y_smooth, z_smooth))
            return smooth_path
        except:
            return spiral_path
    
    return spiral_path

def find_closest_point_index(target_point, points_list):
    """
    Find the index of the closest point in points_list to target_point.
    
    Args:
        target_point (list): (x,y,z) point to find closest to
        points_list (list): List of (x,y,z) points to search
        
    Returns:
        int: Index of closest point
    """
    distances = [np.linalg.norm(np.array(target_point) - np.array(p)) for p in points_list]
    return np.argmin(distances)

def create_hybrid_path(optimized_path, grid_points, width, height, depth, shape_type):
    """
    Combine optimized path with spiral pattern in pocket area.
    
    Args:
        optimized_path (list): Optimized toolpath points
        grid_points (list): All grid points
        width (float): Width of the shape
        height (float): Height of the shape
        depth (float): Depth of the shape
        shape_type (str): Type of shape ('H', 'I', 'L', etc.)
        
    Returns:
        list: Hybrid toolpath combining optimized and spiral paths
    """
    # Separate pocket and non-pocket points
    if shape_type == "H":
        pocket_points = [p for p in grid_points if 1.0 < p[0] < width-1.0 and 
                        height*0.37 < p[1] < height*0.63]
    elif shape_type == "I":
        center_x = width / 2
        bar_width = 0.8
        pocket_points = [p for p in grid_points if center_x - bar_width/2 <= p[0] <= center_x + bar_width/2]
    elif shape_type == "L":
        pocket_points = [p for p in grid_points if p[0] < 0.8 or p[1] < 0.8]
    elif shape_type == "X":
        bar_width = 0.5
        pocket_points = []
        for p in grid_points:
            y1 = (height/width) * p[0]
            y2 = height - (height/width) * p[0]
            if abs(p[1] - y1) < bar_width/2 or abs(p[1] - y2) < bar_width/2:
                pocket_points.append(p)
    elif shape_type == "T":
        center_x = width / 2
        bar_width = 0.8
        top_height = 0.3
        pocket_points = [p for p in grid_points if (p[1] > height - top_height) or 
                       (center_x - bar_width/2 <= p[0] <= center_x + bar_width/2)]
    elif shape_type == "Box":
        # Inner box for pocket
        margin = 0.8
        pocket_points = [p for p in grid_points if 
                        margin <= p[0] <= width-margin and 
                        margin <= p[1] <= height-margin]
    elif shape_type == "Triangle":
        # Inner triangle for pocket
        pocket_points = []
        for p in grid_points:
            y_max = height * (1 - p[0]/width) * 0.8  # Smaller triangle
            if p[1] <= y_max:
                pocket_points.append(p)
    elif shape_type == "Cylinder":
        # Circular pocket
        center_x, center_y = width/2, height/2
        radius = min(width, height)/2 * 0.7
        pocket_points = []
        for p in grid_points:
            dist = math.sqrt((p[0]-center_x)**2 + (p[1]-center_y)**2)
            if dist <= radius:
                pocket_points.append(p)
    elif shape_type == "Pentagon":
        # Inner pentagon for pocket
        center_x, center_y = width/2, height/2
        radius = min(width, height)/2 * 0.7
        angles = np.linspace(0, 2*np.pi, 6)[:-1]  # 5 points
        vertices = [(center_x + radius * np.cos(a), center_y + radius * np.sin(a)) for a in angles]
        poly = Polygon(vertices)
        pocket_points = [p for p in grid_points if poly.contains_point((p[0], p[1]))]
    
    non_pocket_points = [p for p in grid_points if p not in pocket_points]
    
    # Get optimized order for non-pocket points
    optimized_non_pocket = [p for p in optimized_path if p in non_pocket_points]
    
    # Create spiral path for pocket
    spiral_pocket = create_spiral_pocket_path(grid_points, width, height, depth, shape_type)
    if spiral_pocket:
        # Find closest grid points to spiral points
        spiral_indices = [find_closest_point_index(sp, grid_points) for sp in spiral_pocket]
        spiral_points = [grid_points[i] for i in spiral_indices]
    else:
        spiral_points = []
    
    # Find best insertion point for spiral in optimized path
    if optimized_non_pocket and spiral_points:
        # Find point in optimized path closest to spiral start
        spiral_start_point = spiral_points[0]
        closest_idx = min(range(len(optimized_non_pocket)),
                        key=lambda i: np.linalg.norm(
                            np.array(optimized_non_pocket[i]) - 
                            np.array(spiral_start_point)))
        
        # Insert spiral after closest point
        hybrid_points = (optimized_non_pocket[:closest_idx+1] + 
                        spiral_points + 
                        optimized_non_pocket[closest_idx+1:])
    else:
        hybrid_points = optimized_path
    
    # Create smooth transitions between segments with more curves
    if len(hybrid_points) > 3:
        x = [p[0] for p in hybrid_points]
        y = [p[1] for p in hybrid_points]
        z = [p[2] for p in hybrid_points]
        
        # Create spline interpolation for smoother path
        t = np.linspace(0, 1, len(hybrid_points))
        
        # Use more points for smoother curves
        t_new = np.linspace(0, 1, len(hybrid_points)*4)
        
        try:
            # Use higher degree spline for smoother curves
            spl_x = make_interp_spline(t, x, k=3)
            spl_y = make_interp_spline(t, y, k=3)
            spl_z = make_interp_spline(t, z, k=3)
            
            x_smooth = spl_x(t_new)
            y_smooth = spl_y(t_new)
            z_smooth = spl_z(t_new)
            
            smooth_path = list(zip(x_smooth, y_smooth, z_smooth))
            
            # Add some noise to make it look more organic
            smooth_path = [
                (x + random.uniform(-0.01, 0.01),
                y + random.uniform(-0.01, 0.01),
                z + random.uniform(-0.005, 0.005)
            ) for x, y, z in smooth_path]
            
            return smooth_path
        except:
            return hybrid_points
    
    return hybrid_points

# =============================================================================
# VISUALIZATION CLASS
# =============================================================================

class ShapeVisualizer:
    """
    Handles 3D visualization of shapes and toolpaths.
    
    Attributes:
        fig (matplotlib.figure.Figure): Main figure object
        ax1, ax2, ax3 (Axes3D): Subplots for different toolpath views
        ani (FuncAnimation): Animation object
        animation_running (bool): Animation state flag
        current_frame (int): Current animation frame
        max_frames (int): Total number of frames
        optimized_path (list): Optimized toolpath points
        zigzag_path (list): Zigzag toolpath points
        hybrid_path (list): Hybrid toolpath points
        drill_size (int): Visual size of drill indicator
    """
    
    def __init__(self, fig):
        """
        Initialize the visualizer with a matplotlib figure.
        
        Args:
            fig (matplotlib.figure.Figure): Figure to use for visualization
        """
        self.fig = fig
        # Create 3 subplots for different toolpath views
        self.ax1 = self.fig.add_subplot(131, projection='3d')  # Optimized
        self.ax2 = self.fig.add_subplot(132, projection='3d')  # Zigzag
        self.ax3 = self.fig.add_subplot(133, projection='3d')  # Hybrid
        self._setup_lighting()
        self.ani = None
        self.animation_running = False
        self.current_frame = 0
        self.max_frames = 0
        self.optimized_path = []
        self.zigzag_path = []
        self.hybrid_path = []
        self.drill_size = 8  # Default drill size
    
    def _setup_lighting(self):
        """Configure lighting and color scheme for 3D visualization."""
        self.light = LightSource(azdeg=225, altdeg=45)
        self.face_colors = {
            'base': '#4682B4', 'front': '#5F9EA0', 'back': '#B0C4DE',
            'right': '#87CEEB', 'left': '#ADD8E6', 'pocket': '#D3D3D3',
            'top': '#98FB98', 'grid': '#FFFFFF'
        }
    
    def _create_letter_H(self, ax, width, height, depth):
        """
        Create 3D geometry for letter 'H'.
        
        Args:
            ax (Axes3D): Matplotlib 3D axis to draw on
            width (float): Width of the letter
            height (float): Height of the letter
            depth (float): Depth of the letter
        """
        # Left vertical bar vertices
        left_width = 0.8
        bar_thickness = 0.8
        crossbar_height = height * 0.37
        
        left_front = np.array([
            [0, 0, 0], [0, height, 0], [left_width, height, 0], [left_width, 0, 0],
            [0, 0, depth], [0, height, depth], [left_width, height, depth], [left_width, 0, depth]
        ])
        
        # Right vertical bar vertices
        right_front = np.array([
            [width-left_width, 0, 0], [width-left_width, height, 0], [width, height, 0], [width, 0, 0],
            [width-left_width, 0, depth], [width-left_width, height, depth], [width, height, depth], [width, 0, depth]
        ])
        
        # Horizontal bar vertices
        horizontal = np.array([
            [left_width, crossbar_height, 0], 
            [left_width, height-crossbar_height, 0], 
            [width-left_width, height-crossbar_height, 0], 
            [width-left_width, crossbar_height, 0],
            [left_width, crossbar_height, depth], 
            [left_width, height-crossbar_height, depth], 
            [width-left_width, height-crossbar_height, depth], 
            [width-left_width, crossbar_height, depth]
        ])
        
        self._add_shape_parts(ax, [left_front, right_front, horizontal], [1.0, 1.0, 0.9])
    
    def _create_letter_I(self, ax, width, height, depth):
        """
        Create 3D geometry for letter 'I'.
        
        Args:
            ax (Axes3D): Matplotlib 3D axis to draw on
            width (float): Width of the letter
            height (float): Height of the letter
            depth (float): Depth of the letter
        """
        # Simple vertical bar with small horizontal caps
        center_x = width / 2
        bar_width = 0.8
        cap_height = 0.3
        
        # Main vertical bar
        vertical = np.array([
            [center_x - bar_width/2, cap_height, 0],
            [center_x - bar_width/2, height-cap_height, 0],
            [center_x + bar_width/2, height-cap_height, 0],
            [center_x + bar_width/2, cap_height, 0],
            [center_x - bar_width/2, cap_height, depth],
            [center_x - bar_width/2, height-cap_height, depth],
            [center_x + bar_width/2, height-cap_height, depth],
            [center_x + bar_width/2, cap_height, depth]
        ])
        
        # Top cap
        top_cap = np.array([
            [center_x - bar_width, height-cap_height, 0],
            [center_x - bar_width, height, 0],
            [center_x + bar_width, height, 0],
            [center_x + bar_width, height-cap_height, 0],
            [center_x - bar_width, height-cap_height, depth],
            [center_x - bar_width, height, depth],
            [center_x + bar_width, height, depth],
            [center_x + bar_width, height-cap_height, depth]
        ])
        
        # Bottom cap
        bottom_cap = np.array([
            [center_x - bar_width, 0, 0],
            [center_x - bar_width, cap_height, 0],
            [center_x + bar_width, cap_height, 0],
            [center_x + bar_width, 0, 0],
            [center_x - bar_width, 0, depth],
            [center_x - bar_width, cap_height, depth],
            [center_x + bar_width, cap_height, depth],
            [center_x + bar_width, 0, depth]
        ])
        
        self._add_shape_parts(ax, [vertical, top_cap, bottom_cap], [1.0, 0.9, 0.9])
    
    def _create_letter_L(self, ax, width, height, depth):
        """
        Create 3D geometry for letter 'L'.
        
        Args:
            ax (Axes3D): Matplotlib 3D axis to draw on
            width (float): Width of the letter
            height (float): Height of the letter
            depth (float): Depth of the letter
        """
        # Vertical bar with horizontal base - Updated design
        vertical = np.array([
            [0, 0, 0], [0, height, 0], [0.8, height, 0], [0.8, 0, 0],
            [0, 0, depth], [0, height, depth], [0.8, height, depth], [0.8, 0, depth]
        ])
        
        horizontal = np.array([
            [0, 0, 0], [0, 0.8, 0], [width, 0.8, 0], [width, 0, 0],
            [0, 0, depth], [0, 0.8, depth], [width, 0.8, depth], [width, 0, depth]
        ])
        
        self._add_shape_parts(ax, [vertical, horizontal], [1.0, 0.9])
    
    def _create_letter_X(self, ax, width, height, depth):
        """
        Create 3D geometry for letter 'X'.
        
        Args:
            ax (Axes3D): Matplotlib 3D axis to draw on
            width (float): Width of the letter
            height (float): Height of the letter
            depth (float): Depth of the letter
        """
        # Two diagonal bars crossing each other
        bar_width = 0.5
        
        # Diagonal from top-left to bottom-right
        diag1 = np.array([
            [0, height, 0],
            [bar_width, height, 0],
            [width, bar_width, 0],
            [width-bar_width, 0, 0],
            [0, height, depth],
            [bar_width, height, depth],
            [width, bar_width, depth],
            [width-bar_width, 0, depth]
        ])
        
        # Diagonal from bottom-left to top-right
        diag2 = np.array([
            [0, bar_width, 0],
            [0, 0, 0],
            [width-bar_width, height, 0],
            [width, height, 0],
            [0, bar_width, depth],
            [0, 0, depth],
            [width-bar_width, height, depth],
            [width, height, depth]
        ])
        
        self._add_shape_parts(ax, [diag1, diag2], [1.0, 0.9])
    
    def _create_letter_T(self, ax, width, height, depth):
        """
        Create 3D geometry for letter 'T'.
        
        Args:
            ax (Axes3D): Matplotlib 3D axis to draw on
            width (float): Width of the letter
            height (float): Height of the letter
            depth (float): Depth of the letter
        """
        # Horizontal top with vertical center
        bar_width = 0.8
        center_x = width / 2
        top_height = 0.3
        
        # Horizontal top
        horizontal = np.array([
            [0, height-top_height, 0],
            [0, height, 0],
            [width, height, 0],
            [width, height-top_height, 0],
            [0, height-top_height, depth],
            [0, height, depth],
            [width, height, depth],
            [width, height-top_height, depth]
        ])
        
        # Vertical center
        vertical = np.array([
            [center_x - bar_width/2, 0, 0],
            [center_x - bar_width/2, height, 0],
            [center_x + bar_width/2, height, 0],
            [center_x + bar_width/2, 0, 0],
            [center_x - bar_width/2, 0, depth],
            [center_x - bar_width/2, height, depth],
            [center_x + bar_width/2, height, depth],
            [center_x + bar_width/2, 0, depth]
        ])
        
        self._add_shape_parts(ax, [horizontal, vertical], [1.0, 0.9])
    
    def _create_box(self, ax, width, height, depth):
        """
        Create 3D geometry for a box shape.
        
        Args:
            ax (Axes3D): Matplotlib 3D axis to draw on
            width (float): Width of the box
            height (float): Height of the box
            depth (float): Depth of the box
        """
        # Simple box shape
        box = np.array([
            [0, 0, 0], [0, height, 0], [width, height, 0], [width, 0, 0],
            [0, 0, depth], [0, height, depth], [width, height, depth], [width, 0, depth]
        ])
        
        self._add_shape_parts(ax, [box], [1.0])
    
    def _create_triangle(self, ax, width, height, depth):
        """
        Create 3D geometry for a right-angle triangle.
        
        Args:
            ax (Axes3D): Matplotlib 3D axis to draw on
            width (float): Width of the triangle
            height (float): Height of the triangle
            depth (float): Depth of the triangle
        """
        # Right-angle triangle
        triangle = np.array([
            [0, 0, 0], [0, height, 0], [width, 0, 0],
            [0, 0, depth], [0, height, depth], [width, 0, depth]
        ])
        
        self._add_shape_parts(ax, [triangle], [1.0])
    
    def _create_cylinder(self, ax, width, height, depth):
        """
        Create 3D geometry for a cylinder.
        
        Args:
            ax (Axes3D): Matplotlib 3D axis to draw on
            width (float): Width of the cylinder
            height (float): Height of the cylinder
            depth (float): Depth of the cylinder
        """
        # Cylinder approximation with polygons
        center_x, center_y = width/2, height/2
        radius = min(width, height)/2 * 0.9
        n_segments = 32
        
        # Create vertices for top and bottom circles
        angles = np.linspace(0, 2*np.pi, n_segments, endpoint=False)
        bottom_circle = np.array([[center_x + radius * np.cos(a), 
                                 center_y + radius * np.sin(a), 
                                 0] for a in angles])
        top_circle = np.array([[center_x + radius * np.cos(a), 
                              center_y + radius * np.sin(a), 
                              depth] for a in angles])
        
        # Create side faces
        faces = []
        for i in range(n_segments):
            next_i = (i + 1) % n_segments
            faces.append([bottom_circle[i], bottom_circle[next_i], top_circle[next_i], top_circle[i]])
        
        # Create top and bottom faces
        bottom_face = bottom_circle.tolist()
        top_face = top_circle.tolist()
        
        # Add all faces to the plot
        poly = Poly3DCollection(
            faces + [bottom_face, top_face],
            facecolors='#4682B4',
            linewidths=1,
            edgecolors='k',
            zorder=1,
            alpha=0.7
        )
        ax.add_collection3d(poly)
    
    def _create_pentagon(self, ax, width, height, depth):
        """
        Create 3D geometry for a pentagon.
        
        Args:
            ax (Axes3D): Matplotlib 3D axis to draw on
            width (float): Width of the pentagon
            height (float): Height of the pentagon
            depth (float): Depth of the pentagon
        """
        # Pentagon shape
        center_x, center_y = width/2, height/2
        radius = min(width, height)/2 * 0.9
        angles = np.linspace(0, 2*np.pi, 6)[:-1]  # 5 points
        
        # Create vertices for top and bottom pentagons
        bottom_pent = np.array([[center_x + radius * np.cos(a), 
                               center_y + radius * np.sin(a), 
                               0] for a in angles])
        top_pent = np.array([[center_x + radius * np.cos(a), 
                            center_y + radius * np.sin(a), 
                            depth] for a in angles])
        
        # Create side faces
        faces = []
        for i in range(5):
            next_i = (i + 1) % 5
            faces.append([bottom_pent[i], bottom_pent[next_i], top_pent[next_i], top_pent[i]])
        
        # Create top and bottom faces
        bottom_face = bottom_pent.tolist()
        top_face = top_pent.tolist()
        
        # Add all faces to the plot
        poly = Poly3DCollection(
            faces + [bottom_face, top_face],
            facecolors='#4682B4',
            linewidths=1,
            edgecolors='k',
            zorder=1,
            alpha=0.7
        )
        ax.add_collection3d(poly)
    
    def _add_shape_parts(self, ax, parts, color_factors):
        """
        Add shape parts to the plot with proper shading.
        
        Args:
            ax (Axes3D): Matplotlib 3D axis to draw on
            parts (list): List of part vertices
            color_factors (list): List of color adjustment factors
        """
        for vertices, color_factor in zip(parts, color_factors):
            faces = self._get_faces(vertices)
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
    
    def _get_faces(self, vertices):
        """
        Get faces for different types of 3D shapes.
        
        Args:
            vertices (list): List of vertices defining the shape
            
        Returns:
            list: List of faces (each face is a list of vertices)
        """
        if len(vertices) == 8:  # Box-like shape
            return [
                [vertices[0], vertices[1], vertices[2], vertices[3]],  # front
                [vertices[4], vertices[5], vertices[6], vertices[7]],  # back
                [vertices[0], vertices[1], vertices[5], vertices[4]],  # top
                [vertices[2], vertices[3], vertices[7], vertices[6]],  # bottom
                [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
                [vertices[0], vertices[3], vertices[7], vertices[4]]   # left
            ]
        elif len(vertices) == 6:  # Triangle
            return [
                [vertices[0], vertices[1], vertices[2]],  # front
                [vertices[3], vertices[4], vertices[5]],  # back
                [vertices[0], vertices[1], vertices[4], vertices[3]],  # top
                [vertices[1], vertices[2], vertices[5], vertices[4]],  # right
                [vertices[0], vertices[2], vertices[5], vertices[3]]   # left
            ]
    
    def setup_scenes(self, width, height, depth, grid_points, shape_type):
        """
        Set up the 3D visualization scenes for all toolpath types.
        
        Args:
            width (float): Width of the shape
            height (float): Height of the shape
            depth (float): Depth of the shape
            grid_points (list): List of grid points
            shape_type (str): Type of shape ('H', 'I', 'L', etc.)
        """
        # Clear existing plots and recreate axes
        self.fig.clf()
        self.ax1 = self.fig.add_subplot(131, projection='3d')
        self.ax2 = self.fig.add_subplot(132, projection='3d')
        self.ax3 = self.fig.add_subplot(133, projection='3d')
        
        # Create the shape geometry based on type
        if shape_type == "H":
            self._create_letter_H(self.ax1, width, height, depth)
            self._create_letter_H(self.ax2, width, height, depth)
            self._create_letter_H(self.ax3, width, height, depth)
        elif shape_type == "I":
            self._create_letter_I(self.ax1, width, height, depth)
            self._create_letter_I(self.ax2, width, height, depth)
            self._create_letter_I(self.ax3, width, height, depth)
        elif shape_type == "L":
            self._create_letter_L(self.ax1, width, height, depth)
            self._create_letter_L(self.ax2, width, height, depth)
            self._create_letter_L(self.ax3, width, height, depth)
        elif shape_type == "X":
            self._create_letter_X(self.ax1, width, height, depth)
            self._create_letter_X(self.ax2, width, height, depth)
            self._create_letter_X(self.ax3, width, height, depth)
        elif shape_type == "T":
            self._create_letter_T(self.ax1, width, height, depth)
            self._create_letter_T(self.ax2, width, height, depth)
            self._create_letter_T(self.ax3, width, height, depth)
        elif shape_type == "Box":
            self._create_box(self.ax1, width, height, depth)
            self._create_box(self.ax2, width, height, depth)
            self._create_box(self.ax3, width, height, depth)
        elif shape_type == "Triangle":
            self._create_triangle(self.ax1, width, height, depth)
            self._create_triangle(self.ax2, width, height, depth)
            self._create_triangle(self.ax3, width, height, depth)
        elif shape_type == "Cylinder":
            self._create_cylinder(self.ax1, width, height, depth)
            self._create_cylinder(self.ax2, width, height, depth)
            self._create_cylinder(self.ax3, width, height, depth)
        elif shape_type == "Pentagon":
            self._create_pentagon(self.ax1, width, height, depth)
            self._create_pentagon(self.ax2, width, height, depth)
            self._create_pentagon(self.ax3, width, height, depth)
        
        # Add grid points
        xs, ys, zs = zip(*grid_points)
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.scatter(
                xs, ys, zs, 
                color=self.face_colors['grid'], 
                s=40, 
                edgecolor='k',
                linewidth=1,
                zorder=5
            )
        
        self.ax1.set_title('Optimized Toolpath', fontsize=12, pad=15)
        self.ax2.set_title('Zigzag Toolpath', fontsize=12, pad=15)
        self.ax3.set_title('Hybrid Toolpath', fontsize=12, pad=15)
        self.fig.suptitle(f'{shape_type} Shape Toolpath Comparison', fontsize=16, y=0.95)
        
        # Configure view
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.view_init(elev=30, azim=-50)
            ax.set_xlim(0, width)
            ax.set_ylim(0, height)
            ax.set_zlim(0, depth*1.1)
            ax.set_xlabel('X Axis', fontsize=10)
            ax.set_ylabel('Y Axis', fontsize=10)
            ax.set_zlabel('Z Axis', fontsize=10)
            ax.grid(False)
            ax.set_facecolor('white')
    
    def animate_paths(self, optimized_path, zigzag_path, hybrid_path):
        """
        Animate the toolpaths on the 3D visualization.
        
        Args:
            optimized_path (list): Optimized toolpath points
            zigzag_path (list): Zigzag toolpath points
            hybrid_path (list): Hybrid toolpath points
        """
        # Store paths for later use in controls
        self.optimized_path = optimized_path
        self.zigzag_path = zigzag_path
        self.hybrid_path = hybrid_path
        self.current_frame = 0
        self.max_frames = max(len(optimized_path), len(zigzag_path), len(hybrid_path))
        
        # Initialize all paths with better colors and styles
        self.opt_line, = self.ax1.plot([], [], [], 'r-', linewidth=3, zorder=6, alpha=0.8)
        self.opt_drill, = self.ax1.plot([], [], [], 'ko', markersize=self.drill_size, zorder=10)
        
        self.zz_line, = self.ax2.plot([], [], [], 'b-', linewidth=3, zorder=6, alpha=0.8)
        self.zz_drill, = self.ax2.plot([], [], [], 'ko', markersize=self.drill_size, zorder=10)
        
        # Use yellow color for hybrid path with higher contrast
        self.hybrid_line, = self.ax3.plot([], [], [], 'y-', linewidth=4, zorder=6, alpha=0.9)
        self.hybrid_drill, = self.ax3.plot([], [], [], 'ko', markersize=self.drill_size, zorder=10)
        
        # Store path coordinates
        self.opt_x, self.opt_y, self.opt_z = [], [], []
        self.zz_x, self.zz_y, self.zz_z = [], [], []
        self.hybrid_x, self.hybrid_y, self.hybrid_z = [], [], []
        
        # Create animation
        if self.ani:
            self.ani.event_source.stop()
            
        self.ani = FuncAnimation(
            self.fig, 
            self._update_animation,
            frames=self.max_frames,
            interval=50,
            blit=True,
            init_func=lambda: [self.opt_line, self.opt_drill, self.zz_line, 
                             self.zz_drill, self.hybrid_line, self.hybrid_drill]
        )
        self.animation_running = True

    def _update_animation(self, frame):
        """
        Update function for the animation.
        
        Args:
            frame (int): Current animation frame
            
        Returns:
            list: List of artists to be updated
        """
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
        
        # Update zigzag path
        if frame < len(self.zigzag_path):
            x, y, z = self.zigzag_path[frame]
            self.zz_drill.set_data([x], [y])
            self.zz_drill.set_3d_properties([z])
            self.zz_x.append(x)
            self.zz_y.append(y)
            self.zz_z.append(z)
            self.zz_line.set_data(self.zz_x, self.zz_y)
            self.zz_line.set_3d_properties(self.zz_z)
        
        # Update hybrid path
        if frame < len(self.hybrid_path):
            x, y, z = self.hybrid_path[frame]
            self.hybrid_drill.set_data([x], [y])
            self.hybrid_drill.set_3d_properties([z])
            self.hybrid_x.append(x)
            self.hybrid_y.append(y)
            self.hybrid_z.append(z)
            self.hybrid_line.set_data(self.hybrid_x, self.hybrid_y)
            self.hybrid_line.set_3d_properties(self.hybrid_z)
        
        self.current_frame = frame
        return self.opt_line, self.opt_drill, self.zz_line, self.zz_drill, self.hybrid_line, self.hybrid_drill
    
    def set_drill_size(self, size):
        """
        Update the drill size for all paths.
        
        Args:
            size (int): New drill size in points
        """
        self.drill_size = size
        if hasattr(self, 'opt_drill'):
            self.opt_drill.set_markersize(size)
            self.zz_drill.set_markersize(size)
            self.hybrid_drill.set_markersize(size)
            self.fig.canvas.draw_idle()
    
    def toggle_animation(self):
        """Toggle animation play/pause state."""
        if self.ani:
            if self.animation_running:
                self.ani.event_source.stop()
                self.animation_running = False
            else:
                self.ani.event_source.start()
                self.animation_running = True
    
    def step_animation(self, step):
        """
        Step through animation frames.
        
        Args:
            step (int): Number of frames to step (can be negative)
        """
        if not self.ani:
            return
            
        new_frame = self.current_frame + step
        if 0 <= new_frame < self.max_frames:
            self._update_animation(new_frame)
            self.fig.canvas.draw_idle()
    
    def reset_animation(self):
        """Reset animation to the first frame."""
        if not self.ani:
            return
            
        self.opt_x, self.opt_y, self.opt_z = [], [], []
        self.zz_x, self.zz_y, self.zz_z = [], [], []
        self.hybrid_x, self.hybrid_y, self.hybrid_z = [], [], []
        self.current_frame = 0
        self._update_animation(0)
        self.fig.canvas.draw_idle()
    
    def save_animation(self, filename):
        """
        Save animation to a file.
        
        Args:
            filename (str): Path to save the animation
        """
        if not self.ani:
            return
            
        writer = PillowWriter(fps=15)
        self.ani.save(filename, writer=writer)

# =============================================================================
# GRID POINT GENERATION
# =============================================================================

def create_grid_points(width, height, depth, margin, grid_size, shape_type):
    """
    Create grid points for a specific shape type.
    
    Args:
        width (float): Width of the shape
        height (float): Height of the shape
        depth (float): Depth of the shape
        margin (float): Margin around the shape
        grid_size (int): Number of grid points in each dimension
        shape_type (str): Type of shape ('H', 'I', 'L', etc.)
        
    Returns:
        tuple: (grid_points, x_vals, y_vals)
            grid_points: List of (x,y,z) grid points
            x_vals: X-coordinates of grid points
            y_vals: Y-coordinates of grid points
    """
    x_vals = np.linspace(margin, width-margin, grid_size)
    y_vals = np.linspace(margin, height-margin, grid_size)
    
    if shape_type == "H":
        grid_points = [[x, y, depth] for x in x_vals for y in y_vals 
                      if (x < 0.8 or x > width-0.8 or (y > height*0.37 and y < height*0.63))]
    elif shape_type == "I":
        center_x = width / 2
        bar_width = 0.8
        grid_points = [[x, y, depth] for x in x_vals for y in y_vals 
                      if (center_x - bar_width/2 <= x <= center_x + bar_width/2)]
    elif shape_type == "L":
        # Updated grid points for L shape
        grid_points = []
        for x in x_vals:
            for y in y_vals:
                if (x < 0.8 or y < 0.8):  # L shape condition
                    grid_points.append([x, y, depth])
    elif shape_type == "X":
        bar_width = 0.5
        # Points along the diagonals
        grid_points = []
        for x in x_vals:
            y1 = (height/width) * x
            y2 = height - (height/width) * x
            for y in y_vals:
                if (abs(y - y1) < bar_width/2 or abs(y - y2) < bar_width/2):
                    grid_points.append([x, y, depth])
    elif shape_type == "T":
        center_x = width / 2
        bar_width = 0.8
        top_height = 0.3
        grid_points = [[x, y, depth] for x in x_vals for y in y_vals 
                      if (y > height - top_height) or 
                      (center_x - bar_width/2 <= x <= center_x + bar_width/2)]
    elif shape_type == "Box":
        # Simple box shape - all points
        grid_points = [[x, y, depth] for x in x_vals for y in y_vals]
    elif shape_type == "Triangle":
        # Right-angle triangle points
        grid_points = []
        for x in x_vals:
            y_max = height * (1 - x/width)  # Right-angle triangle
            for y in y_vals:
                if y <= y_max:
                    grid_points.append([x, y, depth])
    elif shape_type == "Cylinder":
        # Circular points for cylinder
        center_x, center_y = width/2, height/2
        radius = min(width, height)/2 * 0.9
        grid_points = []
        for x in x_vals:
            for y in y_vals:
                dist = math.sqrt((x-center_x)**2 + (y-center_y)**2)
                if dist <= radius:
                    grid_points.append([x, y, depth])
    elif shape_type == "Pentagon":
        # Pentagon shape points
        center_x, center_y = width/2, height/2
        radius = min(width, height)/2 * 0.9
        angles = np.linspace(0, 2*np.pi, 6)[:-1]  # 5 points
        vertices = [(center_x + radius * np.cos(a), center_y + radius * np.sin(a)) for a in angles]
        poly = Polygon(vertices)
        grid_points = []
        for x in x_vals:
            for y in y_vals:
                if poly.contains_point((x, y)):
                    grid_points.append([x, y, depth])
    
    return grid_points, x_vals, y_vals

# =============================================================================
# MAIN APPLICATION CLASS
# =============================================================================

class ToolpathDashboard:
    """
    Main application class for the Toolpath Optimization Dashboard.
    
    Attributes:
        root (tk.Tk): Main application window
        params (dict): Dictionary of parameter variables
        fig (matplotlib.figure.Figure): Main visualization figure
        visualizer (ShapeVisualizer): Toolpath visualizer instance
        optimized_path (list): Optimized toolpath points
        zigzag_path (list): Zigzag toolpath points
        hybrid_path (list): Hybrid toolpath points
        grid_points (list): Grid points for toolpath generation
    """
    
    def __init__(self, root):
        """
        Initialize the dashboard application.
        
        Args:
            root (tk.Tk): Root window
        """
        self.root = root
        self.root.title("Advanced Toolpath Optimization Dashboard")
        self.root.geometry("1400x900")
        self.root.configure(bg=BG_COLOR)
        
        # Set window icon
        try:
            self.root.iconbitmap("icon.ico")
        except:
            pass
            
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the user interface components."""
        # Create header frame
        header_frame = tk.Frame(self.root, bg=DARK_BG)
        header_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Add logo and title
        logo_label = tk.Label(header_frame, text="⚙️", font=("Arial", 24), bg=DARK_BG, fg="white")
        logo_label.pack(side=tk.LEFT, padx=10)
        
        title_label = tk.Label(header_frame, text="Toolpath Optimization Dashboard", 
                             font=("Arial", 16, "bold"), bg=DARK_BG, fg="white")
        title_label.pack(side=tk.LEFT, padx=10)
        
        # Add help button
        help_button = tk.Button(header_frame, text="Help", command=self.show_help, 
                              bg=HIGHLIGHT_COLOR, fg="white", relief=tk.FLAT)
        help_button.pack(side=tk.RIGHT, padx=10)
        
        # Create main frames
        main_frame = tk.Frame(self.root, bg=BG_COLOR)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        control_frame = ttk.LabelFrame(main_frame, text="Parameters", padding=10, style="Custom.TLabelframe")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        visualization_frame = tk.Frame(main_frame, bg=BG_COLOR)
        visualization_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=5, pady=5)
        
        # Configure styles
        self.configure_styles()
        
        # Default values
        self.params = {
            'shape_type': tk.StringVar(value="H"),
            'width': tk.DoubleVar(value=3.0),
            'height': tk.DoubleVar(value=3.0),
            'depth': tk.DoubleVar(value=1.5),
            'margin': tk.DoubleVar(value=0.1),
            'grid_size': tk.IntVar(value=8),
            'pop_size': tk.IntVar(value=70),
            'generations': tk.IntVar(value=150),
            'mutation_rate': tk.DoubleVar(value=0.5),
            'elitism': tk.IntVar(value=3),
            'tournament_size': tk.IntVar(value=5),
            'animation_speed': tk.IntVar(value=50),
            'use_hybrid': tk.BooleanVar(value=True),
            'drill_size': tk.StringVar(value="Medium")
        }
        
        # Create notebook for organized controls
        notebook = ttk.Notebook(control_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Design tab
        design_tab = ttk.Frame(notebook)
        notebook.add(design_tab, text="Design")
        
        # Shape type selection
        ttk.Label(design_tab, text="Shape Type").grid(row=0, column=0, sticky="w", padx=(0, 5), pady=(0, 5))
        shape_menu = ttk.Combobox(design_tab, textvariable=self.params['shape_type'], 
                                 values=["H", "I", "L", "X", "T", "Box", "Triangle", "Cylinder", "Pentagon"], state="readonly")
        shape_menu.grid(row=0, column=1, columnspan=2, sticky="ew", pady=(0, 10))
        shape_menu.bind("<<ComboboxSelected>>", self.update_dimensions)
        
        # Create parameter controls
        ttk.Label(design_tab, text="Dimensions", font=('Arial', 10, 'bold')).grid(
            row=1, column=0, columnspan=3, pady=(0, 5), sticky="w")
        self.create_slider(design_tab, "Width (m)", 'width', 1.0, 5.0, 2)
        self.create_slider(design_tab, "Height (m)", 'height', 1.0, 5.0, 3)
        self.create_slider(design_tab, "Depth (m)", 'depth', 0.5, 3.0, 4)
        
        ttk.Label(design_tab, text="Grid Settings", font=('Arial', 10, 'bold')).grid(
            row=5, column=0, columnspan=3, pady=(10, 5), sticky="w")
        self.create_slider(design_tab, "Margin (m)", 'margin', 0.05, 0.3, 6)
        self.create_slider(design_tab, "Grid Size", 'grid_size', 4, 20, 7)
        
        # GA tab
        ga_tab = ttk.Frame(notebook)
        notebook.add(ga_tab, text="GA Parameters")
        
        ttk.Label(ga_tab, text="Genetic Algorithm", font=('Arial', 10, 'bold')).grid(
            row=0, column=0, columnspan=3, pady=(0, 5), sticky="w")
        self.create_slider(ga_tab, "Population Size", 'pop_size', 20, 200, 1)
        self.create_slider(ga_tab, "Generations", 'generations', 50, 300, 2)
        self.create_slider(ga_tab, "Mutation Rate", 'mutation_rate', 0.1, 0.9, 3)
        self.create_slider(ga_tab, "Elitism", 'elitism', 1, 10, 4)
        self.create_slider(ga_tab, "Tournament Size", 'tournament_size', 3, 10, 5)
        
        # Options tab
        options_tab = ttk.Frame(notebook)
        notebook.add(options_tab, text="Options")
        
        ttk.Label(options_tab, text="Path Options", font=('Arial', 10, 'bold')).grid(
            row=0, column=0, columnspan=3, pady=(0, 5), sticky="w")
        ttk.Checkbutton(options_tab, text="Use Hybrid Path", variable=self.params['use_hybrid'], 
                       style="Custom.TCheckbutton").grid(row=1, column=0, columnspan=3, pady=(0, 10), sticky="w")
        
        # Drill size selection
        ttk.Label(options_tab, text="Drill Size", font=('Arial', 10, 'bold')).grid(
            row=2, column=0, sticky="w", padx=(0, 5))
        drill_menu = ttk.Combobox(options_tab, textvariable=self.params['drill_size'], 
                                 values=["Small", "Medium", "Large"], state="readonly")
        drill_menu.grid(row=2, column=1, columnspan=2, sticky="ew", pady=(0, 5))
        drill_menu.bind("<<ComboboxSelected>>", lambda e: self.update_drill_size())
        
        ttk.Label(options_tab, text="Animation Speed", font=('Arial', 10, 'bold')).grid(
            row=3, column=0, columnspan=3, pady=(5, 5), sticky="w")
        ttk.Scale(options_tab, from_=10, to=200, variable=self.params['animation_speed'],
                 command=lambda v: self.update_animation_speed(), style="Custom.Horizontal.TScale").grid(
                 row=4, column=0, columnspan=2, sticky="ew", padx=(0, 5))
        ttk.Label(options_tab, textvariable=self.params['animation_speed'], style="Custom.TLabel").grid(
            row=4, column=2, sticky="w")
        
        # Action buttons frame
        action_frame = ttk.Frame(control_frame)
        action_frame.pack(fill=tk.X, pady=(10, 5))
        
        ttk.Button(action_frame, text="Run Optimization", command=self.run_optimization, 
                  style="Accent.TButton").pack(fill=tk.X, pady=5)
        
        # Simulation controls frame
        sim_frame = ttk.LabelFrame(control_frame, text="Simulation Controls", padding=5, style="Custom.TLabelframe")
        sim_frame.pack(fill=tk.X, pady=5)
        
        btn_frame1 = ttk.Frame(sim_frame)
        btn_frame1.pack(fill=tk.X)
        ttk.Button(btn_frame1, text="Play/Pause", command=self.toggle_animation).pack(side=tk.LEFT, expand=True, padx=2)
        ttk.Button(btn_frame1, text="Step Forward", command=lambda: self.step_animation(1)).pack(side=tk.LEFT, expand=True, padx=2)
        
        btn_frame2 = ttk.Frame(sim_frame)
        btn_frame2.pack(fill=tk.X, pady=(5, 0))
        ttk.Button(btn_frame2, text="Step Back", command=lambda: self.step_animation(-1)).pack(side=tk.LEFT, expand=True, padx=2)
        ttk.Button(btn_frame2, text="Reset", command=self.reset_animation).pack(side=tk.LEFT, expand=True, padx=2)
        
        # Export frame
        export_frame = ttk.LabelFrame(control_frame, text="Export", padding=5, style="Custom.TLabelframe")
        export_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(export_frame, text="Save Path Data", command=self.export_path_data).pack(fill=tk.X, pady=2)
        ttk.Button(export_frame, text="Save Image", command=self.export_image).pack(fill=tk.X, pady=2)
        ttk.Button(export_frame, text="Save Animation", command=self.export_animation).pack(fill=tk.X, pady=2)
        
        # Bottom buttons
        bottom_frame = ttk.Frame(control_frame)
        bottom_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(bottom_frame, text="Reset Defaults", command=self.reset_defaults).pack(side=tk.LEFT, expand=True, padx=2)
        ttk.Button(bottom_frame, text="Exit", command=self.root.quit, style="Danger.TButton").pack(side=tk.LEFT, expand=True, padx=2)
        
        # Visualization area
        self.fig = plt.figure(figsize=(14, 6), facecolor='white')
        self.canvas = FigureCanvasTkAgg(self.fig, master=visualization_frame)
        self.canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH, padx=5, pady=5)
        
        # Add toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, visualization_frame)
        self.toolbar.update()
        self.canvas._tkcanvas.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, 
                             anchor=tk.W, style="Status.TLabel")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        
        # Initialize visualizer
        self.visualizer = ShapeVisualizer(self.fig)
        
        # Store paths for export
        self.optimized_path = []
        self.zigzag_path = []
        self.hybrid_path = []
        self.grid_points = []
    
    def configure_styles(self):
        """Configure the visual styles for the UI components."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('.', background=BG_COLOR, foreground=TEXT_COLOR)
        style.configure('TFrame', background=BG_COLOR)
        style.configure('TLabel', background=BG_COLOR, foreground=TEXT_COLOR)
        style.configure('TButton', background=BUTTON_COLOR, foreground='white', 
                       font=('Arial', 10), borderwidth=1)
        style.map('TButton', 
                 background=[('active', HIGHLIGHT_COLOR), ('pressed', DARK_BG)],
                 foreground=[('active', 'white'), ('pressed', 'white')])
        
        style.configure('Accent.TButton', background=ACCENT_COLOR, foreground='white',
                       font=('Arial', 10, 'bold'))
        style.map('Accent.TButton', 
                 background=[('active', HIGHLIGHT_COLOR), ('pressed', DARK_BG)],
                 foreground=[('active', 'white'), ('pressed', 'white')])
        
        style.configure('Danger.TButton', background='#e74c3c', foreground='white',
                       font=('Arial', 10))
        style.map('Danger.TButton', 
                 background=[('active', '#c0392b'), ('pressed', '#7f2318')],
                 foreground=[('active', 'white'), ('pressed', 'white')])
        
        style.configure('Custom.TLabelframe', background=BG_COLOR, 
                       bordercolor=DARK_BG, relief=tk.RAISED)
        style.configure('Custom.TLabelframe.Label', background=BG_COLOR, 
                       foreground=TEXT_COLOR, font=('Arial', 10, 'bold'))
        
        style.configure('Custom.TCheckbutton', background=BG_COLOR)
        style.configure('Custom.Horizontal.TScale', background=BG_COLOR)
        style.configure('Status.TLabel', background=DARK_BG, foreground='white')
        
        style.configure('TEntry', fieldbackground='white')
        style.configure('TCombobox', fieldbackground='white')
    
    def create_slider(self, parent, label, param, min_val, max_val, row):
        """
        Create a labeled slider control.
        
        Args:
            parent: Parent widget
            label (str): Label text
            param (str): Parameter name
            min_val: Minimum value
            max_val: Maximum value
            row (int): Grid row position
        """
        ttk.Label(parent, text=label, style="Custom.TLabel").grid(row=row, column=0, sticky="w", padx=(0, 5), pady=2)
        
        if isinstance(min_val, int) and isinstance(max_val, int):
            slider = ttk.Scale(parent, from_=min_val, to=max_val, variable=self.params[param],
                              command=lambda v: self.params[param].set(round(float(v))),
                              style="Custom.Horizontal.TScale")
            entry = ttk.Entry(parent, textvariable=self.params[param], width=5)
            self.params[param].trace_add('write', 
                lambda *args: slider.set(self.params[param].get()))
        else:
            slider = ttk.Scale(parent, from_=min_val, to=max_val, variable=self.params[param],
                             style="Custom.Horizontal.TScale")
            entry = ttk.Entry(parent, textvariable=self.params[param], width=5)
        
        slider.grid(row=row, column=1, sticky="ew", padx=(0, 5), pady=2)
        entry.grid(row=row, column=2, sticky="w", pady=2)
    
    def show_help(self):
        """Display help information in a message box."""
        help_text = """Toolpath Optimization Dashboard Help:

1. Design Tab:
   - Select shape type (Box, letters, or geometric shapes)
   - Adjust dimensions and grid settings

2. GA Parameters Tab:
   - Configure genetic algorithm settings
   - Population size and generations affect optimization quality
   - Mutation rate controls exploration vs exploitation

3. Options Tab:
   - Toggle hybrid path (combines optimized and spiral paths)
   - Adjust drill size visualization
   - Control animation speed

4. Simulation Controls:
   - Play/pause animation
   - Step through animation frame by frame
   - Reset animation

5. Export:
   - Save path data as CSV
   - Save visualization as image
   - Save animation as GIF

."""
        
        messagebox.showinfo("Help", help_text)
    
    def update_dimensions(self, event=None):
        """
        Update default dimensions when shape type changes.
        
        Args:
            event: Optional event parameter
        """
        shape_type = self.params['shape_type'].get()
        if shape_type == "Box":
            self.params['width'].set(1.0)
            self.params['height'].set(1.0)
            self.params['depth'].set(0.3)
        else:
            self.params['width'].set(3.0)
            self.params['height'].set(3.0)
            self.params['depth'].set(1.5)
    
    def update_drill_size(self):
        """Update the drill size based on user selection."""
        size_map = {
            "Small": 6,
            "Medium": 8,
            "Large": 12
        }
        size = size_map[self.params['drill_size'].get()]
        self.visualizer.set_drill_size(size)
    
    def reset_defaults(self):
        """Reset all parameters to their default values."""
        shape_type = self.params['shape_type'].get()
        defaults = {
            'shape_type': shape_type,
            'width': 1.0 if shape_type == "Box" else 3.0,
            'height': 1.0 if shape_type == "Box" else 3.0,
            'depth': 0.3 if shape_type == "Box" else 1.5,
            'margin': 0.1,
            'grid_size': 8,
            'pop_size': 70,
            'generations': 150,
            'mutation_rate': 0.5,
            'elitism': 3,
            'tournament_size': 5,
            'animation_speed': 50,
            'use_hybrid': True,
            'drill_size': "Medium"
        }
        
        for key, value in defaults.items():
            self.params[key].set(value)
    
    def run_optimization(self):
        """Run the toolpath optimization process."""
        self.status_var.set("Optimizing...")
        self.root.update()
        
        # Get parameters
        params = {k: v.get() for k, v in self.params.items()}
        shape_type = params['shape_type']
        
        try:
            # Run simulation with current parameters
            self.grid_points, x_vals, y_vals = create_grid_points(
                params['width'], params['height'], params['depth'], 
                params['margin'], params['grid_size'], shape_type)
            
            # Optimize path
            optimizer = ToolpathOptimizer(self.grid_points)
            global GA_PARAMS
            GA_PARAMS = {
                'pop_size': params['pop_size'],
                'generations': params['generations'],
                'mutation_rate': params['mutation_rate'],
                'elitism': params['elitism'],
                'tournament_size': params['tournament_size']
            }
            
            optimized_path_indices = optimizer.evolve()
            self.optimized_path = [self.grid_points[i] for i in optimized_path_indices]
            self.zigzag_path = create_zigzag_path(x_vals, y_vals, params['width'], 
                                           params['height'], params['depth'], shape_type)
            
            if params['use_hybrid']:
                self.hybrid_path = create_hybrid_path(self.optimized_path, self.grid_points,
                                                    params['width'], params['height'], 
                                                    params['depth'], shape_type)
            else:
                self.hybrid_path = self.optimized_path.copy()
            
            def find_closest_point_index(target_point, points_list):
                distances = [np.linalg.norm(np.array(target_point) - np.array(p)) for p in points_list]
                return np.argmin(distances)
            
            opt_indices = [find_closest_point_index(p, self.grid_points) for p in self.optimized_path]
            opt_length = optimizer.path_length(opt_indices)
            
            zz_indices = [find_closest_point_index(p, self.grid_points) for p in self.zigzag_path]
            zz_length = optimizer.path_length(zz_indices)
            
            hybrid_indices = [find_closest_point_index(p, self.grid_points) for p in self.hybrid_path]
            hybrid_length = optimizer.path_length(hybrid_indices)
            
            status_text = (f"Optimized: {opt_length:.2f} units | "
                         f"Zigzag: {zz_length:.2f} units | "
                         f"Hybrid: {hybrid_length:.2f} units")
            self.status_var.set(status_text)
            
            self.visualizer.setup_scenes(params['width'], params['height'], 
                                       params['depth'], self.grid_points, shape_type)
            self.visualizer.animate_paths(self.optimized_path, self.zigzag_path, self.hybrid_path)
            
            self.update_drill_size()
            
            self.canvas.draw()
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", f"An error occurred during optimization:\n{str(e)}")
        
        self.status_var.set("Optimization complete")
    
    def toggle_animation(self):
        """Toggle animation play/pause state."""
        self.visualizer.toggle_animation()
    
    def step_animation(self, step):
        """
        Step through animation frames.
        
        Args:
            step (int): Number of frames to step
        """
        self.visualizer.step_animation(step)
    
    def reset_animation(self):
        """Reset animation to the first frame."""
        self.visualizer.reset_animation()
    
    def update_animation_speed(self):
        """Update animation speed based on user setting."""
        if hasattr(self.visualizer, 'ani') and self.visualizer.ani:
            self.visualizer.ani.event_source.interval = 1000 / self.params['animation_speed'].get()
    
    def export_path_data(self):
        """Export path data to a CSV file."""
        if not self.optimized_path or not self.zigzag_path or not self.hybrid_path:
            self.status_var.set("No path data to export")
            messagebox.showwarning("Warning", "Please run optimization first to generate path data")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save path data as"
        )
        
        if not file_path:
            return
            
        try:
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Type', 'X', 'Y', 'Z'])
                
                writer.writerow(['Optimized Path'])
                for point in self.optimized_path:
                    writer.writerow(['', *point])
                
                writer.writerow(['Zigzag Path'])
                for point in self.zigzag_path:
                    writer.writerow(['', *point])
                
                writer.writerow(['Hybrid Path'])
                for point in self.hybrid_path:
                    writer.writerow(['', *point])
                
                writer.writerow(['Grid Points'])
                for point in self.grid_points:
                    writer.writerow(['', *point])
            
            self.status_var.set(f"Path data saved to {os.path.basename(file_path)}")
            messagebox.showinfo("Success", f"Path data successfully saved to:\n{file_path}")
        except Exception as e:
            self.status_var.set(f"Export failed: {str(e)}")
            messagebox.showerror("Error", f"Failed to save path data:\n{str(e)}")
    
    def export_image(self):
        """Export visualization to an image file."""
        if not hasattr(self.visualizer, 'fig'):
            self.status_var.set("No visualization to export")
            messagebox.showwarning("Warning", "Please run optimization first to generate visualization")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
            title="Save image as"
        )
        
        if not file_path:
            return
            
        try:
            self.visualizer.fig.savefig(file_path, dpi=300, bbox_inches='tight')
            self.status_var.set(f"Image saved to {os.path.basename(file_path)}")
            messagebox.showinfo("Success", f"Image successfully saved to:\n{file_path}")
        except Exception as e:
            self.status_var.set(f"Export failed: {str(e)}")
            messagebox.showerror("Error", f"Failed to save image:\n{str(e)}")
    
    def export_animation(self):
        """Export animation to a GIF file."""
        if not hasattr(self.visualizer, 'ani'):
            self.status_var.set("No animation to export")
            messagebox.showwarning("Warning", "Please run optimization first to generate animation")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".gif",
            filetypes=[("GIF files", "*.gif"), ("All files", "*.*")],
            title="Save animation as"
        )
        
        if not file_path:
            return
            
        try:
            self.status_var.set("Exporting animation...")
            self.root.update()
            self.visualizer.save_animation(file_path)
            self.status_var.set(f"Animation saved to {os.path.basename(file_path)}")
            messagebox.showinfo("Success", f"Animation successfully saved to:\n{file_path}")
        except Exception as e:
            self.status_var.set(f"Export failed: {str(e)}")
            messagebox.showerror("Error", f"Failed to save animation:\n{str(e)}")

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = ToolpathDashboard(root)
    root.mainloop()