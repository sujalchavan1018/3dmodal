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
from PIL import Image, ImageTk
import webbrowser

# Modern color scheme
BG_COLOR = "#f0f0f0"
ACCENT_COLOR = "#4a6fa5"
HIGHLIGHT_COLOR = "#6c8fc7"
TEXT_COLOR = "#333333"
BUTTON_COLOR = "#5d8bb7"
DARK_BG = "#2c3e50"

# Parameters with default values
grid_size = 8
pocket_margin = 0.10
pocket_depth = 0.30
letter_width = 3.0
letter_height = 3.0
letter_depth = 1.5
box_size = 1.0

# GA Parameters with default values
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
        return sum(self.distance_matrix[path_indices[i]][path_indices[i+1]] for i in range(len(path_indices)-1))
        
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

def create_zigzag_path(x_vals, y_vals, width, height, depth, design_type):
    path = []
    if design_type == "Box":
        for i, x in enumerate(x_vals):
            if i % 2 == 0:
                path.extend([[x, y, depth] for y in y_vals])
            else:
                path.extend([[x, y, depth] for y in reversed(y_vals)])
    else:
        if design_type == "H":
            for i, x in enumerate(x_vals):
                if i % 2 == 0:
                    for y in y_vals:
                        if (x < 0.8 or x > width-0.8 or (y > height*0.37 and y < height*0.63)):
                            path.append([x, y, depth])
                else:
                    for y in reversed(y_vals):
                        if (x < 0.8 or x > width-0.8 or (y > height*0.37 and y < height*0.63)):
                            path.append([x, y, depth])
        elif design_type == "I":
            center_x = width / 2
            bar_width = 0.8
            for i, y in enumerate(y_vals):
                if i % 2 == 0:
                    for x in np.linspace(center_x - bar_width/2, center_x + bar_width/2, len(x_vals)//2):
                        path.append([x, y, depth])
                else:
                    for x in np.linspace(center_x + bar_width/2, center_x - bar_width/2, len(x_vals)//2):
                        path.append([x, y, depth])
        elif design_type == "L":
            for i, y in enumerate(y_vals):
                if y < 0.8:
                    if i % 2 == 0:
                        for x in x_vals:
                            path.append([x, y, depth])
                    else:
                        for x in reversed(x_vals):
                            path.append([x, y, depth])
                else:
                    if i % 2 == 0:
                        for x in x_vals[:len(x_vals)//4]:
                            path.append([x, y, depth])
                    else:
                        for x in reversed(x_vals[:len(x_vals)//4]):
                            path.append([x, y, depth])
        elif design_type == "X":
            for i, x in enumerate(x_vals):
                y1 = (height/width) * x
                y2 = height - (height/width) * x
                path.append([x, y1, depth])
                path.append([x, y2, depth])
        elif design_type == "T":
            center_x = width / 2
            bar_width = 0.8
            for i, y in enumerate(y_vals):
                if y > height * 0.8:
                    if i % 2 == 0:
                        for x in x_vals:
                            path.append([x, y, depth])
                    else:
                        for x in reversed(x_vals):
                            path.append([x, y, depth])
                else:
                    if i % 2 == 0:
                        for x in np.linspace(center_x - bar_width/2, center_x + bar_width/2, len(x_vals)//4):
                            path.append([x, y, depth])
                    else:
                        for x in np.linspace(center_x + bar_width/2, center_x - bar_width/2, len(x_vals)//4):
                            path.append([x, y, depth])
    return path

def create_spiral_pocket_path(grid_points, width, height, depth, design_type):
    if design_type == "Box":
        pocket_points = [p for p in grid_points if pocket_margin <= p[0] <= width-pocket_margin and 
                        pocket_margin <= p[1] <= height-pocket_margin]
    elif design_type == "H":
        pocket_points = [p for p in grid_points if 1.0 < p[0] < width-1.0 and 
                        height*0.37 < p[1] < height*0.63]
    elif design_type == "I":
        center_x = width / 2
        bar_width = 0.8
        pocket_points = [p for p in grid_points if center_x - bar_width/2 <= p[0] <= center_x + bar_width/2]
    elif design_type == "L":
        pocket_points = [p for p in grid_points if p[0] < 0.8 or p[1] < 0.8]
    elif design_type == "X":
        bar_width = 0.5
        pocket_points = []
        for p in grid_points:
            y1 = (height/width) * p[0]
            y2 = height - (height/width) * p[0]
            if abs(p[1] - y1) < bar_width/2 or abs(p[1] - y2) < bar_width/2:
                pocket_points.append(p)
    elif design_type == "T":
        center_x = width / 2
        bar_width = 0.8
        top_height = 0.3
        pocket_points = [p for p in grid_points if (p[1] > height - top_height) or 
                       (center_x - bar_width/2 <= p[0] <= center_x + bar_width/2)]
    
    if not pocket_points:
        return []
        
    center_x = np.mean([p[0] for p in pocket_points])
    center_y = np.mean([p[1] for p in pocket_points])
    
    pocket_points_sorted = sorted(pocket_points,
                                key=lambda p: (np.arctan2(p[1]-center_y, p[0]-center_x),
                                np.hypot(p[0]-center_x, p[1]-center_y)))
    
    spiral_path = []
    angles = [np.arctan2(p[1]-center_y, p[0]-center_x) for p in pocket_points_sorted]
    dists = [np.hypot(p[0]-center_x, p[1]-center_y) for p in pocket_points_sorted]
    
    angle_sectors = np.linspace(-np.pi, np.pi, 8)
    sector_groups = [[] for _ in range(len(angle_sectors)-1)]
    
    for point in pocket_points_sorted:
        angle = np.arctan2(point[1]-center_y, point[0]-center_x)
        for i in range(len(angle_sectors)-1):
            if angle_sectors[i] <= angle < angle_sectors[i+1]:
                sector_groups[i].append(point)
                break
    
    for i in range(len(sector_groups)):
        if i % 2 == 0:
            sector_groups[i].sort(key=lambda p: np.hypot(p[0]-center_x, p[1]-center_y))
        else:
            sector_groups[i].sort(key=lambda p: -np.hypot(p[0]-center_x, p[1]-center_y))
        spiral_path.extend(sector_groups[i])
    
    if len(spiral_path) > 3:
        x = [p[0] for p in spiral_path]
        y = [p[1] for p in spiral_path]
        z = [p[2] for p in spiral_path]
        
        t = np.linspace(0, 1, len(spiral_path))
        t_new = np.linspace(0, 1, len(spiral_path)*3)
        
        try:
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
    distances = [np.linalg.norm(np.array(target_point) - np.array(p)) for p in points_list]
    return np.argmin(distances)

def create_hybrid_path(optimized_path, grid_points, width, height, depth, design_type):
    if design_type == "Box":
        pocket_points = [p for p in grid_points if pocket_margin <= p[0] <= width-pocket_margin and 
                        pocket_margin <= p[1] <= height-pocket_margin]
    elif design_type == "H":
        pocket_points = [p for p in grid_points if 1.0 < p[0] < width-1.0 and 
                        height*0.37 < p[1] < height*0.63]
    elif design_type == "I":
        center_x = width / 2
        bar_width = 0.8
        pocket_points = [p for p in grid_points if center_x - bar_width/2 <= p[0] <= center_x + bar_width/2]
    elif design_type == "L":
        pocket_points = [p for p in grid_points if p[0] < 0.8 or p[1] < 0.8]
    elif design_type == "X":
        bar_width = 0.5
        pocket_points = []
        for p in grid_points:
            y1 = (height/width) * p[0]
            y2 = height - (height/width) * p[0]
            if abs(p[1] - y1) < bar_width/2 or abs(p[1] - y2) < bar_width/2:
                pocket_points.append(p)
    elif design_type == "T":
        center_x = width / 2
        bar_width = 0.8
        top_height = 0.3
        pocket_points = [p for p in grid_points if (p[1] > height - top_height) or 
                       (center_x - bar_width/2 <= p[0] <= center_x + bar_width/2)]
    
    non_pocket_points = [p for p in grid_points if p not in pocket_points]
    
    optimized_non_pocket = [p for p in optimized_path if p in non_pocket_points]
    
    spiral_pocket = create_spiral_pocket_path(grid_points, width, height, depth, design_type)
    if spiral_pocket:
        spiral_indices = [find_closest_point_index(sp, grid_points) for sp in spiral_pocket]
        spiral_points = [grid_points[i] for i in spiral_indices]
    else:
        spiral_points = []
    
    if optimized_non_pocket and spiral_points:
        spiral_start_point = spiral_points[0]
        closest_idx = min(range(len(optimized_non_pocket)),
                        key=lambda i: np.linalg.norm(
                            np.array(optimized_non_pocket[i]) - 
                            np.array(spiral_start_point)))
        
        hybrid_points = (optimized_non_pocket[:closest_idx+1] + 
                        spiral_points + 
                        optimized_non_pocket[closest_idx+1:])
    else:
        hybrid_points = optimized_path
    
    if len(hybrid_points) > 3:
        x = [p[0] for p in hybrid_points]
        y = [p[1] for p in hybrid_points]
        z = [p[2] for p in hybrid_points]
        
        t = np.linspace(0, 1, len(hybrid_points))
        t_new = np.linspace(0, 1, len(hybrid_points)*2)
        
        try:
            spl_x = make_interp_spline(t, x, k=3)
            spl_y = make_interp_spline(t, y, k=3)
            spl_z = make_interp_spline(t, z, k=3)
            
            x_smooth = spl_x(t_new)
            y_smooth = spl_y(t_new)
            z_smooth = spl_z(t_new)
            
            smooth_path = list(zip(x_smooth, y_smooth, z_smooth))
            return smooth_path
        except:
            return hybrid_points
    
    return hybrid_points

class DesignVisualizer:
    def __init__(self, fig):
        self.fig = fig
        self.ax1 = self.fig.add_subplot(131, projection='3d')
        self.ax2 = self.fig.add_subplot(132, projection='3d')
        self.ax3 = self.fig.add_subplot(133, projection='3d')
        self._setup_lighting()
        self.ani = None
        self.animation_running = False
        self.current_frame = 0
        self.max_frames = 0
        self.optimized_path = []
        self.zigzag_path = []
        self.hybrid_path = []
    
    def _setup_lighting(self):
        self.light = LightSource(azdeg=225, altdeg=45)
        self.face_colors = {
            'base': '#4682B4', 'front': '#5F9EA0', 'back': '#B0C4DE',
            'right': '#87CEEB', 'left': '#ADD8E6', 'pocket': '#D3D3D3',
            'top': '#98FB98', 'grid': '#FFFFFF'
        }
    
    def _create_box(self, ax, width, height, depth):
        vertices = np.array([
            [0, 0, 0], [width, 0, 0], [width, height, 0], [0, height, 0],
            [0, 0, depth], [width, 0, depth], [width, height, depth], [0, height, depth]
        ])
        
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]],
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[1], vertices[2], vertices[6], vertices[5]],
            [vertices[0], vertices[3], vertices[7], vertices[4]]
        ]
        
        pocket_verts = np.array([
            [pocket_margin, pocket_margin, depth], 
            [width-pocket_margin, pocket_margin, depth], 
            [width-pocket_margin, height-pocket_margin, depth], 
            [pocket_margin, height-pocket_margin, depth],
            [pocket_margin, pocket_margin, depth-pocket_depth], 
            [width-pocket_margin, pocket_margin, depth-pocket_depth], 
            [width-pocket_margin, height-pocket_margin, depth-pocket_depth], 
            [pocket_margin, height-pocket_margin, depth-pocket_depth]
        ])
        
        pocket_faces = [
            [pocket_verts[0], pocket_verts[1], pocket_verts[2], pocket_verts[3]],
            [pocket_verts[4], pocket_verts[5], pocket_verts[6], pocket_verts[7]],
            [pocket_verts[0], pocket_verts[1], pocket_verts[5], pocket_verts[4]],
            [pocket_verts[2], pocket_verts[3], pocket_verts[7], pocket_verts[6]],
            [pocket_verts[1], pocket_verts[2], pocket_verts[6], pocket_verts[5]],
            [pocket_verts[0], pocket_verts[3], pocket_verts[7], pocket_verts[4]]
        ]
        
        for face in faces:
            poly = Poly3DCollection(
                [face],
                facecolors=self.face_colors['base'],
                linewidths=1,
                edgecolors='k',
                zorder=1,
                alpha=0.7
            )
            ax.add_collection3d(poly)
        
        for face in pocket_faces:
            poly = Poly3DCollection(
                [face],
                facecolors=self.face_colors['pocket'],
                linewidths=1,
                edgecolors='k',
                zorder=2,
                alpha=0.7
            )
            ax.add_collection3d(poly)
        
        x_vals = np.linspace(pocket_margin, width-pocket_margin, grid_size)
        y_vals = np.linspace(pocket_margin, height-pocket_margin, grid_size)
        grid_lines = []
        for i in range(grid_size):
            for j in range(grid_size-1):
                p1 = [x_vals[j], y_vals[i], depth]
                p2 = [x_vals[j+1], y_vals[i], depth]
                grid_lines.append([p1, p2])
                p1 = [x_vals[i], y_vals[j], depth]
                p2 = [x_vals[i], y_vals[j+1], depth]
                grid_lines.append([p1, p2])
        
        ax.add_collection3d(Line3DCollection(
            grid_lines,
            colors='white',
            linewidths=1.5,
            zorder=4
        ))
    
    def _create_letter_H(self, ax, width, height, depth):
        left_width = 0.8
        bar_thickness = 0.8
        crossbar_height = height * 0.37
        
        left_front = np.array([
            [0, 0, 0], [0, height, 0], [left_width, height, 0], [left_width, 0, 0],
            [0, 0, depth], [0, height, depth], [left_width, height, depth], [left_width, 0, depth]
        ])
        
        right_front = np.array([
            [width-left_width, 0, 0], [width-left_width, height, 0], [width, height, 0], [width, 0, 0],
            [width-left_width, 0, depth], [width-left_width, height, depth], [width, height, depth], [width, 0, depth]
        ])
        
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
        
        self._add_letter_parts(ax, [left_front, right_front, horizontal], [1.0, 1.0, 0.9])
    
    def _create_letter_I(self, ax, width, height, depth):
        center_x = width / 2
        bar_width = 0.8
        cap_height = 0.3
        
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
        
        self._add_letter_parts(ax, [vertical, top_cap, bottom_cap], [1.0, 0.9, 0.9])
    
    def _create_letter_L(self, ax, width, height, depth):
        vertical = np.array([
            [0, 0, 0], [0, height, 0], [0.8, height, 0], [0.8, 0, 0],
            [0, 0, depth], [0, height, depth], [0.8, height, depth], [0.8, 0, depth]
        ])
        
        horizontal = np.array([
            [0, 0, 0], [0, 0.8, 0], [width, 0.8, 0], [width, 0, 0],
            [0, 0, depth], [0, 0.8, depth], [width, 0.8, depth], [width, 0, depth]
        ])
        
        self._add_letter_parts(ax, [vertical, horizontal], [1.0, 0.9])
    
    def _create_letter_X(self, ax, width, height, depth):
        bar_width = 0.5
        
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
        
        self._add_letter_parts(ax, [diag1, diag2], [1.0, 0.9])
    
    def _create_letter_T(self, ax, width, height, depth):
        bar_width = 0.8
        center_x = width / 2
        top_height = 0.3
        
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
        
        self._add_letter_parts(ax, [horizontal, vertical], [1.0, 0.9])
    
    def _add_letter_parts(self, ax, parts, color_factors):
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
        return [
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]],
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[1], vertices[2], vertices[6], vertices[5]],
            [vertices[0], vertices[3], vertices[7], vertices[4]]
        ]
    
    def setup_scenes(self, width, height, depth, grid_points, design_type):
        self.fig.clf()
        self.ax1 = self.fig.add_subplot(131, projection='3d')
        self.ax2 = self.fig.add_subplot(132, projection='3d')
        self.ax3 = self.fig.add_subplot(133, projection='3d')
        
        if design_type == "Box":
            self._create_box(self.ax1, width, height, depth)
            self._create_box(self.ax2, width, height, depth)
            self._create_box(self.ax3, width, height, depth)
        elif design_type == "H":
            self._create_letter_H(self.ax1, width, height, depth)
            self._create_letter_H(self.ax2, width, height, depth)
            self._create_letter_H(self.ax3, width, height, depth)
        elif design_type == "I":
            self._create_letter_I(self.ax1, width, height, depth)
            self._create_letter_I(self.ax2, width, height, depth)
            self._create_letter_I(self.ax3, width, height, depth)
        elif design_type == "L":
            self._create_letter_L(self.ax1, width, height, depth)
            self._create_letter_L(self.ax2, width, height, depth)
            self._create_letter_L(self.ax3, width, height, depth)
        elif design_type == "X":
            self._create_letter_X(self.ax1, width, height, depth)
            self._create_letter_X(self.ax2, width, height, depth)
            self._create_letter_X(self.ax3, width, height, depth)
        elif design_type == "T":
            self._create_letter_T(self.ax1, width, height, depth)
            self._create_letter_T(self.ax2, width, height, depth)
            self._create_letter_T(self.ax3, width, height, depth)
        
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
        self.fig.suptitle(f'{design_type} Toolpath Comparison', fontsize=16, y=0.95)
        
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
        self.optimized_path = optimized_path
        self.zigzag_path = zigzag_path
        self.hybrid_path = hybrid_path
        self.current_frame = 0
        self.max_frames = max(len(optimized_path), len(zigzag_path), len(hybrid_path))
        
        self.opt_line, = self.ax1.plot([], [], [], 'r-', linewidth=2, zorder=6)
        self.opt_drill, = self.ax1.plot([], [], [], 'ko', markersize=8, zorder=10)
        
        self.zz_line, = self.ax2.plot([], [], [], 'b-', linewidth=2, zorder=6)
        self.zz_drill, = self.ax2.plot([], [], [], 'ko', markersize=8, zorder=10)
        
        self.hybrid_line, = self.ax3.plot([], [], [], 'g-', linewidth=2, zorder=6)
        self.hybrid_drill, = self.ax3.plot([], [], [], 'ko', markersize=8, zorder=10)
        
        self.opt_x, self.opt_y, self.opt_z = [], [], []
        self.zz_x, self.zz_y, self.zz_z = [], [], []
        self.hybrid_x, self.hybrid_y, self.hybrid_z = [], [], []
        
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
        if frame < len(self.optimized_path):
            x, y, z = self.optimized_path[frame]
            self.opt_drill.set_data([x], [y])
            self.opt_drill.set_3d_properties([z])
            self.opt_x.append(x)
            self.opt_y.append(y)
            self.opt_z.append(z)
            self.opt_line.set_data(self.opt_x, self.opt_y)
            self.opt_line.set_3d_properties(self.opt_z)
        
        if frame < len(self.zigzag_path):
            x, y, z = self.zigzag_path[frame]
            self.zz_drill.set_data([x], [y])
            self.zz_drill.set_3d_properties([z])
            self.zz_x.append(x)
            self.zz_y.append(y)
            self.zz_z.append(z)
            self.zz_line.set_data(self.zz_x, self.zz_y)
            self.zz_line.set_3d_properties(self.zz_z)
        
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
    
    def toggle_animation(self):
        if self.ani:
            if self.animation_running:
                self.ani.event_source.stop()
                self.animation_running = False
            else:
                self.ani.event_source.start()
                self.animation_running = True
    
    def step_animation(self, step):
        if not self.ani:
            return
            
        new_frame = self.current_frame + step
        if 0 <= new_frame < self.max_frames:
            self._update_animation(new_frame)
            self.fig.canvas.draw_idle()
    
    def reset_animation(self):
        if not self.ani:
            return
            
        self.opt_x, self.opt_y, self.opt_z = [], [], []
        self.zz_x, self.zz_y, self.zz_z = [], [], []
        self.hybrid_x, self.hybrid_y, self.hybrid_z = [], [], []
        self.current_frame = 0
        self._update_animation(0)
        self.fig.canvas.draw_idle()
    
    def save_animation(self, filename):
        if not self.ani:
            return
            
        writer = PillowWriter(fps=15)
        self.ani.save(filename, writer=writer)

def create_grid_points(width, height, depth, margin, grid_size, design_type):
    x_vals = np.linspace(margin, width-margin, grid_size)
    y_vals = np.linspace(margin, height-margin, grid_size)
    
    if design_type == "Box":
        grid_points = [[x, y, depth] for x in x_vals for y in y_vals]
    elif design_type == "H":
        grid_points = [[x, y, depth] for x in x_vals for y in y_vals 
                      if (x < 0.8 or x > width-0.8 or (y > height*0.37 and y < height*0.63))]
    elif design_type == "I":
        center_x = width / 2
        bar_width = 0.8
        grid_points = [[x, y, depth] for x in x_vals for y in y_vals 
                      if (center_x - bar_width/2 <= x <= center_x + bar_width/2)]
    elif design_type == "L":
        grid_points = []
        for x in x_vals:
            for y in y_vals:
                if (x < 0.8 or y < 0.8):
                    grid_points.append([x, y, depth])
    elif design_type == "X":
        bar_width = 0.5
        grid_points = []
        for x in x_vals:
            y1 = (height/width) * x
            y2 = height - (height/width) * x
            for y in y_vals:
                if (abs(y - y1) < bar_width/2 or abs(y - y2) < bar_width/2):
                    grid_points.append([x, y, depth])
    elif design_type == "T":
        center_x = width / 2
        bar_width = 0.8
        top_height = 0.3
        grid_points = [[x, y, depth] for x in x_vals for y in y_vals 
                      if (y > height - top_height) or 
                      (center_x - bar_width/2 <= x <= center_x + bar_width/2)]
    
    return grid_points, x_vals, y_vals

class ToolpathDashboard:
    def __init__(self, root):
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
            'design_type': tk.StringVar(value="Box"),
            'width': tk.DoubleVar(value=1.0),
            'height': tk.DoubleVar(value=1.0),
            'depth': tk.DoubleVar(value=0.3),
            'margin': tk.DoubleVar(value=0.1),
            'grid_size': tk.IntVar(value=8),
            'pop_size': tk.IntVar(value=70),
            'generations': tk.IntVar(value=150),
            'mutation_rate': tk.DoubleVar(value=0.5),
            'elitism': tk.IntVar(value=3),
            'tournament_size': tk.IntVar(value=5),
            'animation_speed': tk.IntVar(value=50),
            'use_hybrid': tk.BooleanVar(value=True)
        }
        
        # Create notebook for organized controls
        notebook = ttk.Notebook(control_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Design tab
        design_tab = ttk.Frame(notebook)
        notebook.add(design_tab, text="Design")
        
        # Design type selection
        ttk.Label(design_tab, text="Design Type").grid(row=0, column=0, sticky="w", padx=(0, 5), pady=(0, 5))
        design_menu = ttk.Combobox(design_tab, textvariable=self.params['design_type'], 
                                 values=["Box", "H", "I", "L", "X", "T"], state="readonly")
        design_menu.grid(row=0, column=1, columnspan=2, sticky="ew", pady=(0, 10))
        design_menu.bind("<<ComboboxSelected>>", self.update_dimensions)
        
        # Create parameter controls
        ttk.Label(design_tab, text="Dimensions", font=('Arial', 10, 'bold')).grid(
            row=1, column=0, columnspan=3, pady=(0, 5), sticky="w")
        self.create_slider(design_tab, "Width (m)", 'width', 0.5, 5.0, 2)
        self.create_slider(design_tab, "Height (m)", 'height', 0.5, 5.0, 3)
        self.create_slider(design_tab, "Depth (m)", 'depth', 0.1, 1.0, 4)
        
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
        
        ttk.Label(options_tab, text="Animation Speed", font=('Arial', 10, 'bold')).grid(
            row=2, column=0, columnspan=3, pady=(5, 5), sticky="w")
        ttk.Scale(options_tab, from_=10, to=200, variable=self.params['animation_speed'],
                 command=lambda v: self.update_animation_speed(), style="Custom.Horizontal.TScale").grid(
                 row=3, column=0, columnspan=2, sticky="ew", padx=(0, 5))
        ttk.Label(options_tab, textvariable=self.params['animation_speed'], style="Custom.TLabel").grid(
            row=3, column=2, sticky="w")
        
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
        self.visualizer = DesignVisualizer(self.fig)
        
        # Store paths for export
        self.optimized_path = []
        self.zigzag_path = []
        self.hybrid_path = []
        self.grid_points = []
    
    def configure_styles(self):
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
        help_text = """Toolpath Optimization Dashboard Help:

1. Design Tab:
   - Select design type (Box or letters)
   - Adjust dimensions and grid settings

2. GA Parameters Tab:
   - Configure genetic algorithm settings
   - Population size and generations affect optimization quality
   - Mutation rate controls exploration vs exploitation

3. Options Tab:
   - Toggle hybrid path (combines optimized and spiral paths)
   - Adjust animation speed

4. Simulation Controls:
   - Play/pause animation
   - Step through animation frame by frame
   - Reset animation

5. Export:
   - Save path data as CSV
   - Save visualization as image
   - Save animation as GIF

For more information, visit the documentation website."""
        
        messagebox.showinfo("Help", help_text)
    
    def update_dimensions(self, event=None):
        design_type = self.params['design_type'].get()
        if design_type == "Box":
            self.params['width'].set(1.0)
            self.params['height'].set(1.0)
            self.params['depth'].set(0.3)
        else:
            self.params['width'].set(3.0)
            self.params['height'].set(3.0)
            self.params['depth'].set(1.5)
    
    def reset_defaults(self):
        design_type = self.params['design_type'].get()
        defaults = {
            'design_type': design_type,
            'width': 1.0 if design_type == "Box" else 3.0,
            'height': 1.0 if design_type == "Box" else 3.0,
            'depth': 0.3 if design_type == "Box" else 1.5,
            'margin': 0.1,
            'grid_size': 8,
            'pop_size': 70,
            'generations': 150,
            'mutation_rate': 0.5,
            'elitism': 3,
            'tournament_size': 5,
            'animation_speed': 50,
            'use_hybrid': True
        }
        
        for key, value in defaults.items():
            self.params[key].set(value)
    
    def run_optimization(self):
        self.status_var.set("Optimizing...")
        self.root.update()
        
        # Get parameters
        params = {k: v.get() for k, v in self.params.items()}
        design_type = params['design_type']
        
        try:
            # Run simulation with current parameters
            self.grid_points, x_vals, y_vals = create_grid_points(
                params['width'], params['height'], params['depth'], 
                params['margin'], params['grid_size'], design_type)
            
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
                                           params['height'], params['depth'], design_type)
            
            if params['use_hybrid']:
                self.hybrid_path = create_hybrid_path(self.optimized_path, self.grid_points,
                                                    params['width'], params['height'], 
                                                    params['depth'], design_type)
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
                                       params['depth'], self.grid_points, design_type)
            self.visualizer.animate_paths(self.optimized_path, self.zigzag_path, self.hybrid_path)
            
            self.canvas.draw()
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", f"An error occurred during optimization:\n{str(e)}")
        
        self.status_var.set("Optimization complete")
    
    def toggle_animation(self):
        self.visualizer.toggle_animation()
    
    def step_animation(self, step):
        self.visualizer.step_animation(step)
    
    def reset_animation(self):
        self.visualizer.reset_animation()
    
    def update_animation_speed(self):
        if hasattr(self.visualizer, 'ani') and self.visualizer.ani:
            self.visualizer.ani.event_source.interval = 1000 / self.params['animation_speed'].get()
    
    def export_path_data(self):
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

if __name__ == "__main__":
    root = tk.Tk()
    app = ToolpathDashboard(root)
    root.mainloop()