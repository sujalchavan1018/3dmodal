import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.animation import FuncAnimation

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

# Setup figure
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

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

# Create 8x8 grid on the top side of the pocket
grid_size = 8
x_vals = np.linspace(0.10, 0.90, grid_size + 1)
y_vals = np.linspace(0.10, 0.90, grid_size + 1)
grid_points = [[x, y, 1] for x in x_vals for y in y_vals]

# Scatter plot for white dots
ax.scatter(*zip(*grid_points), color='white', s=20, zorder=15)  # Mark intersections with white color

# Create line segments for the grid
xy_lines = []
for i in range(grid_size + 1):
    for j in range(grid_size):
        xy_lines.append([[x_vals[j], y_vals[i], 1], [x_vals[j + 1], y_vals[i], 1]])  # Horizontal
        xy_lines.append([[x_vals[i], y_vals[j], 1], [x_vals[i], y_vals[j + 1], 1]])  # Vertical

ax.add_collection3d(Line3DCollection(xy_lines, colors='lightgray', linewidths=1.5))

# Define a zigzag toolpath based on the grid points
toolpath = []
for i in range(grid_size + 1):
    row = grid_points[i * (grid_size + 1):(i + 1) * (grid_size + 1)]
    toolpath.extend(row if i % 2 == 0 else reversed(row))

# Define the drill (represented by a small circle for simplicity)
drill, = ax.plot([], [], [], color='black', marker='o', markersize=10)

# Store traveled points for red line visualization
traveled_x, traveled_y, traveled_z = [], [], []
red_line, = ax.plot([], [], [], color='red', linewidth=2, zorder=5)

# Update function for the animation
def update(i):
    if i < len(toolpath):
        x_pos, y_pos, z_pos = toolpath[i]
        drill.set_data([x_pos], [y_pos])
        drill.set_3d_properties([z_pos])
        
        traveled_x.append(x_pos)
        traveled_y.append(y_pos)
        traveled_z.append(z_pos)
        
        red_line.set_data(traveled_x, traveled_y)
        red_line.set_3d_properties(traveled_z)
    return drill, red_line

# Animation setup
ani = FuncAnimation(fig, update, frames=len(toolpath), interval=100, blit=True)

# Adjust view and axis
ax.view_init(elev=25, azim=-45)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1.1)
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_title('Zigzag Toolpath with Moving Drill and Red Travel Path', fontsize=14, pad=20)

plt.tight_layout()
plt.show()
