import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

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

colors = ['red', 'blue', 'yellow', 'purple', 'orange']

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

# Create chessboard pattern on top side of pocket
grid_size = 8
x_vals = np.linspace(0.10, 0.90, grid_size + 1)
y_vals = np.linspace(0.10, 0.90, grid_size + 1)
chess_lines = []
for x in x_vals:
    chess_lines.append([[x, 0.10, 1], [x, 0.90, 1]])
for y in y_vals:
    chess_lines.append([[0.10, y, 1], [0.90, y, 1]])

ax.add_collection3d(Line3DCollection(chess_lines, colors='black', linewidths=1.5))

# Red Zigzag pattern on bottom surface
num_points = 15
x_path = np.linspace(0.15, 0.85, num_points)
y_path = np.array([0.2, 0.8] * (num_points // 2) + [0.2])
z_path = pocket_depth + 0.001
ax.plot(x_path, y_path, [z_path]*num_points, color='red', linewidth=3, linestyle='-', zorder=10)
ax.scatter(x_path, y_path, [z_path]*num_points, color='white', edgecolors='red', s=40, zorder=12)

ax.view_init(elev=25, azim=-45)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1.1)
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_title('Red Zigzag Path & Chessboard Border on Pocket Top', fontsize=14, pad=20)

plt.tight_layout()
plt.show()
