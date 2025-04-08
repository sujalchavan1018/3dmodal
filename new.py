import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Create figure and 3D axis
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Define cube vertices
vertices = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Bottom face
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # Top face
])

# Define pocket (1 cm deep, increased length and width)
pocket_depth = 0.1
pocket_vertices = np.array([
    [0.15, 0.15, 1], [0.85, 0.15, 1], [0.85, 0.85, 1], [0.15, 0.85, 1],  # Top pocket edges
    [0.15, 0.15, 1 - pocket_depth], [0.85, 0.15, 1 - pocket_depth],
    [0.85, 0.85, 1 - pocket_depth], [0.15, 0.85, 1 - pocket_depth]  # Bottom pocket edges
])

# Define cube and pocket faces
faces = [
    [vertices[j] for j in [0, 1, 2, 3]],  # Bottom face
    [vertices[j] for j in [4, 5, 6, 7]],  # Top face
    [vertices[j] for j in [0, 1, 5, 4]],  # Side face 1
    [vertices[j] for j in [2, 3, 7, 6]],  # Side face 2
    [vertices[j] for j in [1, 2, 6, 5]],  # Side face 3
    [vertices[j] for j in [0, 3, 7, 4]],  # Side face 4
    # Pocket faces
    [pocket_vertices[j] for j in [4, 5, 6, 7]],  # Bottom pocket face (solid light blue)
    [pocket_vertices[j] for j in [0, 1, 5, 4]],  # Pocket side face 1 (solid light blue)
    [pocket_vertices[j] for j in [1, 2, 6, 5]],  # Pocket side face 2 (solid light blue)
    [pocket_vertices[j] for j in [2, 3, 7, 6]],  # Pocket side face 3 (solid light blue)
    [pocket_vertices[j] for j in [3, 0, 4, 7]],  # Pocket side face 4 (solid light blue)
]

# Add cube with pocket
box = Poly3DCollection(faces, alpha=0.7, facecolor=['silver'] * 6 + ['lightblue'] * 5, edgecolor='black', linewidths=1.2)
ax.add_collection3d(box)

# Define zigzag toolpath inside the pocket
num_steps = 10
x_start, x_end = 0.15, 0.85
y_start, y_end = 0.15, 0.85
z_depth = 1 - pocket_depth

zigzag_x = []
zigzag_y = []
zigzag_z = []

for i in range(num_steps):
    y_pos = y_start + (y_end - y_start) * (i / (num_steps - 1))
    if i % 2 == 0:
        zigzag_x.append(np.linspace(x_start, x_end, num_steps))
    else:
        zigzag_x.append(np.linspace(x_end, x_start, num_steps))
    zigzag_y.append(np.full(num_steps, y_pos))
    zigzag_z.append(np.full(num_steps, z_depth))

zigzag_x = np.concatenate(zigzag_x)
zigzag_y = np.concatenate(zigzag_y)
zigzag_z = np.concatenate(zigzag_z)

# Plot the toolpath
ax.plot(zigzag_x, zigzag_y, zigzag_z, linestyle='dashed', color='red')

# Initialize drill bit
drill_bit, = ax.plot([], [], [], 'ro', markersize=6)  # Red circle for drill bit

# Update function for animation
def update(frame):
    drill_bit.set_data([zigzag_x[frame]], [zigzag_y[frame]])  # Wrap in lists
    drill_bit.set_3d_properties([zigzag_z[frame]])  # Wrap in list
    return drill_bit,

# Animation setup
ani = animation.FuncAnimation(fig, update, frames=len(zigzag_x), interval=200, blit=False)

# Set axis limits
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1])

# Improve visualization
ax.view_init(elev=30, azim=45)
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# Show plot
plt.show()
