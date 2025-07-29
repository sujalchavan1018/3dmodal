import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Triangle parameters
width = 3.0
height = 3.0
depth = 1.2

# Create vertices of the triangular prism
vertices = np.array([
    # Base triangle vertices (z=0)
    [0, 0, 0],          # Vertex 1 (bottom left)
    [width, 0, 0],      # Vertex 2 (bottom right)
    [width/2, height, 0], # Vertex 3 (apex)
    
    # Top triangle vertices (z=depth)
    [0, 0, depth],      # Vertex 4
    [width, 0, depth],  # Vertex 5
    [width/2, height, depth]  # Vertex 6
])

# Define the triangular prism faces
faces = [
    # Bottom face
    [vertices[0], vertices[1], vertices[2]],
    # Top face
    [vertices[3], vertices[4], vertices[5]],
    # Front face (rectangle)
    [vertices[0], vertices[1], vertices[4], vertices[3]],
    # Left face (rectangle)
    [vertices[0], vertices[2], vertices[5], vertices[3]],
    # Right face (rectangle)
    [vertices[1], vertices[2], vertices[5], vertices[4]]
]

# Create the figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Create the 3D triangular prism
poly = Poly3DCollection(faces, alpha=0.8, linewidths=1, edgecolor='k')
poly.set_facecolor('#4682B4')  # Steel blue color
ax.add_collection3d(poly)

# Add a grid pattern to the top surface
x = np.linspace(0, width, 10)
y = np.linspace(0, height, 10)
X, Y = np.meshgrid(x, y)
Z = np.full_like(X, depth)

# Only keep points inside the triangle
mask = Y <= height * (1 - np.abs(2*X/width - 1))
ax.scatter(X[mask], Y[mask], Z[mask], color='white', s=20, edgecolor='k')

# Add a circular pocket in the center
center_x, center_y = width/2, height*0.4
theta = np.linspace(0, 2*np.pi, 30)
x_circle = center_x + 0.5 * np.cos(theta)
y_circle = center_y + 0.5 * np.sin(theta)
z_circle = np.full_like(x_circle, depth - 0.2)
ax.plot(x_circle, y_circle, z_circle, color='red', linewidth=2)

# Set viewing angles and labels
ax.view_init(elev=30, azim=-60)
ax.set_xlim(0, width)
ax.set_ylim(0, height)
ax.set_zlim(0, depth)
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_title('3D Triangular Prism with Surface Grid')

plt.tight_layout()
plt.show()