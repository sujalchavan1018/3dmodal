import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Create main window
root = tk.Tk()
root.title("CNC Drill Path")

# Create figure and 3D axis
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')

# Define box (workpiece) vertices
vertices = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Bottom face
    [0, 0, 0.3], [1, 0, 0.3], [1, 1, 0.3], [0, 1, 0.3]  # Top face
])

# Define faces
faces = [
    [vertices[j] for j in [0, 1, 2, 3]],  # Bottom face
    [vertices[j] for j in [4, 5, 6, 7]],  # Top face (cutting surface)
    [vertices[j] for j in [0, 1, 5, 4]],  # Side face 1
    [vertices[j] for j in [2, 3, 7, 6]],  # Side face 2
    [vertices[j] for j in [1, 2, 6, 5]],  # Side face 3
    [vertices[j] for j in [0, 3, 7, 4]]   # Side face 4
]

# Create 3D collection and set color to gray
workpiece = Poly3DCollection(faces, alpha=0.5, facecolor='gray', edgecolor='black')
ax.add_collection3d(workpiece)

# Draw border on top face
border_x = [0, 1, 1, 0, 0]
border_y = [0, 0, 1, 1, 0]
border_z = [0.3, 0.3, 0.3, 0.3, 0.3]
ax.plot(border_x, border_y, border_z, 'k-', linewidth=2)

# Zig-Zag Toolpath (CNC Cutting on Top Surface)
rows, cols = 5, 10
x_vals = np.linspace(0.1, 0.9, cols)
y_vals = np.ones_like(x_vals) * 0.5
z_surface = 0.3

zigzag_x, zigzag_y, zigzag_z = [], [], []
for i in range(rows):
    if i % 2 == 0:
        zigzag_x.extend(x_vals)
    else:
        zigzag_x.extend(x_vals[::-1])
    zigzag_y.extend(y_vals + (i * 0.1 - 0.2))
    zigzag_z.extend([z_surface] * cols)

zigzag_x, zigzag_y, zigzag_z = map(np.array, [zigzag_x, zigzag_y, zigzag_z])

# CNC Drill
drill, = ax.plot([], [], [], 'ro', markersize=8)
drill_body, = ax.plot([], [], [], 'r-', linewidth=3)
cut_path, = ax.plot([], [], [], 'b-', linewidth=2)

# Animation Control Variables
paused = False
frame_index = 0

def update(frame):
    """Updates CNC drill position."""
    global frame_index
    if paused:
        return drill, drill_body, cut_path
    
    frame_index = frame  # Save frame position

    drill.set_data([zigzag_x[frame]], [zigzag_y[frame]])
    drill.set_3d_properties([zigzag_z[frame] + 0.05])

    drill_body.set_data([zigzag_x[frame], zigzag_x[frame]], [zigzag_y[frame], zigzag_y[frame]])
    drill_body.set_3d_properties([zigzag_z[frame] + 0.1, zigzag_z[frame]])

    cut_path.set_data(zigzag_x[:frame+1], zigzag_y[:frame+1])
    cut_path.set_3d_properties(zigzag_z[:frame+1])

    return drill, drill_body, cut_path

# Play/Pause Functions
def toggle_animation():
    """Toggles play/pause state."""
    global paused
    paused = not paused

# Set Axis Limits
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 0.4])
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# Animation Setup
ani = animation.FuncAnimation(fig, update, frames=len(zigzag_x), interval=200, blit=False)

# Embed Matplotlib in Tkinter
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Buttons
play_pause_button = tk.Button(root, text="Play/Pause", command=toggle_animation, font=("Arial", 12))
play_pause_button.pack(pady=10)

# Run Tkinter GUI
root.mainloop()
