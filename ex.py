import open3d as o3d
import numpy as np

# Create the box (workpiece)
box = o3d.geometry.TriangleMesh.create_box(width=10, height=10, depth=5)
box.translate([-5, -5, 0])  # Center it
box.paint_uniform_color([0.7, 0.7, 0.7])  # Grey color

# Create the drill (cutting tool)
drill = o3d.geometry.TriangleMesh.create_cylinder(radius=1, height=6)
drill.paint_uniform_color([0, 0, 1])  # Blue color
drill.translate([0, 0, 6])  # Position above the workpiece

# Create the pocket (cut area)
pocket = o3d.geometry.TriangleMesh.create_box(width=6, height=6, depth=2)
pocket.translate([-3, -3, 0])  # Position at the center

# Perform boolean subtraction (cut pocket from workpiece)
box = box - pocket  # Open3D does not support direct boolean subtraction; use another library like PyMesh if needed

# Visualize the model
o3d.visualization.draw_geometries([box, drill])
