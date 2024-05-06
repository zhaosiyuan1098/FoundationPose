import trimesh
import os


script_dir = os.path.dirname(os.path.abspath(__file__))

mesh = trimesh.load('realsense/data/mouse/mesh/mouse.obj')
mesh.apply_scale(0.001)
mesh.export('realsense/data/mouse/mesh/scaled_down_file.obj')