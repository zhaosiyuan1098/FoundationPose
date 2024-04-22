import trimesh


script_dir = os.path.dirname(os.path.abspath(__file__))

mesh = trimesh.load('data/realsense/mesh/textured_mesh.obj')
mesh.apply_scale(0.001)
mesh.export('data/realsense/mesh/scaled_down_file.obj')