import trimesh
import os

# 获取脚本所在的目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 使用os.path.join连接脚本目录和文件相对路径
file_path = os.path.join(script_dir, 'data/nd_1/mesh/v3_0000.obj')

# 确保文件存在
assert os.path.exists(file_path), f"File not found: {file_path}"

# 加载文件
mesh = trimesh.load(file_path)

mesh.apply_scale(0.001)
mesh.export('data/nd_1/mesh/scaled_down_file.obj')