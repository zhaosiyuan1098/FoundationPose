import os
import yaml
import numpy as np

# 设置文件地址变量
camera_intrinsics_file = 'data/realsense/camera_intrinsics.yaml'
cam_K_file = 'data/realsense/cam_K.txt'

# 读取 camera_intrinsics.yaml 文件
with open(camera_intrinsics_file, 'r') as file:
    data = yaml.safe_load(file)

# 提取 camera_matrix 数据
camera_matrix = data['camera_matrix']['data']

# 将 camera_matrix 数据重塑为 3x3 矩阵
camera_matrix = np.array(camera_matrix).reshape(3, 3)

# 检查 cam_K.txt 文件是否存在，如果不存在则创建
if not os.path.exists(cam_K_file):
    open(cam_K_file, 'w').close()

# 将 camera_matrix 数据写入 cam_K.txt 文件
with open(cam_K_file, 'w') as file:
    for row in camera_matrix:
        file.write(' '.join('{:.18e}'.format(val) for val in row) + '\n')