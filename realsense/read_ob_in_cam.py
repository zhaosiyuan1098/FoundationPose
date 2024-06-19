import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse


class TransformationMatrixFilter:
    def __init__(self, std_inv=10, weight_num=5):
        self.std_inv = std_inv  # 标准差的倒数
        self.weight_num = weight_num  # 权重数量
        self.matrix_list = []  # 初始化矩阵列表
        # 预计算权重
        self.weights = np.exp(-(np.arange(self.weight_num) / self.std_inv) ** 2)[::-1]

    def weighted_pts(self, matrix):
        # 检查矩阵是否全部为NaN
        if np.isnan(matrix).all():
            # 处理全为NaN的情况，这里可以根据需要进行修改
            # 可以返回一个特定的结果，或者继续使用当前状态
            raise ValueError("输入的矩阵不能全为NaN")

        self.matrix_list.append(matrix)

        # 如果矩阵数量超过weight_num，则截取最近的weight_num个矩阵
        if len(self.matrix_list) > self.weight_num:
            self.matrix_list = self.matrix_list[-self.weight_num:]

        # 获取当前有效矩阵的数量和相应的权重
        matrix_num = len(self.matrix_list)
        weights = self.weights[-matrix_num:]

        sum_R = np.zeros((3, 3))
        sum_t = np.zeros(3)
        total_weight = np.sum(weights)

        for matrix, weight in zip(self.matrix_list, weights):
            R = matrix[:3, :3]
            t = matrix[:3, 3]
            sum_R += R * weight
            sum_t += t * weight

        # 处理sum_R全为零的特殊情况，可能由于非有效的加权导致
        if np.isclose(sum_R, 0).all():
            raise ValueError("加权旋转矩阵计算结果异常，无法进行SVD分解")

        U, _, Vt = np.linalg.svd(sum_R)
        normalized_R = U @ Vt
        normalized_t = sum_t / total_weight

        result = np.eye(4)
        result[:3, :3] = normalized_R
        result[:3, 3] = normalized_t

        return result
    


parser = argparse.ArgumentParser()
script_dir = os.path.dirname(os.path.abspath(__file__))
parser.add_argument('--dir', type=str, default='nd_1')  # Change type to str
args = parser.parse_args()
data_dir = os.path.join(script_dir, "data", args.dir) 
masks_dir = os.path.join(data_dir, "ob_in_cam")
directory = masks_dir  # Replace with the actual directory path

# Get a list of all txt files in the directory
txt_files = [file for file in os.listdir(directory) if file.endswith('.txt')]

# Initialize the matrix variable
matrix = np.zeros((len(txt_files), 4, 4))

# Read data from each txt file and store it in the matrix
for i, file in enumerate(txt_files):
    file_path = os.path.join(directory, file)
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for j, line in enumerate(lines):
            values = line.strip().split()
            matrix[i, j] = [float(value) for value in values]

# Print the matrix
print(matrix)


gf = TransformationMatrixFilter(4,10)
sm=[]
z_ori=[]
z_filted=[]
for i in range(matrix.shape[0]):
    filted_matrix = gf.weighted_pts(matrix[i])
    sm.append(filted_matrix)
    z_ori.append(matrix[i][2][3])
    z_filted.append(filted_matrix[2][3])

    # Convert the filtered matrices to z-coordinates

print(111)
plt.figure()
plt.plot(z_ori, label='Original',marker='o')
plt.plot(z_filted, label='Filtered',marker='x')
plt.xlabel('Frame')
plt.ylabel('Z-coordinate')
plt.title('Original vs Filtered Z-coordinate')
plt.legend()
path=os.path.join(data_dir, "smooth.png")
plt.savefig(path)
plt.show()
print(222)

z_ori_mean = np.mean(z_ori)
z_ori_std = np.std(z_ori)
z_filted_mean = np.mean(z_filted)
z_filted_std = np.std(z_filted)

print("Original Z-coordinate:")
print("Mean:", z_ori_mean)
print("Standard Deviation:", z_ori_std)

print("Filtered Z-coordinate:")
print("Mean:", z_filted_mean)
print("Standard Deviation:", z_filted_std)

   

