import os
import sys
import trimesh
import numpy as np
import cv2
import imageio
import numpy as np
from estimater import *
from datareader import *

import matplotlib.pyplot as plt

sys.path.append('/home/siyuan/code/FoundationPose')


class PoseEstimation:
    def __init__(self, est_refine_iter=5, track_refine_iter=2, debug=1, debug_dir="debug"):
        # 初始化函数，设置一些参数和路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(script_dir, "data/mouse_stand_1280")  # 数据目录
        self.mesh_file = os.path.join(self.data_dir, "mesh/scaled_down_file.obj")  # 网格文件路径
        self.est_refine_iter = est_refine_iter  # 估计迭代次数
        self.track_refine_iter = track_refine_iter  # 跟踪迭代次数
        self.debug = debug  # 是否开启调试模式
        self.debug_dir = os.path.join(self.data_dir, debug_dir)  # 调试输出目录
        self.mesh = trimesh.load(self.mesh_file)  # 加载网格模型
        self.scorer = ScorePredictor()  # 分数预测器
        self.refiner = PoseRefinePredictor()  # 姿态细化预测器
        self.glctx = dr.RasterizeCudaContext()  # CUDA上下文
        self.est = FoundationPose(model_pts=self.mesh.vertices, model_normals=self.mesh.vertex_normals, mesh=self.mesh, scorer=self.scorer, refiner=self.refiner, debug_dir=self.debug_dir, debug=self.debug, glctx=self.glctx)  # 基础姿态估计器
        self.reader = YcbineoatReader(video_dir=self.data_dir, shorter_side=None, zfar=np.inf)  # 数据读取器
        self.to_origin, extents = trimesh.bounds.oriented_bounds(self.mesh)  # 获取网格模型的边界框
        self.bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)  # 边界框坐标

    def run_demo(self):
        # 运行演示函数
        for i in range(len(self.reader.color_files)):
            color = self.reader.get_color(i)  # 获取彩色图像
            depth = self.reader.get_depth(i)  # 获取深度图像
            if i == 0:
                mask = self.reader.get_mask(0).astype(bool)  # 获取物体掩码
                pose = self.est.register(K=self.reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=self.est_refine_iter)  # 注册物体姿态
            else:
                pose = self.est.track_one(rgb=color, depth=depth, K=self.reader.K, iteration=self.track_refine_iter)  # 跟踪物体姿态

            os.makedirs(f'{self.reader.video_dir}/ob_in_cam', exist_ok=True)  # 创建输出目录
            np.savetxt(f'{self.reader.video_dir}/ob_in_cam/{self.reader.id_strs[i]}.txt', pose.reshape(4,4))  # 保存姿态矩阵

            center_pose = pose@np.linalg.inv(self.to_origin)  # 将姿态矩阵转换到原点坐标系下
            vis = draw_posed_3d_box(self.reader.K, img=color, ob_in_cam=center_pose, bbox=self.bbox)  # 绘制3D框
            vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=self.reader.K, thickness=3, transparency=0, is_input_rgb=True)  # 绘制坐标轴
            cv2.imshow('1', vis[...,::-1])  # 显示图像
            cv2.waitKey(1)

            # 打印旋转和平移矩阵
            print("Rotation and translation matrix for frame", i, ":")
            print(np.array(pose))

    def estimate_pose(self, image, iteration=5):
        # 估计姿态函数
        color = image
        depth = self.reader.get_depth(0)  # 获取深度图像
        mask = self.reader.get_mask(0).astype(bool)  # 获取物体掩码
        pose = self.est.register(K=self.reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=iteration)  # 注册物体姿态
        return pose

def main():
    demo = PoseEstimation()
    demo.run_demo()

if __name__ == "__main__":
    main()
