import numpy as np
import torch
import cv2
import sys
import os
from segment_anything import sam_model_registry, SamPredictor

import matplotlib.pyplot as plt

class SAMInference:
    def __init__(self, sam_checkpoint, data_dir):
        self.sam_checkpoint = sam_checkpoint
        self.data_dir = data_dir
        self.masks_dir = os.path.join(self.data_dir, "masks")  # 设置保存掩码图像的目录路径
        self.rgb_dir = os.path.join(self.data_dir, "rgb")  # 设置保存RGB图像的目录路径
        self.model_type = "vit_h"  # 设置模型类型
        self.device = "cuda"  # 设置设备类型
        self.sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)  # 根据模型类型和检查点路径创建SAM模型
        self.sam.to(device=self.device)  # 将SAM模型移动到指定设备上
        self.predictor = SamPredictor(self.sam)  # 创建SAM预测器对象

    def load_image(self, image_path):
        image = cv2.imread(image_path)  # 读取图像
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将图像从BGR格式转换为RGB格式
        self.predictor.set_image(image)  # 设置预测器的图像
        return image

    def predict(self, input_point, input_label):
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,  # 输入点的坐标
            point_labels=input_label,  # 输入点的标签
            multimask_output=True,  # 是否输出多个掩码
        )
        return masks, scores

    def save_masks(self, masks):
        for i, mask in enumerate(masks):
            binary_mask = (mask > 0.5).astype(np.uint8) * 255  # 将掩码转换为二值图像
            _, binary_image = cv2.threshold(binary_mask, 128, 255, cv2.THRESH_BINARY)  # 对二值图像进行阈值处理
            cv2.imshow("Binary Image", binary_image)  # 显示二值图像
            key = cv2.waitKey(0)  # 等待按键输入
            cv2.destroyAllWindows()  # 关闭窗口

            if key == ord('s'):  # 如果按下's'键
                cv2.imwrite(os.path.join(self.masks_dir, f'mask_{i+1}.png'), binary_mask)  # 保存掩码图像

def main():
    sam_checkpoint = "/home/siyuan/code/segment-anything/sam_vit_h_4b8939.pth"  # SAM模型的检查点路径
    script_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本所在的目录路径
    data_dir = os.path.join(script_dir, "data/mouse_1280")  # 数据目录路径
    masks_dir = os.path.join(data_dir, "masks")  # 保存掩码图像的目录路径
    rgb_dir = os.path.join(data_dir, "rgb")  # 保存RGB图像的目录路径

    # 初始化SAMInference对象
    sam_inference = SAMInference(sam_checkpoint, data_dir)

    # 加载图像
    image_files = sorted(os.listdir(rgb_dir), key=lambda x: os.path.getctime(os.path.join(rgb_dir, x)))  # 获取RGB图像文件列表，并按创建时间排序
    image_path = os.path.join(rgb_dir, image_files[0])  # 获取第一张图像的路径
    image = sam_inference.load_image(image_path)  # 加载图像

    # 定义输入点和标签
    input_point = np.array([[550,400]])  # 输入点的坐标
    input_label = np.array([1])  # 输入点的标签

    # 预测
    masks, scores = sam_inference.predict(input_point, input_label)

    # 保存掩码图像
    sam_inference.save_masks(masks)

if __name__ == "__main__":
    main()
