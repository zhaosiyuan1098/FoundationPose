import pyrealsense2 as rs
import numpy as np
import cv2
import json
import time
import os

# 定义一个RealSenseRecorder类，用于录制和保存RealSense相机的数据
class RealSenseRecorder:
    def __init__(self, save_dir):
        self.save_dir = save_dir  # 保存数据的目录
        self.pipeline = rs.pipeline()  # 创建一个pipeline对象
        self.config = rs.config()  # 创建一个config对象，用于配置pipeline
        self.align_to = rs.stream.color  # 设置对齐到颜色流
        self.align = rs.align(self.align_to)  # 创建一个align对象，用于对齐深度和颜色帧
        self.RecordStream = False  # 是否开始录制的标志

    # 设置相机和流
    def setup(self):
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == "RGB Camera":
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)

        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

        if device_product_line == "L500":
            self.config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

        profile = self.pipeline.start(self.config)

        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: ", depth_scale)

        clipping_distance_in_meters = 1  # 1 meter
        self.clipping_distance = clipping_distance_in_meters / depth_scale

    # 创建保存数据的子文件夹
    def create_subfolders(self):
        subfolders = ["depth", "rgb", "depth_unaligned", "rgb_unaligned"]
        for subfolder in subfolders:
            path = os.path.join(self.save_dir, subfolder)
            if not os.path.exists(path):
                os.makedirs(path)

    # 录制和保存数据
    def record(self):
        try:
            while True:
                frames = self.pipeline.wait_for_frames()  # 等待获取一帧数据
                aligned_frames = self.align.process(frames)  # 对齐深度和颜色帧

                aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐后的深度帧
                color_frame = aligned_frames.get_color_frame()  # 获取对齐后的颜色帧

                unaligned_depth_frame = frames.get_depth_frame()  # 获取未对齐的深度帧
                unaligned_color_frame = frames.get_color_frame()  # 获取未对齐的颜色帧

                intrinsics = aligned_depth_frame.profile.as_video_stream_profile().intrinsics

                if not aligned_depth_frame or not color_frame:
                    continue

                depth_image = np.asanyarray(aligned_depth_frame.get_data())  # 将深度帧转换为numpy数组
                color_image = np.asanyarray(color_frame.get_data())  # 将颜色帧转换为numpy数组

                grey_color = 153
                depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
                bg_removed = np.where((depth_image_3d > self.clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

                unaligned_depth_image = np.asanyarray(unaligned_depth_frame.get_data())
                unaligned_color_image = np.asanyarray(unaligned_color_frame.get_data())

                # 保存图像和帧
                if self.RecordStream:
                    timestamp = time.time()
                    cv2.imwrite(os.path.join(self.save_dir, "depth", f"{timestamp}.png"), depth_image)
                    cv2.imwrite(os.path.join(self.save_dir, "rgb", f"{timestamp}.png"), color_image)
                    cv2.imwrite(os.path.join(self.save_dir, "depth_unaligned", f"{timestamp}.png"), unaligned_depth_image)
                    cv2.imwrite(os.path.join(self.save_dir, "rgb_unaligned", f"{timestamp}.png"), unaligned_color_image)

                # 显示图像
                cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
                cv2.imshow('Align Example', bg_removed)
                key = cv2.waitKey(1)

                # 's'键被按下 - 开始/停止录制
                if key == ord('s'):
                    self.RecordStream = not self.RecordStream

                # 'q'键被按下 - 退出
                elif key == ord('q'):
                    cv2.destroyAllWindows()
                    break

        finally:
            self.pipeline.stop()  # 停止pipeline

def main():
    # 定义保存数据的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, "data/202405211546")

    # 创建RealSenseRecorder类的实例
    recorder = RealSenseRecorder(save_dir)

    # 设置recorder
    recorder.setup()

    # 在保存目录中创建必要的子文件夹
    recorder.create_subfolders()

    # 开始录制
    recorder.record()

if __name__ == "__main__":
    main()