import cv2
import numpy as np
import pyrealsense2 as rs
import os
import yaml

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

depth_dir="./data/realsense/depth"
rgb_dir="./data/realsense/rgb"

os.makedirs(depth_dir, exist_ok=True)
os.makedirs(rgb_dir, exist_ok=True)

i = 0
try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            print("No frames received")
            continue

        # Read camera intrinsics
        if i == 0:  # Only read intrinsics on the first frame
            depth_sensor = pipeline.get_active_profile().get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()
            intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

            # Create YAML file
            yaml_data = {
                'image_width': intrinsics.width,
                'image_height': intrinsics.height,
                'camera_name': 'realsense',
                'camera_matrix': {
                    'rows': 3,
                    'cols': 3,
                    'data': [
        intrinsics.fx, 0, intrinsics.ppx,
        0, intrinsics.fy, intrinsics.ppy,
        0, 0, 1
    ]
                },
                'distortion_model': 'plumb_bob',
                'distortion_coefficients': {
                    'rows': 1,
                    'cols': 5,
                    'data': intrinsics.coeffs
                },
                'depth_scale': depth_scale
            }

            # Save YAML file
            with open('./data/realsense/camera_intrinsics.yaml', 'w') as file:
                yaml.dump(yaml_data, file)

        # Convert depth frame to a numpy array
        depth_image = np.asanyarray(depth_frame.get_data())
        cv2.imwrite(os.path.join(depth_dir, '{}.png'.format(i)), depth_image)

        # Convert color frame to a numpy array
        color_image = np.asanyarray(color_frame.get_data())
        cv2.imwrite(os.path.join(rgb_dir, '{}.png'.format(i)), color_image)

        # Display the color and depth images (optional)
        cv2.imshow('Color Image', color_image)
        cv2.imshow('Depth Image', depth_image)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        i += 1

finally:
    pipeline.stop()
    cv2.destroyAllWindows()