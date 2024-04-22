import cv2
import numpy as np
import pyrealsense2 as rs
import os

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

i = 0
try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

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