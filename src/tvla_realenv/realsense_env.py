import json
import numpy as np
import cv2
import pyrealsense2 as rs

def list_all_realsense_cameras():
    """枚举所有已连接的RealSense相机，返回设备信息列表"""
    ctx = rs.context()
    devices = ctx.query_devices()
    camera_list = []

    if len(devices) == 0:
        print("未检测到任何RealSense相机！")
        return camera_list

    print(f"共检测到 {len(devices)} 台RealSense相机：")
    for i, dev in enumerate(devices):
        dev_info = {
            "index": i,
            "name": dev.get_info(rs.camera_info.name),
            "serial_number": dev.get_info(rs.camera_info.serial_number),
            "firmware_version": dev.get_info(rs.camera_info.firmware_version),
            "product_line": dev.get_info(rs.camera_info.product_line)
        }
        camera_list.append(dev_info)
        print(f"[{i}] 型号: {dev_info['name']} | 序列号: {dev_info['serial_number']} | 固件: {dev_info['firmware_version']}")
    return camera_list

class RealSenseEnv:
    def __init__(self, serial_number=None):
        self.pipeline = rs.pipeline()

        config = rs.config()
        if serial_number:
            config.enable_device(serial_number)
        else:
            list_all_realsense_cameras()
            assert False
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        self.pipeline.start(config)

        profile = self.pipeline.get_active_profile()

        depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        depth_intrinsics = depth_profile.get_intrinsics()

        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        color_intrinsics = color_profile.get_intrinsics()

        self.align = rs.align(rs.stream.color)

        self.meta_obs = {
            "size": [color_intrinsics.height, color_intrinsics.width],
            "intrinsic": [
                [color_intrinsics.fx, 0.0, color_intrinsics.ppx],
                [0.0, color_intrinsics.fy, color_intrinsics.ppy],
                [0.0, 0.0, 1.0],
            ],
            "distortion": np.array(color_intrinsics.coeffs).tolist(),
            "distortion_model": str(color_intrinsics.model)
        }

    def compute_observation(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return {
            "rgb": color_image[:,:,::-1], # bgr2rgb
            "depth": depth_image,
        } | self.meta_obs

    def reset(self):
        return self.compute_observation()

    def step(self, _):
        return self.compute_observation()

    def close(self):
        self.pipeline.stop()


if __name__ == "__main__":
    env = RealSenseEnv()
    try:
        obs = env.reset()
        with open("cam_intrinsics.json", "w") as f:
            json.dump(env.meta_obs, f, indent=4)
        for _ in range(1000):
            print(obs)
            obs = env.step(obs)

            cv2.imshow("Capture_Video", obs["rgb"][:,:,::-1])
            cv2.waitKey(1)
    finally:
        env.close()
