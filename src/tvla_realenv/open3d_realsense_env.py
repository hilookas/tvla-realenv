import open3d as o3d
import numpy as np

class Open3dRealsenseEnv:
    def __init__(self, serial: str=None):
        if not serial:
            print(o3d.t.io.RealSenseSensor.list_devices())
            assert False

        self.rs = o3d.t.io.RealSenseSensor()
        config = o3d.t.io.RealSenseSensorConfig({
            "serial": serial,
            "color_format": "RS2_FORMAT_RGB8",
            # "color_resolution": "640,480",     # L515配置为640×480工作模式
            "depth_format": "RS2_FORMAT_Z16",
            # "depth_resolution": "640,480",     # L515配置为640×480工作模式
            "fps": "30",
            "visual_preset": "RS2_L500_VISUAL_PRESET_MAX_RANGE"  # L515专用预设
        })
        self.rs.init_sensor(config, 0)
        self.rs.start_capture()

        intrinsic = self.rs.get_metadata().intrinsics.intrinsic_matrix
        depth_scale = self.rs.get_metadata().depth_scale

        # import atexit
        # atexit.register(self.close)

        self.meta_obs = {
            "size": [self.rs.get_metadata().height, self.rs.get_metadata().width],
            "intrinsic": intrinsic.tolist(),
            "depth_scale": depth_scale,
            "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
            "distortion_model": "none",
        }

    def compute_observation(self) -> dict:
        im_rgbd: o3d.t.geometry.RGBDImage = self.rs.capture_frame(True, True)
        # pcd: o3d.t.geometry.PointCloud = o3d.t.geometry.PointCloud.create_from_rgbd_image(im_rgbd, intrinsics=o3d.core.Tensor(self.intrinsic_matrix, dtype=o3d.core.Dtype.Float32), depth_scale=self.depth_scale)

        img_rgb = np.asarray(im_rgbd.color)
        img_depth = np.asarray(im_rgbd.depth)

        return {
            "rgb": img_rgb,
            "depth": img_depth,
        } | self.meta_obs

    def reset(self) -> dict:
        return self.compute_observation()

    def step(self, action=None) -> dict:
        return self.compute_observation()

    def close(self):
        self.rs.stop_capture()

if __name__ == "__main__":
    '''
    View Intel Realsense D405 pointcloud in Open3D viewer
    Src: https://github.com/isl-org/Open3D/issues/6221
    '''

    # from o3d_vis import Open3dVisualizer

    rs_env = Open3dRealsenseEnv("f1471193")
    # o3d_vis = Open3dVisualizer()

    try:
        while True:
            rs_obs = rs_env.step()
            print(rs_obs)
            # rs_obs |= {
            #     "servo_angle": [1, 0, 0, 0, 0, 0]
            # }
            # o3d_vis.render(rs_obs, None)

    finally:
        rs_env.close()
