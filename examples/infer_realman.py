from tvla_realenv.open3d_realsense_env import Open3dRealsenseEnv
from tvla_realenv.realman_env import RealmanEnv
import cv2
import copy
from pathlib import Path
from datetime import datetime
import json
import numpy as np

# Copy from: https://github.com/NVlabs/FoundationPose/blob/main/Utils.py

def draw_xyz_axis(color, ob_in_cam, K=np.eye(3), scale=0.1, thickness=3, transparency=0, is_input_rgb=True, save_path=None):
    """
    Draw XYZ coordinate axes on an image.

    Args:
        color: Input image (RGB or BGR)
        ob_in_cam: Object pose in camera frame (4x4 transformation matrix)
        scale: Scale factor for axis length
        K: Camera intrinsic matrix (3x3)
        thickness: Line thickness for drawing
        transparency: Transparency factor (0-1)
        is_input_rgb: Whether input is RGB (True) or BGR (False)
        save_path: Optional path to save the result image

    Returns:
        Image with XYZ axes drawn
    """
    import cv2

    def project_3d_to_2d(pt, K, ob_in_cam):
        """Project 3D point to 2D image coordinates."""
        pt = pt.reshape(4, 1)
        projected = K @ ((ob_in_cam@pt)[:3,:])
        projected = projected.reshape(-1)
        projected = projected / projected[2]
        return projected.reshape(-1)[:2].round().astype(int)

    # Convert RGB to BGR if needed (OpenCV uses BGR)
    if is_input_rgb:
        color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    xx = np.array([1,0,0,1]).astype(float)
    yy = np.array([0,1,0,1]).astype(float)
    zz = np.array([0,0,1,1]).astype(float)
    xx[:3] = xx[:3]*scale
    yy[:3] = yy[:3]*scale
    zz[:3] = zz[:3]*scale
    origin = tuple(project_3d_to_2d(np.array([0,0,0,1]), K, ob_in_cam))
    xx = tuple(project_3d_to_2d(xx, K, ob_in_cam))
    yy = tuple(project_3d_to_2d(yy, K, ob_in_cam))
    zz = tuple(project_3d_to_2d(zz, K, ob_in_cam))
    line_type = cv2.LINE_AA
    arrow_len = 0
    tmp = color.copy()
    tmp1 = tmp.copy()
    tmp1 = cv2.arrowedLine(tmp1, origin, xx, color=(0,0,255), thickness=thickness, line_type=line_type, tipLength=arrow_len)
    mask = np.linalg.norm(tmp1-tmp, axis=-1)>0
    tmp[mask] = tmp[mask]*transparency + tmp1[mask]*(1-transparency)
    tmp1 = tmp.copy()
    tmp1 = cv2.arrowedLine(tmp1, origin, yy, color=(0,255,0), thickness=thickness, line_type=line_type, tipLength=arrow_len)
    mask = np.linalg.norm(tmp1-tmp, axis=-1)>0
    tmp[mask] = tmp[mask]*transparency + tmp1[mask]*(1-transparency)
    tmp1 = tmp.copy()
    tmp1 = cv2.arrowedLine(tmp1, origin, zz, color=(255,0,0), thickness=thickness, line_type=line_type, tipLength=arrow_len)
    mask = np.linalg.norm(tmp1-tmp, axis=-1)>0
    tmp[mask] = tmp[mask]*transparency + tmp1[mask]*(1-transparency)
    tmp = tmp.astype(np.uint8)

    if save_path:
        cv2.imwrite(save_path, tmp)

    # Convert back to RGB if input was RGB
    if is_input_rgb:
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)

    return tmp

if __name__ == "__main__":
    with open("data/20260115_190746/camera_results.json", "r") as f:
        cam_results = json.load(f)

    cv2.imshow("color", np.eye(100,100,3).astype(np.uint8)*255)
    clicked_point = None
    clicked_point_new_clicked = False
    def on_mouse(event, u, v, flags, param):
        # if clicked, goto the position
        if event == cv2.EVENT_LBUTTONDOWN:
            global clicked_point, clicked_point_new_clicked
            clicked_point = (u, v)
            clicked_point_new_clicked = True
    cv2.setMouseCallback("color", on_mouse)

    env = RealmanEnv("192.168.101.19")
    # env = RealmanEnv("192.168.101.20")
    rs_env = Open3dRealsenseEnv("f1471193")
    try:
        obs = env.reset()
        obs |= rs_env.reset()
        action = {
            "Ttcp2base": obs["Ttcp2base"],
            "gripper_open": obs["gripper_open"],
        }
        disable_robot = False
        while True:
            if not disable_robot:
                obs = env.step(action)
            else:
                obs = env.compute_observation()
            obs |= rs_env.step(action)

            debug_img = obs["rgb"][:,:,::-1]
            # import ipdb; ipdb.set_trace()
            print(obs["Ttcp2base"])
            # debug_img = draw_xyz_axis(debug_img, np.linalg.inv(np.array(cam_results["Tcam2base"])), K=np.array(rs_env.meta_obs["intrinsic"]))

            debug_img = draw_xyz_axis(debug_img, np.linalg.inv(np.array(cam_results["Tcam2base"])) @ obs["Ttcp2base"], K=np.array(rs_env.meta_obs["intrinsic"]), is_input_rgb=False)

            if clicked_point is not None:
                debug_img = cv2.circle(debug_img, clicked_point, 5, (0, 0, 255), -1)

            cv2.imshow("color", debug_img)
            k = cv2.waitKey(1)

            if clicked_point_new_clicked:
                clicked_point_new_clicked = False

                print("Selected center point:", clicked_point)
                u, v = clicked_point
                d = obs["depth"][v, u] / rs_env.meta_obs["depth_scale"]

                x, y, z = np.linalg.inv(np.array(rs_env.meta_obs["intrinsic"])) @ (np.array([u, v, 1]) * d)

                x, y, z, _ = np.array(cam_results["Tcam2base"]) @ np.array([x, y, z, 1])

                z += 0.030 # z+3cm offset

                new_Ttcp2base = copy.deepcopy(obs["Ttcp2base"])
                new_Ttcp2base[:3, 3] = np.array([x, y, z])

                action = {
                    "Ttcp2base": new_Ttcp2base,
                    "gripper_open": obs["gripper_open"],
                }

                # import ipdb; ipdb.set_trace()

            if k == ord('c'):
                action = {
                    "Ttcp2base": obs["Ttcp2base"],
                    "gripper_open": obs["gripper_open"],
                }
                disable_robot = False
                print("Enable robot movement")

                action["gripper_open"] = 0.00
                print("Closed gripper")
            elif k == ord('o'):
                action = {
                    "Ttcp2base": obs["Ttcp2base"],
                    "gripper_open": obs["gripper_open"],
                }
                disable_robot = False
                print("Enable robot movement")

                action["gripper_open"] = 0.09
                print("Open gripper")
            elif k == ord('e'):
                action = {
                    "Ttcp2base": obs["Ttcp2base"],
                    "gripper_open": obs["gripper_open"],
                }
                disable_robot = False
                print("Enable robot movement")
            elif k == ord('d'):
                disable_robot = True
                print("Disable robot movement")
            elif k == ord('q'):
                break
    finally:
        env.close()
        rs_env.close()
