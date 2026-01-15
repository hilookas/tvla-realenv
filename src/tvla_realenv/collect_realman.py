from tvla_realenv.open3d_realsense_env import Open3dRealsenseEnv
from tvla_realenv.realman_env import RealmanEnv
import cv2
import copy
from pathlib import Path
from datetime import datetime
import json

if __name__ == "__main__":
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
        count = 0
        calib_dir = Path("data") / datetime.now().strftime('%Y%m%d_%H%M%S')
        Path(calib_dir).mkdir(parents=True, exist_ok=True)
        with open(calib_dir / "cam_intrinsic.json", "w") as f:
            json.dump(rs_env.meta_obs, f, indent=4)
        while True:
            if not disable_robot:
                obs = env.step(action)
            else:
                obs = env.compute_observation()
            obs |= rs_env.step(action)

            cv2.imshow("Capture_Video", obs["rgb"][:,:,::-1])
            k = cv2.waitKey(1)

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
            elif k == ord('s'):
                with open(calib_dir / "Ttcp2bases.jsonl", 'a+') as f:
                    f.write(f'{obs["Ttcp2base"].tolist()}\n')
                cv2.imwrite(calib_dir / f"{count:04d}.jpg", obs["rgb"][:,:,::-1])
                print(f"Saving {count:04d}.jpg")
                count += 1
            elif k == ord('q'):
                break
    finally:
        env.close()
        rs_env.close()
