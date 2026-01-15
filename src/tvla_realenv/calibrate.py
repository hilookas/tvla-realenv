import argparse
from pathlib import Path
from glob import glob
import numpy as np
import cv2

def main():
    parser = argparse.ArgumentParser(description="Hand-Eye Calibration")
    parser.add_argument("--calib_dir", type=str, default="data/20260115_175549",
                           help="Path to the calibration data directory")
    args = parser.parse_args()

    calib_dir = Path(args.calib_dir)

    with open(calib_dir / "Ttcp2bases.jsonl", "r") as f:
        import json
        Ttcp2bases = [json.loads(line) for line in f.readlines()]
    Ttcp2bases = np.array(Ttcp2bases)

    img_paths = sorted(glob("*.jpg", root_dir=calib_dir), key=lambda x: int(Path(x).stem))

    img_size = None
    obj_points = []
    img_points = []
    Tbase2tcps = []

    XX = 6 #标定板的中长度对应的角点的个数
    YY = 4  #标定板的中宽度对应的角点的个数
    L = 0.02 #标定板一格的长度  单位为米

    objp = np.zeros((XX * YY, 3), np.float32)
    objp[:, :2] = np.mgrid[0:XX, 0:YY].T.reshape(-1, 2) * L     # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y

    for img_path, Ttcp2base in zip(img_paths, Ttcp2bases):
        img_bgr = cv2.imread(calib_dir / img_path)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        img_size = img_gray.shape

        ret, corners = cv2.findChessboardCorners(img_gray, (XX, YY), None)
        Path(calib_dir / "debug").mkdir(exist_ok=True)
        cv2.imwrite(str(calib_dir / "debug" / f"{Path(img_path).stem}.jpg"), cv2.drawChessboardCorners(img_bgr, (XX, YY), corners, ret))

        if ret:
            corners2 = cv2.cornerSubPix(img_gray, corners, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001))  # 在原角点的基础上寻找亚像素角点
            assert [corners2]
            corners2 = corners2.squeeze(1) # (24, 2)

            obj_points.append(objp)
            img_points.append(corners2)
            Tbase2tcps.append(np.linalg.inv(Ttcp2base))

    obj_points = np.array(obj_points)  # (N, 24, 3)
    img_points = np.array(img_points)  # (N, 24, 2)
    Tbase2tcps = np.array(Tbase2tcps)

    with open(calib_dir / "cam_intrinsic.json", "r") as f:
        cam_intrinsic = json.load(f)

    # 标定,得到图案在相机坐标系下的位姿
    ret, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_size[::-1],
        # None, None,

        np.array(cam_intrinsic["intrinsic"]), np.array(cam_intrinsic["distortion"]),
        flags=(cv2.CALIB_USE_INTRINSIC_GUESS |
         cv2.CALIB_FIX_PRINCIPAL_POINT |
         cv2.CALIB_FIX_FOCAL_LENGTH |
         cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 |
         cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6 |
         cv2.CALIB_ZERO_TANGENT_DIST),
    )
    print(np.array(cam_intrinsic["intrinsic"]))
    print(camera_matrix)
    # import ipdb; ipdb.set_trace()

    # void cv::calibrateHandEye(
    #     InputArrayOfArrays R_gripper2base, // hand2eye: base2eef
    #     InputArrayOfArrays t_gripper2base,
    #     InputArrayOfArrays R_target2cam,
    #     InputArrayOfArrays t_target2cam,
    #     OutputArray R_cam2gripper, // hand2eye: cam2base
    #     OutputArray t_cam2gripper,
    #     HandEyeCalibrationMethod method = CALIB_HAND_EYE_TSAI
    # )

    R, t = cv2.calibrateHandEye(Tbase2tcps[:, :3, :3], Tbase2tcps[:, :3, 3], rvecs, tvecs, cv2.CALIB_HAND_EYE_TSAI) # eef2base target2cam -> cam2base?
    Tcam2base = np.eye(4)
    Tcam2base[:3, :3] = R
    Tcam2base[:3, 3] = t.squeeze(1)

    with open(calib_dir / "camera_results.json", "w") as f:
        import json
        json.dump({
            "Tcam2base": Tcam2base.tolist(),
            "camera_matrix": camera_matrix.tolist(),
            "distortion": distortion.tolist(),
            "img_size": img_size,
        }, f, indent=4)

if __name__ == "__main__":
    main()
