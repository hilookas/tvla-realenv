from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e, rm_peripheral_read_write_params_t
import numpy as np

def T_from_realman_xyzrpy(xyzrpy):
    x, y, z, rx, ry, rz = xyzrpy

    T = np.eye(4)
    Rx = np.array([[1, 0, 0],
                [0, np.cos(rx), -np.sin(rx)],
                [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                [0, 1, 0],
                [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                [np.sin(rz), np.cos(rz), 0],
                [0, 0, 1]])
    T[:3, :3] = Rz@Ry@Rx  # 先绕 x轴旋转 再绕y轴旋转  最后绕z轴旋转
    T[:3, 3] = [x, y, z]

    return T

def realman_xyzrpy_from_T(T):
    x = T[0, 3]
    y = T[1, 3]
    z = T[2, 3]

    ry = np.arcsin(-T[2, 0])
    if np.cos(ry) != 0:
        rx = np.arctan2(T[2, 1]/np.cos(ry), T[2, 2]/np.cos(ry))
        rz = np.arctan2(T[1, 0]/np.cos(ry), T[0, 0]/np.cos(ry))
    else:
        rx = 0
        rz = np.arctan2(-T[0, 1], T[1, 1])

    return np.array([x, y, z, rx, ry, rz])

def realman_gripper_value_from_width(width: float) -> int:
    return int(9000 - int(width * 1e5))


def width_from_realman_gripper_value(gripper_value: int) -> float:
    return (9000 - gripper_value) * 1e-5

from pytransform3d.transformations import transform_from
from pytransform3d.rotations import active_matrix_from_angle

T_TCP2REALMANEEF = transform_from(
    active_matrix_from_angle(2, -np.pi / 3) @ np.array([
        [0, 0, 1],
        [0, -1, 0],
        [1, 0, 0],
    ]),
    np.array([0, 0, 0.225])
)

class RealmanEnv:
    def __init__(self, robot_ip: str = "192.168.101.19"):
        # 实例化RoboticArm类
        self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)

        # 创建机械臂连接，打印连接id
        handle = self.arm.rm_create_robot_arm(robot_ip, 8080)
        assert handle.id > 0, "机械臂连接失败，检查ip是否正确"

        ret = self.arm.rm_set_modbus_mode(1,115200,2)
        assert ret == 0, "机械臂modbus设置失败"

        param = rm_peripheral_read_write_params_t(1, 260, 1)
        ret = self.arm.rm_write_single_register(param, 100)
        assert ret == 0, "写夹爪运动速度失败"

    def get_gripper(self):
        param = rm_peripheral_read_write_params_t(1, 258, 1)
        ret, _ = self.arm.rm_read_holding_registers(param)

        param = rm_peripheral_read_write_params_t(1, 259, 1)
        ret, gripper_value_state = self.arm.rm_read_holding_registers(param) # gripper_value 0~9000

        return width_from_realman_gripper_value(gripper_value_state)

    def set_gripper(self, gripper_open):
        gripper_value_cmd = realman_gripper_value_from_width(gripper_open)

        # 设置夹爪目标位置
        param = rm_peripheral_read_write_params_t(1, 258, 1, 2)
        ret = self.arm.rm_write_registers(param, [0, gripper_value_cmd, 0, 0])
        assert ret == 0

        # 执行
        param = rm_peripheral_read_write_params_t(1, 264, 1)
        ret = self.arm.rm_write_single_register(param, 1)
        assert ret == 0

    def compute_observation(self):
        ret, state = self.arm.rm_get_current_arm_state()
        assert ret == 0, "获取机械臂状态失败"

        return {
            "Ttcp2base": T_from_realman_xyzrpy(state["pose"]) @ T_TCP2REALMANEEF,
            "gripper_open": self.get_gripper()
        }

    def reset(self):
        target_joints = np.array([90,0,0,-90,0,-90,60])
        target_gripper = 0.09

        while True:
            ret = self.arm.rm_movej_follow(target_joints)
            assert ret == 0
            ret, state = self.arm.rm_get_current_arm_state()
            assert ret == 0, "获取机械臂状态失败"

            self.set_gripper(target_gripper)

            err = np.linalg.norm(state["joint"] - target_joints)
            err_gripper = abs(self.get_gripper() - target_gripper)
            if err < 0.01 and err_gripper < 0.001:
                break

            print(f"waiting for reset... joint_err: {err}, gripper_err: {err_gripper}")
        return self.compute_observation()

    def step(self, action: dict) -> dict:
        pose_target = realman_xyzrpy_from_T(action["Ttcp2base"] @ np.linalg.inv(T_TCP2REALMANEEF))
        self.arm.rm_movep_follow(pose_target)

        self.set_gripper(action["gripper_open"])

        return self.compute_observation()

    def close(self):
        self.arm.rm_delete_robot_arm()

if __name__ == "__main__":
    # Test
    assert np.allclose(
        realman_xyzrpy_from_T(T_from_realman_xyzrpy([0.1, 0.2, 0.3, 0.1, 0.2, 0.3])),
        [0.1, 0.2, 0.3, 0.1, 0.2, 0.3]
    )

    assert np.allclose(width_from_realman_gripper_value(realman_gripper_value_from_width(0.09)), 0.09)

    env = RealmanEnv()
    try:
        obs = env.reset()
        for _ in range(1000):
            # print(obs)
            # {'Ttcp2base': array([[-8.66028465e-01, -4.99993018e-01, -1.29631160e-03,
            #         -1.20000000e-04],
            #        [-4.99994698e-01,  8.66025554e-01,  2.24530929e-03,
            #         -2.10030000e-01],
            #        [ 0.00000000e+00,  2.59265069e-03, -9.99996639e-01,
            #          3.52504000e-01],
            #        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
            #          1.00000000e+00]]), 'gripper_open': 0.09000000000000001}
            print(obs["Ttcp2base"][:3, 3])
            print(obs["Ttcp2base"][:3, :3])

            obs["gripper_open"] = 0
            obs = env.step(obs)
    finally:
        env.close()
