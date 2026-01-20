from franky import Robot, CartesianMotion, Affine, ReferenceType, JointMotion, Gripper
import numpy as np
import time
import threading

T_TCP2FRANKATCP = [
    [0, 0, 1, 0],
    [0, -1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1.],
]

class FrankaEnv:
    def __init__(self, robot_ip: str = "172.16.0.3"):
        # Connect to the robot
        self.robot = Robot(robot_ip)
        self.robot.recover_from_errors()

        if impedance:
            # Set collision behavior
            self.robot.set_collision_behavior(
                lower_torque_threshold = [200.0] * 7,  # Nm
                upper_torque_threshold = [400.0] * 7,  # Nm
                lower_force_threshold = [100.0] * 6,  # N (linear) and Nm (angular)
                upper_force_threshold = [200.0] * 6,  # N (linear) and Nm (angular)
            )

            # # Set joint impedance
            self.robot.set_joint_impedance([30.0, 30.0, 30.0, 25.0, 25.0, 20.0, 20.0])  # Nm/rad # franky use joint impedance control insde
            self.robot.set_cartesian_impedance([3000.0, 3000.0, 3000.0, 300.0, 300.0, 300.0])  # N/m and Nm/rad])
        else:
            # Set collision behavior
            self.robot.set_collision_behavior(
                lower_torque_threshold = [20.0] * 7,  # Nm
                upper_torque_threshold = [40.0] * 7,  # Nm
                lower_force_threshold = [10.0] * 6,  # N (linear) and Nm (angular)
                upper_force_threshold = [20.0] * 6,  # N (linear) and Nm (angular)
            )

            # Set joint impedance
            self.robot.set_joint_impedance([3000.0, 3000.0, 3000.0, 2500.0, 2500.0, 2000.0, 2000.0])  # Nm/rad
            self.robot.set_cartesian_impedance([3000.0, 3000.0, 3000.0, 300.0, 300.0, 300.0])  # N/m and Nm/rad])

        # Reduce the acceleration and velocity dynamic
        self.robot.relative_dynamics_factor = 0.01

        self.gripper = Gripper(robot_ip)

        self.gripper_lock = threading.Lock()
        self.gripper_target_width = None

        self.gripper_speed = 0.2  # [m/s]
        self.gripper_force = 20.0  # [N]

        self.gripper_thread = None

    def _gripper_send_thread(self):
        while True:
            with self.gripper_lock:
                target_width = self.gripper_target_width

            if target_width is None:
                break

            print("Setting gripper width to:", target_width)

            self.gripper.move(target_width, speed=self.gripper_speed)


    def compute_observation(self):
        return {
            "Ttcp2base": self.robot.current_cartesian_state.pose.end_effector_pose.matrix @ np.array(T_TCP2FRANKATCP),
            "gripper_open": self.gripper.width * 2
        }

    def reset(self):
        # Go to initial position
        self.robot.move(JointMotion([0.0, 0.0, 0.0, -2.2, 0.0, 2.2, 0.7]), asynchronous=False)

        if self.gripper_thread is not None:
            with self.gripper_lock:
                self.gripper_target_width = None
            self.gripper_thread.join()

        self.gripper_target_width = 0.08
        self.gripper.move(self.gripper_target_width, speed=self.gripper_speed) # blocking move

        self.gripper_thread = threading.Thread(target=self._gripper_send_thread, daemon=True)
        self.gripper_thread.start()

        return self.compute_observation()

    def step(self, action: dict) -> dict:
        self.robot.move(CartesianMotion(Affine(action["Ttcp2base"] @ np.linalg.inv(np.array(T_TCP2FRANKATCP))), reference_type=ReferenceType.Absolute), asynchronous=True)

        with self.gripper_lock:
            self.gripper_target_width = action["gripper_open"]

        return self.compute_observation()

    def close(self):
        if self.gripper_thread is not None:
            with self.gripper_lock:
                self.gripper_target_width = None
            self.gripper_thread.join()
            self.gripper_thread = None


if __name__ == "__main__":
    env = FrankaEnv()
    obs = env.reset()
    last_timestamp = time.time()
    while True:
        action = {
            "Ttcp2base": obs["Ttcp2base"],
            "gripper_open": 0.0,
        }
        obs = env.step(action)
        print("gripper_open:", obs["gripper_open"])

        time.sleep(0.01)

        this_timestamp = time.time()
        print("fps:", 1.0 / (this_timestamp - last_timestamp))
        last_timestamp = this_timestamp
