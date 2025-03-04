import time
import math

from allegro_hand.controller import AllegroController
from xarm.wrapper import XArmAPI

import numpy as np

# pregrasp pose
# XARM_HOME_VALUES = [-40.0, 45.0, -75.0, 70.0, 100.0, -160.0]

# stable initial pose
XARM_HOME_VALUES = [0.0, -45.0, -45.0, 0.0, 45.0, -90.0]

# XARM_HOME_VALUES = [0.0, -45.0, 0.0, 0.0, 0.0, -90.0]

# ALLEGRO_HOME_VALUES = [
#     0.0,
#     -0.17453293,
#     0.78539816,
#     0.78539816,
#     0.0,
#     -0.17453293,
#     0.78539816,
#     0.78539816,
#     0.08726646,
#     -0.08726646,
#     0.87266463,
#     0.78539816,
#     1.04719755,
#     0.43633231,
#     0.26179939,
#     0.78539816,
# ]


ALLEGRO_HOME_VALUES = [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]


# XARM_HOME_VALUES = [
#     0.54522133,
#     -0.12438324,
#     -1.46864963,
#     -0.23138203,
#     1.14474761,
#     -0.21687701,
# ]

# ALLEGRO_HOME_VALUES = [
#     -0.32651427,
#     0.85937423,
#     0.84576434,
#     0.71720642,
#     0.11098903,
#     0.62952811,
#     0.75782716,
#     0.70635951,
#     0.46350813,
#     0.52595007,
#     0.71539563,
#     0.69929093,
#     1.15112174,
#     0.90769601,
#     0.71409345,
#     0.75540137,
# ]

# array([ 0.54522133, -0.12438324, -1.46864963, -0.23138203,  1.14474761,
#        -0.21687701, -0.32651427,  0.85937423,  0.84576434,  0.71720642,
#         0.11098903,  0.62952811,  0.75782716,  0.70635951,  0.46350813,
#         0.52595007,  0.71539563,  0.69929093,  1.15112174,  0.90769601,
#         0.71409345,  0.75540137])


class DexArmControl:
    def __init__(self, xarm_ip_address="192.168.1.221"):
        # try:
        #     rospy.init_node("dex_arm")
        # except:
        #     pass

        self.allegro = AllegroController()
        self.arm = XArmAPI(xarm_ip_address, report_type="devlop")

        self.max_hand_joint_vel = 100.0 / 360.0 * 2 * math.pi  # 100 degree / sec
        self.last_xarm_command = None
        self.last_allegro_command = None

        self.arm.motion_enable(enable=False)
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)

        self.reset()

        print("init complete")

    def reset(self):
        if self.arm.has_err_warn:
            self.arm.clean_error()

        self.arm.motion_enable(enable=False)
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)  # 0: position control, 1: servo control
        self.arm.set_state(state=0)

        self.home_robot()

    def set_mode(self, mode=0):
        if self.arm.has_err_warn:
            self.arm.clean_error()

        self.arm.set_mode(mode)
        self.arm.set_state(state=0)

    def home_robot(self):
        # Homing the XArm6
        self.arm.set_servo_angle(angle=XARM_HOME_VALUES, wait=True, is_radian=False)

        # Homing the Allegro Hand
        # self.move_hand(allegro_angles=np.array(ALLEGRO_HOME_VALUES), interpolate=True)

        # self.allegro.hand_pose(ALLEGRO_HOME_VALUES)

    def move_robot(
        self,
        allegro_angles,
        xarm_angles,
        arm_control_mode="position",
        arm_relative=False,
        wait=False,
    ):
        t = time.time()

        print("before control :{}".format(t))

        self.move_hand(allegro_angles, interpolate=True)

        for _ in range(1):
            if arm_control_mode == "position":
                xarm_pose = xarm_angles.copy()
                xarm_pose[:3] = xarm_pose[:3] * 1000

                self.arm.set_position(
                    x=xarm_pose[0],
                    y=xarm_pose[1],
                    z=xarm_pose[2],
                    roll=xarm_pose[3],
                    pitch=xarm_pose[4],
                    yaw=xarm_pose[5],
                    relative=arm_relative,
                    wait=wait,
                    is_radian=True,
                )

            elif arm_control_mode == "position_aa":
                xarm_pose = xarm_angles.copy()
                xarm_pose[:3] = xarm_pose[:3] * 1000

                self.arm.set_position_aa(
                    axis_angle_pose=xarm_pose,
                    relative=arm_relative,
                    wait=wait,
                    is_radian=True,
                )

            else:
                assert arm_control_mode == "servo_angle"

                self.arm.set_servo_angle(
                    angle=xarm_angles, relative=arm_relative, wait=wait, is_radian=True
                )

        self.move_hand(allegro_angles, interpolate=False)

        self.last_xarm_command = xarm_angles
        self.last_allegro_command = allegro_angles

        print("after control :{}".format(t))
        print("slow control fps :{}".format(1 / (time.time() - t)))

    def move_robot_servo(
        self,
        allegro_angles,
        xarm_angles,
        arm_control_mode="servo_angle_j",
        arm_relative=False,
    ):
        t = time.time()

        print("before control :{}".format(t))


        self.move_hand(allegro_angles, interpolate=True)

        for _ in range(1):
            if arm_control_mode == "servo_angle_j":
                assert not arm_relative
                self.arm.set_servo_angle_j(angles=xarm_angles, is_radian=True)

            elif arm_control_mode == "servo_cartesian":
                xarm_pose = xarm_angles.copy()
                xarm_pose[:3] = xarm_pose[:3] * 1000

                self.arm.set_servo_cartesian(
                    mvpose=xarm_pose, is_radian=True, relative=arm_relative
                )

            else:
                assert arm_control_mode == "servo_cartesian_aa"
                xarm_pose = xarm_angles.copy()

                xarm_pose[:3] = xarm_pose[:3] * 1000

                self.arm.set_servo_cartesian_aa(
                    axis_angle_pose=xarm_pose, is_radian=True, relative=arm_relative
                )

        self.move_hand(allegro_angles, interpolate=False)

        self.last_xarm_command = xarm_angles.copy()
        self.last_allegro_command = allegro_angles.copy()

        print("after control :{}".format(t))
        print("control fps :{}".format(1 / (time.time() - t)))
        # time.sleep(1 / fps)
        # print("after sleep :{}".format(t))
        # print("total control fps :{}".format(1 / (time.time() - t)))

    def move_hand(self, allegro_angles, interpolate=True):
        fps = 300.0  # control itself is 333Hz, but ros is 300Hz.
        model_fps = 30.0
        num_steps = int(fps / model_fps)

        # interpolate to make sure the hand does not move too fast
        current_hand_angles = np.asarray(self.allegro.current_joint_pose.position)
        joint_diff = np.abs(allegro_angles - current_hand_angles)

        if self.last_allegro_command is not None:
            if interpolate:
                # joint_diff = np.abs(allegro_angles - self.last_allegro_command)
                # max_diff = np.max(joint_diff) * fps
                # num_steps = int(max_diff / self.max_hand_joint_vel)
                if num_steps > 0:
                    for i in range(num_steps):
                        alpha = (i + 1) / num_steps
                        new_hand_angles = (
                            alpha * allegro_angles
                            + (1 - alpha) * self.last_allegro_command
                        )
                        self.allegro.hand_pose(new_hand_angles)

                        time.sleep(1 / fps)

            self.allegro.hand_pose(allegro_angles)
            time.sleep(1 / fps)

        else:
            if interpolate:
                # max_diff = np.max(joint_diff) * fps
                # num_steps = int(max_diff / self.max_hand_joint_vel)
                if num_steps > 0:
                    for i in range(num_steps):
                        alpha = (i + 1) / num_steps
                        new_hand_angles = (
                            alpha * allegro_angles + (1 - alpha) * current_hand_angles
                        )
                        self.allegro.hand_pose(new_hand_angles)
                        time.sleep(1 / fps)

            self.allegro.hand_pose(allegro_angles)
            time.sleep(1 / fps)

        self.last_allegro_command = allegro_angles.copy()

    def move_arm(self, angles, arm_control_mode="position", arm_relative=False):
        if arm_control_mode == "position":
            xarm_pose = angles.copy()
            xarm_pose[:3] = xarm_pose[:3] * 1000

            self.arm.set_position(
                x=xarm_pose[0],
                y=xarm_pose[1],
                z=xarm_pose[2],
                roll=xarm_pose[3],
                pitch=xarm_pose[4],
                yaw=xarm_pose[5],
                relative=arm_relative,
            )
        elif arm_control_mode == "position_aa":
            xarm_pose = angles.copy()
            xarm_pose[:3] = xarm_pose[:3] * 1000
            self.arm.set_position_aa(axis_angle_pose=xarm_pose, relative=arm_relative)
        else:
            assert arm_control_mode == "servo_angle"
            xarm_angles = angles
            self.arm.set_servo_angle(angles=xarm_angles, is_radian=True)

        self.last_xarm_command = angles

    def move_arm_servo(
        self,
        angles,
        arm_control_mode="servo_angle_j",
        arm_relative=False,
    ):
        if arm_control_mode == "servo_angle_j":
            assert not arm_relative
            xarm_angles = angles.copy()
            self.arm.set_servo_angle_j(angles=xarm_angles, is_radian=True)
        elif arm_control_mode == "servo_cartesian":
            xarm_pose = angles.copy()
            xarm_pose[:3] = xarm_pose[:3] * 1000
            self.arm.set_servo_cartesian(
                mvpose=xarm_pose, is_radian=True, relative=arm_relative
            )
        else:
            assert arm_control_mode == "servo_cartesian_aa"
            xarm_pose = angles.copy()
            xarm_pose[:3] = xarm_pose[:3] * 1000
            self.arm.set_servo_cartesian_aa(
                axis_angle_pose=xarm_pose, is_radian=True, relative=arm_relative
            )

        self.last_xarm_command = angles

    def get_joint_values(self):
        is_error = 1
        while is_error != 0:
            is_error, arm_joint_states = self.arm.get_joint_states(is_radian=True)
            xarm_angles = np.array(arm_joint_states[0])

        allegro_angles = self.allegro.current_joint_pose.position
        allegro_angles = np.array(allegro_angles)

        return xarm_angles, allegro_angles

    def get_arm_position(self):

        is_error = 1

        while is_error != 0:
            is_error, arm_position_aa = self.arm.get_position_aa(is_radian=True)

        arm_position_aa = np.array(arm_position_aa)
        arm_position_aa[:3] = arm_position_aa[:3] / 1000  # change to meters

        return arm_position_aa


if __name__ == "__main__":
    dex_arm = DexArmControl()

    dex_arm.home_robot()

    # for i in range(100):
    #     allegro_angles = np.ones(16)
    #     dex_arm.move_hand(allegro_angles=allegro_angles, interpolate=True)

    #     allegro_angles = np.zeros(16)
    #     dex_arm.move_hand(allegro_angles=allegro_angles, interpolate=True)

    print("Done!")
