from move_dexarm import DexArmControl

import numpy as np
import rospy

if __name__ == "__main__":

    arm_controller = DexArmControl()
    arm_controller.home_robot()
    arm_control_mode_slow = "position_aa"
    arm_control_mode = "servo_cartesian_aa"

    rate = rospy.Rate(30.0)

    warmstart_timestep = 15

    # if warmstart_timestep > 0:
    #     print("Warmstarting...")

    #     arm_controller.move_robot(
    #         allegro_angles=qpos_real[-16:],
    #         xarm_angles=qpos_real[:6],
    #         arm_control_mode=arm_control_mode_slow,
    #         arm_relative=False,
    #         wait=True,
    #     )

    #     if warmstart_timestep == 1:
    #         arm_controller.arm.set_mode(1)
    #         arm_controller.arm.set_state(state=0)

    #     ("first init done.")
    #     first = False

    # else:
    #     arm_controller.move_robot_servo(
    #         allegro_angles,
    #         qpos_real[:6],
    #         arm_control_mode=arm_control_mode,
    #         arm_relative=False,
    #     )

    rate.sleep()