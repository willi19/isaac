import numpy as np
from scipy.spatial.transform import Rotation
from dex_robot.robot_controller.robot_controller import DexArmControl
from dex_robot.xsens.receiver import XSensReceiver
from dex_robot.io.xsens import hand_index
import time
import threading
import transforms3d as t3d
from dex_robot.utils.robot_wrapper import RobotWrapper
import os
from dex_robot.utils.file_io_prev import rsc_path,capture_path
from dex_robot.contact.receiver import SerialReader
import chime
from dex_robot.camera.camera_loader import CameraManager
import argparse
import json

home_wrist_pose = np.load("data/home_pose/contact_test_allegro_eef_frame.npy")
home_hand_pose = np.load("data/home_pose/contact_test_allegro_hand_joint_angle.npy")
# home_qpos = np.load("data/home_pose/allegro_robot_action.npy")

LINK62PALM = np.array(
    [
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [-1, 0, 0, 0],
        [0, 0, 0, 1],
    ]
)

XSENS2ISAAC = np.array(
    [
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ]
)

def homo2cart(h):
    if h.shape == (4, 4):
        t = h[:3, 3]
        R = h[:3, :3]

        axis, angle = t3d.axangles.mat2axangle(R)
        axis_angle = axis * angle
    else:
        raise ValueError("Invalid input shape.")
    return np.concatenate([t, axis_angle])

def listen_for_exit(stop_event):
    """Listens for 'q' key input to safely exit all processes."""
    while not stop_event.is_set():
        user_input = input()
        if user_input.lower() == "q":
            print("\n[INFO] Exiting program...")
            stop_event.set()  # Set the exit flag
            break

def get_latest_index(name):
    save_path = os.path.join(capture_path, name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    files = os.listdir(save_path)
    return str(len(files))
    
robot = RobotWrapper(
    os.path.join(rsc_path, "xarm6", "allegro.urdf")
)
link_list = ["palm_link", "thumb_base", "thumb_proximal", "thumb_medial", "thumb_distal", "thumb_tip", 
             "index_base", "index_proximal", "index_medial", "index_distal", "index_tip", 
             "middle_base", "middle_proximal", "middle_medial", "middle_distal", "middle_tip", 
             "ring_base", "ring_proximal", "ring_medial", "ring_distal", "ring_tip"]

def safety_bound(target_action):
    angle = np.linalg.norm(target_action[3:6])
    axis = target_action[3:6] / angle
    
    R = Rotation.from_rotvec(angle * axis).as_matrix()

    euler = Rotation.from_matrix(R).as_euler("XYZ")
    t = target_action[:3]

    pin_action = np.concatenate([t, euler, target_action[6:]])
    robot.compute_forward_kinematics(pin_action)
    min_z = 10
    link_min_name = None
    for link in link_list:
        link_index = robot.get_link_index(link)
        link_pose = robot.get_link_pose(link_index)
        if link_pose[2,3] < min_z:
            link_min_name = link
        min_z = min(min_z, link_pose[2,3])
    
    # print(max(0, 0.05 - min_z), min_z, link_min_name)
    target_action[2] += max(0, 0.01 - min_z)
    return target_action

arm_control_mode = "servo_cartesian_aa"

def main():
    name = "contact"
    index = get_latest_index(name) 
    save_path = os.path.join(capture_path, name, index)

    arm_controller = DexArmControl(save_path)
    contact_receiver = SerialReader(save_path)

    camera_receiver = CameraManager(save_path, num_cameras=1, is_streaming=False, syncMode=True)
    camera_receiver.start()
    
    traj_cnt = 5
    stop_event = threading.Event()

    input_thread = threading.Thread(target=listen_for_exit, args=(stop_event,))
    input_thread.daemon = True  # Ensures the thread exits when the main program exits
    input_thread.start()

    init_wrist_pose = None
    init_robot_pose = None
    cur_robot_pose = None
    start = True

    state_2_cnt = 0
    
    
    # home_robot(arm_controller)
    init_robot_pose = home_wrist_pose.copy()

    print(init_robot_pose)
    
    cur_robot_pose = init_robot_pose.copy()
    activate_start_time = -1
    activate_end_time = -1

    start = True
    target_action = np.concatenate([homo2cart(cur_robot_pose), home_hand_pose])
    target_action = safety_bound(target_action)
    arm_controller.set_homepose(target_action[:6], target_action[6:])
    arm_controller.home_robot()

    home_start_time = time.time()
    while arm_controller.ready_array[0] != 1:
        if time.time() - home_start_time > 0.3:
            chime.warning()
            home_start_time = time.time()
        time.sleep(0.0008)
    chime.success()
    print("Robot homed.")

    # while not stop_event.is_set():
    #     state = 1
    #     if state == 2:
    #         state_2_cnt += 1
    #         if state_2_cnt > 30:
    #             state_2_cnt = 0
    #             break
            
    #     else:
    #         state_2_cnt = 0

    #     target_action = safety_bound(target_action)
    #     arm_controller.set_robot_servo(
    #                     allegro_angles=target_action[6:],
    #                     xarm_angles=target_action[:6]
    #                 )

    
    arm_controller.quit()
    contact_receiver.quit()
    camera_receiver.quit()
    print("Program terminated.")
    exit(0)

if __name__ == "__main__":
    main()