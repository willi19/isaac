import numpy as np
from scipy.spatial.transform import Rotation
from dex_robot.io.robot_controller import XArmController
import time
import transforms3d as t3d
from dex_robot.utils.robot_wrapper import RobotWrapper
import os
from dex_robot.utils.file_io import rsc_path,capture_path
import chime

home_wrist_pose = np.load("data/home_pose/allegro_eef_frame.npy")
home_hand_pose = np.load("data/home_pose/allegro_hand_joint_angle.npy")
home_qpos = np.load("data/home_pose/allegro_robot_action.npy")

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
    os.path.join(rsc_path, "allegro", "allegro.urdf")
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
    arm_controller = XArmController()

    init_robot_pose = home_wrist_pose.copy()
    cur_robot_pose = init_robot_pose.copy()
    
    arm_controller.set_homepose(homo2cart(cur_robot_pose))
    arm_controller.home_robot()
    home_start_time = time.time()
    while arm_controller.ready_array[0] != 1:
        if time.time() - home_start_time > 0.3:
            chime.warning()
            home_start_time = time.time()
        time.sleep(0.0008)
    chime.success()
    
    arm_controller.quit()
    exit(0)

if __name__ == "__main__":
    main()