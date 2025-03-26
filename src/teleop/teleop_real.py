import numpy as np
from scipy.spatial.transform import Rotation
from dex_robot.robot_controller.robot_controller import DexArmControl
from dex_robot.xsens.receiver import XSensReceiver
from dex_robot.utils import hand_index
import time
import threading
import transforms3d as t3d
from dex_robot.utils.robot_wrapper import RobotWrapper
import os
from dex_robot.utils.file_io_prev import rsc_path,capture_path, shared_path
from dex_robot.contact.receiver import SerialReader
import chime
from dex_robot.camera.camera_loader import CameraManager
import argparse
import json
import paradex
from paradex.utils.io import find_latest_directory
import shutil

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
    parser = argparse.ArgumentParser(description="Teleoperation for real robot")
    parser.add_argument("--name", type=str, required=True, help="Control mode for robot")
    args = parser.parse_args()

    date_time = time.strftime("%Y%m%d_%H%M%S")
    name = args.name

    host = "192.168.0.2"
    port = 9763

    xsens_updater = XSensReceiver()
    xsens_updater.init_server(host, port)

    index = get_latest_index(name) 
    save_path = os.path.join(capture_path, name, index)

    arm_controller = DexArmControl(save_path)
    contact_receiver = SerialReader(save_path)

    camera_receiver = CameraManager(save_path, num_cameras=1, is_streaming=False, syncMode=True)
    camera_receiver.start()
    
    traj_cnt = 2
    stop_event = threading.Event()

    input_thread = threading.Thread(target=listen_for_exit, args=(stop_event,))
    input_thread.daemon = True  # Ensures the thread exits when the main program exits
    input_thread.start()

    init_wrist_pose = None
    init_robot_pose = None
    cur_robot_pose = None
    start = True

    state_2_cnt = 0
    activate_range = {}

    handeye_calib_dir = os.path.join(shared_path, "handeye_calibration")
    handeye_calib_name = find_latest_directory(handeye_calib_dir)
    handeye_calib_path = os.path.join(shared_path, "handeye_calibration", handeye_calib_name, "0", "C2R.npy")

    camparam_dir = os.path.join(shared_path, "cam_param")
    camparam_name = find_latest_directory(camparam_dir)
    camparam_path = os.path.join(shared_path, "cam_param", camparam_name)

    shutil.copyfile(handeye_calib_path, os.path.join(save_path, "C2R.npy"))
    shutil.copytree(camparam_path, os.path.join(save_path, "cam_param"))
    
    for count in range(traj_cnt):
        # home_robot(arm_controller)


        init_robot_pose = home_wrist_pose.copy()
        cur_robot_pose = init_robot_pose.copy()
        activate_range[count] = []
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
        print("count: =========", count)
        print("Robot homed.")

        while not stop_event.is_set():
            # go to home pose
            data = xsens_updater.get_data()
            # print(data["state"])
            state = data["state"]
            
            if state == -1:
                continue

            if state == 0:
                if activate_start_time == -1:
                    activate_start_time = time.time()
                if start:
                    init_wrist_pose = data["hand_pose"][0].copy()
                    start = False
                try:
                    delta_wrists_R = LINK62PALM[:3,:3] @ XSENS2ISAAC[:3,:3].T @ np.linalg.inv(init_wrist_pose[:3,:3]) @ data["hand_pose"][0][:3,:3] @ XSENS2ISAAC[:3,:3] @ LINK62PALM[:3,:3].T
                except:
                    delta_wrists_R = np.eye(3)
                delta_wrists_t = data["hand_pose"][0][:3,3] - init_wrist_pose[:3,3]

                cur_robot_pose = np.zeros((4,4))
                cur_robot_pose[:3,:3] = init_robot_pose[:3,:3] @ delta_wrists_R
                
                cur_robot_pose[:3,3] = delta_wrists_t + init_robot_pose[:3,3]
                cur_robot_pose[3,3] = 1

                hand_joint_angle = np.zeros((20,3))# data["hand_joint_angle"].copy() 
                hand_pose_frame = data["hand_pose"].copy()  
                allegro_angles = np.zeros(16)

                for finger_id in range(4):
                    for joint_id in range(4):
                        if joint_id == 0:
                            rot_mat = np.linalg.inv(hand_pose_frame[0,:3,:3]) @ hand_pose_frame[finger_id * 4 + joint_id + 1, :3,:3]
                        else:
                            rot_mat = np.linalg.inv(hand_pose_frame[hand_index.hand_index_parent[finger_id * 4 + joint_id+1], :3,:3]) @ hand_pose_frame[finger_id * 4 + joint_id + 1, :3,:3]
                        hand_joint_angle[finger_id * 4 + joint_id + 1] = Rotation.from_matrix(rot_mat).as_euler("zyx")
                # zyx euler angle in hand frame = zxy axis angle in robot frame
                allegro_angles[0] = hand_joint_angle[5][0]  # z in robot, y in hand
                allegro_angles[1] = hand_joint_angle[5][2] * 1.2  # y in robot, z in hand
                allegro_angles[2] = hand_joint_angle[6][2] * 0.8
                allegro_angles[3] = hand_joint_angle[7][2] * 0.8

                thumb_meta = np.dot(hand_pose_frame[0,:3,:3].T, hand_pose_frame[1,:3,:3])
                thumb_meta_angle = Rotation.from_matrix(thumb_meta).as_euler("xyz")

                # for drum
                allegro_angles[12] = thumb_meta_angle[0]  # -x in robot, y in hand
                allegro_angles[13] = thumb_meta_angle[1] - 0.5  # y in robot, z in hand
                allegro_angles[14] = hand_joint_angle[2][2] * 1.2
                allegro_angles[15] = hand_joint_angle[3][2] * 1.2

                # for others
                # allegro_angles[4] = thumb_meta_angle[0]  # -x in robot, y in hand
                # allegro_angles[5] = thumb_meta_angle[1] * 0.1  # y in robot, z in hand
                # allegro_angles[6] = hand_joint_angle[2][2] + 1.0
                # allegro_angles[7] = hand_joint_angle[3][2] * 1.2

                allegro_angles[4] = hand_joint_angle[9][0]  # z in robot, y in hand
                allegro_angles[5] = hand_joint_angle[9][2] * 1.2  # y in robot, z in hand
                allegro_angles[6] = hand_joint_angle[10][2] * 0.8
                allegro_angles[7] = hand_joint_angle[11][2] * 0.8

                allegro_angles[8] = hand_joint_angle[13][0]  # z in robot, y in hand
                allegro_angles[9] = hand_joint_angle[13][2] * 1.2  # y in robot, z in hand
                allegro_angles[10] = hand_joint_angle[14][2] * 0.8
                allegro_angles[11] = hand_joint_angle[15][2] * 0.8

                #qpos_sim[6:] = allegro_angles

                target_action = np.concatenate([homo2cart(cur_robot_pose), allegro_angles])
            
            else:
                if activate_start_time != -1 and activate_end_time == -1:
                    activate_end_time = time.time()
                    activate_range[count].append((activate_start_time, activate_end_time))
                    activate_start_time = -1
                    activate_end_time = -1

            if state == 1:
                init_wrist_pose = None
                if not start:
                    init_robot_pose = cur_robot_pose.copy()
                cur_robot_pose = None
                start = True

            if state == 2:
                state_2_cnt += 1
                if state_2_cnt > 30:
                    state_2_cnt = 0
                    break
                
            else:
                state_2_cnt = 0

            target_action = safety_bound(target_action)
            arm_controller.set_robot_servo(
                            allegro_angles=target_action[6:],
                            xarm_angles=target_action[:6]
                        )

    json.dump(activate_range, open(os.path.join(save_path, "activate_range.json"), 'w'))
    arm_controller.quit()
    xsens_updater.quit()
    contact_receiver.quit()
    camera_receiver.quit()
    print("Program terminated.")
    exit(0)

if __name__ == "__main__":
    main()