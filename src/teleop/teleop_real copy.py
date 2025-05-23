import numpy as np
from scipy.spatial.transform import Rotation

from dex_robot.io.robot_controller import XArmController, AllegroController, InspireController
from dex_robot.io.xsens.receiver import XSensReceiver
from dex_robot.io.contact.receiver import SerialReader
from dex_robot.io.camera.camera_loader import CameraManager
from dex_robot.io.robot_controller import retarget

from dex_robot.utils.file_io import rsc_path, capture_path, shared_path

from paradex.utils.io import find_latest_index, find_latest_directory

import time
import threading
import os
import chime
import argparse
import json
import shutil
import transforms3d as t3d

hand_name = "allegro"
arm_name = "xarm"

home_wrist_pose = np.load("data/home_pose/allegro_eef_frame.npy")
home_qpos = np.load("data/home_pose/allegro_robot_action.npy")


def load_homepose(hand_name):
    if hand_name == "allegro":
        return  np.load("data/home_pose/allegro_hand_joint_angle.npy")
    elif hand_name == "inspire":
        return np.zeros(6)+1000
    

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

def copy_calib_files(save_path):

    handeye_calib_dir = os.path.join(shared_path, "handeye_calibration")
    handeye_calib_name = find_latest_directory(handeye_calib_dir)
    handeye_calib_path = os.path.join(shared_path, "handeye_calibration", handeye_calib_name, "0", "C2R.npy")

    camparam_dir = os.path.join(shared_path, "cam_param")
    camparam_name = find_latest_directory(camparam_dir)
    camparam_path = os.path.join(shared_path, "cam_param", camparam_name)

    shutil.copyfile(handeye_calib_path, os.path.join(save_path, "C2R.npy"))
    shutil.copytree(camparam_path, os.path.join(save_path, "cam_param"))


def load_savepath(name):
    if name == None:
        return None
    index = int(find_latest_index(os.path.join(capture_path, name)))+1
    return os.path.join(capture_path, name, str(index))



def initialize_teleoperation(save_path):
    controller = {}
    if save_path != None:
        os.makedirs(save_path, exist_ok=True)
        controller["camera"] = CameraManager(save_path, num_cameras=1, is_streaming=False, syncMode=True)
        

    if arm_name == "xarm":
        controller["arm"] = XArmController(save_path)

    if hand_name == "allegro":
        controller["hand"] = AllegroController(save_path)
        if save_path != None:
            controller["contact"] = SerialReader(save_path)
    elif hand_name == "inspire":
        controller["hand"] = InspireController(save_path)
    
    
    controller["xsens"] = XSensReceiver()

    return controller


def main():
    parser = argparse.ArgumentParser(description="Teleoperation for real robot")
    parser.add_argument("--name", type=str, help="Control mode for robot")
    args = parser.parse_args()
    
    os.makedirs(os.path.join(capture_path, args.name), exist_ok=True)
    save_path = load_savepath(args.name)
    print (f"save_path: {save_path}")
    sensors = initialize_teleoperation(save_path)
    
    traj_cnt = 5
    stop_event = threading.Event()

    input_thread = threading.Thread(target=listen_for_exit, args=(stop_event,))
    input_thread.daemon = True  # Ensures the thread exits when the main program exits
    input_thread.start()

    retargetor = retarget.retargetor(arm_name=arm_name, hand_name=hand_name, home_arm_pose=home_wrist_pose)

    homepose_cnt = 0
    activate_range = {}

    home_hand_pose = load_homepose(hand_name)
    
    for count in range(traj_cnt):
        activate_range[count] = []
        activate_start_time = -1
        activate_end_time = -1

        if hand_name is not None:
            sensors["hand"].set_homepose(home_hand_pose)
            sensors["hand"].home_robot()
            
        if arm_name is not None:
            sensors["arm"].set_homepose(homo2cart(home_wrist_pose))
            sensors["arm"].home_robot()

            # alarm during homing
            home_start_time = time.time()
            while sensors["arm"].ready_array[0] != 1:
                if time.time() - home_start_time > 0.3:
                    chime.warning()
                    home_start_time = time.time()
                time.sleep(0.0008)
            chime.success()
        retargetor.reset()
        print("count: =========", count)
        print("Robot homed.")

        while not stop_event.is_set():
            
            data = sensors["xsens"].get_data()
            state = data["state"]
            if state == -1: # Xsens not ready
                continue

            if state == 0:
                if activate_start_time == -1:
                    activate_start_time = time.time()

                arm_action, hand_action = retargetor.get_action(data)
            
            else: # Saving timing which robot is activated
                if activate_start_time != -1 and activate_end_time == -1:
                    activate_end_time = time.time()
                    activate_range[count].append((activate_start_time, activate_end_time))
                    activate_start_time = -1
                    activate_end_time = -1

            if state == 1:
                retargetor.pause()
                continue

            if state == 2:
                homepose_cnt += 1
                if homepose_cnt > 30:
                    homepose_cnt = 0
                    break
                
            else:
                homepose_cnt = 0

            if arm_name is not None:                
                sensors["arm"].set_target_action(
                                arm_action
                        )
            if hand_name is not None:
                sensors["hand"].set_target_action(
                                hand_action
                            )
            
    if save_path != None:
        json.dump(activate_range, open(os.path.join(save_path, "activate_range.json"), 'w'))
        copy_calib_files(save_path)

    for key in sensors.keys():
        sensors[key].quit()

    print("Program terminated.")
    exit(0)

if __name__ == "__main__":
    main()