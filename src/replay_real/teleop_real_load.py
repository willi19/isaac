import numpy as np
from scipy.spatial.transform import Rotation
from dex_robot.robot_controller.robot_controller import DexArmControl
from dex_robot.xsens.receiver import XSensReceiver

import numpy as np
from dex_robot.utils import hand_index
import time
import threading
from threading import Event
from dex_robot.utils.file_io import load_robot_target_traj, rsc_path
import os
import chime
from dex_robot.utils.robot_wrapper import RobotWrapper
import transforms3d
from dex_robot.contact.receiver import SerialReader
from dex_robot.camera.camera_loader import CameraManager
# home_wrist_pose = np.load("data/home_pose/allegro_eef_frame.npy")
# home_hand_pose = np.load("data/home_pose/allegro_hand_joint_angle.npy")
# home_qpos = np.load("data/home_pose/allegro_robot_action.npy")

XSENS2ISAAC = np.array(
    [
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ]
)

XSENS2XARM = np.array(
    [
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
    ]
)

def homo2cart(h):
    if h.shape == (4, 4):
        t = h[:3, 3]
        R = h[:3, :3]
        q = Rotation.from_matrix(R).as_euler("XYZ")
    else:
        t = h[:3]
        q = h[3:]
        q = Rotation.from_quat(q).as_euler("XYZ")

    return np.concatenate([t, q])

def listen_for_exit(stop_event):
    """Listens for 'q' key input to safely exit all processes."""
    while not stop_event.is_set():
        user_input = input()
        if user_input.lower() == "q":
            print("\n[INFO] Exiting program...")
            stop_event.set()  # Set the exit flag
            break

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
    target_action[2] += max(0, 0.05 - min_z)
    return target_action

demo_path = f"data/teleoperation/bottle"
demo_path_list = os.listdir(demo_path)

arm_control_mode = "servo_cartesian_aa"

def main():
    date_time = time.strftime("%Y%m%d%H%M%S")
    arm_controller = DexArmControl("test/mingi", date_time)
    contact_receiver = SerialReader("test/mingi", date_time)
    camera_receiver = CameraManager("test/mingi", date_time)
    os.makedirs("/home/temp_id/capture/test/mingi", exist_ok=True)

    traj_cnt = 1
    stop_event = threading.Event()

    input_thread = threading.Thread(target=listen_for_exit, args=(stop_event,))
    input_thread.daemon = True  # Ensures the thread exits when the main program exits
    input_thread.start()

    init_wrist_pose = None
    init_robot_pose = None
    cur_robot_pose = None
    start = True

    state_2_cnt = 0

    for count in range(traj_cnt):
        demo_name = demo_path_list[count]
        robot_traj = load_robot_target_traj(os.path.join(demo_path, demo_name))[:200]

        # home_robot(arm_controller)
        # init_robot_pose = np.zeros((4,4))
        # init_robot_pose[:3,:3] = Rotation.from_quat(home_wrist_pose[3:]).as_matrix() @ XSENS2XARM[:3,:3]
        # init_robot_pose[:3,3] = home_wrist_pose[:3]
        # init_robot_pose[3,3] = 1
        
        init_robot_pose = robot_traj[0].copy()
        init_robot_pose = safety_bound(init_robot_pose)
        cur_robot_pose = init_robot_pose.copy()

        
        # arm_controller.set_homepose()
        
        
        arm_controller.set_homepose(init_robot_pose[:6], init_robot_pose[6:])

        start = True
        target_action = init_robot_pose.copy()#np.concatenate([homo2cart(home_wrist_pose), home_hand_pose])

        # arm_controller.ready_array[0] = 0
        arm_controller.home_robot()

        home_start_time = time.time()
        while arm_controller.ready_array[0] != 1:
            if time.time() - home_start_time > 0.3:
                chime.warning()
                home_start_time = time.time()
            time.sleep(0.008)
        chime.success()
        print("count: =========", count)
        print("Robot homed.")

        step = 0
        while not stop_event.is_set():
            state = 2
            if step < len(robot_traj):
                state = 0
            elif step > len(robot_traj):
                state = 2
            # print(state, step)
            if state == 0:
                target_action = robot_traj[step].copy()# np.concatenate(robot_traj[step])


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

            step += 1

            target_action = safety_bound(target_action)
            print(target_action[:6])
            arm_controller.set_robot_servo(
                            allegro_angles=target_action[6:],
                            xarm_angles=target_action[:6]
                        )
            time.sleep(1/30)

    arm_controller.quit()
    contact_receiver.quit()
    print("Program terminated.")
    exit(0)

if __name__ == "__main__":
    main()