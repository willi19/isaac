import numpy as np
from scipy.spatial.transform import Rotation
from dex_robot.robot_controller.robot_controller import DexArmControl
from dex_robot.xsens.receiver import XSensReceiver

import numpy as np
from dex_robot.utils import hand_index
import time
import threading
from threading import Event

home_wrist_pose = np.load("data/home_pose/allegro_eef_frame.npy")
home_hand_pose = np.load("data/home_pose/allegro_hand_joint_angle.npy")
home_qpos = np.load("data/home_pose/allegro_robot_action.npy")

XSENS2ISAAC = np.array(
    [
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
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

def main():
    host = "192.168.0.2"
    port = 9763

    xsens_updater = XSensReceiver()
    xsens_updater.init_server(host, port)
    
    arm_controller = DexArmControl(xarm_home_pose=homo2cart(home_wrist_pose), allegro_home_pose=home_hand_pose)
    # arm_controller.init()
    # arm_control_mode_slow = "position_aa"
    arm_control_mode = "servo_cartesian_aa"

    traj_cnt = 5
    stop_event = threading.Event()

    input_thread = threading.Thread(target=listen_for_exit, args=(stop_event,))
    input_thread.daemon = True  # Ensures the thread exits when the main program exits
    input_thread.start()

    init_wrist_pose = None
    init_robot_pose = None
    cur_robot_pose = None
    start = True

    timestamp = []
    robot_action = []
    robot_state = []
    contact_sensor_value = []

    state_2_cnt = 0

    for count in range(traj_cnt):
        # home_robot(arm_controller)
        init_robot_pose = np.zeros((4,4))
        init_robot_pose[:3,:3] = Rotation.from_quat(home_wrist_pose[3:]).as_matrix()
        init_robot_pose[:3,3] = home_wrist_pose[:3]
        init_robot_pose[3,3] = 1
        cur_robot_pose = init_robot_pose.copy()
        start = True
        target_action = np.concatenate([homo2cart(home_wrist_pose), home_hand_pose])

        while not stop_event.is_set():
            # go to home pose
            arm_controller.ready_array[0] = 0
            arm_controller.home_robot()

            while arm_controller.ready_array[0] != 1:
                # puzz()
                time.sleep(0.0008)
            print("Robot homed.")
            data = xsens_updater.get_data()
            state = data["state"]
            state = 2
            if data["hand_pose"] is None:
                continue

            if state == 0:
                if start:
                    init_wrist_pose = data["hand_pose"][0]
                    start = False
                try:
                    delta_wrists_R = XSENS2ISAAC[:3,:3].T @ np.linalg.inv(init_wrist_pose[:3,:3]) @ data["hand_pose"][0][:3,:3] @ XSENS2ISAAC[:3,:3]
                except:
                    delta_wrists_R = np.eye(3)
                delta_wrists_t = data["hand_pose"][0][:3,3] - init_wrist_pose[:3,3]
                cur_robot_pose = np.zeros((4,4))
                cur_robot_pose[:3,:3] = init_robot_pose[:3,:3] @ delta_wrists_R
                
                cur_robot_pose[:3,3] = delta_wrists_t + init_robot_pose[:3,3]
                cur_robot_pose[3,3] = 1

                qpos_sim = np.zeros(16)
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

            # arm_controller.set_robot_servo(
            #                 allegro_angles=target_action[6:],
            #                 xarm_angles=target_action[:6],
            #                 arm_control_mode=arm_control_mode,
            #                 arm_relative=False,
            #             )
    
    arm_controller.exit()
    xsens_updater.close_server()
    print("Program terminated.")
    exit(0)

if __name__ == "__main__":
    main()