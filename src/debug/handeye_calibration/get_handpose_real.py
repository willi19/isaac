import os
import argparse

import numpy as np
import json
import cv2
from dex_robot.utils.file_io import shared_path, load_camparam, load_c2r, download_path, rsc_path
from paradex.utils.marker import detect_aruco, triangulate, ransac_triangulation
import tqdm
from dex_robot.utils.robot_wrapper import RobotWrapper


import numpy as np
import os
from paradex.utils.io import handeye_calib_path, find_latest_directory
import argparse
import numpy as np
from scipy.linalg import sqrtm
from numpy.linalg import inv
import numpy as np
from scipy.spatial.transform import Rotation as R
from paradex.utils.math import rigid_transform_3D

def fill_framedrop(cam_timestamp):
    frameID = cam_timestamp["frameID"]
    pc_time = np.array(cam_timestamp["pc_time"])
    timestamp = np.array(cam_timestamp["timestamps"])

    time_delta = (1/30)# (pc_time[-1]-pc_time[0])/(frameID[-1]-frameID[0])
    offset = np.mean(pc_time - (np.array(frameID)-1)*time_delta)

    pc_time_nodrop = []
    frameID_nodrop = []

    for i in range(1, frameID[-1]+1):
        frameID_nodrop.append(i)
        pc_time_nodrop.append((i-1)*time_delta+offset)

    return pc_time_nodrop, frameID_nodrop

def load_timestamp(name):
    timestamp_list = []
    frame_num_list = []
    capture_ind = []

    index_list = os.listdir(os.path.join(shared_path, "capture", name))

    for index in index_list:
        timestamp = json.load(open(os.path.join(shared_path, "capture", name, index, "camera_timestamp.json")))
        pc_time, frameID = fill_framedrop(timestamp)

        selected_frame = json.load(open(os.path.join(shared_path, "capture", name, index, "selected_frame.json")))
        frame_num_list.append(len(selected_frame))

        
        for i in range(len(selected_frame)):
            timestamp_list.append([])
            s_f = selected_frame[str(i)]
            for (start, end) in s_f:
                for tmp in range(start-1, end):
                    timestamp_list[-1].append(pc_time[tmp])
                    if frameID[tmp] != tmp+1:
                        print("frameID error: ", frameID[tmp], tmp+1)
                
            timestamp_list[-1] = np.array(timestamp_list[-1])
            capture_ind.append(index)

    return timestamp_list, capture_ind



def logR(T):
    R = T[0:3, 0:3]
    theta = np.arccos((np.trace(R) - 1)/2)
    logr = np.array([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]]) * theta / (2*np.sin(theta))
    return logr

def Calibrate(A, B):
    n_data = len(A)
    M = np.zeros((3,3))
    C = np.zeros((3*n_data, 3))
    d = np.zeros((3*n_data, 1))
    A_ = np.array([])
    for i in range(n_data-1):
        alpha = logR(A[i])
        beta = logR(B[i])
        alpha2 = logR(A[i+1])
        beta2 = logR(B[i+1])
        alpha3 = np.cross(alpha, alpha2)
        beta3  = np.cross(beta, beta2)
        M1 = np.dot(beta.reshape(3,1),alpha.reshape(3,1).T)
        M2 = np.dot(beta2.reshape(3,1),alpha2.reshape(3,1).T)
        M3 = np.dot(beta3.reshape(3,1),alpha3.reshape(3,1).T)
        M = M1+M2+M3
    theta = np.dot(sqrtm(inv(np.dot(M.T, M))), M.T)
    for i in range(n_data):
        rot_a = A[i][0:3, 0:3]
        rot_b = B[i][0:3, 0:3]
        trans_a = A[i][0:3, 3]
        trans_b = B[i][0:3, 3]
        C[3*i:3*i+3, :] = np.eye(3) - rot_a
        d[3*i:3*i+3, 0] = trans_a - np.dot(theta, trans_b)
    b_x  = np.dot(inv(np.dot(C.T, C)), np.dot(C.T, d))
    return theta, b_x

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--name', type=str, default=None)
    # args = parser.parse_args()

    # name_list = [args.name] if args.name else os.listdir(os.path.join(download_path, 'processed'))
    name_list = ["void"]
    
    robot = RobotWrapper(
        os.path.join(rsc_path, "xarm6", "xarm6_allegro_wrist_mounted_rotate.urdf")
    )

    link_index = robot.get_link_index("link6")
    finger_index = {f"{finger_name}_proximal":robot.get_link_index(f"{finger_name}_proximal") for finger_name in ["thumb", "index", "middle", "ring"]}
    finger_marker = {11:"ring_proximal", 13:"middle_proximal", 14:"index_proximal", 10:"thumb_proximal"}
    wrist_index = robot.get_link_index("link6")

    A_list = []
    B_list = []
    
    for name in name_list:
        index_list = os.listdir(os.path.join(download_path, 'processed', name))
        index_list = ["1"]
        
        for index in index_list:
            root_path = os.path.join(download_path, 'processed', name, index)
            
            c2r = load_c2r(root_path)
            robot_action = np.load(os.path.join(root_path, "arm", "state.npy"))
            hand_action = np.load(os.path.join(root_path, "hand", "state.npy"))


            marker_pose = np.load(os.path.join(root_path, "marker_pos.npy"), allow_pickle=True).item()
            serial_list = os.listdir(os.path.join(root_path, "video_extracted"))

            seq_len = len(os.listdir(os.path.join(root_path, "video_extracted", serial_list[0])))

            for serial_num in serial_list:
                if len(os.listdir(os.path.join(root_path, "video_extracted", serial_num))) != seq_len:
                    print("video_extracted error: ", serial_num)
                    break
            
            finger_id_list = [11,13,14]
            marker_id = [262,263,264,265,266]            
            marker_pose = np.load(os.path.join(root_path, "marker_pos.npy"), allow_pickle=True).item()

            intrinsic, extrinsic = load_camparam(root_path)
            # for fid in tqdm.tqdm(range(seq_len)):
            #     marker_3d = np.load(os.path.join(root_path, "marker3d", f"{fid:05d}.npy"), allow_pickle=True).item()
            #     robot_act = np.zeros(22)
            #     robot_act[0:6] = robot_action[fid]
            #     robot.compute_forward_kinematics(robot_act)
            #     link5_id = robot.get_link_index("link5")
            #     robot_pose = robot.get_link_pose(link5_id)

            #     # Verify C2R works
            #     for mid in marker_id:
            #         if mid not in marker_3d or marker_3d[mid] is None:
            #             continue
            #         # marker_pose_h = np.concatenate([marker_pose[mid], np.ones((marker_pose[mid].shape[0], 1))], axis=1)
                    
            #         marker_pose_h = c2r @ robot_pose @ marker_pose[mid]
            #         marker_pose_h /= marker_pose_h[-1]
            #         marker_pose_h = marker_pose_h[:3]
            #         print(marker_pose_h.T-marker_3d[mid])
                

                    
            for fid in tqdm.tqdm(range(seq_len//60-1)):
                fid1 = fid * 60
                fid2 = fid1 + 60

                marker_3d1 = np.load(os.path.join(root_path, "marker3d", f"{fid1:05d}.npy"), allow_pickle=True).item()
                marker_3d2 = np.load(os.path.join(root_path, "marker3d", f"{fid2:05d}.npy"), allow_pickle=True).item()

                robot_action1 = np.concatenate([robot_action[fid1], hand_action[fid1]])
                robot_action2 = np.concatenate([robot_action[fid2], hand_action[fid2]])

            #     # for finger_id in finger_id_list:
            #     #     if finger_id not in marker_3d1 or finger_id not in marker_3d2:
            #     #         continue
            #     #     if marker_3d1[finger_id] is None or marker_3d2[finger_id] is None:
            #     #         continue
            #     #     A = rigid_transform_3D(marker_3d2[finger_id], marker_3d1[finger_id])
                    

            #     #     robot.compute_forward_kinematics(robot_action1)
            #     #     T_r1 = robot.get_link_pose(link_index)
            #     #     T_h1 = robot.get_link_pose(finger_index[finger_marker[finger_id]])
            #     #     T_w1 = robot.get_link_pose(wrist_index)

            #     #     # T_h1 = np.linalg.inv(T_w1) @ T_h1

            #     #     robot.compute_forward_kinematics(robot_action2)
            #     #     T_r2 = robot.get_link_pose(link_index)
            #     #     T_h2 = robot.get_link_pose(finger_index[finger_marker[finger_id]])
            #     #     T_w2 = robot.get_link_pose(wrist_index)

            #     #     # T_h2 = np.linalg.inv(T_w2) @ T_h2

            #     #     #A = np.linalg.inv(T_r1) @ np.linalg.inv(c2r) @ A @ c2r @ T_r2
            #     #     B = T_h1 @ np.linalg.inv(T_h2)

            #     #     A_list.append(A)
            #     #     B_list.append(B)
            #     #     print(A@c2r - c2r@B)

            #     #     # err = A @ B - np.eye(4)
            #     #     # if np.max(np.abs(err)) < 0.05:
            #     #     #     print("Error: ", np.max(np.abs(err)))
            #     #     #     print(A)
            #     #     #     print(B)
            #     #     #     print(np.linalg.inv(A) @ B)

                link5_id = robot.get_link_index(f"link5")
                robot.compute_forward_kinematics(robot_action1)
                robot1 = robot.get_link_pose(link5_id)

                robot.compute_forward_kinematics(robot_action2)
                robot2 = robot.get_link_pose(link5_id)
                
                B = robot1 @ np.linalg.inv(robot2)
                B_list.append(robot1 @ np.linalg.inv(robot2))

                marker1 = []
                marker2 = []
                for mid in marker_3d1:
                    if mid in marker_3d2:
                        if marker_3d1[mid] is None or marker_3d2[mid] is None:
                            continue
                        marker1.append(marker_3d1[mid])
                        marker2.append(marker_3d2[mid])
                
                marker1 = np.vstack(marker1)
                marker2 = np.vstack(marker2)
                A_list.append(rigid_transform_3D(marker2, marker1))
                A = A_list[-1]

                marker1_h = np.concatenate([marker1, np.ones((marker1.shape[0], 1))], axis=1)
                marker2_h = np.concatenate([marker2, np.ones((marker2.shape[0], 1))], axis=1)

                # print(A@c2r - c2r@B)
                # import pdb; pdb.set_trace()
                    
    print(len(A_list))
    X = np.eye(4)
    theta, b_x = Calibrate(A_list, B_list)
    X[0:3, 0:3] = theta
    X[0:3, -1] = b_x.flatten()

    link6 = robot.get_link_pose(link_index)
    wrist = robot.get_link_pose(wrist_index)
    print(np.linalg.inv(link6) @ wrist)
    # print(X, c2r)
    for i in range(len(A_list)):
        err = A_list[i] @ X - X @ B_list[i]
        if np.max(err) > 0.15:
            print(A_list[i] @ X - X @ B_list[i])