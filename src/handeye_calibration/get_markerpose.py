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
from dex_robot.utils.robot_wrapper import RobotWrapper
from dex_robot.utils.file_io import rsc_path

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None, help="Name of the calibration directory.")
    args = parser.parse_args()
    if args.name is None:
        args.name = find_latest_directory(handeye_calib_path)
    
    he_calib_path = os.path.join(handeye_calib_path, args.name)
    index_list = os.listdir(os.path.join(he_calib_path))
    C2R = np.load(os.path.join(he_calib_path, "0", "C2R.npy"))
    marker_pos = {}
    robot = RobotWrapper(
        os.path.join(rsc_path, "xarm6", "xarm6_allegro_wrist_mounted_rotate.urdf")
    )

    marker_id_list = [261,262,263,264,265,266]
    finger_id_list = [11,13,14]
    finger_marker = {11:"ring_proximal", 13:"middle_proximal", 14:"index_proximal", 10:"thumb_proximal"}
    finger_index = {f"{finger_name}_proximal":robot.get_link_index(f"{finger_name}_proximal") for finger_name in ["thumb", "index", "middle", "ring"]}


    for idx in index_list:
        link5_pose = np.load(os.path.join(he_calib_path, idx, "link5.npy"))
        marker_dict = np.load(os.path.join(he_calib_path, idx, "marker_3d.npy"), allow_pickle=True).item()
        robot_action = np.load(os.path.join(he_calib_path, idx, "robot.npy"))
        
        robot.compute_forward_kinematics(robot_action)

        for mid in marker_id_list:
            if mid not in marker_dict:
                continue
            if mid not in marker_pos:
                marker_pos[mid] = []
            # marker_dict[mid] :4x3
            marker_pos[mid].append(np.linalg.inv(link5_pose) @ np.linalg.inv(C2R) @ np.hstack((marker_dict[mid], np.ones((marker_dict[mid].shape[0], 1)))).T)

        for fid in finger_id_list:
            if fid not in marker_dict:
                continue
            if fid not in marker_pos:
                marker_pos[fid] = []
            # marker_dict[mid] :4x3
            finger_pose = robot.get_link_pose(finger_index[finger_marker[fid]])
            marker_pos[fid].append(np.linalg.inv(finger_pose) @ np.linalg.inv(C2R) @ np.hstack((marker_dict[fid], np.ones((marker_dict[fid].shape[0], 1)))).T)

    for mid in marker_pos:
        print(np.std(marker_pos[mid], axis=0), mid)
        marker_pos[mid] = np.mean(marker_pos[mid], axis=0)
    np.save(os.path.join(he_calib_path, "0", "marker_pos.npy"), marker_pos)