import os
import argparse

import numpy as np
import json
import cv2
from dex_robot.utils.file_io import shared_path, load_camparam, load_c2r, download_path, rsc_path
from paradex.utils.marker import detect_aruco, triangulate, ransac_triangulation
import tqdm
from dex_robot.utils.robot_wrapper import RobotWrapper
from dex_robot.visualization.grid_image import grid_image

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=None)
    args = parser.parse_args()

    name_list = [args.name] if args.name else os.listdir(os.path.join(download_path, 'processed'))
    
    robot = RobotWrapper(
        os.path.join(rsc_path, "xarm6", "xarm6_allegro_wrist_mounted_rotate.urdf")
    )
    link_index = robot.get_link_index("link5")
    marker_id = [262,263,264,265,266]

    for name in name_list:
        index_list = os.listdir(os.path.join(download_path, 'processed', name))
        
        for index in index_list:
            root_path = os.path.join(download_path, 'processed', name, index)
            
            intrinsic_list, extrinsic_list = load_camparam(root_path)
            cammat = {}
            for serial_num in list(intrinsic_list.keys()):
                int_mat = intrinsic_list[serial_num]["intrinsics_undistort"]
                ext_mat = extrinsic_list[serial_num]
                cammat[serial_num] = int_mat @ ext_mat

            c2r = load_c2r(root_path)
            robot_action = np.load(os.path.join(root_path, "arm", "state.npy"))
            hand_action = np.load(os.path.join(root_path, "hand", "state.npy"))

            marker_pose = np.load(os.path.join(root_path, "marker_pos.npy"), allow_pickle=True).item()
            
            serial_list = list(intrinsic_list.keys())
            serial_list.sort()

            seq_len = len(os.listdir(os.path.join(root_path, "video_extracted", serial_list[0])))

            for serial_num in serial_list:
                if len(os.listdir(os.path.join(root_path, "video_extracted", serial_num))) != seq_len:
                    print("video_extracted error: ", serial_num)
                    break
            
            os.makedirs(os.path.join(root_path, "marker3d"), exist_ok=True)

            for fid in tqdm.tqdm(range(seq_len)): 
                if not os.path.exists(os.path.join(root_path, "marker3d", f"{fid:05d}.npy")):
                    continue
                marker_3d = np.load(os.path.join(root_path, "marker3d", f"{fid:05d}.npy"), allow_pickle=True).item()
                img_list = {}

                qpos = np.concatenate([robot_action[fid], hand_action[fid]])
                robot.compute_forward_kinematics(qpos)
                link5_pose = robot.get_link_pose(link_index)    

                show = False
                for mid in marker_id:
                    if mid not in marker_3d or marker_3d[mid] is None:
                        continue
                    show = True
                if not show:
                    continue
                show = False

                for serial_num in serial_list:
                    intrinsic = intrinsic_list[serial_num]
                    img = cv2.imread(os.path.join(root_path, "video_extracted", serial_num, f"{fid:05d}.jpeg"))
                    undistort_img = cv2.undistort(img, intrinsic["intrinsics_original"], intrinsic["dist_params"])
                    
                    undist_kypt, ids = detect_aruco(undistort_img)
                    if ids is None:
                        continue
                    
                    show = False
                    for mid in marker_id:
                        if mid not in ids:
                            continue
                        show = True
                    if not show:
                        continue
                    
                    img_tmp = undistort_img.copy()
                    
                    for cor, id in zip(undist_kypt, ids):
                        if id not in marker_id or int(id) not in marker_3d.keys():
                            continue
                        cor = cor.squeeze().astype(int)
                        
                        for i in range(4):
                            cv2.circle(img_tmp, tuple(cor[i]), 5, (0, 0, 255), -1)
                        cv2.putText(img_tmp, str(id), tuple(cor[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            
                        proj_mtx = cammat[serial_num]
                        pt_3d = marker_3d[int(id)]
                        if pt_3d is None:
                            continue
                        pt_3d_hom = np.concatenate([pt_3d, np.ones((4, 1))], axis=1)
                        for i in range(4):
                            pt_2d = proj_mtx @ pt_3d_hom[i].T
                            pt_2d = (pt_2d / pt_2d[2])[:2]
                            cv2.circle(img_tmp, (int(pt_2d[0]),int(pt_2d[1])), 5, (255, 0, 0), -1)  # color blue 
                    
                        marker_pose_h = c2r @ link5_pose @ marker_pose[int(id)]
                        marker_pose_h /= marker_pose_h[-1]
                        marker_pose_h = marker_pose_h[:3].T

                        proj_mtx = cammat[serial_num]
                        pt_3d_hom = np.concatenate([marker_pose_h, np.ones((4, 1))], axis=1)
                        pt_2d = proj_mtx @ pt_3d_hom.T
                        pt_2d = (pt_2d / pt_2d[2])[:2]
                        pt_2d = pt_2d.T

                        for i in range(4):
                            cv2.circle(img_tmp, tuple(pt_2d[i].astype(int)), 5, (0, 255, 0), -1) # color green

                    os.makedirs(os.path.join(root_path, "reproj", serial_num), exist_ok=True)
                    cv2.imwrite(os.path.join(root_path, "reproj", serial_num, f"{fid:05d}.jpeg"), img_tmp)
                    # img_list[serial_num] = img_tmp
                # if len(img_list) == 0:
                #     continue
                # g_image = grid_image(img_list, serial_list)
                # cv2.imshow("original", g_image)
                # cv2.waitKey(0)
                # os.makedirs(os.path.join(root_path, "reproj"), exist_ok=True)
                # cv2.imwrite(os.path.join(root_path, "reproj", f"{fid:05d}.jpeg"), g_image)
                    # for mid in cor_3d.keys():
                    #     # if mid not in ids or cor_3d[mid] is None:
                    #     #     continue
                    #     proj_mtx = cammat[serial_num]
                    #     pt_3d = cor_3d[mid]
                    #     if pt_3d is None:
                    #         continue
                    #     pt_3d_hom = np.concatenate([pt_3d, np.ones((4, 1))], axis=1)

                    #     for i in range(4):
                    #         pt_2d = proj_mtx @ pt_3d_hom[i].T
                    #         pt_2d = (pt_2d / pt_2d[2])[:2]
                    #         cv2.circle(img_tmp, (int(pt_2d[0]),int(pt_2d[1])), 5, (255, 0, 0), -1)
                        
        
                    # img_tmp = cv2.resize(img_tmp, (1024, 768))
                    # cv2.imshow("original", img_tmp)
                    # cv2.waitKey(0)
                    
            # for i in range(seq_len):
            #     marker_3d = np.load(os.path.join(root_path, "marker3d", f"{i:05d}.npy"), allow_pickle=True).item()
            #     print(marker_3d)
            # timestamp_camera = []
            # for fid in tqdm.tqdm(range(seq_len)):
            #     marker_3d = np.load(os.path.join(root_path, "marker3d", f"{fid:05d}.npy"), allow_pickle=True).item()
            #     if len(marker_3d) == 0:
            #         continue
            #     timestamp_camera.append({'time':timestamp[fid], 'marker_3d': marker_3d})

            # robot_timestamp = np.load(os.path.join(shared_path, "capture", name, capture_ind[int(index)], "arm", "timestamp.npy"))
            # robot_value = np.load(os.path.join(shared_path, "capture", name, capture_ind[int(index)], "arm", "state.npy"))
            
            # time_delta = [i*0.01 for i in range(15)]
            # for td in time_delta:
            #     ri = 0
            #     err = 0
            #     cnt = 0
            #     for i in range(len(timestamp_camera)):
            #         while ri < len(robot_timestamp)-1 and abs(robot_timestamp[ri] - timestamp_camera[i]["time"] + td) > abs(robot_timestamp[ri+1] - timestamp_camera[i]["time"] + td):
            #             ri += 1
            #         qpos = np.zeros(22)
            #         qpos[:6] = robot_value[ri]

            #         robot.compute_forward_kinematics(qpos)
            #         link_pose = robot.get_link_pose(link_index)
            #         for marker_id in timestamp_camera[i]["marker_3d"].keys():
            #             cam_marker_pose = timestamp_camera[i]["marker_3d"][marker_id]
            #             if cam_marker_pose is None:
            #                 continue
                        
            #             robot_marker_pose = c2r @ link_pose @ marker_pose[marker_id]
            #             robot_marker_pose = robot_marker_pose[:3, :] / robot_marker_pose[3, :]
            #             robot_marker_pose = robot_marker_pose.T
            #             # print(np.linalg.norm(robot_marker_pose - cam_marker_pose))
            #             err += np.linalg.norm(cam_marker_pose - robot_marker_pose, axis=1).mean()
            #             cnt += 1
            #             # print(i, np.linalg.norm(cam_marker_pose - robot_marker_pose, axis=1).mean(), marker_id)
            #             if np.linalg.norm(cam_marker_pose - robot_marker_pose, axis=1).mean() >0.4:
            #                 print(abs(robot_timestamp[ri] - timestamp_camera[i]["time"] + td), "asdfsadf")
            #                 import pdb; pdb.set_trace()
            #             # print(np.linalg.norm(cam_marker_pose - robot_marker_pose, axis=1).sum())
            #     print("time delta: ", td, "error: ", err / cnt)
                        
