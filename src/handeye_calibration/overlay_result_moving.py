import cv2
import numpy as np
import sys
import argparse
import os

import open3d as o3d
import time
from dex_robot.utils.file_io import shared_path, load_camparam, load_c2r

from dex_robot.renderer.robot_module import robot_info, robot_asset_file, Robot_Module
from dex_robot.renderer.vis_utils_o3d import simpleViewer, get_camLinesSet
import torch
from paradex.utils.io import find_latest_directory
import tqdm
import pyrender
import trimesh

device = torch.device("cuda:0")


def make_grid(img_list, vis_cam_list, org_height, org_width):
    if torch.is_tensor(img_list[0]):
        images_np = torch.stack(img_list).squeeze().detach().cpu().numpy()
    else:
        images_np = np.stack(img_list)
    c_start = int(np.sqrt(len(vis_cam_list)))
    for closest_factor in range(c_start,0,-1):
        if len(vis_cam_list)%closest_factor==0:
            break
    grid_h, grid_w = closest_factor, int(len(vis_cam_list)/closest_factor)
    grid = images_np.reshape(grid_h, grid_w, org_height, org_width, 3).swapaxes(1, 2).reshape(grid_h * org_height, grid_w * org_width, 3)
        
    return grid

if __name__ == "__main__":
    device = torch.device("cuda:0")

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=None)
    root_path = os.path.join(shared_path, 'handeye_calibration')
    args = parser.parse_args()
    if args.name is not None:
        name = args.name
    else:
        name = find_latest_directory(root_path)

    intrinsic, extrinsic = load_camparam(os.path.join(shared_path, 'handeye_calibration', name, '0'))
    cam_list = list(intrinsic.keys())
    c2r = load_c2r(os.path.join(shared_path, 'handeye_calibration', name, "0"))

    index_list = os.listdir(os.path.join(shared_path, 'handeye_calibration', name))

    vis = simpleViewer()
    
    cam_lineset_list = get_camLinesSet(extrinsic, length=0.05)
    vis.add_geometries_list(cam_lineset_list)

    for index in tqdm.tqdm(index_list):
        start_time = time.time()
        robot_traj = np.load(os.path.join(shared_path, 'handeye_calibration', name, index, 'robot.npy')).reshape((1, 22))
        robot_module = Robot_Module(state = robot_traj)
        mesh_list = robot_module.get_mesh(0, c2r)

        combined_mesh = None
        for mesh in mesh_list:
            if combined_mesh is None:
                combined_mesh = mesh
            else:
                combined_mesh+=mesh

        combined_mesh.paint_uniform_color([1.0, 0.0, 0.0])
        vis.add_geometries_dict({"name": "robot", "geometry": combined_mesh})

        vis.tick()

        if vis.reset_flag:
            vis.run()

        vis.remove_geometries("robot")
        time.sleep(0.1)

        # for cam_id in cam_list:  # cam_list는 24개
        #     K = intrinsic[cam_id]['intrinsics_undistort']  # 3x3
        #     E = extrinsic[cam_id]  # 4x4
        #     E = np.vstack((E, np.array([0, 0, 0, 1])))  # 4x4
            
        #     color_img = render_trimesh_fast(combined_mesh_trimesh, extrinsic=E, intrinsic=K, img_size=(1536, 2048))  # (H, W, 3)

        #     rgb_path = os.path.join(shared_path, 'handeye_calibration', name, index, 'undist_image', f'{cam_id}.png')
        #     org_rgb = cv2.imread(rgb_path)
        #     if org_rgb is None:
        #         continue

        #     # 실루엣 기반 마스크 만들기 (간단한 차이 기반)
        #     mask = (color_img != 255).any(axis=2).astype(np.uint8)  # 흰 배경에서 다른 값이면 mesh가 찍힌 영역
        #     mask_3ch = np.stack([mask]*3, axis=2)

        #     alpha = 0.5
        #     blended = org_rgb.copy()
        #     blended[mask_3ch == 1] = (
        #         alpha * color_img[mask_3ch == 1] + (1 - alpha) * org_rgb[mask_3ch == 1]
        #     ).astype(np.uint8)

        #     debug_dir = os.path.join(shared_path, 'handeye_calibration', name, index, 'debug')
        #     # os.makedirs(debug_dir, exist_ok=True)
        #     # cv2.imwrite(os.path.join(debug_dir, f"{cam_id}.png"), blended)
        #     cv2.imshow("blended", color_img)
        #     cv2.waitKey(0)