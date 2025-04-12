import cv2
import numpy as np
import sys
from pathlib import Path
import argparse
import json
import os
import pickle

import open3d as o3d

from dex_robot.utils.file_io import shared_path, load_camparam, load_c2r
from dex_robot.renderer.scene import Scene
from dex_robot.renderer.vis_utils import load_ply_as_pytorch3d_mesh
from dex_robot.renderer.renderer_utils import Batched_RGB_Silhouette_Renderer, convert_mesho3d2py3d


from dex_robot.renderer.robot_module import robot_info, robot_asset_file, Robot_Module
import torch
from paradex.utils.io import find_latest_directory
from dex_robot.visualization.grid_image import grid_image

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


def get_renderer(intrinsic, extrinsic, n_splits=8):
    cam_id_list = []
    
    extrinsic_list = []
    intrinsic_list = []

    for cam_id in intrinsic.keys():
        intrinsic_list.append(np.array(intrinsic[cam_id]['intrinsics_undistort']).reshape((3,3)))
        extrinsic_list.append(extrinsic[cam_id])
        cam_id_list.append(cam_id)

    intrinsic_list = np.array(intrinsic_list)
    extrinsic_list = np.array(extrinsic_list)

    intrinsic_list = torch.tensor(intrinsic_list, dtype=torch.float32, device=device)
    extrinsic_list = torch.tensor(extrinsic_list, dtype=torch.float32, device=device)
    img_sizes = torch.tensor([[1536, 2048]], device=device)

    total = len(cam_id_list)
    split_size = (total + n_splits - 1) // n_splits  # 나머지가 있어도 분할되도록

    renderers = []
    cam_id_chunks = []

    for i in range(n_splits):
        start = i * split_size
        end = min((i + 1) * split_size, total)

        renderer = Batched_RGB_Silhouette_Renderer(
            extrinsic_list[start:end],
            intrinsic_list[start:end],
            img_sizes,
            device
        )
        renderers.append(renderer)
        cam_id_chunks.append(cam_id_list[start:end])

    return renderers, cam_id_chunks


if __name__ == "__main__":
    device = torch.device("cuda:0")

    parser = argparse.ArgumentParser()
    # parser.add_argument('--root_path', type=str, default='/home/jisoo/data2/paradex_processing/captured_data/spray/0')
    root_path = os.path.join(shared_path, 'handeye_calibration')
    name = find_latest_directory(root_path)

    intrinsic, extrinsic = load_camparam(os.path.join(shared_path, 'handeye_calibration', name, '0'))
    cam_list = list(intrinsic.keys())
    c2r = load_c2r(os.path.join(shared_path, 'handeye_calibration', name, "0"))
    renderer_list, cam_id_list_list = get_renderer(intrinsic, extrinsic)

    index_list = os.listdir(os.path.join(shared_path, 'handeye_calibration', name))
    for index in index_list:
        # robot_traj = np.load(os.path.join(shared_path, 'handeye_calibration', name, index, 'robot.npy')).reshape((1, 22))
        # robot_traj[:,6:] = 0
        # robot_module = Robot_Module(state = robot_traj)
        # mesh_list = robot_module.get_mesh(0, c2r)

        # combined_mesh = None
        # for mesh in mesh_list:
        #     if combined_mesh is None:
        #         combined_mesh = mesh
        #     else:
        #         combined_mesh+=mesh

        # combined_mesh.paint_uniform_color([1.0, 0.0, 0.0])
        # combined_mesh_py3d, _,_,_ = convert_mesho3d2py3d(combined_mesh, device)

        with torch.no_grad():
            image_list = {}

            for cam_id_list, renderer in zip(cam_id_list_list, renderer_list):
                for cam_id in cam_id_list:
                    combined_rgb = cv2.imread(os.path.join(shared_path, 'handeye_calibration', name, index, 'debug', f'{cam_id}.png'))
                    image_list[cam_id] = combined_rgb

            cam_id_list = list(image_list.keys())
            cam_id_list.sort()
            grid_image = np.zeros((1536, 2048, 3), dtype=np.uint8)
            grid_w = 6
            grid_h = 4

            for idx, cam_id in enumerate(cam_id_list):
                img = image_list[cam_id]
                img = cv2.resize(img, (2048//grid_w, 1536//grid_h))
                
                row = idx // grid_w
                col = idx % grid_w
                y_start = row * (1536//grid_h)
                y_end = (row + 1) * (1536//grid_h)
                x_start = col * (2048//grid_w)
                x_end = (col + 1) * (2048//grid_w)  

                grid_image[y_start:y_end, x_start:x_end] = img

            os.makedirs(os.path.join(shared_path, 'handeye_calibration', name, 'debug_grid'), exist_ok=True)
            cv2.imwrite(os.path.join(shared_path, 'handeye_calibration', name, 'debug_grid', f'{index}.png'), grid_image)
            # cv2.imshow("grid", grid_image)
            # cv2.waitKey(0)