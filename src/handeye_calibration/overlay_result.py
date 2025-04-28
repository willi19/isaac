import cv2
import numpy as np
import sys
from pathlib import Path
import argparse
import json
import os
import pickle

import open3d as o3d
import time
from dex_robot.utils.file_io import shared_path, load_camparam, load_c2r
from dex_robot.renderer.scene import Scene
from dex_robot.renderer.vis_utils import load_ply_as_pytorch3d_mesh
from dex_robot.renderer.renderer_utils import Batched_RGB_Silhouette_Renderer, convert_mesho3d2py3d


from dex_robot.renderer.robot_module import robot_info, robot_asset_file, Robot_Module
import torch
from paradex.utils.io import find_latest_directory
import tqdm
import pyrender

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
    renderer_list, cam_id_list_list = get_renderer(intrinsic, extrinsic)

    index_list = os.listdir(os.path.join(shared_path, 'handeye_calibration', name))
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
        combined_mesh_py3d, _,_,_ = convert_mesho3d2py3d(combined_mesh, device)
        print("Conversion time: ", time.time() - start_time)
        with torch.no_grad():
            for cam_id_list, renderer in zip(cam_id_list_list, renderer_list):
                start_time = time.time()
                rendered_rgb_batch, rendered_silhouette = renderer.render(combined_mesh_py3d)
                print("Rendering time: ", time.time() - start_time)
                for i, cam_id in enumerate(cam_id_list):
                    rendered_rgb = rendered_rgb_batch[i].cpu().numpy() * 255
                    rendered_rgb = rendered_rgb.astype(np.uint8)

                    org_rgb = cv2.imread(os.path.join(shared_path, 'handeye_calibration', name, index, 'undist_image', f'{cam_id}.png'))
                    # org_rgb = cv2.resize(org_rgb, (1536, 2048))
                    # org_rgb_list.append(org_rgb)
                    # 실루엣: shape (H, W), 값은 0 또는 255
                    alpha = 0.5  # 0.0 ~ 1.0 사이, 비율 조절 가능

                    # 실루엣 마스크: 1이면 로봇 영역
                    mask = (rendered_silhouette[i].cpu().numpy() > 0.5).astype(np.uint8)  # shape: (H, W)
                    mask_3ch = np.stack([mask]*3, axis=2)  # shape: (H, W, 3)

                    # 알파 블렌딩 (mask된 부분만)
                    combined_rgb = org_rgb.copy()
                    combined_rgb[mask_3ch == 1] = (
                        alpha * rendered_rgb[mask_3ch == 1] + (1 - alpha) * org_rgb[mask_3ch == 1]
                    ).astype(np.uint8)

                    # 저장
                    os.makedirs(os.path.join(shared_path, 'handeye_calibration', name, index, 'overlay'), exist_ok=True)
                    cv2.imwrite(os.path.join(shared_path, 'handeye_calibration', name, index, 'overlay', f'{cam_id}.png'), combined_rgb)
