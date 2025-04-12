from dex_robot.renderer.scene import Scene
from dex_robot.utils.file_io import download_path
from dex_robot.renderer.robot_module import Robot_Module
from dex_robot.renderer.render_utils import convert_mesho3d2py3d
import os
import argparse
import numpy as np
import torch
import cv2

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

device = torch.device("cuda:0")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=None)
    args = parser.parse_args()

    name_list = [args.name] if args.name else os.listdir(os.path.join(download_path, 'processed'))
    for name in name_list:
        root_path = os.path.join(download_path, 'processed', name)
        scene = Scene(root_path)
        scene.get_renderer()
        robot_module = Robot_Module(state = scene.robot_traj)
        cam_params = scene.cam_params

        grid_save_dir = 'video/grid_save'
        os.makedirs(grid_save_dir, exist_ok=True)

        for fidx in range(scene.ttl_frame_length):
            robot_mesh_list = robot_module.get_mesh(fidx, scene.C2R)

            combined_mesh = None
            for mesh in robot_mesh_list:
                if combined_mesh is None:
                    combined_mesh = mesh
                else:
                    combined_mesh+=mesh

            # if args.save_mesh:
            #     o3d.io.write_triangle_mesh(str(mesh_save_dir/('%05d.ply'%(fidx))), combined_mesh)

            cam_id = scene.cam_ids[0]
            combined_mesh_py3d, _,_,_ = convert_mesho3d2py3d(combined_mesh, device)

            rendered_rgb_list = []
            org_rgb_list = []
            rendered_silhouette_list = []

            # img_dict = scene.get_images(fidx)

            vis_cam_list = args.tg_cams
            with torch.no_grad():
                for cam_id in  vis_cam_list: #scene.cam_ids:
                    print(f'render {cam_id}')
                    rendered_rgb, rendered_silhouette = scene.render(cam_id, combined_mesh_py3d)# render silhouette and rgb as well
                    rendered_rgb_list.append(rendered_rgb)
                    rendered_silhouette_list.append(rendered_silhouette)
                    org_rgb = cv2.imread(os.path.join(args.root_path,'video_extracted/%s/%05d.jpeg'%(cam_id, fidx)))
                    org_rgb = cv2.resize(org_rgb, (scene.width, scene.height))
                    org_rgb_list.append(org_rgb)


            # org_rgb_list = []
            # for cam_id in vis_cam_list:
            #     img_path = root_path/'video_extracted'/cam_id/('%05d.jpeg'%(fidx))
            #     if os.path.exists(img_path):
            #         org_rgb_list.append(cv2.resize(cv2.imread(str(img_path)), (scene.width, scene.height)))
            #     else:
            #         org_rgb_list.append(np.zeros((scene.height, scene.width, 3)))


            # images_np = torch.stack(rendered_rgb_list).squeeze().detach().cpu().numpy()
            # grid_h = int(np.sqrt(len(vis_cam_list)))
            # grid_w = int(len(vis_cam_list)/grid_h)
            # grid = images_np.reshape(grid_h, grid_w, scene.height, scene.width, 3).swapaxes(1, 2).reshape(grid_h * scene.height, grid_w * scene.width, 3)
            
            rendered_grid = make_grid(rendered_rgb_list, vis_cam_list, scene.height, scene.width)
            rendered_rgb = make_grid(org_rgb_list, vis_cam_list, scene.height, scene.width)

            # cv2.imwrite(str(grid_save_dir/("%05d.png"%(fidx))), grid*255)
            # cv2.resize(grid, ( grid_w * scene.width, grid_h * scene.height))

            
            # org_rgbs_np = np.array(org_rgb_list)
            # org_rgbs_grid = org_rgbs_np.reshape(6, 4, scene.height, scene.width, 3).swapaxes(1, 2).reshape(6 * scene.height, 4 * scene.width, 3)
            # cv2.imwrite(grid_save_dir/"rgb_%05d.png"%(fidx), cv2.resize(org_rgbs_grid, ( 2 * scene.width, 3 * scene.height)))

            cv2.imwrite(str(grid_save_dir/("%05d.png"%(fidx))), np.concatenate((rendered_grid*255, rendered_rgb), axis=0))




