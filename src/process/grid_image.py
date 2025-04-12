import os
from dex_robot.utils.file_io import download_path
import numpy as np
import cv2
import tqdm
import sys
from dex_robot.visualization.convert_codec import change_to_h264

obj_name = 'spray'

root_path = os.path.join(download_path, 'processed', obj_name)
index_list = ["1"]# os.listdir(root_path)

for index in index_list:
    image_path = os.path.join(download_path, 'processed', obj_name, index, 'video_extracted')
    serial_list = os.listdir(image_path)
    serial_list.sort()

    output_video = cv2.VideoWriter("spray_1_tmp.mp4", cv2.VideoWriter_fourcc(*'XVID'), 20, (2048, 1536))

    for fid in tqdm.tqdm(range(153)):
        image_list = {}

        for serial in serial_list:
            image_path_serial = os.path.join(image_path, serial)
            image_path_fid = os.path.join(image_path_serial, f"{fid:05d}.png")
            image_list[serial] = cv2.imread(image_path_fid)
        
        grid_image = np.zeros((1536, 2048, 3), dtype=np.uint8)
        grid_w = 6
        grid_h = 4

        for idx, cam_id in enumerate(serial_list):
            img = image_list[cam_id]
            img = cv2.resize(img, (2048//grid_w, 1536//grid_h))
            
            row = idx // grid_w
            col = idx % grid_w
            y_start = row * (1536//grid_h)
            y_end = (row + 1) * (1536//grid_h)
            x_start = col * (2048//grid_w)
            x_end = (col + 1) * (2048//grid_w)  

            grid_image[y_start:y_end, x_start:x_end] = img
        output_video.write(grid_image)  
    output_video.release()
    change_to_h264("spray_1_tmp.mp4", "spray_1.mp4")