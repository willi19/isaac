import cv2
import numpy as np
import os
from paradex.utils.io import download_dir, home_dir

obj_name_list = ["woodbox"]
for obj_name in obj_name_list:
    index_list = os.listdir(os.path.join(download_dir, "processed", obj_name))
    for index in index_list:
        
        grid_video = cv2.VideoCapture(f"{home_dir}/paradex/video/merged/{obj_name}/{index}.mp4")
        graph_video = cv2.VideoCapture(f"video/contact_graph_orig/{obj_name}/{index}.mp4")

        os.makedirs(f"video/final_full/{obj_name}", exist_ok=True)
        merged_video = cv2.VideoWriter(f"video/final_full/{obj_name}/{index}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (4096, 1536))
        
        while True:
            ret, grid_img = grid_video.read()
            if not ret:
                break
            ret, graph_img = graph_video.read()
            if not ret:
                break

            merge_img = np.zeros((1536, 4096, 3), dtype=np.uint8)
            graph_img = cv2.resize(graph_img, (2048, 1536))

            merge_img[:, 2048:] = graph_img
            merge_img[:, :2048] = grid_img

            merged_video.write(merge_img)
        merged_video.release()
        grid_video.release()