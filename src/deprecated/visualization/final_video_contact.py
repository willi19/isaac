import cv2
import numpy as np
import os
from paradex.utils.io import download_dir, home_dir

obj_name_list = ["contact"]
for obj_name in obj_name_list:
    index_list = os.listdir(os.path.join(download_dir, "processed", obj_name))
    for index in index_list:
        cam_video_1 = cv2.VideoCapture(f"/home/temp_id/download/processed/contact/0/video/22641023.avi")
        cam_video_2 = cv2.VideoCapture(f"/home/temp_id/download/processed/contact/0/video/22645029.avi")
        cam_video_3 = cv2.VideoCapture(f"/home/temp_id/download/processed/contact/0/video/22684253.avi")
        cam_video_4 = cv2.VideoCapture(f"/home/temp_id/download/processed/contact/0/video/23280285.avi")

        contact_video = cv2.VideoCapture(f"video/contact2/{obj_name}/{index}/0.mp4")
        graph_video = cv2.VideoCapture(f"video/contact_graph/{obj_name}/{index}.mp4")
        

        os.makedirs(f"video/final/{obj_name}", exist_ok=True)
        merged_video = cv2.VideoWriter(f"video/final/{obj_name}/{index}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (3072, 1536))

        while True:
            ret, cam_img_1 = cam_video_1.read()
            if not ret:
                break
            ret, cam_img_2 = cam_video_2.read()
            if not ret:
                break
            ret, cam_img_3 = cam_video_3.read()
            if not ret:
                break
            ret, cam_img_4 = cam_video_4.read()
            if not ret:
                break
            cam_img_1 = cv2.resize(cam_img_1, (1024, 768))
            cam_img_2 = cv2.resize(cam_img_2, (1024, 768))
            cam_img_3 = cv2.resize(cam_img_3, (1024, 768))
            cam_img_4 = cv2.resize(cam_img_4, (1024, 768))
            cam_img = np.zeros((1536, 2048, 3), dtype=np.uint8)
            cam_img[:768, :1024] = cam_img_1
            cam_img[:768, 1024:] = cam_img_2
            cam_img[768:, :1024] = cam_img_3
            cam_img[768:, 1024:] = cam_img_4

            ret, contact_img = contact_video.read()
            if not ret:
                break
            ret, graph_img = graph_video.read()
            if not ret:
                break        

            merge_img = np.zeros((1536, 3072, 3), dtype=np.uint8)

            contact_img = cv2.resize(contact_img, (1024, 768))
            graph_img = cv2.resize(graph_img, (1024, 768))

            merge_img[:768, 2048:] = graph_img
            merge_img[768:, 2048:] = contact_img
            merge_img[:, :2048] = cam_img

            merged_video.write(merge_img)
        merged_video.release()
        cam_video_1.release()
        cam_video_2.release()
        cam_video_3.release()
        cam_video_4.release()
        contact_video.release()
        graph_video.release()