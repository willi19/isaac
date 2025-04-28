import os
import argparse
import cv2

import numpy as np
from dex_robot.utils.file_io import download_path, load_camparam, load_c2r
import subprocess
import json
import tqdm

if __name__ == "__main__":
    passer = argparse.ArgumentParser()
    passer.add_argument('--name', type=str, default=None)
    args = passer.parse_args()

    name_list = [args.name] if args.name else os.listdir(os.path.join(download_path, 'processed'))
    for name in name_list:
        index_list = os.listdir(os.path.join(download_path, 'processed', name))
        index_list.sort()
        for index in tqdm.tqdm(index_list[2:3]):
            video_path = os.path.join(download_path, 'processed', name, index, 'video')
            video_list = os.listdir(video_path)

            # intrinsic_list, _ = load_camparam(os.path.join(download_path, 'processed', name, index))
            
            for video in tqdm.tqdm(video_list):
                if "avi" not in video:
                    continue
                video_full_path = os.path.join(video_path, video)
                if not os.path.exists(video_full_path):
                    continue
                
                serial = video.split(".")[0]
                cap = cv2.VideoCapture(video_full_path)
                # timestamp = json.load(open(os.path.join(video_path, video.split('-')[0] + "_timestamp.json")))

                save_path = os.path.join(download_path, 'processed', name, index, 'video_extracted', serial)
                os.makedirs(os.path.join(save_path), exist_ok=True)
                # intrinsic = intrinsic_list[serial]
                # for i in tqdm.tqdm(range(1, timestamp["frameID"][-1]+1)):
                #     if i not in timestamp["frameID"]:
                #         continue
                #     ret, frame = cap.read()
                #     if not ret:
                #         continue
                #     undistorted_img = cv2.undistort(frame, intrinsic["intrinsics_original"], intrinsic["dist_params"], None, intrinsic["intrinsics_undistort"])
                #     cv2.imwrite(os.path.join(save_path, f"{i:05d}.png"), undistorted_img)
                
                # fid = 0
                # while True:
                #     ret, frame = cap.read()
                #     if not ret:
                #         break
                #     # undistorted_img = cv2.undistort(frame, intrinsic["intrinsics_original"], intrinsic["dist_params"], None, intrinsic["intrinsics_undistort"])
                #     cv2.imwrite(os.path.join(save_path, f"{fid:05d}.png"), frame)
                #     fid += 1
                # cap.release()
                output_pattern = os.path.join(save_path, "%05d.png")
                cmd = [
                    "ffmpeg",
                    "-i", video_full_path,
                    "-vsync", "0",  # preserve frame timing
                    output_pattern
                ]
                subprocess.run(cmd)#, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)