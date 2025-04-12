import os
from dex_robot.utils.file_io import load_camparam, shared_path
import numpy as np
import tqdm
import subprocess
import argparse
import cv2

h, w = 1536, 2048

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, nargs="+", default=None, help="Object name(s) to process")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    args = parser.parse_args()
    if args.name is None:
        args.name = os.listdir(f"{shared_path}/processed")

    
    root_path = f"{shared_path}/processed"
    for obj_name in args.name:
        print(obj_name)
        index_list = os.listdir(f"{root_path}/{obj_name}")
        for index in index_list:
            intrinsic, _ = load_camparam(f"{root_path}/{obj_name}/{index}")

            proj_video_list = os.listdir(f"video/projection/{obj_name}/{index}")
            video_list = os.listdir(f"{root_path}/{obj_name}/{index}/video")
            
            os.makedirs(f"video/overlay/{obj_name}/{index}", exist_ok=True)

            for proj_video in tqdm.tqdm(proj_video_list):
                serial_num = proj_video.split(".")[0]
                video_name = f"{serial_num}.avi"
                out_path = f"video/overlay/{obj_name}/{index}/{serial_num}.mp4"

                if not args.overwrite and os.path.exists(out_path):
                    continue

                proj_video = cv2.VideoCapture(f"video/projection/{obj_name}/{index}/{proj_video}")
                video = cv2.VideoCapture(f"{root_path}/{obj_name}/{index}/video/{video_name}")
                temp_path = f"video/overlay/{obj_name}/{index}/{serial_num}_temp.mp4"
                out = cv2.VideoWriter(temp_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))


                int_mat = np.array(intrinsic[serial_num]['Intrinsics']).reshape(3,3)
                old_int_mat = np.array([[int_mat[0,0], 0, w//2],[0, int_mat[0,0], h//2],[0, 0, 1]])
                H = int_mat @ np.linalg.inv(old_int_mat)
                
                len1 = int(proj_video.get(cv2.CAP_PROP_FRAME_COUNT))
                len2 = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                length = min(len1, len2)

                for i in tqdm.tqdm(range(length)):
                    ret1, proj_frame = proj_video.read()
                    ret2, frame = video.read()

                    if not ret1 or not ret2:
                        break

                    # Warp the image to simulate the new intrinsics
                    warped = cv2.warpPerspective(proj_frame, H, (w, h))
                    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                    gamma=0.8
                    alpha = np.clip(gray*gamma, 0, 1)[:, :, np.newaxis]  # (H, W, 1)
                    
                    alpha = np.repeat(alpha, 3, axis=2)
                    warped = (warped * alpha).astype(np.uint8)
                    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
                    mask_inv = cv2.bitwise_not(mask)
                    frame_masked = cv2.bitwise_and(frame, frame, mask=mask)

                    # ✅ 해당 영역만 밝기 줄이기
                    dimmed_frame_masked = (frame_masked * 0.2).astype(np.uint8)

                    # ✅ 원본 frame에 다시 덮어쓰기
                    frame_dimmed = frame.copy()
                    frame_dimmed[mask > 0] = dimmed_frame_masked[mask > 0]

                    # ✅ 그 위에 warped를 합성 (그대로 밝기 조정된 배경 위에 덮기)
                    fg_region = cv2.bitwise_and(warped, warped, mask=mask)
                    # bg_region = cv2.bitwise_and(frame_dimmed, frame_dimmed, mask=cv2.bitwise_not(mask))
                    frame = cv2.add(fg_region, frame_dimmed)
                    out.write(frame)

                out.release()
                proj_video.release()
                video.release()
                try:
                    subprocess.run(["ffmpeg", "-y", "-i", temp_path, "-c:v", "libx264", out_path])
                    os.remove(temp_path)
                except:
                    print("Failed to encode video")
                