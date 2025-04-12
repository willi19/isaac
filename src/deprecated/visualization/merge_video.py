import cv2
import numpy as np
import os
from paradex.utils.io import download_dir, home_dir
import argparse
import subprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, nargs="+", default=None, help="Object name(s) to process")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing videos")
    args = parser.parse_args()

    if args.name is None:
        args.name = os.listdir("/home/temp_id/shared_data/processed")

    for obj_name in args.name:
        index_list = os.listdir(os.path.join(download_dir, "processed", obj_name))
        for index in index_list:

            pose_video = cv2.VideoCapture(f"video/pose/{obj_name}/{index}.mp4")
            contact_video = cv2.VideoCapture(f"video/contact/{obj_name}/{index}.mp4")
            grid_video = cv2.VideoCapture(f"video/grid/{obj_name}/{index}.mp4")
            graph_video = cv2.VideoCapture(f"video/contact_graph/{obj_name}/{index}.mp4")

            if not args.overwrite and os.path.exists(f"video/final/{obj_name}/{index}.mp4"):
                print(f"Skipping {obj_name}/{index} (video exists)")
                continue

            os.makedirs(f"video/final/{obj_name}", exist_ok=True)
            output_video_path = f"video/final/{obj_name}/{index}.mp4"
            temp_video_path = f"video/final/{obj_name}/{index}_tmp.mp4"

            merged_video = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 7, (4096, 1536))
            
            cnt = 0
            while True:
                finished = True
                ret, pose_img = pose_video.read()
                if not ret:
                    pose_img = np.zeros((768, 1024, 3), dtype=np.uint8)
                else:
                    finished = False

                ret, contact_img = contact_video.read()
                if not ret:
                    contact_img = np.zeros((768, 1024, 3), dtype=np.uint8)
                else:
                    finished = False

                ret, grid_img = grid_video.read()
                if not ret:
                    grid_img = np.zeros((1536, 2048, 3), dtype=np.uint8)
                else:
                    finished = False

                ret, graph_img = graph_video.read()
                if not ret:
                    graph_img = np.zeros((768, 2048, 3), dtype=np.uint8)
                else:
                    finished = False

                if finished:
                    break
                
                cnt += 1
                merge_img = np.zeros((1536, 4096, 3), dtype=np.uint8)

                pose_img = cv2.resize(pose_img, (1024, 768))
                contact_img = cv2.resize(contact_img, (1024, 768))
                grid_img = cv2.resize(grid_img, (2048, 1536))
                graph_img = cv2.resize(graph_img, (2048, 768))

                merge_img[:768, 2048:] = graph_img
                merge_img[768:, 3072:] = pose_img
                merge_img[768:, 2048:3072] = contact_img
                merge_img[:, :2048] = grid_img

                merged_video.write(merge_img)


            merged_video.release()
            pose_video.release()
            contact_video.release()
            grid_video.release()
            graph_video.release()
            print(f"Temporary video saved: {temp_video_path}")
            if cnt == 0:
                print(f"Skipping {obj_name}/{index} (no frames)")
                os.remove(temp_video_path)
                continue
            # ✅ FFmpeg을 사용하여 H.264로 변환
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",  # 기존 파일 덮어쓰기
                "-i", temp_video_path,  # 입력 파일
                "-c:v", "libx264",  # 비디오 코덱: H.264
                "-preset", "slow",  # 압축률과 속도 조절 (slow = 고품질)
                "-crf", "23",  # 품질 설정 (낮을수록 고품질, 18~23 추천)
                "-pix_fmt", "yuv420p",  # 픽셀 포맷 (H.264 표준 호환)
                output_video_path
            ]

            # FFmpeg 실행
            try:
                subprocess.run(ffmpeg_cmd, check=True)
                print(f"✅ H.264 encoded video saved: {output_video_path}")
                os.remove(temp_video_path)  # 변환 후 임시 파일 삭제
            except subprocess.CalledProcessError as e:
                print(f"❌ FFmpeg encoding failed: {e}")

