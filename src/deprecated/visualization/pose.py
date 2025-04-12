import os
import time
import pinocchio as pin
from dex_robot.simulate.simulator import simulator
from dex_robot.retargeting.retargeting_config import RetargetingConfig
from dex_robot.utils.file_io import (
    load_contact_value,
    load_robot_traj
)
import numpy as np
import argparse
import subprocess

# Viewer setting
save_video = True
save_state = False
view_physics = False
view_replay = True
headless = False

simulator = simulator(
    None,
    view_physics,
    view_replay,
    headless,
    save_video,
    save_state,
    fixed=False,
    add_plane=False
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, nargs="+", default=None, help="Object name(s) to process")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")

    args = parser.parse_args()
    if args.name is None:
        args.name = os.listdir("/home/temp_id/shared_data/processed")

    for obj_name in args.name:
        demo_path = f"/home/temp_id/shared_data/processed/{obj_name}"
        demo_path_list = os.listdir(demo_path)
        for demo_name in demo_path_list:
            video_path = f"/home/temp_id/isaac/video/pose/{obj_name}/{demo_name}.mp4"
            save_path = f"result/{obj_name}/{demo_name}"
            if not args.overwrite and os.path.exists(video_path):
                print(f"Skipping existing file: {video_path}")
                continue
                
            temp_path = video_path.replace(".mp4", "_temp.mp4") 
            simulator.load_camera()
            simulator.set_savepath(temp_path, save_path)
            
            robot_traj = load_robot_traj(os.path.join(demo_path, demo_name))
            T = robot_traj.shape[0]

            for step in range(T):
                target_action = np.zeros(22)
                target_action[6:] = robot_traj[step, 6:]
                simulator.step(target_action, target_action, None)#robot_traj[step], obj_traj[step])

            print(demo_name, obj_name)
            simulator.save()

            # Convert to H.264
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",  # Overwrite existing file
                "-i", temp_path,  # Input file
                "-c:v", "libx264",  # Video codec: H.264
                "-preset", "slow",  # Compression rate and speed control (slow = high quality)
                "-crf", "23",  # Quality setting (lower is higher quality, recommended 18~23)
                "-pix_fmt", "yuv420p",  # Pixel format (H.264 standard compatible)
                video_path
            ]

            try:
                subprocess.run(ffmpeg_cmd, check=True)
                print(f"✅ H.264 encoded video saved: {video_path}")
                os.remove(temp_path)  # 변환 후 임시 파일 삭제
            except subprocess.CalledProcessError as e:
                print(f"❌ FFmpeg encoding failed: {e}")
            
            print("Video processing complete. Output saved as", video_path)

