import os
from dex_robot.simulate.simulator import simulator
from dex_robot.utils.file_io import (
    load_contact_value,
)
import numpy as np
from dex_robot.utils import robot_wrapper
from dex_robot.contact.index import sensor_name, contact_sensor_idx
from dex_robot.contact.process import process_contact

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
            video_path = f"/home/temp_id/isaac/video/contact/{obj_name}/{demo_name}.mp4"
            save_path = f"result/{obj_name}/{demo_name}"
            if not args.overwrite and os.path.exists(video_path):
                print(f"Skipping existing file: {video_path}")
                continue

            temp_path = video_path.replace(".mp4", "_temp.mp4")

            simulator.load_camera()
            simulator.set_savepath(temp_path, save_path)
            simulator.set_color(color_dict={i: (0.4, 0.4, 0.7) for i in range(31)})
            contact_value = load_contact_value(os.path.join(demo_path, demo_name))
            contact_value = process_contact(contact_value)

            T = contact_value.shape[0]

            for step in range(T):
                target_action = np.zeros(22)
                simulator.step(target_action, target_action, None)
                color_dict = {}
                for ri, ci in contact_sensor_idx.items():
                    val = contact_value[step, ci] / 100
                    if val < 0:
                        val = 0
                    if val > 1:
                        val = 1
                    color_dict[ri] = (0.5+0.5*val, 0.5-0.2*val, 0.8-0.5*val)
                simulator.set_color(color_dict)
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

