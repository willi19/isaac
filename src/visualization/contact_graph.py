import os
import argparse
from dex_robot.visualization.contact_graph import plot_contact_data
max_time_seconds = 3  # Duration to plot in seconds
sampling_rate = 30  # Adjust based on actual data rate (e.g., 100 Hz)
frame_rate = 30  # Frames per second for the video


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, nargs="+", default=None, help="Object name(s) to process")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing videos")
    args = parser.parse_args()

    if args.name is None:
        args.name = os.listdir("/home/temp_id/shared_data/processed")

    for obj_name in args.name:
        demo_path_root = f"/home/temp_id/shared_data/processed/{obj_name}"
        index_list = os.listdir(demo_path_root)
        
        for demo_name in index_list:
            demo_path = os.path.join(demo_path_root, demo_name)
            output_video_path = f"video/contact_graph/{obj_name}/{demo_name}.mp4"
            if not args.overwrite and os.path.exists(output_video_path):
                print(f"Skipping {demo_path} (video exists)")
                continue
            plot_contact_data(demo_path, output_video_path, sampling_rate, frame_rate)
            # except Exception as e:
            #     print(f"Error processing {demo_path}: {e}")
            #     continue
            # print(f"Processed {demo_path}")