import os
import json
from paradex.utils.merge_video import merge_video_synced
from paradex.utils.io import home_dir
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge video files.")
    parser.add_argument("--name", type=str, nargs="+", help="List of objects to merge.", default=None)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")

    args = parser.parse_args()

    if args.name is None:
        args.name = os.listdir(os.path.join(home_dir, "download", "processed"))

    for name in args.name:
        root_dir = os.path.join(home_dir, "download", "processed", name)
        ind_list = os.listdir(root_dir)
        for ind in ind_list:
            input_dir = os.path.join(root_dir, ind, "video")
            output_file = os.path.join("video","grid", name, f"{ind}.mp4")
            if not args.overwrite and os.path.exists(output_file):
                print(f"Skipping existing file: {output_file}")
                continue
            try:
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                merge_video_synced(input_dir, output_file)
            except Exception as e:
                print(f"Error processing {input_dir}: {e}")
                continue
            print(f"Processed {input_dir}")
