import os
import argparse
from dex_robot.process.check_finished import check_finished
from dex_robot.visualization.viz_path import *
from dex_robot.visualization.plot import plot_final
from paradex.utils.upload_file import copy_to_nfs, get_total_size
from dex_robot.utils.file_io import shared_path, home_path, download_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, nargs="+", default=None, help="Object name(s) to process")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing videos")
    args = parser.parse_args()

    if args.name is None:
        args.name = os.listdir(f"{download_path}/processed")

    for obj_name in args.name:
        demo_path_root = f"{download_path}/processed/{obj_name}"
        index_list = os.listdir(demo_path_root)

        for ind in index_list:
            if check_finished(obj_name, ind):
                print(f"Skipping {obj_name}/{ind} (video exists)")
                continue
#             try:
            plot_final(obj_name, ind, args.overwrite)
            source_path = f"video/final/{obj_name}/{ind}.mp4"
            destination_path = f"{shared_path}/video/final/{obj_name}/{ind}.mp4"
            total_size = get_total_size(source_path, destination_path)  
            copy_to_nfs(source_path, destination_path, total_size)
                                            
            # except:
            #     print(f"Failed to plot {obj_name}/{ind}")
            #     continue

