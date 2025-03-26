import os
from dex_robot.process.check_finished import check_finished
from paradex.utils.io import shared_dir, home_dir, download_dir
from paradex.utils.upload_file import get_total_size, copy_to_nfs
import json

pc_name = home_dir.split("/")[-1]
pc_list = json.load(open(f"{shared_dir}/current_pc.json", "r"))["current_pc"]
pc_len = len(pc_list)
obj_list = os.listdir("/home/temp_id/shared_data/processed")

cnt = 0
for obj_name in obj_list:
    index_list = os.listdir(f"/home/temp_id/shared_data/processed/{obj_name}")
    for index in index_list:
        if check_finished(obj_name, index):
            print(f"Skipping {obj_name}/{index} (video exists)")
            continue
        if pc_list[cnt%pc_len] != pc_name:
            cnt += 1
            continue
        source_path = f"{shared_dir}/processed/{obj_name}/{index}"
        destination_path = f"{download_dir}/processed/{obj_name}/{index}"

        total_size = get_total_size(source_path, destination_path)
        copy_to_nfs(source_path, destination_path, total_size)
        cnt += 1
