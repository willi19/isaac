import os
import shutil
from dex_robot.utils.file_io import load_camparam, load_c2r, shared_path

homedir = os.path.expanduser("~")

# c2r_dir = "/home/temp_id/shared_data/handeye_calibration/20250324_023211/0/C2R.npy"
# cam_param_dir = "/home/temp_id/shared_data/cam_param/20250323204657"

obj_list = os.listdir("/home/temp_id/shared_data/capture")

for obj_name in obj_list:
    index_list = os.listdir(f"/home/temp_id/shared_data/capture/{obj_name}")
    for index in index_list:
        c2r_dir = os.path.join(shared_path, "capture", obj_name, index, "C2R.npy")
        cam_param_dir = os.path.join(shared_path, "capture", obj_name, index, "cam_param")

        if not os.path.exists(c2r_dir):
            print(f"{c2r_dir} not found")
            continue
        if not os.path.exists(cam_param_dir):
            print(f"{cam_param_dir} not found")
            continue
        
        for target_index in range(int(index)*5, int(index)*5+5):
            target_dir = f"/home/temp_id/shared_data/processed/{obj_name}/{target_index}"
            if not os.path.exists(target_dir):
                continue
            if not os.path.exists(f"{target_dir}/C2R.npy"):
                # os.remove(f"{target_dir}/C2R.npy")
                shutil.copy(c2r_dir, f"{target_dir}/C2R.npy")

            if not os.path.exists(f"{target_dir}/cam_param"):
                # shutil.rmtree(f"{target_dir}/cam_param")
                shutil.copytree(cam_param_dir, f"{target_dir}/cam_param")