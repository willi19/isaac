import os
from dex_robot.utils.file_io import shared_path

def check_finished(name, index):
    return os.path.exists(f"{shared_path}/video/final/{name}/{index}.mp4")