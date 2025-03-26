import os
from dex_robot.utils.robot_wrapper import RobotWrapper
from dex_robot.utils.file_io_prev import rsc_path
import numpy as np

def get_latest_dir(path):
    dirs = os.listdir(path)
    dirs = [d for d in dirs if os.path.isdir(os.path.join(path, d))]
    dirs.sort()
    return dirs[-1]

robot = RobotWrapper(
    os.path.join(rsc_path, "xarm6", "xarm6_allegro_wrist_mounted_rotate.urdf")
)
link_index = robot.get_link_index("link5")

home_path = os.path.expanduser("~")
he_calib_path = os.path.join(home_path,"shared_data","handeye_calibration")
name = get_latest_dir(he_calib_path)


index_list = os.listdir(os.path.join(he_calib_path,name))
for idx in index_list:
    qpos = np.load(os.path.join(he_calib_path,name,idx,"robot.npy"))
    robot.compute_forward_kinematics(qpos)
    link_pose = robot.get_link_pose(link_index)
    np.save(os.path.join(he_calib_path,name,idx,"link5.npy"),link_pose)