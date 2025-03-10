# Redefine Grasp Start & End
import numpy as np
from dex_robot.utils.metric import get_pickplace_timing
from dex_robot.utils.file_io import load_obj_traj, load_target_traj

obj_list = ["bottle", "book", "bowl"]
