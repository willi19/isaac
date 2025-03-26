import os
import time
import pinocchio as pin
from dex_robot.simulate.simulator import simulator
from dex_robot.retargeting.retargeting_config import RetargetingConfig
from dex_robot.utils.file_io import (
    load_contact_value,
    load_robot_traj
)
from dex_robot.utils.data_parser import load_robotwrist_pose
import numpy as np
from dex_robot.utils.metric import get_pickplace_timing, get_grasp_timing
from dex_robot.utils import robot_wrapper

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


contact_sensor_idx = {7: 12, 9:5, 10:4, 12:3, 14:8, 15:7, 17:6, 19:11, 20:10, 22:9, 25:2, 26:1, 27:0}
#     robot contact_sensor
#       7     12, 13, 14
#       9         5
#      10         4
#      12         3
#      14         8
#      15         7
#      17         6
#      19        11
#      20        10
#      22         9
#      25         2
#      26         1
#      27         0

# obj_name_list = ["greenbowl", "bluecar","clock","donut_light","lotionbottle","orangebox","pencilcase"]
obj_name_list = ["bluecar","clock","lotionbottle","orangebox","pencilcase"]

for obj_name in obj_name_list:
    demo_path = f"/home/temp_id/shared_data/processed/{obj_name}"
    demo_path_list = os.listdir(demo_path)
    for demo_name in demo_path_list:
        video_path = f"/home/temp_id/isaac/video/contact/{obj_name}/{demo_name}"
        save_path = f"result/{obj_name}/{demo_name}.mp4"

        simulator.load_camera()
        simulator.set_savepath(video_path, save_path)
        contact_value = load_contact_value(os.path.join(demo_path, demo_name))
        robot_traj = load_robot_traj(os.path.join(demo_path, demo_name))
        T = robot_traj.shape[0]

        for step in range(T):
            target_action = np.zeros(22)
            target_action[6:] = robot_traj[step, 6:]
            simulator.step(target_action, target_action, None)#robot_traj[step], obj_traj[step])
            
            color_dict = {}
            for ri, ci in contact_sensor_idx.items():
                val = (contact_value[step, ci]-contact_value[0,ci]) / (1000)
                if val < 0:
                    val = 0
                if val > 1:
                    val = 1

                color_dict[ri] = (val,0, 1-val)
            simulator.set_color(color_dict)
        print(demo_name, obj_name)
        simulator.save()

