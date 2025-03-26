import os
from dex_robot.retargeting.retargeting_config import RetargetingConfig
from dex_robot.utils.file_io import (
    load_contact_value,
    load_robot_traj
)
from dex_robot.utils.data_parser import load_robotwrist_pose
import numpy as np
from dex_robot.utils.metric import get_pickplace_timing, get_grasp_timing
from dex_robot.utils import robot_wrapper
import matplotlib.pyplot as plt
# Viewer setting
obj_name = "greenbowl"




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
obj_name_list = ["contact_debug"]
for obj_name in obj_name_list:
    demo_path = f"/home/temp_id/shared_data/capture/{obj_name}"
    demo_path_list = os.listdir(demo_path)
    for demo_name in demo_path_list:
        # video_path = f"video/{obj_name}/{demo_name}.mp4"
        # save_path = f"result/{obj_name}/{demo_name}.mp4"

        # simulator.set_savepath(video_path, save_path)
        # simulator.set_camera((0.3, 0, 0), (0, 0, 0))
        # os.makedirs(f"image/contact/{obj_name}", exist_ok=True)
        print(demo_name)
        contact_value = load_contact_value(os.path.join(demo_path, demo_name))
        contact_value = contact_value[:,:15] - contact_value[0,:15]
        #contact_value = np.clip(contact_value, 0, 1000)
        os.makedirs(f"image/contact/{obj_name}", exist_ok=True)
        plt.plot(contact_value)
        plt.legend([f"sensor {i}" for i in range(15)])
        plt.savefig(f"image/contact/{obj_name}/{demo_name}.png")
        plt.close()
        # plt.show()
        # robot_traj = load_robot_traj(os.path.join(demo_path, demo_name))
        # T = robot_traj.shape[0]

        # for step in range(T):
        #     target_action = np.zeros(22)
        #     target_action[6:] = robot_traj[step, 6:]
        #     simulator.step(target_action, target_action, None)#robot_traj[step], obj_traj[step])
            
        #     color_dict = {}
        #     for ri, ci in contact_sensor_idx.items():
        #         val = contact_value[step, ci] / (128*256)
        #         color_dict[ri] = (val,0, 1-val)
        #     simulator.set_color(color_dict)

        # simulator.save()

