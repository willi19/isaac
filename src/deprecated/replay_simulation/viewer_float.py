import os
import time
from dex_robot.simulate.simulator import simulator
from dex_robot.utils.file_io_prev import (
    load_robot_traj,
    load_obj_traj,
    load_robot_target_traj,
)
from dex_robot.retargeting.retargeting_config import RetargetingConfig
from dex_robot.utils.data_parser import load_robotwrist_pose
import numpy as np

# Viewer setting
obj_name = "bottle"
save_video = False
save_state = False
view_physics = False
view_replay = True
headless = False

simulator = simulator(
    obj_name,
    view_physics,
    view_replay,
    headless,
    save_video,
    save_state,
    fixed=False
)

demo_path = f"data/teleoperation/{obj_name}"
demo_path_list = os.listdir(demo_path)

dof_names_list = simulator.get_dof_names()


try:
    for demo_name in demo_path_list:
        robot_traj = load_robot_traj(os.path.join(demo_path, demo_name))
        obj_traj = load_obj_traj(os.path.join(demo_path, demo_name))[obj_name]
        target_traj = load_robot_target_traj(os.path.join(demo_path, demo_name))

        T = robot_traj.shape[0]
        print("Demo name: ", demo_name)

        start = time.time()

        video_path = f"video/{obj_name}/{demo_name}.mp4"
        save_path = f"result/{obj_name}/{demo_name}.mp4"

        simulator.set_savepath(video_path, save_path)

        for step in range(T):
            target_pose = load_robotwrist_pose(target_traj[step])

            simulator.step(target_traj[step], target_traj[step], obj_traj[step])

        print(time.time() - start, "render seconds")
        print(T / 30, "original seconds")
        simulator.save()
finally:
    simulator.terminate()
