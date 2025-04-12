import os
import time
import pinocchio as pin
from dex_robot.simulate.simulator import simulator
from dex_robot.retargeting.retargeting_config import RetargetingConfig
from dex_robot.utils.file_io_prev import (
    load_robot_traj,
    load_obj_traj,
    load_robot_target_traj,
    load_mesh
)
from dex_robot.utils.data_parser import load_robotwrist_pose
import numpy as np
from dex_robot.utils.metric import get_pickplace_timing

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
    fixed=True
)

demo_path = f"data/teleoperation/{obj_name}"
demo_path_list = os.listdir(demo_path)

dof_names_list = simulator.get_dof_names()
config_path = "teleop/xarm6_allegro_hand_right_6d.yml"
config = RetargetingConfig.load_from_file(config_path)
config.set_default_target_joint_names(dof_names_list)
phys_retarget_fn = config.build()

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

    mesh = load_mesh(obj_name)
    pick, place = get_pickplace_timing(obj_traj, mesh)

    for step in range(pick-80+1, pick+1):
        # step = 0
    #for step in range(T):
        # print(robot_traj[0])
        phys_retarget_fn.set_init_qpos(robot_traj[step])
        phys_retarget_fn.reset()

        target_pose = load_robotwrist_pose(target_traj[step])
        
        # print(robot_traj[0])

        # target_action = phys_retarget_fn.inverse_kinematics(target_pose)
        target_action = robot_traj[step]
        #target_action = np.zeros(22)
        simulator.step(target_action, target_action, obj_traj[step])#robot_traj[step], obj_traj[step])
        # print(obj_traj[0])

    print(time.time() - start, "render seconds")
    print(T / 30, "original seconds")
    simulator.save()

