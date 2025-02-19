import os
import numpy as np
import time
from dex_robot.simulate.simulator import simulator
from dex_robot.utils.file_io import (
    load_obj_traj,
    load_hand_pos,
)
from dex_robot.retargeting.retargeting_config import RetargetingConfig
from dex_robot.utils.data_parser import get_hand_tip_pos

# Viewer setting
obj_name = "bottle"
save_video = True
save_state = True
view_physics = True
view_replay = True
headless = False

simulator = simulator(
    obj_name,
    view_physics,
    view_replay,
    headless,
    save_video,
    save_state,
)

demo_path = f"data/human_demo/{obj_name}"
demo_path_list = os.listdir(demo_path)

# Load retargeting model
dof_names_list = simulator.get_dof_names()

config_path = "teleop/xarm6_allegro_hand_right_position.yml"
config = RetargetingConfig.load_from_file(config_path)
config.set_default_target_joint_names(dof_names_list)

vis_retarget_fn = config.build()
phys_retarget_fn = config.build()

try:
    for demo_name in demo_path_list:
        print("Demo name: ", demo_name)

        hand_pos = load_hand_pos(os.path.join(demo_path, demo_name))
        traj_dict = load_obj_traj(os.path.join(demo_path, demo_name))

        obj_traj = traj_dict[obj_name]
        wrist_traj = traj_dict["right_wrist"]

        T = hand_pos.shape[0]

        handtip_pos_wf = get_hand_tip_pos(hand_pos)

        start = time.time()
        handtip_pos_rbf_tot = (
            np.einsum("sij,snj->sni", wrist_traj[:, :3, :3], handtip_pos_wf)
            + wrist_traj[:, :3, 3][:, None, :]
        )

        video_path = f"video/{obj_name}_retarget/{demo_name}.mp4"
        save_path = f"result/{obj_name}_retarget/{demo_name}.mp4"

        simulator.set_savepath(video_path, save_path)

        for step in range(T):
            wrist_T = wrist_traj[step]
            handtip_pos_rbf = handtip_pos_rbf_tot[step]

            phys_action = phys_retarget_fn.retarget(handtip_pos_rbf[:, :]).astype(
                np.float32
            )

            vis_action = vis_retarget_fn.retarget(handtip_pos_rbf[:, :]).astype(
                np.float32
            )

            simulator.step(phys_action, vis_action, obj_traj[step])

        print(time.time() - start, "render seconds")
        print(T / 30, "original seconds")
        simulator.save()
finally:
    simulator.terminate()
