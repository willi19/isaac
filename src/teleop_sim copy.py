from dex_robot.simulate.simulator_fixed import simulator
from dex_robot.retargeting.retargeting_config import RetargetingConfig
from dex_robot.teleop.teleoperator import TeleOperator
from dex_robot.utils.data_parser import load_robotwrist_pose
import hydra
import os
import numpy as np
from scipy.spatial.transform import Rotation

os.environ["HYDRA_FULL_ERROR"] = "1"


def homo2cart(h):
    if h.shape == (4, 4):
        t = h[:3, 3]
        R = h[:3, :3]
        q = Rotation.from_matrix(R).as_euler("XYZ")
    else:
        t = h[:3]
        q = h[3:]
        q = Rotation.from_quat(q).as_euler("XYZ")

    return np.concatenate([t, q])


@hydra.main(version_base="1.2", config_path="config/teleop", config_name="teleop")
def main(configs):
    teleop = TeleOperator(configs)
    teleop.run()

    # Viewer setting
    obj_name = "bottle"
    save_video = True
    save_state = True
    view_physics = False
    view_replay = True
    num_sphere = 3
    headless = False

    sim = simulator(
        obj_name,
        view_physics,
        view_replay,
        num_sphere,
        headless,
        save_video,
        save_state,
    )
    
    video_path = f"video/teleop_sim.mp4"
    save_path = f"result/teleop_sim.pkl"

    sim.set_savepath(video_path, save_path)

    home_wrist_pose = np.load("data/home_pose/allegro_eef_frame.npy")
    home_hand_pose = np.load("data/home_pose/allegro_hand_joint_angle.npy")
    home_pose = np.load("data/home_pose/allegro_robot_action.npy")

    target_action = np.concatenate([homo2cart(home_wrist_pose), home_hand_pose])

    init_obj_pose = np.array(
        [
            [0.53174365, 0.8465994, 0.02276282, 100.49357095],
            [-0.84589686, 0.5331071, 0.01597875, -20.19108522],
            [0.00139258, -0.0277516, 0.99961209, 20.03901474],
            [
                0.0,
                0.0,
                0.0,
                1.0,
            ],
        ]
    )
    robot_pose_homo_prev = None
    step = 0
    while True:
        robot_action = teleop.get_retargeted_action()

        if robot_action is not None:

            UNITY2ISAAC = np.array(
                [[0, 0, 1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
            )
            robot_pose_homo = robot_action["endeff_coords"]
            step += 1
            bias = Rotation.from_euler("xyz", [step, 0, 0], degrees=True).as_matrix()
            wrist_pose_homo = (
                UNITY2ISAAC @ robot_action["transformed_hand_frame"]
            )  # "endeff_coords"]
            init_pose_homo = UNITY2ISAAC @ robot_action["init_hand_frame"]
            robot_init_pose = UNITY2ISAAC @ robot_action["robot_init_H"]
            hand_pose = robot_action["desired_angles"]

            if robot_pose_homo_prev is not None:
                if (
                    np.linalg.norm(
                        robot_pose_homo[:3, :3] - robot_pose_homo_prev[:3, :3]
                    )
                    > 0.1
                ):
                    print(robot_pose_homo_prev)
                    print(robot_pose_homo)
                    print("pose changed")
            robot_pose_homo_prev = robot_pose_homo.copy()

            target_action = np.zeros(22)

            target_action[:6] = homo2cart(robot_pose_homo)
            target_action[6:] = hand_pose

            sim.step(
                target_action,
                target_action,
                init_obj_pose,
                [init_pose_homo[:3, 3], robot_init_pose[:3, 3], robot_pose_homo[:3, 3]],
            )

        else:
            sim.step(target_action, target_action, init_obj_pose)


if __name__ == "__main__":
    main()
# 1 0 0
# 0 1 0
# 0 0 1

# Z X Y

# 0 1 0
# 0 0 1
# 1 0 0

# Y Z X

# 0 0 1
# 1 0 0
# 0 1 0
