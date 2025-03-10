import numpy as np
import numpy as np
from dex_robot.utils.robot_wrapper import RobotWrapper
import os
from dex_robot.utils.file_io import rsc_path
from scipy.spatial.transform import Rotation
from dex_robot.retargeting.retargeting_config import RetargetingConfig
from dex_robot.utils.file_io import load_robot_traj, load_robot_target_traj
import transforms3d
# qpos = np.array(
#     [
#         -1.60879326e00,
#         3.84235412e-01,
#         -1.66013885e00,
#         8.71842742e-01,
#         2.15076590e00,
#         -2.86453414e00,
#         -2.78144508e-01,
#         6.62080042e-01,
#         1.92028394e-02,
#         9.74186805e-03,
#         -1.92939319e-01,
#         7.84291433e-01,
#         1.53865569e-02,
#         2.75749406e-04,
#         -8.21577607e-02,
#         4.52628330e-01,
#         1.28469169e-02,
#         6.19256319e-03,
#         8.02574948e-01,
#         5.03368393e-01,
#         1.96162272e-01,
#         -1.39480406e-02,
#     ]
# )

robot = RobotWrapper(
    os.path.join(rsc_path, "xarm6/xarm6_allegro_wrist_mounted_rotate.urdf")
)


robot_traj = load_robot_traj("data/teleoperation/bottle/1")
robot_target = load_robot_target_traj("data/teleoperation/bottle/1")

for step in range(20, 25):
    qpos = robot_traj[step]
    # print(robot_target[500][:3])
    # for s in ["XYZ","XZY","YZX","YXZ","ZXY","ZYX"]:
    #     print(Rotation.from_euler(s,robot_target[0][:3]).as_matrix().T, s)
    # print(Rotation.from_euler("ZYX",robot_target[0][:3]).as_matrix())

    # print(qpos)
    robot.compute_forward_kinematics(qpos)
    link_index = robot.get_link_index("link6")
    link_pose = robot.get_link_pose(link_index)

    palm_index = robot.get_link_index("palm_link")
    palm_pose = robot.get_link_pose(palm_index)
    print(palm_pose.T @ link_pose)
    #UNITY2ISSAC = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    # link_pose = link_pose
    # rotmat = Rotation.from_euler("XYZ",robot_target[step][:3]).as_matrix()
    angle = np.linalg.norm(robot_target[step][3:6])
    axis = robot_target[step][3:6] / angle

    rotmat = Rotation.from_rotvec(angle * axis).as_matrix()
    
    # print(cart)

    hand_pose = qpos[6:]
    # os.makedirs("data/home_pose", exist_ok=True)
    # np.save("data/home_pose/allegro_hand_joint_angle.npy", hand_pose)
    # np.save("data/home_pose/allegro_eef_frame.npy", link_pose)
    # np.save("data/home_pose/allegro_robot_action.npy", qpos)

# Link6 Position: (0.07518112, -0.4967044, 0.4066923)
# Link6 Rotation (Quaternion): (-0.6292906, -0.5308945, -0.5658871, 0.04377607)
# -0.56 -0.594 -0.572 0.083
# print(Rotation.from_quat([-0.56, -0.594, -0.572, 0.083]).as_matrix())