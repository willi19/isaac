from dex_robot.utils.metric import get_grasp_timing
from dex_robot.utils.file_io import load_obj_traj, load_robot_traj, load_robot_target_traj, load_mesh, rsc_path
import os
import open3d as o3d
from dex_robot.utils import robot_wrapper
import numpy as np
import pickle

obj_name_list = ["bottle"]
urdf_path = os.path.join(rsc_path, "xarm6", "xarm6_allegro_wrist_mounted_rotate.urdf")
robot = robot_wrapper.RobotWrapper(urdf_path)
# palm_id = robot.get_link_index("palm_link")

link_list = ["palm_link", "thumb_base", "thumb_proximal", "thumb_medial", "thumb_distal", "thumb_tip", 
             "index_base", "index_proximal", "index_medial", "index_distal", "index_tip", 
             "middle_base", "middle_proximal", "middle_medial", "middle_distal", "middle_tip", 
             "ring_base", "ring_proximal", "ring_medial", "ring_distal", "ring_tip"]

link_ids = {link_name: robot.get_link_index(link_name) for link_name in link_list}

def downsample_indices(n, target_length):
    # 0~n
    r = n // target_length
    ret = [0]
    for i in range(target_length):
        if i < n % target_length:
            ret.append(ret[-1] + r + 1)
        else:
            ret.append(ret[-1] + r)
    return ret

for obj_name in obj_name_list:
    teleop_root_path = f"data/teleoperation/{obj_name}"  # Replace with your actual path
    teleop_demo_path_list = os.listdir(teleop_root_path)

    demo_list = os.listdir(teleop_root_path)
    grasp_traj_length_list = []
    grasp_range = {}

    for demo_ind in demo_list:
        demo_path = os.path.join(teleop_root_path, demo_ind)
        # Load trajectories
        obj_traj = load_obj_traj(demo_path)[obj_name]
        robot_traj = load_robot_traj(demo_path)
        T = robot_traj.shape[0]
        robot_target_traj = load_robot_target_traj(demo_path)

        hand_joint_pose = {joint_name: [] for joint_name in link_list}
        wrist_pose = []

        for i in range(T):
            robot.compute_forward_kinematics(robot_traj[i])
            for joint_name in link_list:
                hand_joint_pose[joint_name].append(robot.get_link_pose(link_ids[joint_name]))

            wrist_pose.append(robot.get_link_pose(link_ids["palm_link"]))

        wrist_pose = np.array(wrist_pose)

        grasp_start, grasp_end = get_grasp_timing(obj_traj, obj_name, wrist_pose, robot_traj[:,6:])
        grasp_traj_length_list.append(grasp_end - grasp_start+1)
        grasp_range[demo_ind] = {"start": grasp_start, "end": grasp_end, "wrist_pose":wrist_pose, "robot_traj":robot_traj, "robot_target_traj":robot_target_traj, "obj_traj":obj_traj, "hand_joint_pose":hand_joint_pose}

    target_length = min(grasp_traj_length_list)

    os.makedirs(f"data/processed/{obj_name}", exist_ok=True)
    for demo_ind in demo_list:
        os.makedirs(f"data/processed/{obj_name}/{demo_ind}", exist_ok=True)
        save_path = f"data/processed/{obj_name}/{demo_ind}"

        grasp_start = grasp_range[demo_ind]["start"]
        grasp_end = grasp_range[demo_ind]["end"]
        indices = downsample_indices(grasp_end - grasp_start, target_length)
        
        obj_traj = {obj_name: grasp_range[demo_ind]["obj_traj"][grasp_start:grasp_end+1][indices]}
        robot_traj = grasp_range[demo_ind]["robot_traj"][grasp_start:grasp_end+1][indices]
        robot_target_traj = grasp_range[demo_ind]["robot_target_traj"][grasp_start:grasp_end+1][indices]
        wrist_pose = grasp_range[demo_ind]["wrist_pose"][grasp_start:grasp_end+1][indices]
        hand_joint_pose = {joint_name : np.array(pose)[grasp_start:grasp_end+1][indices] for joint_name, pose in grasp_range[demo_ind]["hand_joint_pose"].items()}

        # np.save(os.path.join(save_path, "obj_traj.npy"), obj_traj[grasp_start:grasp_end+1][indices])
        pickle.dump(obj_traj, open(os.path.join(save_path, "obj_traj.pickle"), "wb"))
        pickle.dump(hand_joint_pose, open(os.path.join(save_path, "hand_joint_pose.pickle"), "wb"))
        np.save(os.path.join(save_path, "robot_qpos.npy"), robot_traj)
        np.save(os.path.join(save_path, "target_qpos.npy"), robot_target_traj)
        np.save(os.path.join(save_path, "wrist_pose.npy"), wrist_pose)
