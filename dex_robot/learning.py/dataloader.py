import torch
from torch.utils.data import Dataset
import numpy as np
import os
from dex_robot.utils import robot_wrapper
from dex_robot.utils.file_io_prev import load_obj_traj, load_robot_traj, load_robot_target_traj, load_hand_joint_pose

class RobotHandDataset(Dataset):
    def __init__(self, object_name, frame_type, finger_joint_list): 
        """
        PyTorch Dataset for processing robot and hand poses.
        
        Args:
            object_name (str): Name of the object being manipulated
            frame_type (str): "object" or "wrist" - determines transformation frame
            finger_joint_list (list): List of joint keys to extract hand joint poses
        """
        data_root = f"data/processed/{object_name}"
        demo_list = os.listdir(data_root)

        assert frame_type in ["object", "wrist"], "frame_type should be 'object' or 'wrist'"
        self.frame_type = frame_type
        self.finger_joint_list = finger_joint_list

        self.pose_sequences = []
        self.hand_sequences = []

        for demo_ind in demo_list:
            demo_path = os.path.join(data_root, demo_ind)
            obj_traj = load_obj_traj(demo_path)[object_name]
            robot_traj = load_robot_traj(demo_path)
            hand_joint_pose = load_hand_joint_pose(demo_path)
            target_traj = load_robot_target_traj(demo_path)

            T = robot_traj.shape[0]

            wrist_pose = np.array([robot_wrapper.RobotWrapper.get_link_pose(robot_traj[i], "palm_link") for i in range(T)])
            object_pose = np.array(obj_traj)

            # 객체 기준 손목 pose 변환
            if frame_type == "object":
                obj_inv = np.linalg.inv(object_pose)  # (48, 4, 4)
                transformed_pose = np.einsum("tij,tjk->tik", obj_inv, wrist_pose)  # wrist in object frame
                transformed_target = None#
            else:
                wrist_inv = np.linalg.inv(wrist_pose)  # (48, 4, 4)
                transformed_pose = np.einsum("tij,tjk->tik", wrist_inv, object_pose)  # object in wrist frame
                transformed_target = None#

            # Hand joint pose를 finger_joint_list 순서대로 벡터로 변환
            hand_joint_vec = np.array([[hand_joint_pose[t][joint] for joint in finger_joint_list] for t in range(T)])  # (48, len(finger_joint_list))

            # 슬라이딩 윈도우 변환 (43x5)
            self.pose_sequences.extend([transformed_pose[i:i+5] for i in range(43)])  # (43, 5, 4, 4)
            self.hand_sequences.extend([hand_joint_vec[i:i+5] for i in range(43)])  # (43, 5, len(finger_joint_list))

    def __len__(self):
        return len(self.pose_sequences)  # 총 43개 시퀀스

    def __getitem__(self, idx):
        pose_seq = torch.tensor(self.pose_sequences[idx], dtype=torch.float32)  # (5, 4, 4)
        hand_seq = torch.tensor(self.hand_sequences[idx], dtype=torch.float32)  # (5, len(finger_joint_list))
        return pose_seq, hand_seq
