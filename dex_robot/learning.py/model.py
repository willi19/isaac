import torch
import torch.nn.functional as F
from typing import Dict, List, Union
import pytorch_kinematics as pk
import os
from torch import nn
import transforms3d as t3d
import numpy as np

class KinematicsLayer(nn.Module):
    def __init__(
        self,
        urdf_str_or_path: str,
        end_links: List[str],
        global_transform: bool = False,
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()

        if os.path.exists(urdf_str_or_path):
            with open(urdf_str_or_path) as f:
                self.chain: pk.chain.Chain = pk.build_chain_from_urdf(f.read()).to(dtype=dtype, device=device)
        else:
            self.chain: pk.chain.Chain = pk.build_chain_from_urdf(urdf_str_or_path).to(dtype=dtype, device=device)



        self.end_links = end_links
        self.global_transform = global_transform
        self.dof: int = len(self.chain.get_joint_parameter_names())

    def forward(self, qpos: torch.Tensor):
        tf3ds: Dict[str, pk.Transform3d] = {}
        device, dtype, batch_size = qpos.device, qpos.dtype, qpos.shape[0]

        identiy = torch.eye(4, device=device, dtype=dtype)
        tf3ds["base_link"] = pk.Transform3d(batch_size, matrix=identiy)

        identiy = torch.eye(4, device=device, dtype=dtype)
        identiy[:3, 3] = torch.tensor([0, 0, 0], dtype=dtype, device=device)
        tf3ds["palm"] = pk.Transform3d(batch_size, matrix=identiy)

        start = 0 if not self.global_transform else 6
        # for _, serial_chain in enumerate(self.serial_chains):
        #     # hard code for now
        #     joint_num = 4
        #     tf3ds.update(serial_chain.forward_kinematics(qpos[:, start : start + joint_num], end_only=False))
        #     start += joint_num
        tf3ds.update(self.chain.forward_kinematics(qpos[:, start:]))

        rot_mat = axis_angle_to_matrix(qpos[:, 3:6])
        glb_tf3d = pk.Transform3d(rot=rot_mat, pos=qpos[:, 0:3], device=device, dtype=dtype)
        tf3ds = {name: glb_tf3d.compose(tf3d) for name, tf3d in tf3ds.items()}
