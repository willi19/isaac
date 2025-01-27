import numpy as np
import torch
import nlopt
from dex_robot.retargeting.robot_wrapper import RobotWrapper
from typing import List

# Refer to
# https://github.com/dexsuite/dex-retargeting/blob/main/dex_retargeting/optimizer.py

# Remove fixed joint value during optimization
# Remove mimic joint(used for gripper)
# Optimizing every joint. Excluding joints do almost the same as using fixed joint

# Change the optimzier to instantiate its own robot wrapper
# Cause when retargeting concurrently, the robot wrapper should not be shared


class Optimizer:
    retargeting_modes = ["position"]

    def __init__(
        self,
        urdf_path: str,
        joint_order: List[str],
        retargeting_mode: str,
        retarget_fn_args: dict,
    ):
        self.robot = RobotWrapper(urdf_path)
        self.num_joints = self.robot.dof

        self.opt = nlopt.opt(nlopt.LD_SLSQP, self.num_joints)
        self.opt_dof = self.num_joints

        idx_pin2target = [self.robot.get_joint_index(name) for name in joint_order]
        self.idx_pin2target = np.array(idx_pin2target)
        if retargeting_mode not in self.retargeting_modes:
            raise ValueError(
                f"Unsupported retargeting mode: {retargeting_mode}, available modes: {self.retargeting_modes}"
            )
        self.get_objective_function_wrapper(retargeting_mode, retarget_fn_args)

    def get_objective_function_wrapper(self, retargeting_mode, retarget_fn_args):
        if retargeting_mode == "position":
            self.body_names = retarget_fn_args["target_link_names"]
            huber_delta = retarget_fn_args.get("huber_delta", 0.02)
            norm_delta = retarget_fn_args.get("norm_delta", 0)

            self.huber_loss = torch.nn.L1Loss()  # (beta=huber_delta)
            self.norm_delta = norm_delta
            # Sanity check and cache link indices
            self.target_link_indices = self.robot.get_link_indices(self.body_names)

            self.opt.set_ftol_abs(1e-7)
            self.objective_function_wrapper = self.get_position_objective_function

    def set_joint_limit(self, joint_limits: np.ndarray, epsilon=1e-3):
        if joint_limits.shape != (self.opt_dof, 2):
            raise ValueError(
                f"Expect joint limits have shape: {(self.opt_dof, 2)}, but get {joint_limits.shape}"
            )
        self.opt.set_lower_bounds((joint_limits[:, 0] - epsilon).tolist())
        self.opt.set_upper_bounds((joint_limits[:, 1] + epsilon).tolist())

    def retarget(self, ref_value, init_qpos=None):
        objective_fn = self.get_objective_function(
            ref_value, np.array(init_qpos).astype(np.float32)
        )

        self.opt.set_min_objective(objective_fn)
        try:
            qpos = self.opt.optimize(init_qpos)
            return np.array(qpos, dtype=np.float32)
        except RuntimeError as e:
            print(e)
            return np.array(init_qpos, dtype=np.float32)

    def get_objective_function(self, target_pos, init_qpos):
        return self.objective_function_wrapper(target_pos, init_qpos)

    def get_position_objective_function(self, target_pos, init_qpos):
        torch_target_pos = torch.as_tensor(target_pos)
        torch_target_pos.requires_grad_(False)

        qpos = np.zeros(self.num_joints)

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos[self.idx_pin2target] = (
                x.copy()
            )  # re-order the indexs issac -> pinocchio

            self.robot.compute_forward_kinematics(qpos)
            target_link_poses = [
                self.robot.get_link_pose(index) for index in self.target_link_indices
            ]
            body_pos = np.stack([pose[:3, 3] for pose in target_link_poses], axis=0)

            # Torch computation for accurate loss and grad
            torch_body_pos = torch.as_tensor(body_pos)
            torch_body_pos.requires_grad_()

            # Loss term for kinematics retargeting based on 3D position error
            huber_distance = self.huber_loss(torch_body_pos, torch_target_pos)
            result = huber_distance.cpu().detach().item()

            if grad.size > 0:
                jacobians = []
                for i, index in enumerate(self.target_link_indices):
                    link_body_jacobian = self.robot.compute_single_link_local_jacobian(
                        qpos, index
                    )[:3, ...]
                    link_pose = target_link_poses[i]
                    link_rot = link_pose[:3, :3]
                    link_kinematics_jacobian = link_rot @ link_body_jacobian
                    jacobians.append(link_kinematics_jacobian)

                # Note: the joint order in this jacobian is consistent pinocchio
                jacobians = np.stack(jacobians, axis=0)
                huber_distance.backward()
                grad_pos = torch_body_pos.grad.cpu().numpy()[:, None, :]

                # Convert the jacobian from pinocchio order to target order
                jacobians = jacobians[..., self.idx_pin2target]

                # Compute the gradient to the qpos
                grad_qpos = np.matmul(grad_pos, jacobians)
                grad_qpos = grad_qpos.mean(1).sum(0)
                grad_qpos += 2 * self.norm_delta * (x - init_qpos)

                grad[:] = grad_qpos[:]

            return result

        return objective
