import argparse
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import tqdm
import trimesh
from einops import rearrange, repeat
from torch import nn, optim
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from cgf.data import YCB_CLASSES, YCB_SIZE, AllegroDataset
from cgf.data import sample as sample_data
from cgf.models import SeqAllegroQpos2AllegroCVAET
from cgf.robotics import KinematicsLayer, object_contact_loss
from cgf.transformation import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_euler_angles,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    split_axis_angle,
    rotate,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

END_LINKS = ["link_3.0_tip", "link_7.0_tip", "link_11.0_tip", "link_15.0_tip"]  # end links for the allegro hand
URDF_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data/original_robot/allegro_hand_ros/allegro_hand_description/"
)


def train(data_root, target_mano_side, exp_root):
    ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    exp_dir = os.path.join(exp_root, ts)
    os.makedirs(exp_dir, exist_ok=True)
    print(f"target_mano_side: {target_mano_side}")

    dataset = AllegroDataset(
        os.path.join(data_root, "processed"),
        target_ycb_ids="train",
        target_mano_side=target_mano_side,
        last_frame_only=False,
    )
    print(f"len(dataset): {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=32, num_workers=8, shuffle=True, pin_memory=True)

    model = SeqAllegroQpos2AllegroCVAET()
    model = model.cuda()
    model = model.train()

    if target_mano_side == "left":
        urdf_path = os.path.join(URDF_DIR, "allegro_hand_description_left.urdf")
    else:
        urdf_path = os.path.join(URDF_DIR, "allegro_hand_description_right.urdf")

    kinematics_layer = KinematicsLayer(
        urdf_path, END_LINKS, global_transform=True, return_geometry=False, dtype=DTYPE, device=DEVICE
    )

    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    loss_fn = nn.MSELoss()

    max_epoch_num = 1500
    for epoch in range(max_epoch_num):
        for batch_idx, batch_data in enumerate(dataloader):
            qpos = batch_data["qpos"].cuda()  # B, N, 22
            query_t = batch_data["query_t"].cuda()  # B, 20
            query_t = repeat(query_t, "b n -> b n 1")
            obj_pts = torch.transpose(batch_data["obj_pts"], 1, 2).cuda()  # B, 3, 2000
            aug_rots = batch_data["aug_rot"].cuda()  # B, 3, 3

            batch_size = qpos.shape[0]
            qpos_rot = qpos[..., 3:6]
            qpos_rot_6d = matrix_to_rotation_6d(axis_angle_to_matrix(qpos_rot))

            rot_obj_pts = torch.bmm(aug_rots, obj_pts)

            with torch.no_grad():
                tf3ds = kinematics_layer(rearrange(qpos, "b n d -> (b n) d"))
                tf_mats = [rearrange(tf3ds[link].get_matrix(), "(b n) r c -> b n r c", b=batch_size) for link in tf3ds]
                tls = [tf_mat[..., :3, 3] for tf_mat in tf_mats]
                tls = rearrange(tls, "l b n d -> b n l d")
                rot_mats = [tf_mat[..., :3, :3] for tf_mat in tf_mats]
                rot_mats = rearrange(rot_mats, "l b n r c -> b n l r c")
                rot_mats_6d = matrix_to_rotation_6d(rot_mats)

            query_t.requires_grad_(True)
            pred, mean, log_var, z = model(rot_obj_pts, qpos, query_t)

            if not torch.isfinite(pred).all():
                print("pred contains INF or NaN!")

            gt = torch.cat([qpos[..., 0:3], qpos_rot_6d, qpos[..., 6:]], dim=-1)
            assert pred.shape == gt.shape

            theta_loss = loss_fn(pred, gt)
            xyz_loss = loss_fn(pred[..., 0:3], gt[..., 0:3]) * 10
            kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mean**2 - log_var.exp(), dim=1), dim=0) * 0.001

            pred_rot_6d = pred[..., 3 : 3 + 6]
            pred_rot = matrix_to_axis_angle(rotation_6d_to_matrix(pred_rot_6d))
            pred = torch.cat([pred[..., 0:3], pred_rot, pred[..., 3 + 6 :]], dim=-1)
            pred_tf3ds = kinematics_layer(rearrange(pred, "b n d -> (b n) d"))
            pred_tf_mats = [
                rearrange(pred_tf3ds[link].get_matrix(), "(b n) r c -> b n r c", b=batch_size) for link in pred_tf3ds
            ]
            pred_tls = [pred_tf_mat[..., :3, 3] for pred_tf_mat in pred_tf_mats]
            pred_tls = rearrange(pred_tls, "l b n d -> b n l d")
            pred_rot_mats = [pred_tf_mat[..., :3, :3] for pred_tf_mat in pred_tf_mats]
            pred_rot_mats = rearrange(pred_rot_mats, "l b n r c -> b n l r c")
            pred_rot_mats_6d = matrix_to_rotation_6d(pred_rot_mats)
            assert pred_tls.shape == tls.shape
            assert pred_rot_mats_6d.shape == rot_mats_6d.shape
            vert_loss = loss_fn(pred_tls, tls) * 10
            rot_loss = loss_fn(pred_rot_mats_6d, rot_mats_6d) * 0.5

            # only add the contact loss to the last frame
            assert torch.allclose(query_t[:, 4], torch.zeros_like(query_t[:, 4]))
            end_indices = [list(tf3ds.keys()).index(link) for link in END_LINKS]
            end_tls = tls[:, 4, end_indices]
            rot_obj_pts = rearrange(rot_obj_pts, "b n d -> b d n")
            contact_loss = object_contact_loss(end_tls, rot_obj_pts) * 50

            loss = theta_loss + kl_loss + xyz_loss + vert_loss + rot_loss + contact_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print(
                    f"{epoch:3}/{batch_idx:4}, loss: {loss.item():.4f}\n"
                    f"theta_loss: {theta_loss.item():.4f}, kl_loss: {kl_loss.item():.4f}, xyz_loss: {xyz_loss.item():.4f}, "
                    f"vert_loss: {vert_loss.item():.4f}, rot_loss: {rot_loss.item():.4f}, contact_loss: {contact_loss.item():.4f}"
                )

        scheduler.step()
        if epoch % 100 == 0:
            ckpt = {
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "schedule": scheduler.state_dict(),
                "epoch": epoch,
            }
            torch.save(ckpt, os.path.join(exp_dir, f"{epoch}.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=os.path.join(os.path.dirname(__file__), "..", "data"))
    parser.add_argument("--output_dir", type=str, default=os.path.join(os.path.dirname(__file__), "..", "output"))
    parser.add_argument("--mano_side", type=str, choices=["left", "right"], default="left")

    parser.add_argument("--mode", type=str, choices=["train", "sample"], default="train")
    parser.add_argument("--ts", type=str, default=None)
    parser.add_argument("--ckpt_step", type=int, default=1000)
    args = parser.parse_args()

    train(args.data_root, target_mano_side=args.mano_side, exp_root=args.output_dir)
