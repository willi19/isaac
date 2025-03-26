import pickle
import os
import numpy as np
import json

rsc_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "rsc",
)

home_path = os.path.expanduser("~")
shared_path = os.path.join(home_path, "shared_data")
capture_path = os.path.join(shared_path, "capture")

# # File io
# def load_obj_traj(demo_path):
#     # Load object trajectory
#     obj_traj = pickle.load(open(os.path.join(demo_path, "obj_traj.pickle"), "rb"))
#     return obj_traj


def load_robot_traj(demo_path):
    # Load robot trajectory
    arm_traj = np.load(os.path.join(demo_path, "arm", "state.npy"))
    hand_traj = np.load(os.path.join(demo_path, "hand", "state.npy"))
    robot_traj = np.concatenate([arm_traj, hand_traj], axis=-1)

    return robot_traj


def load_robot_target_traj(demo_path):
    arm_traj = np.load(os.path.join(demo_path, "arm", "action.npy"))
    hand_traj = np.load(os.path.join(demo_path, "hand", "action.npy"))
    robot_traj = np.concatenate([arm_traj, hand_traj], axis=-1)
    return robot_traj

def load_contact_value(demo_path):
    contact_value = np.load(os.path.join(demo_path, "contact", "data.npy"))
    return contact_value

def load_mesh(obj_name):
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(
        os.path.join(rsc_path, obj_name, f"{obj_name}.obj")
    )
    return mesh

def load_camparam(demo_path):
    intrinsic = json.load(open(os.path.join(demo_path, "cam_param", "intrinsics.json"), "r"))
    extrinsic = json.load(open(os.path.join(demo_path, "cam_param", "extrinsics.json"), "r"))
    return intrinsic, extrinsic

def load_c2r(demo_path):
    C2R = np.load(os.path.join(demo_path, "C2R.npy"))
    return C2R