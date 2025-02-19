import pickle
import os
import numpy as np


rsc_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "rsc",
)


# File io
def load_obj_traj(demo_path):
    # Load object trajectory
    obj_traj = pickle.load(open(os.path.join(demo_path, "obj_traj.pickle"), "rb"))
    return obj_traj


def load_robot_traj(demo_path):
    # Load robot trajectory
    robot_traj = np.load(os.path.join(demo_path, "robot_qpos.npy"))
    return robot_traj


def load_robot_target_traj(demo_path):
    # Load robot trajectory
    robot_traj = np.load(os.path.join(demo_path, "target_qpos.npy"))
    return robot_traj


def load_hand_pos(demo_path):
    # Load_target trajectory
    target_traj = np.load(os.path.join(demo_path, "hand_joint.npy"))
    return target_traj


def load_mesh(obj_name):
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(
        os.path.join(rsc_path, obj_name, f"{obj_name}.obj")
    )
    return mesh
