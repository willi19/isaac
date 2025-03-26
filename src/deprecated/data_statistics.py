import os
import pickle
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


# File io
def load_obj_traj(demo_path):
    # Load object trajectory
    obj_traj = pickle.load(open(os.path.join(demo_path, "obj_traj.pickle"), "rb"))
    return obj_traj


def load_robot_traj(demo_path):
    # Load target trajectory
    robot_traj = np.load(os.path.join(demo_path, "robot_qpos.npy"))
    return robot_traj


def compute_mesh_to_ground_distance(obj_pose, mesh):
    """
    Compute the minimum distance between the mesh's lowest point and the ground (z=0).

    Parameters:
        obj_traj (list or np.ndarray): Object trajectory where each element contains
                                       the position (x, y, z) of the object.
        mesh (open3d.geometry.TriangleMesh): Mesh representing the object.

    Returns:
        list: A list of distances (minimum z-values relative to the ground).
    """

    # Translate the mesh based on the object's pose
    # mesh_copy = mesh.clone()
    # mesh_copy = mesh_copy.transform(obj_pose)

    # Get the vertices of the mesh
    vertices = np.asarray(mesh.vertices).copy()

    vertices = obj_pose[:3, :3] @ vertices.T + obj_pose[:3, 3][:, None]

    # Find the minimum z-coordinate (closest point to the ground)
    min_z = np.min(vertices[2, :]) + 0.0585
    distance_to_ground = min_z  # Ensure no negative distances

    distances = distance_to_ground

    return distances


def compute_distance(traj1, traj2):
    """
    Compute the average Euclidean distance between two trajectories.

    Parameters:
        traj1 (np.ndarray): First trajectory of shape (T, 4, 4).
        traj2 (np.ndarray): Second trajectory of shape (T, 4, 4).

    Returns:
        float: Average Euclidean distance between the two trajectories.
    """
    distances = []
    for pose1, pose2 in zip(traj1, traj2):
        if pose1.shape == (4, 4):
            pose1 = pose1[:3, 3]
            pose2 = pose2[:3, 3]
        dist = np.linalg.norm(pose1 - pose2)
        distances.append(dist)
    return np.max(distances)


def main():
    obj_name = "bottle"
    teleop_root_path = f"data/teleoperation/{obj_name}"  # Replace with your actual path
    teleop_demo_path_list = os.listdir(teleop_root_path)

    mesh = o3d.io.read_triangle_mesh(f"rsc/{obj_name}/{obj_name}.obj")
    pickplace_timing = np.zeros((len(teleop_demo_path_list) + 100, 2))
    pickplace_std = []
    pickpose = []
    tot_pickpose = []
    tot_pose = []
    for demo_name in teleop_demo_path_list:
        demo_path = os.path.join(teleop_root_path, demo_name)

        # Load trajectories
        obj_traj = load_obj_traj(demo_path)[obj_name]
        robot_traj = load_robot_traj(demo_path)

        T = obj_traj.shape[0]
        heights = []

        for step in range(T):
            # Compute object heights
            h = compute_mesh_to_ground_distance(obj_traj[step], mesh)
            heights.append(h)

        pick, place = get_pickplace_timing(heights)

        # standard deviation during pick and place

        hand_qpos = robot_traj[pick:place, 6:]
        if pick < place - 1:
            pickplace_std.append(
                np.std(hand_qpos, axis=0) / np.std(robot_traj[:, 6:], axis=0)
            )
            # if (
            #     np.max(np.std(hand_qpos, axis=0) / np.std(robot_traj[:, 6:], axis=0))
            #     > 0.5
            # ):
            #     print(demo_name)

        # standard deviation of picking pose
        pickpose.append(hand_qpos[0])
        tot_pickpose.append(hand_qpos)
        tot_pose.append(robot_traj[:, 6:])
        # pickplace_timing[int(demo_name)][0] = pick
        # pickplace_timing[int(demo_name)][1] = place
    # np.save("pickplace_timing.npy", pickplace_timing)
    pickplace_std = np.array(pickplace_std)
    pickplace_std = np.mean(pickplace_std, axis=0)
    print(
        pickplace_std, "percentage of std during whole trajectory and during pickplace"
    )

    pickpose = np.array(pickpose)
    tot_pickpose = np.concatenate(tot_pickpose, axis=0)
    tot_pose = np.concatenate(tot_pose, axis=0)

    pickpose_std = np.std(pickpose, axis=0)
    print(pickpose_std, "std of picking pose")
    tot_pickpose_std = np.std(tot_pickpose, axis=0)
    print(tot_pickpose_std, "std of picking pose during pickplace")
    tot_pose_std = np.std(tot_pose, axis=0)
    print(tot_pose_std, "std of whole trajectory")


if __name__ == "__main__":
    main()
