import os
import numpy as np
import pickle
from scipy.spatial import ConvexHull
import open3d as o3d
from dex_robot.utils.robot_wrapper import RobotWrapper
from dex_robot.utils.file_io import load_obj_traj, load_robot_traj, rsc_path

if os.path.exists("processed/convex_hull_mesh.obj"):
    mesh = o3d.io.read_triangle_mesh("processed/convex_hull_mesh.obj")

else:

    # Viewer setting
    obj_name_list = ["bottle", "book", "smallbowl1"]
    joint_cors = []
    robot = RobotWrapper(
        os.path.join(rsc_path, "xarm6/xarm6_allegro_wrist_mounted_rotate.urdf")
    )

    joint_frame_id = []
    for i, joint_name in enumerate(robot.joint_names):
        joint_frame_id.append(i)

    for obj_name in obj_name_list:
        demo_path = f"data/teleoperation/{obj_name}"
        demo_path_list = os.listdir(demo_path)

        for demo_name in demo_path_list:
            action = load_robot_traj(os.path.join(demo_path, demo_name))

            for ts in range(len(action)):
                q = action[ts]
                robot.compute_forward_kinematics(q)

                for frame_id in joint_frame_id:
                    joint_cor = robot.get_joint_pose(frame_id).copy()[:3, 3]
                    joint_cors.append(joint_cor)

    joint_cors = np.array(joint_cors)
    hull = ConvexHull(joint_cors)

    vertices = hull.points[hull.vertices]
    inv_index = {v: i for i, v in enumerate(hull.vertices)}
    faces = np.array(
        [[inv_index[index] for index in simplex] for simplex in hull.simplices]
    )

    reversed_faces = faces[:, ::-1]
    all_faces = np.vstack((faces, reversed_faces))

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)

    mesh.triangles = o3d.utility.Vector3iVector(all_faces)
    mesh.compute_vertex_normals()

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(joint_cors)

# 시각화
mesh.paint_uniform_color([0.1, 0.1, 0.7])
pcd.paint_uniform_color([1, 0, 0])
o3d.visualization.draw_geometries([pcd, mesh])

# .obj 파일로 저장
o3d.io.write_triangle_mesh("processed/convex_hull_mesh.obj", mesh)
