import os
import numpy as np
from isaacgym import gymapi, gymtorch
import pickle
from scipy.spatial.transform import Rotation as R
import cv2
import time

# Viewer setting
obj_name = "smallbowl1"
save_video = True
view_physics = True
view_replay = True
headless = False

os.makedirs(f"video/{obj_name}", exist_ok=True)


#  utils
def sim_to_real_allegro(joint_pos: np.array):
    # joint order of Allegro in sim and real are different
    allegro_angles_tmp = joint_pos.copy()
    allegro_angles = np.zeros_like(allegro_angles_tmp)
    allegro_angles[0:4] = allegro_angles_tmp[0:4]
    allegro_angles[4:12] = allegro_angles_tmp[8:16]
    allegro_angles[12:16] = allegro_angles_tmp[4:8]
    return allegro_angles


def real_to_sim_allegro(joint_pos: np.array):
    # joint order of Allegro in sim and real are different
    allegro_angles_tmp = joint_pos.copy()
    allegro_angles = np.zeros_like(allegro_angles_tmp)
    allegro_angles[0:4] = allegro_angles_tmp[0:4]
    allegro_angles[8:16] = allegro_angles_tmp[4:12]
    allegro_angles[4:8] = allegro_angles_tmp[12:16]
    return allegro_angles


def adjust_finger_index(joint_pos: np.array):
    # joint order of Allegro urdf is index middle ring thumb
    # joint order in real is index middle ring thumb

    allegro_angles_tmp = joint_pos.copy()
    allegro_angles = np.zeros_like(allegro_angles_tmp)
    allegro_angles[4:8] = allegro_angles_tmp[8:12]  # ring
    allegro_angles[8:12] = allegro_angles_tmp[12:16]  # middle
    allegro_angles[12:16] = allegro_angles_tmp[4:8]  # thumb
    return allegro_angles


# File io
def load_obj_traj(demo_path):
    # Load object trajectory
    obj_traj = pickle.load(open(os.path.join(demo_path, "obj_traj.pickle"), "rb"))
    return obj_traj


def load_target_traj(demo_path):
    # Load target trajectory
    target_traj = np.load(os.path.join(demo_path, "robot_qpos.npy"))
    return target_traj


# initialize environment
def load_env(gym, sim, assets):
    # Load environment
    actor_handle = {}
    env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 1)

    robot_pose = gymapi.Transform()
    robot_pose.p = gymapi.Vec3(0, 0, 0)

    # 객체 추가
    object_pose = gymapi.Transform()
    object_pose.p = gymapi.Vec3(0.5, 0.0, 0.0)

    if view_physics:
        actor_handle["robot"] = gym.create_actor(
            env, assets["robot"], robot_pose, "robot", 1, 0
        )
        actor_handle["object"] = gym.create_actor(
            env, assets["object"], object_pose, "object", 1, 0
        )

        obj_props = gym.get_actor_rigid_shape_properties(env, actor_handle["object"])

        obj_props[0].restitution = 0.1
        obj_props[0].friction = 0.0
        gym.set_actor_rigid_shape_properties(env, actor_handle["object"], obj_props)

        props = gym.get_actor_dof_properties(env, actor_handle["robot"])
        props["driveMode"].fill(gymapi.DOF_MODE_POS)

        props["stiffness"][:6] = 1000.0  # pgain for arm
        props["damping"][:6] = 10.0  # dgain for arm

        props["stiffness"][6:] = 500.0  # pgain for hand
        props["damping"][6:] = 10.0  # dgain for hand

        gym.set_actor_dof_properties(env, actor_handle["robot"], props)

    if view_replay:

        actor_handle["robot_replay"] = gym.create_actor(
            env, assets["vis_robot"], robot_pose, "robot_replay", 2, 1
        )
        actor_handle["object_replay"] = gym.create_actor(
            env, assets["vis_object"], object_pose, "object_replay", 3, 0
        )

        for obj_name in ["robot_replay", "object_replay"]:
            rigid_body_props = gym.get_actor_rigid_body_properties(
                env, actor_handle[obj_name]
            )

            for prop in rigid_body_props:
                prop.flags = gymapi.RIGID_BODY_DISABLE_GRAVITY  # Disable gravity flag

            gym.set_actor_rigid_body_properties(
                env,
                actor_handle[obj_name],
                rigid_body_props,
                recomputeInertia=False,
            )

            props = gym.get_actor_dof_properties(env, actor_handle[obj_name])
            props["driveMode"].fill(gymapi.DOF_MODE_NONE)
            gym.set_actor_dof_properties(env, actor_handle[obj_name], props)
        for i in range(29):
            gym.set_rigid_body_color(
                env,
                actor_handle["robot_replay"],
                i,
                gymapi.MESH_VISUAL_AND_COLLISION,
                gymapi.Vec3(0.4, 0.4, 0.6),
            )
        # for finger_name in ["thumb", "index", "middle", "ring"]:
        #     for joint_name in ["base", "proximal", "medial", "distal", "tip"]:
        #         ind = gym.find_actor_rigid_body_index(
        #             env,
        #             actor_handle["robot_replay"],
        #             f"{finger_name}_{joint_name}",
        #             gymapi.DOMAIN_ACTOR,
        #         )
        #         gym.set_rigid_body_color(
        #             env,
        #             actor_handle["robot_replay"],
        #             ind,
        #             gymapi.MESH_VISUAL_AND_COLLISION,
        #             gymapi.Vec3(0.4, 0.4, 0.6),
        #         )
        # for link_name in [
        #     "link_base",
        #     "link1",
        #     "link2",
        #     "link3",
        #     "link4",
        #     "link5",
        #     "link6",
        # ]:
        #     ind = gym.find_actor_rigid_body_index(
        #         env,
        #         actor_handle["robot_replay"],
        #         link_name,
        #         gymapi.DOMAIN_ACTOR,
        #     )
        #     gym.set_rigid_body_color(
        #         env,
        #         actor_handle["robot_replay"],
        #         ind,
        #         gymapi.MESH_VISUAL_AND_COLLISION,
        #         gymapi.Vec3(0.4, 0.4, 0.6),
        #     )
        gym.set_rigid_body_color(
            env,
            actor_handle["object_replay"],
            0,
            gymapi.MESH_VISUAL_AND_COLLISION,
            gymapi.Vec3(0.4, 0.4, 0.6),
        )
    return env, actor_handle


def add_plane(gym, sim):
    # configure the ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)  # z-up!
    plane_params.distance = 0.0525
    plane_params.static_friction = 1
    plane_params.dynamic_friction = 0.8
    plane_params.restitution = 0

    # create the ground plane
    gym.add_ground(sim, plane_params)


def set_viewer(sim):
    cam_props = gymapi.CameraProperties()

    viewer = gym.create_viewer(sim, cam_props)
    # Define camera position (e.g., behind and slightly above the actor)
    camera_position = np.array([2, 0, 1])

    # Calculate direction vector (camera looks at the actor)
    camera_direction = -camera_position
    camera_direction = camera_direction / np.linalg.norm(camera_direction)  # Normalize

    # Set viewer camera transform
    gym.viewer_camera_look_at(
        viewer, None, gymapi.Vec3(*camera_position), gymapi.Vec3(0, 0, 0)
    )
    return viewer


def generate_sim(gym):
    # 시뮬레이션 설정
    sim_params = gymapi.SimParams()
    # set common parameters
    sim_params.dt = 1 / 60
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

    # set PhysX-specific parameters
    sim_params.physx.use_gpu = True
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.contact_offset = 0.01
    sim_params.physx.rest_offset = 0.0

    sim_params.flex.solver_type = 5

    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    return sim


def add_assets(gym, sim):
    asset_root = os.path.join(os.getcwd(), "rsc")
    robot_asset_file = "xarm6/xarm6_allegro_wrist_mounted_rotate.urdf"
    object_asset_file = f"{obj_name}/{obj_name}.urdf"
    robot_asset_options = gymapi.AssetOptions()
    robot_asset_options.fix_base_link = True
    robot_asset_options.armature = 0.001
    robot_asset_options.thickness = 0.002

    object_asset_options = gymapi.AssetOptions()
    object_asset_options.override_inertia = True
    object_asset_options.mesh_normal_mode = (
        gymapi.COMPUTE_PER_VERTEX
    )  # Use per-vertex normals
    object_asset_options.vhacd_enabled = True
    object_asset_options.vhacd_params = gymapi.VhacdParams()
    object_asset_options.vhacd_params.resolution = 300000

    robot_asset = gym.load_asset(sim, asset_root, robot_asset_file, robot_asset_options)
    object_asset = gym.load_asset(
        sim, asset_root, object_asset_file, object_asset_options
    )

    vis_object_asset_options = gymapi.AssetOptions()
    vis_object_asset_options.disable_gravity = True
    vis_object_asset = gym.load_asset(
        sim, asset_root, object_asset_file, vis_object_asset_options
    )

    vis_robot_asset_options = gymapi.AssetOptions()
    vis_robot_asset_options.disable_gravity = True
    vis_robot_asset_options.fix_base_link = True
    vis_robot_asset = gym.load_asset(
        sim, asset_root, robot_asset_file, vis_robot_asset_options
    )

    assets = {
        "robot": robot_asset,
        "object": object_asset,
        "vis_object": vis_object_asset,
        "vis_robot": vis_robot_asset,
    }
    return assets


def load_camera(gym, env, width, height):
    # Add a camera to capture frames
    camera_props = gymapi.CameraProperties()
    camera_props.horizontal_fov = 75.0
    camera_props.width = width
    camera_props.height = height
    camera_handle = gym.create_camera_sensor(env, camera_props)

    gym.set_camera_location(
        camera_handle,
        env,
        gymapi.Vec3(2, 0, 1),
        gymapi.Vec3(0, 0, 0),
    )
    return camera_handle


# =================================================================================================
gym = gymapi.acquire_gym()
sim = generate_sim(gym)

add_plane(gym, sim)
assets = add_assets(gym, sim)

(env, actor_handle) = load_env(gym, sim, assets)
if not headless:
    viewer = set_viewer(sim)

demo_path = f"data/teleoperation/{obj_name}"
demo_path_list = os.listdir(demo_path)

if save_video:
    frame_width = 1920
    frame_height = 1080
    fps = 30
    camera_handle = load_camera(gym, env, frame_width, frame_height)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Codec

print()
for demo_name in demo_path_list[:3]:
    robot_target = load_target_traj(os.path.join(demo_path, demo_name))
    obj_traj = load_obj_traj(os.path.join(demo_path, demo_name))[obj_name]

    T = robot_target.shape[0]
    print("Demo name: ", demo_name)

    if save_video:
        output_filename = f"{demo_name}.mp4"
        out = cv2.VideoWriter(
            f"video/{obj_name}/{output_filename}",
            fourcc,
            fps,
            (frame_width, frame_height),
        )

    start = time.time()
    for step in range(T):
        # Convert rotation matrix to quaternion
        rotmat = obj_traj[step][:3, :3]
        r = R.from_matrix(rotmat)
        quat = r.as_quat()  # [qx, qy, qz, qw]

        pos = obj_traj[step][:3, 3]

        action = robot_target[step].astype(np.float32)
        # action[6:] = sim_to_real_allegro(action[6:])
        # action[6:] = adjust_finger_index(action[6:])

        if view_physics:
            if step == 0:
                object_rb_state = gym.get_actor_rigid_body_states(
                    env, actor_handle["object"], gymapi.STATE_POS
                )
                object_rb_state["pose"]["r"].fill((quat[0], quat[1], quat[2], quat[3]))
                object_rb_state["pose"]["p"].fill((pos[0], pos[1], pos[2]))

                gym.set_actor_rigid_body_states(
                    env, actor_handle["object"], object_rb_state, gymapi.STATE_POS
                )

                robot_dof_state = gym.get_actor_dof_states(
                    env, actor_handle["robot"], gymapi.STATE_POS
                )
                robot_dof_state["pos"] = action

                gym.set_actor_dof_states(
                    env, actor_handle["robot"], robot_dof_state, gymapi.STATE_POS
                )
            if step != T - 1:
                next_action = robot_target[step + 1].astype(np.float32)
                gym.set_actor_dof_position_targets(
                    env, actor_handle["robot"], next_action
                )

        if view_replay:
            robot_dof_state = gym.get_actor_dof_states(
                env, actor_handle["robot_replay"], gymapi.STATE_POS
            )
            robot_dof_state["pos"] = action

            gym.set_actor_dof_states(
                env, actor_handle["robot_replay"], robot_dof_state, gymapi.STATE_POS
            )

            object_rb_state = gym.get_actor_rigid_body_states(
                env, actor_handle["object_replay"], gymapi.STATE_POS
            )
            object_rb_state["pose"]["r"].fill((quat[0], quat[1], quat[2], quat[3]))
            object_rb_state["pose"]["p"].fill((pos[0], pos[1], pos[2]))

            gym.set_actor_rigid_body_states(
                env, actor_handle["object_replay"], object_rb_state, gymapi.STATE_POS
            )

        gym.simulate(sim)
        gym.fetch_results(sim, True)

        if not headless:
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)

        if save_video:
            gym.render_all_camera_sensors(sim)

            image = gym.get_camera_image(
                sim, env, camera_handle, gymapi.IMAGE_COLOR
            ).astype(np.uint8)
            # Convert to OpenCV format (BGRA to BGR)
            frame = image.reshape(frame_height, frame_width, 4)[:, :, :3]

            # Write frame to video
            out.write(frame)

    print(time.time() - start, "render seconds")
    print(T / 30, "original seconds")
    #  gym.destroy_sim(sim)
