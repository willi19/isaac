import os
import numpy as np
from isaacgym import gymapi, gymtorch
import pickle
from scipy.spatial.transform import Rotation as R
import cv2
import time
from dex_robot.retargeting.retargeting_config import RetargetingConfig
import transforms3d as t3d
from ..utils.file_io import rsc_path


class simulator:
    def __init__(
        self,
        obj_name,
        view_physics=True,
        view_replay=True,
        num_sphere=0,
        headless=False,
        save_video=True,
        save_state=False,
    ):
        self.obj_name = obj_name

        self.view_physics = view_physics
        self.view_replay = view_replay
        self.num_sphere = num_sphere

        self.gym = gymapi.acquire_gym()
        self.sim = self.generate_sim()

        self.add_plane()
        self.add_assets(obj_name)

        (self.env, self.actor_handle) = self.load_env()

        self.headless = headless
        self.save_video = save_video
        self.save_state = save_state

        if not headless:
            self.set_viewer()
        self.step_idx = 0

    def set_savepath(self, video_path, state_path):
        if self.save_video:
            self.output_filename = video_path
            os.makedirs(os.path.dirname(self.output_filename), exist_ok=True)
            self.frame_width = 1920
            self.frame_height = 1080
            self.fps = 30
            self.camera_handle = self.load_camera()
            self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.out = cv2.VideoWriter(
                self.output_filename,
                self.fourcc,
                self.fps,
                (self.frame_width, self.frame_height),
            )

        if self.save_state:
            self.state_path = state_path
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)

            self.history = {"robot": [], "object": []}

    def save_stateinfo(self):

        robot_dof_state = self.gym.get_actor_dof_states(
            self.env, self.actor_handle["robot"], gymapi.STATE_POS
        )
        self.history["robot"].append(robot_dof_state["pos"])

        object_rb_state = self.gym.get_actor_rigid_body_states(
            self.env, self.actor_handle["object"], gymapi.STATE_POS
        )

        obj_quat = np.array(
            [
                object_rb_state["pose"]["r"]["x"][0],
                object_rb_state["pose"]["r"]["y"][0],
                object_rb_state["pose"]["r"]["z"][0],
                object_rb_state["pose"]["r"]["w"][0],
            ]
        )
        obj_rotmat = R.from_quat(obj_quat).as_matrix()

        obj_pos = np.array(
            [
                object_rb_state["pose"]["p"]["x"][0],
                object_rb_state["pose"]["p"]["y"][0],
                object_rb_state["pose"]["p"]["z"][0],
            ]
        )

        obj_T = np.eye(4)
        obj_T[:3, :3] = obj_rotmat
        obj_T[:3, 3] = obj_pos

        self.history["object"].append(obj_T)

    def step(self, action, viz_action, obj_pose, sphere_pos=None):
        assert obj_pose.shape == (4, 4)

        obj_quat = R.from_matrix(obj_pose[:3, :3]).as_quat()
        obj_pos = obj_pose[:3, 3]

        action = action.astype(np.float32)
        viz_action = viz_action.astype(np.float32)

        if self.view_physics:
            if self.step_idx == 0:
                robot_dof_state = self.gym.get_actor_dof_states(
                    self.env, self.actor_handle["robot"], gymapi.STATE_POS
                )

                robot_dof_state["pos"] = action

                self.gym.set_actor_dof_states(
                    self.env,
                    self.actor_handle["robot"],
                    robot_dof_state,
                    gymapi.STATE_POS,
                )

                object_rb_state = self.gym.get_actor_rigid_body_states(
                    self.env, self.actor_handle["object"], gymapi.STATE_POS
                )
                object_rb_state["pose"]["r"].fill(
                    (obj_quat[0], obj_quat[1], obj_quat[2], obj_quat[3])
                )
                object_rb_state["pose"]["p"].fill((obj_pos[0], obj_pos[1], obj_pos[2]))

                self.gym.set_actor_rigid_body_states(
                    self.env,
                    self.actor_handle["object"],
                    object_rb_state,
                    gymapi.STATE_POS,
                )

            self.save_stateinfo()
            self.gym.set_actor_dof_position_targets(
                self.env, self.actor_handle["robot"], action
            )

        if self.view_replay:
            robot_dof_state = self.gym.get_actor_dof_states(
                self.env, self.actor_handle["robot_replay"], gymapi.STATE_POS
            )
            robot_dof_state["pos"] = viz_action

            self.gym.set_actor_dof_states(
                self.env,
                self.actor_handle["robot_replay"],
                robot_dof_state,
                gymapi.STATE_POS,
            )

            object_rb_state = self.gym.get_actor_rigid_body_states(
                self.env, self.actor_handle["object_replay"], gymapi.STATE_POS
            )
            object_rb_state["pose"]["r"].fill(
                (obj_quat[0], obj_quat[1], obj_quat[2], obj_quat[3])
            )
            object_rb_state["pose"]["p"].fill((obj_pos[0], obj_pos[1], obj_pos[2]))

            self.gym.set_actor_rigid_body_states(
                self.env,
                self.actor_handle["object_replay"],
                object_rb_state,
                gymapi.STATE_POS,
            )
            robot_rb_state = self.gym.get_actor_rigid_body_states(
                self.env, self.actor_handle["robot_replay"], gymapi.STATE_POS
            )

            # 2. 링크 이름과 인덱스 매핑
            rigid_body_names = self.gym.get_actor_rigid_body_names(
                self.env, self.actor_handle["robot_replay"]
            )

            # 3. `link6`의 인덱스 찾기
            link6_index = rigid_body_names.index("link6")

            # 4. `link6`의 pose 가져오기
            link6_pose = robot_rb_state["pose"][
                link6_index
            ]  # Pose 정보 (position & rotation)

            # 5. 위치(Position)와 회전(Quaternion) 분리
            # link6_position = link6_pose["p"]  # (x, y, z)
            # link6_rotation = link6_pose["r"]  # (qx, qy, qz, qw)

            # 6. 출력 확인
            # print(f"Link6 Position: {link6_position}")
            # print(f"Link6 Rotation (Quaternion): {link6_rotation}")

        if self.num_sphere > 0 and sphere_pos is not None:
            for i in range(self.num_sphere):
                sp = sphere_pos[i]
                object_rb_state = self.gym.get_actor_rigid_body_states(
                    self.env, self.actor_handle[f"sphere_{i}"], gymapi.STATE_POS
                )

                object_rb_state["pose"]["p"].fill((sp[0], sp[1], sp[2]))

                self.gym.set_actor_rigid_body_states(
                    self.env,
                    self.actor_handle[f"sphere_{i}"],
                    object_rb_state,
                    gymapi.STATE_POS,
                )

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        if not self.headless:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)

        if self.save_video:
            self.gym.render_all_camera_sensors(self.sim)

            frame = self.gym.get_camera_image(
                self.sim, self.env, self.camera_handle, gymapi.IMAGE_COLOR
            ).astype(np.uint8)

            frame = frame.reshape((self.frame_height, self.frame_width, 4))[:, :, :3]
            frame = frame[:, :, ::-1]
            self.out.write(frame)
        self.step_idx += 1

    def save(self):
        self.step_idx = 0
        if self.save_video:
            self.out.release()

        if self.save_state:
            pickle.dump(self.history, open(self.state_path, "wb"))
            self.history = {}

    def load_env(self):
        actor_handle = {}
        env = self.gym.create_env(
            self.sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 1
        )

        robot_pose = gymapi.Transform()
        robot_pose.p = gymapi.Vec3(0, 0, 0)

        # 객체 추가
        object_pose = gymapi.Transform()
        object_pose.p = gymapi.Vec3(0.5, 0.0, 0.0)

        if self.view_physics:
            actor_handle["robot"] = self.gym.create_actor(
                env, self.assets["robot"], robot_pose, "robot", 1, 0
            )
            actor_handle["object"] = self.gym.create_actor(
                env, self.assets["object"], object_pose, "object", 1, 0
            )

            obj_props = self.gym.get_actor_rigid_shape_properties(
                env, actor_handle["object"]
            )

            obj_props[0].restitution = 0.1
            obj_props[0].friction = 0.0

            self.gym.set_actor_rigid_shape_properties(
                env, actor_handle["object"], obj_props
            )

            props = self.gym.get_actor_dof_properties(env, actor_handle["robot"])
            props["driveMode"].fill(gymapi.DOF_MODE_POS)

            props["stiffness"][:6] = 1000.0  # pgain for arm
            props["damping"][:6] = 10.0  # dgain for arm

            props["stiffness"][6:] = 500.0  # pgain for hand
            props["damping"][6:] = 10.0  # dgain for hand

            self.gym.set_actor_dof_properties(env, actor_handle["robot"], props)

        if self.view_replay:

            actor_handle["robot_replay"] = self.gym.create_actor(
                env, self.assets["vis_robot"], robot_pose, "robot_replay", 2, 1
            )
            actor_handle["object_replay"] = self.gym.create_actor(
                env, self.assets["vis_object"], object_pose, "object_replay", 3, 0
            )

            for obj_name in ["robot_replay", "object_replay"]:
                rigid_body_props = self.gym.get_actor_rigid_body_properties(
                    env, actor_handle[obj_name]
                )

                for prop in rigid_body_props:
                    prop.flags = (
                        gymapi.RIGID_BODY_DISABLE_GRAVITY
                    )  # Disable gravity flag

                self.gym.set_actor_rigid_body_properties(
                    env,
                    actor_handle[obj_name],
                    rigid_body_props,
                    recomputeInertia=False,
                )

                props = self.gym.get_actor_dof_properties(env, actor_handle[obj_name])
                props["driveMode"].fill(gymapi.DOF_MODE_NONE)
                self.gym.set_actor_dof_properties(env, actor_handle[obj_name], props)

            for i in range(29):
                self.gym.set_rigid_body_color(
                    env,
                    actor_handle["robot_replay"],
                    i,
                    gymapi.MESH_VISUAL_AND_COLLISION,
                    gymapi.Vec3(0.4, 0.4, 0.6),
                )

            self.gym.set_rigid_body_color(
                env,
                actor_handle["object_replay"],
                0,
                gymapi.MESH_VISUAL_AND_COLLISION,
                gymapi.Vec3(0.4, 0.4, 0.6),
            )
        if self.num_sphere > 0:
            for i in range(self.num_sphere):
                actor_handle[f"sphere_{i}"] = self.gym.create_actor(
                    env, self.assets["vis_sphere"], object_pose, f"sphere_{i}", 4, 0
                )
        return env, actor_handle

    def add_plane(self):
        # configure the ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)  # z-up!
        plane_params.distance = 0.0525
        plane_params.static_friction = 1
        plane_params.dynamic_friction = 0.8
        plane_params.restitution = 0

        # create the ground plane
        self.gym.add_ground(self.sim, plane_params)

    def set_viewer(self):
        cam_props = gymapi.CameraProperties()

        viewer = self.gym.create_viewer(self.sim, cam_props)
        # Define camera position (e.g., behind and slightly above the actor)
        camera_position = np.array([2, 0, 1])

        # Calculate direction vector (camera looks at the actor)
        camera_direction = -camera_position
        camera_direction = camera_direction / np.linalg.norm(
            camera_direction
        )  # Normalize

        # Set viewer camera transform
        self.gym.viewer_camera_look_at(
            viewer, None, gymapi.Vec3(*camera_position), gymapi.Vec3(0, 0, 0)
        )
        self.viewer = viewer

    def generate_sim(self):
        # 시뮬레이션 설정
        sim_params = gymapi.SimParams()
        # set common parameters
        sim_params.dt = 1 / 30
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

        sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        return sim

    def add_assets(self, obj_name):
        self.assets = {}

        asset_root = rsc_path
        robot_asset_file = "xarm6/xarm6_allegro_wrist_mounted_rotate.urdf"
        object_asset_file = f"{obj_name}/{obj_name}.urdf"

        if self.view_physics:
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

            robot_asset = self.gym.load_asset(
                self.sim, asset_root, robot_asset_file, robot_asset_options
            )
            object_asset = self.gym.load_asset(
                self.sim, asset_root, object_asset_file, object_asset_options
            )

            self.assets["robot"] = robot_asset
            self.assets["object"] = object_asset

        if self.view_replay:
            vis_object_asset_options = gymapi.AssetOptions()
            vis_object_asset_options.disable_gravity = True
            vis_object_asset = self.gym.load_asset(
                self.sim, asset_root, object_asset_file, vis_object_asset_options
            )

            vis_robot_asset_options = gymapi.AssetOptions()
            vis_robot_asset_options.disable_gravity = True
            vis_robot_asset_options.fix_base_link = True
            vis_robot_asset = self.gym.load_asset(
                self.sim, asset_root, robot_asset_file, vis_robot_asset_options
            )

            self.assets["vis_object"] = vis_object_asset
            self.assets["vis_robot"] = vis_robot_asset

        if self.num_sphere > 0:
            sphere_asset_options = gymapi.AssetOptions()
            sphere_asset_options.disable_gravity = True
            sphere_asset_options.fix_base_link = True
            self.assets["vis_sphere"] = self.gym.create_sphere(
                self.sim, 0.05, sphere_asset_options
            )

    def load_camera(self):
        # Add a camera to capture frames
        camera_props = gymapi.CameraProperties()
        camera_props.horizontal_fov = 75.0
        camera_props.width = 1920
        camera_props.height = 1080
        camera_handle = self.gym.create_camera_sensor(self.env, camera_props)

        self.gym.set_camera_location(
            camera_handle,
            self.env,
            gymapi.Vec3(2, 0, 1),
            gymapi.Vec3(0, 0, 0),
        )
        return camera_handle

    def terminate(self):
        if self.save_video:
            self.out.release()
        if self.save_state:
            pickle.dump(self.history, open(self.state_path, "wb"))

        self.gym.destroy_env(self.env)
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

        print("Simulation terminated")

    def get_dof_names(self):
        if self.view_physics:
            return self.gym.get_asset_dof_names(self.assets["robot"])
        elif self.view_replay:
            return self.gym.get_asset_dof_names(self.assets["vis_robot"])
        else:
            raise ValueError("No robot loaded")
