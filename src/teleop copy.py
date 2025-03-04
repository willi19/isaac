import os
from pathlib import Path
import sys
from typing import Dict, List
import time

import rospy

from isaacgym import gymapi, gymtorch

import torch
import numpy as np
import transforms3d

from dex_retargeting.constants import RobotName, HandType, RobotType, get_default_config_path, RetargetingType
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting
from hand_viewer_isaac import HandISAACViewer


XSENS2XARM = np.array(
    [
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
    ]
)


XSENS2ISAAC = np.array(
    [
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
    ]
)

XARM2ALLEGRO = np.array(
    [
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
    ]
)

XSENS2ALLEGRO = XSENS2XARM @ XARM2ALLEGRO


def prepare_position_retargeting(joint_pos: np.array, link_hand_indices: np.ndarray):
    link_pos = joint_pos[link_hand_indices]
    return link_pos


def prepare_vector_retargeting(joint_pos: np.array, link_hand_indices_pairs: np.ndarray):
    joint_pos = joint_pos @ XSENS2XARM
    origin_link_pos = joint_pos[link_hand_indices_pairs[0]]
    task_link_pos = joint_pos[link_hand_indices_pairs[1]]
    return task_link_pos - origin_link_pos


class LPFilter:
    def __init__(self, control_freq, cutoff_freq):
        dt = 1 / control_freq
        wc = cutoff_freq * 2 * np.pi
        y_cos = 1 - np.cos(wc * dt)
        self.alpha = -y_cos + np.sqrt(y_cos**2 + 2 * y_cos)
        self.y = 0
        self.is_init = False

    def next(self, x):
        self.y = self.y + self.alpha * (x - self.y)
        return self.y.copy()

    def init(self, y):
        self.y = y.copy()
        self.is_init = True


class RobotHandISAACViewer(HandISAACViewer):
    def __init__(
        self,
        is_real: bool = False,
        visualize_sim: bool = True,
        retargeting_type: RetargetingType = RetargetingType.position,
        robot_names: List[RobotName] = [RobotName.allegro],
        robot_type_list: List[RobotType] = [RobotType.hand],
        hand_type: HandType = HandType.right,
        headless=False,
        robot_urdf_dir_list: List[str] = [],
        hand_urdf_dir: str = "../../assets/robots/hands/human_mano/hand_r_wrist_box.xml",
        y_offset: float = 0.8,
        arm_control_mode="servo_cartesian_aa",
        arm_relative=False,
    ):
        """
        Args:
            is_real (bool, optional): True if using real robot. Defaults to False.
            visualize_sim (bool, optional): True if using Issac for visualization. Defaults to True.
            headless (bool, optional): True if using headless server. Defaults to False.

            y_offset (float, optional): Offset of robots for visualization in Isaac. Defaults to 0.8.

            retargeting_type (RetargetingType, optional): Retargeting type. Defaults to RetargetingType.position.

            hand_type (HandType, optional): Hand type. Defaults to HandType.right.
            hand_urdf_dir (str, optional): _description_. Defaults to "../../assets/robots/hands/human_mano/hand_r_wrist_box.xml".

            robot_names (List[RobotName], optional): List of robot names. Defaults to [RobotName.allegro].
            robot_type_list (List[RobotType], optional): List of robot types. Defaults to [RobotType.hand].
            robot_urdf_dir_list (List[str], optional): List of robot urdf dir. Defaults to [].


            arm_control_mode (str, optional): Arm control mode. Only used in real robot teleoperation. Defaults to "servo_cartesian_aa".
            arm_relative (bool, optional): Whether to use relative movement of teleoperator or not. Defaults to False.
        """
        self.is_real = is_real
        self.visualize_sim = visualize_sim

        if visualize_sim:
            super().__init__(headless=headless)

        # set robot related info
        self.robot_type_list = robot_type_list
        self.num_robots = len(robot_names)

        if self.num_robots > 1:
            raise ValueError("Currently only support one robot")

        if self.robot_type_list[0] == RobotType.hand:
            assert (
                arm_control_mode == "servo_cartesian_aa"
            ), "Currently only support servo_cartesian_aa for floating hand urdf"
        else:
            assert self.robot_type_list[0] == RobotType.handarm
            # assert arm_control_mode == "servo_angle_j", "Currently only support servo_angle_j for handarm urdf"

        # set pose offset for each robot
        self.pose_offsets = [-y_offset * i for i in range(self.num_robots + 1)]

        self.robot_assets = []
        self.robot_rb_handles = []
        self.robot_actor_indices = []

        self.robots = []
        self.robot_file_names: List[str] = []

        # robot urdf dir
        self.robot_urdf_dir_list = robot_urdf_dir_list

        # set control mode and whether to use relative values for XArm control.
        self.arm_control_mode = arm_control_mode
        self.arm_relative = arm_relative

        # set retargeting info
        self.retargeting_type = retargeting_type
        self.retargetings: List[SeqRetargeting] = []

        # Load optimizer and filter
        for robot_name in robot_names:
            config_path = get_default_config_path(robot_name, self.retargeting_type, hand_type, online=True)
            config = RetargetingConfig.load_from_file(config_path)

            # set robot urdf dir
            self.robot_urdf_dir_list.append(config.urdf_path)
            retargeting = config.build()

            robot = retargeting.optimizer.robot
            robot_file_name = Path(config.urdf_path).stem

            self.robots.append(robot)
            self.robot_file_names.append(robot_file_name)
            self.retargetings.append(retargeting)

    def load_hand(self):
        # Load humand hand and robot hands
        if self.visualize_sim:
            # load human hand
            super().load_hand()

            # load robot hand
            asset_options = gymapi.AssetOptions()
            asset_options.flip_visual_attachments = False
            asset_options.fix_base_link = False
            asset_options.collapse_fixed_joints = True
            asset_options.disable_gravity = False

            for i, robot_urdf_dir in enumerate(self.robot_urdf_dir_list):
                robot_path = Path(robot_urdf_dir)
                robot_urdf_root = str(robot_path.parent)
                robot_urdf_file = robot_path.name
                robot_name = robot_path.stem

                robot_asset = self.gym.load_asset(self.sim, robot_urdf_root, robot_urdf_file, asset_options)
                self.robot_assets.append(robot_asset)

                initial_pose = gymapi.Transform()
                initial_pose.p = gymapi.Vec3(0, self.pose_offsets[i + 1], 0.0)
                initial_pose.r = gymapi.Quat(0, 0, 0, 1)

                robot_actor = self.gym.create_actor(self.env, robot_asset, initial_pose, robot_name, 0, -1)
                self.robot_rb_handles.append(robot_actor)

                robot_indices = self.gym.get_actor_index(self.env, robot_actor, gymapi.DOMAIN_SIM)
                self.robot_actor_indices.append(robot_indices)

            # set root state, dof state tensor
            self.root_state_tensors_ = self.gym.acquire_actor_root_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.root_state_tensors = gymtorch.wrap_tensor(self.root_state_tensors_)

            dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
            self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)

        else:
            pass  # no simulation loading

    def render_hand_teleoperate_stream(self, stream: Dict, fps=60, y_offset=0.8):
        """
        Get current robot state from ros, and set it to the robot. (not implemented yet)
        Also perform real-time retargeting and publish robot target joint topic.
        """
        assert (
            self.is_real or self.visualize_sim
        ), "Need to set is_real=True or visualize_sim=True to render hand stream"

        if self.is_real:
            from move_dexarm import DexArmControl

            arm_controller = DexArmControl()

        if self.visualize_sim:
            # Set camera view
            cam_pos = gymapi.Vec3(1.5, -0.05, 1.0)
            cam_target = gymapi.Vec3(0, -0.05, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

            self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)

        first = True
        first_pelvis_pose = None
        warmstart_timestep = 15

        xarm_position_ema = None
        xarm_axisangle_ema = None

        # for replay
        # qpos_list = np.load("save_target_qpos_list_thumbdown.npy")

        # for recording
        save_robot_qpos_list = []
        save_robot_wrist_list = []
        save_target_qpos_list = []

        save_hand_pose_list = []
        save_whole_body_pose_list = []

        save_time_list = []

        homepath = Path.home()

        exp_name = "test_0221"
        scene_name = "test_1"
        is_save = False
        os.makedirs(homepath / f"ws/shared_data/teleoperation/{exp_name}", exist_ok=True)
        os.makedirs(homepath / f"ws/shared_data/teleoperation/{exp_name}/{scene_name}", exist_ok=True)

        ind = 0

        # rate = rospy.Rate(60)
        
        start_time = time.time()
        while True:
            current_time = time.time()
            print("current_time: ", current_time - start_time)
            if current_time - start_time > 500 and is_save: # break after 500 sec
                save_robot_qpos_list = np.array(save_robot_qpos_list)
                save_robot_wrist_list = np.array(save_robot_wrist_list)
                save_target_qpos_list = np.array(save_target_qpos_list)
                save_hand_pose_list = np.array(save_hand_pose_list)
                save_whole_body_pose_list = np.array(save_whole_body_pose_list)
                save_time_list = np.array(save_time_list)

                np.save(
                    homepath / f"ws/shared_data/teleoperation/{exp_name}/{scene_name}/robot_qpos_list.npy",
                    save_robot_qpos_list,
                )
                np.save(
                    homepath / f"ws/shared_data/teleoperation/{exp_name}/{scene_name}/robot_wrist_list.npy",
                    save_robot_wrist_list,
                )
                np.save(
                    homepath / f"ws/shared_data/teleoperation/{exp_name}/{scene_name}/target_qpos_list.npy",
                    save_target_qpos_list,
                )
                np.save(
                    homepath / f"ws/shared_data/teleoperation/{exp_name}/{scene_name}/hand_pose_list.npy",
                    save_hand_pose_list,
                )
                np.save(
                    homepath / f"ws/shared_data/teleoperation/{exp_name}/{scene_name}/whole_body_pose_list.npy",
                    save_whole_body_pose_list,
                )
                np.save(
                    homepath / f"ws/shared_data/teleoperation/{exp_name}/{scene_name}/timestemp.npy",
                    save_time_list,
                )

                print("Data saved")

                break

            end_time = time.time()
            if end_time - current_time - 1/60 > 0:
                time.sleep(end_time - current_time - 1/60)
            # print("sleep_time: ", time() - sleep_time)

            if warmstart_timestep < 0:
                tmp_time = time.time() - cycle_time
                print("cycle time :{}".format(tmp_time))
                save_time_list.append(tmp_time)

            cycle_time = time.time()

            warmstart_timestep -= 1
            data = stream.get_data()


            if len(data) == 0:
                print("No data received")
                continue

            hand_pose_frame = data["hand_pose"].copy()
            print(hand_pose_frame)
            hand_joint_angle = data["hand_joint_angle"].copy()

            whole_body_pose_frame = data["pose_data"][0].copy()
            pelvis_pose = whole_body_pose_frame[0]

            if warmstart_timestep < 0:
                save_hand_pose_list.append(hand_pose_frame.copy())
                save_whole_body_pose_list.append(whole_body_pose_frame.copy())

            # compute hand pose relative to pelvis at first frame
            hand_pose_frame = [np.linalg.inv(pelvis_pose) @ pose for pose in hand_pose_frame]

            if first:
                first_pelvis_pose = pelvis_pose
            else:
                rel_pelvis_pose = np.linalg.inv(first_pelvis_pose) @ pelvis_pose
                hand_pose_frame = [rel_pelvis_pose @ pose for pose in hand_pose_frame]

            # for human hand wrist
            xsens_human_wrist_rotmat = hand_pose_frame[0][:3, :3]  # rotation of wrist
            xsens_human_wrist_position = hand_pose_frame[0][:3, 3]  # translation of wrist

            isaac_human_wrist_rotmat = xsens_human_wrist_rotmat.copy() @ XSENS2ISAAC

            isaac_human_wrist_position = xsens_human_wrist_position
            issac_human_wrist_quat = transforms3d.quaternions.mat2quat(isaac_human_wrist_rotmat)

            if self.visualize_sim:
                # set root state tensor for human hand
                self.gym.refresh_actor_root_state_tensor(self.sim)
                self.root_state_tensors[self.hand_rb_handles[0]][:3] = torch.from_numpy(isaac_human_wrist_position)
                self.root_state_tensors[self.hand_rb_handles[0]][6:7] = torch.from_numpy(issac_human_wrist_quat[0:1])
                self.root_state_tensors[self.hand_rb_handles[0]][3:6] = torch.from_numpy(issac_human_wrist_quat[1:4])
                self.gym.set_actor_root_state_tensor(self.sim, self.root_state_tensors_)

                # set dof states for human hand
                hand_dof_states = self.gym.get_actor_dof_states(self.env, self.hand_actor_indices[0], gymapi.STATE_ALL)
                # hand_dof_states["pos"] = hand_axisangle[0].numpy() # TODO: get axisangle from streamer
                self.gym.set_actor_dof_states(self.env, self.hand_actor_indices[0], hand_dof_states, gymapi.STATE_POS)

            # for XArm hand wrist (if using floating hand, just use human_wrist values. XArm's last joint frame is upside down, so need to flip it)
            xarm_wrist_rotmat = np.dot(xsens_human_wrist_rotmat, XSENS2XARM)
            xarm_wrist_position = isaac_human_wrist_position
            xarm_wrist_axisangle, norm = transforms3d.axangles.mat2axangle(xarm_wrist_rotmat)
            xarm_wrist_eulerangle = transforms3d.euler.mat2euler(xarm_wrist_rotmat, "sxyz")
            xarm_wrist_axisangle = xarm_wrist_axisangle * norm
            xarm_wrist_pos_axisangle = np.concatenate([xarm_wrist_position, xarm_wrist_axisangle])

            # LPF filter for wrist position and rotation angle. (Currently not used)
            alpha = 1.0

            if first:
                xarm_position_ema = xarm_wrist_position
                xarm_axisangle_ema = xarm_wrist_axisangle

            xarm_wrist_position = (1 - alpha) * xarm_position_ema + alpha * xarm_wrist_position
            xarm_wrist_axisangle = (1 - alpha) * xarm_axisangle_ema + alpha * xarm_wrist_axisangle

            xarm_position_ema = xarm_wrist_position.copy()
            xarm_axisangle_ema = xarm_wrist_axisangle.copy()
            # Manus hand position values for retargeting
            joint = np.array([hand_pose_frame[i][:3, 3] for i in range(len(hand_pose_frame))])

            xarm_prev = None
            # Retargeting for each robot hand
            for i, retargeting in zip(range(self.num_robots), self.retargetings):
                t = time.time()

                qpos_sim = np.zeros(self.robots[i].dof)
                qpos_real = np.zeros(self.robots[i].dof)

                if self.retargeting_type != RetargetingType.direct:

                    indices = retargeting.optimizer.target_link_human_indices

                    if self.retargeting_type == RetargetingType.position:
                        ref_value = joint[indices, :]
                        qpos_sim = retargeting.retarget(ref_value)

                    else:
                        # currently only support vector retargeting (no Dexpilot)
                        assert self.retargeting_type == RetargetingType.vector
                        assert self.robot_type_list[i] == RobotType.hand

                        origin_indices = indices[0, :]
                        task_indices = indices[1, :]
                        ref_value = joint[task_indices, :] - joint[origin_indices, :]
                        qpos_sim = retargeting.retarget(ref_value)
                        qpos_sim[:3] = isaac_human_wrist_position
                        qpos_sim[3:6] = transforms3d.euler.mat2euler(isaac_human_wrist_rotmat, "sxyz")

                    allegro_angles = qpos_sim[6:].copy()

                    if self.robot_type_list[i] == RobotType.handarm:
                        qpos_real[:6] = qpos_sim[:6].copy()

                    else:
                        assert self.robot_type_list[i] == RobotType.hand
                        qpos_real[:3] = xarm_wrist_position
                        qpos_real[3:6] = xarm_wrist_axisangle

                else:
                    assert self.retargeting_type == RetargetingType.direct
                    # assert self.robot_type_list[i] == RobotType.hand

                    # This is wrong for qpos_sim. Need to use IK for RobotType handarm, use position and euler angle for RobotType hand.
                    qpos_sim[:3] = xarm_wrist_position
                    qpos_sim[3:6] = xarm_wrist_eulerangle

                    qpos_real[:3] = xarm_wrist_position * 1.5
                    qpos_real[0] = qpos_real[0] - 0.2
                    qpos_real[2] = qpos_real[2] - 0.2
                    qpos_real[3:6] = xarm_wrist_axisangle

                    # TODO : figure this out, make it as a function.
                    allegro_angles = np.zeros(16)
                    # zyx euler angle in hand frame = zxy axis angle in robot frame
                    allegro_angles[0] = hand_joint_angle[5][0]  # z in robot, y in hand
                    allegro_angles[1] = hand_joint_angle[5][2] * 1.2  # y in robot, z in hand
                    allegro_angles[2] = hand_joint_angle[6][2] * 0.8
                    allegro_angles[3] = hand_joint_angle[7][2] * 0.8

                    thumb_meta = np.dot(hand_pose_frame[0].T, hand_pose_frame[1])
                    thumb_meta_angle = transforms3d.euler.mat2euler(thumb_meta, "sxyz")

                    # for drum
                    allegro_angles[4] = thumb_meta_angle[0]  # -x in robot, y in hand
                    allegro_angles[5] = thumb_meta_angle[1] - 0.5  # y in robot, z in hand
                    allegro_angles[6] = hand_joint_angle[2][2] * 1.2
                    allegro_angles[7] = hand_joint_angle[3][2] * 1.2

                    # for others
                    # allegro_angles[4] = thumb_meta_angle[0]  # -x in robot, y in hand
                    # allegro_angles[5] = thumb_meta_angle[1] * 0.1  # y in robot, z in hand
                    # allegro_angles[6] = hand_joint_angle[2][2] + 1.0
                    # allegro_angles[7] = hand_joint_angle[3][2] * 1.2

                    allegro_angles[8] = hand_joint_angle[9][0]  # z in robot, y in hand
                    allegro_angles[9] = hand_joint_angle[9][2] * 1.2  # y in robot, z in hand
                    allegro_angles[10] = hand_joint_angle[10][2] * 0.8
                    allegro_angles[11] = hand_joint_angle[11][2] * 0.8

                    allegro_angles[12] = hand_joint_angle[13][0]  # z in robot, y in hand
                    allegro_angles[13] = hand_joint_angle[13][2] * 1.2  # y in robot, z in hand
                    allegro_angles[14] = hand_joint_angle[14][2] * 0.8
                    allegro_angles[15] = hand_joint_angle[15][2] * 0.8

                    qpos_sim[6:] = allegro_angles

                # joint order of Allegro in sim and real are different
                allegro_angles_tmp = allegro_angles.copy()
                allegro_angles = np.zeros_like(allegro_angles_tmp)

                # Todo: Need to change this hard-coded mapping
                allegro_angles[0:4] = allegro_angles_tmp[0:4]
                allegro_angles[4:12] = allegro_angles_tmp[8:16]
                allegro_angles[12:16] = allegro_angles_tmp[4:8]
                qpos_real[6:] = allegro_angles

                # if first:
                #     xarm_prev_position = xarm6_position
                #     xarm_prev_rotmat = xarm6_rotmat

                # xarm6_rel_position = xarm6_position - xarm_prev_position
                # xarm6_rel_rotmat = np.dot(xarm_prev_rotmat.T, xarm6_rotmat)
                # xarm6_rel_axisangle, norm = transforms3d.axangles.mat2axangle(xarm6_rel_rotmat)
                # xarm6_rel_axisangle = xarm6_rel_axisangle * norm
                # xarm6_rel_pos_axisangle = np.concatenate([xarm6_rel_position, xarm6_rel_axisangle])

                # print("Retargeting fps :{}".format(1 / (time() - t)))

                if warmstart_timestep < 0:
                    save_target_qpos_list.append(qpos_real.copy())

                # import ipdb

                # ipdb.set_trace()

                # qpos_real = qpos_list[ind]
                # if ind < 15:
                #     qpos_real[:3] = qpos_real[:3].copy() / 1000
                # print(ind, qpos_real)
                # ind = ind + 1
                # allegro_angles = qpos_real[6:].copy()

                if warmstart_timestep > 0:
                    print("Warmstarting...")

                    if self.is_real:

                        if self.arm_control_mode == "servo_angle_j":
                            arm_controller.arm.set_servo_angle(
                                angle=qpos_real[:6], is_radian=True, relative=False, wait=True
                            )

                        else:
                            qpos_real[:6][:3] = qpos_real[:6][:3] * 1000

                            arm_controller.arm.set_position_aa(axis_angle_pose=qpos_real[:6], wait=True, is_radian=True)

                        if warmstart_timestep == 1:
                            arm_controller.arm.set_mode(1)
                            arm_controller.arm.set_state(state=0)

                    ("first init done.")
                    first = False

                else:
                    if self.is_real:

                        arm_controller.move_robot_servo(
                            allegro_angles, qpos_real[:6], arm_control_mode=self.arm_control_mode, arm_relative=False
                        )

                    first = False

                # if warmstart_timestep < 0:

                #     xarm_angles_real, allegro_angles_real = arm_controller.get_joint_values()
                #     cur_qpos = np.concatenate([xarm_angles_real, allegro_angles_real])
                #     save_robot_qpos_list.append(cur_qpos.copy())

                #     xarm_wrist = arm_controller.get_arm_position()
                #     save_robot_wrist_list.append(xarm_wrist.copy())

                # print("after_control: {}".format(time()))
                # print("control fps :{}".format(1 / (time() - t)))

                if self.visualize_sim:
                    # set dof states for each robot hand
                    robot_dof_states = self.gym.get_actor_dof_states(
                        self.env, self.robot_actor_indices[i], gymapi.STATE_ALL
                    )

                    robot_dof_states["pos"] = qpos_sim

                    if self.is_real:
                        xarm_angles_real, allegro_angles_real = arm_controller.get_joint_values()

                        if self.robot_type_list[i] == RobotType.handarm:
                            robot_dof_states["pos"][:6] = xarm_angles_real[:6]
                        else:
                            assert self.robot_type_list[i] == RobotType.hand

                            # Currently this doesn't match for rotation
                            robot_dof_states["pos"][:6] = arm_controller.get_arm_position()

                        # joint order of Allegro in sim and real are different

                        allegro_angles_tmp = allegro_angles_real.copy()
                        allegro_angles = np.zeros_like(allegro_angles_tmp)

                        allegro_angles[0:4] = allegro_angles_tmp[0:4]
                        allegro_angles[8:16] = allegro_angles_tmp[4:12]
                        allegro_angles[4:8] = allegro_angles_tmp[12:16]
                        robot_dof_states["pos"][6:] = allegro_angles

                    self.gym.set_actor_dof_states(
                        self.env, self.robot_actor_indices[i], robot_dof_states, gymapi.STATE_POS
                    )

            if self.visualize_sim:
                self.gym.fetch_results(self.sim, True)
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                self.gym.sync_frame_time(self.sim)
