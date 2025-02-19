from utils import robot_wrapper
from mplib import pose


class IK_solver:
    def __init__(self, urdf_path: str):
        self.robot = robot_wrapper.RobotWrapper(urdf_path)

    def inverse_kinematics(self, target_pos, init_qpos):
        qpos = self.robot.inverse_kinematics(target_pos, init_qpos)
        return qpos

    def retarget(self, ref_value, init_qpos):
        qpos = self.robot.compute_IK_CLIK_JL(ref_value, init_qpos)
        return qpos

    def set_init_qpos(self, init_qpos):
        self.robot.set_init_qpos(init_qpos)
