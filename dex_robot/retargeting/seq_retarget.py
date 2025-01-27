import time
import numpy as np

# XARM_HOME_VALUES = [0.0, -40.0, 0.0, 0.0, 0.0, -90.0]
# XARM_HOME_VALUES_RADIAN = [math.radians(v) for v in XARM_HOME_VALUES]

# ALLEGRO_HOME_VALUES = [
#     0.0,
#     -0.17453293,
#     0.78539816,
#     0.78539816,
#     0.0,
#     -0.17453293,
#     0.78539816,
#     0.78539816,
#     0.08726646,
#     -0.08726646,
#     0.87266463,
#     0.78539816,
#     1.04719755,
#     0.43633231,
#     0.26179939,
#     0.78539816,
# ]


class SeqRetargeting:
    def __init__(
        self,
        optimizer=None,  # dex_robot.retargeting.optimizer.Optimizer
        lp_filter=None,  # dex_robot.retargeting.filter.LPFilter
    ):

        self.optimizer = optimizer
        robot = self.optimizer.robot

        # Joint limit
        joint_limits = robot.joint_limits.copy()
        self.optimizer.set_joint_limit(joint_limits[self.optimizer.idx_pin2target])
        self.joint_limits = joint_limits[self.optimizer.idx_pin2target]

        # Temporal information
        self.init_qpos = joint_limits.mean(1)[self.optimizer.idx_pin2target].astype(
            np.float32
        )

        self.last_qpos = self.init_qpos.copy()
        # for Allegro and XArm, we use the home values as the initial values
        # self.last_qpos = np.concatenate([ALLEGRO_HOME_VALUES, XARM_HOME_VALUES_RADIAN])

        self.accumulated_time = 0
        self.num_retargeting = 0

        # Filter
        self.filter = lp_filter

    def retarget(self, ref_value):
        tic = time.perf_counter()

        qpos = self.optimizer.retarget(
            ref_value=ref_value,  # .astype(np.float32),
            init_qpos=np.clip(
                self.last_qpos, self.joint_limits[:, 0], self.joint_limits[:, 1]
            ),
        )
        self.accumulated_time += time.perf_counter() - tic
        self.num_retargeting += 1
        self.last_qpos = qpos
        robot_qpos = np.zeros(self.optimizer.robot.dof)
        robot_qpos[self.optimizer.idx_pin2target] = qpos

        if self.filter is not None:
            robot_qpos = self.filter.next(robot_qpos)
        return robot_qpos

    def print_status(self):
        min_value = self.optimizer.opt.last_optimum_value()
        print(
            f"Retargeting {self.num_retargeting} times takes: {self.accumulated_time}s"
        )
        print(f"Last distance: {min_value}")

    def reset(self):
        self.last_qpos = self.init_qpos.copy()
        self.num_retargeting = 0
        self.accumulated_time = 0

    @property
    def joint_names(self):
        return self.optimizer.robot.dof_joint_names
