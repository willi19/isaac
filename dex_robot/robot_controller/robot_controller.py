import time
import math

from allegro_hand.controller import AllegroController
from xarm.wrapper import XArmAPI

import numpy as np
from multiprocessing import Process, shared_memory, Event
import datetime
# stable initial pose
XARM_HOME_VALUES = [10.0, -45.0, -45.0, 0.0, 45.0, -90.0]

ALLEGRO_HOME_VALUES = [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]


class DexArmControl:
    def __init__(self, xarm_ip_address="192.168.1.221", xarm_home_pose=None, allegro_home_pose=None):

        self.allegro = AllegroController()
        self.arm = XArmAPI(xarm_ip_address, report_type="devlop")
        
        self.xarm_home_pose = XARM_HOME_VALUES#[v for v in xarm_home_pose]#xarm_home_pose.copy()
        self.allegro_home_pose = allegro_home_pose.copy()
        
        self.max_hand_joint_vel = 100.0 / 360.0 * 2 * math.pi  # 100 degree / sec
        self.last_xarm_command = None
        self.last_allegro_command = None

        self.arm.motion_enable(enable=False)
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)

        self.reset()
        
        self.arm_state_hist = []
        self.arm_state_timestamp = []
        
        self.arm_action_hist = []
        self.arm_action_time_hist = []

        self.hand_state_hist = []
        self.hand_state_timestamp = []
        
        self.hand_action_hist = []
        self.hand_action_time_hist = []

        self.exit = Event()

        self.shm = {}
        self.create_shared_memory("ready", 1 * np.dtype(np.int32).itemsize)
        self.ready_array = np.ndarray((1,), dtype=np.int32, buffer=self.shm["ready"].buf)
        self.ready_array[0] = -1

        self.create_shared_memory("hand_target_action", 16 * np.dtype(np.float32).itemsize)
        self.hand_target_action_array = np.ndarray((16,), dtype=np.float32, buffer=self.shm["hand_target_action"].buf)

        self.create_shared_memory("arm_target_action", 6 * np.dtype(np.float32).itemsize)
        self.arm_target_action_array = np.ndarray((6,), dtype=np.float32, buffer=self.shm["arm_target_action"].buf)

        self.hand_target_action_array[:] = self.allegro_home_pose.copy()
        self.arm_target_action_array[:] = self.xarm_home_pose.copy()

        self.arm_process = Process(target=self.move_arm)
        self.arm_process.start()

        self.hand_process = Process(target=self.move_hand)
        self.hand_process.start()

    def reset(self):
        if self.arm.has_err_warn:
            self.arm.clean_error()

        self.arm.motion_enable(enable=False)
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)  # 0: position control, 1: servo control
        self.arm.set_state(state=0)

    # def set_mode(self, mode=0):
    #     if self.arm.has_err_warn:
    #         self.arm.clean_error()

    #     self.arm.set_mode(mode)
    #     self.arm.set_state(state=0)
    
    def is_ready(self):
        return self.ready_array[0] == 1

    def create_shared_memory(self, name, size):
        try:
            existing_shm = shared_memory.SharedMemory(name=name)
            existing_shm.close()
            existing_shm.unlink()
        except FileNotFoundError:
            pass
        self.shm[name] = shared_memory.SharedMemory(create=True, name=name, size=size)


    def home_robot(self):
        self.ready_array[0] = 0
        
        self.arm_target_action_array = self.xarm_home_pose.copy()
        self.hand_target_action_array = self.allegro_home_pose.copy()

    def move_hand(self):
        fps = 60
        while not self.exit.is_set():
            start_time = time.time()
            if self.ready_array[0] == 1: 
                current_hand_angles = np.asarray(self.allegro.current_joint_pose.position)
                self.allegro.hand_pose(self.hand_target_action_array.copy())
                
                self.hand_state_hist.append(current_hand_angles.copy())
                self.hand_state_timestamp.append(start_time)
                
                self.hand_action_hist.append(self.hand_target_action_array.copy())
                self.hand_action_time_hist.append(start_time)
                
            elif self.ready_array[0] == 0:
                self.allegro.hand_pose(self.hand_target_action_array.copy())

            end_time = time.time()
            time.sleep(max(0, 1 / fps - (end_time - start_time)))

    def move_arm(
        self
    ):
        fps = 100
        while not self.exit.is_set():
            start_time = time.time()
            if self.ready_array[0] == 1:
                continue
                # current_arm_angles = np.asarray(self.arm.get_joint_states(is_radian=True)[1][0])
                # angles = self.arm_target_action_array.copy()
                # angles[:3] = angles[:3] * 1000

                # self.arm.set_servo_cartesian_aa(angles, is_radian=True, relative=False)
                
                # self.arm_state_hist.append(current_arm_angles.copy())
                # self.arm_state_timestamp.append(start_time)
                
                # self.arm_action_hist.append(self.arm_target_action_array.copy())
                # self.arm_action_time_hist.append(start_time)

            elif self.ready_array[0] == 0:
                # self.arm.set_mode(0)  # 0: position control, 1: servo control
                # self.arm.set_state(state=0)

                self.arm.set_servo_angle(angle=XARM_HOME_VALUES, is_radian=False, wait=True)
                self.set_mode(1)
                self.ready_array[0] = 1

            end_time = time.time()
            time.sleep(max(0, 1 / fps - (end_time - start_time)))

    def quit(self):
        self.exit.set()
        self.arm_process.join()
        self.hand_process.join()
        
        now_string = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        np.save(f"arm_state_hist_{now_string}.npy", self.arm_state_hist)
        np.save(f"arm_state_timestamp_{now_string}.npy", self.arm_state_timestamp)

        np.save(f"arm_action_hist_{now_string}.npy", self.arm_action_hist)
        np.save(f"arm_action_time_hist_{now_string}.npy", self.arm_action_time_hist)

        np.save(f"hand_state_hist_{now_string}.npy", self.hand_state_hist)
        np.save(f"hand_state_timestamp_{now_string}.npy", self.hand_state_timestamp)

        np.save(f"hand_action_hist_{now_string}.npy", self.hand_action_hist)
        np.save(f"hand_action_time_hist_{now_string}.npy", self.hand_action_time_hist)


        self.ready.close()
        self.hand_target_action.close()
        self.arm_target_action.close()

        self.arm.motion_enable(enable=False)
        self.arm.disconnect()
        self.allegro.disconnect()

        print("Exiting...")

if __name__ == "__main__":
    dex_arm = DexArmControl()

    dex_arm.home_robot()

    # for i in range(100):
    #     allegro_angles = np.ones(16)
    #     dex_arm.move_hand(allegro_angles=allegro_angles, interpolate=True)

    #     allegro_angles = np.zeros(16)
    #     dex_arm.move_hand(allegro_angles=allegro_angles, interpolate=True)

    print("Done!")
