import time
import math

from allegro_hand.controller import AllegroController
from xarm.wrapper import XArmAPI

import numpy as np
from multiprocessing import Process, shared_memory, Event, Lock

import os


class DexArmControl:
    def __init__(self, save_path, xarm_ip_address="192.168.1.221"):

        self.xarm_ip_address = xarm_ip_address
        self.xarm_home_pose = None#[v for v in xarm_home_pose]#xarm_home_pose.copy()
        self.allegro_home_pose = None
        
        self.capture_path = save_path
        os.makedirs(self.capture_path, exist_ok=True)

        self.max_hand_joint_vel = 100.0 / 360.0 * 2 * math.pi  # 100 degree / sec
        self.last_xarm_command = None
        self.last_allegro_command = None

        self.arm_state_hist = np.zeros((60000, 6), dtype=np.float64)
        self.arm_action_hist = np.zeros((60000, 6), dtype=np.float64)
        self.arm_timestamp = np.zeros((60000, 1), dtype=np.float64)
        
        
        self.hand_state_hist = np.zeros((60000, 16), dtype=np.float64)
        self.hand_timestamp = np.zeros((60000, 1), dtype=np.float64)        
        self.hand_action_hist = np.zeros((60000, 16), dtype=np.float64)


        self.first_home = True
        self.home_cnt = 0

        self.exit = Event()

        self.shm = {}
        self.create_shared_memory("ready", 1 * np.dtype(np.int32).itemsize)
        self.ready_array = np.ndarray((1,), dtype=np.int32, buffer=self.shm["ready"].buf)
        self.ready_array[0] = -1

        self.create_shared_memory("hand_target_action", 16 * np.dtype(np.float32).itemsize)
        self.hand_target_action_array = np.ndarray((16,), dtype=np.float32, buffer=self.shm["hand_target_action"].buf)

        self.create_shared_memory("arm_target_action", 6 * np.dtype(np.float32).itemsize)
        self.arm_target_action_array = np.ndarray((6,), dtype=np.float32, buffer=self.shm["arm_target_action"].buf)

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
        time.sleep(1)

    def set_homepose(self, xarm_home_pose, allegro_home_pose):
        self.xarm_home_pose = xarm_home_pose.copy()
        self.allegro_home_pose = allegro_home_pose.copy()
    
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
        self.arm_target_action_array[:] = self.xarm_home_pose.copy()
        self.hand_target_action_array[:] = self.allegro_home_pose.copy()
        self.ready_array[0] = 0

    def move_hand(self):
        fps = 60
        self.hand_cnt = 0
        self.allegro = AllegroController()
        self.hand_lock = Lock()
        while not self.exit.is_set():
            start_time = time.time()
            action = self.hand_target_action_array.copy()

            if self.ready_array[0] == 1: 
                current_hand_angles = np.asarray(self.allegro.current_joint_pose.position)
                self.allegro.hand_pose(action)
                
                self.hand_state_hist[self.hand_cnt] = current_hand_angles.copy()
                self.hand_timestamp[self.hand_cnt] = start_time

                self.hand_action_hist[self.hand_cnt] = self.hand_target_action_array.copy()
                self.hand_cnt += 1
                
            elif self.ready_array[0] == 0:
                self.allegro.hand_pose(action)

            end_time = time.time()
            time.sleep(max(0, 1 / fps - (end_time - start_time)))
        
        os.makedirs(os.path.join(self.capture_path, "hand"), exist_ok=True)
        np.save(os.path.join(self.capture_path, "hand", f"state.npy"), self.hand_state_hist[:self.hand_cnt])
        np.save(os.path.join(self.capture_path, "hand", f"timestamp.npy"), self.hand_timestamp[:self.hand_cnt])
        np.save(os.path.join(self.capture_path, "hand", f"action.npy"), self.hand_action_hist[:self.hand_cnt])
        

    def move_arm(
        self
    ):
        self.arm = XArmAPI(self.xarm_ip_address, report_type="devlop")
        self.arm.motion_enable(enable=False)
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)

        self.reset()
        
        fps = 100
        self.arm_cnt = 0
        self.lock_arm = Lock()

        while not self.exit.is_set():
            start_time = time.time()
            with self.lock_arm:        
                angles = self.arm_target_action_array.copy()
                angles[:3] *= 1000
            if self.ready_array[0] == 100:
                
                current_arm_angles = np.asarray(self.arm.get_joint_states(is_radian=True)[1][0][:6])
                

                self.arm.set_servo_cartesian_aa(angles, is_radian=True, relative=False)
                
                self.arm_state_hist[self.arm_cnt] = current_arm_angles.copy()
                self.arm_timestamp[self.arm_cnt] = start_time
                
                self.arm_action_hist[self.arm_cnt] = angles.copy()
                self.arm_cnt += 1
                

            elif self.ready_array[0] == 0:
                if not self.first_home:
                    self.arm.set_mode(0)  # 0: position control, 1: servo control
                    self.arm.set_state(state=0)
                self.first_home = False

                angles = self.arm_target_action_array.copy()
                angles[:3] *= 1000

                self.arm.set_position_aa(axis_angle_pose=angles, is_radian=True, wait=True)
                self.arm.set_mode(1)
                self.arm.set_state(state=0)
                time.sleep(0.1)
                
                self.ready_array[0] = 1

            end_time = time.time()
            time.sleep(max(0, 1 / fps - (end_time - start_time)))

        
        os.makedirs(os.path.join(self.capture_path, "arm"), exist_ok=True)
        np.save(os.path.join(self.capture_path, "arm", f"state.npy"), self.arm_state_hist[:self.arm_cnt])
        np.save(os.path.join(self.capture_path, "arm", f"action.npy"), self.arm_action_hist[:self.arm_cnt])
        np.save(os.path.join(self.capture_path, "arm", f"timestamp.npy"), self.arm_timestamp[:self.arm_cnt])
        
        self.arm.motion_enable(enable=False)
        self.arm.disconnect()

    def set_robot_servo(self,allegro_angles, xarm_angles):
        self.arm_target_action_array[:] = xarm_angles.copy()
        self.hand_target_action_array[:] = allegro_angles.copy()
        
    def quit(self):
        self.exit.set()
        self.arm_process.join()
        self.hand_process.join()

        for key in self.shm.keys():
            self.shm[key].close()
            self.shm[key].unlink()

        print("Exiting...")

if __name__ == "__main__":
    dex_arm = DexArmControl()

    dex_arm.home_robot()
    print("Done!")
