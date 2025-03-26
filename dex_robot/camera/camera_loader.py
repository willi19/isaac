import threading
import PySpin
import json
import time
from multiprocessing import shared_memory, Lock, Value, Event, Process
from dex_robot.camera import camera
import numpy as np
import os

homedir = os.path.expanduser("~")

class CameraManager:
    def __init__(self, save_path, num_cameras=1, is_streaming=False, syncMode=True, shared_memories={}, update_flags={}):
        self.num_cameras = num_cameras

        self.is_streaming = is_streaming
        self.save_path = save_path

        self.stop_event = Event()
        self.shared_memories = shared_memories
        self.update_flags = update_flags
        self.locks = {}

        self.capture_threads = []
        self.frame_cnt = 0
        self.syncMode = syncMode


    def configure_camera(self, cam):
        nodemap = cam.GetNodeMap()
        node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode("AcquisitionMode"))
        node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName("Continuous")
        node_acquisition_mode.SetIntValue(node_acquisition_mode_continuous.GetValue())
        

    def create_shared_memory(self, camera_index, shape, dtype):
        """
        Creates shared memory and lock for a camera.
        """
        shm_name = f"camera_{camera_index}_shm"
        try:
            existing_shm = shared_memory.SharedMemory(name=shm_name)
            existing_shm.close()
            existing_shm.unlink()
        except FileNotFoundError:
            pass
    

        shm = shared_memory.SharedMemory(create=True, name=shm_name, size=np.prod(shape) * np.dtype(dtype).itemsize)
        self.shared_memories[camera_index] = {
            "name": shm_name,
            "shm": shm,
            "array": np.ndarray(shape, dtype=dtype, buffer=shm.buf),
            "lock": Lock(),
        }
        self.update_flags[camera_index] = Value('i', 0)  # 0: not updated, 1: updated

        print(f"Shared memory created for camera {camera_index}.")

    def capture_video(self, camera_index):
        system = PySpin.System.GetInstance()
        cam_list = system.GetCameras()
        
        if cam_list.GetSize() <= camera_index:
            print(f"Camera index {camera_index} is out of range.")
            cam_list.Clear()
            system.ReleaseInstance()
            return

        camPtr = cam_list.GetByIndex(camera_index)
        lens_info = json.load(open("config/camera/lens.json", "r"))
        cam_info = json.load(open("config/camera/camera.json", "r"))

        # if self.is_streaming:
        #     shm_info = self.shared_memories[camera_index]
        #     update_flag = self.update_flags[camera_index]
        cam = camera.Camera(camPtr, lens_info, cam_info, self.save_path, syncMode=self.syncMode)

        if not self.is_streaming:
            cam.set_record()
        try:
            start_time = time.time()
            while not self.stop_event.is_set():
                cam.get_capture()
                # frame, ret = cam.get_capture()
                # cnt += 1
                # if ret and self.is_streaming:
                #     with shm_info["lock"]:
                #         np.copyto(shm_info["array"], img)
                #         update_flag.value = 1  # Mark as updated
        except Exception as e:
            print(e, repr(e))
        finally:
            if not self.is_streaming:
                cam.set_record()
            cam.stop_camera()
            del camPtr
            cam_list.Clear()
            system.ReleaseInstance()
    
    def start(self):
        frame_shape = (1536, 2048, 3)  # Example shape for each frame (RGB)
        frame_dtype = np.uint8
        if self.is_streaming:
            for i in range(self.num_cameras):
                self.create_shared_memory(i, frame_shape, frame_dtype)

        self.capture_threads = [
            Process(target=self.capture_video, args=(i,))
            for i in range(self.num_cameras)
        ]

        for p in self.capture_threads:
            p.start()


    def quit(self):
        self.stop_event.set()
        
        # Wait for all threads to finish
        for p in self.capture_threads:
            p.join()

        print("All capture threads have stopped.")
        
if __name__ == "__main__":
    manager = CameraManager(name="test", is_streaming=False, syncMode=False)
    manager.start()
    start_time = time.time()
    while time.time() - start_time < 5:
        # print(time.time()-start_time, "s\n")
        time.sleep(0.03)
    manager.quit()
