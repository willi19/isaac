import time
import json
import numpy as np
from pathlib import Path
import PySpin as ps

from .camera_setting import CameraConfig
import os

class Camera(CameraConfig):
    def __init__(
        self,
        camPtr,
        lens_info,
        cam_info,
        videoPath,
        syncMode=False,
    ):
        camPtr.Init()  # initialize camera
        self.device_nodemap = camPtr.GetTLDeviceNodeMap()  #
        self.stream_nodemap = camPtr.GetTLStreamNodeMap()  #
        self.nodeMap = camPtr.GetNodeMap()  #
        self.serialnum = self.get_serialnum()
        settingDict = lens_info[str(cam_info[self.serialnum]["lens"])]
        saveVideo = False
        super().__init__(settingDict, saveVideo)

        self.cam = camPtr
        
        self.is_recording = False

        self.timestamps = dict([("timestamps", []), ("frameID", []), ("pc_time", [])])
        
        self.syncMode = syncMode  # True : triggered, False : no trigger,
        self.saveVideo = saveVideo  # true : save in video, false : stream viewer
        # Check for spinview attribute for details

        self.configureSettings(self.nodeMap)
        self.configureBuffer(self.stream_nodemap)
        self.configurePacketSize(self.nodeMap)

        self.VideoPath = videoPath

        self.cam.BeginAcquisition()  # Start acquiring images from camera

    def get_serialnum(self):
        serialnum_entry = self.device_nodemap.GetNode(
            "DeviceSerialNumber"
        )  # .GetValue()
        serialnum = ps.CStringPtr(serialnum_entry).GetValue()
        return serialnum

    def get_capture(self, return_img=True):
        pImageRaw = self.cam.GetNextImage()  # get from buffer
        framenum = pImageRaw.GetFrameID()
        capture_time = time.time()
        if not pImageRaw.IsIncomplete():
            chunkData = pImageRaw.GetChunkData()
            ts = chunkData.GetTimestamp()
            self.timestamps["timestamps"].append(ts)
            self.timestamps["frameID"].append(framenum)
            self.timestamps["pc_time"].append(capture_time)

        else:
            print(ps.Image_GetImageStatusDescription(pImageRaw.GetImageStatus()))
            
        pImageRaw.Release()
        return 

    def stop_camera(self):
        self.cam.EndAcquisition()
        self.cam.DeInit()
        del self.cam
        return

    # for saving file
    def set_record(self):
        if self.is_recording:
            print("Stop Recording")
            os.makedirs(self.VideoPath, exist_ok=True)
            self.is_recording = False

            json.dump(
                self.timestamps, open(os.path.join(self.VideoPath,"camera_timestamp.json"), "w"), indent="\t"
            )
            print("Video Save finished")
        else:
            self.is_recording = True
            print("Start Recording")
        return

    def configureSettings(self, nodeMap):
        self.configureGain(nodeMap)
        self.configureThroughPut(nodeMap)
        # configureTrigger(nodeMap)
        if not self.syncMode:
            self.configureFrameRate(nodeMap)  # we use trigger anyway
        else:
            self.configureTrigger(nodeMap)
        self.configureExposure(nodeMap)
        self.configureAcquisition(nodeMap)
        # Set Exposure time, Gain, Throughput limit, Trigger mode,
        self.configureChunk(nodeMap)  # getting timestamp
        # self.configureBuffer(nodeMap)
        return
