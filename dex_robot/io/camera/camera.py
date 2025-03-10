import time
import json
from datetime import datetime
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
        saveVideoPath=None,
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
        self.saveVideoPath = saveVideoPath
        # Check for spinview attribute for details

        self.configureSettings(self.nodeMap)
        self.configureBuffer(self.stream_nodemap)
        self.configurePacketSize(self.nodeMap)

        # self.image_processor = None
        
        # self.videoName = None
        # self.saveVideoPath = saveVideoPath

        # self.videoStream = ps.SpinVideo()
        # video_option = ps.AVIOption()

        # # # Set the video file format (e.g., MP4, AVI)
        # video_option.frameRate = 30  # Set the desired frame rate
        # video_option.height=1536
        # video_option.width=2048
        # self.videoOption = video_option
        self.cam.BeginAcquisition()  # Start acquiring images from camera

    def get_serialnum(self):
        serialnum_entry = self.device_nodemap.GetNode(
            "DeviceSerialNumber"
        )  # .GetValue()
        serialnum = ps.CStringPtr(serialnum_entry).GetValue()
        return serialnum

    def get_now(self):
        now = datetime.now()
        return now.strftime("%Y%m%d%H%M%S")

    def get_capture(self, return_img=True):
        pImageRaw = self.cam.GetNextImage()  # get from buffer
        framenum = pImageRaw.GetFrameID()
        capture_time = time.time()
        # print(framenum, capture_time)
        if not pImageRaw.IsIncomplete():
            chunkData = pImageRaw.GetChunkData()
            # print("chunkd : ", time.time() - before)
            ts = chunkData.GetTimestamp()
            # print("tstamp : ", time.time() - before)
            # if self.image_processor is not None:
            #     pImageConv = self.image_processor.Convert(
            #         pImageRaw, ps.PixelFormat_BayerRG8
            #     )
            # else:
            #     pImageConv = pImageRaw
            # print("conveted : ", time.time() - before)
            
            # retImage = pImageConv
            self.timestamps["timestamps"].append(ts)
            self.timestamps["frameID"].append(framenum)
            self.timestamps["pc_time"].append(capture_time)
            # retcode=True
            
            # if self.is_recording:
            #     try:
            #         self.videoStream.Append(retImage)
            #     except Exception as e:
            #         print(e)
            
        else:
            print(ps.Image_GetImageStatusDescription(pImageRaw.GetImageStatus()))
            
            

        pImageRaw.Release()



        # return retImage, retcode
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
            self.is_recording = False
            stampname = (
                self.videoName + ".json"
            )
            json.dump(
                self.timestamps, open(self.saveVideoPath + "/" + stampname, "w"), indent="\t"
            )
            # self.videoStream.Close()
            print("Video Save finished")
        else:
            self.is_recording = True
            # self.videoStream.SetMaximumFileSize(0)  # no limited size for the file
            self.videoName = self.serialnum + "_"+self.get_now()
            # savePath = self.saveVideoPath +"/" + self.videoName
            # self.videoStream.Open(str(savePath), self.videoOption)
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
