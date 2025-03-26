import json
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt

home_path = os.path.expanduser("~")
shared_path = os.path.join(home_path, "shared_data")
sensor_dict = {
    "contact": ["data"]
    }

def get_pc_time_range(path):
    range_list = json.load(open(os.path.join(path, "activate_range.json"), 'r'))
    return range_list

def fill_framedrop(cam_timestamp):
    frameID = cam_timestamp["frameID"]
    pc_time = np.array(cam_timestamp["pc_time"])
    timestamp = np.array(cam_timestamp["timestamps"])

    time_delta = (pc_time[-1]-pc_time[0])/(frameID[-1]-frameID[0])
    # print(time_delta, (pc_time[-1]-pc_time[0])/(frameID[-1]-frameID[0]))
    pc_time_nodrop = []
    frameID_nodrop = []

    for i in range(len(frameID)):
        for j in range(frameID[i]-i):
            pc_time_nodrop.append(pc_time[i]-time_delta*(frameID[i]-i-1-j))
            frameID_nodrop.append(i+j+1)    
    return pc_time_nodrop, frameID_nodrop

def get_selected_frame(pc_time, frameID, active_range):
    selected_frame = {}
    pc_time_idx = 0

    for index, range_list in active_range.items():
        selected_frame[index] = []
        for (start, end) in range_list:
            while pc_time_idx < len(pc_time) and pc_time[pc_time_idx] < start:
                pc_time_idx += 1
            start_time_idx = pc_time_idx

            while pc_time_idx < len(pc_time) and pc_time[pc_time_idx] < end:
                pc_time_idx += 1
            end_time_idx = pc_time_idx-1
            if start_time_idx > end_time_idx:
                continue
            selected_frame[index].append((frameID[start_time_idx], frameID[end_time_idx]))
    return selected_frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    name = "contact"
    root_path = os.path.join(shared_path, 'capture', name)
    index_list = list(map(int, os.listdir(root_path)))
    index_list.sort()

    index_offset = 0
    for index in index_list:
        capture_path = os.path.join(root_path, str(index))
        camera_timestamp = json.load(open(os.path.join(capture_path, "camera_timestamp.json"), 'r'))

        pc_time, frameID = fill_framedrop(camera_timestamp)
        sensor_timestamp = {}
        sensor_value = {}
        
        for sensor_name in list(sensor_dict.keys()):
            sensor_timestamp[sensor_name] = np.load(os.path.join(capture_path, sensor_name, "timestamp.npy"))
            sensor_value[sensor_name] = {}
            for data_name in sensor_dict[sensor_name]:
                sensor_value[sensor_name][data_name] = np.load(os.path.join(capture_path, sensor_name, data_name+".npy"))
                
        print(len(pc_time), len(sensor_value["contact"]["data"]))
        
        selected_frame = json.load(open(os.path.join(capture_path, "selected_frame.json"), 'r'))

        save_path = os.path.join(shared_path,"processed",name)
        os.makedirs(save_path, exist_ok=True)

        sensor_idx_offset = {sensor_name: 0 for sensor_name in list(sensor_dict.keys())}
        
        for idx, range_list in selected_frame.items():
            tot_idx = int(idx)+index_offset
            os.makedirs(os.path.join(save_path, str(tot_idx)), exist_ok=True)
            selected_sensor_value = {}
            for sensor_name in ["contact"]:
                selected_sensor_value[sensor_name] = {}
                for data_name in sensor_dict[sensor_name]:
                    selected_sensor_value[sensor_name][data_name] = []

            for (start, end) in range_list:
                start_idx = start-1
                end_idx = end-1

                for i in range(start_idx, end_idx+1):
                    t = pc_time[i]
                    for sensor_name in ["contact"]:
                        ts = sensor_idx_offset[sensor_name]
                        while ts < len(sensor_timestamp[sensor_name])-1 and abs(sensor_timestamp[sensor_name][ts] - t) > abs(sensor_timestamp[sensor_name][ts+1] - t):
                            ts += 1
                        sensor_idx_offset[sensor_name] = ts
                        for data_name in sensor_dict[sensor_name]:
                            selected_sensor_value[sensor_name][data_name].append(sensor_value[sensor_name][data_name][ts])

            for sensor_name in ["contact"]:
                os.makedirs(os.path.join(save_path, str(tot_idx), sensor_name), exist_ok=True)
                for data_name in sensor_dict[sensor_name]:
                    np.save(os.path.join(save_path, str(tot_idx), sensor_name, data_name+".npy"), selected_sensor_value[sensor_name][data_name])

            

        index_offset += len(selected_frame.keys())
