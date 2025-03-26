import json
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt

home_path = os.path.expanduser("~")
shared_path = os.path.join(home_path, "shared_data")
sensor_dict = {
    "contact": ["data"],
    "arm": ["state", "action"],
    "hand": ["state", "action"]
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

    offset = 0
    for i in range(len(frameID)):
        for j in range(frameID[i]-i-offset):
            pc_time_nodrop.append(pc_time[i]-time_delta*(frameID[i]-i-1-j))
            frameID_nodrop.append(i+j+1)    
            offset = frameID[i] - i - 1
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
    parser.add_argument('--name', type=str, default=None)
    args = parser.parse_args()


    if args.name == None:
        name_list = os.listdir(os.path.join(shared_path, 'capture'))
        name_list.sort()
    else:
        name_list = [args.name]
    
    for name in name_list:
        root_path = os.path.join(shared_path, 'capture', name)
        index_list = list(map(int, os.listdir(root_path)))
        index_list.sort()

        index_offset = 0
        for index in index_list:
            capture_path = os.path.join(root_path, str(index))
            camera_timestamp = json.load(open(os.path.join(capture_path, "camera_timestamp.json"), 'r'))

            # plt.plot(camera_timestamp["timestamps"] - np.min(camera_timestamp["timestamps"]))
            # plt.plot(camera_timestamp["frameID"])
            pc_time, frameID = fill_framedrop(camera_timestamp)
            # plt.plot(frameID, pc_time)
            # plt.plot(camera_timestamp["frameID"], camera_timestamp["pc_time"])
            # plt.plot(pc_time)
            # plt.plot(camera_timestamp["pc_time"])
            sensor_timestamp = {}
            sensor_value = {}
            
            for sensor_name in list(sensor_dict.keys()):
                sensor_timestamp[sensor_name] = np.load(os.path.join(capture_path, sensor_name, "timestamp.npy"))
                sensor_value[sensor_name] = {}

                for data_name in sensor_dict[sensor_name]:
                    sensor_value[sensor_name][data_name] = np.load(os.path.join(capture_path, sensor_name, data_name+".npy"))
                    
            # plt.plot(camera_timestamp["pc_time"] - np.min(camera_timestamp["pc_time"]), linestyle=None)
            # plt.plot(pc_time - np.min(pc_time), linestyle=None)
            # for sensor_name in list(sensor_dict.keys()):
            #    plt.plot(sensor_timestamp[sensor_name] - np.min(sensor_timestamp[sensor_name]),linestyle=None)
            # plt.show()

            # active_range = get_pc_time_range(capture_path)
            
            # selected_frame = get_selected_frame(pc_time, frameID, active_range)

            # json.dump(selected_frame, open(os.path.join(capture_path, "selected_frame.json"), 'w'))

            save_path = os.path.join(shared_path,"processed_full",name)
            os.makedirs(save_path, exist_ok=True)

            sensor_idx_offset = {sensor_name: 0 for sensor_name in list(sensor_dict.keys())}
            
            
            os.makedirs(os.path.join(save_path, str(index)), exist_ok=True)
            selected_sensor_value = {}
            for sensor_name in ["contact", "arm", "hand"]:
                selected_sensor_value[sensor_name] = {}
                for data_name in sensor_dict[sensor_name]:
                    selected_sensor_value[sensor_name][data_name] = []
            
            start = 1
            end = len(pc_time)

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
                os.makedirs(os.path.join(save_path, str(index), sensor_name), exist_ok=True)
                for data_name in sensor_dict[sensor_name]:
                    np.save(os.path.join(save_path, str(index), sensor_name, data_name+".npy"), selected_sensor_value[sensor_name][data_name])



    #     at = 0
    #     selected_frame = []
    #     selected_frame_tmp = []

    #     for i in range(len(pc_time)):
    #         while at < len(arm_timestamp)-1 and abs(pc_time[i] - arm_timestamp[at][0]) > abs(pc_time[i] - arm_timestamp[at+1][0]):
    #             at += 1
    #         diff = abs(pc_time[i] - arm_timestamp[at][0])
    #         if diff < 1 / 100:
    #             selected_frame.append(i+1)

    #     json.dump(selected_frame, open(os.path.join(capture_path, "selected_frame_"+dt+".json"), 'w'))

    # arm_state = np.load(os.path.join(capture_path, "robot", "arm_state_hist_"+dt+".npy"))
    # arm_action = np.load(os.path.join(capture_path, "robot", "arm_action_hist_"+dt+".npy"))
    # hand_state = np.load(os.path.join(capture_path, "robot", "hand_state_hist_"+dt+".npy"))
    # hand_action = np.load(os.path.join(capture_path, "robot", "hand_action_hist_"+dt+".npy"))
    # contact_value = np.load(os.path.join(capture_path, "contact", "data"+"_"+dt+".npy"))

    # arm_state_select = []
    # arm_action_select = []
    # hand_state_select = []
    # hand_action_select = []
    # contact_value_select = []

    # at = 0
    # ct = 0
    # ht = 0
    # for f in selected_frame:
    #     pc_t = pc_time[f-1]
    #     while at < len(arm_timestamp)-1 and abs(pc_t - arm_timestamp[at][0]) > abs(pc_t - arm_timestamp[at+1][0]):
    #         at += 1
    #     arm_action_select.append(arm_action[at])
    #     arm_state_select.append(arm_state[at])
        
    #     while ct < len(contact_timestamp)-1 and abs(pc_t - contact_timestamp[ct]) > abs(pc_t - contact_timestamp[ct+1]):
    #         ct += 1
    #     contact_value_select.append(contact_value[ct])
        
    #     while ht < len(hand_timestamp)-1 and abs(pc_t - hand_timestamp[ht]) > abs(pc_t - hand_timestamp[ht+1]):
    #         ht += 1
    #     hand_action_select.append(hand_action[ht])
    #     hand_state_select.append(hand_state[ht])

    # np.save(os.path.join(capture_path, "robot", "arm_state_select_"+dt+".npy"), arm_state_select)
    # np.save(os.path.join(capture_path, "robot", "arm_action_select_"+dt+".npy"), arm_action_select)
    # np.save(os.path.join(capture_path, "robot", "hand_state_select_"+dt+".npy"), hand_state_select)
    # np.save(os.path.join(capture_path, "robot", "hand_action_select_"+dt+".npy"), hand_action_select)
    # np.save(os.path.join(capture_path, "contact", "data_select"+"_"+dt+".npy"), contact_value_select)