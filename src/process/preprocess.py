import json
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import shutil

home_path = os.path.expanduser("~")
shared_path = os.path.join(home_path, "shared_data")
sensor_dict = {
    "contact": ["data"],
    "arm": ["state", "action"],
    "hand": ["state", "action"]
    }

def fill_framedrop(cam_timestamp):
    frameID = cam_timestamp["frameID"]
    pc_time = np.array(cam_timestamp["pc_time"])
    timestamp = np.array(cam_timestamp["timestamps"])

    time_delta = (1/30)# (pc_time[-1]-pc_time[0])/(frameID[-1]-frameID[0])
    offset = np.mean(pc_time - (np.array(frameID)-1)*time_delta)

    pc_time_nodrop = []
    frameID_nodrop = []

    for i in range(1, frameID[-1]+1):
        frameID_nodrop.append(i)
        pc_time_nodrop.append((i-1)*time_delta+offset)

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
    parser.add_argument('--name', nargs="+", type=str, default=None)
    args = parser.parse_args()

    if args.name == None:
        name_list = os.listdir(os.path.join(shared_path, 'capture'))
        name_list.sort()

    else:
        name_list = args.name
    for name in name_list:
        
        root_path = os.path.join(shared_path, 'capture', name)
        index_list = list(map(int, os.listdir(root_path)))
        index_list.sort()

        index_offset = 0
        for index in index_list:
            c2r_dir = os.path.join(shared_path, "capture", name, str(index), "C2R.npy")
            cam_param_dir = os.path.join(shared_path, "capture", name, str(index), "cam_param")

            if not os.path.exists(c2r_dir):
                print(f"{c2r_dir} not found")
                continue
            if not os.path.exists(cam_param_dir):
                print(f"{cam_param_dir} not found")
                continue
            
            for target_index in range(int(index)*2, int(index)*2+2):
                target_dir = f"/home/temp_id/shared_data/processed/{name}/{target_index}"
                if not os.path.exists(target_dir):
                    continue
                if not os.path.exists(f"{target_dir}/C2R.npy"):
                    # os.remove(f"{target_dir}/C2R.npy")
                    shutil.copy(c2r_dir, f"{target_dir}/C2R.npy")

                if not os.path.exists(f"{target_dir}/cam_param"):
                    # shutil.rmtree(f"{target_dir}/cam_param")
                    shutil.copytree(cam_param_dir, f"{target_dir}/cam_param")

            capture_path = os.path.join(root_path, str(index))
            if not os.path.exists(os.path.join(capture_path, "camera_timestamp.json")):
                print(f"camera_timestamp.json not found in {capture_path}")
                continue

            camera_timestamp = json.load(open(os.path.join(capture_path, "camera_timestamp.json"), 'r'))

            pc_time, frameID = fill_framedrop(camera_timestamp)
            sensor_timestamp = {}
            sensor_value = {}
            
            sensor_exist = True
            for sensor_name in list(sensor_dict.keys()):
                sensor_path = os.path.join(capture_path, sensor_name)
                if not os.path.exists(sensor_path) or not os.path.exists(os.path.join(sensor_path, "timestamp.npy")):
                    print(f"{sensor_name} data not found in {capture_path}")
                    sensor_exist = False
                    continue

                sensor_timestamp[sensor_name] = np.load(os.path.join(capture_path, sensor_name, "timestamp.npy"))
                sensor_value[sensor_name] = {}

                for data_name in sensor_dict[sensor_name]:
                    sensor_value[sensor_name][data_name] = np.load(os.path.join(capture_path, sensor_name, data_name+".npy"))
            
            if not sensor_exist:
                continue
            
            if not os.path.exists(os.path.join(capture_path, "activate_range.json")):
                print(f"activate_range.json not found in {capture_path}")
                continue

            active_range = json.load(open(os.path.join(capture_path, "activate_range.json"), 'r')) 
            selected_frame = get_selected_frame(pc_time, frameID, active_range)

            json.dump(selected_frame, open(os.path.join(capture_path, "selected_frame.json"), 'w'))

            save_path = os.path.join(shared_path,"processed",name)
            os.makedirs(save_path, exist_ok=True)

            sensor_idx_offset = {sensor_name: 0 for sensor_name in list(sensor_dict.keys())}
            selected_pc_time = {"cam":[],"contact":[],"arm":[],"hand":[]}

            for idx, range_list in selected_frame.items():
                tot_idx = int(idx)+index_offset
                os.makedirs(os.path.join(save_path, str(tot_idx)), exist_ok=True)

                done = True
                for sensor_name in list(sensor_dict.keys()):
                    os.makedirs(os.path.join(save_path, str(tot_idx), sensor_name), exist_ok=True)
                    for data_name in sensor_dict[sensor_name]:
                        if not os.path.exists(os.path.join(save_path, str(tot_idx), sensor_name, data_name+".npy")):
                            done = False
                            break
                # if done:
                #     continue

                        
                selected_sensor_value = {}
                for sensor_name in list(sensor_dict.keys()):
                    selected_sensor_value[sensor_name] = {}
                    for data_name in sensor_dict[sensor_name]:
                        selected_sensor_value[sensor_name][data_name] = []
                
                # import pdb; pdb.set_trace()
                diff_tmp = []
                for (start, end) in range_list:
                    start_idx = start-1
                    end_idx = end-1

                    for i in range(start_idx, end_idx+1):
                        t = pc_time[i]
                        selected_pc_time["cam"].append(t)
                        for sensor_name in list(sensor_dict.keys()):
                            ts = sensor_idx_offset[sensor_name]
                            while ts < len(sensor_timestamp[sensor_name])-1 and abs(sensor_timestamp[sensor_name][ts] - t) > abs(sensor_timestamp[sensor_name][ts+1] - t):
                                ts += 1
                            sensor_idx_offset[sensor_name] = ts
                            selected_pc_time[sensor_name].append(sensor_timestamp[sensor_name][ts])
                            if sensor_name == "arm":
                                diff_tmp.append(np.abs(sensor_timestamp[sensor_name][ts] - t))

                            for data_name in sensor_dict[sensor_name]:
                                selected_sensor_value[sensor_name][data_name].append(sensor_value[sensor_name][data_name][ts])
                print(np.max(diff_tmp))
                for sensor_name in list(sensor_dict.keys()):
                    os.makedirs(os.path.join(save_path, str(tot_idx), sensor_name), exist_ok=True)
                    for data_name in sensor_dict[sensor_name]:
                        np.save(os.path.join(save_path, str(tot_idx), sensor_name, data_name+".npy"), selected_sensor_value[sensor_name][data_name])

                # min_start = min(selected_pc_time['arm'][0], selected_pc_time['hand'][0], selected_pc_time['contact'][0], selected_pc_time['cam'][0])
                # plt.plot(selected_pc_time["cam"]-min_start, 'ro', linestyle='', markersize=1)
                # plt.plot(selected_pc_time["contact"]-min_start, 'go', linestyle='', markersize=1)
                # plt.plot(selected_pc_time["arm"]-min_start, 'bo', linestyle='', markersize=1)
                # plt.plot(selected_pc_time["hand"]-min_start, 'yo', linestyle='', markersize=1)
                # plt.savefig(os.path.join(save_path, str(tot_idx), "time.png"))
                # plt.close()

            index_offset += len(active_range.keys())
    