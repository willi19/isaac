import os
import argparse

import numpy as np
from dex_robot.utils.file_io import shared_path, load_camparam, load_c2r, download_path, rsc_path
import tqdm
import matplotlib.pyplot as plt
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="void")
    args = parser.parse_args()

    name_list = [args.name] if args.name else os.listdir(os.path.join(shared_path, 'capture'))
    for name in name_list:
        index_list = os.listdir(os.path.join(shared_path, 'capture', name))
        
        for index in index_list:
            active_range = json.load(open(os.path.join(shared_path, 'capture', name, index, 'activate_range.json')))
            root_path = os.path.join(shared_path, 'capture', name, index)
            
            intrinsic_list, extrinsic_list = load_camparam(root_path)
            
            robot_action = np.load(os.path.join(root_path, "arm", "state.npy"))
            robot_timestamp = np.load(os.path.join(root_path, "arm", "timestamp.npy"))

            print(np.mean(1 / (robot_timestamp[1:] - robot_timestamp[:-1])))
            print(np.std(1 / (robot_timestamp[1:] - robot_timestamp[:-1])))
            
            
            hand_action = np.load(os.path.join(root_path, "hand", "state.npy"))

            serial_list = list(intrinsic_list.keys())
            serial_list.sort()

            seq_len = len(robot_action)
            fuck = []
            
            prev_pos = robot_action[0]
            for fid in range(1, seq_len):# tqdm.tqdm(range(seq_len)): 
                if np.linalg.norm(robot_action[fid] - prev_pos) == 0:
                    # print(f"Same frame at {fid} with {robot_action[fid]} and {prev_pos}")
                    rt = robot_timestamp[fid]
                    in_range = False
                    for range_list in active_range.values():
                        for (start, end) in range_list:
                            if start < rt and rt < end:
                                fuck.append(1)
                                in_range = True
                    if not in_range:
                        fuck.append(0.5)
                else:
                    fuck.append(0)
                prev_pos = robot_action[fid]

            os.makedirs(f"timediff/{name}", exist_ok=True)
            plt.plot(robot_timestamp[1:] - robot_timestamp[0],fuck, marker='o', linestyle='None')
            plt.savefig(f"timediff/{name}/{index}.png")
            plt.cla()
            for i in range(seq_len//50-1):
                print(np.mean(fuck[i*50:(i+1)*50]))
            print("---------------------------")