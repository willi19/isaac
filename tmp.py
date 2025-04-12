import cv2
import os
from dex_robot.visualization.convert_codec import change_to_h264
import tqdm
import numpy as np
import random

dir = "/home/temp_id/download/processed/void/1/hand"
hand_action = np.load(os.path.join(dir, "action.npy"))
print(hand_action.shape)
idx = 0
finger_name = ["index", "middle", "ring", "thumb"]
joint_limit = []

joint_limit.append([-0.57, 0.57])
joint_limit.append([-0.296, 1.71])

for i in range(10):
    # hand_action_round = hand_action[i]
    # hand_action_round = np.round(hand_action_round, 3)
    hand_action = np.zeros(16)# np.load("data/calibration_pose/hand_{}.npy".format(i))

    for j in range(3):
        for k in range(2):
            mid = (joint_limit[k][0] + joint_limit[k][1]) / 2
            if i % 2 == 0:
                hand_action[4*j+k] = random.uniform(joint_limit[k][0], mid)
            else:
                hand_action[4*j+k] = random.uniform(mid, joint_limit[k][1])
    np.save(os.path.join("data/calibration_pose", f"hand_{i}.npy"), hand_action)

    print(hand_action)
    print("==========================")