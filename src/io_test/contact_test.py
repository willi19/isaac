import datetime
import time
from dex_robot.io.contact.receiver import SerialReader
from dex_robot.io.contact.process import process_contact, moving_average

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # reader = SerialReader(save_path="test_capture/mingi")

    # start_time = time.time()
    # while time.time() - start_time < 10:
    #     print(time.time() - start_time)
    #     time.sleep(0.01)  # Keep main process alive

    
    # print("\n[INFO] Exiting...")
    # reader.quit()  # Stop the serial reader process safely

    data_path = "/home/temp_id/shared_data/capture/void/0/contact"
    value = np.load(f"{data_path}/data.npy")
    value = value - np.mean(value, axis=0)
    value = moving_average(value, 3)
    value = np.clip(value, -100, 100)
    timestamp = np.load(f"{data_path}/timestamp.npy")
    timestamp = timestamp - timestamp[0]

    # for i in range(15):
    #     plt.plot(timestamp, value[:,i:i+1])# - value[0])
    #     plt.legend(["sensor" + str(j) for j in range(i,i+1)])
    #     plt.show()
    plt.plot(timestamp, value)
    plt.legend(["sensor" + str(i) for i in range(15)])
    plt.show()

    print(np.std(value, axis=0))
    print(np.mean(value, axis=0))
    print(np.min(value, axis=0))
    print(np.max(value, axis=0))