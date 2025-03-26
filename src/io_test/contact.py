import datetime
import time
from dex_robot.contact.receiver import SerialReader
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    reader = SerialReader(save_path="test_capture/mingi")

    start_time = time.time()
    while time.time() - start_time < 10:
        print(time.time() - start_time)
        time.sleep(0.01)  # Keep main process alive

    
    print("\n[INFO] Exiting...")
    reader.quit()  # Stop the serial reader process safely

    value = np.load("test_capture/mingi/contact/data.npy")
    timestamp = np.load("test_capture/mingi/contact/timestamp.npy")

    print(value.shape)
    print(timestamp.shape)
    plt.plot(timestamp, value - value[0])
    plt.legend(["sensor" + str(i) for i in range(15)])
    plt.show()