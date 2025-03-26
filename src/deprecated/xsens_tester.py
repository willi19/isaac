import sys
import time
import argparse
# from IPython import embed
import numpy as np
import datetime
import time
import pickle
from copy import deepcopy
import socket
import time
import struct
import argparse

# from render_args import add_render_args
# import matplotlib.pyplot as plt
# import seaborn as sns
import threading
import numpy as np
import time
import re
import matplotlib.pyplot as plt


host_ip = "192.168.0.2"
port = 9763

server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((host_ip, port))

header_format = "!6s I B B I B B B B 2s H"
header_struct = struct.Struct(header_format)

cnt = 0
xsens_time = -1
xsens_time_tot = 0
pc_time_tot = 0
start_time = -1

pc_list = [
]
xsens_list = []
for i in range(500):
    
    t1 = time.monotonic()
    data, addr = server_socket.recvfrom(4096)
    t2 = time.monotonic()

    wait_time = (time.monotonic()-t1 )*1000
    # print((time.time()-t1)*1000, "wait time")
    recv_time = time.monotonic()  #

    data_header = header_struct.unpack(data[:24])  # header is 24 bytes
    parsed_data = {}

    # Use this if network is unstable or multiple protocol is used for this port

    header_id_string = data_header[0].decode("utf-8")
    assert header_id_string[:4] == "MXTP"
    message_id = int(header_id_string[4:])
    # self.buffer.put((recv_time, message_id, data_header, data[24:]))
    # print(self.cnt / (time.time()-start_time))
    if message_id == 25:
        
        
        # print(data[28:].decode("utf-8"), re.split(r'[:.]', data[28:].decode("utf-8")))
        _new_xsens_time = list(map(int, re.split(r'[:.]', data[28:].decode("utf-8"))))
        new_xsens_time = _new_xsens_time[0]
        new_xsens_time = new_xsens_time * 60 + _new_xsens_time[1]
        new_xsens_time = new_xsens_time * 60 + _new_xsens_time[2]
        new_xsens_time = new_xsens_time * 1000 + _new_xsens_time[3]
        
        if xsens_time != -1:
            pc_time_tot += (t2 - start_time)*1000
            xsens_time_tot += new_xsens_time - xsens_time
            print(xsens_time_tot - pc_time_tot, (new_xsens_time - xsens_time) // 16, (new_xsens_time - xsens_time))
        pc_list.append(pc_time_tot)
        xsens_list.append(xsens_time_tot)
        xsens_time = new_xsens_time
        start_time = t2
    cnt += 1

plt.plot(xsens_list[100:], pc_list[100:],'o')
plt.plot(xsens_list[100:], xsens_list[100:],'x')

plt.ylabel("PC Time (s)")   # X축 라벨
plt.xlabel("Xsens Time (ms)")  # Y축 라벨

# 그래프 제목
plt.title("PC Time vs. Xsens Time")

plt.savefig("./PC_XSENS_COMM.png")