import subprocess

command = [
    "roslaunch",
    "allegro_hand",
    "allegro_hand.launch",
    "VISUALIZE:=true",
    "AUTO_CAN:=false",
    "CAN_DEVICE:=/dev/pcan32"
]

# Run the command
process = subprocess.Popen(command)
