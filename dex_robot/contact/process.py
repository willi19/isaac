import numpy as np

def process_contact(input_data, trim=500):
    input_data = input_data - input_data[0]
    return np.clip(input_data[:,:15], 0, trim)