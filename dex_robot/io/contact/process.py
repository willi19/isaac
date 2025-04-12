import numpy as np


def moving_average(data, window_size=5):
    """간단한 이동 평균 필터"""
    cumsum = np.cumsum(np.insert(data, 0, 0, axis=0), axis=0)
    smoothed = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    padding = np.tile(data[0], (window_size - 1, 1))
    return np.vstack((padding, smoothed))

def process_contact(input_data, trim=100, window_size=5):
    # log_data = np.log(input_data)
    # input_data = (log_data - log_data[0]) * 50
    input_data = input_data - input_data[0]
    input_data = moving_average(input_data, window_size=window_size)
    return input_data[:, :15]
    # return np.clip(input_data[:, :15], 0, trim)