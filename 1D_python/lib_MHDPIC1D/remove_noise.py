import numpy as np


def smoothing_U_left(U, index_interface_mhd_start, index_interface_mhd_end, window_size = 3):

    smoothed_U = U.copy().astype(np.float64)

    for i in range(8):
        tmp_U = np.convolve(U[i, :], np.ones(window_size) / window_size, mode="same")

        index_start = index_interface_mhd_start
        index_end = index_interface_mhd_end - window_size//2 + 1
        smoothed_U[i, index_start : index_end] = tmp_U[index_start : index_end]
        #smoothed_U[i, index_start : index_start + window_size//2] = smoothed_U[i, index_start + window_size//2]
        smoothed_U[i, index_end - window_size//2 : index_end] = smoothed_U[i, index_end - window_size//2]

    return smoothed_U


def smoothing_U_right(U, index_interface_mhd_start, index_interface_mhd_end, window_size = 3):

    smoothed_U = U.copy().astype(np.float64)

    for i in range(8):
        tmp_U = np.convolve(U[i, :], np.ones(window_size) / window_size, mode="same")

        index_start = index_interface_mhd_start
        index_end = index_interface_mhd_end - window_size//2 + 1
        smoothed_U[i, index_start : index_end] = tmp_U[index_start : index_end]
        smoothed_U[i, index_start : index_start + window_size//2] = smoothed_U[i, index_start + window_size//2]
        #smoothed_U[i, index_end - window_size//2 : index_end] = smoothed_U[i, index_end - window_size//2]

    return smoothed_U

