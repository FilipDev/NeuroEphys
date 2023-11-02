import numpy as np
import math
import scipy.signal as signal

from numpy import ndarray

from analysis import kernels


def decompress_timestamp_data(timestamps: np.ndarray, precision: int):
    """
    Decompresses timestamp data into a timeseries binary array, while aggregating based on precision if necessary
    :param timestamps: An array of the timestamps
    :param precision: The number of significant digits to include
    :return: The decompressed values as a binary array

    >>> decompress_timestamp_data([4.402, 5.591, 7.063, 9.18, 12.024, 16.364, 21.146, 23.243, 24.675, 29.257], 0)
    array([0., 0., 0., 0., 1., 0., 1., 1., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1.])
    """
    decompressed_range = int(round(timestamps[-1], precision) * 10 ** precision) + 1

    binary_data = np.zeros(decompressed_range)

    for timestamp in timestamps:
        timestamp = int(round(timestamp, precision) * 10 ** precision)
        binary_data[timestamp] = binary_data[timestamp] + 1

    return binary_data


def apply_kernel(binary_data: np.ndarray, steps_per_second: float, window_size: float, step_size: float,
                 fft=True, **kwargs):
    """
    Applies a kernel on the inputted binary data, used for density calculations
    :param binary_data: The binary input data to apply the kernel on
    :param window_size: The size of the window in seconds, determines what is included in the kernel
    :param step_size: The step size in seconds for indexing the input data
    :param steps_per_second: The conversion rate from window_size and step_size units (which are in seconds) to index differences for indexing the input data
    
    >>> apply_kernel([0.943, 1.826, 2.56], 10, 1, 0.1, kernel_type='simple')
    [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 1.]
    """

    kernel_type = kwargs.get("kernel_type", "simple")
    kernel = kernels.get_kernel(kernel_type)

    # Makes the window size proportional to the precision of the data in significant digits
    window_size_in_seconds = window_size
    window_size *= steps_per_second

    # Makes the step size proportional to the precision of the data in significant digits
    step_size *= steps_per_second

    convolved = signal.fftconvolve(binary_data, [kernel(window_size/2, i, window_size=window_size, **kwargs) for i in range(int(window_size))], mode='same')

    #convolved = np.convolve(binary_data, [kernel(window_size/2, i, window_size=window_size, **kwargs) for i in range(int(window_size))], mode='same')

    print("Convolved")

    return convolved

def stamps_histogram(timestamps: np.ndarray, order=1):
    """
    Computes the histogram of the timestamps
    :param timestamps: An array of timestamps
    :param order: How many skips to make between each timestamp
    :return: The histogram of the timestamps
    """
    relative_times = convert_timestamps_to_relative_times_nth_order(timestamps, order)
    return np.histogram(relative_times, bins=200)


def convert_timestamps_to_relative_times_nth_order(timestamps: np.ndarray, order):
    """
    Converts timestamps to relative times

    :param order: How many skips to make between each timestamp
    :param timestamps: An array of timestamps
    :return: An array of relative times
    """
    relative_times = np.zeros(len(timestamps))
    for i in range(len(timestamps)):
        relative_times[i] = (timestamps[i] - timestamps[(i - order) if i > order - 1 else 0]) / order
    return relative_times


def normalize(timeseries: np.ndarray):
    """
    Sets the minimum value of a timeseries to 0
    :param timeseries: The timeseries
    :return: The normalized timeseries
    """
    return timeseries - timeseries.min()


def digitize_timeseries(timeseries: np.ndarray, n_bins: int):
    """
    Digitizes a timeseries into a given number of bins

    :param timeseries: The timeseries
    :param n_bins: The number of bins
    :return: The digitized timeseries
    """
    bins = np.linspace(np.floor(np.min(timeseries)), np.ceil(np.max(timeseries)), num=n_bins+3)
    # TODO: For some reason, the n_bins is really weird
    v = np.digitize(timeseries, bins)
    v = v - np.median(np.unique(v))
    v = (np.abs(v) - 1) * np.sign(v)
    return v

def find_average_value_of_x_when_y(X: np.ndarray, Y: np.ndarray, x, y) -> float:
    """
    Finds the average value of x in X when y is present in Y
    :param X: The first stream
    :param Y: The second stream
    :param x: The value in the first stream
    :param y: The value in the second stream
    :return: The average value of x in X when y is present in Y
    """
    if len(X) == 0 or len(Y) == 0:
        return 0
    if x is None or y is None:
        return 0

    total_value_of_xs = 0
    for i in range(len(X)):
        if X[i] == x and Y[i] == y:
           total_value_of_xs += X[i]

    average_value_of_x_when_y = total_value_of_xs / len(X)

    return average_value_of_x_when_y

def value_mutual_information(X: np.ndarray, Y: np.ndarray, x, y) -> float:
    """
    Computes the mutual information between the values x and y in respective streams X and Y

    :param X: The first stream
    :param Y: The second stream
    :param x: The value in the first stream
    :param y: The value in the second stream
    :return: The mutual information between x and y
    """
    if len(X) == 0 or len(Y) == 0:
        return 0
    if x is None or y is None:
        return 0

    prob_x = find_probability(X, x)
    prob_y = find_probability(Y, y)
    prob_xy = find_mutual_probability(X, Y, x, y)
    if prob_x == 0 or prob_y == 0 or prob_xy == 0:
        return 0

    return prob_xy * math.log(prob_xy / (prob_x * prob_y), math.e)


def stream_mutual_information(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Computes the mutual information between two streams.

    X and Y must be the same length
    The streams MUST be digitized.

    :param X: The first stream
    :param Y: The second stream
    :return: The mutual information
    """

    if len(X) != len(Y):
        raise ValueError("X and Y must have the same length, X: {}, Y: {}".format(len(X), len(Y)))

    sum = 0
    for x in np.unique(X):
        for y in np.unique(Y):
            sum += value_mutual_information(X, Y, x, y)

    return sum


def all_values_mutual_information(X: np.ndarray, Y: np.ndarray) -> dict:
    """
    Computes the mutual information for all values between two streams.
    Returns a 2D dictionary of mutual information values

    X and Y must be the same length
    The streams MUST be digitized.

    :param X: The first stream
    :param Y: The second stream
    :return: The mutual information of all values
    """

    if len(X) != len(Y):
        raise ValueError("X and Y must have the same length, X: {}, Y: {}".format(len(X), len(Y)))

    unique_x, unique_y = np.unique(X), np.unique(Y)
    value_mutual_informations = {}

    for x in unique_x:
        x = int(x)
        value_mutual_informations[x] = {}
        for y in unique_y:
            value_mutual_informations[x][y] = value_mutual_information(X, Y, x, y)

    return value_mutual_informations


def find_probability(X: np.ndarray, x) -> float:
    """
    Computes the probability of a specific value occurring in a stream
    :param X: The stream
    :param x: The value
    :return: The probability
    """

    if len(X) == 0:
        return 0
    if x is None:
        return 0

    return np.sum(X == x) / len(X)


def find_mutual_probability(X: np.ndarray, Y: np.ndarray, x, y) -> float:
    """
    Computes the mutual probability of two specific values occurring simultaneously in two respective concurrent streams

    :param X: The first stream
    :param Y: The second stream
    :param x: The value in the first stream
    :param y: The value in the second stream
    :return: The mutual probability of x and y in X and Y
    """

    if len(X) != len(Y):
        raise ValueError("X and Y must have the same length, X: {}, Y: {}".format(len(X), len(Y)))
    if len(X) == 0 or len(Y) == 0:
        return 0
    if x is None or y is None:
        return 0

    mutual_instances = np.sum((X == x) & (Y == y))

    return mutual_instances / len(X)
