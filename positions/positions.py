import math

import numpy as np
import scipy

from analysis import analyzer


def read_positions(position_matrix_file, start=0, end=None):
    """
    :param position_matrix_file: The path to the position matrix
    :param start: The start index
    :param end: The end index
    """
    mat = scipy.io.loadmat(position_matrix_file)

    if end is None:
        data = mat['HSpos'][start:, 0:3]  # 0 is time, 1 is x, 2 is y
    else:
        data = mat['HSpos'][start:end, 0:3]  # 0 is time, 1 is x, 2 is y
    data = data[:, 0:3]
    return data


def correct_data(data, precision=0):
    data = data[:]
    prior = []
    for i in range(len(data)):
        datum = data[i]
        for n in range(1, 3):
            if math.isnan(datum[n]):
                data[i][n] = prior[n - 1]  # Sets the value to prior value if nan
                #  TODO: This is likely causing confounds in the data.
                #  Don't take into account value pairs in MI where one value is NaN
                # Look up commands that can handle missing values
                # Make this a separate function
        if i > 0:
            if abs(datum[0] - data[i - 1, 0]) > 1 * 10 ** 9:  # Detects jumps in time
                data[i, 0] = data[i, 0] / 100.0  # Convert to Neuralynx units (0.1ms)
                #   print(f"Aberrant datum at {i} converted to Neuralynx units")
        data[i, 0] = round(data[i, 0] / 10 ** 4,
                           precision)  # Converts time to seconds, and rounds to precision
        if (datum[1] == 0) and (datum[2] == 0):
            print(i, datum)
        prior = [datum[1], datum[2]]
    return data


def calculate_distances(data):
    distances = np.zeros(len(data), dtype=object)
    for i in range(1, len(data)):
        datum = data[i]
        distances[i] = math.sqrt((datum[1] - data[i - 1, 1]) ** 2 +
                                 (datum[2] - data[i - 1, 2]) ** 2)
    return np.column_stack((data[:, 0], distances))


def smooth_values(data):
    """
    Smooths the data
    :return: The smoothed data
    """
    fps = 30
    # TODO: Why is this 0.3
    data_smoothed = analyzer.apply_kernel(data[:, 1], fps, 0.3, kernel_type='gaussian'), \
                    analyzer.apply_kernel(data[:, 2], fps, 0.3, kernel_type='gaussian')
    return np.column_stack((data[:, 0], data_smoothed[0], data_smoothed[1]))


def calculate_angles(data):
    """
    Calculates angles as arc tangents, the angle between two sequential vectors in the path
    Essentially it provides absolute angles for each line segment on the path
    :param data: The data, organized in an n x 3 matrix, where n is the number of points
    :return: The angles
    """

    print(data)

    x_values, y_values = data[:, 1], data[:, 2]

    angles = np.zeros(len(x_values), dtype=object)
    for i in range(len(x_values)):
        if i < len(x_values) - 2:
            angles[i + 1] = math.atan2(y_values[i + 1] - y_values[i],
                                       x_values[i + 1] - x_values[i])
    print(data[:, 0])
    angles = np.column_stack((data[:, 0], angles))
    return angles


def calculate_angular_velocity(angles):
    """
    Calculates the angular velocity timeseries of the path, the rate of change of the arc tangents
    Essentially it provides the series of relative angles to each line segment on the path
    :return: The angular velocity
    """
    angle_changes = np.zeros(len(angles), dtype=object)

    for i, angle in enumerate(angles[:, 1]):
        if i < len(angles) - 2:
            angle_changes[i + 1] = (angles[i + 1, 1] - angle + np.pi) % (
                    2 * np.pi) - np.pi

    return np.column_stack((angles[:, 0], angle_changes))


def threshold_angular_velocity(angular_velocity, distances, threshold):
    thresholded_angular_velocity = np.where(distances[:, 1] > threshold, angular_velocity[:, 1], 0)
    thresholded_angular_velocity[0] = 0
    thresholded_angular_velocity[1] = 0
    return np.column_stack((angular_velocity[:, 0], thresholded_angular_velocity))
