import scipy
import math
import numpy as np
from matplotlib import pyplot as plt

from analysis import analyzer


class PositionMatrix:

    def __init__(self, position_matrix_file, frame_rate=30, precision=0, start=0, end=None):
        """
        :param position_matrix_file: The path to the position matrix
        :param precision: The precision, defined in terms of seconds
        :param start: The start index
        :param end: The end index
        """
        self.frame_rate = frame_rate
        self.precision = precision  # Default precision is 0, converting to seconds
        self.speed_distribution = None
        self.rotational_velocities = None
        self.angular_velocities = None
        self.angles = None
        self.distances = [0]
        self.data_smoothed = None
        self.data = None
        self.position_matrix_file = position_matrix_file
        self._read_positions(start, end)

    def __len__(self):
        """
        :return: The length of the data
        """
        return len(self.data[:, 0])

    def __getitem__(self, index):
        """
        Gets the item at the index
        :param key: The index
        :return: The item
        """
        return self.data[index]

    def __setitem__(self, key, value):
        self.data[key] = value

    def _read_positions(self, start, end):
        """
        Reads the positions from the position matrix
        :param start: The start index
        :param end: The end index
        """
        mat = scipy.io.loadmat(self.position_matrix_file)
        if end is None:
            self.data = mat['HSpos'][start:, 0:3]  # 0 is time, 1 is x, 2 is y
        else:
            self.data = mat['HSpos'][start:end, 0:3]  # 0 is time, 1 is x, 2 is y
        data = self.data
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
                self.distances.append(math.sqrt((datum[1] - data[i - 1, 1]) ** 2 + (datum[2] - data[i - 1, 2]) ** 2))
            data[i, 0] = round(data[i, 0] / 10 ** 4, self.precision)  # Converts time to seconds, and rounds to precision
            if (datum[1] == 0) and (datum[2] == 0):
                print(i, datum)
            prior = [datum[1], datum[2]]
        print("Finished reading positions from file", self.position_matrix_file)

    def compute_speed_distribution(self):
        """
        Calculates the speed distribution of the data
        :return: The speed distribution
        """
        log_distances = np.log(self.distances)
        self.speed_distribution = np.histogram(log_distances, bins=200, range=(0, 20))[0]
        return self.speed_distribution

    def smooth_values(self):
        """
        Smooths the data
        :return: The smoothed data
        """
        fps = 30
        # TODO: Why is this 0.3
        self.data_smoothed = analyzer.apply_kernel(self.data[:, 1], fps, 0.3, 1 / fps), analyzer.apply_kernel(
            self.data[:, 2], fps, 0.3, 1 / fps, kernel_type='gaussian')
        return self.data_smoothed

    def find_angles(self):
        """
        Calculates angles as arc tangents, the angle between two sequential vectors in the path
        Essentially it provides absolute angles for each line segment on the path
        :return: The angles
        """

        # Smooths data if not already smoothed
        if self.data_smoothed is None:
            self.smooth_values()
        x_values, y_values = self.data_smoothed[0], self.data_smoothed[1]

        angles = np.zeros(len(x_values), dtype=object)
        for i in range(len(x_values)):
            if i < len(x_values) - 2:
                angles[i + 1] = math.atan2(y_values[i + 1] - y_values[i],
                                              x_values[i + 1] - x_values[i])

        self.angles = angles
        return self.angles

    def find_angular_velocity(self):
        """
        Calculates the angular velocity timeseries of the path, the rate of change of the arc tangents
        Essentially it provides the series of relative angles to each line segment on the path
        :return: The angular velocity
        """
        if self.angles is None:
            self.find_angles()
        angles = self.angles
        angle_changes = np.zeros(len(angles), dtype=object)

        for i, angle in enumerate(angles):
            if i < len(angles) - 2:
                angle_changes[i + 1] = (angles[i + 1] - angle + np.pi) % (
                        2 * np.pi) - np.pi

        self.angular_velocities = angle_changes
        return self.angular_velocities

    def compute_rotational_velocity(self):
        """
        Calculates the rotational velocity timeseries of the path, the rate of change of the arc tangents
        :return: The rotational velocity
        """
        if self.angular_velocities is None:
            self.find_angular_velocity()

        self.rotational_velocities = np.zeros(len(self.angular_velocities))

        distance_threshold = 10
        max_rotation = 24
        min_rotation = 2
        prior = 0
        for i in range(len(self.angular_velocities)):
            # TODO: Rework
            # If the number of pixels moved is greater than the threshold
            if self.distances[i] > distance_threshold:
                # Why is it multiplied by 100
                self.rotational_velocities[i] = 100 * self.angular_velocities[i]
                # If the rotational velocity is greater than max_rotation make it equal to prior
                # Smooths out sharp abrupt turns, defined as turns that are > max_rotation degree in one frame
                if np.abs(self.rotational_velocities[i]) > max_rotation:
                    self.rotational_velocities[i] = prior
                # If the rotational velocity is less than min_rotation make it equal to 0
                # Smooths out sharp abrupt turns, defined as turns that are < 2 degree in one frame
                if np.abs(self.rotational_velocities[i]) < min_rotation:
                    self.rotational_velocities[i] = 0
                prior = self.rotational_velocities[i]
            else:
                # If distance is less than distance threshold set rotational velocity to prior (smooths out noise)
                self.rotational_velocities[i] = prior
        return self.rotational_velocities

    def plot(self):
        """
        Plots the data
        """
        if self.data_smoothed is None:
            self.smooth_values()
        plt.scatter(self.data_smoothed[0], self.data_smoothed[1], 0.1)
        plt.show()

    def to_csv(self):
        # TODO: Implement
        pass
