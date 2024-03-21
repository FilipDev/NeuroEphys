import glob
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analysis.analyzer
import analysis.analyzer as analyzer
import positions.positions2 as positions
import spikes.t_reader
import spikes.t_reader as reader
import utils.utils as utils
from scipy.stats import poisson
from statsmodels.stats.proportion import proportion_confint


def calculate_errors(counts, total_counts_per_bin, alpha, precision=0):
    lower_bounds, upper_bounds = [], []
    for count, total_count in zip(counts, total_counts_per_bin):
        ci_low, ci_high = proportion_confint(count / 10 ** precision / 2, total_count / 10 ** precision, alpha=alpha,
                                             method='wilson')
        lower_bounds.append(ci_low * total_count)
        upper_bounds.append(ci_high * total_count)

    errors = np.array(upper_bounds) - np.array(lower_bounds)
    return errors


def read_t_files_d(file_glob):
    t_files = glob.glob(file_glob)
    print(t_files[0])
    read_t_files = reader.read_t_files(t_files)
    return [spikes.reader.decompress_timestamp_data(read_t_files[i][1], precision=t_precision) for i in
            range(len(read_t_files))]


def read_t_files(file_glob):
    t_files = glob.glob(file_glob)
    print(t_files[0])
    read_t_files = reader.read_raw_t_files(t_files)
    return [read_t_files[i][1] for i in
            range(len(read_t_files))]


def read_sync_file(file_path):
    pos_t = {}
    t_pos = {}
    read = False
    for line in open(file_path):
        if line == "<BODY>\n":
            read = True
            continue
        if line == "</BODY>\n":
            break
        if read:
            pos_mat_time = line.split("Start=")[1].split(">")[0]
            t_file_time = line.split("ENUSCC>")[1].split("<")[0]
            pos_t[int(pos_mat_time)] = int(t_file_time)
            t_pos[int(t_file_time)] = int(pos_mat_time)
    return pos_t, t_pos


def get_cell_firing_rates_for_directions(t_file, angular_velocities, angular_velocity_threshold, distance_threshold):
    fire_rates = {"left": [], "right": [], "none": []}

    for timepoint, rotation in enumerate(angular_velocities):
        if distances[timepoint] > distance_threshold:  # If the distance is greater than the threshold
            fire_rate = t_file[timepoint]
            if fire_rate > 1e-4:  # If the fire rate is non-zero, in debug mode with or True
                if not abs(rotation) > angular_velocity_threshold * 4:
                    if rotation < -angular_velocity_threshold:  # If the rotation is past the left threshold
                        fire_rates["left"].append(fire_rate)
                    elif rotation > angular_velocity_threshold:  # If the rotation is past the right threshold
                        fire_rates["right"].append(fire_rate)
                    else:  # If the rotation is between the thresholds
                        fire_rates["none"].append(fire_rate)

    return fire_rates


def group_by_time_units(times, values):
    data_df = pd.DataFrame({'time': times, 'value': values})
    data_df_grouped_by_time = data_df.groupby('time')
    return np.array(data_df_grouped_by_time['value'].mean().reset_index()['value'])


def synchronize(ab_sync_dict):
    output = np.array([np.array([]), np.array([])])
    for item in ab_sync_dict.items():
        np.append(output[0], item[0])
        np.append(output[1], item[1])
    return output


if __name__ == '__main__':

    # region Setting parameters
    plot = {"position matrix": True,
            }
    debug = False

    t_precision = 1
    shuffle = False  # Shuffle the angular velocities
    # endregion

    # region Import position matrix
    position_matrix = positions.PositionMatrix('../data/HSpos_080602_ps17_160704.mat', start=1, precision=t_precision)
    position_matrix.data = position_matrix.data[:100000, :3]  # Only take time, x, and y

    # The second part of the smi file shows you the start timestamp in Neuralynx units and is used to determine when
    # to start the correlation between the spikes and the position tracking

    print(position_matrix.data[:, 0])

    # region Plot position matrix
    if plot["position matrix"]:
        plt.title("Position Matrix")
        position_matrix.plot()
    # endregion
    # endregion

    # region Compute angular velocities
    angular_velocities = position_matrix.find_angular_velocity()

    angular_velocities = group_by_time_units(position_matrix.data[:, 0],
                                             angular_velocities)

    av_unshuffled = np.array(angular_velocities)
    if shuffle:
        np.random.shuffle(angular_velocities)
    print("Sum of squares difference between shuffled and unshuffled: ", sum([x for x in (angular_velocities -
                                                                                     av_unshuffled) ** 2]))
    # endregion

    # region Compute distances
    distances = group_by_time_units(position_matrix.data[:, 0], position_matrix.distances[:10000])
    d_unshuffled = np.array(distances)
    if shuffle:
        np.random.shuffle(distances)
    # endregion

    # region Import T files
    decompressed_t_files = read_t_files("../data/t_files/*.t")

    #print(decompressed_t_files[8][:-10])
    print([x * 1000 for x in decompressed_t_files[8][-10:]])

