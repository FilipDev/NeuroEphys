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


def read_t_files(file_glob):
    t_files = glob.glob(file_glob)
    print(t_files[0])
    read_t_files = reader.read_t_files(t_files)
    return [spikes.reader.decompress_timestamp_data(read_t_files[i][1], precision=t_precision) for i in
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

    pps = 30  # Positions per second (framerate of video)
    position_frames = 75000 * pps
    t_precision = 1
    shuffle = False  # Shuffle the angular velocities
    # endregion

    # region Import position matrix
    position_matrix = positions.PositionMatrix('../data/HSpos_080602_ps17_160704.mat', start=1, precision=t_precision)
    position_matrix.data = position_matrix.data[:position_frames, :3]  # Only take time, x, and y

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
    distances = group_by_time_units(position_matrix.data[:, 0], position_matrix.distances[:position_frames])
    d_unshuffled = np.array(distances)
    if shuffle:
        np.random.shuffle(distances)
    # endregion

    # region Import T files
    decompressed_t_files = read_t_files("../data/t_files/*.t")

    if debug:
        print("t_file length: ", len(decompressed_t_files[0]))
    # endregion

    # region Apply kernels
    sigma = 11  # 10.1 seconds

    #for i, t_file in enumerate(decompressed_t_files):
        #decompressed_t_files[i] = analyzer.apply_kernel(t_file, 1, 5, kernel_type='gaussian', sigma=3)

    #angular_velocities = analyzer.apply_kernel(angular_velocities, 1, 9, kernel_type='gaussian', sigma=7)
    #distances = analyzer.apply_kernel(distances, 1, 9, kernel_type='gaussian', sigma=7)
    # endregion

    # region Plot distance histogram
    plt.hist(distances, bins=50)
    plt.title("Distance Histogram")
    plt.xlim(0, 11)
    plt.show()
    # endregion

    if False:
        for i, t_file in enumerate(decompressed_t_files):
            plt.scatter(t_file[:len(angular_velocities)], angular_velocities)
            plt.title(f"Firing Rate vs Angular Velocity Neuron{i}")
            plt.show()

    for i, t_file in enumerate(decompressed_t_files):
        plt.scatter(t_file[:len(angular_velocities)], distances)
        plt.title(f"Firing Rate vs Distances Neuron{i}")
        plt.show()

    exit()

    # region Mixed plot
    plt.plot(distances)
    plt.plot(angular_velocities)
    plt.plot(decompressed_t_files[2][:len(distances)])
    plt.show()
    # endregion

    # region Set thresholds
    angular_velocities_std = np.std(angular_velocities)
    distances_std = np.std(distances)

    distance_threshold = distances_std
    angular_velocity_threshold = angular_velocities_std / 2
    # endregion

    # region Eliminate angular velocities below distance threshold
    for i in range(len(distances)):
        if distances[i] < distance_threshold:
            angular_velocities[i] = math.nan
    # endregion

    # region Plot angular velocity histogram with thresholds marked
    plt.hist(angular_velocities, bins=50)
    plt.axvline(x=-angular_velocity_threshold, color='r', linestyle='--')
    plt.axvline(x=angular_velocity_threshold, color='r', linestyle='--')
    plt.show()
    # endregion

    # region Loop through all T files and collect the fire rates for each direction and timepoint
    cell_directional_firing_rates = []

    for i, t_file in enumerate(decompressed_t_files):
        firing_rates = get_cell_firing_rates_for_directions(t_file, angular_velocities, angular_velocity_threshold,
                                                            distance_threshold)
        cell_directional_firing_rates.append(firing_rates)
    # endregion

    # region Plot the average cell firing rates for each direction
    for i, cell_firing_rates in enumerate(cell_directional_firing_rates):
        left_avg = sum(cell_firing_rates["left"]) / len(cell_firing_rates["left"])  # / lefts_over_rights
        none_avg = sum(cell_firing_rates["none"]) / len(cell_firing_rates["none"])
        right_avg = sum(cell_firing_rates["right"]) / len(cell_firing_rates["right"])
        # 2.576 is the 99% confidence interval
        left_conf = 2.576 * np.std(cell_firing_rates["left"]) / np.sqrt(len(cell_firing_rates["left"]) - 1)
        none_conf = 2.576 * np.std(cell_firing_rates["none"]) / np.sqrt(len(cell_firing_rates["none"]) - 1)
        right_conf = 2.576 * np.std(cell_firing_rates["right"]) / np.sqrt(len(cell_firing_rates["right"]) - 1)
        plt.plot([left_avg, none_avg, right_avg])
        plt.xticks([0, 1, 2], ["Left", "None", "Right"])
        plt.xlabel("Direction")
        plt.errorbar([0, 1, 2], [left_avg, none_avg, right_avg], yerr=[left_conf, none_conf, right_conf])
        plt.ylabel("Average Firing Rate")
        plt.title(f"cell {i}")
        plt.show()

        # Assuming cell_directional_firing_rates is your data
        # and i is defined

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharey=True)
        plt.suptitle(f"Cell {i} Directional Firing Rates")

        bin_edges = np.histogram_bin_edges(cell_directional_firing_rates[i]["left"], bins=20)

        left_over_right = len(cell_directional_firing_rates[i]["left"]) / len(cell_directional_firing_rates[i]["right"])

        # Left histogram (regular)
        axes[0].hist(np.array(cell_directional_firing_rates[i]["left"]) / left_over_right, orientation='horizontal',
                     color='red',
                     bins=bin_edges)
        max_left = axes[0].get_xlim()[0]
        axes[0].set_title('Left')
        axes[0].set_xlabel('Frequency')
        axes[0].invert_xaxis()  # Invert the x-axis to have the histogram face right
        axes[0].set_yticks(bin_edges)
        axes[0].grid(True, which='both', axis='both', color='gray', linestyle='-', linewidth=0.5)  # Add grid

        # Right histogram (mirrored)
        axes[1].hist(cell_directional_firing_rates[i]["right"], orientation='horizontal', bins=bin_edges)
        max_right = axes[0].get_xlim()[0]
        axes[1].set_title('Right')
        axes[1].set_xlabel('Frequency')
        axes[1].set_yticks(bin_edges)
        axes[1].grid(True, which='both', axis='both', color='gray', linestyle='-', linewidth=0.5)  # Add grid

        max_frequency = max([max_left, max_right])
        axes[0].set_xlim([max_frequency, 0])
        axes[1].set_xlim([0, max_frequency])

        plt.subplots_adjust(wspace=0)  # Adjust the space between the histograms to be zero
        plt.show()

    # endregion

    exit()

    # Loop through all T files and plot the histograms
    for t_file in range(len(decompressed_t_files)):
        # Firing rate histograms for each direction
        hist_right = np.histogram(cell_directional_firing_rates[t_file]["right"], bins=10)
        hist_none = np.histogram(cell_directional_firing_rates[t_file]["none"], bins=10)
        hist_left = np.histogram(cell_directional_firing_rates[t_file]["left"], bins=10)

        # Total of each direction
        sum_left = sum(hist_left[0])
        sum_right = sum(hist_right[0])

        # Total of each direction for each firing rate
        totals = hist_right[0] + hist_left[0] + hist_none[0]

        # Total of all firings
        total_sum = sum(totals)

        # Calculating proportion of total of each direction
        # Determines the proportion of how many more left turns there were than right turns
        left_prop_of_total = sum_left / total_sum
        right_prop_of_total = sum_right / total_sum
        lefts_over_rights = left_prop_of_total / right_prop_of_total

        '''
        # Plotting total frequency of each firing rate
        plt.plot(hist_none[1][:-1], totals)
        plt.title("Fire rate frequencies for t_file {}".format(t_file))
        plt.ylabel("Frequency")
        plt.xlabel("Firing rate")
        plt.show()
        '''

        # Adding epsilon to totals to avoid division by zero
        epsilon = 1e-10
        totals_safe = totals + epsilon

        # Adjust histogram so that frequency is proportion of total of each direction

        # Now we divide the left histogram by the proportion of lefts over rights to make our adjustment
        sm_hist_left = hist_left[0] / lefts_over_rights  # / totals_safe
        sm_hist_right = hist_right[0]  # / totals_safe
        sm_hist_none = hist_none[0]  # / totals_safe

        # Compute 95% confidence intervals

        total_counts_per_bin = sm_hist_left + sm_hist_right + sm_hist_none  # Add counts for 'none' direction if needed

        alpha = 0.05  # 95% CI

        left_errors = calculate_errors(sm_hist_left, total_counts_per_bin, alpha, precision=2)
        right_errors = calculate_errors(sm_hist_right, total_counts_per_bin, alpha, precision=2)

        # Plotting histograms with error bars
        fig, ax1 = plt.subplots()

        ax1.set_xlabel('Firing rate (Hz)')
        ax1.set_ylabel('Frequency')
        ax1.errorbar(hist_left[1][:-1] * 10 ** t_precision, sm_hist_right / 10 ** t_precision,
                     yerr=right_errors / 10 ** t_precision,
                     color="tab:red", fmt='-o')  #
        # Added error bars
        ax1.errorbar(hist_left[1][:-1] * 10 ** t_precision, sm_hist_left / 10 ** t_precision,
                     yerr=left_errors / 10 ** t_precision,
                     color="tab:blue", fmt='-o')  #
        # Added error bars
        plt.legend(["Right", "Left"])

        plt.title("t_file {}".format(t_file))
        plt.show()

        # Compute the difference between the left and right histograms
        ss_diffs = (sm_hist_right - sm_hist_left) ** 2
        ss_difference = np.sum(ss_diffs) / sum(hist_left[0] + hist_right[0])
        # print(f"ss_difference: {ss_difference}, t_file: {t_file}")
        print(ss_difference)

    if debug:
        # Real time debugging
        while True:
            command = input("Please enter your next command: \n")
            try:
                exec(command)
            except Exception as e:
                print(e)
