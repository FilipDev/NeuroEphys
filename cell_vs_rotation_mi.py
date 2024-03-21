import glob
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analysis.analyzer as analyzer
import positions.positions2 as positions
import spikes.t_reader
import spikes.t_reader as reader
import utils.utils as utils

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    plot = {"position matrix": True,
            "rotational velocities": True,
            "rotation distribution": True,
            "binned rotational velocities": False,
            "mutual information totals": True,
            "mutual information right-left differences": True,
            "mutual information matrices": True
            }
    debug = False

    # Import position matrix
    position_matrix = positions.PositionMatrix('../data/HSpos_080602_ps17_160704.mat', start=1)
    position_matrix.data = position_matrix.data[45000:47000, :3]  # Only take time, x, and y

    if plot["position matrix"]:
        # Plot position matrix
        plt.title("Position Matrix")
        position_matrix.plot()

    # Compute rotational velocities
    rotational_velocities = position_matrix.find_rotational_velocity()

    # TODO: Figure out if by second is too long, if so, may need to reduce to 100ms
    # Create a dataframe to find the average rotational velocity in each second
    rotational_velocity_dataframe = pd.DataFrame({'time': position_matrix.data[:, 0], 'value': rotational_velocities})
    rotational_velocity_dataframe_grouped_by_time = rotational_velocity_dataframe.groupby('time')
    rotational_velocities_seconds = rotational_velocity_dataframe_grouped_by_time['value'].mean().reset_index()

    if plot["rotational velocities"]:
        # Plot rotational velocities
        plt.plot(rotational_velocities_seconds['time'], rotational_velocities_seconds['value'])
        plt.title("Rotational Velocities")
        plt.show()

    # Compute binned rotational velocities with 9 bins
    rv_bins = 21
    binned_rotational_velocities = analyzer.digitize_timeseries(rotational_velocities_seconds['value'],
                                                                rv_bins)

    if plot["binned rotational velocities"]:
        # Plot binned rotational velocities
        plt.title("Binned Rotational Velocities")
        plt.plot(binned_rotational_velocities)
        plt.show()

    if plot["rotation distribution"]:
        # Plot histogram of binned rotational velocities
        plt.title("Histogram of Binned Rotational Velocities")
        plt.hist(binned_rotational_velocities)
        plt.show()
        plt.title("Histogram of Unbinned Rotational Velocities")
        plt.hist(rotational_velocities_seconds['value'], bins=21)
        plt.show()
        plt.title("Histogram of Unbinned Angles")
        plt.hist(position_matrix.angular_velocities, bins=101)
        plt.show()

    exit()

    # Select and read all T files
    t_files = glob.glob('../data/t_files/*.t')
    read_t_files = reader.read_t_files(t_files)
    decompressed_t_files = [spikes.reader.decompress_timestamp_data(read_t_files[i][1], 0) for i in range(len(read_t_files))]

    # Initialize mutual information
    all_val_mutual_infos = []
    stream_mutual_infos = []

    # Compute mutual information for each T file with the binned rotational velocities
    for i, t_file in enumerate(decompressed_t_files):
        # Trim either the T File or the Binned Rotational Velocities to match
        t_file_data, binned_rotational_velocities = utils.trim_to_match(t_file, binned_rotational_velocities)

        # Find the mutual information for each value pair
        all_val_mutual_infos.append(analyzer.all_values_mutual_information(t_file_data, binned_rotational_velocities))

        # Find the mutual information for each stream pair (T File, Binned Rotational Velocities)
        stream_mutual_infos.append(analyzer.stream_mutual_information(t_file_data, binned_rotational_velocities))

    # Find the average MI for each T file as it corresponds to left and right rotational velocities
    right_left_differences = []  # Positive if right, negative if left
    for i, mmi_all_values in enumerate(all_val_mutual_infos):  # Go through all mutual information for each T file

        # Converts the dictionary of values (which allows for negative indexing) to a 2D array
        # min_x_y: [0] is the minimum value of the first dimension (activation level),
        # and [1] is the minimum value of the second dimension (rotation value)
        val_map, min_x_y = utils.two_dimensional_dict_to_value_map(mmi_all_values)

        max_val_index = np.unravel_index(np.argmax(val_map), val_map.shape)  # TODO: What does this do?
        max_val = val_map[max_val_index]
        print("T file: ", i)

        std_val_map = np.std(val_map[1:])

        total_average = np.mean(val_map[1:])
        rotation_indices = np.arange(val_map.shape[1]) + min_x_y[1]
        print("Rotation indices: ", rotation_indices)
        left_normalized = np.sum(val_map[1:, rotation_indices < 0] / std_val_map)
        right_normalized = np.sum(val_map[1:, rotation_indices > 0] / std_val_map)
        center_total = np.sum(val_map[1:, rotation_indices == 0] / std_val_map)

        print("Left normalized: ", left_normalized)
        print("Right normalized: ", right_normalized)
        print("Center total: ", center_total)
        print("Total average: ", total_average)

        # Comparison method 1
        #left_right_differences.append((right_average - center_average) - (left_average - center_average))

        # Comparison method 2
        #right_left_differences.append((right_average - left_average) / center_average)

        # Comparison method 3
        right_left_differences.append((right_normalized ** 2 - left_normalized ** 2) / total_average ** 2)

        #left_right_differences.append(left_average / total_average)
        #right_left_differences.append(right_average / total_average - left_average / total_average)


    avg = np.abs(np.average(right_left_differences))
    print("Avg absolute right-left diff: ", avg)

    if plot["mutual information right-left differences"]:
        # Plot total mutual information for each T file
        plt.title("Mutual Information Right-Left Difference By T File")
        plt.xticks(np.arange(len(right_left_differences)))
        plt.bar(range(len(right_left_differences)), right_left_differences)
        plt.show()

    if plot["mutual information totals"]:
        # Plot total mutual information for each T file
        plt.title("Mutual Information By T File")
        plt.xticks(np.arange(len(t_files)))
        plt.bar(range(len(stream_mutual_infos)), stream_mutual_infos)
        plt.show()

    # Print highest MI TFile
    max_mi_tfile_index = np.argmax(stream_mutual_infos)
    print("Most MI TFile: ", max_mi_tfile_index, " MI: ", stream_mutual_infos[max_mi_tfile_index])

    if plot["mutual information matrices"]:
        # Plot the MIs for each T File in terms of neuronal activity and rotational velocity
        for i, mmi_all_values in enumerate(all_val_mutual_infos):
            val_map, min_x_y = utils.two_dimensional_dict_to_value_map(mmi_all_values)
            max_val_index = np.unravel_index(np.argmax(val_map), val_map.shape)
            max_val = val_map[max_val_index]
            plt.imshow(val_map, cmap='hot', interpolation='nearest')
            plt.colorbar(label='Value')
            plt.title(f"T File: {i}")
            plt.gca().invert_yaxis()
            plt.xlabel("Rotational Velocity")
            plt.ylabel("Spikes per Second")
            plt.xticks(np.arange(rv_bins), np.arange(rv_bins) - math.floor(rv_bins / 2))
            plt.show()

    if debug:
        # Real time debugging
        while True:
            command = input("Please enter your next command: \n")
            try:
                exec(command)
            except Exception as e:
                print(e)
