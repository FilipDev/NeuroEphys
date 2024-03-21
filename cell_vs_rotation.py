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
            }
    debug = False

    # Import position matrix
    position_matrix = positions.PositionMatrix('../data/HSpos_080602_ps17_160704.mat', start=1)
    position_matrix.data = position_matrix.data[:40000, :3]  # Only take time, x, and y

    if plot["position matrix"]:
        # Plot position matrix
        plt.title("Position Matrix")
        position_matrix.plot()

    # Compute rotational velocities
    angular_velocities = position_matrix.find_angular_velocity()

    # Select and read all T files
    t_files = glob.glob('../data/t_files/*.t')
    print(t_files[0])
    read_t_files = reader.read_t_files(t_files)
    decompressed_t_files = [spikes.reader.decompress_timestamp_data(read_t_files[i][1], 0) for i in range(len(read_t_files))]

    neuron_fire_angles = [list() for i in range(len(decompressed_t_files))]

    for i, t_file in enumerate(decompressed_t_files):
        for timepoint, value in enumerate(t_file[:40000]):
            if debug:
                print(timepoint, value)
            if value > 0:
                for k in range(int(value)):
                    if debug:
                        print("Added angle {} with timepoint {} to neuron {}'s fire history {} times".format(angular_velocities[timepoint], timepoint, i, value))
                    if abs(angular_velocities[timepoint]) > 0.1:
                        if position_matrix.distances[timepoint] > 7:
                            neuron_fire_angles[i].append(angular_velocities[timepoint])

    for t_file in range(len(decompressed_t_files)):
        hist = np.histogram(neuron_fire_angles[t_file], bins=50)
        plt.plot(hist[1][:-1], hist[0])
        plt.show()

    for i in range(len(neuron_fire_angles)):
        average_angle = sum(neuron_fire_angles[i]) / len(neuron_fire_angles[i])
        print(i, average_angle)

    if debug:
        # Real time debugging
        while True:
            command = input("Please enter your next command: \n")
            try:
                exec(command)
            except Exception as e:
                print(e)
