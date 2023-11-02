import glob

import matplotlib.pyplot as plt
import numpy as np

import analysis.analyzer as analyzer
import spikes.reader as reader
import plot.plotter as plotter

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    plot = {
            "cell vs cell": True,
            "single cell": False
            }
    debug = True

    # Select and read all T files
    t_files = glob.glob('../../data/t_files/*.t')
    read_t_files = reader.read_t_files(t_files)

    significant_digits = 1

    # Decompress and smooth
    print("Decompressing and smoothing")
    decompressed_t_files = [analyzer.decompress_timestamp_data(read_t_files[i][1], significant_digits) for i in range(len(read_t_files))]
    smoothed_t_files = [analyzer.apply_kernel(
        decompressed_t_files[i], 10**significant_digits, window_size=5000, step_size=10,
        kernel_type="gaussian", sigma=5000
    ) for i in range(len(decompressed_t_files))]
    print("Done")

    # Plot single cell histograms
    if plot["single cell"]:
        print("Plotting single cell histograms")
        for i in range(len(smoothed_t_files)):
            plt.title("Histogram of cell {}, firing rate per second".format(i))
            plt.hist(smoothed_t_files[i] / np.std(smoothed_t_files[i]), bins=100, range=(1,20))
            plt.show()

    # Plot cell vs cell heatmaps
    if plot["cell vs cell"]:
        print("Plotting cell vs cell heatmaps")
        for i in range(len(smoothed_t_files)):
            for j in range(len(smoothed_t_files)):
                if not (i == 11 and j == 1) and not (i == 6 and j == 7):  # Here I select two cell pairs to plot
                    continue
                if i == j:  # Don't plot cell against itself
                    continue
                plotter.plot_cell_vs_cell(smoothed_t_files[i], smoothed_t_files[j], i, j)

    if debug:
        # Real time debugging
        while True:
            command = input("Please enter your next command: \n")
            try:
                exec(command)
            except Exception as e:
                print(e)
