import glob

import matplotlib.pyplot as plt
import numpy as np

import analysis.analyzer as analyzer
import spikes.reader as reader

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    plot = {
            "cell vs cell": True,
            "single cell": False
            }

    # Select and read all T files
    t_files = glob.glob('../../data/t_files/*.t')
    read_t_files = reader.read_t_files(t_files)

    significant_digits = 1

    # Decompress and smooth
    print("Decompressing and smoothing")
    decompressed_t_files = [analyzer.decompress_timestamp_data(read_t_files[i][1], significant_digits) for i in range(len(read_t_files))]
    smoothed_t_files = [analyzer.apply_kernel(
        decompressed_t_files[i], 10**significant_digits, window_size=100, step_size=10,
        kernel_type="gaussian", sigma=100
    ) for i in range(len(decompressed_t_files))]
    print("Done")

    # Plot single cell histograms
    if plot["single cell"]:
        print("Plotting single cell histograms")
        for i in range(len(smoothed_t_files)):
            plt.title("Histogram of cell {}, firing rate per second".format(i))
            plt.hist(smoothed_t_files[i] / np.std(smoothed_t_files[i]), bins=100, range=(1,20))
            plt.show()

    # Plot cell vs cell scatterplots
    if plot["cell vs cell"]:
        print("Plotting cell vs cell scatterplots")
        for i in range(len(smoothed_t_files)):
            for j in range(len(smoothed_t_files)):
                if not (i == 9 and j == 7):
                    continue
                if i == j:
                    continue
                plt.title("Cell {} vs Cell {}, firing rate per second".format(i, j))
                # Creating a 2D histogram
                heatmap, xedges, yedges = np.histogram2d(smoothed_t_files[i], smoothed_t_files[j], bins=50)
                extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                heatmap[0, 0] = 0
                heatmap = np.arctan(heatmap / np.std(heatmap))

                # Plotting the heatmap
                plt.clf()
                plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='hot')
                plt.title("Cell {} vs Cell {}, firing rate per second".format(i, j))
                plt.xlabel("Cell {}".format(i))
                plt.ylabel("Cell {}".format(j))
                plt.show()