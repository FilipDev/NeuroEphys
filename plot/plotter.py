import matplotlib.pyplot as plt
import numpy as np

# TODO: Generalize this to other forms of two dimensional data
def plot_cell_vs_cell(cell_1_data, cell_2_data, cell_1_n, cell_2_n):
    # Creating a 2D histogram
    heatmap, xedges, yedges = np.histogram2d(cell_1_data, cell_2_data, bins=50)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    heatmap[0, 0] = 0
    heatmap = np.arctan(heatmap / 2 / np.std(heatmap))

    # Plotting the heatmap
    plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='hot')
    plt.title("Cell {} vs Cell {}, firing rate per second".format(cell_1_n, cell_2_n))
    plt.xlabel("Cell {}".format(cell_1_n))
    plt.ylabel("Cell {}".format(cell_2_n))
    plt.show()

# TODO: Transfer remaining plotting code to plotter
