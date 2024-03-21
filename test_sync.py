import numpy as np

from analysis import analyzer
from sync import sync
from positions import positions
from spikes import t_reader
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dd = '../data/'

    #print(sync_data[0][0])

    position_data = positions.read_positions(dd + "HSpos_080602_ps17_160704.mat", start=0)

    uncorrected = position_data.copy()

    position_data = positions.correct_data(np.array([np.array([(t[0] - position_data[0][0]), t[1], t[2]]) for t in
                                            position_data]), precision=2)

    position_data = positions.smooth_values(position_data)

    print(position_data[:-50])

    t_files = t_reader.read_t_files_decompressed(dd + "t_files/*.t")

    print(len(t_files[0]))

    angles = positions.calculate_angles(position_data)

    angles[:, 1] = analyzer.apply_kernel(angles[:, 1], 30, 5, sigma=5, kernel_type="gaussian")

    distances = positions.calculate_distances(position_data)

    angular_velocities = positions.calculate_angular_velocity(angles)

    #angular_velocities = positions.threshold_angular_velocity(angular_velocities, distances, threshold=0.3)

    angular_velocities[:, 1] = np.round(angular_velocities[:, 1].astype(float), decimals=2)

    angular_velocities[:, 1] = analyzer.apply_kernel(angular_velocities[:, 1], 30, 11, sigma=11,
                                                     kernel_type="gaussian")

    import matplotlib.ticker as ticker

    # Assuming 'uncorrected' and 'angular_velocities' are defined and have the correct shapes
    # Set the figure size to stretch the graph
    fig = plt.figure(figsize=(10, 5))  # Adjust the size as needed

    plt.plot(uncorrected[:30 * 30, 0], angular_velocities[:30 * 30, 1])

    # Set the number of ticks on the x-axis
    ax = plt.gca()  # Get the current axes instance
    ax.xaxis.set_major_locator(ticker.MaxNLocator(20))  # Increase the number of ticks on the x-axis

    # Set the format for the ticks to show 4 significant digits
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.4g'))  # 'g' uses significant digits

    # Rotate the tick labels and set label size
    ax.tick_params(axis='x', which='major', labelsize=8, labelrotation=45)

    # Enable vertical grid at each x tick mark
    ax.grid(axis='x', which='major', linestyle='-', linewidth=0.5)  # Customize grid style as needed

    plt.show()

    #seconds_angular_velocities = analyzer.group_by_time_units(position_data[:, 0], angles[:, 1])

    import csv

    with open('C:\\Users\\filip\\Desktop\\Lab\\Project\\data\\080602_ps17_2016-07-04_13-22-47\\modified '
              'video\\first30secs-x.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for t, row in enumerate(angular_velocities[:30*30]):
            #writer.writerow(angular_velocities[t])
            writer.writerow([uncorrected[t, 0], position_data[t, 1]])

    with open('C:\\Users\\filip\\Desktop\\Lab\\Project\\data\\080602_ps17_2016-07-04_13-22-47\\modified '
              'video\\first30secs-y.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for t, row in enumerate(angular_velocities[:30*30]):
            #writer.writerow(angular_velocities[t])
            writer.writerow([uncorrected[t, 0], position_data[t, 2]])