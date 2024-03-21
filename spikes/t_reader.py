import glob
import random
from pathlib import Path
import os
import numpy as np

# TODO: Convert to object oriented Spikes class
from analysis import analyzer


def read_t_files(t_files) -> list:
    """
    Reads a collection of T-files and returns a list of spike timestamps in seconds for each T-file.
    :param t_files: The collection of T-files to be read
    :return: A tuple containing the sample rate and an nparray of spike timestamps in seconds for each T-file

    # Todo: Add the ability to read the sample rate from the T-file

    >>> read_t_files(["/TT9_11/TT9_11.t"])
    [10000, [7.9362, 18.3298, 54.0931, 69.2228, 97.5874, 140.3649, 146.8417, 147.9937, 171.2386, 177.0499, ...]]
    """

    all_processed_timestamps = []
    for i, T_file in enumerate(t_files):
        # Split the T_file
        directory, base_name = os.path.split(T_file)
        file_extension = Path(base_name).suffix

        sample_rate = 10000
        with open(T_file, 'rb') as f:
            bin_data = f.read()
            text = bin_data.decode('latin-1').split("%%ENDHEADER")[0]
            sample_rate_str = text.split("(")[1].split(")")[0]
            if sample_rate_str == "tenths of msecs":
                sample_rate = 10000
            else:
                print("The sample rate in the file header is: \"{}\"\n"
                      .format(sample_rate_str))
                sample_rate = int(input("Please input the sample rate (Hz) in digits: "))
                print("The sample rate was successfully set to: {}Hz\n".format(sample_rate))

        # Open the file in binary read mode
        with open(T_file, 'rb') as f:

            # Depending on the file extension, read the spike data in different formats
            if file_extension in ['.raw64']:
                # Read data as 64-bit integers and multiply by 10000
                timestamps = np.fromfile(f, dtype=np.uint64) * 10000
            elif file_extension in ['.raw32']:
                # Read data as 32-bit integers and multiply by 10000
                timestamps = np.fromfile(f, dtype=np.uint32) * 10000
            elif file_extension in ['.t64', '._t64', '.k64']:
                # Read data as 64-bit integers
                timestamps = np.fromfile(f, dtype=np.uint64)
            elif file_extension in ['.t32', '.t', '._t', '.k32', '.k']:
                # Read data as 32-bit integers
                timestamps = np.fromfile(f, dtype=np.uint32)
            else:
                # If the file extension is not recognized, raise an error
                raise ValueError('LoadSpikes: Unknown file extension.')

        processed_timestamps = np.zeros(len(timestamps))

        # Process the timestamps by dividing them by 10000
        for m, timestamp in enumerate(timestamps):
            processed_timestamps[m] = timestamp / sample_rate

        # Sort the timestamps
        processed_timestamps = sorted(processed_timestamps)

        # Append the sorted timestamps to the final list
        all_processed_timestamps.append((sample_rate, processed_timestamps))

        # Print the first 10 timestamp strings
        print(f"T File: {T_file} of index {i} was read: {processed_timestamps[:10]} with sample rate of"
              f" {sample_rate}Hz")

    return all_processed_timestamps


def read_t_files_decompressed(file_glob, precision=0):
    """
    Reads a collection of T-files and returns the number of spikes in each seconds for each T-file.
    :param file_glob:
    :param precision:
    :return:
    """
    t_files = glob.glob(file_glob)
    print(t_files[0])
    read = read_t_files(t_files)
    return [decompress_timestamp_data(read[i][1], precision=precision) for i in
            range(len(read))]


def generate_sample_timestamp_data(n_timestamps: int, min_max_difference: tuple, precision: int):
    """
    Generates artificial timestamp data for testing analysis
    :param n_timestamps: The amount of timestamps to generate
    :param min_max_difference: The window of allowed differences from the previous value
    :param precision: The precision of the timestamps in significant digits
    :return: The generated timestamp data

    >>> generate_timestamp_data(3, (0, 1), 3)
    [0.943 1.826 2.56]
    """

    timestamps = np.zeros(n_timestamps)
    for n in range(n_timestamps):
        last_element = timestamps[n - 1] if len(timestamps) > 0 else 0
        timestamps[n] = round(last_element + random.uniform(min_max_difference[0], min_max_difference[1]), precision)
    return timestamps


def decompress_timestamp_data(timestamps: np.ndarray, precision: int):
    """
    Decompresses timestamp data into a timeseries binary array, while aggregating based on precision if necessary
    :param timestamps: An array of the timestamps
    :param precision: The number of significant digits to include
    :return: The decompressed values as a binary array

    >>> decompress_timestamp_data([4.402, 5.591, 7.063, 9.18, 12.024, 16.364, 21.146, 23.243, 24.675, 29.257], 0)
    array([0., 0., 0., 0., 1., 0., 1., 1., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1.])
    """
    decompressed_range = int(round(timestamps[-1], precision) * 10 ** precision) + 1

    binary_data = np.zeros(decompressed_range)

    for timestamp in timestamps:
        timestamp = int(round(timestamp, precision) * 10 ** precision)
        binary_data[timestamp] = binary_data[timestamp] + 1

    return binary_data
