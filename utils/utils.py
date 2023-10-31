import math

import numpy as np


def trim_to_match(X, Y):
    lens = [len(X), len(Y)]
    argmin = np.argmin(lens)
    min_index = lens[argmin]
    if argmin == 1:
        X = X[:min_index]
    elif argmin == 0:
        Y = Y[:min_index]
    return X, Y


def two_dimensional_dict_to_value_map(d):
    min_x = np.min(list(d.keys()))
    min_y = math.inf
    max_x = np.max(list(d.keys()))
    max_y = np.max(list(d[min_x].keys()))
    print(d)

    for x in d.keys():
        for y in d[x]:
            if y < min_y:
                min_y = y
    min_x, min_y, max_x, max_y = int(min_x), int(min_y), int(max_x), int(max_y)
    print(min_x, min_y, max_x, max_y)
    val_map = np.zeros((int(max_x - min_x + 1), int(max_y - min_y + 1)))
    for x in d.keys():
        for y in d[x].keys():
            x, y = int(x), int(y)
            val_map[x - min_x][y - min_y] = d[x][y]

    return val_map, (min_x, min_y)
