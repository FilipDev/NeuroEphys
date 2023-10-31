import numpy as np
import math


def get_sliding_window_factor(center: float, x: float, size: float) -> float:
    return int(abs(x - center) < size / 2)


def get_gaussian_factor(center: float, x: float, window_size: float, **kwargs) -> float:
    """
    :param center: The mean value of the gaussian (its center)
    :param x: The point to evaluate
    :param window_size: The maximum size of the window
    :param kwargs: sigma parameter (default: 1)
    :return: The gaussian factor of the point
    """
    if abs(x - center) > window_size / 2:
        return 0
    sigma = float(kwargs.get('sigma', 1))
    gaussian = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - center) / sigma) ** 2)
    return gaussian


def get_factor(kernel, center, x, window_size, **kwargs):
    return kernel(center, x, window_size, **kwargs)


kernel_types = {
    'simple': get_sliding_window_factor,
    'gaussian': get_gaussian_factor
}


def get_kernel(kernel_type):
    return kernel_types.get(kernel_type)
