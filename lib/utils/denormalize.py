import numpy as np
from torchvision.transforms.functional import normalize


def denormalize(tensor, mean, std):
    mean = np.array(mean)
    std = np.array(std)
    mean = -mean / std
    std = 1 / std

    if isinstance(tensor, np.ndarray):
        return (tensor - mean.reshape(-1, 1, 1)) / std.reshape(-1, 1, 1)
    return normalize(tensor, mean, std)