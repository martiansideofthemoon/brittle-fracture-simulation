"""A list of functions to deform meshes."""
import numpy as np


def twist(positions):
    """Make one twist on the object."""
    positions[:, 1], positions[:, 2] = \
        positions[:, 1] * np.sin(positions[:, 0] / 2) + \
        positions[:, 2] * np.cos(positions[:, 0] / 2), \
        -positions[:, 1] * np.cos(positions[:, 0] / 2) + \
        positions[:, 2] * np.sin(positions[:, 0] / 2)
    return positions


def stretch(positions):
    """Stretch the object in Y direction."""
    positions[:, 1] = 1.1 * positions[:, 1]
    return positions


def squash(positions):
    """Squash the object in Y direction."""
    positions[:, 1] = 0.9 * positions[:, 1]
    return positions
