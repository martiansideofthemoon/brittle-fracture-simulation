"""A list of functions to provide external acceleration."""
import numpy as np


def gravity_ground(body):
    """Adding interface for gravity and ground."""
    # Assume a ground at position y = -5
    external = np.zeros([body.m_points.shape[0], 3])
    external += np.array([0, -9.8, 0])
    collision = 50 * -1 * np.minimum(body.positions[:, 1] + 5, 0)
    external[:, 1] += collision
    return external


def stretch(body):
    """Stretch the corners in all directions."""
    external = np.zeros([body.m_points.shape[0], 3])
    external[:8] += 80 * (body.m_points[:8] - np.array([0.5, 0.5, 0.5]))
    return external
