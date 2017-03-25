"""This file creates a basic FEM simulation."""
import numpy as np

from utils.helpers import *
from utils.vtk_interface import VTKInterface

# Assuming the object is glass
constants = [1.04E4, 1.04E4, 0, 6760]
density = 2588
time_step = 0.01

m_points, cells = VTKInterface.read('data/cube.2.vtk')
volume = get_volume(cells, m_points)
mass = get_mass(cells, m_points, density)
positions = np.copy(m_points)
positions[:, 2] = 1.1 * positions[:, 2]
beta = get_beta(cells, m_points)
velocities = np.zeros(m_points.shape)
gravity = -1 * np.array([0, 0, 0])


def get_acc(points, velocities):
    """The function makes things easier to type."""
    global cells
    global volume
    global mass
    global beta
    global constants
    return get_accel(cells, points, velocities, volume, mass, beta, constants)

acceleration = get_acc(positions, velocities)
h = time_step
h2 = h / 2.0
h6 = h / 6.0
for i in range(1000):
    q1 = velocities + h2 * acceleration
    k1 = get_acc(positions + h2 * velocities, q1)
    q2 = velocities + h2 * k1
    k2 = get_acc(positions + h2 * q1, q2)
    q3 = velocities + h * k2
    k3 = get_acc(positions + h * q2, q3)
    positions = positions + h * (velocities + h6 * (acceleration + k1 + k2))
    velocities = velocities + h6 * (acceleration + 2 * k1 + 2 * k2 + k3)
    acceleration = get_acc(positions, velocities)
    # if stress.any() != 0:
    #     import pdb
    #     pdb.set_trace()
    VTKInterface.write(positions, cells, 'output/sphere' + str(i) + '.vtk')
