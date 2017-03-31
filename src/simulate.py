"""This file creates a basic FEM simulation."""
import numpy as np

from utils.helpers import *
from utils.vtk_interface import VTKInterface

# Assuming the object is glass
constants = [1.04E4, 1.04E4, 0, 6760]
density = 2588
h = 0.001 # time_step
out_time_step = 0.032 # 30 fps
thres_high = 10
thres_low = 0.000001

m_points, cells = VTKInterface.read('data/cube.2.vtk')
volume = get_volume(cells, m_points)
mass = get_mass(cells, m_points, density)
positions = np.copy(m_points)
positions[:, 2] = 1.5 * positions[:, 2]
positions[:, 1] = 1.5 * positions[:, 1]
beta = get_beta(cells, m_points)
velocities = np.zeros(m_points.shape)
gravity = -0 * np.array([0, 9.8, 0])


def get_acc(points, velocities):
    """The function makes things easier to type."""
    global cells
    global volume
    global mass
    global beta
    global constants
    global gravity
    return get_accel(cells, points, velocities, volume, mass, beta, constants) + gravity


def get_update(positions, velocities):
    global h
    global out_time_step
    global thres_high
    h2 = h / 2.
    h6 = h / 6.
    pos_update = np.zeros(positions.shape)
    vel_update = np.zeros(velocities.shape)
    acceleration = get_acc(positions, velocities)
    for j in range(int(out_time_step/h)):
        q1 = velocities + vel_update + h2 * acceleration
        k1 = get_acc(positions + pos_update + h2 * velocities + vel_update, q1)
        q2 = velocities + vel_update + h2 * k1
        k2 = get_acc(positions + pos_update + h2 * q1, q2)
        q3 = velocities + vel_update + h * k2
        k3 = get_acc(positions + pos_update + h * q2, q3)
        pos_update = pos_update + h * (velocities + vel_update + h6 * (acceleration + k1 + k2))
        vel_update = vel_update + h6 * (acceleration + 2 * k1 + 2 * k2 + k3)
        acceleration = get_acc(positions + pos_update, velocities + vel_update)
        if np.any(np.abs(pos_update) > thres_high) or np.isnan(pos_update).any() or np.isinf(pos_update).any() :#or \
           # np.any(np.abs(vel_update) > thres_high) or np.isnan(vel_update).any() or np.isinf(vel_update).any():
            return pos_update, vel_update
    return pos_update, vel_update

for i in range(100):
    pos_update, vel_update = get_update(positions, velocities)
    while (np.all(np.abs(pos_update) < thres_low) or np.all(np.abs(vel_update) < thres_low) ) and h < out_time_step:
        h = h * 2.
        pos_update, vel_update = get_update(positions, velocities)
        print i, h

    while np.any(np.abs(pos_update) > thres_high) or np.isnan(pos_update).any() or np.isinf(pos_update).any() :#or \
          # np.any(np.abs(vel_update) > thres_high) or np.isnan(vel_update).any() or np.isinf(vel_update).any():
        print np.any(np.abs(pos_update) > thres_high), np.isnan(pos_update).any(), np.isinf(pos_update).any()
        # print np.any(np.abs(vel_update) > thres_high), np.isnan(vel_update).any(), np.isinf(vel_update).any()
        h = h / 2.
        pos_update, vel_update = get_update(positions, velocities)
        print i, h

    print i, h
    positions = positions + pos_update
    velocities = velocities + vel_update
    VTKInterface.write(positions, cells, 'output/sphere' + str(i) + '.vtk')
