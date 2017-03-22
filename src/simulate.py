"""This file creates a basic FEM simulation."""
import numpy as np

from utils.helpers import *
from utils.vtk_interface import VTKInterface

# Assuming the object is glass
constants = [1.04E8, 1.04E8, 0, 6760]
density = 2588
time_step = 0.1

m_points, cells = VTKInterface.read('data/sphere.1.vtk')
points = np.copy(m_points)
beta = get_beta(cells, m_points)
velocities = np.zeros(m_points.shape)
gravity = -1 * np.array([0, 9.8, 0])

for i in np.arange(0, 10, time_step):
    dpos, dvel = get_derivatives(cells, points, velocities, beta)
    stress = get_stress(dpos, dvel, constants)
    internal_f = get_internal(cells, points, beta, stress)
    mass = get_mass(cells, points, density)
    internal_acc = internal_f / mass[:, None]
    acceleration = np.add(internal_acc, gravity)
    points = np.add(points, time_step * velocities)
    velocities = np.add(velocities, time_step * acceleration)
    VTKInterface.write(points, cells, 'output/sphere' + str(int(i * 10)) + '.vtk')
