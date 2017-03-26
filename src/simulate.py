"""This file creates a basic FEM simulation."""
import numpy as np

from utils.helpers import *
from utils.vtk_interface import VTKInterface

# Assuming the object is glass
constants = [1.04E8, 1.04E8, 0, 6760]
# constants = [1, 1, 0, 0]
density = 2588
# density = 1
time_step = 0.0001

m_points, cells = VTKInterface.read('data/cube.1.vtk')
points = np.copy(m_points) * np.array([1.0, 2, 1.0])
beta = get_beta(cells, m_points)
velocities = np.zeros(m_points.shape)
gravity = 0 * np.array([0, 9.8, 0])

mass = get_mass(cells, m_points, density)
volume = get_volume(cells, m_points)

for i in np.arange(0, 100):
    dpos, dvel = get_derivatives(cells, points, velocities, beta)
    stress = get_stress(dpos, dvel, constants)
    internal_f = get_internal(cells, points, beta, stress, volume)
    print stress.max(), stress.min() 
    print internal_f.max(), internal_f.min()
    print np.sum(mass), mass.min(), mass.max()
    internal_acc = internal_f / mass[:, None]
    acceleration = np.add(internal_acc, gravity)
    points = np.add(points, time_step * velocities)
    velocities = np.add(velocities, time_step * acceleration)
    VTKInterface.write(points, cells, 'output/sphere' + str(int(i)) + '.vtk')
