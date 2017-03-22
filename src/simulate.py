"""This file creates a basic FEM simulation."""
import numpy as np

from utils.helpers import *
from utils.vtk_interface import VTKInterface

constants = [1.04E8, 1.04E8, 0, 6760]
density = 2588

points, cells = VTKInterface.read('data/sphere.1.vtk')
beta = get_beta(cells, points)
velocities = np.zeros(points.shape)
dpos, dvel = get_derivatives(cells, points, velocities, beta)
stress = get_stress(dpos, dvel, [1, 2, 3, 4])
internal_f = get_internal(cells, points, beta, stress)
mass = get_mass(cells, points, density)

