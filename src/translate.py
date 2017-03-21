#!/usr/bin/python
"""This is a test script that translates a sphere."""
import numpy as np

from utils.vtk_interface import VTKInterface

points, cells = VTKInterface.read('data/sphere.1.vtk')

for i in range(0, 100):
    if i % 10 == 0:
        print i
    points1 = np.add(points, (i + 1) * np.array([0.1, 0, 0]))
    VTKInterface.write(points1, cells, 'output/sphere' + str(i) + '.vtk')
