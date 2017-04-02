"""This file creates a basic FEM simulation."""
import numpy as np
import utils.helpers as helpers
import utils.deformations as deformations

from utils.vtk_interface import VTKInterface


class Body(object):
    """This class represents a mesh of points."""

    def __init__(self, filename, density):
        """Read an object off a VTK file."""
        m_points, cells = VTKInterface.read(filename)
        self.m_points = m_points
        self.cells = cells
        self.beta = helpers.get_beta(cells, m_points)
        self.volume = helpers.get_volume(cells, m_points)
        self.density = density
        self.mass = helpers.get_mass(cells, m_points, density)
        self.positions = np.copy(m_points)
        self.velocities = np.zeros(m_points.shape)

    def update(self, pos_update, vel_update):
        """Update the positions and velocities of the points."""
        self.positions += pos_update
        self.velocities += vel_update

    def deform(self, deformation):
        """Modify positions to deform the mesh."""
        self.positions = deformation(self.positions)


class Simulate(object):
    """This class runs the RK4 simulation on an object."""

    def __init__(self, constants, sim_step, fps, body, namespace):
        """The standard initialization function."""
        self.lame = constants['lame']
        self.density = constants['density']
        self.h = sim_step
        self.out_rate = sim_step * int((1.0 / fps) / sim_step)
        self.thres_high = constants['thresholds']['high']
        self.thres_low = constants['thresholds']['low']
        self.body = body
        self.namespace = namespace

    def get_acc(self, pos, vel):
        """Wrapper for helper function get_accel()."""
        return helpers.get_accel(
            cells=self.body.cells,
            points=pos,
            velocities=vel,
            volume=self.body.volume,
            mass=self.body.mass,
            beta=self.body.beta,
            constants=self.lame
        )

    def get_update(self):
        """Integrate object's trajectory using RK4."""
        positions = self.body.positions
        velocities = self.body.velocities
        h = self.h
        out_rate = self.out_rate
        h2 = h / 2.0
        h6 = h / 6.0
        pos_update = np.zeros(positions.shape)
        vel_update = np.zeros(velocities.shape)
        acceleration = self.get_acc(positions, velocities)
        for j in range(int(out_rate / h)):
            q1 = velocities + vel_update + h2 * acceleration
            k1 = self.get_acc(
                pos=(positions + pos_update + h2 * velocities + vel_update),
                vel=q1
            )
            q2 = velocities + vel_update + h2 * k1
            k2 = self.get_acc(
                pos=(positions + pos_update + h2 * q1),
                vel=q2
            )
            q3 = velocities + vel_update + h * k2
            k3 = self.get_acc(
                pos=(positions + pos_update + h * q2),
                vel=q3
            )
            pos_update = pos_update + \
                h * (velocities + vel_update + h6 * (acceleration + k1 + k2))
            vel_update = vel_update + \
                h6 * (acceleration + 2 * k1 + 2 * k2 + k3)
            # Acceleration for next timestep
            acceleration = self.get_acc(
                pos=(positions + pos_update),
                vel=(velocities + vel_update)
            )
            if np.any(np.abs(pos_update) > self.thres_high) or \
               np.isnan(pos_update).any() or \
               np.isinf(pos_update).any():
                # np.any(np.abs(vel_update) > thres_high) or \
                # np.isnan(vel_update).any() or \
                # np.isinf(vel_update).any():
                return pos_update, vel_update
        return pos_update, vel_update

    def run(self, num_frames):
        """The number of frames to render in the scene."""
        thres_high = self.thres_high
        thres_low = self.thres_low
        out_rate = self.out_rate
        for i in range(num_frames):
            # First trying to speed up simulation
            pos_update, vel_update = self.get_update()
            while (np.all(np.abs(pos_update) < thres_low) or
                   np.all(np.abs(vel_update) < thres_low)) and \
                    self.h < (out_rate / 2.0):
                self.h = self.h * 2.
                pos_update, vel_update = self.get_update()
                print i, self.h
            # Now slowing down simulation to get reasonable results
            while np.any(np.abs(pos_update) > thres_high) or \
                    np.isnan(pos_update).any() or \
                    np.isinf(pos_update).any():
                # np.any(np.abs(vel_update) > thres_high) or \
                # np.isnan(vel_update).any() or \
                # np.isinf(vel_update).any():
                print np.any(np.abs(pos_update) > thres_high), \
                    np.isnan(pos_update).any(), \
                    np.isinf(pos_update).any()
                self.h = self.h / 2.
                pos_update, vel_update = self.get_update()
                print i, self.h
            print i, self.h
            self.body.update(pos_update, vel_update)
            self.write(i)

    def write(self, frame):
        """Output the frame to a VTK file."""
        VTKInterface.write(
            self.body.positions,
            self.body.cells,
            'output/' + self.namespace + str(frame) + ".vtk"
        )


# Assuming the object is glass
constants = {
    'lame': [1.04E4, 1.04E4, 0, 6760],
    'density': 2588,
    'thresholds': {
        'high': 10,
        'low': 0.000001
    }
}

body = Body('data/cube.2.vtk', constants['density'])
body.deform(deformations.squash)
sim = Simulate(constants, 0.001, 30, body, 'squashcube')
sim.run(100)
