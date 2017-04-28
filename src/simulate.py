"""This file creates a basic FEM simulation."""
import numpy as np
import sys
import utils.helpers as helpers
import utils.deformations as deformations
import utils.external as external

from utils.vtk_interface import VTKInterface
from utils.povray_interface import POVRayInterface


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

    def clone_point(self, index):
        """Helper function for fracture() routine."""
        self.m_points = \
            np.concatenate((self.m_points, self.m_points[[index]]), axis=0)
        self.positions = \
            np.concatenate((self.positions, self.positions[[index]]), axis=0)
        self.velocities = \
            np.concatenate((self.velocities, self.velocities[[index]]), axis=0)
        # Return index of newly added point
        return (self.m_points.shape[0] - 1)

    def delete_last(self):
        """Helper function for fracture() routine."""
        self.m_points = self.m_points[:-1]
        self.positions = self.positions[:-1]
        self.velocities = self.velocities[:-1]

    def add_pt(self, position, cell):
        """Helper function to add a point using barycentric coordinates."""
        positions = np.transpose(self.positions[self.cells[cell]])
        ones = np.ones((1, positions.shape[1]))
        orig = np.linalg.inv(np.concatenate([positions, ones], axis=0))
        bary = orig.dot(np.concatenate([position, np.array([1])]))
        m_points = np.transpose(self.m_points[self.cells[cell]])
        m_point = np.concatenate([m_points, ones], axis=0).dot(bary)[:3]
        velocities = np.transpose(self.velocities[self.cells[cell]])
        velocity = np.concatenate([velocities, ones], axis=0).dot(bary)[:3]
        # Adding the point to the mesh
        self.m_points = \
            np.concatenate((self.m_points, np.array([m_point])), axis=0)
        self.positions = \
            np.concatenate((self.positions, np.array([position])), axis=0)
        self.velocities = \
            np.concatenate((self.velocities, np.array([velocity])), axis=0)
        # Return index of newly added point
        return (self.m_points.shape[0] - 1)

    def recompute(self):
        """Recompute mesh parameters."""
        self.beta = helpers.get_beta(self.cells, self.m_points)
        self.volume = helpers.get_volume(self.cells, self.m_points)
        self.mass = helpers.get_mass(self.cells, self.m_points, self.density)

    def split(self, point, normal):
        """Split a point by separating tetrahedra at that location."""
        new_point = self.clone_point(point)
        connected_cells, point_index = np.where(self.cells == point)
        dot_prods = {}
        new_indices = []
        old_indices = []
        for i, cell in enumerate(connected_cells):
            points = self.positions[self.cells[cell]]
            vectors = points - self.positions[point]
            mags = np.sqrt(np.sum(np.square(vectors), axis=1))
            mags = np.maximum(mags, 0.00001)
            dots = np.einsum('ij,j->i', vectors, normal) / mags
            for j in range(0, 4):
                dot_prods[self.cells[cell, j]] = dots[j]
            # 0.1 is cos(90 - 0.1*180/pi)
            if np.amin(dots) >= -0.1:
                # Assign the cell to `new_point`. Assumed to be n+
                new_indices.append([cell, point_index[i]])
            elif np.amax(dots) <= 0.1:
                old_indices.append([cell, point_index[i]])
                # Keep it assigned to `point`. Assumed to be n-
                pass
            else:
                old_indices.append([cell, point_index[i]])
                pass
        # Remove point if nothing assigned to it
        if len(new_indices) == 0 or len(old_indices) == 0:
            self.delete_last()
        else:
            for pair in new_indices:
                self.cells[pair[0]][pair[1]] = new_point
        return connected_cells, point_index, dot_prods

    def fracture(self, fracture_point):
        """Break the object and perform local remeshing."""
        point = fracture_point['point']
        normal = fracture_point['vector']
        print self.m_points[point]
        # Cloning the fracture point
        connected_cells, point_index, dot_prods = self.split(point, normal)
        closest = sorted(dot_prods, key=lambda x: abs(dot_prods[x]))
        closest = closest[1:]
        for c in closest[1:]:
            if abs(dot_prods[c]) < 0.1:
                _, _, _ = self.split(c, normal)
        self.recompute()


class Simulate(object):
    """This class runs the RK4 simulation on an object."""

    def __init__(self, constants, sim_step, fps, body, namespace, rule, ext):
        """The standard initialization function."""
        self.lame = constants['lame']
        self.density = constants['density']
        self.toughness = constants['toughness']
        self.h = sim_step
        self.out_rate = sim_step * int((1.0 / fps) / sim_step)
        self.thres_high = constants['thresholds']['high']
        self.thres_low = constants['thresholds']['low']
        self.body = body
        self.namespace = namespace
        self.rule = rule
        self.external_acc = ext

    def get_acc(self, pos, vel):
        """Wrapper for helper function get_accel()."""
        internal, sep_tensor = helpers.get_accel(
            cells=self.body.cells,
            points=pos,
            velocities=vel,
            volume=self.body.volume,
            mass=self.body.mass,
            beta=self.body.beta,
            constants=self.lame
        )
        external = self.external_acc(self.body)
        total = internal + external
        return total, sep_tensor

    def get_update(self):
        """Integrate object's trajectory using Explicit Euler."""
        positions = self.body.positions
        velocities = self.body.velocities
        h = self.h
        out_rate = self.out_rate
        pos_update = np.zeros(positions.shape)
        vel_update = np.zeros(velocities.shape)
        sep_tensor = None
        acceleration, _ = self.get_acc(positions, velocities)
        for j in range(int(out_rate / h)):
            pos_update = pos_update + \
                h * (velocities)
            vel_update = vel_update + \
                h * (acceleration)
            # Acceleration for next timestep
            acceleration, sep_tensor = self.get_acc(
                pos=(positions + pos_update),
                vel=(velocities + vel_update)
            )
            if np.any(np.abs(pos_update) > self.thres_high) or \
               np.isnan(pos_update).any() or \
               np.isinf(pos_update).any():
                # np.any(np.abs(vel_update) > thres_high) or \
                # np.isnan(vel_update).any() or \
                # np.isinf(vel_update).any():
                return pos_update, vel_update, sep_tensor
        return pos_update, vel_update, sep_tensor

    def get_update_rk4(self):
        """Integrate object's trajectory using RK4."""
        positions = self.body.positions
        velocities = self.body.velocities
        h = self.h
        out_rate = self.out_rate
        h2 = h / 2.0
        h6 = h / 6.0
        pos_update = np.zeros(positions.shape)
        vel_update = np.zeros(velocities.shape)
        sep_tensor = None
        acceleration, _ = self.get_acc(positions, velocities)
        for j in range(int(out_rate / h)):
            q1 = velocities + vel_update + h2 * acceleration
            k1, _ = self.get_acc(
                pos=(positions + pos_update + h2 * velocities + vel_update),
                vel=q1
            )
            q2 = velocities + vel_update + h2 * k1
            k2, _ = self.get_acc(
                pos=(positions + pos_update + h2 * q1),
                vel=q2
            )
            q3 = velocities + vel_update + h * k2
            k3, _ = self.get_acc(
                pos=(positions + pos_update + h * q2),
                vel=q3
            )
            pos_update = pos_update + \
                h * (velocities + vel_update + h6 * (acceleration + k1 + k2))
            vel_update = vel_update + \
                h6 * (acceleration + 2 * k1 + 2 * k2 + k3)
            # Acceleration for next timestep
            acceleration, sep_tensor = self.get_acc(
                pos=(positions + pos_update),
                vel=(velocities + vel_update)
            )
            if np.any(np.abs(pos_update) > self.thres_high) or \
               np.isnan(pos_update).any() or \
               np.isinf(pos_update).any():
                # np.any(np.abs(vel_update) > thres_high) or \
                # np.isnan(vel_update).any() or \
                # np.isinf(vel_update).any():
                return pos_update, vel_update, sep_tensor
        return pos_update, vel_update, sep_tensor

    def run(self, num_frames):
        """The number of frames to render in the scene."""
        if self.rule == "rk4":
            get_update = self.get_update_rk4
        else:
            get_update = self.get_update
        thres_high = self.thres_high
        thres_low = self.thres_low
        out_rate = self.out_rate
        sep_tensor = None
        for i in range(num_frames):
            # First trying to speed up simulation
            pos_update, vel_update, sep_tensor = get_update()
            # while (np.all(np.abs(pos_update) < thres_low) or
            #        np.all(np.abs(vel_update) < thres_low)) and \
            #         self.h < (out_rate / 2.0):
            #     self.h = self.h * 2.
            #     pos_update, vel_update, sep_tensor = get_update()
            #     print i, self.h
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
                pos_update, vel_update, sep_tensor = get_update()
                print i, self.h
            print i, self.h
            self.body.update(pos_update, vel_update)

            # Check whether object can separate at a point
            # breaking = True
            # while breaking is True:
            breaking, fracture_point = self.check_separation(sep_tensor)
            if breaking is True:
                print "Fractured!"
                self.body.fracture(fracture_point)
            self.write(i)

    def check_separation(self, sep_tensor):
        """The code checks whether object has potential to fracture."""
        e_val, e_vec = np.linalg.eig(sep_tensor)
        e_val_plus = np.maximum(e_val, 0)
        greatest = np.argmax(np.reshape(e_val_plus, [-1]))
        indices = (greatest / 3, greatest % 3)
        if e_val_plus[indices] > self.toughness:
            fracture_point = {
                'point': indices[0],
                'vector': e_vec[indices[0], :, indices[1]]
            }
            return True, fracture_point
        else:
            return False, None

    def write(self, frame):
        """Output the frame to a VTK file."""
        VTKInterface.write(
            self.body.positions,
            self.body.cells,
            'output/' + self.namespace + str(frame) + ".vtk"
        )
        POVRayInterface.write(
            self.body.positions,
            self.body.cells,
            'output/' + self.namespace + str(frame) + ".pov"
        )


# Assuming the object is glass
constants = {
    'lame': [1.04E4, 1.04E4, 0, 6760],
    'density': 2588,
    'toughness': 40,
    'thresholds': {
        'high': 10,
        'low': 0.000001
    }
}

body = Body('data/cube.2.vtk', constants['density'])

#body.deform(deformations.stretch)

sim = Simulate(constants=constants,
               sim_step=0.001,
               fps=30,
               body=body,
               namespace='cube_pull.vtk',
               rule='rk4',
               ext=external.stretch)
sim.run(100)
