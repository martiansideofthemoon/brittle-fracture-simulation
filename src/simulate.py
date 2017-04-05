"""This file creates a basic FEM simulation."""
import numpy as np
import sys
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
            np.concatenate((self.m_points, m_point), axis=0)
        self.positions = \
            np.concatenate((self.positions, position), axis=0)
        self.velocities = \
            np.concatenate((self.velocities, velocity), axis=0)
        # Return index of newly added point
        return (self.m_points.shape[0] - 1)

    def recompute(self):
        """Recompute mesh parameters."""
        self.beta = helpers.get_beta(self.cells, self.m_points)
        self.volume = helpers.get_volume(self.cells, self.m_points)
        self.mass = helpers.get_mass(self.cells, self.m_points, self.density)

    def fracture(self, fracture_point):
        """Break the object and perform local remeshing."""
        point = fracture_point['point']
        normal = fracture_point['vector']
        # Cloning the fracture point
        new_point = self.clone_point(point)
        connected_cells, point_index = np.where(self.cells == point)
        cross_cells = []
        cross_point_indices = []
        for i, cell in enumerate(connected_cells):
            points = self.positions[self.cells[cell]]
            vectors = points - self.positions[point]
            dots = np.einsum('ij,j->i', vectors, normal)
            # TODO - Add thresholding for nodes close to plain
            if np.amin(dots) >= 0:
                # Assign the cell to `new_point`. Assumed to be n+
                self.cells[cell][point_index[i]] = new_point
            elif np.amax(dots) <= 0:
                # Keep it assigned to `point`. Assumed to be n-
                pass
            else:
                cross_cells.append(cell)
                cross_point_indices.append(point_index[i])
        # Work on splitting up the tetrahedrons crossing an element
        for i, cell in enumerate(cross_cells):
            # Each cell needs to be broken into 3 tetrahedra
            points = self.positions[self.cells[cell]]
            vectors = points - self.positions[point]
            dots = np.einsum('ij,j->i', vectors, normal)
            plus = np.where(dots > 0)[0]
            neg = np.where(dots < 0)[0]
            if len(plus) == 2 and len(neg) == 1:
                p0 = helpers.intersect(point, normal, points[plus[0]], points[neg[0]])
                p0 = self.add_pt(p0, cell)
                p1 = helpers.intersect(point, normal, points[plus[1]], points[neg[0]])
                p1 = self.add_pt(p1, cell)
                # Fetched the intersection point indices
                # Time to remesh system
                self.cells[cell] = np.array([point, p0, p1, self.cells[cell, neg[0]]])
                cell1 = np.array([new_point, p0, self.cells[cell, plus[0]], self.cells[cell, plus[1]]])
                cell2 = np.array([new_point, p0, p1, self.cells[cell, plus[1]]])
                
            elif len(plus) == 1 and len(neg) == 2:
                p1 = helpers.intersect(point, normal, points[plus[0]], points[neg[0]])
                p1 = self.add_pt(p1, cell)
                p2 = helpers.intersect(point, normal, points[plus[0]], points[neg[1]])
                p2 = self.add_pt(p2, cell)
            else:
                print "Error!"
                sys.exit()
        self.recompute()


class Simulate(object):
    """This class runs the RK4 simulation on an object."""

    def __init__(self, constants, sim_step, fps, body, namespace):
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
        thres_high = self.thres_high
        thres_low = self.thres_low
        out_rate = self.out_rate
        sep_tensor = None
        for i in range(num_frames):
            # First trying to speed up simulation
            breaking = True
            while breaking is True:
                pos_update, vel_update, sep_tensor = self.get_update()
                while (np.all(np.abs(pos_update) < thres_low) or
                       np.all(np.abs(vel_update) < thres_low)) and \
                        self.h < (out_rate / 2.0):
                    self.h = self.h * 2.
                    pos_update, vel_update, sep_tensor = self.get_update()
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
                    pos_update, vel_update, sep_tensor = self.get_update()
                    print i, self.h
                # Check whether object can separate at a point
                breaking, fracture_point = self.check_separation(sep_tensor)
                if breaking is True:
                    self.body.fracture(fracture_point)
            print i, self.h
            self.body.update(pos_update, vel_update)
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


# Assuming the object is glass
constants = {
    'lame': [1.04E4, 1.04E4, 0, 6760],
    'density': 2588,
    'toughness': 200,
    'thresholds': {
        'high': 10,
        'low': 0.000001
    }
}

body = Body('data/cube.2.vtk', constants['density'])
body.deform(deformations.twist)
sim = Simulate(constants, 0.001, 30, body, 'squashcube')
sim.run(100)
