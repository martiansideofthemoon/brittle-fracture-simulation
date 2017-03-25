"""A set of helper functions to calculate useful information."""
import numpy as np

from numpy.linalg import inv


def get_accel(cells, points, velocities, volume, mass, beta, constants):
    """The function returns the acceleration."""
    dpos, dvel = get_derivatives(cells, points, velocities, beta)
    stress, elastic, viscous, strain, rate = get_stress(dpos, dvel, constants)
    internal_f = get_internal(cells, points, beta, stress, volume)
    internal_acc = internal_f / mass[:, None]
    return internal_acc
    # acceleration = np.add(internal_acc, gravity)


def point_vectors(cells, points):
    """Return vectors used in other functions."""
    p1 = points[cells[:, 0]]
    p2 = points[cells[:, 1]]
    p3 = points[cells[:, 2]]
    p4 = points[cells[:, 3]]
    return p1, p2, p3, p4


def get_volume(cells, points):
    """Vectorised volume from the cell and a tuple of points."""
    p1, p2, p3, p4 = point_vectors(cells, points)
    term1 = np.cross(p2 - p4, p3 - p4)
    term2 = p1 - p4
    return np.abs(np.einsum('ij,ij->i', term1, term2)) / 6.0


def get_beta(cells, points):
    """Vectorised computation of beta values, equation (16)."""
    p1, p2, p3, p4 = point_vectors(cells, points)
    m_points = np.stack([p1, p2, p3, p4], axis=2)
    ones = np.ones((m_points.shape[0], 1, m_points.shape[2]))
    orig = np.concatenate([m_points, ones], axis=1)
    return inv(orig)


def get_derivatives(cells, points, velocities, beta):
    """Vectorised computation of derivatives, equation (21) and (22)."""
    p1, p2, p3, p4 = point_vectors(cells, points)
    v1, v2, v3, v4 = point_vectors(cells, velocities)

    positions = np.stack([p1, p2, p3, p4], axis=2)
    velocities = np.stack([v1, v2, v3, v4], axis=2)
    delta = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [0, 0, 0]])
    prod1 = np.matmul(beta, delta)
    # einsum() applies a matrix multiplication across dimension 2 and 3
    dpos = np.matmul(positions, prod1)
    dvel = np.matmul(velocities, prod1)
    return dpos, dvel


def get_stress(dpos, dvel, constants):
    """
    Vectorized computation of stress tensors from dpos and dvel.

    constants - (lambda, mu, phi, psi) - equation (7), (8)
    """
    # dpos_T = np.transpose(dpos, [0, 2, 1])
    # dvel_T = np.transpose(dvel, [0, 2, 1])
    strain = np.einsum('ijk,ijl->ikl', dpos, dpos) - np.eye(3)
    rate = np.add(
        np.einsum('ijk,ijl->ikl', dpos, dvel),
        np.einsum('ijk,ijl->ikl', dvel, dpos)
    )
    # Implementation of equation (7)
    str_trace = np.trace(strain, axis1=1, axis2=2)
    elastic = \
        constants[0] * np.einsum('i,jk->ijk', str_trace, np.eye(3)) + \
        2 * constants[1] * strain
    rate_trace = np.trace(rate, axis1=1, axis2=2)
    viscous = \
        constants[2] * np.einsum('i,jk->ijk', rate_trace, np.eye(3)) + \
        2 * constants[3] * rate
    return np.add(elastic, viscous), elastic, viscous, strain, rate


def get_internal(cells, points, beta, stress, volume):
    """Compute internal forces on all nodes."""
    p1, p2, p3, p4 = point_vectors(cells, points)
    positions = np.stack([p1, p2, p3, p4], axis=2)
    # This is because the fourth column of beta is not used in summation
    beta = beta[:, :, :3]
    # einsum() implementation of equation (27)
    force = -0.5 * np.einsum('h,hmj,hjl,hik,hkl->hmi', volume, positions, beta, beta, stress)
    forces_node = np.zeros(points.shape)
    np.add.at(forces_node, cells, np.transpose(force, (0, 2, 1)))
    return forces_node


def get_mass(cells, points, density):
    """Compute the mass of each point, contributed by cells."""
    volume = get_volume(cells, points)
    volume = np.expand_dims(volume, axis=1)
    mass = np.zeros((points.shape[0],))
    volume = 0.25 * density * np.repeat(volume, 4, axis=1)
    np.add.at(mass, cells, volume)
    return mass
