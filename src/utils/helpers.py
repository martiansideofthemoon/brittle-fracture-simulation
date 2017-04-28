"""A set of helper functions to calculate useful information."""
import numpy as np

from numpy.linalg import inv


def intersect(plane_point, plane_vec, pt1, pt2):
    """The function intersects a line and a plane."""
    ray_dir = pt2 - pt1
    ndotu = plane_vec.dot(ray_dir)
    w = pt1 - plane_point
    si = -plane_vec.dot(w) / ndotu
    return (pt1 + si * ray_dir)


def get_accel(cells, points, velocities, volume, mass, beta, constants):
    """The function returns the acceleration."""
    dpos, dvel = get_derivatives(cells, points, velocities, beta)
    stress, tensile = get_stress(dpos, dvel, constants)
    internal_f, sep_tensor = get_internal(cells, points, beta, stress, tensile, volume)
    internal_acc = internal_f / mass[:, None] + np.array([0, -9.8, 0])
    collision = 50 * -1 * np.minimum(points[:, 1] + 5, 0)
    internal_acc[:, 1] += collision
    return internal_acc, sep_tensor
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
    stress = elastic + viscous

    # Implementation of force decomposition
    e_val, e_vec = np.linalg.eig(stress)
    e_val_plus = np.maximum(e_val, 0)
    tensile = np.matmul(e_vec * e_val_plus[..., np.newaxis], e_vec.transpose([0, 2, 1]))
    return stress, tensile


def get_internal(cells, points, beta, stress, tensile, volume):
    """Compute internal forces on all nodes."""
    p1, p2, p3, p4 = point_vectors(cells, points)
    positions = np.stack([p1, p2, p3, p4], axis=2)
    # This is because the fourth column of beta is not used in summation
    beta = beta[:, :, :3]
    # einsum() implementation of equation (27)
    force = -0.5 * np.einsum('h,hmj,hjl,hik,hkl->hmi', volume, positions, beta, beta, stress)
    force_tensile = -0.5 * np.einsum('h,hmj,hjl,hik,hkl->hmi', volume, positions, beta, beta, tensile)
    force_compressive = force - force_tensile

    forces_node = np.zeros(points.shape)
    forces_tensile_node = np.zeros(points.shape)

    np.add.at(forces_node, cells, np.transpose(force, (0, 2, 1)))
    np.add.at(forces_tensile_node, cells, np.transpose(force_tensile, (0, 2, 1)))
    forces_compressive_node = forces_node - forces_tensile_node

    forces_tensile_node = forces_tensile_node[..., np.newaxis]                      # n_pts * 3 * 1
    forces_compressive_node = forces_compressive_node[..., np.newaxis]

    # print forces_tensile_node.shape
    # print forces_compressive_node.shape

    # print np.any(np.linalg.norm(forces_tensile_node, axis=1, keepdims=True) == 0)
    # print np.any(np.linalg.norm(forces_compressive_node, axis=1, keepdims=True) == 0)

    # equation 32
    t1_denom = np.linalg.norm(forces_tensile_node, axis=1, keepdims=True)
    t3_denom = np.linalg.norm(forces_compressive_node, axis=1, keepdims=True)
    t1_denom[t1_denom == 0] = 1
    t3_denom[t3_denom == 0] = 1

    term1 = np.einsum('ijl,ikl->ijk', forces_tensile_node, forces_tensile_node) / t1_denom      # n_pts * 3 * 3
    term3 = np.einsum('ijl,ikl->ijk', forces_compressive_node, forces_compressive_node) / t3_denom

    force_tensile = force_tensile.transpose((0, 2, 1))[..., np.newaxis]             # n_cells * 4 * 3 * 1
    force_compressive = force_compressive.transpose((0, 2, 1))[..., np.newaxis]

    t2_denom = np.linalg.norm(force_tensile, axis=2, keepdims=True)
    t4_denom = np.linalg.norm(force_compressive, axis=2, keepdims=True)
    t2_denom[t2_denom == 0] = 1
    t4_denom[t4_denom == 0] = 1

    t2 = np.einsum('abik,abjk->abij', force_tensile, force_tensile) / t2_denom
    t4 = np.einsum('abik,abjk->abij', force_compressive, force_compressive) / t4_denom

    sep_tensor = -term1 + term3
    np.add.at(sep_tensor, cells, t2 - t4)
    sep_tensor = sep_tensor / 2.

    return forces_node, sep_tensor


def get_mass(cells, points, density):
    """Compute the mass of each point, contributed by cells."""
    volume = get_volume(cells, points)
    volume = np.expand_dims(volume, axis=1)
    mass = np.zeros((points.shape[0],))
    volume = 0.25 * density * np.repeat(volume, 4, axis=1)
    np.add.at(mass, cells, volume)
    return mass
