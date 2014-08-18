from math import sqrt
import numpy as np


def dist_from_plane(unit_normal, intercept, point):
    """Perpendicular distance of a point from a plane (or line)."""
    return abs(np.dot(unit_normal, point) + intercept)


def projection_onto_plane(unit_normal, intercept, point):
    """Euclidean projection of a point onto a plane (or line)."""
    d = dist_from_plane(unit_normal, intercept, point)
    return point - d * unit_normal


def proj(unit_normals, intercepts, point, max_iter=100, verbose=1.,
         tol=1e-10):
    """Euclidean projection of a point onto the intersection of
    the nonnegative orthant and a set of hyperplanes."""
    x = point
    error = np.inf
    for k in xrange(max_iter):
        old_x = x.copy()

        if verbose:
            print "Iteration %03i/%03i: x=%s, ||dx||=%5.2e" % (
                k + 1, max_iter, x, error)
        if error < tol:
            if verbose:
                print "Converged."
            break

        for unit_normal, intercept in zip(unit_normals, intercepts):
            x = projection_onto_plane(unit_normal, intercept, x)
        x = np.maximum(x, 0.)
        error = sqrt(((x - old_x) ** 2).sum())
    return x


def evil(L, r, z, proj_C=lambda v: np.maximum(v, 0.), L_norm=None,
         max_iter=100, verbose=1, tol=1e-4):
    """Projects the point z onto the intersection of the hyperplane "Ex=r" and
    the convex set C.
    """
    from scipy import linalg
    if L_norm is None:
        L_norm = linalg.svdvals(L)[0]
    gamma = 1.99 / L_norm ** 2
    v = np.zeros(L.shape[0])
    x = np.zeros(L.shape[1])
    b = 1
    for k in xrange(max_iter):
        old_x = x.copy()
        x = proj_C(z - L.T.dot(v)) + b
        v += gamma * (L.dot(x) - r)
        b *= .5  # we want: \sum_{k}{b_k} < \infty
        error = sqrt(((x - old_x) ** 2).sum())
        if verbose:
            print "\tIteration %03i/%03i: x=%s, error=%5.2e" % (
                x, k + 1, max_iter, error)
        if error < tol:
            if verbose:
                print "\tConverged after %i iterations." % (k + 1)
            break
    return x

if __name__ == "__main__":
    # print ">>Projection of the point (0, -1) onto the line x - y = 1."
    # x = proj([np.array([1, -1]) / sqrt(2.)], [1. / sqrt(2.)],
    #          np.array([0, -1]))
    # print "x* = ", x
    # print "_" * 80

    # print ('>>Projection of the point (0, 1, 1) onto the line of '
    #        'intercetion of ''planes "x=0" and "y=0", namely the line'
    #        ' "x = y = 0".')
    # x = proj([np.array([1., 0., 0.]), np.array([0., 1., 0.])],
    #          [0., 0.], np.array([0., 1., 1.]))
    # print "x* =", x
    # print "_" * 80

    print evil(np.array([[1, -2.]]), 0., np.array([2, -1.]))
