# Author: DOHMATOB Elvis Dopgima <gmdopp@gmail.com>

from math import sqrt
import numpy as np
from scipy import linalg


def proj_simplex(v, z=1.):
    """Euclidean projection onto a simplex."""
    v = np.array(v)
    assert v.ndim == 1
    n = len(v)
    u = v.copy()
    u.sort()
    u = u[::-1]
    cs = np.cumsum(u)
    cs -= z
    cs /= np.arange(1., n + 1)
    rho = np.nonzero(u - cs > 0.)[0][-1]
    theta = cs[rho]
    return np.maximum(v - theta, 0.)


def solve_matrix_game(A, max_iter=1000, tol=1e-6, verbose=1):
    """Solve matrix game using Chambolle-Pock algorithm."""
    A = np.array(A)
    n, m = A.shape
    x = np.zeros(m)
    y = np.zeros(n)
    xbar = x.copy()
    L = linalg.svdvals(A)[0] ** 2
    tau = sigma = .99 / sqrt(L)
    assert tau * sigma * L <= 1.
    theta = 1.
    old_value = np.inf
    for k in xrange(max_iter):
        old_x = x.copy()
        y += sigma * np.dot(A, xbar)
        y = proj_simplex(y)
        Aty = np.dot(A.T, y)
        value = np.dot(x, Aty)
        value_delta = old_value - value
        old_value = value
        if verbose:
            print "Iteration %03i/%03i: value=%g, change=%g" % (
                k + 1, max_iter, value, value_delta)
        value_delta = abs(value_delta)
        if value_delta < tol:
            if verbose:
                print "Converged after %i iterations (|change| < %g)." % (
                    k + 1, tol)
            break
        x -= tau * Aty
        x = proj_simplex(x)
        xbar = x + theta * (x - old_x)

    return x, y


if __name__ == "__main__":
    print solve_matrix_game(10 * np.random.randn(250, 150))
