import numpy as np
from scipy import linalg


def best_response(A, E, e, y0, max_iter=1000, tol=1e-2):
    """Computes best response to an opponent's realization plan in a two-person
    sequential game.

    Parameters
    ----------
    A: 2D array of size (m, n)
        Payoff matrix.

    E: 2D array of shape (p, n)
        Constraints matrix for our realization plans (x)

    e: 1D array of length p
        Our realization plans are constrained like so: Ex = e, x>=0.

    y0: 1D array of length m
        Fixed realization plan of opponent (player 2).

    """

    _, n = A.shape
    assert E.shape[1] == n
    p, _ = E.shape
    x = np.zeros(n)
    xbar = x.copy()
    zeta = np.zeros(p)
    L = linalg.svdvals(np.vstack((A, E)))[0]
    tau = sigma = .99 / L
    assert tau * sigma * L * L < 1.
    const = tau * A.T.dot(y0)
    r = 0
    history = []
    for k in xrange(max_iter):
        old_x = x.copy()
        residue = e - E.dot(xbar)
        r = (k * r + np.dot(residue, residue)) / (k + 1.)
        history.append(r)
        print "Iteration %03i/%03i: running average ||e - Exk||^2 = %g" % (
            k + 1, max_iter, r)
        if r < tol:
            print "Converged (running average ||e - Exk||^2 < %g)" % tol
            break
        zeta += sigma * (residue)
        x += const + E.T.dot(zeta)

        x = np.maximum(x, 0.)

        xbar = 2 * x - old_x

    return x, history


if __name__ == "__main__":
    A = np.zeros((5, 3))
    A[2:4, 1:] = [[1, -1], [-2, 4]]
    A[-1, 0] = 1
    A = A.T
    E = np.array([[1, 0, 0, 0, 0], [-1, 1, 1, 0, 0], [-1, 0, 0, 1, 1]])
    e = np.eye(E.shape[0])[0]
    y0 = np.array([1, .66, .33])
    xstar, history = best_response(A, E, e, y0)
    print "x* =", xstar
    print E.dot(xstar) - e

    import pylab as pl
    history = np.array(history)
    history -= history.min()
    pl.loglog(history)
    pl.ylabel("(excess) running average ||e - Exk||^2")
    pl.xlabel("k")
    pl.show()
